# -*- coding: utf-8 -*-
"""
napping_lite.py
===============
Version autonome du pipeline de napping — sans Qt, sans QThread.

Basée sur test_projection_identite.py (stable 15+ min, thread principal).

Fonctionnalités :
  - Tracking KCF (cam_top) + ArUco (cam_bottom) + identité stable
  - Projecteur : fond blanc + cercles colorés (identique test_projection_identite.py)
  - Fenêtre grille 700×700 à l'écran : points rouges = positions tasses, axes -10/10
  - Export CSV enrichi — 3 sources de coordonnées par tasse :
      · cam_top (pos_mm EMA-lissée, via tracker KCF + PoseConverter)
      · tracker  (position brute du tracker KCF, avant EMA et avant match)
      · cam_bottom (position brute ArUco, frame par frame)
  - Export JSON associations.json :
      · pour chaque aruco_id : liste des tracker_id successifs + log bind/unbind
  - Enregistrement vidéo cam_top via VideoWriterThread (non bloquant)
  - Saisie participant / protocole au démarrage via input()

COLONNES CSV :
  frame, timestamp,
  ID_{id}_x_camtop, ID_{id}_y_camtop,   ← position EMA lissée (cam_top)
  ID_{id}_x_tracker, ID_{id}_y_tracker,  ← position brute tracker KCF
  ID_{id}_x_bottom, ID_{id}_y_bottom     ← position brute ArUco (cam_bottom)

  Chaque groupe est répété pour chaque tag dans ARUCO_CUP_IDS.
  Cellule vide si la source n'a pas de donnée ce frame.

TOUCHES :
  q=quitter  r=reset trackers  +/-=seuil détection
  c=toggle correction 3D  i=debug identités
"""

import csv
import cv2
import json
import numpy as np
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.core.vision.camera_manager import CameraManager
from src.core.cup_tracking.cam_bottom_thread import CamBottomThread as CamBottomThreadCore
from src.core.cup_tracking.video_writer_thread import VideoWriterThread
from src.core.cup_tracking.cup_identity_manager import CupIdentityManager, CupState
from src.core.config.app_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from src.core.utils.paths import config_path, data_path
from src.core.projection.display_manager import DisplayManager
from src.core.projection.draw_utils import DrawUtils


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CAM_TOP_INDEX  = 0
CAM_BOT_INDEX  = 1

CAPTURE_W      = 1920
CAPTURE_H      = 1080
PROCESS_W      = 640
PROCESS_H      = 360
PROC_TO_NATIVE = CAPTURE_W / PROCESS_W
TABLE_SIZE_MM  = 597.0
BOUNDS_MARGIN_MM = 30.0
CUP_HEIGHT_MM  = 95.0

PROJ_W         = 3840
PROJ_H         = 2160
PROJ_SCREEN_ID = 1
RING_RADIUS    = 120
RING_THICKNESS = 10

# Grille fenêtre PC — axes en unités "index" comme RecordWindow
GRID_X_MIN  = -10.0
GRID_X_MAX  =  10.0
GRID_Y_MIN  = -10.0
GRID_Y_MAX  =  10.0
GRID_X_LEG  = "x"
GRID_Y_LEG  = "y"
GRID_SIZE   = 700   # pixels — image carrée 700×700

# Couleurs état identité
COLOR_MATCHED  = (0, 220, 0)
COLOR_AIRBORNE = (0, 160, 255)
COLOR_LOST     = (0, 0, 220)
COLOR_UNKNOWN  = (180, 180, 180)

# Détection masque
V_THRESHOLD  = 110
AREA_MIN     = 800
AREA_MAX     = 40_000
CIRC_MIN     = 0.15
ASPECT_MIN   = 0.25
ASPECT_MAX   = 3.5
MARGIN       = 20

# MOSSE
DETECT_INTERVAL_S = 0.10
MAX_LOST_FRAMES   = 15
MATCH_MIN_SCORE   = 0.01
MAX_DRIFT_RATIO   = 0.6
STABILITY_FRAMES  = 8
MAX_DRIFT_PX = 25

BBOX_GROW_RATIO   = 1.4   # bbox MOSSE autorisée à grandir de 40% max
                           # au delà → tracker a perdu sa cible
ARUCO_DRIFT_MM    = 80.0  # distance max tracker↔ArUco avant de considérer hijacking
ARUCO_DRIFT_FRAMES = 2    # nombre de frames consécutives de drift avant invalidation

# EMA
EMA_ALPHA       = 0.35
EMA_MAX_JUMP_MM = 200.0
OFFSET_ALPHA = 0.15

# CSV
CSV_FLUSH_EVERY = 200

# Vidéo
VIDEO_W   = 960
VIDEO_H   = 540
VIDEO_FPS = 30.0

# Purge identity manager
PURGE_EVERY_FRAMES = 300
PURGE_MAX_AGE_S    = 20.0
# IDs ArUco des tasses utilisées dans la session
# À adapter selon le protocole
ARUCO_CUP_IDS = [0,1,2,3,4,5,6,7,8]


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires géométriques  (inchangés)
# ══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA, max_jump=EMA_MAX_JUMP_MM):
        self._a, self._mj = alpha, max_jump
        self._val = None

    def update(self, x, y):
        if self._val is None:
            self._val = (x, y); return self._val
        dx, dy = x - self._val[0], y - self._val[1]
        if (dx*dx + dy*dy)**.5 > self._mj:
            self._val = (x, y); return self._val
        self._val = (self._a*x + (1-self._a)*self._val[0],
                     self._a*y + (1-self._a)*self._val[1])
        return self._val

    def reset(self): self._val = None

class KalmanFilter2D:
    """
    Filtre de Kalman 2D à modèle cinématique vitesse constante.
    Utilisé UNIQUEMENT pour l'export CSV — la projection reste sur EMA.

    État interne : [x, vx, y, vy]
    Mesure       : [x, y]

    Paramètres (calibrés sur tes données) :
      process_noise = 2.0  — confiance dans le modèle cinématique
      measure_noise = 8.0  — confiance dans la mesure KCF brute
      max_jump_mm   = 200  — même garde-fou que EMAFilter

    Interface identique à EMAFilter :
      update(x, y) → (xf, yf)
      reset()
    """

    def __init__(
        self,
        process_noise: float = 2.0,
        measure_noise: float = 8.0,
        max_jump_mm:   float = 200.0,
    ):
        self._pn = process_noise
        self._mn = measure_noise
        self._mj = max_jump_mm

        # Modèle cinématique vitesse constante, dt = 1 frame
        self._F = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # On observe uniquement la position, pas la vitesse
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)

        self._Q = self._pn * np.eye(4, dtype=np.float64)
        self._R = self._mn * np.eye(2, dtype=np.float64)

        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._x is None:
            self._x = np.array([x, 0.0, y, 0.0], dtype=np.float64)
            self._P = np.eye(4, dtype=np.float64) * 100.0
            return float(x), float(y)

        # Garde-fou : saut impossible → réinitialisation
        dx = x - float(self._x[0])
        dy = y - float(self._x[2])
        if (dx * dx + dy * dy) ** 0.5 > self._mj:
            self._x = np.array([x, 0.0, y, 0.0], dtype=np.float64)
            self._P = np.eye(4, dtype=np.float64) * 100.0
            return float(x), float(y)

        # Predict
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # Update
        z     = np.array([x, y], dtype=np.float64)
        y_res = z - self._H @ x_pred
        S     = self._H @ P_pred @ self._H.T + self._R
        K     = P_pred @ self._H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ y_res
        self._P = (np.eye(4) - K @ self._H) @ P_pred

        return float(self._x[0]), float(self._x[2])

    def reset(self) -> None:
        self._x = None
        self._P = None

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """Vitesse estimée (vx, vy) en mm/frame. None si non initialisé."""
        if self._x is None:
            return None
        return float(self._x[1]), float(self._x[3])

    @property
    def speed_mm_per_frame(self) -> float:
        v = self.velocity
        return 0.0 if v is None else (v[0] ** 2 + v[1] ** 2) ** 0.5




def _cup_base_correction(cx, cy, rvec, tvec, K):
    K_inv = np.linalg.inv(K)
    ray   = K_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
    ray  /= np.linalg.norm(ray)
    R, _  = cv2.Rodrigues(rvec)
    n, o  = R[:, 2], tvec.reshape(3)
    denom = np.dot(n, ray)
    if abs(denom) < 1e-9: return cx, cy
    t = np.dot(n, o) / denom
    if t < 0: return cx, cy
    pt_cam   = ray * t
    pt_table = R.T @ (pt_cam - o)
    x_top, y_top = float(pt_table[0]), float(pt_table[1])
    dist_zero = np.zeros((5, 1))
    pt_mid  = np.array([[x_top, y_top, CUP_HEIGHT_MM/2]], dtype=np.float64)
    pm, _   = cv2.projectPoints(pt_mid, rvec, tvec, K, dist_zero)
    pt_base = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
    pb, _   = cv2.projectPoints(pt_base, rvec, tvec, K, dist_zero)
    return float(pb[0, 0, 0]), cy + (cy - float(pm[0, 0, 1]))


def _is_on_table(x, y):
    lo, hi = -BOUNDS_MARGIN_MM, TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    return lo <= x <= hi and lo <= y <= hi


def _center(bbox):
    x, y, w, h = bbox
    return x + w/2.0, y + h/2.0


def _iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix*iy; union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0


def _match_score(b1, b2):
    iou  = _iou(b1, b2)
    x, y, w, h = b1
    diag = max(np.sqrt(w**2+h**2), 1.0)
    cx1,cy1 = _center(b1); cx2,cy2 = _center(b2)
    d = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) / diag
    if d > 2.0: return 0.0
    return iou + 0.4 * max(0.0, 1.0 - d)


def _drift_ok(prev, new):
    x, y, w, h = prev
    diag = max(np.sqrt(w**2+h**2), 1.0)
    cx1,cy1 = _center(prev); cx2,cy2 = _center(new)
    dist = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)
    return dist < MAX_DRIFT_RATIO * diag and dist < MAX_DRIFT_PX


def _identity_color(state: Optional[CupState]) -> tuple:
    if state is None:               return COLOR_UNKNOWN
    if state == CupState.MATCHED:   return COLOR_MATCHED
    if state == CupState.AIRBORNE:  return COLOR_AIRBORNE
    return COLOR_LOST


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter  (inchangé)
# ══════════════════════════════════════════════════════════════════════════════

class PoseConverter:
    def __init__(self, pose_path):
        d         = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec = np.array(d["rvec"],          dtype=np.float64)
        self.tvec = np.array(d["tvec"],          dtype=np.float64)
        self.K    = np.array(d["camera_matrix"], dtype=np.float64)
        h_path = pose_path.replace("camtop_table_pose.json", "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(f"H_top_to_bottom.json introuvable → {h_path}")
        self._H = np.array(
            json.load(open(h_path))["H_top_to_bottom"], dtype=np.float32)
        print(f"[PoseConverter] OK  fx={self.K[0,0]:.1f}")

    def pixel_to_mm(self, u, v):
        K_inv = np.linalg.inv(self.K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray  /= np.linalg.norm(ray)
        R, _  = cv2.Rodrigues(self.rvec)
        n, o  = R[:, 2], self.tvec.reshape(3)
        denom = np.dot(n, ray)
        if abs(denom) < 1e-9: return None
        t = np.dot(n, o) / denom
        if t < 0: return None
        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - o)
        pt2 = cv2.perspectiveTransform(
            np.array([[[float(pt_table[0]), float(pt_table[1])]]],
                     dtype=np.float32), self._H)
        return float(pt2[0, 0, 0]), float(pt2[0, 0, 1])


def _proc_to_mm(cx_p, cy_p, converter, use_3d):
    cx_n = cx_p * PROC_TO_NATIVE
    cy_n = cy_p * PROC_TO_NATIVE
    if use_3d:
        cx_n, cy_n = _cup_base_correction(
            cx_n, cy_n, converter.rvec, converter.tvec, converter.K)
    mm = converter.pixel_to_mm(cx_n, cy_n)
    if mm is None or not _is_on_table(*mm): return None
    return mm


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedCup  (inchangé sauf update_mm qui renvoie aussi la pos brute)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    cup_id:         int
    ema:            EMAFilter
    kalman_csv:     KalmanFilter2D
    cv_tracker:     object
    bbox:           tuple
    pos_mm:         Optional[Tuple[float, float]] = None
    pos_mm_raw:     Optional[Tuple[float, float]] = None
    pos_mm_kalman:  Optional[Tuple[float, float]] = None
    lost_frames:    int   = 0
    active:         bool  = True

    # Offset correction cam_top → cam_bottom (inchangé)
    proj_offset:    Tuple[float, float] = (0.0, 0.0)
    _offset_ema_x:  float = 0.0
    _offset_ema_y:  float = 0.0
    _offset_init:   bool  = False

    # ── NOUVEAU — contrainte taille bbox ─────────────────────────────────────
    # Enregistrée à l'init, comparée à chaque update MOSSE.
    # Si MOSSE retourne une bbox > BBOX_GROW_RATIO × ref → tracker invalidé.
    bbox_ref_w:     int   = 0
    bbox_ref_h:     int   = 0

    # ── NOUVEAU — compteur de drift ArUco ────────────────────────────────────
    # Incrémenté chaque frame où pos_mm s'éloigne de ArUco de plus de ARUCO_DRIFT_MM.
    # Remis à 0 dès que l'écart repasse sous le seuil ou que ArUco n'est pas visible.
    # Quand il atteint ARUCO_DRIFT_FRAMES → tracker invalidé.
    aruco_drift_count: int = 0

    def update_kcf(self, frame: np.ndarray) -> bool:
        if not self.active:
            return False

        ok, raw = self.cv_tracker.update(frame)
        if not ok:
            self.active = False
            return False

        rx, ry, rw, rh = raw
        nb = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))

        # Contrainte 1 — dérive position (inchangée)
        if not _drift_ok(self.bbox, nb):
            self.active = False
            return False

        # Contrainte 2 — bbox qui gonfle (NOUVEAU)
        # MOSSE dont la bbox grossit a perdu sa cible et suit une zone vide.
        if self.bbox_ref_w > 0 and self.bbox_ref_h > 0:
            if (nb[2] > self.bbox_ref_w * BBOX_GROW_RATIO or
                    nb[3] > self.bbox_ref_h * BBOX_GROW_RATIO):
                print(f"[TrackedCup] #{self.cup_id} : bbox gonflée "
                      f"({nb[2]}×{nb[3]} vs ref {self.bbox_ref_w}×{self.bbox_ref_h}) "
                      f"→ invalidé")
                self.active = False
                return False

        self.bbox = nb
        return True

    def check_aruco_drift(
        self,
        aruco_pos_mm: Optional[Tuple[float, float]],
    ) -> bool:
        '''
        Vérifie la cohérence entre pos_mm (tracker) et aruco_pos_mm (cam_bottom).

        Appelée depuis la boucle principale APRÈS update_tracking(),
        uniquement pour les trackers dont on connaît l'aruco_id.

        Retourne True si le tracker est toujours valide, False si invalidé.

        Logique :
          - ArUco absent ou tasse AIRBORNE → pas de vérification (drift_count = 0)
          - ArUco présent ET distance > ARUCO_DRIFT_MM → drift_count++
          - Si drift_count >= ARUCO_DRIFT_FRAMES → tracker invalidé
          - ArUco présent ET distance ok → drift_count = 0
        '''
        if aruco_pos_mm is None or self.pos_mm is None:
            self.aruco_drift_count = 0
            return True

        dx = self.pos_mm[0] - aruco_pos_mm[0]
        dy = self.pos_mm[1] - aruco_pos_mm[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist > ARUCO_DRIFT_MM:
            self.aruco_drift_count += 1
            if self.aruco_drift_count >= ARUCO_DRIFT_FRAMES:
                print(f"[TrackedCup] #{self.cup_id} : drift ArUco "
                      f"{dist:.0f}mm depuis {self.aruco_drift_count} frames → invalidé")
                self.active = False
                return False
        else:
            self.aruco_drift_count = 0

        return True

    def reinit_kcf(self, frame: np.ndarray, bbox: tuple) -> bool:
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(4, min(w, fw - x))
        h = max(4, min(h, fh - y))
        t = cv2.legacy.TrackerMOSSE_create()
        try:
            t.init(frame, (x, y, w, h))
            self.cv_tracker       = t
            self.bbox             = (x, y, w, h)
            self.bbox_ref_w       = w          # ← réinitialiser la référence
            self.bbox_ref_h       = h
            self.aruco_drift_count = 0          # ← reset compteur drift
            self.active           = True
            return True
        except Exception as e:
            print(f"[KCF] reinit cup#{self.cup_id}: {e}")
            self.active = False
            return False

    def update_mm(self, converter, use_3d: bool) -> None:
        cx, cy = _center(self.bbox)
        mm = _proc_to_mm(cx, cy, converter, use_3d)
        if mm is not None:
            self.pos_mm_raw   = mm
            xs, ys = self.ema.update(*mm)
            self.pos_mm       = (xs, ys)
            xk, yk = self.kalman_csv.update(*mm)
            self.pos_mm_kalman = (xk, yk)


# ══════════════════════════════════════════════════════════════════════════════
#  CupDetector  (inchangé)
# ══════════════════════════════════════════════════════════════════════════════

class CupDetector:
    def __init__(self):
        self.v_threshold = V_THRESHOLD
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        mask = self._mask(frame)
        return self._bboxes(frame, mask), mask

    def _mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kc, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self._ko, iterations=1)
        return self._rm(m)

    @staticmethod
    def _rm(mask):
        h, w = mask.shape; f = mask.copy()
        for x in range(w):
            if f[0,x]:   cv2.floodFill(f, None, (x,0),   0)
            if f[h-1,x]: cv2.floodFill(f, None, (x,h-1), 0)
        for y in range(h):
            if f[y,0]:   cv2.floodFill(f, None, (0,y),   0)
            if f[y,w-1]: cv2.floodFill(f, None, (w-1,y), 0)
        return f

    def _bboxes(self, frame, mask):
        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if not (AREA_MIN <= a <= AREA_MAX): continue
            hull = cv2.convexHull(cnt)
            ha   = cv2.contourArea(hull); hp = cv2.arcLength(hull, True)
            if hp < 1 or ha < 1: continue
            if 4*np.pi*ha/hp**2 < CIRC_MIN: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0 and not (ASPECT_MIN <= w/float(h) <= ASPECT_MAX): continue
            x1=max(0,x-MARGIN); y1=max(0,y-MARGIN)
            x2=min(fw,x+w+MARGIN); y2=min(fh,y+h+MARGIN)
            out.append((x1, y1, x2-x1, y2-y1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager  (inchangé)
# ══════════════════════════════════════════════════════════════════════════════

class TrackingManager:
    def __init__(self, converter, identity_manager: CupIdentityManager):
        self._conv   = converter
        self._im     = identity_manager
        self._use_3d = True
        self._cups:   Dict[int, TrackedCup] = {}
        self._pending: Dict[tuple, int]     = {}
        self._next_id = 0

    def _notify(self):
        positions = {c.cup_id: c.pos_mm for c in self._cups.values() if c.pos_mm}
        self._im.update_trackers(positions)

    def update_tracking(self, frame):
        for cup in list(self._cups.values()):
            ok = cup.update_kcf(frame)
            if ok:
                cup.update_mm(self._conv, self._use_3d); cup.lost_frames = 0
            else:
                cup.lost_frames += 1
        for cid in [c for c,v in self._cups.items() if v.lost_frames >= MAX_LOST_FRAMES]:
            del self._cups[cid]
            print(f"[Manager] Supprimé cup#{cid}")
        self._notify()
        return list(self._cups.values())

    def update_detection(self, frame, detected):
        if not detected:
            self._pending.clear(); return
        cups = list(self._cups.values())
        used_dets = set(); used_cups = set()
        if cups:
            S = np.zeros((len(cups), len(detected)), dtype=np.float32)
            for i, cup in enumerate(cups):
                for j, det in enumerate(detected):
                    S[i, j] = _match_score(cup.bbox, det)
            for idx in np.argsort(-S.ravel()):
                i, j = divmod(int(idx), len(detected))
                if i in used_cups or j in used_dets: continue
                if S[i, j] < MATCH_MIN_SCORE: break
                cup = cups[i]; det = detected[j]
                cx1,cy1 = _center(cup.bbox); cx2,cy2 = _center(det)
                if np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) <  MAX_DRIFT_PX:
                    cup.reinit_kcf(frame, det)
                    cup.update_mm(self._conv, self._use_3d)
                    cup.lost_frames = 0
                used_cups.add(i); used_dets.add(j)
        cur_keys = set()
        for j, det in enumerate(detected):
            if j in used_dets: continue
            cx, cy = _center(det)
            key = (int(cx/20), int(cy/20))
            cur_keys.add(key)
            cnt = self._pending.get(key, 0) + 1
            self._pending[key] = cnt
            if cnt >= STABILITY_FRAMES:
                self._create(frame, det); self._pending.pop(key, None)
        for k in list(self._pending):
            if k not in cur_keys: del self._pending[k]
        self._notify()

    def force_reset(self, frame, detected):
        self._cups.clear()
        for det in detected: self._create(frame, det)
        print(f"[Manager] Reset → {len(detected)} tasse(s)")
        self._notify()

    def _create(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw - 1)); y = max(0, min(y, fh - 1))
        w = max(4, min(w, fw - x)); h = max(4, min(h, fh - y))
        t = cv2.legacy.TrackerMOSSE_create()
        try:
            t.init(frame, (x, y, w, h))
        except Exception as e:
            print(f"[Manager] init KCF: {e}"); return
        cid = self._next_id; self._next_id += 1
        cup = TrackedCup(cup_id=cid, ema=EMAFilter(), kalman_csv=KalmanFilter2D(),
                         cv_tracker=t, bbox=(x, y, w, h))
        cup.update_mm(self._conv, self._use_3d)
        # ── NOUVEAU — enregistrer taille de référence ──────────────────────
        cup.bbox_ref_w = w
        cup.bbox_ref_h = h
        # ──────────────────────────────────────────────────────────────────
        self._cups[cid] = cup
        print(f"[Manager] Nouveau cup#{cid}  bbox=({x},{y},{w},{h})")


# ══════════════════════════════════════════════════════════════════════════════
#  Grille fenêtre PC  (inchangée)
# ══════════════════════════════════════════════════════════════════════════════

def _build_grid_background() -> np.ndarray:
    SIZE = GRID_SIZE
    img  = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255

    X_MIN, X_MAX = GRID_X_MIN, GRID_X_MAX
    Y_MIN, Y_MAX = GRID_Y_MIN, GRID_Y_MAX

    def px_x(v): return int((v - X_MIN) / (X_MAX - X_MIN) * SIZE)
    def px_y(v): return int((v - Y_MIN) / (Y_MAX - Y_MIN) * SIZE)

    def spacing(delta):
        s = delta / 40
        if s <= 0: return 1.0
        p = 10 ** int(np.floor(np.log10(s)))
        r = s / p
        if r < 2: return 2 * p
        if r < 5: return 5 * p
        return 10 * p

    sx = spacing(X_MAX - X_MIN)
    sy = spacing(Y_MAX - Y_MIN)

    x = np.ceil(X_MIN / sx) * sx
    while x <= X_MAX + 1e-9:
        cv2.line(img, (px_x(x), px_y(Y_MIN)), (px_x(x), px_y(Y_MAX)),
                 (211, 211, 211), 1, cv2.LINE_AA)
        x = round(x + sx, 10)

    y = np.ceil(Y_MIN / sy) * sy
    while y <= Y_MAX + 1e-9:
        cv2.line(img, (px_x(X_MIN), px_y(y)), (px_x(X_MAX), px_y(y)),
                 (211, 211, 211), 1, cv2.LINE_AA)
        y = round(y + sy, 10)

    cv2.line(img, (px_x(X_MIN), px_y(0)), (px_x(X_MAX), px_y(0)),
             (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (px_x(0), px_y(Y_MIN)), (px_x(0), px_y(Y_MAX)),
             (0, 0, 0), 2, cv2.LINE_AA)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    color_txt  = (0, 0, 0)

    x = np.ceil(X_MIN / sx) * sx
    while x <= X_MAX + 1e-9:
        xr = round(x, 8)
        if abs(xr) > 1e-6:
            label = f"{xr:g}.0" if xr == int(xr) else f"{xr:g}"
            tw, th = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.putText(img, label,
                        (px_x(xr) + 3, px_y(0) + th + 4),
                        font, font_scale, color_txt, thickness, cv2.LINE_AA)
        x = round(x + sx, 10)

    y = np.ceil(Y_MIN / sy) * sy
    while y <= Y_MAX + 1e-9:
        yr = round(y, 8)
        if abs(yr) > 1e-6:
            label = f"{yr:g}.0" if yr == int(yr) else f"{yr:g}"
            cv2.putText(img, label,
                        (px_x(0) + 5, px_y(yr) - 3),
                        font, font_scale, color_txt, thickness, cv2.LINE_AA)
        y = round(y + sy, 10)

    cv2.putText(img, GRID_X_LEG,
                (px_x(X_MAX) - 20, px_y(0) - 8),
                font, font_scale, color_txt, thickness, cv2.LINE_AA)
    cv2.putText(img, GRID_Y_LEG,
                (px_x(0) - 16, px_y(Y_MAX) + 14),
                font, font_scale, color_txt, thickness, cv2.LINE_AA)

    return img


def _mm_to_grid_px(x_mm: float, y_mm: float) -> Tuple[int, int]:
    x_idx = (x_mm / TABLE_SIZE_MM) * (GRID_X_MAX - GRID_X_MIN) + GRID_X_MIN
    y_idx = (y_mm / TABLE_SIZE_MM) * (GRID_Y_MAX - GRID_Y_MIN) + GRID_Y_MIN
    px = int((x_idx - GRID_X_MIN) / (GRID_X_MAX - GRID_X_MIN) * GRID_SIZE)
    py = int((y_idx - GRID_Y_MIN) / (GRID_Y_MAX - GRID_Y_MIN) * GRID_SIZE)
    return px, py


def _draw_grid_frame(
    grid_bg: np.ndarray,
    cups: list,
    labels: dict,
    identity_manager: CupIdentityManager,
) -> np.ndarray:
    vis = grid_bg.copy()

    for cup in cups:
        if cup.pos_mm is None:
            continue
        ident = identity_manager.get_identity(cup.cup_id)
        if ident is None:
            continue

        x_mm, y_mm = cup.pos_mm
        px, py = _mm_to_grid_px(x_mm, y_mm)

        if not (0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE):
            continue

        cv2.circle(vis, (px, py), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        label = str(ident.aruco_id)
        cv2.putText(vis, label,
                    (px + 9, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  CSV enrichi — 3 sources de coordonnées par tag
# ══════════════════════════════════════════════════════════════════════════════

class CsvWriter:
    """
    Version patchée de CsvWriter avec 4 sources de coordonnées par tasse.

    Colonnes par tasse (ordre inchangé sauf ajout _filtered) :
        ID_{id}_x_ema,      ID_{id}_y_ema       ← EMA lissée (comme x_camtop avant)
        ID_{id}_x_raw,      ID_{id}_y_raw       ← brut KCF   (comme x_tracker avant)
        ID_{id}_x_filtered, ID_{id}_y_filtered  ← Kalman     ← NOUVEAU
        ID_{id}_x_bottom,   ID_{id}_y_bottom    ← ArUco cam_bottom (inchangé)

    Le renommage camtop→ema et tracker→raw est purement documentaire :
    les valeurs sont strictement identiques à la session précédente.
    """

    import csv as _csv
    from datetime import datetime as _dt

    def __init__(self, output_path: str, cup_ids: list):
        import csv
        from datetime import datetime

        self._csv    = csv
        self._dt     = datetime
        self._frame_index = 0
        self._buffer: list = []

        self._fieldnames = ['frame', 'timestamp']
        for cid in cup_ids:
            self._fieldnames += [
                f'ID_{cid}_x_ema',       f'ID_{cid}_y_ema',        # anciennement _camtop
                f'ID_{cid}_x_raw',       f'ID_{cid}_y_raw',        # anciennement _tracker
                f'ID_{cid}_x_filtered',  f'ID_{cid}_y_filtered',   # NOUVEAU Kalman
                f'ID_{cid}_x_bottom',    f'ID_{cid}_y_bottom',     # inchangé
            ]

        self._file   = open(output_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            extrasaction='ignore',
            restval='',
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[CSV] créé — {len(cup_ids)} tasse(s) × 4 sources")
        print(f"[CSV] colonnes : {self._fieldnames}")

    def push(
        self,
        ema_by_aruco:      Dict[int, Tuple[float, float]],  # pos_mm (EMA)
        raw_by_aruco:      Dict[int, Tuple[float, float]],  # pos_mm_raw (brut)
        filtered_by_aruco: Dict[int, Tuple[float, float]],  # pos_mm_kalman (Kalman)
        bottom_by_aruco:   Dict[int, Tuple[float, float]],  # ArUco cam_bottom
    ) -> None:
        from datetime import datetime
        self._frame_index += 1
        row: dict = {
            'frame':     self._frame_index,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        }

        for aruco_id, (x, y) in ema_by_aruco.items():
            row[f'ID_{aruco_id}_x_ema'] = round(float(x), 2)
            row[f'ID_{aruco_id}_y_ema'] = round(float(y), 2)

        for aruco_id, (x, y) in raw_by_aruco.items():
            row[f'ID_{aruco_id}_x_raw'] = round(float(x), 2)
            row[f'ID_{aruco_id}_y_raw'] = round(float(y), 2)

        for aruco_id, (x, y) in filtered_by_aruco.items():
            row[f'ID_{aruco_id}_x_filtered'] = round(float(x), 2)
            row[f'ID_{aruco_id}_y_filtered'] = round(float(y), 2)

        for aruco_id, (x, y) in bottom_by_aruco.items():
            row[f'ID_{aruco_id}_x_bottom'] = round(float(x), 2)
            row[f'ID_{aruco_id}_y_bottom'] = round(float(y), 2)

        self._buffer.append(row)
        if len(self._buffer) >= 50:   # flush toutes les 50 frames (~2s) au lieu de 200
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()

    def close(self):
        self.flush()
        self._file.close()
        print(f"[CSV] fermé → {self._frame_index} frames total")


# ══════════════════════════════════════════════════════════════════════════════
#  CSV brut trackers — zéro donnée perdue, même avant le match
# ══════════════════════════════════════════════════════════════════════════════

class TrackerRawCsvWriter:
    """
    Enregistre TOUTES les positions de TOUS les trackers KCF actifs,
    frame par frame, sans aucune condition sur le match ArUco.

    Une ligne = un tracker actif pour ce frame.
    Si N trackers sont actifs, N lignes sont écrites pour ce frame.

    Colonnes :
        frame        — numéro de frame global (synchronisé avec session_main.csv)
        timestamp    — horodatage ISO milliseconde
        tracker_id   — identifiant interne KCF (0, 1, 2, …)
        x_mm_raw     — position brute KCF, avant filtre EMA (mm)
        y_mm_raw     — position brute KCF, avant filtre EMA (mm)
        x_mm_ema     — position EMA-lissée (mm) — vide si pas encore calculée
        y_mm_ema     — position EMA-lissée (mm) — vide si pas encore calculée
        aruco_id     — tag ArUco associé à ce tracker (vide avant le match)
        state        — état de l'identité : PENDING, MATCHED, AIRBORNE, LOST,
                       ou UNMATCHED si le tracker n'a pas encore d'identité

    Clé de jointure avec session_main.csv : (frame, aruco_id) une fois matché.
    """

    FIELDNAMES = [
        'frame', 'timestamp',
        'tracker_id',
        'x_mm_raw', 'y_mm_raw',
        'x_mm_ema', 'y_mm_ema',
        'aruco_id', 'state',
    ]

    def __init__(self, output_path: str):
        self._frame_index = 0
        self._buffer: list = []

        self._file   = open(output_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
            extrasaction='ignore',
            restval='',
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[CSV-raw] créé → {output_path}")

    def push(
        self,
        frame_index: int,
        cups: list,                          # liste de TrackedCup
        identity_manager: CupIdentityManager,
    ) -> None:
        """
        Écrit une ligne par TrackedCup actif dans ce frame.

        Paramètres :
          frame_index      — le même numéro de frame que session_main.csv
          cups             — liste des TrackedCup retournés par TrackingManager
          identity_manager — pour résoudre tracker_id → aruco_id + state
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        for cup in cups:
            # tracker_id interne KCF
            tid = cup.cup_id

            # Résolution de l'identité — peut être None si pas encore matché
            ident = identity_manager.get_identity(tid)
            aruco_id = ident.aruco_id if ident is not None else ''
            if ident is not None:
                state = ident.state.name          # MATCHED / PENDING / AIRBORNE / LOST
            else:
                state = 'UNMATCHED'               # tracker créé, pas encore en PENDING

            row: dict = {
                'frame':     frame_index,
                'timestamp': ts,
                'tracker_id': tid,
                'aruco_id':  aruco_id,
                'state':     state,
            }

            if cup.pos_mm_raw is not None:
                row['x_mm_raw'] = round(float(cup.pos_mm_raw[0]), 2)
                row['y_mm_raw'] = round(float(cup.pos_mm_raw[1]), 2)

            if cup.pos_mm is not None:
                row['x_mm_ema'] = round(float(cup.pos_mm[0]), 2)
                row['y_mm_ema'] = round(float(cup.pos_mm[1]), 2)

            self._buffer.append(row)

        if len(self._buffer) >= CSV_FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()

    def close(self):
        self.flush()
        self._file.close()
        print("[CSV-raw] fermé")


# ══════════════════════════════════════════════════════════════════════════════
#  Export JSON des associations tracker↔tag
# ══════════════════════════════════════════════════════════════════════════════

def _write_associations_json(
    json_path: str,
    identity_manager: CupIdentityManager,
    participant_id: str,
    protocol_name: str,
    session_start: str,
    total_frames: int,
) -> None:
    """
    Écrit associations.json avec :
      - Métadonnées de session
      - Pour chaque aruco_id :
          · tous les tracker_id qui lui ont été associés (sans doublons)
          · log chronologique de tous les événements bind/unbind
    """
    history  = identity_manager.get_tracker_history_by_aruco()
    full_log = identity_manager.get_association_log()

    associations = {}
    for aruco_id, tracker_ids in history.items():
        associations[str(aruco_id)] = {
            "tracker_ids_used": tracker_ids,
            "bind_count": sum(
                1 for ev in full_log
                if ev["aruco_id"] == aruco_id and ev["event"] == "bind"
            ),
            "unbind_count": sum(
                1 for ev in full_log
                if ev["aruco_id"] == aruco_id and ev["event"] == "unbind"
            ),
        }

    output = {
        "session": {
            "participant_id":  participant_id,
            "protocol_name":   protocol_name,
            "session_start":   session_start,
            "total_frames":    total_frames,
        },
        "associations": associations,
        "event_log":    full_log,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[JSON] associations écrites → {json_path}")
    for aid, info in associations.items():
        print(f"  ArUco#{aid} → trackers utilisés: {info['tracker_ids_used']}  "
              f"(bind×{info['bind_count']} unbind×{info['unbind_count']})")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Napping Lite — pipeline autonome")
    print("=" * 60)

    protocol_name  = input("Nom du protocole  : ").strip() or "PROTO_TEST"
    participant_id = input("ID participant    : ").strip() or "P001"

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_start = datetime.now().isoformat(timespec='seconds')
    basename      = f"{protocol_name}_{participant_id}_{timestamp}"

    out_dir = str(data_path("sessions", "lite"))
    os.makedirs(out_dir, exist_ok=True)

    csv_path         = os.path.join(out_dir, f"{basename}.csv")
    trackers_raw_path = os.path.join(out_dir, f"{basename}_trackers_raw.csv")
    video_path       = os.path.join(out_dir, f"{basename}.mp4")
    json_path        = os.path.join(out_dir, f"{basename}_associations.json")

    print(f"\nCSV principal  → {csv_path}")
    print(f"CSV trackers   → {trackers_raw_path}")
    print(f"Associations   → {json_path}")
    print(f"Vidéo          → {video_path}\n")

    # ── Config ───────────────────────────────────────────────────────────────
    H_proj = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    converter        = PoseConverter(str(config_path("camtop_table_pose.json")))
    identity_manager = CupIdentityManager()
    detector         = CupDetector()
    manager          = TrackingManager(converter=converter, identity_manager=identity_manager)
    csv_writer            = CsvWriter(csv_path, cup_ids=ARUCO_CUP_IDS)
    tracker_raw_writer    = TrackerRawCsvWriter(trackers_raw_path)

    # ── VideoWriter ──────────────────────────────────────────────────────────
    video_writer = VideoWriterThread(
        output_path=video_path, width=VIDEO_W, height=VIDEO_H, fps=VIDEO_FPS)
    video_writer.start()

    # ── CamBottom ────────────────────────────────────────────────────────────
    cam_bot_manager = CameraManager(
        camera_index=CAM_BOT_INDEX,
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    cam_bot_manager.open_camera()
    cam_bot = CamBottomThreadCore(
        camera_manager=cam_bot_manager,
        pose_path=str(config_path("cambottom_table_pose.json")),
        identity_manager=identity_manager,
        valid_cup_ids=set(ARUCO_CUP_IDS),
        show_preview=False)
    cam_bot.start()

    # ── Undistort cam_top ────────────────────────────────────────────────────
    map1 = map2 = None
    try:
        c    = json.load(open(config_path("camera_calibration_top.json")))
        K_r  = np.array(c["camera_matrix"], dtype=np.float64)
        dist = np.array(c["dist_coeffs"],   dtype=np.float64)
        sx = PROCESS_W / CAPTURE_W
        sy = PROCESS_H / CAPTURE_H
        K_small = K_r.copy()
        K_small[0,0] *= sx; K_small[1,1] *= sy
        K_small[0,2] *= sx; K_small[1,2] *= sy
        nK, _ = cv2.getOptimalNewCameraMatrix(
            K_small, dist, (PROCESS_W, PROCESS_H), 1, (PROCESS_W, PROCESS_H))
        map1, map2 = cv2.initUndistortRectifyMap(
            K_small, dist, None, nK, (PROCESS_W, PROCESS_H), cv2.CV_16SC2)
        print("[Main] Undistort cam_top OK")
    except FileNotFoundError:
        print("[Main] Pas de camera_calibration_top.json — frames brutes")

    # ── Projecteur ───────────────────────────────────────────────────────────
    dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
    proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

    # ── Grille fenêtre PC ────────────────────────────────────────────────────
    grid_bg = _build_grid_background()
    print("[Main] Grille PC calculée")

    # ── CamTop ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_TOP_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[Main] ERREUR cam_top {CAM_TOP_INDEX}")
        cam_bot.stop(); video_writer.stop(); return

    # ── Attente cam_bottom ────────────────────────────────────────────────────
    print("[Main] Attente cam_bottom (3s max)…")
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 3.0:
        if cam_bot.get_aruco_positions(): break
        time.sleep(0.05)

    # ── Détection initiale ────────────────────────────────────────────────────
    ret, frame0 = cap.read()
    if not ret:
        print("[Main] Impossible de lire frame0")
        cam_bot.stop(); video_writer.stop(); return
    small0 = cv2.resize(frame0, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_LINEAR)
    if map1 is not None:
        small0 = cv2.remap(small0, map1, map2, cv2.INTER_LINEAR)
    del frame0
    init_bboxes, _ = detector.detect(small0)
    print(f"[Main] Détection initiale : {len(init_bboxes)} tasse(s)")
    manager.force_reset(small0, init_bboxes)

    print("\nq=quitter  r=reset  +/-=seuil  c=3D  i=debug\n")

    use_3d_correction = False
    last_det_t        = time.monotonic()
    fps_t0            = time.monotonic()
    fps_count         = 0
    fps               = 0.0
    frame_count       = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Boucle principale
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        ret, frame_native = cap.read()
        if not ret or frame_native is None:
            continue

        frame_small = cv2.resize(
            frame_native, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_LINEAR)
        if map1 is not None:
            frame_small = cv2.remap(frame_small, map1, map2, cv2.INTER_LINEAR)

        video_writer.push_frame(frame_native)
        del frame_native

        manager._use_3d = use_3d_correction

        cups = manager.update_tracking(frame_small)

        # ── Vérification drift ArUco + suppression immédiate si invalidé ─────
        aruco_pos = cam_bot.get_aruco_positions()
        to_remove = []
        for cup in list(manager._cups.values()):
            ident = identity_manager.get_identity(cup.cup_id)
            if ident is None:
                continue
            if ident.state != CupState.MATCHED:
                cup.aruco_drift_count = 0
                continue
            aruco_mm = aruco_pos.get(ident.aruco_id)
            still_valid = cup.check_aruco_drift(aruco_mm)
            if not still_valid:
                to_remove.append(cup.cup_id)

        for cid in to_remove:
            if cid in manager._cups:
                del manager._cups[cid]
                print(f"[Main] Tracker #{cid} supprimé immédiatement (drift ArUco)")

        cups = list(manager._cups.values())
        
        # if frame_count % 300 == 0 and len(manager._cups) > 0:
        # Cette condition est vraie toutes les 300 frames, soit environ toutes les 12 secondes à 25 FPS.
        # On vérifie qu’il y a au moins une tasse suivie avant de lancer la mise à jour.
        
            # t0 = time.monotonic()
        #On enregistre le temps de départ pour mesurer la durée dexécution du bloc.

            # for cup in manager._cups.values():
            #     cup.update_kcf(frame_small)
        #Pour chaque tasse, on appelle la méthode update_kcf() qui met à jour le tracker KCF avec la nouvelle image (frame_small).

            # kcf_ms = (time.monotonic() - t0) * 1000
        # On calcule le temps total d’exécution en millisecondes
            # print(f"[BENCH] {len(manager._cups)} trackers KCF : {kcf_ms:.1f}ms  ({kcf_ms/len(manager._cups):.1f}ms/tracker)")
            
        
        if frame_count % 300 == 0 and len(manager._cups) > 0:
            import time as _t
            N = 10  # répéter 10 fois pour avoir une mesure stable
            t0 = _t.perf_counter()  # perf_counter > monotonic sur Windows
            for _ in range(N):
                for cup in manager._cups.values():
                    cup.cv_tracker.update(frame_small)
            elapsed_ms = (_t.perf_counter() - t0) * 1000 / N
            n = len(manager._cups)
            print(f"[BENCH] {n} trackers MOSSE : {elapsed_ms:.2f}ms total  "
                f"({elapsed_ms/n:.2f}ms/tracker)  →  max théorique {1000/elapsed_ms:.0f} FPS")

        #Depuis la dernière détection, si plus de DETECT_INTERVAL_S secondes se sont écoulées, on lance une nouvelle détection sur l’image réduite (frame_small) et on met à jour le manager avec les nouvelles détections. On récupère ensuite la liste des tasses suivies (cups) et on met à jour le temps de la dernière détection (last_det_t).
        now = time.monotonic()
        if now - last_det_t >= DETECT_INTERVAL_S:
            det_bboxes, _ = detector.detect(frame_small)
            manager.update_detection(frame_small, det_bboxes)
            cups       = list(manager._cups.values())
            last_det_t = now

        frame_count += 1
        if frame_count % PURGE_EVERY_FRAMES == 0:
            identity_manager.purge_stale_identities(PURGE_MAX_AGE_S)

        labels = identity_manager.get_labels()

        # ── Collecte des 4 sources de positions pour le CSV ──────────────────────────
        ema_by_aruco:      Dict[int, Tuple[float, float]] = {}   # EMA → _ema
        raw_by_aruco:      Dict[int, Tuple[float, float]] = {}   # brut → _raw
        filtered_by_aruco: Dict[int, Tuple[float, float]] = {}   # Kalman → _filtered
        bottom_by_aruco:   Dict[int, Tuple[float, float]] = \
            identity_manager.get_raw_aruco_positions()

        # ── Collecte positions + calcul offset cam_top → cam_bottom ──────────────
        

        for cup in cups:
            ident = identity_manager.get_identity(cup.cup_id)
            if ident is None:
                continue
            aruco_id = ident.aruco_id
            if cup.pos_mm is not None:
                ema_by_aruco[aruco_id] = cup.pos_mm
            if cup.pos_mm_raw is not None:
                raw_by_aruco[aruco_id] = cup.pos_mm_raw
            if cup.pos_mm_kalman is not None:
                filtered_by_aruco[aruco_id] = cup.pos_mm_kalman

            # Mise à jour offset si cam_bottom a une donnée pour cette tasse ce frame
            if cup.pos_mm is not None and aruco_id in bottom_by_aruco:
                bx, by = bottom_by_aruco[aruco_id]
                tx, ty = cup.pos_mm
                dx, dy = bx - tx, by - ty

                if not cup._offset_init:
                    cup._offset_ema_x = dx
                    cup._offset_ema_y = dy
                    cup._offset_init  = True
                else:
                    cup._offset_ema_x = OFFSET_ALPHA * dx + (1 - OFFSET_ALPHA) * cup._offset_ema_x
                    cup._offset_ema_y = OFFSET_ALPHA * dy + (1 - OFFSET_ALPHA) * cup._offset_ema_y

                cup.proj_offset = (cup._offset_ema_x, cup._offset_ema_y)

        # ── Projecteur : fond blanc + cercles ────────────────────────────
        proj_frame[:] = 255

        for cup in cups:
            if cup.pos_mm is None:
                continue
            x_mm = cup.pos_mm[0] + cup.proj_offset[0]
            y_mm = cup.pos_mm[1] + cup.proj_offset[1]
            pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
            pxy = cv2.perspectiveTransform(pt, H_proj)
            px  = int(pxy[0, 0, 0])
            py  = int(pxy[0, 0, 1])
            state = identity_manager.get_state(cup.cup_id)
            color = _identity_color(state)
            label = labels.get(cup.cup_id, f"?#{cup.cup_id}")
            m = RING_RADIUS + RING_THICKNESS + 30
            if m <= px <= PROJ_W - m and m <= py <= PROJ_H - m:
                cv2.circle(proj_frame, (px, py),
                        RING_RADIUS, color, RING_THICKNESS,
                        lineType=cv2.LINE_AA)
                txt_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
                tx = px - txt_size[0] // 2
                ty = py + RING_RADIUS + 60
                cv2.putText(proj_frame, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5,
                            cv2.LINE_AA)

        dm.display_image_on_projector_monitor(proj_frame)

        # ── CSV principal — 1 ligne/frame, indexé par aruco_id ───────────────
        csv_writer.push(ema_by_aruco, raw_by_aruco, filtered_by_aruco, bottom_by_aruco)

        # ── CSV trackers brut — 1 ligne/tracker/frame, zéro perte ────────────
        tracker_raw_writer.push(
            frame_index=csv_writer._frame_index,   # même compteur de frame
            cups=cups,
            identity_manager=identity_manager,
        )

        # ── Fenêtre grille PC ─────────────────────────────────────────────────
        grid_vis = _draw_grid_frame(grid_bg, cups, labels, identity_manager)
        cv2.imshow("Napping Grille", grid_vis)
        del grid_vis

        # ── Preview cam_top ───────────────────────────────────────────────────
        preview = cv2.resize(frame_small, (960, 540))
        ps = 960 / PROCESS_W
        for cup in cups:
            bx, by, bw, bh = cup.bbox
            state = identity_manager.get_state(cup.cup_id)
            color = _identity_color(state)
            cv2.rectangle(preview,
                          (int(bx*ps), int(by*ps)),
                          (int((bx+bw)*ps), int((by+bh)*ps)), color, 2)
            cx, cy = _center(cup.bbox)
            cv2.drawMarker(preview, (int(cx*ps), int(cy*ps)),
                           (0,0,255), cv2.MARKER_CROSS, 14, 2)
            lbl = labels.get(cup.cup_id, f"?#{cup.cup_id}")
            cv2.putText(preview, lbl,
                        (int(bx*ps), max(20, int(by*ps)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(preview,
                    f"FPS:{fps:.1f}  cups:{len(cups)}"
                    f"  seuil:{detector.v_threshold}"
                    f"  3D:{'ON' if use_3d_correction else 'OFF'}"
                    f"  {participant_id}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        cv2.imshow(" CamTop", preview)

        # FPS
        fps_count += 1
        now2 = time.monotonic()
        if now2 - fps_t0 >= 1.0:
            fps       = fps_count / (now2 - fps_t0)
            fps_count = 0; fps_t0 = now2
            print(f"[Main] FPS={fps:.1f}  cups={len(cups)}  frames={frame_count}")

        key = cv2.waitKey(1) & 0xFF
        del preview

        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            detector.v_threshold = min(245, detector.v_threshold + 5)
            print(f"seuil → {detector.v_threshold}")
        elif key == ord('-'):
            detector.v_threshold = max(10, detector.v_threshold - 5)
            print(f"seuil → {detector.v_threshold}")
        elif key == ord('r'):
            det_bboxes, _ = detector.detect(frame_small)
            manager.force_reset(frame_small, det_bboxes)
            identity_manager.reset()
            for cup in manager._cups.values():
                cup.kalman_csv.reset()
            print(f"Reset — seuil={detector.v_threshold}")
        elif key == ord('c'):
            use_3d_correction = not use_3d_correction
            manager._use_3d   = use_3d_correction
            print(f"Correction 3D → {'ON' if use_3d_correction else 'OFF'}")
        elif key == ord('i'):
            print(identity_manager.debug_summary())

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    print("\n[Main] Arrêt...")
    cam_bot.stop()
    cam_bot.wait(2000)
    cam_bot_manager.close_camera()
    cap.release()
    video_writer.stop()
    csv_writer.close()
    tracker_raw_writer.close()

    # Export JSON associations
    _write_associations_json(
        json_path        = json_path,
        identity_manager = identity_manager,
        participant_id   = participant_id,
        protocol_name    = protocol_name,
        session_start    = session_start,
        total_frames     = frame_count,
    )

    cv2.destroyAllWindows()
    proj_frame[:] = 0
    dm.display_image_on_projector_monitor(proj_frame)
    cv2.waitKey(1)
    print(f"\n[Main] Terminé — {frame_count} frames")
    print(f"  CSV principal  : {csv_path}")
    print(f"  CSV trackers   : {trackers_raw_path}")
    print(f"  Associations   : {json_path}")
    print(f"  Vidéo          : {video_path}")


if __name__ == "__main__":
    main()