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
  - Export CSV coordonnées x,y par tasse, flush toutes les 200 frames
  - Enregistrement vidéo cam_top via VideoWriterThread (non bloquant)
  - Saisie participant / protocole au démarrage via input()

TOUCHES :
  q=quitter  r=reset trackers  +/-=seuil détection
  c=toggle correction 3D  i=debug identités
"""

import cv2
import json
import numpy as np
import os
import pandas as pd
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

# KCF
DETECT_INTERVAL_S = 0.15
MAX_LOST_FRAMES   = 20
MATCH_MIN_SCORE   = 0.01
MAX_DRIFT_RATIO   = 0.8
STABILITY_FRAMES  = 8

# EMA
EMA_ALPHA       = 0.35
EMA_MAX_JUMP_MM = 200.0

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
ARUCO_CUP_IDS = [6, 8]

# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires géométriques
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
    return np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) < MAX_DRIFT_RATIO * diag


def _identity_color(state: Optional[CupState]) -> tuple:
    if state is None:               return COLOR_UNKNOWN
    if state == CupState.MATCHED:   return COLOR_MATCHED
    if state == CupState.AIRBORNE:  return COLOR_AIRBORNE
    return COLOR_LOST


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter
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
#  TrackedCup
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    cup_id:      int
    ema:         EMAFilter
    cv_tracker:  object
    bbox:        tuple
    pos_mm:      Optional[Tuple[float, float]] = None
    lost_frames: int  = 0
    active:      bool = True

    def update_kcf(self, frame):
        if not self.active: return False
        ok, raw = self.cv_tracker.update(frame)
        if ok:
            rx, ry, rw, rh = raw
            nb = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))
            if _drift_ok(self.bbox, nb):
                self.bbox = nb; return True
            self.active = False; return False
        self.active = False; return False

    def reinit_kcf(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t = cv2.TrackerKCF_create()
        try:
            t.init(frame, (x,y,w,h))
            self.cv_tracker=t; self.bbox=(x,y,w,h); self.active=True; return True
        except Exception as e:
            print(f"[KCF] reinit cup#{self.cup_id}: {e}")
            self.active=False; return False

    def update_mm(self, converter, use_3d):
        cx, cy = _center(self.bbox)
        mm = _proc_to_mm(cx, cy, converter, use_3d)
        if mm is not None:
            xs, ys = self.ema.update(*mm)
            self.pos_mm = (xs, ys)


# ══════════════════════════════════════════════════════════════════════════════
#  CupDetector
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
#  TrackingManager
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
                if np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) < 15:
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
        x,y,w,h = bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t = cv2.TrackerKCF_create()
        try: t.init(frame, (x,y,w,h))
        except Exception as e:
            print(f"[Manager] init KCF: {e}"); return
        cid = self._next_id; self._next_id += 1
        cup = TrackedCup(cup_id=cid, ema=EMAFilter(), cv_tracker=t, bbox=(x,y,w,h))
        cup.update_mm(self._conv, self._use_3d)
        self._cups[cid] = cup
        print(f"[Manager] Nouveau cup#{cid}  bbox=({x},{y},{w},{h})")


# ══════════════════════════════════════════════════════════════════════════════
#  Grille fenêtre PC
# ══════════════════════════════════════════════════════════════════════════════

def _build_grid_background() -> np.ndarray:
    """
    Grille 700×700 fidèle à GraphicsScene :
    traits fins gris clair, axes noirs, graduations décimales.
    """
    SIZE = GRID_SIZE
    img  = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255

    X_MIN, X_MAX = GRID_X_MIN, GRID_X_MAX
    Y_MIN, Y_MAX = GRID_Y_MIN, GRID_Y_MAX

    def px_x(v): return int((v - X_MIN) / (X_MAX - X_MIN) * SIZE)
    def px_y(v): return int((v - Y_MIN) / (Y_MAX - Y_MIN) * SIZE)

    # Pas de grille — identique à GraphicsScene.calculate_grid_spacing
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

    # Lignes verticales — gris très clair, épaisseur 1
    x = np.ceil(X_MIN / sx) * sx
    while x <= X_MAX + 1e-9:
        cv2.line(img, (px_x(x), px_y(Y_MIN)), (px_x(x), px_y(Y_MAX)),
                 (211, 211, 211), 1, cv2.LINE_AA)
        x = round(x + sx, 10)

    # Lignes horizontales
    y = np.ceil(Y_MIN / sy) * sy
    while y <= Y_MAX + 1e-9:
        cv2.line(img, (px_x(X_MIN), px_y(y)), (px_x(X_MAX), px_y(y)),
                 (211, 211, 211), 1, cv2.LINE_AA)
        y = round(y + sy, 10)

    # Axes principaux — noirs, épaisseur 2
    cv2.line(img, (px_x(X_MIN), px_y(0)), (px_x(X_MAX), px_y(0)),
             (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (px_x(0), px_y(Y_MIN)), (px_x(0), px_y(Y_MAX)),
             (0, 0, 0), 2, cv2.LINE_AA)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    color_txt  = (0, 0, 0)

    # Graduations axe X — format décimal comme GraphicsScene (1.0, 2.0...)
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

    # Graduations axe Y
    y = np.ceil(Y_MIN / sy) * sy
    while y <= Y_MAX + 1e-9:
        yr = round(y, 8)
        if abs(yr) > 1e-6:
            label = f"{yr:g}.0" if yr == int(yr) else f"{yr:g}"
            cv2.putText(img, label,
                        (px_x(0) + 5, px_y(yr) - 3),
                        font, font_scale, color_txt, thickness, cv2.LINE_AA)
        y = round(y + sy, 10)

    # Légendes axes
    cv2.putText(img, GRID_X_LEG,
                (px_x(X_MAX) - 20, px_y(0) - 8),
                font, font_scale, color_txt, thickness, cv2.LINE_AA)
    cv2.putText(img, GRID_Y_LEG,
                (px_x(0) - 16, px_y(Y_MAX) + 14),
                font, font_scale, color_txt, thickness, cv2.LINE_AA)

    return img

def _mm_to_grid_px(x_mm: float, y_mm: float) -> Tuple[int, int]:
    """
    Convertit une position en mm (espace table 0-597)
    vers un pixel dans la grille 700×700 (espace index -10/+10).

    Même logique que GraphicsScene.pixel_to_index_x/y inversée :
      index = px / (grid_xmax - grid_xmin) * (x_max - x_min) + x_min
    Ici on fait l'inverse :
      px = (index - x_min) / (x_max - x_min) * GRID_SIZE
    Et index = mm / TABLE_SIZE_MM * (x_max - x_min) + x_min
    """
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

        # Point rouge plein — rayon 6, propre
        cv2.circle(vis, (px, py), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Label — juste le numéro ArUco, sans "Cup#"
        label = str(ident.aruco_id)
        cv2.putText(vis, label,
                    (px + 9, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  CSV
# ══════════════════════════════════════════════════════════════════════════════

import csv

class CsvWriter:
    def __init__(self, output_path: str, cup_ids: list):
        self._frame_index = 0
        self._buffer: list = []

        # Colonnes fixes dès le départ — tous les IDs connus
        self._fieldnames = ['frame', 'timestamp']
        for cid in cup_ids:
            self._fieldnames.append(f'ID_{cid}_x')
            self._fieldnames.append(f'ID_{cid}_y')

        self._file   = open(output_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            extrasaction='ignore',
            restval='',   # cellule vide si la tasse n'est pas détectée ce frame
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[CSV] créé — colonnes : {self._fieldnames}")

    def push(self, data_out: list):
        self._frame_index += 1
        row = {
            'frame':     self._frame_index,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        }
        for cup_id, x_mm, y_mm in data_out:
            row[f'ID_{cup_id}_x'] = round(float(x_mm), 2)
            row[f'ID_{cup_id}_y'] = round(float(y_mm), 2)
        self._buffer.append(row)
        if len(self._buffer) >= CSV_FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()
        print(f"[CSV] flush → {self._frame_index} frames")

    def close(self):
        self.flush()
        self._file.close()
        print(f"[CSV] fermé → {self._frame_index} frames total")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Napping Lite — pipeline autonome")
    print("=" * 60)

    protocol_name  = input("Nom du protocole  : ").strip() or "PROTO_TEST"
    participant_id = input("ID participant    : ").strip() or "P001"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename  = f"{protocol_name}_{participant_id}_{timestamp}"

    out_dir = str(data_path("sessions", "lite"))
    os.makedirs(out_dir, exist_ok=True)

    csv_path   = os.path.join(out_dir, f"{basename}.csv")
    video_path = os.path.join(out_dir, f"{basename}.mp4")

    print(f"\nCSV   → {csv_path}")
    print(f"Vidéo → {video_path}\n")

    # ── Config ───────────────────────────────────────────────────────────────
    H_proj = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    converter        = PoseConverter(str(config_path("camtop_table_pose.json")))
    identity_manager = CupIdentityManager()
    detector         = CupDetector()
    manager          = TrackingManager(converter=converter, identity_manager=identity_manager)
    csv_writer = CsvWriter(csv_path, cup_ids=ARUCO_CUP_IDS)

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
        show_preview=False)
    cam_bot.start()

    # ── Undistort cam_top — maps pour PROCESS_W×PROCESS_H ───────────────────
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

    # ── Projecteur — fond blanc (identique test_projection_identite.py) ──────
    dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
    proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

    # ── Grille fenêtre PC — calculée UNE SEULE FOIS ──────────────────────────
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

    use_3d_correction = True
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

        # resize D'ABORD, remap ENSUITE sur 640×360
        frame_small = cv2.resize(
            frame_native, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_LINEAR)
        if map1 is not None:
            frame_small = cv2.remap(frame_small, map1, map2, cv2.INTER_LINEAR)

        # Vidéo native — push avant del (non bloquant)
        video_writer.push_frame(frame_native)
        del frame_native

        manager._use_3d = use_3d_correction

        # KCF update chaque frame
        cups = manager.update_tracking(frame_small)

        # Recalage HSV périodique
        now = time.monotonic()
        if now - last_det_t >= DETECT_INTERVAL_S:
            det_bboxes, _ = detector.detect(frame_small)
            manager.update_detection(frame_small, det_bboxes)
            cups       = list(manager._cups.values())
            last_det_t = now

        # Purge identités stale
        frame_count += 1
        if frame_count % PURGE_EVERY_FRAMES == 0:
            identity_manager.purge_stale_identities(PURGE_MAX_AGE_S)

        labels = identity_manager.get_labels()

        # ── Projecteur : fond blanc + cercles ────────────────────────────────
        proj_frame[:] = 255
        data_out = []

        for cup in cups:
            if cup.pos_mm is None: continue
            x_mm, y_mm = cup.pos_mm
            pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
            pxy = cv2.perspectiveTransform(pt, H_proj)
            px  = int(pxy[0, 0, 0])
            py  = int(pxy[0, 0, 1])

            state = identity_manager.get_state(cup.cup_id)
            color = _identity_color(state)
            label = labels.get(cup.cup_id, f"?#{cup.cup_id}")
            ident = identity_manager.get_identity(cup.cup_id)

            # ── Projecteur — toujours affiché ─────────────────────────────
            m = RING_RADIUS + RING_THICKNESS + 30
            if m <= px <= PROJ_W - m and m <= py <= PROJ_H - m:
                cv2.circle(proj_frame, (px, py),
                        RING_RADIUS, color, RING_THICKNESS,
                        lineType=cv2.LINE_AA)
                txt_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
                tx = px - txt_size[0]//2
                ty = py + RING_RADIUS + 60
                cv2.putText(proj_frame, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5,
                            cv2.LINE_AA)

            # ── CSV — uniquement si identité ArUco confirmée ───────────────
            if ident is not None:
                data_out.append((ident.aruco_id, x_mm, y_mm))

        dm.display_image_on_projector_monitor(proj_frame)

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_writer.push(data_out)

        # ── Fenêtre grille PC — points rouges ─────────────────────────────────
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
    cv2.destroyAllWindows()
    proj_frame[:] = 0
    dm.display_image_on_projector_monitor(proj_frame)
    cv2.waitKey(1)
    print(f"[Main] Terminé — {frame_count} frames")
    print(f"  CSV   : {csv_path}")
    print(f"  Vidéo : {video_path}")


if __name__ == "__main__":
    main()