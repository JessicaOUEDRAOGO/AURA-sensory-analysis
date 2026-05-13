# -*- coding: utf-8 -*-
"""
cup_tracking_pipeline.py
=========================
Pipeline complet de tracking des tasses — remplace :
  - algorithm_analysis.py
  - cam_top_thread.py
  - projection_loop.py

Architecture directement issue de test_projection_identite.py,
enveloppée dans un QObject compatible avec RecordWindow.

Flux :
  CamBottomThread  →  CupIdentityManager  ←  boucle principale
  cam_top (cv2)    →  KCF + HSV           →  positions mm
  positions mm     →  H_table_to_proj     →  projecteur
  positions mm     →  data_signal         →  RecordWindow.update_ui

Interface publique (identique à Algorithm_Analysis) :
  __init__(parent, display_manager, image_background, ...)
  _prepare_threads()          appelé par RecordWindow après open_camera
  detect_and_process()        démarre la boucle (dans QThread)
  stop()
  set_show_grid(bool)
  update_background_image(np.ndarray)
  state_popUpCamera_changed()
  data_signal(dict)
  finished_signal()
  runtime_K                   injecté par RecordWindow
"""

import cv2
import json
import numpy as np
import os
import time
import threading
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from src.core.utils.paths import config_path, data_path
from src.core.projection.display_manager import DisplayManager
from src.core.vision.camera_manager import CameraManager
from src.core.cup_tracking.cam_bottom_thread import CamBottomThread
from src.core.cup_tracking.cup_identity_manager import CupIdentityManager, CupState
from src.core.config.app_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES — identiques à test_projection_identite.py
# ══════════════════════════════════════════════════════════════════════════════

CAPTURE_W          = 1920
CAPTURE_H          = 1080
PROCESS_W          = 640
PROCESS_H          = 360
PROC_TO_NATIVE     = CAPTURE_W / PROCESS_W      # 3.0
TABLE_SIZE_MM      = 597.0
BOUNDS_MARGIN_MM   = 30.0
CUP_HEIGHT_MM      = 95.0

DETECT_INTERVAL_S  = 0.15
MAX_LOST_FRAMES    = 20
MATCH_MIN_SCORE    = 0.01
MAX_DRIFT_RATIO    = 0.8
STILL_THRESHOLD_PX = 15
STABILITY_FRAMES   = 8

EMA_ALPHA          = 0.35
EMA_MAX_JUMP_MM    = 200.0

V_THRESHOLD        = 110
AREA_MIN           = 800
AREA_MAX           = 40_000
CIRC_MIN           = 0.15
ASPECT_MIN         = 0.25
ASPECT_MAX         = 3.5
MARGIN             = 20

PROJ_W             = 3840
PROJ_H             = 2160
RING_RADIUS        = 120
RING_THICKNESS     = 10

# Couleurs selon état identité
COLOR_MATCHED      = (0, 220, 0)
COLOR_AIRBORNE     = (0, 160, 255)
COLOR_LOST         = (0, 0, 220)
COLOR_UNKNOWN      = (180, 180, 180)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires géométriques — identiques à test_projection_identite.py
# ══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA, max_jump=EMA_MAX_JUMP_MM):
        self._a, self._mj, self._v = alpha, max_jump, None

    def update(self, x, y):
        if self._v is None:
            self._v = (x, y); return self._v
        dx, dy = x - self._v[0], y - self._v[1]
        if (dx*dx + dy*dy)**.5 > self._mj:
            self._v = (x, y); return self._v
        self._v = (self._a*x + (1-self._a)*self._v[0],
                   self._a*y + (1-self._a)*self._v[1])
        return self._v

    def reset(self): self._v = None


def _cup_base_correction(cx, cy, rvec, tvec, K):
    dz = np.zeros((5, 1), dtype=np.float64)
    Ki = np.linalg.inv(K)
    r  = Ki @ np.array([cx, cy, 1.0]); r /= np.linalg.norm(r)
    R, _ = cv2.Rodrigues(rvec)
    n, o = R[:, 2], tvec.reshape(3)
    d = np.dot(n, r)
    if abs(d) < 1e-9: return cx, cy
    t = np.dot(n, o) / d
    if t < 0: return cx, cy
    pt = R.T @ (r*t - o)
    pm, _ = cv2.projectPoints(
        np.array([[pt[0], pt[1], CUP_HEIGHT_MM/2]]), rvec, tvec, K, dz)
    pb, _ = cv2.projectPoints(
        np.array([[pt[0], pt[1], 0.0]]),            rvec, tvec, K, dz)
    return float(pb[0, 0, 0]), cy + (cy - float(pm[0, 0, 1]))


def _is_on_table(x, y):
    lo, hi = -BOUNDS_MARGIN_MM, TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    return lo <= x <= hi and lo <= y <= hi


def _center(b): return b[0] + b[2]/2., b[1] + b[3]/2.


def _iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix*iy; union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.


def _match_score(bt, bd):
    iou = _iou(bt, bd); x,y,w,h = bt
    diag = max((w**2+h**2)**.5, 1.)
    cx1,cy1 = _center(bt); cx2,cy2 = _center(bd)
    dn = ((cx1-cx2)**2+(cy1-cy2)**2)**.5 / diag
    return 0. if dn > 2. else iou + .4*max(0., 1.-dn)


def _drift_ok(prev, new):
    x,y,w,h = prev; diag = max((w**2+h**2)**.5, 1.)
    cx1,cy1 = _center(prev); cx2,cy2 = _center(new)
    return ((cx1-cx2)**2+(cy1-cy2)**2)**.5 < MAX_DRIFT_RATIO * diag


def _identity_color(state: Optional[CupState]) -> tuple:
    if state is None:               return COLOR_UNKNOWN
    if state == CupState.MATCHED:   return COLOR_MATCHED
    if state == CupState.AIRBORNE:  return COLOR_AIRBORNE
    return COLOR_LOST


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter — identique à test_projection_identite.py
# ══════════════════════════════════════════════════════════════════════════════

class _PoseConverter:
    def __init__(self, pose_path: str):
        d         = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec = np.array(d["rvec"],          dtype=np.float64)
        self.tvec = np.array(d["tvec"],          dtype=np.float64)
        self.K    = np.array(d["camera_matrix"], dtype=np.float64)
        h_path = pose_path.replace("camtop_table_pose.json", "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(f"H_top_to_bottom.json manquant → {h_path}")
        self._H = np.array(
            json.load(open(h_path))["H_top_to_bottom"], dtype=np.float32)
        print(f"[Pipeline] PoseConverter OK  fx={self.K[0,0]:.1f}")

    def pixel_to_mm(self, u, v):
        Ki = np.linalg.inv(self.K)
        r  = Ki @ np.array([u, v, 1.0]); r /= np.linalg.norm(r)
        R, _ = cv2.Rodrigues(self.rvec)
        n, o = R[:, 2], self.tvec.reshape(3)
        d = np.dot(n, r)
        if abs(d) < 1e-9: return None
        t = np.dot(n, o) / d
        if t < 0: return None
        pt = R.T @ (r*t - o)
        p2 = cv2.perspectiveTransform(
            np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32), self._H)
        return float(p2[0, 0, 0]), float(p2[0, 0, 1])


def _proc_to_mm(cx_p, cy_p, conv: _PoseConverter):
    cx_n = cx_p * PROC_TO_NATIVE
    cy_n = cy_p * PROC_TO_NATIVE
    cx_n, cy_n = _cup_base_correction(cx_n, cy_n, conv.rvec, conv.tvec, conv.K)
    mm = conv.pixel_to_mm(cx_n, cy_n)
    if mm is None or not _is_on_table(*mm): return None
    return mm


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedCup — identique à test_projection_identite.py
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
            rx,ry,rw,rh = raw
            nb = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))
            if _drift_ok(self.bbox, nb):
                self.bbox = nb; return True
        self.active = False; return False

    def reinit_kcf(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x,y,w,h = bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t = cv2.TrackerKCF_create()
        try:
            t.init(frame, (x,y,w,h))
            self.cv_tracker=t; self.bbox=(x,y,w,h); self.active=True
            return True
        except Exception as e:
            print(f"[Pipeline] reinit KCF cup#{self.cup_id}: {e}")
            self.active=False; return False

    def update_mm(self, conv):
        cx, cy = _center(self.bbox)
        mm = _proc_to_mm(cx, cy, conv)
        if mm:
            xs, ys = self.ema.update(*mm)
            self.pos_mm = (xs, ys)


# ══════════════════════════════════════════════════════════════════════════════
#  CupDetector — identique à test_projection_identite.py
# ══════════════════════════════════════════════════════════════════════════════

class _CupDetector:
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
            x,y,w,h = cv2.boundingRect(cnt)
            if h > 0 and not (ASPECT_MIN <= w/float(h) <= ASPECT_MAX): continue
            x1=max(0,x-MARGIN); y1=max(0,y-MARGIN)
            x2=min(fw,x+w+MARGIN); y2=min(fh,y+h+MARGIN)
            out.append((x1, y1, x2-x1, y2-y1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager — identique à test_projection_identite.py
# ══════════════════════════════════════════════════════════════════════════════

class _TrackingManager:
    def __init__(self, conv: _PoseConverter, identity_manager: CupIdentityManager):
        self._conv             = conv
        self._identity_manager = identity_manager
        self._cups:   Dict[int, TrackedCup] = {}
        self._pending: Dict[tuple, int]     = {}
        self._next_id = 0

    def _notify(self):
        positions = {cup.cup_id: cup.pos_mm
                     for cup in self._cups.values() if cup.pos_mm}
        self._identity_manager.update_trackers(positions)

    def update_tracking(self, frame) -> List[TrackedCup]:
        for cup in list(self._cups.values()):
            ok = cup.update_kcf(frame)
            if ok:
                cup.update_mm(self._conv); cup.lost_frames = 0
            else:
                cup.lost_frames += 1
        for cid in [c for c,v in self._cups.items()
                    if v.lost_frames >= MAX_LOST_FRAMES]:
            del self._cups[cid]
            print(f"[Pipeline] KCF perdu cup#{cid} → supprimé")
        self._notify()
        return list(self._cups.values())

    def update_detection(self, frame, detected):
        if not detected:
            self._pending.clear(); return

        cups      = list(self._cups.values())
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
                if ((cx1-cx2)**2+(cy1-cy2)**2)**.5 < STILL_THRESHOLD_PX:
                    cup.reinit_kcf(frame, det)
                    cup.update_mm(self._conv)
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
                self._create(frame, det)
                self._pending.pop(key, None)
        for k in list(self._pending):
            if k not in cur_keys: del self._pending[k]

        self._notify()

    def force_reset(self, frame, detected):
        self._cups.clear()
        for det in detected:
            self._create(frame, det)
        print(f"[Pipeline] Reset → {len(detected)} tasse(s)")
        self._notify()

    def _create(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x,y,w,h = bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t = cv2.TrackerKCF_create()
        try: t.init(frame, (x,y,w,h))
        except Exception as e:
            print(f"[Pipeline] init KCF: {e}"); return
        cid = self._next_id; self._next_id += 1
        cup = TrackedCup(cup_id=cid, ema=EMAFilter(), cv_tracker=t, bbox=(x,y,w,h))
        cup.update_mm(self._conv)
        self._cups[cid] = cup
        print(f"[Pipeline] Nouveau cup#{cid}  bbox=({x},{y},{w},{h})")


# ══════════════════════════════════════════════════════════════════════════════
#  CupTrackingPipeline — remplace Algorithm_Analysis
# ══════════════════════════════════════════════════════════════════════════════

class CupTrackingPipeline(QObject):
    """
    Remplace Algorithm_Analysis + CamTopThread + ProjectionLoop.
    Interface strictement compatible avec RecordWindow.
    """

    data_signal     = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_manager: DisplayManager,
        image_background: np.ndarray,
        record_window=None,
        output_dir: str = None,
        output_name: str = "data",
        modules_enabled: dict = None,
        assets=None,
        timeline_steps=None,
        protocol=None,
        grid_size: int = 700,
        **kwargs,
    ):
        super().__init__()

        self.parent           = parent        # RecordWindow
        self.display_manager  = display_manager
        self.record_window    = record_window
        self.grid_size        = int(grid_size)
        self.modules_enabled  = modules_enabled or {}
        self.running          = False
        self.show_preview     = False
        self.show_grid        = False

        self.image_background       = self._validate_bg(image_background)
        self.image_background_clean = self.image_background.copy()

        # runtime_K injecté par RecordWindow après undistort
        self.runtime_K: Optional[np.ndarray] = None

        # Chemins de pose
        self._pose_top_path    = str(config_path("camtop_table_pose.json"))
        self._pose_bottom_path = str(config_path("cambottom_table_pose.json"))

        # Identités
        self._identity_manager = CupIdentityManager()

        # Threads / objets créés dans _prepare_threads()
        self._cam_bottom:      Optional[CamBottomThread] = None
        self._cam_bot_manager: Optional[CameraManager]   = None
        self._converter:       Optional[_PoseConverter]  = None
        self._detector:        Optional[_CupDetector]    = None
        self._manager:         Optional[_TrackingManager] = None
        self._H_proj:          Optional[np.ndarray]      = None
        self._map1 = self._map2 = None
        self._threads_ready = False

        # CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            os.makedirs(data_path(), exist_ok=True)
            output_dir = data_path()
        else:
            os.makedirs(output_dir, exist_ok=True)
        self.output_csv  = os.path.join(output_dir, f"{output_name}_{timestamp}.csv")
        self.data_buffer = []

        print("[Pipeline] Initialisé")

    # ── Compatibilité RecordWindow ────────────────────────────────────────────

    def set_hands_provider(self, func): pass
    def on_consigne_key_pressed(self): pass

    def state_popUpCamera_changed(self):
        self.show_preview = not self.show_preview
        print(f"[Pipeline] preview → {'ON' if self.show_preview else 'OFF'}")

    def set_show_grid(self, show: bool):
        self.show_grid = bool(show)

    def update_background_image(self, new_bg: np.ndarray):
        self.image_background = self._validate_bg(new_bg)

    # ── Préparation ──────────────────────────────────────────────────────────

    def _prepare_threads(self) -> None:
        print("[Pipeline] Préparation...")

        # ── H_table_to_proj ──────────────────────────────────────────────────
        try:
            self._H_proj = np.array(
                json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
                dtype=np.float32)
            print("[Pipeline] H_table_to_proj OK")
        except Exception as e:
            print(f"[Pipeline] ERREUR H_table_to_proj : {e}"); return

        # ── PoseConverter cam_top ────────────────────────────────────────────
        try:
            self._converter = _PoseConverter(self._pose_top_path)
        except Exception as e:
            print(f"[Pipeline] ERREUR PoseConverter : {e}"); return

        # ── Undistort cam_top ────────────────────────────────────────────────
        try:
            c    = json.load(open(config_path("camera_calibration_top.json")))
            Kr   = np.array(c["camera_matrix"], dtype=np.float64)
            dist = np.array(c["dist_coeffs"],   dtype=np.float64)
            nK, _ = cv2.getOptimalNewCameraMatrix(
                Kr, dist, (CAPTURE_W, CAPTURE_H), 1, (CAPTURE_W, CAPTURE_H))
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                Kr, dist, None, nK, (CAPTURE_W, CAPTURE_H), cv2.CV_16SC2)
            print("[Pipeline] Undistort cam_top OK")
        except FileNotFoundError:
            print("[Pipeline] Pas de camera_calibration_top.json — frames brutes")

        # ── CameraManager cam_bottom ─────────────────────────────────────────
        cam_bot_id = getattr(
            getattr(self.parent, "parent", None), "camera_bottom_id", 1)
        self._cam_bot_manager = CameraManager(
            camera_index=cam_bot_id,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        )
        try:
            self._cam_bot_manager.open_camera()
            if self.runtime_K is not None:
                self._cam_bot_manager.K = self.runtime_K
            print(f"[Pipeline] CameraManager cam_bottom OK (index={cam_bot_id})")
        except Exception as e:
            print(f"[Pipeline] ERREUR cam_bottom : {e}"); return

        # ── CamBottomThread ──────────────────────────────────────────────────
        self._cam_bottom = CamBottomThread(
            camera_manager   = self._cam_bot_manager,
            pose_path        = self._pose_bottom_path,
            identity_manager = self._identity_manager,
            show_preview     = False,
        )
        if self.runtime_K is not None:
            self._cam_bottom.set_camera_matrix(self.runtime_K)

        # ── Detector + Manager ───────────────────────────────────────────────
        self._detector = _CupDetector()
        self._manager  = _TrackingManager(
            conv=self._converter,
            identity_manager=self._identity_manager,
        )

        self._threads_ready = True
        print("[Pipeline] Prêt")

    # ── Boucle principale ────────────────────────────────────────────────────

    def detect_and_process(self) -> None:
        if not self._threads_ready:
            print("[Pipeline] ERREUR : _prepare_threads() non appelé")
            self.finished_signal.emit(); return

        self.running = True

        # Démarrer cam_bottom
        self._cam_bottom.start()

        # Attendre la première détection ArUco (3s max)
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < 3.0:
            if self._cam_bottom.get_aruco_positions():
                break
            time.sleep(0.05)

        # Ouvrir cam_top
        cam_top_id = getattr(
            getattr(self.parent, "parent", None), "camera_top_id", 0)
        cap = cv2.VideoCapture(cam_top_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[Pipeline] ERREUR cam_top {cam_top_id}")
            self._cam_bottom.stop()
            self.finished_signal.emit(); return

        # Détection initiale
        ret, frame0 = cap.read()
        if ret and frame0 is not None:
            if self._map1 is not None:
                frame0 = cv2.remap(frame0, self._map1, self._map2, cv2.INTER_LINEAR)
            small0 = cv2.resize(frame0, (PROCESS_W, PROCESS_H))
            init_bboxes, _ = self._detector.detect(small0)
            print(f"[Pipeline] Détection initiale : {len(init_bboxes)} tasse(s)")
            self._manager.force_reset(small0, init_bboxes)

        # Frame projecteur pré-allouée
        proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

        last_det_t     = time.monotonic()
        fps_t0         = time.monotonic()
        fps_count      = 0
        fps            = 0.0
        signal_counter = 0   # ← CORRECTION 1 : compteur pour throttler data_signal

        print("[Pipeline] Boucle démarrée")

        while self.running:
            ret, frame_native = cap.read()
            if not ret or frame_native is None:
                time.sleep(0.005); continue

            if self._map1 is not None:
                frame_native = cv2.remap(
                    frame_native, self._map1, self._map2, cv2.INTER_LINEAR)

            frame_small = cv2.resize(
                frame_native, (PROCESS_W, PROCESS_H),
                interpolation=cv2.INTER_LINEAR)

            # KCF update chaque frame
            cups = self._manager.update_tracking(frame_small)

            # Recalage HSV périodique
            now = time.monotonic()
            if now - last_det_t >= DETECT_INTERVAL_S:
                det_bboxes, _ = self._detector.detect(frame_small)
                self._manager.update_detection(frame_small, det_bboxes)
                cups       = list(self._manager._cups.values())
                last_det_t = now

            # Labels d'identité
            labels = self._identity_manager.get_labels()

            # ── Projection ────────────────────────────────────────────────
            proj_frame[:] = 255
            data_out      = []

            for cup in cups:
                if cup.pos_mm is None: continue
                x_mm, y_mm = cup.pos_mm
                pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
                pxy = cv2.perspectiveTransform(pt, self._H_proj)
                px  = int(pxy[0, 0, 0])
                py  = int(pxy[0, 0, 1])

                state = self._identity_manager.get_state(cup.cup_id)
                color = _identity_color(state)
                label = labels.get(cup.cup_id, f"?#{cup.cup_id}")

                # Aruco ID pour data_signal (marker_id ArUco ou cup_id si inconnu)
                ident  = self._identity_manager.get_identity(cup.cup_id)
                out_id = ident.aruco_id if ident else cup.cup_id

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

                data_out.append((out_id, [x_mm, y_mm]))

            # ── Affichage projecteur ──────────────────────────────────────
            self.display_manager.display_image_on_projector_monitor(proj_frame)

            # ── Signal UI throttlé (~10 fps) ──────────────────────────────
            # CORRECTION 2 : on n'émet data_signal qu'une frame sur 3
            # pour ne pas saturer le thread Qt principal et éviter les saccades
            signal_counter += 1
            if signal_counter % 3 == 0:
                self.data_signal.emit({"data": data_out})
            self._save_to_buffer(data_out)

            # ── Preview optionnel ─────────────────────────────────────────
            if self.show_preview:
                self._draw_preview(frame_small, cups, labels)

            # ── waitKey unique par frame ──────────────────────────────────
            # CORRECTION 3 : un seul cv2.waitKey par frame, ici en fin de boucle.
            # Il ne doit plus exister dans DisplayManager ni dans _draw_preview.
            cv2.waitKey(1)

            # ── FPS ───────────────────────────────────────────────────────
            fps_count += 1
            now2 = time.monotonic()
            if now2 - fps_t0 >= 1.0:
                fps       = fps_count / (now2 - fps_t0)
                fps_count = 0; fps_t0 = now2
                print(f"[Pipeline] FPS={fps:.1f}  cups={len(cups)}")

        # ── Nettoyage ─────────────────────────────────────────────────────
        print("[Pipeline] Arrêt...")
        self._cam_bottom.stop()
        self._cam_bottom.wait(2000)
        self._cam_bot_manager.close_camera()
        cap.release()
        if self.show_preview:
            cv2.destroyWindow("Pipeline Preview")
        proj_frame[:] = 0
        self.display_manager.display_image_on_projector_monitor(proj_frame)
        cv2.waitKey(1)   # flush final après l'écran noir
        self._save_csv()
        print("[Pipeline] Terminé")
        self.finished_signal.emit()

    def stop(self) -> None:
        print("[Pipeline] STOP demandé")
        self.running = False

    # ── Preview ──────────────────────────────────────────────────────────────

    def _draw_preview(self, frame_small, cups, labels):
        preview = cv2.resize(frame_small, (960, 540))
        ps      = 960 / PROCESS_W
        for cup in cups:
            bx,by,bw,bh = cup.bbox
            state = self._identity_manager.get_state(cup.cup_id)
            color = _identity_color(state)
            cv2.rectangle(preview,
                          (int(bx*ps), int(by*ps)),
                          (int((bx+bw)*ps), int((by+bh)*ps)), color, 2)
            cx, cy = _center(cup.bbox)
            cv2.drawMarker(preview, (int(cx*ps), int(cy*ps)),
                           (0,0,255), cv2.MARKER_CROSS, 14, 2)
            label = labels.get(cup.cup_id, f"?#{cup.cup_id}")
            cv2.putText(preview, label,
                        (int(bx*ps), max(20, int(by*ps)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("Pipeline Preview", preview)
        # PAS de waitKey ici — géré par la boucle principale (correction 3)

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _save_to_buffer(self, data_out: list) -> None:
        frame_data = {
            "frame":     len(self.data_buffer) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for out_id, pos in data_out:
            frame_data[f"ID_{out_id}_x"] = float(pos[0])
            frame_data[f"ID_{out_id}_y"] = float(pos[1])
        self.data_buffer.append(frame_data)

    def _save_csv(self) -> None:
        if not self.data_buffer: return
        pd.DataFrame(self.data_buffer).to_csv(self.output_csv, index=False)
        print(f"[Pipeline] CSV → {self.output_csv}")

    # ── Utilitaire ────────────────────────────────────────────────────────────

    def is_enabled(self, key: str, default: bool = False) -> bool:
        v = self.modules_enabled.get(key, default)
        if v is None: return False
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return v != 0
        if isinstance(v, str): return v.strip().lower() in ("1","true","yes","on")
        return bool(v)

    @staticmethod
    def _validate_bg(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image_background invalide : type={type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image_background invalide : shape={image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image


# Alias pour compatibilité avec l'import existant dans RecordWindow
Algorithm_Analysis = CupTrackingPipeline
