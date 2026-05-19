# -*- coding: utf-8 -*-
"""
test_projection_identite.py
============================
Fusion cam_top (KCF) + cam_bottom (ArUco) avec identité stable par tasse.

NOUVEAUTÉS PAR RAPPORT À test_projection_blanc.py :
  • CamBottomThread tourne en parallèle et détecte les tags ArUco
  • CupIdentityManager fait le match position mm tag ↔ tracker KCF
  • Chaque tasse affiche son ID ArUco dans la projection
  • L'identité est conservée en vol (tasse levée) puis vérifiée à la repose
  • Couleur du cercle selon l'état : vert=MATCHED, orange=AIRBORNE, rouge=LOST

TOUCHES :
  q=quitter  d=masque  +/-=seuil  r=reset  c=toggle correction 3D
  i=debug identités dans la console
"""

import cv2
import json
import numpy as np
import os

from src.core.vision.camera_manager import CameraManager
from src.core.cup_tracking.cam_bottom_thread import CamBottomThread as CamBottomThreadCore
from src.core.config.app_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


# Importer nos deux nouveaux modules
from src.core.cup_tracking.cup_identity_manager import CupIdentityManager, CupState


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

# Caméra du haut (tracking KCF)
CAM_TOP_INDEX  = 0
CAPTURE_W      = 1920
CAPTURE_H      = 1080
PROCESS_W      = 640
PROCESS_H      = 360
TABLE_SIZE_MM  = 597.0

# Caméra du bas (ArUco)
CAM_BOT_INDEX  = 1   # ← adapter selon votre configuration

# Projecteur
PROJ_W         = 3840
PROJ_H         = 2160
PROJ_SCREEN_ID = 1
RING_RADIUS    = 120
RING_THICKNESS = 10

# Couleurs selon état d'identité
COLOR_MATCHED  = (0, 220, 0)     # vert   — identité confirmée
COLOR_AIRBORNE = (0, 160, 255)   # orange — en vol
COLOR_LOST     = (0, 0, 220)     # rouge  — identité perdue
COLOR_UNKNOWN  = (180, 180, 180) # gris   — pas encore identifié

# Detection masque
V_THRESHOLD    = 110
AREA_MIN       = 800
AREA_MAX       = 40_000
CIRC_MIN       = 0.15
ASPECT_MIN     = 0.25
ASPECT_MAX     = 3.5
MARGIN         = 20

# Conversion
PROC_TO_NATIVE  = CAPTURE_W / PROCESS_W
CUP_HEIGHT_MM   = 95.0

# EMA
EMA_ALPHA       = 0.35
EMA_MAX_JUMP_MM = 200.0

# Bornes table
BOUNDS_MARGIN_MM = 30.0

# KCF
DETECT_INTERVAL_S = 0.15
MAX_LOST_FRAMES   = 20
MATCH_MIN_SCORE   = 0.01
MAX_DRIFT_RATIO   = 0.8

# ArUco
CALIBRATION_TAG_IDS = {40, 41, 42, 43}   # tags de calibration à ignorer


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires (identiques à test_projection_blanc.py)
# ══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA, max_jump=EMA_MAX_JUMP_MM):
        self._a, self._max_jump = alpha, max_jump
        self._val = None

    def update(self, x, y):
        if self._val is None:
            self._val = (x, y); return self._val
        dx, dy = x - self._val[0], y - self._val[1]
        if (dx*dx + dy*dy)**.5 > self._max_jump:
            self._val = (x, y); return self._val
        self._val = (self._a*x + (1-self._a)*self._val[0],
                     self._a*y + (1-self._a)*self._val[1])
        return self._val

    def reset(self): self._val = None


def cup_base_pixel_correction(cx, cy, rvec, tvec, K):
    K_inv = np.linalg.inv(K)
    ray   = K_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
    ray  /= np.linalg.norm(ray)
    R, _  = cv2.Rodrigues(rvec)
    n, o  = R[:, 2], tvec.reshape(3)
    denom = np.dot(n, ray)
    if abs(denom) < 1e-9: return cx, cy
    t = np.dot(n, o) / denom
    if t < 0: return cx, cy
    pt_cam = ray * t
    pt_table = R.T @ (pt_cam - o)
    x_top, y_top = float(pt_table[0]), float(pt_table[1])
    dist_zero = np.zeros((5, 1))
    pt_mid = np.array([[x_top, y_top, CUP_HEIGHT_MM/2]], dtype=np.float64)
    proj_mid, _ = cv2.projectPoints(pt_mid, rvec, tvec, K, dist_zero)
    cy_mid = float(proj_mid[0, 0, 1])
    pt_base = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
    proj_base, _ = cv2.projectPoints(pt_base, rvec, tvec, K, dist_zero)
    return float(proj_base[0, 0, 0]), cy + (cy - cy_mid)


def is_on_table(x, y):
    lo, hi = -BOUNDS_MARGIN_MM, TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    return lo <= x <= hi and lo <= y <= hi


def _center(bbox):
    x, y, w, h = bbox
    return x + w/2.0, y + h/2.0


def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0


def _match_score(b1, b2):
    iou  = _iou(b1, b2)
    x, y, w, h = b1
    diag = max(np.sqrt(w**2+h**2), 1.0)
    cx1, cy1 = _center(b1)
    cx2, cy2 = _center(b2)
    d = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) / diag
    if d > 2.0: return 0.0
    return iou + 0.4 * max(0.0, 1.0 - d)


def _drift_ok(prev, new):
    x, y, w, h = prev
    diag = max(np.sqrt(w**2+h**2), 1.0)
    cx1, cy1 = _center(prev)
    cx2, cy2 = _center(new)
    return np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) < MAX_DRIFT_RATIO * diag


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter (cam_top)
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


def proc_to_mm(cx_proc, cy_proc, converter, use_3d):
    cx_n = cx_proc * PROC_TO_NATIVE
    cy_n = cy_proc * PROC_TO_NATIVE
    if use_3d:
        cx_n, cy_n = cup_base_pixel_correction(
            cx_n, cy_n, converter.rvec, converter.tvec, converter.K)
    mm = converter.pixel_to_mm(cx_n, cy_n)
    if mm is None or not is_on_table(*mm):
        return None
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
            new_bbox = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))
            if _drift_ok(self.bbox, new_bbox):
                self.bbox = new_bbox; return True
            self.active = False; return False
        self.active = False; return False

    def reinit_kcf(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw-1)); y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x)); h = max(4, min(h, fh-y))
        tracker = cv2.TrackerKCF_create()
        try:
            tracker.init(frame, (x, y, w, h))
            self.cv_tracker = tracker; self.bbox = (x,y,w,h)
            self.active = True; return True
        except Exception as e:
            print(f"[KCF] Echec reinit cup_id={self.cup_id}: {e}")
            self.active = False; return False

    def update_mm(self, converter, use_3d):
        cx, cy = _center(self.bbox)
        mm = proc_to_mm(cx, cy, converter, use_3d)
        if mm is not None:
            xs, ys = self.ema.update(*mm)
            self.pos_mm = (xs, ys)


# ══════════════════════════════════════════════════════════════════════════════
#  CupDetector (masque HSV)
# ══════════════════════════════════════════════════════════════════════════════

class CupDetector:
    def __init__(self):
        self.v_threshold = V_THRESHOLD
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    def detect(self, frame):
        mask = self._mask(frame)
        return self._bboxes(frame, mask), mask

    def _mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kc, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._ko, iterations=1)
        return self._remove_border_blobs(mask)

    @staticmethod
    def _remove_border_blobs(mask):
        h, w  = mask.shape
        flood = mask.copy()
        for x in range(w):
            if flood[0,x]:   cv2.floodFill(flood, None, (x,0),   0)
            if flood[h-1,x]: cv2.floodFill(flood, None, (x,h-1), 0)
        for y in range(h):
            if flood[y,0]:   cv2.floodFill(flood, None, (0,y),   0)
            if flood[y,w-1]: cv2.floodFill(flood, None, (w-1,y), 0)
        return flood

    def _bboxes(self, frame, mask):
        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (AREA_MIN <= area <= AREA_MAX): continue
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)
            if hull_peri < 1 or hull_area < 1: continue
            if 4 * np.pi * hull_area / hull_peri**2 < CIRC_MIN: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0 and not (ASPECT_MIN <= w/float(h) <= ASPECT_MAX): continue
            x1 = max(0, x-MARGIN); y1 = max(0, y-MARGIN)
            x2 = min(fw, x+w+MARGIN); y2 = min(fh, y+h+MARGIN)
            out.append((x1, y1, x2-x1, y2-y1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager (avec injection identity_manager)
# ══════════════════════════════════════════════════════════════════════════════

class TrackingManager:
    def __init__(self, converter, identity_manager: CupIdentityManager):
        self._converter        = converter
        self._identity_manager = identity_manager
        self._use_3d           = True
        self._cups:   Dict[int, TrackedCup] = {}
        self._next_id = 0
        self._pending: Dict[tuple, int]     = {}
        self.STABILITY_FRAMES = 8

    def _notify_identity_manager(self) -> None:
        """Envoie les positions actuelles des trackers au gestionnaire d'identité."""
        positions = {cup.cup_id: cup.pos_mm
                     for cup in self._cups.values()
                     if cup.pos_mm is not None}
        self._identity_manager.update_trackers(positions)

    def update_tracking(self, frame):
        for cup in list(self._cups.values()):
            ok = cup.update_kcf(frame)
            if ok:
                cup.update_mm(self._converter, self._use_3d)
                cup.lost_frames = 0
            else:
                cup.lost_frames += 1

        for cid in [c for c, v in self._cups.items()
                    if v.lost_frames >= MAX_LOST_FRAMES]:
            del self._cups[cid]
            print(f"[Manager] Supprime cup_id={cid}")

        self._notify_identity_manager()
        return list(self._cups.values())

    def update_detection(self, frame, detected):
        if not detected:
            self._pending.clear()
            return

        cups      = list(self._cups.values())
        used_dets = set()
        used_cups = set()

        if cups:
            scores = np.zeros((len(cups), len(detected)), dtype=np.float32)
            for i, cup in enumerate(cups):
                for j, det in enumerate(detected):
                    scores[i, j] = _match_score(cup.bbox, det)

            for idx in np.argsort(-scores.ravel()):
                i, j = divmod(int(idx), len(detected))
                if i in used_cups or j in used_dets: continue
                if scores[i, j] < MATCH_MIN_SCORE: break
                cup = cups[i]; det = detected[j]
                cx_kcf, cy_kcf = _center(cup.bbox)
                cx_det, cy_det = _center(det)
                dist = np.sqrt((cx_kcf-cx_det)**2+(cy_kcf-cy_det)**2)
                if dist < 15:
                    cup.reinit_kcf(frame, det)
                    cup.update_mm(self._converter, self._use_3d)
                    cup.lost_frames = 0
                used_cups.add(i); used_dets.add(j)

        current_pending_keys = set()
        for j, det in enumerate(detected):
            if j not in used_dets:
                cx, cy = _center(det)
                key = (int(cx/20), int(cy/20))
                current_pending_keys.add(key)
                count = self._pending.get(key, 0) + 1
                self._pending[key] = count
                if count >= self.STABILITY_FRAMES:
                    self._create(frame, det)
                    self._pending.pop(key, None)

        for key in list(self._pending.keys()):
            if key not in current_pending_keys:
                del self._pending[key]

        self._notify_identity_manager()

    def force_reset(self, frame, detected):
        self._cups.clear()
        for det in detected:
            self._create(frame, det)
        print(f"[Manager] Reset → {len(detected)} tasse(s)")
        self._notify_identity_manager()

    def _create(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw-1)); y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x)); h = max(4, min(h, fh-y))
        tracker = cv2.TrackerKCF_create()
        try:
            tracker.init(frame, (x,y,w,h))
        except Exception as e:
            print(f"[Manager] Echec init KCF: {e}"); return
        cid = self._next_id; self._next_id += 1
        cup = TrackedCup(cup_id=cid, ema=EMAFilter(), cv_tracker=tracker, bbox=(x,y,w,h))
        cup.update_mm(self._converter, self._use_3d)
        self._cups[cid] = cup
        print(f"[Manager] Nouveau cup_id={cid}  bbox=({x},{y},{w},{h})")

    def reset(self):
        self._cups.clear(); self._next_id = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Thread cam_bottom (ArUco) — léger, autonome
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  Affichage — couleur et label selon l'état d'identité
# ══════════════════════════════════════════════════════════════════════════════

def identity_color(state: Optional[CupState]) -> tuple:
    if state is None:               return COLOR_UNKNOWN
    if state == CupState.MATCHED:   return COLOR_MATCHED
    if state == CupState.AIRBORNE:  return COLOR_AIRBORNE
    return COLOR_LOST


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Chargement config ────────────────────────────────────────────────────
    H_proj = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    converter        = PoseConverter(str(config_path("camtop_table_pose.json")))
    identity_manager = CupIdentityManager()
    detector         = CupDetector()
    manager          = TrackingManager(
        converter=converter,
        identity_manager=identity_manager,
    )

    bot_pose_path   = str(config_path("cambottom_table_pose.json"))
    cam_bot_manager = CameraManager(
        camera_index=CAM_BOT_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )
    cam_bot_manager.open_camera()

    cam_bot = CamBottomThreadCore(
        camera_manager   = cam_bot_manager,
        pose_path        = bot_pose_path,
        identity_manager = identity_manager,
        show_preview     = False,
    )
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
        K_small[0, 0] *= sx; K_small[1, 1] *= sy
        K_small[0, 2] *= sx; K_small[1, 2] *= sy
        nK, _ = cv2.getOptimalNewCameraMatrix(
            K_small, dist, (PROCESS_W, PROCESS_H), 1, (PROCESS_W, PROCESS_H))
        map1, map2 = cv2.initUndistortRectifyMap(
            K_small, dist, None, nK, (PROCESS_W, PROCESS_H), cv2.CV_16SC2)
        print("[Main] Undistort OK")
    except FileNotFoundError:
        print("[Main] Pas de camera_calibration_top.json — frames brutes")

    # ── Projecteur ───────────────────────────────────────────────────────────
    dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
    dm.resolution = (PROJ_W, PROJ_H)
    proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

    # ── Capture cam_top ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_TOP_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[Main] ERREUR cam_top {CAM_TOP_INDEX}")
        cam_bot.stop(); return

    # ── Attendre que cam_bottom détecte au moins un tag avant de démarrer ───
    print("[Main] Attente détection initiale cam_bottom (3s max)…")
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 3.0:
        if cam_bot.get_aruco_positions():
            break
        time.sleep(0.05)

    # Détection initiale cam_top
    ret, frame0 = cap.read()
    if not ret:
        print("[Main] Impossible de lire frame0")
        cam_bot.stop(); return
    frame_small0 = cv2.resize(frame0, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_LINEAR)
    if map1 is not None:
        frame_small0 = cv2.remap(frame_small0, map1, map2, cv2.INTER_LINEAR)
    del frame0
    init_bboxes, _ = detector.detect(frame_small0)
    print(f"[Main] Détection initiale : {len(init_bboxes)} tasse(s)")
    manager.force_reset(frame_small0, init_bboxes)

    print("q=quitter  d=masque  +/-=seuil  r=reset  c=toggle 3D  i=debug IDs")

    show_mask         = False
    use_3d_correction = True
    last_det_t        = time.monotonic()
    last_mask         = None
    fps_t0            = time.monotonic()
    fps_count         = 0
    fps               = 0.0
    frame_count       = 0
    while True:
        ret, frame_native = cap.read()
        if not ret or frame_native is None:
            continue

        frame_small = cv2.resize(
            frame_native, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_LINEAR)
        if map1 is not None:
            frame_small = cv2.remap(frame_small, map1, map2, cv2.INTER_LINEAR)
        del frame_native

        manager._use_3d = use_3d_correction

        # KCF update chaque frame
        cups = manager.update_tracking(frame_small)

        # Recalage HSV périodique
        now = time.monotonic()
        if now - last_det_t >= DETECT_INTERVAL_S:
            det_bboxes, last_mask = detector.detect(frame_small)
            manager.update_detection(frame_small, det_bboxes)
            cups       = list(manager._cups.values())
            last_det_t = now
        frame_count += 1
        if frame_count % 300 == 0:
            identity_manager.purge_stale_identities(20.0)

        # Récupérer les labels d'identité
        labels = identity_manager.get_labels()   # {tracker_id: "Cup#N"}

        # ── Projection ───────────────────────────────────────────────────────
        proj_frame[:] = 255
        proj_results  = []
        for cup in cups:
            if cup.pos_mm is None:
                continue
            x_mm, y_mm = cup.pos_mm
            pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
            pxy = cv2.perspectiveTransform(pt, H_proj)
            px  = int(pxy[0, 0, 0])
            py  = int(pxy[0, 0, 1])

            state = identity_manager.get_state(cup.cup_id)
            color = identity_color(state)
            label = labels.get(cup.cup_id, f"?#{cup.cup_id}")

            m = RING_RADIUS + RING_THICKNESS + 30
            if m <= px <= PROJ_W - m and m <= py <= PROJ_H - m:
                # Cercle coloré selon état
                cv2.circle(proj_frame, (px, py),
                           RING_RADIUS, color, RING_THICKNESS,
                           lineType=cv2.LINE_AA)
                # Label ArUco sous le cercle
                txt_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
                tx = px - txt_size[0] // 2
                ty = py + RING_RADIUS + 60
                cv2.putText(proj_frame, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5,
                            cv2.LINE_AA)

            proj_results.append((cup.cup_id, x_mm, y_mm, px, py, label, state))

        dm.display_image_on_projector_monitor(proj_frame)

        # ── Preview cam_top ──────────────────────────────────────────────────
        preview = cv2.resize(frame_small, (960, 540))
        ps      = 960 / PROCESS_W

        for cup in cups:
            bx, by, bw, bh = cup.bbox
            state = identity_manager.get_state(cup.cup_id)
            color = identity_color(state)
            cv2.rectangle(preview,
                          (int(bx*ps), int(by*ps)),
                          (int((bx+bw)*ps), int((by+bh)*ps)),
                          color, 2)
            cx, cy = _center(cup.bbox)
            cv2.drawMarker(preview,
                           (int(cx*ps), int(cy*ps)),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            label = labels.get(cup.cup_id, f"?#{cup.cup_id}")
            cv2.putText(preview, label,
                        (int(bx*ps), int(by*ps) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for idx, (tid, xs, ys, px, py, label, state) in enumerate(proj_results):
            state_str = state.name if state else "UNKNOWN"
            color_txt = identity_color(state)
            cv2.putText(preview,
                        f"{label}[{state_str}]: ({xs:.0f},{ys:.0f})mm",
                        (10, 40 + 22*idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_txt, 1)

        corr_lbl = "3D:ON" if use_3d_correction else "3D:OFF"
        cv2.putText(preview,
                    f"FPS:{fps:.1f}  cups:{len(cups)}"
                    f"  seuil:{detector.v_threshold}  {corr_lbl}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Overlay positions ArUco (petits cercles bleus)
        for aid, (ax, ay) in cam_bot.get_aruco_positions().items():
            pt  = np.array([[[ax, ay]]], dtype=np.float32)
            # Projeter en pixels preview via H inverse (approx)
            # On utilise juste pour debug un affichage mm en texte
            cv2.putText(preview,
                        f"A{aid}:({ax:.0f},{ay:.0f})",
                        (10, 540 - 20 - aid*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        cv2.imshow("CamTop preview", preview)

        if show_mask and last_mask is not None:
            cv2.imshow("Masque", cv2.resize(last_mask, (960, 540)))

        # FPS
        fps_count += 1
        now2 = time.monotonic()
        if now2 - fps_t0 >= 1.0:
            fps       = fps_count / (now2 - fps_t0)
            fps_count = 0
            fps_t0    = now2
        
        # ── Touches ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        del preview
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Masque")
        elif key in (ord('+'), ord('=')):
            detector.v_threshold = min(245, detector.v_threshold + 5)
            print(f"seuil → {detector.v_threshold}")
        elif key == ord('-'):
            detector.v_threshold = max(10, detector.v_threshold - 5)
            print(f"seuil → {detector.v_threshold}")
        elif key == ord('r'):
            detector.v_threshold = V_THRESHOLD
            det_bboxes, _ = detector.detect(frame_small)
            manager.force_reset(frame_small, det_bboxes)
            identity_manager.reset()
            print(f"Reset — seuil → {V_THRESHOLD}")
        elif key == ord('c'):
            use_3d_correction = not use_3d_correction
            manager._use_3d   = use_3d_correction
            print(f"Correction 3D → {'ON' if use_3d_correction else 'OFF'}")
        elif key == ord('i'):
            print(identity_manager.debug_summary())

    # ── Nettoyage ────────────────────────────────────────────────────────────
    cam_bot.stop()
    cam_bot.wait(2000)
    cam_bot_manager.close_camera()
    cap.release()
    cv2.destroyAllWindows()
    proj_frame[:] = 0
    dm.display_image_on_projector_monitor(proj_frame)
    print("[Main] Terminé")


if __name__ == "__main__":
    main()
