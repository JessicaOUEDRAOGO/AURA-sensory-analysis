# -*- coding: utf-8 -*-
"""
test_projection_blanc.py  —  VERSION KCF ROBUSTE
=================================================
Projecteur : fond blanc + anneaux verts sur les tasses
Tracking   : KCF (visuel, frame-par-frame) + recalage HSV periodique
Conversion : pixel → mm via PoseConverter 3D + H_top_to_bottom
Touches    : q=quitter  d=masque  +/-=seuil  r=reset  c=toggle correction 3D

ARCHITECTURE (inspiree de cup_tracking_pipeline.py) :
  Chaque frame :
    1. KCF update sur frame_small → bbox mise a jour visuellement
    2. Conversion centre bbox → mm via chaine geometrique complete
  Toutes les DETECT_INTERVAL_S secondes :
    3. Detection masque HSV → bboxes
    4. Association greedy (score IoU + distance) → identite stable
    5. Reinitialisation KCF sur bboxes HSV → corrige derive
    6. Creation nouveaux trackers / suppression perdus

  Avantage : le masque ne doit PAS voir la tasse a chaque frame.
  KCF la suit visuellement entre les recalages.
  => main qui approche, mouvement rapide : tracking maintenu.

CHAINE DE CONVERSION :
  1. frame_native (1920x1080) → undistort
  2. resize → frame_small (640x360)
  3. KCF → (cx_proc, cy_proc) centre bbox
  4. cx_nat = cx_proc * PROC_TO_NATIVE (x3)
  5. [optionnel] cup_base_pixel_correction()
  6. PoseConverter.pixel_to_mm() → (x_mm, y_mm) repere commun
  7. Validation bornes table
  8. EMA lissage
  9. perspectiveTransform(H_table_to_proj) → pixel projecteur
"""

import cv2
import json
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRES
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX   = 0
CAPTURE_W      = 1920
CAPTURE_H      = 1080
PROCESS_W      = 640
PROCESS_H      = 360
TABLE_SIZE_MM  = 597.0

PROJ_W         = 3840
PROJ_H         = 2160
PROJ_SCREEN_ID = 1
RING_RADIUS    = 120
RING_THICKNESS = 10
RING_COLOR     = (0, 200, 0)

# Detection masque
V_THRESHOLD    = 110
AREA_MIN       = 800
AREA_MAX       = 40_000
CIRC_MIN       = 0.15
ASPECT_MIN     = 0.25
ASPECT_MAX     = 3.5
MARGIN         = 20

# Conversion
PROC_TO_NATIVE  = CAPTURE_W / PROCESS_W   # 3.0
CUP_HEIGHT_MM   = 95.0

# EMA
EMA_ALPHA       = 0.35
EMA_MAX_JUMP_MM = 200.0

# Bornes table
BOUNDS_MARGIN_MM = 30.0

# KCF + recalage
DETECT_INTERVAL_S = 0.15   # recalage HSV toutes les 150ms
MAX_LOST_FRAMES   = 20     # frames avant suppression tracker
MATCH_MIN_SCORE   = 0.01   # score minimum pour association
MAX_DRIFT_RATIO   = 0.8    # derive max KCF (fraction diagonale bbox)


# ══════════════════════════════════════════════════════════════════════════════
#  EMA Filter
# ══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    def __init__(self, alpha: float = EMA_ALPHA,
                 max_jump: float = EMA_MAX_JUMP_MM):
        self._a        = alpha
        self._max_jump = max_jump
        self._val: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._val is None:
            self._val = (x, y)
            return self._val
        dx, dy = x - self._val[0], y - self._val[1]
        if (dx * dx + dy * dy) ** 0.5 > self._max_jump:
            self._val = (x, y)
            return self._val
        self._val = (
            self._a * x + (1 - self._a) * self._val[0],
            self._a * y + (1 - self._a) * self._val[1],
        )
        return self._val

    def reset(self) -> None:
        self._val = None


# ══════════════════════════════════════════════════════════════════════════════
#  Correction 3D perspective
# ══════════════════════════════════════════════════════════════════════════════

def cup_base_pixel_correction(cx_px: float, cy_px: float,
                               rvec: np.ndarray, tvec: np.ndarray,
                               K: np.ndarray) -> Tuple[float, float]:
    dist_zero = np.zeros((5, 1), dtype=np.float64)
    K_inv = np.linalg.inv(K)
    ray   = K_inv @ np.array([cx_px, cy_px, 1.0], dtype=np.float64)
    ray  /= np.linalg.norm(ray)
    R, _  = cv2.Rodrigues(rvec)
    n, o  = R[:, 2], tvec.reshape(3)
    denom = np.dot(n, ray)
    if abs(denom) < 1e-9:
        return cx_px, cy_px
    t = np.dot(n, o) / denom
    if t < 0:
        return cx_px, cy_px
    pt_cam   = ray * t
    pt_table = R.T @ (pt_cam - o)
    x_top, y_top = float(pt_table[0]), float(pt_table[1])
    pt_mid  = np.array([[x_top, y_top, CUP_HEIGHT_MM / 2.0]], dtype=np.float64)
    proj_mid, _ = cv2.projectPoints(pt_mid, rvec, tvec, K, dist_zero)
    cy_mid  = float(proj_mid[0, 0, 1])
    pt_base = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
    proj_base, _ = cv2.projectPoints(pt_base, rvec, tvec, K, dist_zero)
    cx_base = float(proj_base[0, 0, 0])
    return cx_base, cy_px + (cy_px - cy_mid)


# ══════════════════════════════════════════════════════════════════════════════
#  Validation bornes
# ══════════════════════════════════════════════════════════════════════════════

def is_on_table(x_mm: float, y_mm: float) -> bool:
    lo = -BOUNDS_MARGIN_MM
    hi = TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    return lo <= x_mm <= hi and lo <= y_mm <= hi


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter
# ══════════════════════════════════════════════════════════════════════════════

class PoseConverter:
    def __init__(self, pose_path: str):
        d         = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec = np.array(d["rvec"],          dtype=np.float64)
        self.tvec = np.array(d["tvec"],          dtype=np.float64)
        self.K    = np.array(d["camera_matrix"], dtype=np.float64)
        print(f"[Pose] fx={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}")
        h_path = pose_path.replace("camtop_table_pose.json",
                                   "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(
                f"\n[Pose] ERREUR : H_top_to_bottom.json introuvable → {h_path}\n"
                "Lance test_camtop_projection.py pour le generer.\n"
            )
        self._H = np.array(
            json.load(open(h_path))["H_top_to_bottom"], dtype=np.float32)
        print("[Pose] H_top_to_bottom OK")

    def pixel_to_mm(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        K_inv = np.linalg.inv(self.K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray  /= np.linalg.norm(ray)
        R, _  = cv2.Rodrigues(self.rvec)
        n, o  = R[:, 2], self.tvec.reshape(3)
        denom = np.dot(n, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(n, o) / denom
        if t < 0:
            return None
        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - o)
        pt2 = cv2.perspectiveTransform(
            np.array([[[float(pt_table[0]), float(pt_table[1])]]], dtype=np.float32),
            self._H)
        return float(pt2[0, 0, 0]), float(pt2[0, 0, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  Conversion proc → mm
# ══════════════════════════════════════════════════════════════════════════════

def proc_to_mm(cx_proc: float, cy_proc: float,
               converter: PoseConverter,
               use_3d: bool) -> Optional[Tuple[float, float]]:
    cx_n = cx_proc * PROC_TO_NATIVE
    cy_n = cy_proc * PROC_TO_NATIVE
    if use_3d:
        cx_n, cy_n = cup_base_pixel_correction(
            cx_n, cy_n, converter.rvec, converter.tvec, converter.K)
    mm = converter.pixel_to_mm(cx_n, cy_n)
    if mm is None:
        return None
    if not is_on_table(*mm):
        return None
    return mm


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires geometriques
# ══════════════════════════════════════════════════════════════════════════════

def _center(bbox: Tuple) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _iou(b1: Tuple, b2: Tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def _match_score(b_tracker: Tuple, b_det: Tuple) -> float:
    iou  = _iou(b_tracker, b_det)
    x, y, w, h = b_tracker
    diag = max(np.sqrt(w**2 + h**2), 1.0)
    cx1, cy1 = _center(b_tracker)
    cx2, cy2 = _center(b_det)
    dist_norm = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) / diag
    if dist_norm > 2.0:
        return 0.0
    return iou + 0.4 * max(0.0, 1.0 - dist_norm)


def _drift_ok(prev: Tuple, new: Tuple) -> bool:
    x, y, w, h = prev
    diag = max(np.sqrt(w**2 + h**2), 1.0)
    cx1, cy1 = _center(prev)
    cx2, cy2 = _center(new)
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) < MAX_DRIFT_RATIO * diag


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedCup
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    cup_id:      int
    ema:         EMAFilter
    cv_tracker:  object
    bbox:        Tuple[int, int, int, int]
    pos_mm:      Optional[Tuple[float, float]] = None
    lost_frames: int  = 0
    active:      bool = True

    def update_kcf(self, frame_proc: np.ndarray) -> bool:
        if not self.active:
            return False
        ok, raw = self.cv_tracker.update(frame_proc)
        if ok:
            rx, ry, rw, rh = raw
            new_bbox = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))
            if _drift_ok(self.bbox, new_bbox):
                self.bbox = new_bbox
                return True
            self.active = False
            return False
        self.active = False
        return False

    def reinit_kcf(self, frame_proc: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> bool:
        fh, fw = frame_proc.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw-1))
        y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x))
        h = max(4, min(h, fh-y))
        tracker = cv2.TrackerKCF_create()
        try:
            tracker.init(frame_proc, (x, y, w, h))
            self.cv_tracker = tracker
            self.bbox       = (x, y, w, h)
            self.active     = True
            return True
        except Exception as e:
            print(f"[KCF] Echec reinit cup_id={self.cup_id}: {e}")
            self.active = False
            return False

    def update_mm(self, converter: PoseConverter, use_3d: bool) -> None:
        cx, cy = _center(self.bbox)
        mm = proc_to_mm(cx, cy, converter, use_3d)
        if mm is not None:
            xs, ys = self.ema.update(*mm)
            self.pos_mm = (xs, ys)


# ══════════════════════════════════════════════════════════════════════════════
#  Detecteur masque
# ══════════════════════════════════════════════════════════════════════════════

class CupDetector:
    def __init__(self):
        self.v_threshold = V_THRESHOLD
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray):
        mask = self._mask(frame)
        return self._bboxes(frame, mask), mask

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kc, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._ko, iterations=1)
        mask = self._remove_border_blobs(mask)
        return mask

    @staticmethod
    def _remove_border_blobs(mask: np.ndarray) -> np.ndarray:
        h, w  = mask.shape
        flood = mask.copy()
        for x in range(w):
            if flood[0,   x]: cv2.floodFill(flood, None, (x, 0),   0)
            if flood[h-1, x]: cv2.floodFill(flood, None, (x, h-1), 0)
        for y in range(h):
            if flood[y,   0]: cv2.floodFill(flood, None, (0,   y), 0)
            if flood[y, w-1]: cv2.floodFill(flood, None, (w-1, y), 0)
        return flood

    def _bboxes(self, frame: np.ndarray, mask: np.ndarray):
        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (AREA_MIN <= area <= AREA_MAX):
                continue
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)
            if hull_peri < 1 or hull_area < 1:
                continue
            if 4 * np.pi * hull_area / hull_peri ** 2 < CIRC_MIN:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0 and not (ASPECT_MIN <= w/float(h) <= ASPECT_MAX):
                continue
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(fw, x + w + MARGIN)
            y2 = min(fh, y + h + MARGIN)
            out.append((x1, y1, x2-x1, y2-y1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager
# ══════════════════════════════════════════════════════════════════════════════

class TrackingManager:
    def __init__(self, converter: PoseConverter):
        self._converter = converter
        self._use_3d    = True
        self._cups:     Dict[int, TrackedCup] = {}
        self._next_id   = 0
        self._pending: Dict[Tuple[int,int], int] = {}  # clé→nb frames vues
        self.STABILITY_FRAMES = 8  # ~270ms à 30fps — main détectée < 270ms → ignorée

    def update_tracking(self, frame_proc: np.ndarray) -> List[TrackedCup]:
        """KCF update chaque frame."""
        for cup in list(self._cups.values()):
            ok = cup.update_kcf(frame_proc)
            if ok:
                cup.update_mm(self._converter, self._use_3d)
                cup.lost_frames = 0
            else:
                cup.lost_frames += 1

        for cid in [c for c, v in self._cups.items()
                    if v.lost_frames >= MAX_LOST_FRAMES]:
            del self._cups[cid]
            print(f"[Manager] Supprime cup_id={cid}")

        return list(self._cups.values())

    STABILITY_FRAMES  = 8    # frames pour créer un nouveau track
    RECALIB_MIN_STILL = 5    # frames immobiles avant d'accepter un recalage

    def update_detection(self, frame_proc, detected):
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
                if i in used_cups or j in used_dets:
                    continue
                if scores[i, j] < MATCH_MIN_SCORE:
                    break

                cup = cups[i]
                det = detected[j]

                # ── NOUVEAU : recaler KCF seulement si la tasse est immobile ──
                # Si le centre du track KCF actuel et le centre de la détection
                # HSV sont proches → tasse posée → recalage OK
                # Si ils sont loin → tasse en mouvement (main+tasse fusionnés)
                # → on garde KCF tel quel, pas de recalage
                cx_kcf, cy_kcf = _center(cup.bbox)
                cx_det, cy_det = _center(det)
                dist = np.sqrt((cx_kcf - cx_det)**2 + (cy_kcf - cy_det)**2)

                # Seuil : si la détection HSV a bougé de plus de 15px
                # par rapport au KCF → probablement fusion main+tasse → skip
                if dist < 15:
                    cup.reinit_kcf(frame_proc, det)
                    cup.update_mm(self._converter, self._use_3d)
                    cup.lost_frames = 0

                used_cups.add(i)
                used_dets.add(j)

        # Nouvelles détections → stabilité requise
        current_pending_keys = set()
        for j, det in enumerate(detected):
            if j not in used_dets:
                cx, cy = _center(det)
                key = (int(cx / 20), int(cy / 20))
                current_pending_keys.add(key)
                count = self._pending.get(key, 0) + 1
                self._pending[key] = count
                if count >= self.STABILITY_FRAMES:
                    self._create(frame_proc, det)
                    self._pending.pop(key, None)

        for key in list(self._pending.keys()):
            if key not in current_pending_keys:
                del self._pending[key]

    def force_reset(self, frame_proc: np.ndarray,
                    detected: List[Tuple]) -> None:
        self._cups.clear()
        for det in detected:
            self._create(frame_proc, det)
        print(f"[Manager] Reset → {len(detected)} tasse(s)")

    def _create(self, frame_proc: np.ndarray, bbox: Tuple) -> None:
        fh, fw = frame_proc.shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, fw-1))
        y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x))
        h = max(4, min(h, fh-y))
        tracker = cv2.TrackerKCF_create()
        try:
            tracker.init(frame_proc, (x, y, w, h))
        except Exception as e:
            print(f"[Manager] Echec init KCF: {e}")
            return
        cid = self._next_id
        self._next_id += 1
        cup = TrackedCup(
            cup_id=cid,
            ema=EMAFilter(),
            cv_tracker=tracker,
            bbox=(x, y, w, h),
        )
        cup.update_mm(self._converter, self._use_3d)
        self._cups[cid] = cup
        print(f"[Manager] Nouveau cup_id={cid}  bbox=({x},{y},{w},{h})")

    def reset(self) -> None:
        self._cups.clear()
        self._next_id = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    H_proj = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    converter = PoseConverter(str(config_path("camtop_table_pose.json")))
    detector  = CupDetector()
    manager   = TrackingManager(converter=converter)

    # Undistort maps
    map1 = map2 = None
    try:
        c    = json.load(open(config_path("camera_calibration_top.json")))
        K_r  = np.array(c["camera_matrix"], dtype=np.float64)
        dist = np.array(c["dist_coeffs"],   dtype=np.float64)
        nK, _ = cv2.getOptimalNewCameraMatrix(
            K_r, dist, (CAPTURE_W, CAPTURE_H), 1, (CAPTURE_W, CAPTURE_H))
        map1, map2 = cv2.initUndistortRectifyMap(
            K_r, dist, None, nK, (CAPTURE_W, CAPTURE_H), cv2.CV_16SC2)
        print("[Test] Undistort OK")
    except FileNotFoundError:
        print("[Test] Pas de camera_calibration_top.json — frames brutes")

    dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
    dm.resolution = (PROJ_W, PROJ_H)
    proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[Test] ERREUR cam {CAMERA_INDEX}")
        return

    # Detection initiale sur premiere frame
    ret, frame0 = cap.read()
    if not ret:
        print("[Test] Impossible de lire la premiere frame")
        return
    if map1 is not None:
        frame0 = cv2.remap(frame0, map1, map2, cv2.INTER_LINEAR)
    frame_small0 = cv2.resize(frame0, (PROCESS_W, PROCESS_H))
    init_bboxes, _ = detector.detect(frame_small0)
    print(f"[Test] Detection initiale : {len(init_bboxes)} tasse(s)")
    manager.force_reset(frame_small0, init_bboxes)

    print("q=quitter  d=masque  +/-=seuil  r=reset  c=toggle correction 3D")

    show_mask         = False
    use_3d_correction = True
    last_det_t        = time.monotonic()
    last_mask         = None
    fps_t0            = time.monotonic()
    fps_count         = 0
    fps               = 0.0

    while True:
        ret, frame_native = cap.read()
        if not ret or frame_native is None:
            continue

        if map1 is not None:
            frame_native = cv2.remap(frame_native, map1, map2, cv2.INTER_LINEAR)

        frame_small = cv2.resize(
            frame_native, (PROCESS_W, PROCESS_H),
            interpolation=cv2.INTER_LINEAR)

        manager._use_3d = use_3d_correction

        # KCF update chaque frame
        cups = manager.update_tracking(frame_small)

        # Recalage HSV periodique
        now = time.monotonic()
        if now - last_det_t >= DETECT_INTERVAL_S:
            det_bboxes, last_mask = detector.detect(frame_small)
            manager.update_detection(frame_small, det_bboxes)
            cups       = list(manager._cups.values())
            last_det_t = now

        # Projection
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
            kcf_only = cup.lost_frames > 0
            proj_results.append((cup.cup_id, x_mm, y_mm, px, py, kcf_only))
            m = RING_RADIUS + RING_THICKNESS
            if m <= px <= PROJ_W - m and m <= py <= PROJ_H - m:
                color = (0, 180, 255) if kcf_only else RING_COLOR
                cv2.circle(proj_frame, (px, py),
                           RING_RADIUS, color, RING_THICKNESS,
                           lineType=cv2.LINE_AA)

        dm.display_image_on_projector_monitor(proj_frame)

        # Preview
        preview = cv2.resize(frame_small, (960, 540))
        ps      = 960 / PROCESS_W

        for cup in cups:
            bx, by, bw, bh = cup.bbox
            color = (0, 180, 255) if cup.lost_frames > 0 else (0, 255, 0)
            cv2.rectangle(preview,
                          (int(bx*ps), int(by*ps)),
                          (int((bx+bw)*ps), int((by+bh)*ps)),
                          color, 1)
            cx, cy = _center(cup.bbox)
            cv2.drawMarker(preview,
                           (int(cx*ps), int(cy*ps)),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

        for idx, (tid, xs, ys, px, py, kcf_only) in enumerate(proj_results):
            src       = "KCF" if kcf_only else "OK"
            color_txt = (0, 180, 255) if kcf_only else (0, 220, 80)
            cv2.putText(preview,
                        f"T{tid}[{src}]: ({xs:.0f},{ys:.0f})mm",
                        (10, 40 + 22 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_txt, 1)

        corr_lbl = "3D:ON" if use_3d_correction else "3D:OFF"
        cv2.putText(preview,
                    f"FPS:{fps:.1f}  cups:{len(cups)}"
                    f"  seuil:{detector.v_threshold}(+/-)  {corr_lbl}(c)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("CamTop preview", preview)

        if show_mask and last_mask is not None:
            cv2.imshow("Masque", cv2.resize(last_mask, (960, 540)))

        fps_count += 1
        now2 = time.monotonic()
        if now2 - fps_t0 >= 1.0:
            fps       = fps_count / (now2 - fps_t0)
            fps_count = 0
            fps_t0    = now2
            print(f"[Test] FPS={fps:.1f}  cups={len(cups)}"
                  f"  seuil={detector.v_threshold}  3D={use_3d_correction}")
            for (tid, xs, ys, px, py, kcf_only) in proj_results:
                src = "KCF" if kcf_only else "mask"
                print(f"  T{tid}[{src}]  mm=({xs:.1f},{ys:.1f})"
                      f"  proj=({px},{py})")

        key = cv2.waitKey(1) & 0xFF
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
            print(f"Reset — seuil → {V_THRESHOLD}")
        elif key == ord('c'):
            use_3d_correction = not use_3d_correction
            manager._use_3d   = use_3d_correction
            print(f"Correction 3D → {'ON' if use_3d_correction else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    proj_frame[:] = 0
    dm.display_image_on_projector_monitor(proj_frame)
    print("[Test] Termine")


if __name__ == "__main__":
    main()