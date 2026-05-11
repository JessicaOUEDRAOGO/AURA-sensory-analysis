# -*- coding: utf-8 -*-
"""
cam_top_thread.py
=================
Thread caméra du haut — KCF tracking PERMANENT sur toutes les tasses.

Architecture identique à test_projection_blanc.py :
  - KCF tourne en permanence sur toutes les tasses visibles
  - Recalage HSV périodique (toutes les 150ms)
  - Conversion pixel → mm via PoseConverter + H_top_to_bottom
  - EMA sur les positions mm

Rôle dans le système global :
  - Cam_bottom (ArUco) → identité des tasses + position quand POSEE
  - Cam_top (KCF)     → position quand SOULEVEE (tracking continu)
  - CupStateBuffer    → choisit quelle source utiliser selon l'état

Cam_top ne sait pas si une tasse est posée ou levée.
Elle tracke tout, tout le temps. CupStateBuffer décide.

Prints de debug :
  [CamTop] T{marker_id} cam_top=({x:.0f},{y:.0f})mm  cam_bottom=({x:.0f},{y:.0f})mm
"""

import cv2
import json
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.utils.paths import config_path


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CAPTURE_W         = 1920
CAPTURE_H         = 1080
PROCESS_W         = 640
PROCESS_H         = 360
PROC_TO_NATIVE    = CAPTURE_W / PROCESS_W   # 3.0
TABLE_SIZE_MM     = 597.0
BOUNDS_MARGIN_MM  = 30.0
CUP_HEIGHT_MM     = 95.0

# KCF
DETECT_INTERVAL_S  = 0.15   # recalage HSV toutes les 150ms
MAX_LOST_FRAMES    = 20
MATCH_MIN_SCORE    = 0.01
MAX_DRIFT_RATIO    = 0.8
STILL_THRESHOLD_PX = 15     # recalage accepté si détection HSV proche du KCF

# Stabilité création nouveau track (évite de tracker la main)
STABILITY_FRAMES   = 8

# EMA
EMA_ALPHA          = 0.35
EMA_MAX_JUMP_MM    = 200.0

# Détection masque
V_THRESHOLD        = 110
AREA_MIN           = 800
AREA_MAX           = 40_000
CIRC_MIN           = 0.15
ASPECT_MIN         = 0.25
ASPECT_MAX         = 3.5
MARGIN             = 20

# Debug print toutes les N secondes
DEBUG_PRINT_INTERVAL_S = 1.0


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
#  PoseConverter — chaîne exacte de test_projection_blanc.py
# ══════════════════════════════════════════════════════════════════════════════

class _PoseConverter:
    def __init__(self, pose_path: str):
        d         = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec = np.array(d["rvec"],          dtype=np.float64)
        self.tvec = np.array(d["tvec"],          dtype=np.float64)
        self.K    = np.array(d["camera_matrix"], dtype=np.float64)
        print(f"[CamTop] PoseConverter fx={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}")

        h_path = pose_path.replace("camtop_table_pose.json",
                                   "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(
                f"[CamTop] H_top_to_bottom.json introuvable → {h_path}")
        self._H    = np.array(
            json.load(open(h_path))["H_top_to_bottom"], dtype=np.float32)
        self._H_inv = np.linalg.inv(self._H.astype(np.float64))
        print("[CamTop] H_top_to_bottom OK")

    def pixel_to_mm(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Pixel natif undistordu → (x_mm, y_mm) repère commun."""
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

    def mm_to_pixel_proc(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        (x_mm, y_mm) repère commun → pixel proc.
        Inverse de pixel_to_mm — utilisé pour initialiser KCF depuis pos ArUco.
        """
        # Repasser en repère cam_top via H_inv
        pt_top = cv2.perspectiveTransform(
            np.array([[[x_mm, y_mm]]], dtype=np.float32),
            self._H_inv.astype(np.float32))
        x_top = float(pt_top[0, 0, 0])
        y_top = float(pt_top[0, 0, 1])

        # Projeter ce point table (Z=0) → pixel natif
        dist_zero = np.zeros((5, 1), dtype=np.float64)
        pt_3d = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
        px_nat, _ = cv2.projectPoints(
            pt_3d, self.rvec, self.tvec, self.K, dist_zero)
        cx_nat = float(px_nat[0, 0, 0])
        cy_nat = float(px_nat[0, 0, 1])

        # Natif → proc
        return cx_nat / PROC_TO_NATIVE, cy_nat / PROC_TO_NATIVE


# ══════════════════════════════════════════════════════════════════════════════
#  Correction perspective hauteur tasse
# ══════════════════════════════════════════════════════════════════════════════

def _cup_base_correction(cx_px: float, cy_px: float,
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


def _proc_to_mm(cx_proc: float, cy_proc: float,
                converter: _PoseConverter,
                use_3d: bool = True) -> Optional[Tuple[float, float]]:
    cx_n = cx_proc * PROC_TO_NATIVE
    cy_n = cy_proc * PROC_TO_NATIVE
    if use_3d:
        cx_n, cy_n = _cup_base_correction(
            cx_n, cy_n, converter.rvec, converter.tvec, converter.K)
    mm = converter.pixel_to_mm(cx_n, cy_n)
    if mm is None:
        return None
    x_mm, y_mm = mm
    lo, hi = -BOUNDS_MARGIN_MM, TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    if not (lo <= x_mm <= hi and lo <= y_mm <= hi):
        return None
    return x_mm, y_mm


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires géométriques
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
#  TrackedCup — une tasse trackée avec son identité ArUco
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    marker_id:   int      # identité ArUco — JAMAIS changée
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
            print(f"[CamTop] Echec reinit KCF marker_id={self.marker_id}: {e}")
            self.active = False
            return False

    def update_mm(self, converter: _PoseConverter,
                  use_3d: bool = True) -> None:
        cx, cy = _center(self.bbox)
        mm = _proc_to_mm(cx, cy, converter, use_3d)
        if mm is not None:
            xs, ys = self.ema.update(*mm)
            self.pos_mm = (xs, ys)


# ══════════════════════════════════════════════════════════════════════════════
#  Détecteur masque HSV
# ══════════════════════════════════════════════════════════════════════════════

class _CupDetector:
    def __init__(self):
        self.v_threshold = V_THRESHOLD
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mask = self._mask(frame)
        return self._bboxes(frame, mask)

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

    def _bboxes(self, frame: np.ndarray,
                mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
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
            x1 = max(0, x - MARGIN);  y1 = max(0, y - MARGIN)
            x2 = min(fw, x+w+MARGIN); y2 = min(fh, y+h+MARGIN)
            out.append((x1, y1, x2-x1, y2-y1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager — identique à test_projection_blanc
# ══════════════════════════════════════════════════════════════════════════════

class _TrackingManager:
    """
    Tracking permanent comme test_projection_blanc.
    Différence : les tracks sont identifiés par marker_id ArUco
    plutôt que par un ID auto-incrémenté.

    Association ArUco → track :
      - Au démarrage : détection HSV → bboxes → association par proximité
        avec les positions ArUco connues → marker_id assigné
      - Si ArUco inconnu → track avec ID temporaire négatif jusqu'à association
    """

    def __init__(self, converter: _PoseConverter):
        self._converter  = converter
        self._detector   = _CupDetector()
        self._cups:      Dict[int, TrackedCup] = {}   # clé = marker_id
        self._pending:   Dict[Tuple[int, int], int] = {}
        self._last_det_t = 0.0
        self._tmp_id     = -1   # IDs temporaires négatifs avant association ArUco

    def update_tracking(self, frame_proc: np.ndarray) -> List[TrackedCup]:
        """KCF update chaque frame — identique à test_projection_blanc."""
        for cup in list(self._cups.values()):
            ok = cup.update_kcf(frame_proc)
            if ok:
                cup.update_mm(self._converter)
                cup.lost_frames = 0
            else:
                cup.lost_frames += 1

        for mid in [m for m, c in self._cups.items()
                    if c.lost_frames >= MAX_LOST_FRAMES]:
            del self._cups[mid]
            print(f"[CamTop] KCF perdu marker_id={mid} → supprimé")

        return list(self._cups.values())

    def update_detection(self, frame_proc: np.ndarray,
                         known_positions: Dict[int, Tuple[float, float]]) -> None:
        """
        Recalage HSV périodique.
        known_positions : {marker_id: (x_mm, y_mm)} depuis CupStateBuffer.
        Permet d'assigner le bon marker_id aux bboxes HSV.
        """
        now = time.monotonic()
        if now - self._last_det_t < DETECT_INTERVAL_S:
            return
        self._last_det_t = now

        detected = self._detector.detect(frame_proc)
        if not detected:
            self._pending.clear()
            return

        cups      = list(self._cups.values())
        used_dets = set()
        used_cups = set()

        # ── Étape 1 : recaler les tracks existants ────────────────────
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
                cx_kcf, cy_kcf = _center(cup.bbox)
                cx_det, cy_det = _center(det)
                dist = np.sqrt((cx_kcf-cx_det)**2 + (cy_kcf-cy_det)**2)
                if dist < STILL_THRESHOLD_PX:
                    cup.reinit_kcf(frame_proc, det)
                    cup.update_mm(self._converter)
                    cup.lost_frames = 0
                used_cups.add(i)
                used_dets.add(j)

        # ── Étape 2 : nouvelles détections → créer tracks ────────────
        # Stabilité requise (évite de tracker la main)
        current_keys = set()
        for j, det in enumerate(detected):
            if j in used_dets:
                continue
            cx, cy = _center(det)
            key = (int(cx / 20), int(cy / 20))
            current_keys.add(key)
            count = self._pending.get(key, 0) + 1
            self._pending[key] = count
            if count >= STABILITY_FRAMES:
                # Associer au marker_id ArUco le plus proche
                marker_id = self._assign_marker_id(det, known_positions)
                self._create(frame_proc, det, marker_id)
                self._pending.pop(key, None)

        for key in list(self._pending.keys()):
            if key not in current_keys:
                del self._pending[key]

    def init_from_aruco(self, frame_proc: np.ndarray,
                        known_positions: Dict[int, Tuple[float, float]]) -> None:
        """
        Initialisation au démarrage depuis les positions ArUco connues.
        Cherche la bbox HSV la plus proche de chaque tasse connue.
        """
        detected = self._detector.detect(frame_proc)
        if not detected:
            print("[CamTop] init_from_aruco: aucune bbox HSV détectée")
            return

        used_dets = set()
        for marker_id, pos_mm in known_positions.items():
            if marker_id in self._cups:
                continue  # déjà tracké

            # Projeter pos_mm → pixel proc
            cx_ref, cy_ref = self._converter.mm_to_pixel_proc(*pos_mm)

            # Bbox HSV la plus proche
            best_j    = None
            best_dist = float('inf')
            for j, det in enumerate(detected):
                if j in used_dets:
                    continue
                cx_det, cy_det = _center(det)
                d = np.sqrt((cx_det-cx_ref)**2 + (cy_det-cy_ref)**2)
                if d < best_dist:
                    best_dist = d
                    best_j    = j

            # Tolérance large — 200px proc (~200mm sur la table)
            if best_j is not None and best_dist < 200.0:
                self._create(frame_proc, detected[best_j], marker_id)
                used_dets.add(best_j)
                print(f"[CamTop] Init KCF marker_id={marker_id} "
                      f"dist={best_dist:.0f}px bbox={detected[best_j]}")
            else:
                print(f"[CamTop] Init KCF marker_id={marker_id} "
                      f"— bbox trop loin ({best_dist:.0f}px > 200px)")

    def _assign_marker_id(self, bbox: Tuple,
                          known_positions: Dict[int, Tuple[float, float]]) -> int:
        """
        Assigne le marker_id ArUco le plus proche à une bbox HSV.
        Retourne un ID temporaire négatif si aucun ArUco proche.
        """
        cx_proc, cy_proc = _center(bbox)
        best_id   = None
        best_dist = float('inf')
        for marker_id, pos_mm in known_positions.items():
            if marker_id in self._cups:
                continue
            cx_ref, cy_ref = self._converter.mm_to_pixel_proc(*pos_mm)
            d = np.sqrt((cx_proc-cx_ref)**2 + (cy_proc-cy_ref)**2)
            if d < best_dist:
                best_dist = d
                best_id   = marker_id

        if best_id is not None and best_dist < 200.0:
            return best_id

        # Aucun ArUco connu proche → ID temporaire
        tid = self._tmp_id
        self._tmp_id -= 1
        return tid

    def _create(self, frame_proc: np.ndarray,
                bbox: Tuple, marker_id: int) -> None:
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
            print(f"[CamTop] Echec init KCF marker_id={marker_id}: {e}")
            return
        cup = TrackedCup(
            marker_id=marker_id,
            ema=EMAFilter(),
            cv_tracker=tracker,
            bbox=(x, y, w, h),
        )
        cup.update_mm(self._converter)
        self._cups[marker_id] = cup
        print(f"[CamTop] KCF créé marker_id={marker_id} bbox=({x},{y},{w},{h})")

    def get_all_pos_mm(self) -> Dict[int, Tuple[float, float]]:
        return {mid: cup.pos_mm for mid, cup in self._cups.items()
                if cup.pos_mm is not None}


# ══════════════════════════════════════════════════════════════════════════════
#  CamTopThread
# ══════════════════════════════════════════════════════════════════════════════

class CamTopThread(QThread):
    """
    Thread caméra du haut — tracking KCF permanent.

    Signaux :
      fps_signal(float)
      pos_signal(int, float, float) : (marker_id, x_mm, y_mm)
    """

    fps_signal = pyqtSignal(float)
    pos_signal = pyqtSignal(int, float, float)

    def __init__(
        self,
        cup_state_buffer,
        camera_index: int,
        pose_path:    str,
        cam_width:    int  = 1920,
        cam_height:   int  = 1080,
        show_preview: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.cup_state_buffer = cup_state_buffer
        self.camera_index     = camera_index
        self.cam_width        = cam_width
        self.cam_height       = cam_height
        self.show_preview     = show_preview
        self.running          = False

        if not os.path.isfile(pose_path):
            raise FileNotFoundError(
                f"[CamTop] pose_path introuvable → {pose_path}")

        self._converter = _PoseConverter(pose_path)
        self._manager   = _TrackingManager(self._converter)

        self._map1 = self._map2 = None
        self._load_undistort()

        self._fps_count  = 0
        self._fps_t0     = 0.0
        self._debug_t0   = 0.0
        self._initialized = False   # première init depuis ArUco faite?

    def _load_undistort(self) -> None:
        try:
            c    = json.load(open(config_path("camera_calibration_top.json")))
            K_r  = np.array(c["camera_matrix"], dtype=np.float64)
            dist = np.array(c["dist_coeffs"],   dtype=np.float64)
            nK, _ = cv2.getOptimalNewCameraMatrix(
                K_r, dist, (CAPTURE_W, CAPTURE_H), 1, (CAPTURE_W, CAPTURE_H))
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                K_r, dist, None, nK, (CAPTURE_W, CAPTURE_H), cv2.CV_16SC2)
            print("[CamTop] Undistort OK")
        except FileNotFoundError:
            print("[CamTop] Pas de calibration — frames brutes")

    def run(self) -> None:
        self.running    = True
        self._fps_count = 0
        self._fps_t0    = time.monotonic()
        self._debug_t0  = time.monotonic()
        print(f"[CamTop] Thread démarré — cam={self.camera_index}")

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[CamTop] ERREUR : cam {self.camera_index} inaccessible")
            return

        while self.running:
            ret, frame_native = cap.read()
            if not ret or frame_native is None:
                continue

            if self._map1 is not None:
                frame_native = cv2.remap(
                    frame_native, self._map1, self._map2, cv2.INTER_LINEAR)

            frame_proc = cv2.resize(
                frame_native, (PROCESS_W, PROCESS_H),
                interpolation=cv2.INTER_LINEAR)

            # Positions ArUco connues depuis CupStateBuffer
            all_cups       = self.cup_state_buffer.get_all()
            known_positions = {
                mid: (c["last_pos"][0], c["last_pos"][1])
                for mid, c in all_cups.items()
            }

            # Initialisation au démarrage dès qu'ArUco détecte des tasses
            if not self._initialized and known_positions:
                self._manager.init_from_aruco(frame_proc, known_positions)
                self._initialized = True

            # KCF update chaque frame
            cups = self._manager.update_tracking(frame_proc)

            # Recalage HSV périodique
            self._manager.update_detection(frame_proc, known_positions)

            # Publier positions dans CupStateBuffer
            for cup in cups:
                if cup.pos_mm is not None:
                    self.cup_state_buffer.update_from_top(
                        cup.marker_id, list(cup.pos_mm))
                    self.pos_signal.emit(
                        cup.marker_id, cup.pos_mm[0], cup.pos_mm[1])

            # Preview
            if self.show_preview:
                self._draw_preview(frame_proc, cups)

            # FPS + debug print
            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                fps = self._fps_count / (now - self._fps_t0)
                self.fps_signal.emit(fps)
                self._fps_count = 0
                self._fps_t0    = now

                # Print positions cam_top vs cam_bottom
                top_pos  = self._manager.get_all_pos_mm()
                bot_cups = self.cup_state_buffer.get_all()

                print(f"[CamTop] FPS={fps:.1f}  tracks={len(cups)}")
                all_ids = set(top_pos.keys()) | set(bot_cups.keys())
                for mid in sorted(all_ids):
                    t_pos = top_pos.get(mid)
                    b_pos = bot_cups.get(mid, {}).get("last_pos")
                    state = bot_cups.get(mid, {}).get("state", "?")
                    t_str = (f"({t_pos[0]:.0f},{t_pos[1]:.0f})mm"
                             if t_pos else "N/A")
                    b_str = (f"({b_pos[0]:.0f},{b_pos[1]:.0f})mm"
                             if b_pos else "N/A")
                    print(f"  #{mid}[{state}]  "
                          f"cam_top={t_str}  cam_bottom={b_str}")

        cap.release()
        if self.show_preview:
            cv2.destroyWindow("CamTop Preview")
        print("[CamTop] Thread arrêté")

    def stop(self) -> None:
        self.running = False

    def _draw_preview(self, frame_proc: np.ndarray,
                      cups: List[TrackedCup]) -> None:
        preview = cv2.resize(frame_proc, (960, 540))
        ps      = 960 / PROCESS_W
        for cup in cups:
            bx, by, bw, bh = cup.bbox
            col = (0, 180, 255) if cup.lost_frames > 0 else (0, 255, 0)
            cv2.rectangle(preview,
                          (int(bx*ps), int(by*ps)),
                          (int((bx+bw)*ps), int((by+bh)*ps)), col, 2)
            cx, cy = _center(cup.bbox)
            cv2.drawMarker(preview,
                           (int(cx*ps), int(cy*ps)),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            if cup.pos_mm:
                cv2.putText(preview,
                            f"#{cup.marker_id} "
                            f"({cup.pos_mm[0]:.0f},{cup.pos_mm[1]:.0f})mm",
                            (int(bx*ps), max(20, int(by*ps)-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imshow("CamTop Preview", preview)
        cv2.waitKey(1)