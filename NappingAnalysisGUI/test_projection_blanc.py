# -*- coding: utf-8 -*-
"""
test_projection_blanc.py  —  VERSION FINALE PATCHÉE
=====================================================
Projecteur : fond blanc + anneaux verts sur les tasses
Détection  : seuillage gris sur frame réduite 640×360
Conversion : pixel → mm via PoseConverter 3D + H_top_to_bottom
Touches    : q=quitter  d=masque  +/-=seuil  r=reset  c=toggle correction 3D

═══════════════════════════════════════════════════════════════════════════════
CHAÎNE DE CONVERSION EXACTE (dans l'ordre) :
  1. frame_native (1920×1080) → undistort (pinhole, map1/map2)
  2. resize → frame_small (640×360) pour la détection
  3. CupDetector → (cx_small, cy_small) = centre BASE de la tasse  [FIX-1]
  4. cx_nat = cx_small * PROC_TO_NATIVE  (×3, small→native)
  5. [optionnel] cup_base_pixel_correction(cx_nat, cy_nat)          [FIX-2]
     → corrige le décalage perspective via rvec/tvec/CUP_HEIGHT_MM
  6. PoseConverter.pixel_to_mm(cx_nat, cy_nat)
       a. rayon → intersection plan table → (x_top, y_top) repère cam_top
       b. perspectiveTransform(H_top_to_bottom) → (x_mm, y_mm) repère commun
  7. Validation bornes table                                        [FIX-4]
  8. Lissage EMA + tracking temporel                                [FIX-3/6]
  9. perspectiveTransform(H_table_to_proj) → pixel projecteur
═══════════════════════════════════════════════════════════════════════════════

CORRECTIONS vs version originale :
  [FIX-1] CENTROÏDE BASE    : fitEllipse sur moitié inférieure du contour
  [FIX-2] CORRECTION 3D     : cup_base_pixel_correction() analytique
  [FIX-3] EMA               : lissage exponentiel α=0.4 sur positions mm
  [FIX-4] BORNES            : rejette positions hors table
  [FIX-5] CIRCULARITÉ       : seuil 0.20→0.30, ratio d'aspect ajouté
  [FIX-6] TRACKING TEMPOREL : association distance pour identité stable
  [FIX-H] H OBLIGATOIRE     : FileNotFoundError si H_top_to_bottom.json absent
           (le fallback miroir Y était faux pour cette orientation de caméra)
  [FIX-scale] PROC→NATIVE   : cx * PROC_TO_NATIVE (multiply ×3, pas divide)
"""

import cv2
import json
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
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

# Détection
V_THRESHOLD    = 110
AREA_MIN       = 800
AREA_MAX       = 25_000
CIRC_MIN       = 0.30        # [FIX-5] était 0.20
ASPECT_MAX     = 3.5
MARGIN         = 20

# [FIX-scale] proc→native : MULTIPLIER par ce facteur (pas diviser)
PROC_TO_NATIVE = CAPTURE_W / PROCESS_W   # 3.0

# [FIX-1] Centroïde base
CUP_HEIGHT_MM       = 90.0   # hauteur physique tasse (mm) — À MESURER
CUP_BASE_FACTOR     = 0.88   # fallback : base ≈ y + h × FACTOR
FIT_ELLIPSE_MIN_PTS = 8

# [FIX-3] EMA
EMA_ALPHA       = 0.40
EMA_MAX_JUMP_MM = 120.0

# [FIX-4] Bornes
BOUNDS_MARGIN_MM = 30.0

# [FIX-6] Tracking temporel
ASSOC_MAX_DIST_MM = 150.0


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-3] Filtre EMA
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
        if (dx*dx + dy*dy) ** 0.5 > self._max_jump:
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
#  [FIX-6] Tracker temporel
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCupTracker:
    def __init__(self, max_dist_mm: float = ASSOC_MAX_DIST_MM):
        self._max_dist  = max_dist_mm
        self._filters:  Dict[int, EMAFilter] = {}
        self._last_pos: Dict[int, Tuple[float, float]] = {}
        self._next_id   = 0

    def update(self, detections_mm: List[Tuple[float, float]]) \
            -> List[Tuple[int, float, float]]:
        if not detections_mm:
            self._filters.clear()
            self._last_pos.clear()
            return []

        ids         = list(self._last_pos.keys())
        result      = []
        used_tracks = set()
        used_dets   = set()

        if ids:
            dists = np.full((len(ids), len(detections_mm)), np.inf)
            for i, tid in enumerate(ids):
                px, py = self._last_pos[tid]
                for j, (dx, dy) in enumerate(detections_mm):
                    dists[i, j] = ((px-dx)**2 + (py-dy)**2) ** 0.5

            flat = np.argsort(dists.ravel())
            for idx in flat:
                i, j = divmod(int(idx), len(detections_mm))
                if i in used_tracks or j in used_dets:
                    continue
                if dists[i, j] > self._max_dist:
                    break
                tid    = ids[i]
                xs, ys = self._filters[tid].update(*detections_mm[j])
                self._last_pos[tid] = (xs, ys)
                result.append((tid, xs, ys))
                used_tracks.add(i)
                used_dets.add(j)

        for j, (xr, yr) in enumerate(detections_mm):
            if j not in used_dets:
                tid = self._next_id
                self._next_id += 1
                f      = EMAFilter()
                xs, ys = f.update(xr, yr)
                self._filters[tid]   = f
                self._last_pos[tid]  = (xs, ys)
                result.append((tid, xs, ys))

        seen = {r[0] for r in result}
        for tid in list(self._filters.keys()):
            if tid not in seen:
                del self._filters[tid]
                del self._last_pos[tid]

        return result

    def reset(self) -> None:
        self._filters.clear()
        self._last_pos.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-1] Centre BASE de la tasse
# ══════════════════════════════════════════════════════════════════════════════

def _base_center_from_contour(cnt: np.ndarray,
                               x: int, y: int, w: int, h: int) \
        -> Tuple[float, float]:
    """
    fitEllipse sur les points du contour dans la moitié inférieure
    de la bbox → centre de l'ellipse de base de la tasse.
    Fallback : centroïde X + cy = y + h × CUP_BASE_FACTOR.
    """
    M  = cv2.moments(cnt)
    cx = M["m10"] / M["m00"] if M["m00"] > 1 else x + w / 2.0

    pts_low = cnt[cnt[:, 0, 1] >= y + h * 0.55]
    if len(pts_low) >= FIT_ELLIPSE_MIN_PTS:
        try:
            (ex, ey), _, _ = cv2.fitEllipse(pts_low)
            if (x - MARGIN <= ex <= x + w + MARGIN and
                    y + h * 0.4 <= ey <= y + h + MARGIN):
                return float(ex), float(ey)
        except cv2.error:
            pass

    return float(cx), float(y + h * CUP_BASE_FACTOR)


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-2] Correction 3D perspective
# ══════════════════════════════════════════════════════════════════════════════

def cup_base_pixel_correction(cx_px: float, cy_px: float,
                               rvec: np.ndarray, tvec: np.ndarray,
                               K: np.ndarray) -> Tuple[float, float]:
    """
    Corrige le pixel centroïde apparent → pixel correspondant à la base (Z=0).
    Entrée/sortie : coordonnées pixels NATIVES undistorduées.
    """
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

    pt_mid = np.array([[x_top, y_top, CUP_HEIGHT_MM / 2.0]], dtype=np.float64)
    proj_mid, _ = cv2.projectPoints(pt_mid, rvec, tvec, K, dist_zero)
    cy_mid = float(proj_mid[0, 0, 1])

    pt_base = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
    proj_base, _ = cv2.projectPoints(pt_base, rvec, tvec, K, dist_zero)
    cx_base = float(proj_base[0, 0, 0])

    delta_y = cy_px - cy_mid
    cy_corr = cy_px + delta_y
    return cx_base, cy_corr


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-4] Validation bornes
# ══════════════════════════════════════════════════════════════════════════════

def is_on_table(x_mm: float, y_mm: float) -> bool:
    lo = -BOUNDS_MARGIN_MM
    hi = TABLE_SIZE_MM + BOUNDS_MARGIN_MM
    return lo <= x_mm <= hi and lo <= y_mm <= hi


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter — chaîne pixel → mm repère commun [FIX-H]
# ══════════════════════════════════════════════════════════════════════════════

class PoseConverter:
    """
    pixel natif undistordu → (x_mm, y_mm) repère commun.

    Chaîne :
      pixel → rayon → plan table → (x_top, y_top) repère cam_top brut
      → H_top_to_bottom → (x_mm, y_mm) repère cam_bottom (commun)

    H_top_to_bottom est OBLIGATOIRE.
    La matrice mesurée montre une rotation ~90° + inversion, pas un simple
    miroir Y. Le fallback de la version originale était incorrect.
    """
    def __init__(self, pose_path: str):
        d          = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec  = np.array(d["rvec"],          dtype=np.float64)
        self.tvec  = np.array(d["tvec"],          dtype=np.float64)
        self.K     = np.array(d["camera_matrix"], dtype=np.float64)
        print(f"[Pose] fx={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}")

        h_path = pose_path.replace("camtop_table_pose.json",
                                   "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(
                f"\n[Pose] ERREUR : H_top_to_bottom.json introuvable → {h_path}\n"
                "Ce fichier est obligatoire (la transformation top→bottom\n"
                "n'est pas un simple miroir Y pour cette configuration).\n"
                "Lance test_camtop_projection.py pour le générer.\n"
            )
        self._H = np.array(
            json.load(open(h_path))["H_top_to_bottom"], dtype=np.float32)
        print("[Pose] H_top_to_bottom ✅")

    def pixel_to_mm(self, u: float, v: float) \
            -> Optional[Tuple[float, float]]:
        """
        Pixel (natif undistordu) → (x_mm, y_mm) repère commun.
        Étape 1 : intersection rayon/plan → repère cam_top brut
        Étape 2 : H_top_to_bottom → repère commun
        """
        K_inv = np.linalg.inv(self.K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray  /= np.linalg.norm(ray)
        R, _  = cv2.Rodrigues(self.rvec)
        n     = R[:, 2]
        o     = self.tvec.reshape(3)
        denom = np.dot(n, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(n, o) / denom
        if t < 0:
            return None
        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - o)
        x_top    = float(pt_table[0])
        y_top    = float(pt_table[1])

        pt2 = cv2.perspectiveTransform(
            np.array([[[x_top, y_top]]], dtype=np.float32), self._H)
        return float(pt2[0, 0, 0]), float(pt2[0, 0, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  Détecteur tasses [FIX-1, FIX-5]
# ══════════════════════════════════════════════════════════════════════════════

class CupDetector:
    def __init__(self):
        self.v_threshold = V_THRESHOLD
        self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) \
            -> Tuple[List[Tuple[float, float, int, int]], np.ndarray]:
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

    def _bboxes(self, frame: np.ndarray, mask: np.ndarray) \
            -> List[Tuple[float, float, int, int]]:
        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (AREA_MIN <= area <= AREA_MAX):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            if 4 * np.pi * area / perim ** 2 < CIRC_MIN:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 0 and w / float(h) > ASPECT_MAX:
                continue
            cx, cy = _base_center_from_contour(cnt, x, y, w, h)
            bx1 = max(0, x - MARGIN);      by1 = max(0, y - MARGIN)
            bx2 = min(fw, x + w + MARGIN); by2 = min(fh, y + h + MARGIN)
            out.append((cx, cy, bx2 - bx1, by2 - by1))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    H_proj = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    # Lève FileNotFoundError si H_top_to_bottom.json absent [FIX-H]
    converter = PoseConverter(str(config_path("camtop_table_pose.json")))
    detector  = CupDetector()
    tracker   = SimpleCupTracker()

    # Undistort maps — résolution native (pixel_to_mm attend du natif)
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

    print("q=quitter  d=masque  +/-=seuil  r=reset  c=toggle correction 3D")
    show_mask         = False
    use_3d_correction = True
    fps_t0            = time.monotonic()
    fps_count         = 0
    fps               = 0.0

    while True:
        ret, frame_native = cap.read()
        if not ret or frame_native is None:
            continue

        # 1. Undistort natif
        if map1 is not None:
            frame_native = cv2.remap(frame_native, map1, map2, cv2.INTER_LINEAR)

        # 2. Resize → proc
        frame_small = cv2.resize(frame_native, (PROCESS_W, PROCESS_H),
                                  interpolation=cv2.INTER_LINEAR)

        # 3. Détection [FIX-1]
        bboxes, mask = detector.detect(frame_small)

        detections_mm: List[Tuple[float, float]] = []
        debug_pts = []

        for (cx_s, cy_s, bw, bh) in bboxes:
            # 4. proc → native  [FIX-scale]  MULTIPLY
            cx_n = cx_s * PROC_TO_NATIVE
            cy_n = cy_s * PROC_TO_NATIVE

            # 5. Correction 3D perspective [FIX-2]
            if use_3d_correction:
                cx_n, cy_n = cup_base_pixel_correction(
                    cx_n, cy_n,
                    converter.rvec, converter.tvec, converter.K)

            # 6. pixel → mm (repère cam_top → H_top_to_bottom → repère commun)
            mm = converter.pixel_to_mm(cx_n, cy_n)
            if mm is None:
                continue
            x_mm, y_mm = mm

            # 7. Validation bornes [FIX-4]
            if not is_on_table(x_mm, y_mm):
                continue

            detections_mm.append((x_mm, y_mm))
            debug_pts.append((cx_s, cy_s, x_mm, y_mm))

        # 8. Tracking + EMA [FIX-3/6]
        tracked = tracker.update(detections_mm)

        # 9. Projection
        proj_frame[:] = 255
        proj_results  = []
        for (tid, x_smooth, y_smooth) in tracked:
            pt  = np.array([[[x_smooth, y_smooth]]], dtype=np.float32)
            pxy = cv2.perspectiveTransform(pt, H_proj)
            px  = int(pxy[0, 0, 0])
            py  = int(pxy[0, 0, 1])
            proj_results.append((tid, x_smooth, y_smooth, px, py))
            m = RING_RADIUS + RING_THICKNESS
            if m <= px <= PROJ_W - m and m <= py <= PROJ_H - m:
                cv2.circle(proj_frame, (px, py),
                           RING_RADIUS, RING_COLOR, RING_THICKNESS,
                           lineType=cv2.LINE_AA)

        dm.display_image_on_projector_monitor(proj_frame)

        # Preview
        preview = cv2.resize(frame_small, (960, 540))
        ps      = 960 / PROCESS_W
        for (cx_s, cy_s, x_mm, y_mm) in debug_pts:
            cv2.drawMarker(preview,
                           (int(cx_s * ps), int(cy_s * ps)),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for (tid, xs, ys, px, py) in proj_results:
            cv2.putText(preview,
                        f"T{tid}: ({xs:.0f},{ys:.0f})mm",
                        (10, 40 + 22 * tid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
        corr_lbl = "3D:ON" if use_3d_correction else "3D:OFF"
        cv2.putText(preview,
                    f"FPS:{fps:.1f}  det:{len(bboxes)}"
                    f"  seuil:{detector.v_threshold}(+/-)  {corr_lbl}(c)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("CamTop preview", preview)
        if show_mask:
            cv2.imshow("Masque", cv2.resize(mask, (960, 540)))

        fps_count += 1
        now = time.monotonic()
        if now - fps_t0 >= 1.0:
            fps       = fps_count / (now - fps_t0)
            fps_count = 0
            fps_t0    = now
            print(f"[Test] FPS={fps:.1f}  det={len(bboxes)}"
                  f"  seuil={detector.v_threshold}  3D={use_3d_correction}")
            for (tid, xs, ys, px, py) in proj_results:
                print(f"  T{tid}  mm=({xs:.1f},{ys:.1f})  proj=({px},{py})")

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
            tracker.reset()
            print(f"Reset — seuil → {V_THRESHOLD}")
        elif key == ord('c'):
            use_3d_correction = not use_3d_correction
            print(f"Correction 3D → {'ON' if use_3d_correction else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    proj_frame[:] = 0
    dm.display_image_on_projector_monitor(proj_frame)
    print("[Test] Terminé")


if __name__ == "__main__":
    main()
