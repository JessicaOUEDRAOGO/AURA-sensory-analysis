# -*- coding: utf-8 -*-
"""
cup_tracking_pipeline.py
========================
Pipeline : detection HSV (top-view) + tracking CSRT + conversion px->mm

La detection YOLO COCO echoue en vue top-down (modele entraine sur vues
de cote). On utilise a la place une segmentation HSV robuste adaptee a
des tasses blanches/creme vues du dessus sur fond gris uniforme.

Architecture :
  HsvCupDetector  : segmentation HSV -> contours -> bboxes
  TrackingManager : N trackers CSRT, association par IoU avec les detections
  CupTopTracker   : tracker CSRT + conversion pixel -> mm (fourni)

Dépendances :
    pip install opencv-contrib-python numpy

Usage :
    python cup_tracking_pipeline.py
    python cup_tracking_pipeline.py --camera 2
    python cup_tracking_pipeline.py --video test.mp4
    python cup_tracking_pipeline.py --pose camtop_table_pose.json

Touches :
    q / Echap  -> quitter
    r          -> redétection HSV forcée
    d          -> debug : affiche le masque HSV
    +/-        -> ajuster V_min en live (luminosite minimale des tasses)
    p          -> pause
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Résolution des chemins ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_CONFIG_DIR = _SCRIPT_DIR.parents[2] / "config" / "pose"
_SEARCH_DIRS = [_CONFIG_DIR, _SCRIPT_DIR, Path.cwd()]


def _resolve_path(p: str) -> str:
    c = Path(p)
    if c.is_absolute():
        return str(c)
    for base in _SEARCH_DIRS:
        r = base / p
        if r.exists():
            return str(r.resolve())
    return str(_CONFIG_DIR / p)

# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

try:
    from src.core.Hand_tracking.cup_top_tracker import CupTopTracker
except ImportError as exc:
    raise ImportError(
        "Impossible d'importer CupTopTracker. "
        "Placez cup_top_tracker.py dans le meme dossier."
    ) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  Parametres de detection HSV — calibres pour tasses blanches/creme top-view
# ══════════════════════════════════════════════════════════════════════════════

# Plage HSV : tasses blanches/creme (faible saturation, haute valeur)
# H : 0-180 (toutes teintes, car blanc = pas de teinte)
# S : 0-80  (peu sature -> blanc / creme)
# V : 160-255 (lumineux)
HSV_LOWER = np.array([0,   0, 150], dtype=np.uint8)
HSV_UPPER = np.array([180, 90, 255], dtype=np.uint8)

# Aire contour en pixels carres
AREA_MIN = 1_500
AREA_MAX = 60_000

# Circularite minimale (1.0 = cercle parfait) — tasse vue de dessus ~ 0.5-0.8
CIRCULARITY_MIN = 0.40

# Ratio W/H de la bounding rect (tasse ~ronde)
ASPECT_MIN = 0.35
ASPECT_MAX = 2.8

# Marge en pixels autour de la bbox detecee avant tracking
BBOX_MARGIN = 18

# IOU pour appariement detection <-> tracker existant
IOU_MATCH_THRESHOLD = 0.20

# Frames perdues avant suppression du tracker
MAX_LOST_FRAMES = 60

# Intervalle entre deux passes de detection (secondes)
DETECT_INTERVAL_S = 2.0

# Couleurs par cup_id (BGR)
_PALETTE = [
    (0,   210,  80),
    (0,   160, 255),
    (60,   60, 220),
    (200,   0, 200),
    (0,   220, 220),
    (0,   180, 255),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Detecteur HSV — specialise vue top-down
# ══════════════════════════════════════════════════════════════════════════════

class HsvCupDetector:
    """
    Detecte les tasses blanches/creme vues de dessus par segmentation HSV.

    Approche :
      1. Convertir BGR -> HSV
      2. Appliquer un masque de couleur (blanc/creme : S faible, V eleve)
      3. Fermeture morphologique pour boucher l'interieur de la tasse
      4. Extraction de contours + filtres geometriques (aire, circularite, ratio)

    Ajustement live : modifier v_min avec les touches + et - pendant l'execution.
    """

    def __init__(self,
                 hsv_lower:       np.ndarray = HSV_LOWER.copy(),
                 hsv_upper:       np.ndarray = HSV_UPPER.copy(),
                 area_min:        int   = AREA_MIN,
                 area_max:        int   = AREA_MAX,
                 circularity_min: float = CIRCULARITY_MIN,
                 margin:          int   = BBOX_MARGIN):
        self.hsv_lower       = hsv_lower.copy()
        self.hsv_upper       = hsv_upper.copy()
        self.area_min        = area_min
        self.area_max        = area_max
        self.circularity_min = circularity_min
        self.margin          = margin
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))

    # ── API ───────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Retourne [(x, y, w, h), ...] des tasses detectees."""
        mask = self._mask(frame)
        return self._bboxes(frame, mask)

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Retourne le masque binaire (pour overlay debug)."""
        return self._mask(frame)

    @property
    def v_min(self) -> int:
        return int(self.hsv_lower[2])

    @v_min.setter
    def v_min(self, v: int) -> None:
        self.hsv_lower[2] = np.clip(v, 0, 254)

    # ── Implementation ────────────────────────────────────────────────────────

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # Fermeture : bouche l'interieur creux de la tasse (reflet sombre)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=3)
        # Ouverture : retire les petits artefacts lumineux (reflets, bordures)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
        return mask

    def _bboxes(self,
                frame: np.ndarray,
                mask:  np.ndarray) -> List[Tuple[int, int, int, int]]:
        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (self.area_min <= area <= self.area_max):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim < 1.0:
                continue
            circ = 4.0 * np.pi * area / (perim ** 2)
            if circ < self.circularity_min:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)
            if not (ASPECT_MIN <= ratio <= ASPECT_MAX):
                continue

            # Marge + clamp
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(fw, x + w + self.margin)
            y2 = min(fh, y + h + self.margin)
            out.append((x1, y1, x2 - x1, y2 - y1))

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Structure TrackedCup
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    cup_id:      int
    tracker:     CupTopTracker
    bbox:        Tuple[int, int, int, int]
    pos_mm:      Optional[Tuple[float, float]] = None
    lost_frames: int = 0
    color:       Tuple[int, int, int] = field(
        default_factory=lambda: (0, 200, 80)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Gestionnaire de trackers CSRT
# ══════════════════════════════════════════════════════════════════════════════

def _iou(b1: Tuple, b2: Tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


class TrackingManager:
    """
    Orchestre N trackers CSRT.
    Cycle par frame :
      1. update CSRT    -> nouvelle position de chaque tasse
      2. associate      -> appariement detection HSV <-> tracker (IoU)
      3. new            -> nouvelles detections -> nouveaux trackers
      4. purge          -> trackers perdus depuis trop longtemps
    """

    def __init__(self, pose_path: str):
        self._pose_path = pose_path
        self._cups:    Dict[int, TrackedCup] = {}
        self._next_id: int = 0

    def update(self,
               frame:    np.ndarray,
               detected: List[Tuple[int, int, int, int]]
               ) -> List[TrackedCup]:

        # 1. Update CSRT
        for cup in list(self._cups.values()):
            ok, pos_mm = cup.tracker.update(frame)
            if ok:
                cup.pos_mm      = pos_mm
                cup.lost_frames = 0
                # Recuperer la bbox interne du tracker OpenCV
                _, raw = cup.tracker._tracker.update(frame)
                rx, ry, rw, rh = raw
                cup.bbox = (int(rx), int(ry), max(4, int(rw)), max(4, int(rh)))
            else:
                cup.lost_frames += 1

        # 2. Association detection -> tracker (greedy IoU)
        unmatched = list(detected)
        for cup in self._cups.values():
            if not unmatched:
                break
            ious   = [_iou(cup.bbox, b) for b in unmatched]
            best_i = int(np.argmax(ious))
            if ious[best_i] >= IOU_MATCH_THRESHOLD:
                new_bbox = unmatched.pop(best_i)
                cup.bbox        = new_bbox
                cup.lost_frames = 0
                if not cup.tracker.is_active:
                    cup.tracker.start(frame, new_bbox, cup.cup_id)
                    print(f"[Manager] Reinit CSRT cup_id={cup.cup_id}")

        # 3. Nouvelles detections
        for bbox in unmatched:
            self._create(frame, bbox)

        # 4. Purge
        for cid in [c for c, v in self._cups.items()
                    if v.lost_frames >= MAX_LOST_FRAMES]:
            self._cups[cid].tracker.stop()
            del self._cups[cid]
            print(f"[Manager] Supprime cup_id={cid}")

        return list(self._cups.values())

    def force_reset(self,
                    frame:    np.ndarray,
                    detected: List[Tuple[int, int, int, int]]) -> None:
        for cup in self._cups.values():
            cup.tracker.stop()
        self._cups.clear()
        for bbox in detected:
            self._create(frame, bbox)
        print(f"[Manager] Reset -> {len(detected)} tasse(s)")

    def _create(self, frame: np.ndarray, bbox: Tuple) -> None:
        cid     = self._next_id
        self._next_id += 1
        tracker = CupTopTracker(self._pose_path)
        ok = tracker.start(frame, bbox, cid)
        if ok:
            self._cups[cid] = TrackedCup(
                cup_id=cid,
                tracker=tracker,
                bbox=bbox,
                color=_PALETTE[cid % len(_PALETTE)],
            )
            print(f"[Manager] Nouveau tracker cup_id={cid} bbox={bbox}")
        else:
            print(f"[Manager] Echec CSRT bbox={bbox}")


# ══════════════════════════════════════════════════════════════════════════════
#  Rendu visuel
# ══════════════════════════════════════════════════════════════════════════════

def _dashed_rect(img, pt1, pt2, color, thickness=1, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


def draw_overlay(frame:    np.ndarray,
                 cups:     List[TrackedCup],
                 det_bboxes: List[Tuple[int, int, int, int]],
                 mask:     Optional[np.ndarray],
                 debug:    bool,
                 fps:      float,
                 v_min:    int) -> np.ndarray:
    out = frame.copy()

    # Overlay masque HSV (debug)
    if debug and mask is not None:
        green = np.zeros_like(out)
        green[:, :, 1] = mask
        out = cv2.addWeighted(out, 0.65, green, 0.35, 0)
        # Bboxes brutes de detection
        for (x, y, w, h) in det_bboxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)

    # Trackers CSRT
    for cup in cups:
        x, y, w, h = cup.bbox
        col  = cup.color
        lost = cup.lost_frames > 0

        if lost:
            _dashed_rect(out, (x, y), (x + w, y + h), col, 2)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)

        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(out, (cx, cy), col,
                       cv2.MARKER_CROSS, markerSize=18, thickness=2)

        if cup.pos_mm is not None:
            xmm, ymm = cup.pos_mm
            label = f"#{cup.cup_id}  {xmm:+.1f} , {ymm:+.1f} mm"
        else:
            label = f"#{cup.cup_id}  perdu ({cup.lost_frames}f)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        ty = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 8, ty + 2),
                      (15, 15, 15), -1)
        cv2.putText(out, label, (x + 4, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 1, cv2.LINE_AA)

    # HUD
    lines = [
        f"FPS: {fps:.1f}",
        f"Tasses: {len(cups)}",
        f"V_min: {v_min}  (+/-)",
    ]
    if debug:
        lines.append("[DEBUG HSV ON]")
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (10, 26 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (210, 210, 210), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def run(camera_index: int  = 0,
        video_path:   str  = "",
        pose_path:    str  = "camtop_table_pose.json",
        undistort:    bool = False,
        dist_path:    str  = "") -> None:

    pose_path = Path(pose_path)

    if not pose_path.is_absolute():
        pose_path = Path(__file__).resolve().parents[3] / "config" / pose_path

    if not pose_path.exists():
        raise FileNotFoundError(
            f"camtop_table_pose.json introuvable.\n"
            
        )

    # Source video
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"[Pipeline] Video : {video_path}")
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[Pipeline] Camera index={camera_index}")

    if not cap.isOpened():
        raise IOError("Impossible d'ouvrir la source video.")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Pipeline] Resolution : {fw}x{fh}")

    # Undistort
    map1 = map2 = None
    if undistort and dist_path and Path(dist_path).exists():
        import json as _json
        d    = _json.load(open(dist_path))
        K    = np.array(d["camera_matrix"], dtype=np.float64)
        dist = np.array(d["dist_coeffs"],   dtype=np.float64)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (fw, fh), cv2.CV_16SC2)
        print("[Pipeline] Undistort active.")

    detector = HsvCupDetector()
    manager  = TrackingManager(pose_path)

    # Premiere frame
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Impossible de lire la premiere frame.")
    if map1 is not None:
        frame0 = cv2.remap(frame0, map1, map2, cv2.INTER_LINEAR)

    init_det = detector.detect(frame0)
    print(f"[Pipeline] Detection initiale : {len(init_det)} tasse(s)"
          f"  (V_min={detector.v_min})")
    manager.force_reset(frame0, init_det)

    # Etat
    debug_mode     = False
    paused         = False
    last_det_t     = time.monotonic()
    t_prev         = time.monotonic()
    measured_fps   = 0.0
    last_det: List[Tuple[int, int, int, int]] = init_det
    last_mask: Optional[np.ndarray] = None

    print("[Pipeline] q=quitter  r=redetection  d=debug  +/-=V_min  p=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("[Pipeline] Fin de la video.")
                break
            if map1 is not None:
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Detection periodique
        now = time.monotonic()
        if not paused and (now - last_det_t) >= DETECT_INTERVAL_S:
            last_det   = detector.detect(frame)
            last_mask  = detector.mask(frame) if debug_mode else None
            last_det_t = now
            if last_det:
                print(f"\n[HSV] {len(last_det)} detection(s)")

        # Debug mask toujours a jour si actif
        if debug_mode and not paused:
            last_mask = detector.mask(frame)

        # Update trackers
        cups = manager.update(frame, last_det if not paused else [])

        # Console
        if cups and not paused:
            parts = [
                f"#{c.cup_id}=({c.pos_mm[0]:+.1f},{c.pos_mm[1]:+.1f})mm"
                if c.pos_mm else f"#{c.cup_id}=perdu"
                for c in cups
            ]
            print(f"\r[Positions] {'  '.join(parts)}     ", end="", flush=True)

        # FPS
        if not paused:
            t_now        = time.monotonic()
            measured_fps = 0.9 * measured_fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev       = t_now

        # Rendu
        out = draw_overlay(
            frame, cups,
            det_bboxes=last_det if debug_mode else [],
            mask=last_mask,
            debug=debug_mode,
            fps=measured_fps,
            v_min=detector.v_min,
        )
        if paused:
            cv2.putText(out, "PAUSE", (fw // 2 - 70, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 80, 255), 3)

        cv2.imshow("Cup Tracker -- HSV + CSRT", out)

        # Touches
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            print()
            det  = detector.detect(frame)
            manager.force_reset(frame, det)
            last_det   = det
            last_det_t = time.monotonic()
            print(f"[Pipeline] Redetection : {len(det)} tasse(s)")
        elif key == ord('d'):
            debug_mode = not debug_mode
            last_mask  = detector.mask(frame) if debug_mode else None
            print(f"\n[Pipeline] Debug HSV : {'ON' if debug_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            detector.v_min = detector.v_min - 5   # baisser = detecte plus sombre
            print(f"\n[Pipeline] V_min={detector.v_min}")
        elif key == ord('-'):
            detector.v_min = detector.v_min + 5   # monter = plus strict
            print(f"\n[Pipeline] V_min={detector.v_min}")
        elif key == ord('p'):
            paused = not paused
            print(f"\n[Pipeline] {'PAUSE' if paused else 'REPRISE'}")

    print()
    cap.release()
    cv2.destroyAllWindows()
    print("[Pipeline] Termine.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection HSV top-view + Tracking CSRT de tasses"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera",    type=int, default=0,
                     help="Index camera OpenCV (defaut: 0)")
    src.add_argument("--video",     type=str, default="",
                     help="Fichier video (mode demo)")
    p.add_argument("--pose",        type=str, default="camtop_table_pose.json",
                   help="Chemin vers camtop_table_pose.json")
    p.add_argument("--undistort",   action="store_true",
                   help="Appliquer la correction de distorsion")
    p.add_argument("--dist",        type=str, default="",
                   help="JSON camera_matrix + dist_coeffs (si --undistort)")
    p.add_argument("--v-min",       type=int, default=int(HSV_LOWER[2]),
                   help=f"Luminosite HSV minimale des tasses (defaut:{int(HSV_LOWER[2])})")
    p.add_argument("--area-min",    type=int, default=AREA_MIN,
                   help=f"Aire min contour px2 (defaut:{AREA_MIN})")
    p.add_argument("--area-max",    type=int, default=AREA_MAX,
                   help=f"Aire max contour px2 (defaut:{AREA_MAX})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Surcharge des parametres HSV depuis CLI
    HSV_LOWER[2] = args.v_min
    AREA_MIN     = args.area_min
    AREA_MAX     = args.area_max
    run(
        camera_index=args.camera,
        video_path=args.video,
        pose_path=args.pose,
        undistort=args.undistort,
        dist_path=args.dist,
    )