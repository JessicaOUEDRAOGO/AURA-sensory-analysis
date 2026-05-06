# -*- coding: utf-8 -*-
"""
cup_tracking_pipeline.py
========================
Detection HSV (tasses noires sur table blanche retro-eclairee)
+ Tracking KCF leger pour interpolation inter-frames
+ Identite stable via association hongroise (distance centre + IoU)

Architecture :
  - HsvCupDetector  : seuillage luminosite -> contours -> bboxes
  - StableTracker   : KCF par tasse, reinitialise depuis HSV a chaque detection
  - Association     : algorithme hongrois simplifie pour identite stable
                      meme lors des croisements

Pourquoi KCF et non CSRT :
  CSRT : ~40ms/tasse -> 4 tasses -> ~160ms -> 6 FPS max
  KCF  :  ~5ms/tasse -> 4 tasses ->  ~20ms -> 30+ FPS

Usage :
    python cup_tracking_pipeline.py
    python cup_tracking_pipeline.py --camera 2
    python cup_tracking_pipeline.py --tracker kcf|csrt|mil
    python cup_tracking_pipeline.py --v-thresh 110

Touches :
    q / Echap -> quitter
    r         -> redetection HSV forcee
    d         -> debug masque HSV
    + / -     -> ajuster V_seuil (detecte plus/moins sombre)
    p         -> pause
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Resolution des chemins ────────────────────────────────────────────────────
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


try:
    from src.core.Hand_tracking.cup_top_tracker import CupTopTracker
except ImportError as exc:
    raise ImportError(
        "Impossible d'importer CupTopTracker. "
        "Placez cup_top_tracker.py dans le meme dossier."
    ) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════════════════════════════════

# Seuil de luminosite : pixels plus sombres que V_THRESHOLD = tasse
V_THRESHOLD_INIT = 110

# Parametres geometriques des contours
# Aires calibrees pour PROCESS_WIDTH=640 (frame full HD reduite au 1/3)
# Tasse vue de cote a 640px de large : ~40x60px = ~2400px2
AREA_MIN         = 800
AREA_MAX         = 25_000
CIRCULARITY_MIN  = 0.20
ASPECT_MIN       = 0.25
ASPECT_MAX       = 3.5
BBOX_MARGIN      = 20

# Tracker OpenCV a utiliser : "kcf" (rapide), "csrt" (precis), "mil"
TRACKER_TYPE     = "kcf"

# Resolution interne de traitement (detection + tracking)
# La frame est redimensionnee a cette largeur avant tout traitement.
# L affichage reste en resolution native.
# Plus c est petit -> plus c est rapide, mais moins de detail.
# 640 : optimal pour KCF sur CPU  |  960 : bon compromis  |  1280 : precis
PROCESS_WIDTH    = 640

# Intervalle entre deux passes de detection HSV completes (secondes)
DETECT_INTERVAL_S = 0.15   # ~7 detections/sec — assez frequent pour reinit rapide

# Distance max acceptable entre deux frames pour le meme tracker
# (en fraction de la diagonale de la bbox)
MAX_DRIFT_RATIO  = 0.8

# Score d'association minimum pour apparier une detection a un tracker
MATCH_MIN_SCORE  = 0.01

# Frames perdues consecutives avant suppression
MAX_LOST_FRAMES  = 20

# Palette de couleurs par cup_id (BGR)
_PALETTE = [
    (0,   210,  80),
    (0,   160, 255),
    (60,   60, 220),
    (200,   0, 200),
    (0,   220, 220),
    (255, 140,   0),
    (0,   200, 200),
    (180,   0, 180),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Detecteur HSV — tasses sombres sur fond blanc
# ══════════════════════════════════════════════════════════════════════════════

class HsvCupDetector:
    """
    Detecte les tasses noires/sombres sur table blanche retro-eclairee.
    Utilise un seuillage en niveaux de gris plutot que HSV car les tasses
    peuvent etre de differentes couleurs (noir, bleu marine, vert fonce...).
    """

    def __init__(self,
                 v_threshold:     int   = V_THRESHOLD_INIT,
                 area_min:        int   = AREA_MIN,
                 area_max:        int   = AREA_MAX,
                 circularity_min: float = CIRCULARITY_MIN,
                 margin:          int   = BBOX_MARGIN):
        self.v_threshold     = v_threshold
        self.area_min        = area_min
        self.area_max        = area_max
        self.circularity_min = circularity_min
        self.margin          = margin
        # Noyaux calibres pour PROCESS_WIDTH=640
        # A 640px : tasse ~80-150px de large -> fermeture de 9px suffit
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @property
    def v_min(self) -> int:
        return self.v_threshold

    @v_min.setter
    def v_min(self, v: int) -> None:
        self.v_threshold = int(np.clip(v, 10, 245))

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mask = self._mask(frame)
        return self._bboxes(frame, mask)

    def mask(self, frame: np.ndarray) -> np.ndarray:
        return self._mask(frame)

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
        # iterations=1 au lieu de 3 : evite la fusion de tasses proches
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
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
            if 4.0 * np.pi * area / (perim ** 2) < self.circularity_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (ASPECT_MIN <= w / float(h) <= ASPECT_MAX):
                continue
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(fw, x + w + self.margin)
            y2 = min(fh, y + h + self.margin)
            out.append((x1, y1, x2 - x1, y2 - y1))
        return out


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


def _dist_centers(b1: Tuple, b2: Tuple) -> float:
    cx1, cy1 = _center(b1)
    cx2, cy2 = _center(b2)
    return float(np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2))


def _match_score(tracker_bbox: Tuple, det_bbox: Tuple) -> float:
    """
    Score d appariement tracker <-> detection.
    Combine IoU et proximite des centres.
    Retourne une valeur >= 0 ; plus c est grand, meilleur est le match.
    """
    iou = _iou(tracker_bbox, det_bbox)
    x, y, w, h = tracker_bbox
    diag = max(np.sqrt(w**2 + h**2), 1.0)
    dist_norm = _dist_centers(tracker_bbox, det_bbox) / diag
    # Score positif uniquement si assez proche
    if dist_norm > 2.0:
        return 0.0
    return iou + 0.4 * max(0.0, 1.0 - dist_norm)


def _make_tracker(tracker_type: str) -> cv2.Tracker:
    t = tracker_type.lower()
    if t == "csrt":
        return cv2.TrackerCSRT_create()
    elif t == "mil":
        return cv2.TrackerMIL_create()
    else:  # kcf par defaut
        return cv2.TrackerKCF_create()


# ══════════════════════════════════════════════════════════════════════════════
#  Structure TrackedCup
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    cup_id:       int
    pose_tracker: CupTopTracker          # pour la conversion px -> mm
    cv_tracker:   cv2.Tracker            # tracker OpenCV leger (KCF)
    bbox:         Tuple[int,int,int,int]
    pos_mm:       Optional[Tuple[float,float]] = None
    lost_frames:  int  = 0
    active:       bool = True
    color:        Tuple[int,int,int] = field(default_factory=lambda: (0,200,80))

    def update_cv(self, frame: np.ndarray) -> bool:
        """Met a jour le tracker KCF. Retourne True si succes."""
        if not self.active:
            return False
        ok, raw = self.cv_tracker.update(frame)
        if ok:
            rx, ry, rw, rh = raw
            new_bbox = (int(rx), int(ry), max(4,int(rw)), max(4,int(rh)))
            # Contrainte de derive : rejette si le centre a trop bouge
            if _drift_ok(self.bbox, new_bbox):
                self.bbox = new_bbox
                return True
            else:
                self.active = False
                return False
        self.active = False
        return False

    def reinit_cv(self, frame: np.ndarray,
                  bbox: Tuple[int,int,int,int],
                  tracker_type: str = TRACKER_TYPE) -> bool:
        """Reinitialise le tracker KCF sur une nouvelle bbox."""
        self.cv_tracker = _make_tracker(tracker_type)
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw-1));  y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x));  h = max(4, min(h, fh-y))
        try:
            self.cv_tracker.init(frame, (x, y, w, h))
            self.bbox   = (x, y, w, h)
            self.active = True
            return True
        except Exception as e:
            print(f"[TrackedCup] Echec reinit KCF cup_id={self.cup_id}: {e}")
            self.active = False
            return False

    def update_pos_mm(self, frame: np.ndarray,
                      scale: float = 1.0) -> None:
        """
        Calcule la position mm depuis la bbox courante.
        scale : facteur pour remonter les coords proc -> coords full HD.
                (ex: 1920/640 = 3.0 si le tracker tourne sur frame 640px)
                La camera_matrix est calibree en resolution native -> on doit
                passer des coordonnees full HD a _pixel_to_table_mm.
        """
        cx, cy = _center(self.bbox)
        # Remonter en coordonnees full HD avant la conversion geometrique
        cx_full = cx / scale if scale != 1.0 else cx
        cy_full = cy / scale if scale != 1.0 else cy
        result = self.pose_tracker._pixel_to_table_mm(cx_full, cy_full)
        if result is not None:
            self.pos_mm = result


def _drift_ok(prev: Tuple, new: Tuple) -> bool:
    x, y, w, h = prev
    diag = max(np.sqrt(w**2 + h**2), 1.0)
    return _dist_centers(prev, new) < MAX_DRIFT_RATIO * diag


# ══════════════════════════════════════════════════════════════════════════════
#  Gestionnaire de trackers — identite stable
# ══════════════════════════════════════════════════════════════════════════════

class TrackingManager:
    """
    Gestion des trackers avec identite stable lors des croisements.

    Strategie :
      Chaque frame :
        1. KCF update (rapide, ~5ms/tasse) sur frame proc (640px)
        2. Calcul position mm avec remontee en coords full HD (x 1/scale)
      Toutes les DETECT_INTERVAL_S secondes (ou sur commande) :
        3. Detection HSV sur frame proc -> bboxes proc
        4. Association hongroise (score IoU + distance) -> identite stable
        5. Reinitialisation KCF sur bboxes HSV (corrige la derive)
        6. Creation de nouveaux trackers pour les nouvelles detections
        7. Suppression des trackers perdus trop longtemps

    scale : facteur proc->full (ex: 1920/640=3.0).
            Stocke dans le manager et passe a update_pos_mm pour que
            _pixel_to_table_mm recoive toujours des coordonnees full HD.
    """

    def __init__(self, pose_path: str, tracker_type: str = TRACKER_TYPE,
                 scale: float = 1.0):
        self._pose_path    = pose_path
        self._tracker_type = tracker_type
        self._scale        = scale   # proc -> full HD
        self._cups:  Dict[int, TrackedCup] = {}
        self._next_id: int = 0

    # ── API ───────────────────────────────────────────────────────────────────

    def update_tracking(self, frame: np.ndarray) -> List[TrackedCup]:
        """
        Appel rapide : met a jour KCF + calcule positions mm.
        A appeler chaque frame.
        """
        for cup in list(self._cups.values()):
            ok = cup.update_cv(frame)
            if ok:
                cup.update_pos_mm(frame, scale=self._scale)
                cup.lost_frames = 0
            else:
                cup.lost_frames += 1

        # Purge rapide des tres perdus
        for cid in [c for c, v in self._cups.items()
                    if v.lost_frames >= MAX_LOST_FRAMES]:
            self._cups[cid].pose_tracker.stop()
            del self._cups[cid]
            print(f"[Manager] Supprime cup_id={cid}")

        return list(self._cups.values())

    def update_detection(self, frame: np.ndarray,
                         detected: List[Tuple[int,int,int,int]]) -> None:
        """
        Appel periodique : realigne les trackers KCF sur les detections HSV.
        Garantit l identite stable et corrige la derive du KCF.
        """
        if not detected:
            return

        cups   = list(self._cups.values())
        used_d = set()   # indices de detected deja assignes
        used_t = set()   # cup_ids deja assignes

        # ── Association hongroise simplifiee (greedy sur score decroissant) ──
        # Construire la matrice de score
        if cups:
            scores = np.zeros((len(cups), len(detected)), dtype=np.float32)
            for i, cup in enumerate(cups):
                for j, det in enumerate(detected):
                    scores[i, j] = _match_score(cup.bbox, det)

            # Greedy : prendre le meilleur score disponible a chaque etape
            flat = np.argsort(-scores.ravel())
            for idx in flat:
                i, j = divmod(int(idx), len(detected))
                if i in used_t or j in used_d:
                    continue
                if scores[i, j] < MATCH_MIN_SCORE:
                    break
                cup = cups[i]
                det = detected[j]
                # Reinitialiser KCF sur la bbox HSV (corrige la derive)
                cup.reinit_cv(frame, det, self._tracker_type)
                cup.update_pos_mm(frame, scale=self._scale)
                cup.lost_frames = 0
                used_t.add(i)
                used_d.add(j)

        # ── Nouvelles detections sans tracker associe ─────────────────────────
        for j, det in enumerate(detected):
            if j not in used_d:
                self._create(frame, det)

    def force_reset(self, frame: np.ndarray,
                    detected: List[Tuple[int,int,int,int]]) -> None:
        for cup in self._cups.values():
            cup.pose_tracker.stop()
        self._cups.clear()
        for det in detected:
            self._create(frame, det)
        print(f"[Manager] Reset -> {len(detected)} tasse(s)")

    # ── Interne ───────────────────────────────────────────────────────────────

    def _create(self, frame: np.ndarray,
                bbox: Tuple[int,int,int,int]) -> None:
        cid = self._next_id
        self._next_id += 1

        # CupTopTracker est utilise UNIQUEMENT pour _pixel_to_table_mm()
        # On charge la pose mais on n appelle pas start() car on n utilise
        # pas son tracker CSRT interne — le tracking est fait par KCF.
        pose_tracker = CupTopTracker(self._pose_path)

        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw-1));  y = max(0, min(y, fh-1))
        w = max(4, min(w, fw-x));  h = max(4, min(h, fh-y))

        cv_trk = _make_tracker(self._tracker_type)
        try:
            cv_trk.init(frame, (x, y, w, h))
        except Exception as e:
            print(f"[Manager] Echec init KCF cup_id={cid}: {e}")
            return

        cup = TrackedCup(
            cup_id=cid,
            pose_tracker=pose_tracker,
            cv_tracker=cv_trk,
            bbox=(x, y, w, h),
            color=_PALETTE[cid % len(_PALETTE)],
        )
        cup.update_pos_mm(frame, scale=self._scale)
        self._cups[cid] = cup
        print(f"[Manager] Nouveau cup_id={cid} bbox=({x},{y},{w},{h})")


# ══════════════════════════════════════════════════════════════════════════════
#  Rendu visuel
# ══════════════════════════════════════════════════════════════════════════════

def _dashed_rect(img, pt1, pt2, color, thickness=1, gap=8):
    x1, y1 = pt1; x2, y2 = pt2
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x,y1), (min(x+gap,x2),y1), color, thickness)
        cv2.line(img, (x,y2), (min(x+gap,x2),y2), color, thickness)
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x1,y), (x1,min(y+gap,y2)), color, thickness)
        cv2.line(img, (x2,y), (x2,min(y+gap,y2)), color, thickness)


def draw_overlay(frame:      np.ndarray,
                 cups:       List[TrackedCup],
                 det_bboxes: List[Tuple],
                 mask:       Optional[np.ndarray],
                 debug:      bool,
                 fps:        float,
                 v_seuil:    int) -> np.ndarray:
    out = frame.copy()

    if debug and mask is not None:
        green = np.zeros_like(out)
        green[:,:,1] = mask
        out = cv2.addWeighted(out, 0.65, green, 0.35, 0)
        for (x,y,w,h) in det_bboxes:
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,255), 1)

    for cup in cups:
        x, y, w, h = cup.bbox
        col  = cup.color
        lost = cup.lost_frames > 0

        if lost:
            _dashed_rect(out, (x,y), (x+w,y+h), col, 2)
        else:
            cv2.rectangle(out, (x,y), (x+w,y+h), col, 2)

        cx, cy = x + w//2, y + h//2
        cv2.drawMarker(out, (cx,cy), col,
                       cv2.MARKER_CROSS, markerSize=18, thickness=2)

        if cup.pos_mm is not None:
            xmm, ymm = cup.pos_mm
            label = f"#{cup.cup_id}  {xmm:+.1f} , {ymm:+.1f} mm"
        else:
            label = f"#{cup.cup_id}  calcul..."

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        ty = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ty-th-4), (x+tw+8, ty+2), (15,15,15), -1)
        cv2.putText(out, label, (x+4, ty-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, col, 1, cv2.LINE_AA)

    lines = [f"FPS: {fps:.1f}",
             f"Tasses: {len(cups)}",
             f"V_seuil: {v_seuil}  (+/-)",
             "[DEBUG HSV ON]" if debug else ""]
    for i, txt in enumerate(t for t in lines if t):
        cv2.putText(out, txt, (10, 26+28*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (210,210,210), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def run(camera_index:  int   = 0,
        video_path:    str   = "",
        pose_path:     str   = "camtop_table_pose.json",
        tracker_type:  str   = TRACKER_TYPE,
        undistort:     bool  = False,
        dist_path:     str   = "") -> None:

    pose_path = Path(pose_path)

    if not pose_path.is_absolute():
        pose_path = Path(__file__).resolve().parents[3] / "config" / pose_path

    if not pose_path.exists():
        raise IOError(f"Fichier de pose introuvable : {pose_path}")
        

    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"[Pipeline] Video : {video_path}")
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        print(f"[Pipeline] Camera index={camera_index}")

    if not cap.isOpened():
        raise IOError("Impossible d'ouvrir la source video.")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Pipeline] Resolution : {fw}x{fh}  "
          f"Tracker : {tracker_type.upper()}")

    map1 = map2 = None
    if undistort and dist_path and Path(dist_path).exists():
        import json as _j
        d    = _j.load(open(dist_path))
        K    = np.array(d["camera_matrix"], dtype=np.float64)
        dist = np.array(d["dist_coeffs"],   dtype=np.float64)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (fw,fh), cv2.CV_16SC2)
        print("[Pipeline] Undistort active.")

    detector = HsvCupDetector()

    # ── Calcul du facteur de redimensionnement pour le traitement ─────────────
    # Tout le traitement (detection HSV + KCF) se fait sur une frame reduite.
    # Les bboxes sont ensuite remises a l echelle pour l affichage.
    scale     = PROCESS_WIDTH / fw          # ex: 640/1920 = 0.333
    proc_w    = PROCESS_WIDTH
    proc_h    = int(fh * scale)
    print(f"[Pipeline] Frame traitement : {proc_w}x{proc_h}  "
          f"(scale={scale:.3f})  Tracker : {tracker_type.upper()}")

    manager  = TrackingManager(pose_path, tracker_type, scale=1.0/scale)

    def to_proc(frame_full: np.ndarray) -> np.ndarray:
        return cv2.resize(frame_full, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

    def scale_bboxes_up(bboxes):
        """Remonte les bboxes de l espace proc vers l espace full."""
        inv = 1.0 / scale
        return [(int(x*inv), int(y*inv), int(w*inv), int(h*inv))
                for x, y, w, h in bboxes]

    def scale_bboxes_down(bboxes):
        """Reduit les bboxes de l espace full vers l espace proc."""
        return [(int(x*scale), int(y*scale), int(w*scale), int(h*scale))
                for x, y, w, h in bboxes]

    # Init sur la premiere frame
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Impossible de lire la premiere frame.")
    if map1 is not None:
        frame0 = cv2.remap(frame0, map1, map2, cv2.INTER_LINEAR)

    proc0    = to_proc(frame0)
    init_det_proc = detector.detect(proc0)
    init_det      = scale_bboxes_up(init_det_proc)
    print(f"[Pipeline] Detection initiale : {len(init_det)} tasse(s)  "
          f"(V_seuil={detector.v_min})")
    manager.force_reset(proc0, init_det_proc)

    debug_mode    = False
    paused        = False
    last_det_t    = time.monotonic()
    t_prev        = time.monotonic()
    measured_fps  = 0.0
    last_det_disp: List[Tuple] = init_det          # bboxes en coords full (affichage)
    last_mask_disp: Optional[np.ndarray] = None    # masque redimensionne pour affichage
    proc_frame    = proc0

    print("[Pipeline] q=quitter  r=redetection  d=debug  +/-=seuil  p=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("[Pipeline] Fin de la video.")
                break
            if map1 is not None:
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            proc_frame = to_proc(frame)

        # ── Tracking KCF rapide sur frame reduite — chaque frame ──────────────
        cups_proc = manager.update_tracking(proc_frame)

        # ── Detection HSV periodique sur frame reduite ────────────────────────
        now = time.monotonic()
        if not paused and (now - last_det_t) >= DETECT_INTERVAL_S:
            det_proc       = detector.detect(proc_frame)
            last_det_t     = now
            manager.update_detection(proc_frame, det_proc)
            cups_proc      = list(manager._cups.values())
            last_det_disp  = scale_bboxes_up(det_proc)
            if debug_mode:
                mask_proc       = detector.mask(proc_frame)
                last_mask_disp  = cv2.resize(mask_proc, (fw, fh),
                                             interpolation=cv2.INTER_NEAREST)

        if debug_mode and last_mask_disp is None:
            mask_proc      = detector.mask(proc_frame)
            last_mask_disp = cv2.resize(mask_proc, (fw, fh),
                                        interpolation=cv2.INTER_NEAREST)
        if not debug_mode:
            last_mask_disp = None

        # Remonter les bboxes proc -> full pour l affichage
        cups_disp = []
        for c in cups_proc:
            bx, by, bw, bh = c.bbox
            inv = 1.0 / scale
            c_disp = TrackedCup(
                cup_id=c.cup_id,
                pose_tracker=c.pose_tracker,
                cv_tracker=c.cv_tracker,
                bbox=(int(bx*inv), int(by*inv), int(bw*inv), int(bh*inv)),
                pos_mm=c.pos_mm,
                lost_frames=c.lost_frames,
                active=c.active,
                color=c.color,
            )
            cups_disp.append(c_disp)

        # Console
        if cups_proc and not paused:
            parts = [
                f"#{c.cup_id}=({c.pos_mm[0]:+.1f},{c.pos_mm[1]:+.1f})mm"
                if c.pos_mm else f"#{c.cup_id}=?"
                for c in cups_proc
            ]
            print(f"\r[Pos] {'  '.join(parts)}     ", end="", flush=True)

        # FPS
        if not paused:
            t_now        = time.monotonic()
            measured_fps = 0.9*measured_fps + 0.1/max(t_now-t_prev, 1e-6)
            t_prev       = t_now

        out = draw_overlay(frame, cups_disp,
                           det_bboxes=last_det_disp if debug_mode else [],
                           mask=last_mask_disp,
                           debug=debug_mode,
                           fps=measured_fps,
                           v_seuil=detector.v_min)
        if paused:
            cv2.putText(out, "PAUSE", (fw//2-70, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,80,255), 3)

        cv2.imshow("Cup Tracker -- HSV + KCF", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            print()
            det_proc      = detector.detect(proc_frame)
            manager.force_reset(proc_frame, det_proc)
            last_det_disp = scale_bboxes_up(det_proc)
            last_det_t    = time.monotonic()
            print(f"[Pipeline] Redetection : {len(det_proc)} tasse(s)")
        elif key == ord('d'):
            debug_mode     = not debug_mode
            last_mask_disp = None
            print(f"\n[Pipeline] Debug : {'ON' if debug_mode else 'OFF'}")
        elif key in (ord('+'), ord('=')):
            detector.v_min = detector.v_min + 5
            print(f"\n[Pipeline] V_seuil={detector.v_min}")
        elif key == ord('-'):
            detector.v_min = detector.v_min - 5
            print(f"\n[Pipeline] V_seuil={detector.v_min}")
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
        description="Detection HSV + Tracking KCF de tasses (table retro-eclairee)"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera",   type=int, default=0)
    src.add_argument("--video",    type=str, default="")
    p.add_argument("--pose",       type=str, default="camtop_table_pose.json")
    p.add_argument("--tracker",    type=str, default=TRACKER_TYPE,
                   choices=["kcf","csrt","mil"],
                   help="Algorithme de tracking (defaut: kcf)")
    p.add_argument("--v-thresh",   type=int, default=V_THRESHOLD_INIT,
                   help=f"Seuil luminosite tasse (defaut:{V_THRESHOLD_INIT})")
    p.add_argument("--area-min",   type=int, default=AREA_MIN)
    p.add_argument("--area-max",   type=int, default=AREA_MAX)
    p.add_argument("--undistort",  action="store_true")
    p.add_argument("--dist",       type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    V_THRESHOLD_INIT = args.v_thresh
    AREA_MIN         = args.area_min
    AREA_MAX         = args.area_max
    run(
        camera_index=args.camera,
        video_path=args.video,
        pose_path=args.pose,
        tracker_type=args.tracker,
        undistort=args.undistort,
        dist_path=args.dist,
    )