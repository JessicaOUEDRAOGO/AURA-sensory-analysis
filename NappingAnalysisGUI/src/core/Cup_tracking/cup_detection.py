# -*- coding: utf-8 -*-
"""
cup_tracking_pipeline.py
========================
Pipeline complet :
  1. Ouverture de la cam_top (ou fichier vidéo en mode démo)
  2. Détection des tasses via YOLOv8 (classe "cup", COCO id=41)
  3. Instanciation d'un tracker CSRT par tasse détectée
  4. Conversion de la position pixel → mm via CupTopTracker
  5. Affichage temps réel + console

Dépendances :
    pip install ultralytics opencv-contrib-python numpy

Usage :
    python cup_tracking_pipeline.py                           # webcam 0
    python cup_tracking_pipeline.py --camera 2               # autre index
    python cup_tracking_pipeline.py --video myvideo.mp4      # fichier vidéo
    python cup_tracking_pipeline.py --pose camtop_table_pose.json
    python cup_tracking_pipeline.py --model yolov8n.pt --conf 0.4

Touches :
    q / Échap  → quitter
    r          → forcer une redétection YOLO immédiate
    d          → activer / désactiver l'overlay de debug YOLO
    p          → pause / reprise
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Import du tracker de position mm ─────────────────────────────────────────
try:
    from src.core.Hand_tracking.cup_top_tracker import CupTopTracker
except ImportError as exc:
    raise ImportError(
        "Impossible d'importer CupTopTracker.\n"
        "Assurez-vous que cup_top_tracker.py est dans le même dossier "
        "ou dans le PYTHONPATH."
    ) from exc

# ── Import YOLO ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics non installé.\n"
        "Installez-le avec : pip install ultralytics"
    ) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  Constantes globales
# ══════════════════════════════════════════════════════════════════════════════

# Classe COCO correspondant à "cup" (id=41, 0-indexé)
YOLO_CUP_CLASS_ID   = 41

# Confiance YOLO minimale pour qu'une détection soit retenue
YOLO_CONF_THRESHOLD = 0.40

# Marge (pixels) ajoutée autour de la bbox YOLO avant de passer au tracker
CSRT_BBOX_MARGIN    = 12

# IOU minimale pour associer une détection YOLO à un tracker existant
IOU_MATCH_THRESHOLD = 0.25

# Nombre de frames perdues consécutives avant de supprimer un tracker
MAX_LOST_FRAMES     = 45

# Intervalle (secondes) entre deux passes YOLO complètes
YOLO_INTERVAL_S     = 2.0

# Couleurs par cup_id (BGR)
_PALETTE = [
    (0,   210,  80),   # vert vif
    (0,   160, 255),   # orange
    (220,  60,  60),   # bleu acier
    (200,   0, 200),   # violet
    (0,   220, 220),   # cyan
    (255, 180,   0),   # bleu clair
]


# ══════════════════════════════════════════════════════════════════════════════
#  Détecteur YOLO
# ══════════════════════════════════════════════════════════════════════════════

class YoloCupDetector:
    """
    Encapsule YOLOv8 pour détecter uniquement les tasses (classe 41 COCO).

    Avec un modèle fin-tuné sur vos propres tasses :
        detector = YoloCupDetector("mon_modele.pt", cup_class_id=0)

    Avec le modèle générique pré-entraîné :
        detector = YoloCupDetector("yolov8n.pt", cup_class_id=41)
    """

    def __init__(self,
                 model_path:    str   = "yolov8n.pt",
                 cup_class_id:  int   = YOLO_CUP_CLASS_ID,
                 conf:          float = YOLO_CONF_THRESHOLD,
                 margin:        int   = CSRT_BBOX_MARGIN,
                 device:        str   = ""):
        """
        model_path   : chemin vers un .pt YOLOv8 (téléchargé auto si absent)
        cup_class_id : id de la classe à détecter (41=cup COCO, 0 si fin-tuné)
        conf         : seuil de confiance (0-1)
        margin       : marge en pixels ajoutée à chaque bbox
        device       : "" -> auto, "cpu", "cuda:0", "mps"
        """
        print(f"[YoloCupDetector] Chargement du modèle : {model_path}")
        self._model        = YOLO(model_path)
        self._cup_class_id = cup_class_id
        self._conf         = conf
        self._margin       = margin
        self._device       = device or None
        print(f"[YoloCupDetector] Pret — classe={cup_class_id}  conf>={conf}")

    # ── API publique ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Lance YOLO sur `frame` et retourne les bboxes des tasses detectées
        sous la forme [(x, y, w, h), ...] en pixels.
        """
        results = self._model.predict(
            frame,
            classes=[self._cup_class_id],
            conf=self._conf,
            verbose=False,
            device=self._device,
        )

        bboxes: List[Tuple[int, int, int, int]] = []
        fh, fw = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])

            # Ajout de la marge + clamp dans les limites de la frame
            x1 = max(0, x1 - self._margin)
            y1 = max(0, y1 - self._margin)
            x2 = min(fw, x2 + self._margin)
            y2 = min(fh, y2 + self._margin)

            w = max(4, x2 - x1)
            h = max(4, y2 - y1)

            bboxes.append((x1, y1, w, h))
            print(f"[YOLO] Tasse detectee  conf={conf_val:.2f}  "
                  f"bbox=({x1},{y1},{w},{h})")

        return bboxes


# ══════════════════════════════════════════════════════════════════════════════
#  Structure de données
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
    Gere le cycle de vie de plusieurs trackers CSRT simultanes.

    Par frame :
      1. update CSRT  — chaque tracker predit sa nouvelle position
      2. associate    — bboxes YOLO appariees aux trackers via IoU
      3. new          — détections non appariées créent un nouveau tracker
      4. purge        — trackers perdus depuis MAX_LOST_FRAMES supprimes
    """

    def __init__(self, pose_path: str):
        self._pose_path = pose_path
        self._cups:    Dict[int, TrackedCup] = {}
        self._next_id: int = 0

    # ── API ───────────────────────────────────────────────────────────────────

    def update(self,
               frame:       np.ndarray,
               yolo_bboxes: List[Tuple[int, int, int, int]]
               ) -> List[TrackedCup]:
        """
        Met a jour tous les trackers CSRT, associe les detections YOLO,
        crée les nouveaux trackers, purge les perdus.
        Retourne la liste des TrackedCup actifs.
        """
        # ── 1. Mise a jour CSRT ───────────────────────────────────────────────
        for cup in list(self._cups.values()):
            ok, pos_mm = cup.tracker.update(frame)
            if ok:
                cup.pos_mm      = pos_mm
                cup.lost_frames = 0
                # Recuperer la bbox a jour depuis le tracker interne OpenCV
                _, raw_bbox = cup.tracker._tracker.update(frame)
                rx, ry, rw, rh = raw_bbox
                cup.bbox = (int(rx), int(ry), int(rw), int(rh))
            else:
                cup.lost_frames += 1

        # ── 2. Appariement YOLO → trackers (greedy IoU) ───────────────────────
        unmatched = list(yolo_bboxes)
        for cup in self._cups.values():
            if not unmatched:
                break
            ious   = [_iou(cup.bbox, b) for b in unmatched]
            best_i = int(np.argmax(ious))
            if ious[best_i] >= IOU_MATCH_THRESHOLD:
                new_bbox = unmatched.pop(best_i)
                cup.bbox        = new_bbox
                cup.lost_frames = 0
                # Reinitialiser le tracker si tracking perdu
                if not cup.tracker.is_active:
                    cup.tracker.start(frame, new_bbox, cup.cup_id)
                    print(f"[TrackingManager] Reinit tracker cup_id={cup.cup_id}")

        # ── 3. Nouvelles detections → nouveaux trackers ────────────────────────
        for bbox in unmatched:
            self._create(frame, bbox)

        # ── 4. Purge ──────────────────────────────────────────────────────────
        to_delete = [cid for cid, c in self._cups.items()
                     if c.lost_frames >= MAX_LOST_FRAMES]
        for cid in to_delete:
            self._cups[cid].tracker.stop()
            del self._cups[cid]
            print(f"[TrackingManager] Supprime cup_id={cid} (perdu >={MAX_LOST_FRAMES}f)")

        return list(self._cups.values())

    def force_redetect(self,
                       frame:       np.ndarray,
                       yolo_bboxes: List[Tuple[int, int, int, int]]) -> None:
        for cup in self._cups.values():
            cup.tracker.stop()
        self._cups.clear()
        for bbox in yolo_bboxes:
            self._create(frame, bbox)
        print(f"[TrackingManager] Reset — {len(yolo_bboxes)} tasse(s)")

    # ── Interne ───────────────────────────────────────────────────────────────

    def _create(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        cup_id  = self._next_id
        self._next_id += 1
        tracker = CupTopTracker(self._pose_path)
        ok = tracker.start(frame, bbox, cup_id)
        if ok:
            color = _PALETTE[cup_id % len(_PALETTE)]
            self._cups[cup_id] = TrackedCup(
                cup_id=cup_id,
                tracker=tracker,
                bbox=bbox,
                color=color,
            )
            print(f"[TrackingManager] Nouveau tracker cup_id={cup_id} bbox={bbox}")
        else:
            print(f"[TrackingManager] Echec init CSRT bbox={bbox} — ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  Rendu visuel
# ══════════════════════════════════════════════════════════════════════════════

def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


def draw_overlay(frame:       np.ndarray,
                 cups:        List[TrackedCup],
                 yolo_bboxes: List[Tuple[int, int, int, int]],
                 debug:       bool  = False,
                 fps:         float = 0.0) -> np.ndarray:
    out = frame.copy()

    # Bboxes YOLO brutes (debug)
    if debug:
        for (x, y, w, h) in yolo_bboxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(out, "YOLO", (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # Trackers CSRT
    for cup in cups:
        x, y, w, h = cup.bbox
        col  = cup.color
        lost = cup.lost_frames > 0

        if lost:
            _draw_dashed_rect(out, (x, y), (x + w, y + h), col, 1)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)

        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(out, (cx, cy), col,
                       cv2.MARKER_CROSS, markerSize=16, thickness=2)

        if cup.pos_mm is not None:
            xmm, ymm = cup.pos_mm
            label = f"#{cup.cup_id}  {xmm:+.1f} , {ymm:+.1f} mm"
        else:
            label = f"#{cup.cup_id}  perdu ({cup.lost_frames}f)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        ty = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 8, ty + 2),
                      (15, 15, 15), -1)
        cv2.putText(out, label, (x + 4, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, col, 1, cv2.LINE_AA)

    # HUD
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(out, f"Tasses: {len(cups)}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 1, cv2.LINE_AA)
    if debug:
        cv2.putText(out, "[DEBUG YOLO]", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def run(camera_index: int   = 0,
        video_path:   str   = "",
        pose_path:    str   = "camtop_table_pose.json",
        model_path:   str   = "yolov8n.pt",
        cup_class_id: int   = YOLO_CUP_CLASS_ID,
        conf:         float = YOLO_CONF_THRESHOLD,
        undistort:    bool  = False,
        dist_path:    str   = "") -> None:

    pose_path = Path(pose_path)

    if not pose_path.is_absolute():
        pose_path = Path(__file__).resolve().parents[3] / "config" / pose_path

    if not pose_path.exists():
        raise FileNotFoundError(
            f"Fichier de pose introuvable : {pose_path}\n"
            "Fournissez --pose <chemin_vers_camtop_table_pose.json>"
        )

    # Ouverture source
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

    # Undistort optionnel
    map1 = map2 = None
    if undistort and dist_path and Path(dist_path).exists():
        import json as _json
        d    = _json.load(open(dist_path))
        K    = np.array(d["camera_matrix"], dtype=np.float64)
        dist = np.array(d["dist_coeffs"],   dtype=np.float64)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (fw, fh), cv2.CV_16SC2
        )
        print("[Pipeline] Undistort active.")

    # Composants
    detector = YoloCupDetector(
        model_path=model_path,
        cup_class_id=cup_class_id,
        conf=conf,
    )
    manager = TrackingManager(pose_path)

    # Premiere frame + detection initiale
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Impossible de lire la premiere frame.")
    if map1 is not None:
        frame0 = cv2.remap(frame0, map1, map2, cv2.INTER_LINEAR)

    init_bboxes = detector.detect(frame0)
    print(f"[Pipeline] Init YOLO : {len(init_bboxes)} tasse(s)")
    manager.force_redetect(frame0, init_bboxes)

    # Etat boucle
    debug_mode           = False
    paused               = False
    last_yolo_t          = time.monotonic()
    t_prev               = time.monotonic()
    measured_fps         = 0.0
    last_yolo_bboxes: List[Tuple[int, int, int, int]] = init_bboxes

    print("[Pipeline] Demarrage — q=quitter  r=redetection  d=debug  p=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("[Pipeline] Fin de la video.")
                break
            if map1 is not None:
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Passe YOLO periodique
        now = time.monotonic()
        if not paused and (now - last_yolo_t) >= YOLO_INTERVAL_S:
            last_yolo_bboxes = detector.detect(frame)
            last_yolo_t      = now
            if last_yolo_bboxes:
                print(f"[YOLO] {len(last_yolo_bboxes)} detection(s)")

        # Mise a jour trackers CSRT
        cups = manager.update(frame, last_yolo_bboxes if not paused else [])

        # Console
        if cups and not paused:
            parts = [
                f"#{c.cup_id}=({c.pos_mm[0]:+.1f},{c.pos_mm[1]:+.1f})mm"
                if c.pos_mm else f"#{c.cup_id}=perdu"
                for c in cups
            ]
            print(f"\r[Positions] {'  '.join(parts)}          ",
                  end="", flush=True)

        # FPS
        if not paused:
            t_now        = time.monotonic()
            measured_fps = 0.9 * measured_fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev       = t_now

        # Rendu
        out = draw_overlay(
            frame, cups,
            yolo_bboxes=last_yolo_bboxes if debug_mode else [],
            debug=debug_mode,
            fps=measured_fps,
        )
        if paused:
            cv2.putText(out, "PAUSE", (fw // 2 - 60, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 80, 255), 3)

        cv2.imshow("Cup Tracker — YOLO + CSRT", out)

        # Touches
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            print()
            bboxes = detector.detect(frame)
            manager.force_redetect(frame, bboxes)
            last_yolo_bboxes = bboxes
            last_yolo_t      = time.monotonic()
            print(f"[Pipeline] Redetection forcee : {len(bboxes)} tasse(s)")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"\n[Pipeline] Debug YOLO : {'ON' if debug_mode else 'OFF'}")
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
        description="Detection YOLO + Tracking CSRT de tasses via cam_top"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera",   type=int,   default=0,
                     help="Index OpenCV de la camera (defaut: 0)")
    src.add_argument("--video",    type=str,   default="",
                     help="Fichier video (mode demo)")
    p.add_argument("--pose",       type=str,   default="camtop_table_pose.json",
                   help="Chemin vers camtop_table_pose.json")
    p.add_argument("--model",      type=str,   default="yolov8n.pt",
                   help="Modele YOLO .pt (telecharge auto si absent)")
    p.add_argument("--class-id",   type=int,   default=YOLO_CUP_CLASS_ID,
                   help=f"ID classe YOLO pour 'cup' "
                        f"(COCO defaut: {YOLO_CUP_CLASS_ID} ; "
                        f"0 si modele fin-tune mono-classe)")
    p.add_argument("--conf",       type=float, default=YOLO_CONF_THRESHOLD,
                   help=f"Seuil confiance YOLO (defaut: {YOLO_CONF_THRESHOLD})")
    p.add_argument("--undistort",  action="store_true",
                   help="Appliquer la correction de distorsion")
    p.add_argument("--dist",       type=str,   default="",
                   help="JSON camera_matrix + dist_coeffs (si --undistort)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        camera_index=args.camera,
        video_path=args.video,
        pose_path=args.pose,
        model_path=args.model,
        cup_class_id=args.class_id,
        conf=args.conf,
        undistort=args.undistort,
        dist_path=args.dist,
    )