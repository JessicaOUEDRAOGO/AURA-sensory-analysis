# -*- coding: utf-8 -*-
"""
cup_top_tracker.py — Tracker CSRT pour la tasse vue depuis la cam_top
======================================================================

Responsabilités :
  1. Initialiser un tracker CSRT OpenCV sur une bbox fournie
  2. Mettre à jour le tracker frame par frame
  3. Convertir la position pixel (centre bbox) → coordonnées mm dans le
     repère table, via la pose camtop_table_pose.json (même méthode que
     pose_top_cam_pixel_to_mm.py)

Interface publique :
    tracker = CupTopTracker(pose_path)
    tracker.start(frame_top, bbox_pixel)   # bbox = (x, y, w, h) en pixels
    ok, pos_mm = tracker.update(frame_top) # pos_mm = (x_mm, y_mm) ou None
    tracker.stop()
    tracker.is_active                      # bool

Conversion bbox → mm :
    On prend le centre de la bbox (cx, cy) et on applique pixel_to_table_mm()
    identique à celui de pose_top_cam_pixel_to_mm.py.
    La cam_top est supposée avoir une image déjà undistordue (même pipeline
    que HandTrackingThread qui applique undistort avant d'envoyer à MediaPipe).
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class CupTopTracker:
    """
    Encapsule un tracker CSRT OpenCV pour une tasse unique.

    - start()  : initialise le tracker sur la frame et la bbox données
    - update() : met à jour et retourne (True, pos_mm) ou (False, None)
    - stop()   : libère le tracker
    """

    def __init__(self, pose_path: str):
        """
        pose_path : chemin vers camtop_table_pose.json
                    (contient rvec, tvec, camera_matrix)
        """
        self._tracker:   Optional[cv2.TrackerCSRT] = None
        self._active:    bool  = False
        self._cup_id:    Optional[int] = None

        # Chargement de la pose cam_top
        data     = json.load(open(pose_path, "r", encoding="utf-8"))
        self._rvec = np.array(data["rvec"],          dtype=np.float64)
        self._tvec = np.array(data["tvec"],          dtype=np.float64)
        self._K    = np.array(data["camera_matrix"], dtype=np.float64)
        # dist_coeffs = 0 car l'image est préalablement undistordue
        self._dist = np.zeros((5, 1), dtype=np.float64)

        # CSRT — paramètres par défaut suffisants pour un fond de table stable.
        # Si drift observé : augmenter filter_lr (0.02→0.06) via cv2.TrackerCSRT_Params()
        # Si accrochage fond : activer use_segmentation=True

    # ------------------------------------------------------------------
    # Propriétés publiques
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def cup_id(self) -> Optional[int]:
        return self._cup_id

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def start(self, frame: np.ndarray, bbox_pixel: Tuple[int, int, int, int],
              cup_id: int) -> bool:
        """
        Initialise le tracker CSRT.

        frame       : frame BGR de la cam_top (déjà undistordue)
        bbox_pixel  : (x, y, w, h) en pixels — centre de la tasse
        cup_id      : ID ArUco hérité de la cam_bottom

        Retourne True si l'init a réussi.
        """
        self.stop()  # sécurité : libère un tracker précédent éventuel

        x, y, w, h = bbox_pixel
        fh, fw = frame.shape[:2]

        # Debug systématique pour diagnostiquer les échecs
        print(f"[CupTopTracker] init — cup_id={cup_id} "
              f"bbox_raw=({x},{y},{w},{h}) frame=({fw}x{fh}) "
              f"dtype={frame.dtype} channels={frame.shape[2] if frame.ndim==3 else 1}")

        # Sanity check : bbox dans les limites de la frame
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(4, min(w, fw - x))
        h = max(4, min(h, fh - y))
        print(f"[CupTopTracker] bbox_clipped=({x},{y},{w},{h})")

        self._tracker = cv2.TrackerCSRT_create()   # paramètres par défaut

        # OpenCV 4.13+ : init() retourne None (pas True/False) — on vérifie
        # l'absence d'exception plutôt que la valeur de retour
        try:
            self._tracker.init(frame, (x, y, w, h))
            ok_check, _ = self._tracker.update(frame)
            init_ok = True

            # DEBUG visuel — a retirer une fois le tracking valide
            debug = frame.copy()
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(debug, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
            cv2.imwrite(f"debug_csrt_init_cup{cup_id}.jpg", debug)
            cv2.imwrite(f"debug_csrt_crop_cup{cup_id}.jpg", frame[y:y+h, x:x+w])
            print(f"[CupTopTracker] Debug sauvegarde : debug_csrt_init_cup{cup_id}.jpg")

        except Exception as e:
            print(f"[CupTopTracker] Exception init : {e}")
            init_ok = False

        if init_ok:
            self._active  = True
            self._cup_id  = cup_id
            print(f"[CupTopTracker] START OK — cup_id={cup_id} bbox=({x},{y},{w},{h})")
        else:
            self._tracker = None
            print(f"[CupTopTracker] ERREUR init CSRT — cup_id={cup_id} bbox=({x},{y},{w},{h})")

        return init_ok

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Met à jour le tracker sur la nouvelle frame.

        Retourne :
            (True,  (x_mm, y_mm))  si le tracker a réussi
            (False, None)           si perdu ou non actif
        """
        if not self._active or self._tracker is None:
            return False, None

        ok, bbox = self._tracker.update(frame)

        if not ok:
            print(f"[CupTopTracker] Tracker perdu — cup_id={self._cup_id}")
            return False, None

        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        pos_mm = self._pixel_to_table_mm(cx, cy)
        if pos_mm is None:
            # Intersection impossible (rayon parallèle au plan ou derrière caméra)
            return False, None

        return True, pos_mm

    def stop(self) -> None:
        """Libère le tracker proprement."""
        if self._active:
            print(f"[CupTopTracker] STOP — cup_id={self._cup_id}")
        self._tracker = None
        self._active  = False
        self._cup_id  = None

    # ------------------------------------------------------------------
    # Conversion pixel → mm (même logique que pose_top_cam_pixel_to_mm.py)
    # ------------------------------------------------------------------
    def _pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        K_inv = np.linalg.inv(self._K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def _intersect_ray_plane(self, ray: np.ndarray) -> Optional[np.ndarray]:
        R, _         = cv2.Rodrigues(self._rvec)
        normal       = R[:, 2]
        plane_origin = self._tvec.reshape(3)
        denom        = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        return t * ray if t >= 0 else None

    def _camera_to_table(self, pt_cam: np.ndarray) -> Tuple[float, float]:
        R, _ = cv2.Rodrigues(self._rvec)
        pt   = R.T @ (pt_cam - self._tvec.reshape(3))
        return float(pt[0]), float(pt[1])

    def _pixel_to_table_mm(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Convertit un pixel (u, v) en (x_mm, y_mm) dans le repère table."""
        ray    = self._pixel_to_ray(u, v)
        pt_cam = self._intersect_ray_plane(ray)
        if pt_cam is None:
            return None
        return self._camera_to_table(pt_cam)

    # ------------------------------------------------------------------
    # Utilitaire : projeter une position mm de la table en pixel cam_top
    # (utilisé pour construire la bbox initiale depuis la position ArUco)
    # ------------------------------------------------------------------
    def table_mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Projette un point (x_mm, y_mm, 0) du repère table vers les pixels
        de la cam_top. Utile pour construire la bbox d'initialisation CSRT
        depuis la dernière position ArUco connue.
        """
        pt_3d = np.array([[x_mm, y_mm, 0.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(pt_3d, self._rvec, self._tvec,
                                         self._K, self._dist)
        px, py = projected[0][0]
        return float(px), float(py)