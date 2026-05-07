# -*- coding: utf-8 -*-
"""
cam_bottom_thread.py
====================
Thread dédié à la caméra du bas (ArUco).

Responsabilités :
  - Lire la cam_bottom en continu
  - Détecter les tags ArUco
  - Convertir pixel → mm (repère table)
  - Écrire dans CupStateBuffer via update_from_bottom()
  - Émettre fps_signal pour debug

Ce thread ne fait RIEN d'autre. Pas de projection, pas de KCF, pas de mains.

FPS attendu : 30fps (ou la limite de la caméra).
Goulot actuel évité :
  - Avant : CSRT (~40ms) dans la même boucle → 5fps
  - Maintenant : ArUco seul (~3-8ms / frame) → 30fps
"""

import cv2
import json
import numpy as np
import time
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.config.app_config import CALIBRATION_TAG_IDS
from src.core.utils.paths import config_path
from src.core.vision.camera_manager import CameraManager


class CamBottomThread(QThread):
    """
    Thread caméra basse — détection ArUco uniquement.

    Signaux :
      fps_signal(float)          : FPS mesuré (émis chaque seconde)
      markers_signal(dict)       : {marker_id: [x_mm, y_mm]} cette frame
                                   (utile pour debug visuel, optionnel)
    """

    fps_signal     = pyqtSignal(float)
    markers_signal = pyqtSignal(dict)   # optionnel, pour debug/affichage

    def __init__(
        self,
        cup_state_buffer,           # CupStateBuffer partagé
        camera_manager: CameraManager,
        pose_path: str,
        calibration_tag_ids: set = None,
        show_preview: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.cup_state_buffer    = cup_state_buffer
        self.camera_manager      = camera_manager
        self.pose_path           = pose_path
        self.calibration_tag_ids = calibration_tag_ids or set(CALIBRATION_TAG_IDS)
        self.show_preview        = show_preview
        self.running             = False

        # Chargement pose cam_bottom
        pose      = json.load(open(pose_path, "r", encoding="utf-8"))
        self._rvec = np.array(pose["rvec"], dtype=np.float64)
        self._tvec = np.array(pose["tvec"], dtype=np.float64)
        self._K    = np.array(pose["camera_matrix"], dtype=np.float64)

        # Détecteur ArUco
        aruco_dict        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params            = cv2.aruco.DetectorParameters()
        self._detector    = cv2.aruco.ArucoDetector(aruco_dict, params)

        # FPS interne
        self._fps_count = 0
        self._fps_t0    = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Calibration : possibilité de surcharger K après undistort
    # ──────────────────────────────────────────────────────────────────

    def set_camera_matrix(self, K: np.ndarray) -> None:
        """Surcharge la matrice K (appelé par RecordWindow après undistort)."""
        self._K = K.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────
    # Boucle principale
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running    = True
        self._fps_count = 0
        self._fps_t0    = time.monotonic()

        print("[CamBottom] Thread démarré")

        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None or frame.size == 0:
                continue

            detected = self._process_frame(frame)

            # Mise à jour buffer partagé (thread-safe)
            self.cup_state_buffer.update_from_bottom(detected)

            # Signal optionnel pour debug UI
            if detected:
                self.markers_signal.emit(detected)

            # FPS
            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                fps = self._fps_count / (now - self._fps_t0)
                self.fps_signal.emit(fps)
                print(f"[CamBottom] FPS={fps:.1f}  markers={list(detected.keys())}")
                self._fps_count = 0
                self._fps_t0    = now

            # Preview optionnel (debug)
            if self.show_preview and frame is not None:
                self._show_preview(frame, detected)

        print("[CamBottom] Thread arrêté")
        if self.show_preview:
            cv2.destroyWindow("CamBottom Preview")

    def stop(self) -> None:
        self.running = False

    # ──────────────────────────────────────────────────────────────────
    # Traitement ArUco
    # ──────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> dict:
        """
        Détecte les ArUco et retourne {marker_id: [x_mm, y_mm]}.
        Filtre les tags de calibration.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return {}

        # Filtrer les tags de calibration
        valid = [
            i for i in range(len(ids))
            if int(ids[i][0]) not in self.calibration_tag_ids
        ]
        if not valid:
            return {}

        detected = {}
        for i in valid:
            marker_id = int(ids[i][0])
            pts       = corners[i][0]
            cx        = float(np.mean(pts[:, 0]))
            cy        = float(np.mean(pts[:, 1]))
            mm        = self._pixel_to_table(cx, cy)
            if mm is not None:
                detected[marker_id] = list(mm)

        return detected

    def _pixel_to_table(self, u: float, v: float):
        """Convertit un pixel cam_bottom → coordonnées mm repère table."""
        K_inv = np.linalg.inv(self._K)
        ray   = K_inv @ np.array([u, v, 1.0])
        ray   = ray / np.linalg.norm(ray)

        R, _         = cv2.Rodrigues(self._rvec)
        normal       = R[:, 2]
        plane_origin = self._tvec.reshape(3)

        denom = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        if t < 0:
            return None

        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - self._tvec.reshape(3))
        return float(pt_table[0]), float(pt_table[1])

    # ──────────────────────────────────────────────────────────────────
    # Preview debug
    # ──────────────────────────────────────────────────────────────────

    def _show_preview(self, frame: np.ndarray, detected: dict) -> None:
        preview = frame.copy()
        for marker_id, (x_mm, y_mm) in detected.items():
            cv2.putText(
                preview, f"ID{marker_id} ({x_mm:.0f},{y_mm:.0f})mm",
                (10, 30 + marker_id * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        cv2.imshow("CamBottom Preview",
                   cv2.resize(preview, (960, 540), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)
