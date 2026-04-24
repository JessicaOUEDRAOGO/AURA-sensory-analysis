# -*- coding: utf-8 -*-
"""
calibration_service.py
~~~~~~~~~~~~~~~~~~~~~~
Service de calibration simplifié.

Nouveau pipeline (actif) :
  - cambottom_table_pose.json  →  rvec, tvec, camera_matrix
  - H_table_to_proj.json       →  H_table_to_proj
  - camera_calibration_bottom.json  →  K, dist (undistort)

Les matrices H_proj / H_inv_proj / H_graph / H_inv_graph et le fichier
calibration_data.json sont SUPPRIMÉS du pipeline.
"""

import os
import cv2

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

from src.core.utils.paths import asset_path


class Calibration:
    def __init__(self, parent, cam_width, cam_height, grid_size, image_background):
        self.parent = parent
        self.cam_width  = cam_width
        self.cam_height = cam_height
        self.grid_size  = grid_size

        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.update_frame)

        self.frame = None

        self.image_background = image_background
        self.image_width  = image_background.shape[1]
        self.image_height = image_background.shape[0]

    # ------------------------------------------------------------------
    # Aperçu caméra
    # ------------------------------------------------------------------
    def run(self):
        self.timer.start(33)  # ~30 FPS
        return False

    def update_frame(self, last_frame=None):
        if last_frame is not None:
            self.parent.display_manager.show_frame(last_frame)
            return last_frame

        frame = self.parent.camera_manager.get_frame()
        if frame is None:
            print("Erreur : Impossible de capturer une image de la caméra.")
            self.timer.stop()
            return None

        self.frame = frame

        frame_small = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        self.parent.display_manager.show_frame(frame_small)
        return frame

    # ------------------------------------------------------------------
    # Vérification rapide des fichiers JSON du nouveau pipeline
    # ------------------------------------------------------------------
    def check_new_pipeline_files(self, label_status=None):
        """
        Vérifie que les 3 fichiers JSON du nouveau pipeline sont présents.
        Affiche une icône de validation si label_status est fourni.
        Retourne True si tout est OK, False sinon.
        """
        from src.core.utils.paths import config_path

        required = [
            "cambottom_table_pose.json",
            "H_table_to_proj.json",
            "camera_calibration_fisheye.json",
        ]

        missing = []
        for fname in required:
            path = config_path(fname)
            if not os.path.exists(path):
                missing.append(fname)

        if missing:
            print(f"[Calibration] Fichiers manquants : {missing}")
            return False

        print("[Calibration] Tous les fichiers JSON du pipeline sont présents.")

        if label_status is not None:
            validate_icon = asset_path("icons", "Validate.png")
            if os.path.exists(validate_icon):
                label_status.setPixmap(QPixmap(validate_icon))

        return True

    def __del__(self):
        pass