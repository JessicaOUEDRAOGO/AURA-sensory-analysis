# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PyQt6 import uic, QtWidgets
from PyQt6.QtGui import QPixmap

from src.core.utils.paths import gui_path, asset_path
from src.ui.controllers.key_handler import KeyHandler

from src.core.vision.camera_manager import CameraManager
from src.core.projection.display_manager import DisplayManager
from src.core.calibration.calibration_service import Calibration


class CalibrationMenu(QtWidgets.QWidget):
    def __init__(self, parent, image_background, cam_width=3840, cam_height=2160, grid_size=700):
        super().__init__()
        uic.loadUi(gui_path("Calibration_Menu.ui"), self)

        self.parent     = parent  # MainApp
        self.cam_width  = cam_width
        self.cam_height = cam_height
        self.grid_size  = grid_size

        self.key_handler = KeyHandler(self)

        # Caméra + preview dans le QLabel de la UI
        self.camera_manager = CameraManager(
            camera_index=self.parent.camera_bottom_id,
            width=self.cam_width,
            height=self.cam_height
        )
        self.display_manager = DisplayManager(label=self.label_frame)

        # Service calibration (preview caméra uniquement dans le nouveau pipeline)
        self.calibration = Calibration(
            self,
            self.cam_width,
            self.cam_height,
            self.grid_size,
            image_background
        )

        # UI state
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)

        # Image caméra éteinte
        cam_closed_path = asset_path("images", "camera_eteinte.png")
        self.image_cam_closed = cv2.imread(cam_closed_path)
        if self.image_cam_closed is None:
            self.image_cam_closed = np.zeros((480, 640, 3), dtype=np.uint8)

        # Icône status initiale
        if hasattr(self, "label_status"):
            self.label_status.setPixmap(QPixmap(asset_path("icons", "Invalidate.png")))

        # Signals
        self.pushButton_startCam.clicked.connect(self.start_cam)
        self.pushButton_closeCam.clicked.connect(self.close_camera)
        self.pushButton_StartCali.clicked.connect(self.check_pipeline_files)
        self.pushButton_retour.clicked.connect(self.go_to_record)

        # Afficher preview caméra éteinte au démarrage
        self.display_manager.show_frame(self.image_cam_closed)

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Caméra
    # ------------------------------------------------------------------
    def start_cam(self):
        self.pushButton_closeCam.setEnabled(True)
        self.pushButton_StartCali.setEnabled(True)
        self.pushButton_startCam.setEnabled(False)

        if hasattr(self, "label_status"):
            self.label_status.setPixmap(QPixmap(asset_path("icons", "Invalidate.png")))

        self.camera_manager.open_camera()
        self.calibration.run()

    def close_camera(self):
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)

        if self.calibration and self.calibration.timer:
            self.calibration.timer.stop()

        if self.camera_manager:
            self.camera_manager.close_camera()

        self.display_manager.show_frame(self.image_cam_closed)

    # ------------------------------------------------------------------
    # Vérification pipeline (remplace start_cali legacy)
    # ------------------------------------------------------------------
    def check_pipeline_files(self):
        """
        Vérifie que les 3 fichiers JSON du nouveau pipeline sont présents.
        N'injecte plus aucune matrice dans record_window — Algorithm_Analysis
        les charge lui-même au démarrage de l'enregistrement.
        """
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)

        # Stop preview + fermer caméra
        if self.calibration and self.calibration.timer:
            self.calibration.timer.stop()
        if self.camera_manager:
            self.camera_manager.close_camera()

        # Vérification
        ok = self.calibration.check_new_pipeline_files(
            label_status=self.label_status if hasattr(self, "label_status") else None
        )

        if ok:
            print("✅ Fichiers pipeline OK — prêt à enregistrer")
            QtWidgets.QMessageBox.information(
                self,
                "Pipeline OK",
                "Tous les fichiers de calibration sont présents.\n"
                "Tu peux lancer l'enregistrement."
            )
        else:
            msg = (
                "Fichiers manquants dans config/ :\n"
                "  • cambottom_table_pose.json\n"
                "  • H_table_to_proj.json\n"
                "  • camera_calibration_bottom.json\n\n"
                "Lance les scripts de calibration correspondants."
            )
            QtWidgets.QMessageBox.warning(self, "Fichiers manquants", msg)
            if hasattr(self, "label_status"):
                self.label_status.setPixmap(QPixmap(asset_path("icons", "Invalidate.png")))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def go_to_record(self):
        self.close_camera()
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)