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

        self.parent = parent  # MainApp
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.grid_size = grid_size

        # Key handler (évite crash)
        self.key_handler = KeyHandler(self)

        # Cam + affichage preview dans le QLabel de la UI
        self.camera_manager = CameraManager(
            camera_index=self.parent.camera_bottom_id,
            width=self.cam_width,
            height=self.cam_height
        )
        self.display_manager = DisplayManager(label=self.label_frame)

        # Service calibration (utilise self.parent.camera_manager -> ici parent = CalibrationMenu donc OK)
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

        # Status icon init (optionnel)
        invalidate_icon = asset_path("icons", "Invalidate.png")
        if hasattr(self, "label_status"):
            self.label_status.setPixmap(QPixmap(invalidate_icon))

        # Signals
        self.pushButton_startCam.clicked.connect(self.start_cam)
        self.pushButton_closeCam.clicked.connect(self.close_camera)
        self.pushButton_StartCali.clicked.connect(self.start_cali)
        self.pushButton_retour.clicked.connect(self.go_to_record)

        # Afficher preview caméra éteinte au démarrage
        self.display_manager.show_frame(self.image_cam_closed)

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)

    # ---------------------------------------------------------------------
    # Camera
    # ---------------------------------------------------------------------
    def start_cam(self):
        self.pushButton_closeCam.setEnabled(True)
        self.pushButton_StartCali.setEnabled(True)
        self.pushButton_startCam.setEnabled(False)

        if hasattr(self, "label_status"):
            self.label_status.setPixmap(QPixmap(asset_path("icons", "Invalidate.png")))

        # Ouvrir caméra puis lancer timer calibration (preview)
        self.camera_manager.open_camera()
        self.calibration.run()

    def close_camera(self):
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)

        # Stop timer calibration
        if self.calibration and self.calibration.timer:
            self.calibration.timer.stop()

        # Fermer caméra
        if self.camera_manager:
            self.camera_manager.close_camera()

        # Preview caméra éteinte
        self.display_manager.show_frame(self.image_cam_closed)

    # ---------------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------------
    def start_cali(self):
        """
        Lance la calibration (détection 4 tags 40-43), calcule H/H_inv
        et injecte les matrices dans record_window.
        """
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)

        try:
            H_proj, H_inv_proj, H_graph, H_inv_graph = self.calibration.start_calib(self.label_status)

            # Stop preview
            if self.calibration and self.calibration.timer:
                self.calibration.timer.stop()

            # Fermer caméra
            if self.camera_manager:
                self.camera_manager.close_camera()

            # Injecter dans RecordWindow
            rw = self.parent.record_window
            rw.calib_data["H"] = H_proj
            rw.calib_data["H_inv"] = H_inv_proj
            rw.calib_data["H_graph"] = H_graph
            rw.calib_data["H_inv_graph"] = H_inv_graph

            # Autoriser Start
            rw.pushButton_Start.setEnabled(True)

            print("✅ Calibration OK")
            print("H_proj:\n", H_proj)
            print("H_inv_proj:\n", H_inv_proj)

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur calibration", str(e))
            if hasattr(self, "label_status"):
                self.label_status.setPixmap(QPixmap(asset_path("icons", "Invalidate.png")))

    # ---------------------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------------------
    def go_to_record(self):
        self.close_camera()
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)
