# -*- coding: utf-8 -*-
from src.core.utils.paths import gui_path, asset_path
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import uic, QtWidgets
import cv2
import numpy as np

from src.core.vision.camera_manager import CameraManager
from src.core.projection.display_manager import DisplayManager
from src.core.config.app_config import CAMERA_WIDTH, CAMERA_HEIGHT


class SettingsMenu(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        # UI
        uic.loadUi(gui_path("Settings_Menu.ui"), self)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.parent = parent

        # -----------------------------
        # SETTINGS INIT
        # -----------------------------
        self.initial_proj_screen_id = self.parent.settings["projector_screen_id"]
        self.initial_cam_id = self.parent.settings["camera_id"]
        self.initial_projector_resolution = self.parent.settings["projector_resolution"]

        # Valeurs temporaires
        self.temp_proj_screen_id = self.initial_proj_screen_id
        self.temp_cam_id = self.initial_cam_id
        self.temp_projector_resolution = self.initial_projector_resolution

        # -----------------------------
        # SETUP UI FIELDS
        # -----------------------------
        self.lineEdit_NumProjScren.setText(str(self.initial_proj_screen_id))
        self.lineEdit_NumCam.setText(str(self.initial_cam_id))
        self.checkBox_4K.setChecked(self.initial_projector_resolution == (3840, 2160))
        self.checkBox_FullHD.setChecked(self.initial_projector_resolution == (1920, 1080))

        # DisplayManager local pour preview dans le label
        self.display_manager = DisplayManager(
            label=self.label_CameraPreview,
            projector_screen_id=self.initial_proj_screen_id
        )

        # Camera preview
        self.camera_manager = None
        self.timer = None

        # Connexions
        self.pushButton_ScrenTest.clicked.connect(self.test_projection_screen)

        self.pushButton_StartCam.clicked.connect(self.start_camera)
        self.pushButton_StartCam.setEnabled(True)

        self.pushButton_StopCam.clicked.connect(self.stop_camera)
        self.pushButton_StopCam.setEnabled(False)

        self.checkBox_4K.clicked.connect(self.set_4k)
        self.checkBox_FullHD.clicked.connect(self.set_fullhd)

        self.pushButton_Appliquer.clicked.connect(self.apply_and_close)

        # Image caméra éteinte
        self.image_cam_closed = cv2.imread(asset_path("images", "camera_eteinte.png"))
        if self.image_cam_closed is None:
            self.image_cam_closed = np.zeros((480, 640, 3), dtype=np.uint8)

        self.display_manager.show_frame(self.image_cam_closed)

    def test_projection_screen(self):
        try:
            screen_id = int(self.lineEdit_NumProjScren.text())
            self.temp_proj_screen_id = screen_id

            self.parent.display_manager.display_image_on_projector_monitor(
                self.parent.image_background,
                screen_id=screen_id
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"ID écran invalide : {e}")

    def start_camera(self):
        self.pushButton_StopCam.setEnabled(True)
        self.pushButton_StartCam.setEnabled(False)

        try:
            cam_id = int(self.lineEdit_NumCam.text())
            self.temp_cam_id = cam_id

            if self.camera_manager is not None:
                self.stop_camera()

            # La caméra reste verrouillée en 1920x1080
            self.camera_manager = CameraManager(
                camera_index=cam_id,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT
            )
            self.camera_manager.open_camera()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_camera_frame)
            self.timer.start(30)

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Impossible de démarrer la caméra : {e}")
            self.pushButton_StopCam.setEnabled(False)
            self.pushButton_StartCam.setEnabled(True)

    def update_camera_frame(self):
        try:
            frame = self.camera_manager.get_frame() if self.camera_manager else None
            if frame is not None:
                self.display_manager.show_frame(frame)
            else:
                error_img = cv2.imread(asset_path("images", "CamError.png"))
                if error_img is not None:
                    self.display_manager.show_frame(error_img)
        except Exception:
            error_img = cv2.imread(asset_path("images", "CamError.png"))
            if error_img is not None:
                self.display_manager.show_frame(error_img)

    def stop_camera(self):
        self.pushButton_StopCam.setEnabled(False)
        self.pushButton_StartCam.setEnabled(True)

        if self.timer:
            self.timer.stop()
            self.timer = None

        if self.camera_manager:
            self.camera_manager.close_camera()
            self.camera_manager = None

        try:
            self.display_manager.close_display()
        except Exception:
            pass

        self.display_manager.show_frame(self.image_cam_closed)

    def set_4k(self):
        self.checkBox_4K.setChecked(True)
        self.checkBox_FullHD.setChecked(False)
        self.temp_projector_resolution = (3840, 2160)

    def set_fullhd(self):
        self.checkBox_4K.setChecked(False)
        self.checkBox_FullHD.setChecked(True)
        self.temp_projector_resolution = (1920, 1080)

    def apply_and_close(self):
        self.stop_camera()

        # Appliquer settings dans MainApp
        self.parent.settings["projector_screen_id"] = self.temp_proj_screen_id
        self.parent.settings["camera_id"] = self.temp_cam_id
        self.parent.settings["projector_resolution"] = self.temp_projector_resolution

        # Update DisplayManager principal
        self.parent.display_manager.projector_screen_id = self.temp_proj_screen_id
        self.parent.display_manager.resolution = self.temp_projector_resolution

        # Mise à jour éventuelle d'autres composants
        if hasattr(self.parent, "record_window") and self.parent.record_window is not None:
            if hasattr(self.parent.record_window, "camera_manager") and self.parent.record_window.camera_manager:
                if hasattr(self.parent.record_window.camera_manager, "camera_index"):
                    self.parent.record_window.camera_manager.camera_index = self.temp_cam_id

        self.parent.display_manager.display_image_on_projector_monitor(
            self.parent.image_background,
            screen_id=self.temp_proj_screen_id
        )
        self.close()

    def closeEvent(self, event):
        self.stop_camera()

        self.parent.display_manager.display_image_on_projector_monitor(
            self.parent.image_background,
            screen_id=self.initial_proj_screen_id
        )
        event.accept()