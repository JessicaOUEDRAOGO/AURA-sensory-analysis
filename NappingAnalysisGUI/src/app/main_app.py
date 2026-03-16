# -*- coding: utf-8 -*-
from src.core.storage.db import init_db
import sys
import cv2
from PyQt6 import QtWidgets
import numpy as np
from src.core.utils.paths import asset_path
from src.core.projection.display_manager import DisplayManager
from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.service import ProtocolService

from src.ui.views.ui_main_menu_page import MainMenuPage
from src.ui.views.ui_protocol_home import ProtocolHomeWindow
from src.ui.views.ui_record_window import RecordWindow
from src.ui.views.ui_calibration_menu import CalibrationMenu
from src.ui.views.ui_reality_augmented_window import RealityAugementedWindow
from src.ui.views.ui_reality_augmented_bg import RealityAugementedWindowWithBG
from src.ui.views.ui_background_window import BackgroundWindow
from src.ui.views.ui_projection_background_bg import ProjectionBackgroundWindowWithBG
from src.ui.views.ui_settings_menu import SettingsMenu


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # SETTINGS
        # --------------------------------------------------
        from src.core.config.app_config import (
            CAMERA_ID,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            PROJECTOR_SCREEN_ID,
            PROJECTOR_WIDTH,
            PROJECTOR_HEIGHT,
            GRID_SIZE,
        )
        # SETTINGS GLOBALS
        # --------------------------------------------------
        self.settings = {
            "projector_screen_id": 1,
            "camera_id": 0,
            "projector_resolution": (3840, 2160)
        }
        self.projector_screen_id = PROJECTOR_SCREEN_ID
        self.projector_resolution = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

        self.camera_id = CAMERA_ID
        self.cam_width = CAMERA_WIDTH
        self.cam_height = CAMERA_HEIGHT

        self.grid_size = GRID_SIZE
        self.setWindowTitle("Projective Augmented Reality & Napping Collection Data")
        self.resize(1033, 1061)

        self.stacked_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Protocol services
        self.current_protocol = None
        self.protocol_repo = ProtocolRepository()
        self.protocol_service = ProtocolService(self.protocol_repo)

        # --------------------------------------------------
        # BACKGROUND (projecteur)
        # --------------------------------------------------
        # self.image_background = cv2.imread(asset_path("textures", "blanc_4k_carre_mid.png"))
        # if self.image_background is not None:
        #     print("Background shape:", self.image_background.shape)
        # else:
        #     print("Background image introuvable")
        # if self.image_background is None:
        #     self.image_background = 255 * (cv2.UMat(2160, 3840, cv2.CV_8UC3).get())
        self.image_background = np.full((2160, 3840, 3), 255, dtype=np.uint8)
        print("Background shape:", self.image_background.shape)
        # --------------------------------------------------
        # DISPLAY MANAGER
        # --------------------------------------------------
        self.display_manager = DisplayManager(projector_screen_id=self.projector_screen_id)
        self.display_manager.resolution = self.projector_resolution
        self.display_manager.display_image_on_projector_monitor(self.image_background)

        # --------------------------------------------------
        # UI PAGES
        # --------------------------------------------------
        self.main_menu = MainMenuPage(self)  # QWidget page
        self.protocol_home = ProtocolHomeWindow(self, self.protocol_service)
        self.settings_window = SettingsMenu(self)  # dialog

        nbr_tag = 5
        self.record_window = RecordWindow(
            self,
            nbr_tag,
            self.image_background,
            display_manager=self.display_manager,
            cam_width=self.cam_width,
            cam_height=self.cam_height,
            grid_size=self.grid_size,
        )

        self.calibration_window = CalibrationMenu(
            self,
            self.image_background,
            cam_width=self.cam_width,
            cam_height=self.cam_height,
            grid_size=self.grid_size,
        )

        self.RA_window = RealityAugementedWindowWithBG(self)
        self.Background_Window = ProjectionBackgroundWindowWithBG(self)

        # Stack (QUE des QWidget)
        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.protocol_home)
        self.stacked_widget.addWidget(self.record_window)
        self.stacked_widget.addWidget(self.calibration_window)
        self.stacked_widget.addWidget(self.RA_window)
        self.stacked_widget.addWidget(self.Background_Window)
        self.stacked_widget.setCurrentWidget(self.main_menu)

        # --------------------------------------------------
        # INITIALISATION PROTOCOLE
        # --------------------------------------------------
        self.current_protocol_id = None
        self.init_default_protocol()
        self.current_protocol_id = self.record_window.active_protocol_id
        self.current_protocol_locked = False

    def init_default_protocol(self):
        repo = ProtocolRepository()
        service = ProtocolService(repo)

        try:
            protocol = repo.get_by_name("PROTO_TEST")

            if protocol is None:
                protocol = service.create_new(name="PROTO_TEST", instruction_type="image")
                print("PROTO_TEST créé")
            else:
                print("PROTO_TEST déjà existant")

            if self.current_protocol is None:
                self.current_protocol = protocol
                print("Fallback protocole courant :", protocol.name, protocol.id)

        except Exception as e:
            print("Erreur init protocole :", e)


def main():
    print("\n============================================================")
    print("Projective Augmented Reality & Napping Collection Data")
    print("============================================================\n")

    init_db()

    app = QtWidgets.QApplication(sys.argv)
    qt_window = MainApp()
    qt_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
