# -*- coding: utf-8 -*-
"""
main_app.py — VERSION v5
==========================
Nettoyage vs v4 :
  - HandTrackingThread supprimé
  - HandStateBuffer supprimé
  - FrameBuffer supprimé
  - shared_hand_buffer supprimé
  - shared_frame_buffer supprimé
  - start_hand_tracking() / stop_hand_tracking() supprimés
  - MainApp ne gère plus aucun thread de caméra
    (CamBottomThread et CamTopThread sont démarrés par Algorithm_Analysis)

MainApp ne contient plus que :
  - La configuration des fenêtres Qt
  - Le DisplayManager
  - La navigation entre écrans
"""

from src.core.storage.db import init_db
import sys
import cv2
import numpy as np
from PyQt6 import QtWidgets

from src.core.utils.paths import asset_path
from src.core.projection.display_manager import DisplayManager
from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.service import ProtocolService

from src.ui.views.ui_main_menu_page         import MainMenuPage
from src.ui.views.ui_protocol_home          import ProtocolHomeWindow
from src.ui.views.ui_record_window          import RecordWindow
from src.ui.views.ui_calibration_menu       import CalibrationMenu
from src.ui.views.ui_reality_augmented_bg   import RealityAugementedWindowWithBG
from src.ui.views.ui_projection_background_bg import ProjectionBackgroundWindowWithBG
from src.ui.views.ui_settings_menu          import SettingsMenu


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        from src.core.config.app_config import (
            CAMERA_ID,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            PROJECTOR_SCREEN_ID,
            PROJECTOR_WIDTH,
            PROJECTOR_HEIGHT,
            GRID_SIZE,
        )

        # IDs des caméras
        self.camera_top_id    = 0   # cam_top  (vue de dessus, KCF)
        self.camera_bottom_id = 1   # cam_bottom (vue du bas, ArUco)

        self.settings = {
            "projector_screen_id":  1,
            "camera_id":            self.camera_bottom_id,
            "projector_resolution": (PROJECTOR_WIDTH, PROJECTOR_HEIGHT),
        }
        self.projector_screen_id  = PROJECTOR_SCREEN_ID
        self.projector_resolution = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

        self.camera_id  = CAMERA_ID
        self.cam_width  = CAMERA_WIDTH
        self.cam_height = CAMERA_HEIGHT
        self.grid_size  = GRID_SIZE

        self.setWindowTitle("Projective Augmented Reality & Napping Collection Data")
        self.resize(1033, 1061)

        self.stacked_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.current_protocol = None
        self.protocol_repo    = ProtocolRepository()
        self.protocol_service = ProtocolService(self.protocol_repo)

        # Image de fond projecteur
        self.image_background = np.zeros(
            (PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8
        )

        # DisplayManager
        self.display_manager = DisplayManager(
            projector_screen_id=self.projector_screen_id
        )
        self.display_manager.resolution = self.projector_resolution
        self.display_manager.display_image_on_projector_monitor(self.image_background)

        # Fenêtres Qt
        self.main_menu       = MainMenuPage(self)
        self.protocol_home   = ProtocolHomeWindow(self, self.protocol_service)
        self.settings_window = SettingsMenu(self)

        self.record_window = RecordWindow(
            self,
            nbr_Tag=5,
            image_background=self.image_background,
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

        self.RA_window         = RealityAugementedWindowWithBG(self)
        self.Background_Window = ProjectionBackgroundWindowWithBG(self)

        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.protocol_home)
        self.stacked_widget.addWidget(self.record_window)
        self.stacked_widget.addWidget(self.calibration_window)
        self.stacked_widget.addWidget(self.RA_window)
        self.stacked_widget.addWidget(self.Background_Window)
        self.stacked_widget.setCurrentWidget(self.main_menu)

        self.current_protocol_id     = None
        self.current_protocol_locked = False
        self.init_default_protocol()
        self.current_protocol_id = self.record_window.active_protocol_id

    # ──────────────────────────────────────────────────────────────────
    # Protocole par défaut
    # ──────────────────────────────────────────────────────────────────

    def init_default_protocol(self):
        repo    = ProtocolRepository()
        service = ProtocolService(repo)
        try:
            protocol = repo.get_by_name("PROTO_TEST")
            if protocol is None:
                protocol = service.create_new(
                    name="PROTO_TEST", instruction_type="image"
                )
                print("[Main] PROTO_TEST créé")
            else:
                print("[Main] PROTO_TEST déjà existant")
            if self.current_protocol is None:
                self.current_protocol = protocol
        except Exception as e:
            print(f"[Main] Erreur init protocole : {e}")

    # ──────────────────────────────────────────────────────────────────
    # Fermeture
    # ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # RecordWindow.stop_recording() gère l'arrêt des threads
        if self.record_window:
            self.record_window.stop_recording()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n============================================================")
    print("Projective Augmented Reality & Napping Collection Data — v5")
    print("============================================================\n")

    init_db()

    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
