# -*- coding: utf-8 -*-
from src.core.storage.db import init_db
import sys
import cv2
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt

from src.core.utils.paths import gui_path, asset_path
from src.core.projection.display_manager import DisplayManager
from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.service import ProtocolService

from src.ui.views.ui_record_window import RecordWindow
from src.ui.views.ui_calibration_menu import CalibrationMenu
from src.ui.views.ui_reality_augmented_window import RealityAugementedWindow
from src.ui.views.ui_background_window import BackgroundWindow
from src.ui.views.ui_settings_menu import SettingsMenu


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # SETTINGS
        # --------------------------------------------------
        self.settings = {
            "projector_screen_id": 1,
            "camera_id": 0,
            "resolution": (3840, 2160),
        }

        self.cam_width = 3840
        self.cam_height = 2160
        self.grid_size = 700

        self.setWindowTitle("Projective Augmented Reality & Napping Collection Data")
        self.resize(1033, 1061)

        self.stacked_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # --------------------------------------------------
        # BACKGROUND
        # --------------------------------------------------
        self.image_background = cv2.imread(asset_path("textures", "blanc_4k_carre_mid.png"))
        if self.image_background is None:
            self.image_background = 255 * (cv2.UMat(2160, 3840, cv2.CV_8UC3).get())

        # --------------------------------------------------
        # DISPLAY MANAGER
        # --------------------------------------------------
        self.display_manager = DisplayManager(
            projector_screen_id=self.settings["projector_screen_id"]
        )
        self.display_manager.resolution = self.settings["resolution"]
        self.display_manager.display_image_on_projector_monitor(self.image_background)

        # --------------------------------------------------
        # UI PAGES
        # --------------------------------------------------
        self.main_menu = MainMenu(self)

        nbr_tag = 5
        self.record_window = RecordWindow(
            self,
            nbr_tag,
            self.image_background,
            display_manager=self.display_manager,
            cam_width=self.cam_width,
            cam_height=self.cam_height,
            grid_size=self.grid_size
        )

        self.calibration_window = CalibrationMenu(
            self,
            self.image_background,
            cam_width=self.cam_width,
            cam_height=self.cam_height,
            grid_size=self.grid_size
        )

        self.RA_window = RealityAugementedWindow(self)
        self.Background_Window = BackgroundWindow(self)

        # Stack
        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.record_window)
        self.stacked_widget.addWidget(self.calibration_window)
        self.stacked_widget.addWidget(self.RA_window)
        self.stacked_widget.addWidget(self.Background_Window)
        self.stacked_widget.setCurrentWidget(self.main_menu)

        # --------------------------------------------------
        # INITIALISATION PROTOCOLE (FIX FOREIGN KEY)
        # --------------------------------------------------
        self.init_default_protocol()

    # --------------------------------------------------
    # Création automatique d’un protocole par défaut
    # --------------------------------------------------
    def init_default_protocol(self):
        repo = ProtocolRepository()
        service = ProtocolService(repo)

        try:
            protocol = repo.get_by_name("PROTO_TEST")

            if protocol is None:
                protocol = service.create_new(
                    name="PROTO_TEST",
                    instruction_type="image"
                )
                print("PROTO_TEST créé")

            else:
                print("PROTO_TEST déjà existant")

            # Injection dans RecordWindow
            self.record_window.active_protocol_id = protocol.id
            self.record_window.active_participant_id = "P001"

            print("Protocol utilisé :", protocol.id)

        except Exception as e:
            print("Erreur init protocole :", e)


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------
class MainMenu(QtWidgets.QMainWindow):
    def __init__(self, parent: MainApp):
        super().__init__()
        uic.loadUi(gui_path("Main_Menu.ui"), self)
        self.parent = parent

        self.pushButton_Record.clicked.connect(self.go_to_record)
        self.pushButton_2_ARS.clicked.connect(self.go_to_RA)
        self.pushButton_background.clicked.connect(self.go_to_background)
        self.pushButton_Settings.clicked.connect(self.go_to_settings)
        self.pushButton_Quit.clicked.connect(self.quit_app)

    def go_to_RA(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.RA_window)

    def go_to_record(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)

    def go_to_background(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.Background_Window)

    def go_to_settings(self):
        dlg = SettingsMenu(self.parent)
        dlg.exec()

    def quit_app(self):
        QtWidgets.QApplication.quit()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("\n============================================================")
    print("Projective Augmented Reality & Napping Collection Data")
    print("============================================================\n")

    # Initialisation base
    init_db()

    app = QtWidgets.QApplication(sys.argv)
    qt_window = MainApp()
    qt_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
