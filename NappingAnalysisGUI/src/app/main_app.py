# -*- coding: utf-8 -*-
import sys
import cv2
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt

from src.core.utils.paths import gui_path, asset_path
from src.core.projection.display_manager import DisplayManager

from src.ui.views.ui_record_window import RecordWindow
from src.ui.views.ui_calibration_menu import CalibrationMenu
from src.ui.views.ui_reality_augmented_window import RealityAugementedWindow
from src.ui.views.ui_background_window import BackgroundWindow
from src.ui.views.ui_settings_menu import SettingsMenu


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Settings
        self.settings = {
            "projector_screen_id": 1,
            "camera_id": 0,
            "resolution": (3840, 2160),
        }

        self.cam_width = 3840
        self.cam_height = 2160
        self.grid_size = 700

        self.setWindowTitle("Projective Augemented Reality & Napping Collection Data")
        self.resize(1033, 1061)

        self.stacked_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Charger le background
        self.image_background = cv2.imread(asset_path("textures", "blanc_4k_carre_mid.png"))
        if self.image_background is None:
            # fallback safe
            self.image_background = (255 * (cv2.imread(asset_path("textures", "blanc_4k_carre_mid.png")) is not None))
            # si vraiment None -> image noire
            if self.image_background is None:
                self.image_background = 255 * (cv2.UMat(2160, 3840, cv2.CV_8UC3).get())

        # Display manager (projecteur)
        self.display_manager = DisplayManager(projector_screen_id=self.settings["projector_screen_id"])
        self.display_manager.resolution = self.settings["resolution"]
        self.display_manager.display_image_on_projector_monitor(self.image_background)

        # Pages UI
        self.main_menu = MainMenu(self)  # page menu

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
        self.parent.RA_window.setFocus()

    def go_to_record(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)
        self.parent.record_window.setFocus()

    def go_to_background(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.Background_Window)
        self.parent.Background_Window.setFocus()

    def go_to_settings(self):
        dlg = SettingsMenu(self.parent)
        dlg.exec()


    def quit_app(self):
        print("Fermeture de l'application.\n")
        QtWidgets.QApplication.quit()


def main():
    print("\n=====================================================================================")
    print("Projective Augmented Reality & Napping Collection Data")
    print("")
    print("Programme de Collect et de Gestion des mesures pour les expérimentations de Napping")
    print("Ainsi que la gestion de la réalité augmentée pour l'affichage sur le vidéoprojecteur")
    print("Réalisé par Mathieu ZEMAN sous la direction de Guillaume LAVOUÉ")
    print("dans le cadre d'un partenariat entre le LIRIS et l'INSTITUT LYFE")
    print("- Date : 08/04/2024")
    print("- Version 1.0.0")
    print("=====================================================================================\n")
    print("Lancement de l'application...")

    app = QtWidgets.QApplication(sys.argv)
    qt_window = MainApp()
    qt_window.show()
    print("Application lancée avec succès.\n")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
