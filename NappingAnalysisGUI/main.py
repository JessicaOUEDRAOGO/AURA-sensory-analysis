# -*- coding: utf-8 -*-
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt
import sys
import cv2

from logic.DisplayManager import DisplayManager

from logic.ui_record_window import RecordWindow
from logic.ui_calibration_menu import CalibrationMenu
from logic.ui_reality_augmented_window import RealityAugementedWindow
from logic.ui_background_window import BackgroundWindow
from logic.ui_settings_menu import SettingsMenu

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialise settings AVANT toute utilisation
        self.settings = {
            "projector_screen_id": 0,
            "camera_id": 0,
            "resolution": (3840, 2160)
        }

        self.cam_width = 3840  # Largeur de la caméra
        self.cam_height = 2160  # Hauteur de la caméra
        self.grid_size = 700

        self.setWindowTitle("Projective Augemented Reality & Napping Collection Data")
        self.resize(1033, 1061)

        self.stacked_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Charger les interfaces UI
        self.main_menu = MainMenu(self)
        
        self.image_background = cv2.imread("./assets/textures/blanc_4k_carre_mid.png")
        
        self.display_manager = DisplayManager(
            projector_screen_id=self.settings["projector_screen_id"]
        )
        self.display_manager.resolution = self.settings["resolution"]
        self.display_manager.display_image_on_projector_monitor(self.image_background)

        nbr_tag = 5    
        self.record_window = RecordWindow(self, nbr_tag, self.image_background, display_manager=self.display_manager, cam_width=self.cam_width, cam_height=self.cam_height, grid_size=self.grid_size)
        self.calibration_window = CalibrationMenu(self, self.image_background, cam_width=self.cam_width, cam_height=self.cam_height, grid_size=self.grid_size)
        self.RA_window = RealityAugementedWindow(self)
        self.Background_Window = BackgroundWindow(self)

        # Ajouter les menus au QStackedWidget
        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.record_window)
        self.stacked_widget.addWidget(self.calibration_window)
        self.stacked_widget.addWidget(self.RA_window)
        self.stacked_widget.addWidget(self.Background_Window)

        # Afficher le menu principal au lancement
        self.stacked_widget.setCurrentWidget(self.main_menu)

class MainMenu(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        uic.loadUi("./gui/Main_Menu.ui", self)
        self.parent = parent  # Référence à MainApp

        # Bouton pour ouvrir RecordWindow
        self.pushButton_Record.clicked.connect(self.go_to_record)
        self.pushButton_2_ARS.clicked.connect(self.go_to_RA)
        self.pushButton_background.clicked.connect(self.go_to_background)
        self.pushButton_Settings.clicked.connect(self.go_to_settings)
        
        # Bouton pour quitter l'application
        self.pushButton_Quit.clicked.connect(self.quit_app)

    def go_to_RA(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.RA_window)
        self.parent.RA_window.setFocus()  # <-- Ajoute cette ligne

    def go_to_record(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)  # Changer de page
        self.parent.record_window.setFocus()  # <-- Ajoute cette ligne

    def go_to_background(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.Background_Window)
        self.parent.Background_Window.setFocus()  # <-- Ajoute cette ligne

    def go_to_settings(self):
        self.settings_window = SettingsMenu(self.parent)
        self.settings_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def quit_app(self):
        print("Fermeture de l'application.\n")
        QtWidgets.QApplication.quit()  # Ferme complètement l'application

# === Main ===
def main():
    global qt_window
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

    # Lancer Flask en thread
    # flask_thread = threading.Thread(target=launch_flask)
    # flask_thread.daemon = True
    # flask_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    qt_window = MainApp()
    qt_window.show()
    print("Application lancée avec succès.\n")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()