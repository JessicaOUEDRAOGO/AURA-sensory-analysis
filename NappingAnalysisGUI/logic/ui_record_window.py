from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer, QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QLabel, QHBoxLayout, QGraphicsView, QFrame
from logic.key_handler import KeyHandler
from logic.graphics_scene import GraphicsScene
from logic.CameraManager import CameraManager
from logic.Algorithm_Calibration import Calibration
from logic.Algorithm_Analysis import Algorithm_Analysis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
import glob
import os
from PyQt6 import QtCore

class RecordWindow(QtWidgets.QWidget):
    def __init__(self, parent, nbr_Tag, image_background, display_manager = None, cam_width = 3840, cam_height = 2160, grid_size = 700):
        super().__init__()
        uic.loadUi("./gui/Record_Menu.ui", self)
        
        self.parent = parent  # Référence à MainApp
        self.nbr_Tag = nbr_Tag  # Nombre de tags à afficher
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.algorithm_thread = None
        self.algorithm_analysis = None
        self.grid_size = grid_size
        self.image_background = image_background

        self.key_handler = KeyHandler(self)

        self.camera_manager = CameraManager(camera_index = self.parent.settings["camera_id"], width=self.cam_width, height=self.cam_height)
        self.display_manager = display_manager

        self.calib_data =  {
            "H": None,
            "H_inv": None,
            "H_graph": None,
            "H_inv_graph": None,
            "grid_size": grid_size
        }

        self.image_background_clean = image_background.copy()
        self.image_background_with_grid = image_background.copy()

        self.checkboxes = []

        self.pushButton_Stop.setEnabled(False)
        self.pushButton_Start.setEnabled(False)

        # Timer-related attributes
        self.timer = QTimer(self)  # QTimer instance
        self.timer.timeout.connect(self.update_timer_label)  # Connect timeout signal to update method
        self.elapsed_time = 0  # Time elapsed in seconds
        self.timer_started = False  # Flag to indicate if the timer has started

        # Frame counter attributes
        self.frame_count = 0  # Initialize frame count

        self.calibration_window = None

        self.scene = None

        self.default_xmin = -10
        self.default_xmax = 10
        self.default_ymin = -10
        self.default_ymax = 10
        self.default_xleg = "x"
        self.default_yleg = "y"
        self.setup_graphics_view(self.default_xmin, self.default_xmax, self.default_ymin, self.default_ymax, self.default_xleg, self.default_yleg, self.grid_size)

        # Écrire les valeurs par défaut dans les QLineEdit
        self.set_bounds_to_inputs()
        
        # Bouton pour revenir au MainMenu
        self.pushButton_return.clicked.connect(self.go_to_main)

        # Connexion du bouton de mise à jour
        self.pushButton_UpdateChanges.clicked.connect(self.update_bounds)

        # Connexion du bouton de Calibration
        self.pushButton_calibration.clicked.connect(self.start_calibration)

        self.pushButton_Start.clicked.connect(self.start_recording)
        self.pushButton_Stop.clicked.connect(self.stop_recording)
        self.pushButton_loadCalibration.clicked.connect(self.loadCalib)
        # Configurer le QGraphicsView

        # filepath: main.py
        # ...dans RecordWindow.__init__...
        self.checkBox_DisplayGrid.stateChanged.connect(self.on_display_grid_checkbox_changed)
        # ...et après update_bounds...
        self.pushButton_UpdateChanges.clicked.connect(self.refresh_projector_background)

        self.create_tags()

    def keyPressEvent(self, event):
        if self.timer_started == False:
            self.key_handler.handle_key(event)
        else:
            self.key_handler.handle_key_for_program_started(event)
        super().keyPressEvent(event)

    def refresh_projector_background(self):
        # Récupérer les bornes actuelles
        x_min, x_max, y_min, y_max, x_legend, y_legend = self.get_bounds_from_inputs()
        if self.checkBox_DisplayGrid.isChecked():
            from logic.DrawUtils import DrawUtils
            img_with_grid = DrawUtils.draw_math_grid_on_image(
                self.image_background_clean, x_min, x_max, y_min, y_max, x_legend, y_legend, self.grid_size
            )
            self.image_background_with_grid = img_with_grid
            self.display_manager.display_image_on_projector_monitor(img_with_grid)
        else:
            self.display_manager.display_image_on_projector_monitor(self.image_background_clean)

    def get_latest_background_path(self):
        """
        Retourne le chemin du dernier background PNG sauvegardé dans ./assets/textures.
        """
        files = glob.glob("./assets/textures/background_*.png")
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        print(f"Latest background file: {latest_file}")
        return latest_file

    def start_recording(self):
        self.algorithm_analysis = Algorithm_Analysis(self, self.display_manager,
            self.calib_data["H"], self.calib_data["H_inv"],
            self.calib_data["H_graph"], self.calib_data["H_inv_graph"],
            self.image_background,
            record_window=self  # <-- Ajoute ceci
        )
        self.algorithm_analysis.set_show_grid(self.checkBox_DisplayGrid.isChecked())

        self.key_handler.algorithm_analysis = self.algorithm_analysis  # Mettre à jour le key_handler avec l'instance d'analyse

        self.connect_checkbox_to_algorithm()

        self.camera_manager.open_camera()

        # Créer un thread
        self.algorithm_thread = QThread()
        self.algorithm_analysis.moveToThread(self.algorithm_thread)

        # Connecter les signaux et slots
        self.algorithm_thread.started.connect(self.algorithm_analysis.detect_and_process)
        self.algorithm_analysis.data_signal.connect(self.update_ui)  # Mettre à jour l'interface utilisateur
        self.algorithm_analysis.data_signal.connect(self.start_timer_on_first_frame)  # Start timer on first frame
        self.algorithm_thread.finished.connect(self.algorithm_thread.deleteLater)

        # Démarrer le thread
        self.algorithm_thread.start()
        
        # Reset frame count
        self.frame_count = 0
        self.update_frame_label()  # Reset the label

        # Reset timer state
        self.elapsed_time = 0
        self.timer_started = False  # Reset the flag
        self.update_timer_label()  # Reset the timer label

        self.pushButton_Start.setEnabled(False)
        self.pushButton_Stop.setEnabled(True)

    def start_timer_on_first_frame(self, data):
        """
        Start the timer when the first frame is received.
        """
        if not self.timer_started:
            self.timer_started = True
            self.timer.start(1000)  # Start the timer to update every second

    def stop_recording(self):
        if self.algorithm_analysis:
            self.algorithm_analysis.stop()
        if self.algorithm_thread:
            if self.algorithm_thread.isRunning():  # Vérifie si le thread est encore actif
                self.algorithm_thread.quit()
                self.algorithm_thread.wait()
            self.algorithm_thread = None  # Réinitialise le thread pour éviter des références invalides
        
        if self.camera_manager:
            self.camera_manager.close_camera()

        # Stop the timer
        self.timer.stop()

        self.pushButton_Start.setEnabled(True)
        self.pushButton_Stop.setEnabled(False)

    def update_ui(self, data):
        """
        Met à jour l'interface utilisateur avec les données reçues.
        Affiche les marqueurs détectés dans la GraphicsScene.
        :param data: Dictionnaire contenant les données des marqueurs.
        """
        projector_coords_ArUco = data["data"]

        # Increment frame count and update label
        self.frame_count += 1
        self.update_frame_label()

        # Effacer les anciens points de la scène
        self.scene.clear_markers()

        # Suivre les IDs des marqueurs détectés
        detected_ids = set()

        # Ajouter les nouveaux points et ID des marqueurs
        for marker_id, graph_coords in projector_coords_ArUco:
            x, y = graph_coords

            detected_ids.add(marker_id)

            # Vérifier si la checkbox correspondante est cochée
            checkbox_index = marker_id - 44  # Si tes tags sont 44 à 49
            if 0 <= checkbox_index < len(self.checkboxes) and self.checkboxes[checkbox_index].isChecked():
                # Ajouter le marqueur à la scène
                self.scene.add_marker(x, y, marker_id)

            # Mettre à jour les labels correspondants
            label_posx_nbr = self.findChild(QLabel, f"label_posx_nbr_{marker_id}")
            label_posy_nbr = self.findChild(QLabel, f"label_posy_nbr_{marker_id}")
            if label_posx_nbr and label_posy_nbr:
                label_posx_nbr.setText(f"{self.scene.pixel_to_index_x(x):.2f}")
                label_posy_nbr.setText(f"{self.scene.pixel_to_index_y(y):.2f}")

        # Réinitialiser les labels des marqueurs non détectés
        for i in range(len(self.checkboxes)):
            marker_id = i + 1  # Les IDs commencent à 1
            if marker_id not in detected_ids:
                label_posx_nbr = self.findChild(QLabel, f"label_posx_nbr_{marker_id}")
                label_posy_nbr = self.findChild(QLabel, f"label_posy_nbr_{marker_id}")
                if label_posx_nbr and label_posy_nbr:
                    label_posx_nbr.setText("?")
                    label_posy_nbr.setText("?")

    def update_frame_label(self):
        """
        Update the frame label with the current frame count.
        """
        self.label_nbr_frame.setText(str(self.frame_count))

    def update_timer_label(self):
        """
        Update the timer label with the elapsed time.
        """
        self.elapsed_time += 1
        self.label_timer.setText(f"Timer : {self.elapsed_time} sec")

    def start_calibration(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.calibration_window)  # Changer de page

    def loadCalib(self):

        self.calibration = Calibration(self, self.cam_width, self.cam_height, self.grid_size, self.image_background)
        
        H_proj, H_inv_proj, H_graph, H_inv_graph  = self.calibration.load_calib()

        self.parent.record_window.calib_data["H"] = H_proj
        self.parent.record_window.calib_data["H_inv"] = H_inv_proj
        self.parent.record_window.calib_data["H_graph"] = H_graph
        self.parent.record_window.calib_data["H_inv_graph"] = H_inv_graph
        self.parent.record_window.pushButton_Start.setEnabled(True)

    def go_to_main(self):

        self.stop_recording()
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)  # Revenir au menu principal

    def get_bounds_from_inputs(self):
        """Récupère les valeurs des QLineEdit et les convertit en float."""
        try:
            x_min = float(self.lineEdit_xmin.text())
            x_max = float(self.lineEdit_xmax.text())
            y_min = float(self.lineEdit_ymin.text())
            y_max = float(self.lineEdit_ymax.text())
            x_legend = self.lineEdit_legx.text()
            y_legend = self.lineEdit_legy.text()
            return x_min, x_max, y_min, y_max, x_legend, y_legend
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer des valeurs numériques valides.")
            return None
        
    def set_bounds_to_inputs(self):
        """Écrit les valeurs par défaut dans les QLineEdit."""
        self.lineEdit_xmin.setText(str(self.default_xmin))
        self.lineEdit_xmax.setText(str(self.default_xmax))
        self.lineEdit_ymin.setText(str(self.default_ymin))
        self.lineEdit_ymax.setText(str(self.default_ymax))
        self.lineEdit_legx.setText(str(self.default_xleg))
        self.lineEdit_legy.setText(str(self.default_yleg))

    def update_bounds(self):
        """Met à jour les bornes de la scène avec les nouvelles valeurs."""
        x_min, x_max, y_min, y_max, x_legend, y_legend = self.get_bounds_from_inputs()
        self.scene.update_bounds(x_min, x_max, y_min, y_max, x_legend, y_legend)

    def setup_graphics_view(self, x_min_label, x_max_label, y_min_label, y_max_label, default_xleg, default_yleg, grid_size):
        """Configure la scène et la vue graphique avec une grille."""
        size = self.graphicsView.viewport().size()
        self.scene = GraphicsScene(grid_size=grid_size, status_mathsElement = True, x_min=x_min_label, x_max=x_max_label, y_min=y_min_label, y_max=y_max_label, x_legend = default_xleg, y_legend = default_yleg)  # Taille de la grille
        self.scene.setSceneRect(0, 0, size.width(), size.height())  # Définir la taille de la scène

        self.graphicsView.setScene(self.scene)  # Assigner la scène au QGraphicsView

        # Options d'affichage
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
    
    def create_tags(self):
        """
        Crée dynamiquement des tags dans la scroll area en utilisant le layout existant.
        """
        # Liste pour stocker les checkboxes

        # Créez un widget pour le contenu de la Scroll Area
        content_widget = QWidget()
        
        # Créez un layout pour ce widget
        content_layout = QVBoxLayout(content_widget)
        
        # Créer les tags
        for i in range(44,50):
            # Créer un QHBoxLayout pour chaque élément
            horizontal_layout = QHBoxLayout()
            horizontal_layout.setObjectName(f"horizontalLayout_Tag{i}")
            
            # Créer une case à cocher pour chaque tag
            checkBox = QCheckBox(f"TAG {i}", content_widget)  # Correction ici
            checkBox.setObjectName(f"checkBox_Tag{i}")
            checkBox.setChecked(True)  # Définir la case à cocher comme cochée par défaut
            horizontal_layout.addWidget(checkBox)
            self.checkboxes.append(checkBox)  # Ajouter à la liste

            # Labels pour les positions (exemples ici)
            label_posx = QLabel(f"pos x :", content_widget)
            label_posx.setObjectName(f"label_posx_{i}")
            label_posx.setMinimumSize(50, 70)
            label_posx.setMaximumSize(70, 70)
            label_posx.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            horizontal_layout.addWidget(label_posx)

            label_posx_nbr = QLabel(f"?", content_widget)
            label_posx_nbr.setObjectName(f"label_posx_nbr_{i}")
            label_posx_nbr.setMinimumSize(50, 70)
            label_posx_nbr.setMaximumSize(70, 70)
            horizontal_layout.addWidget(label_posx_nbr)

            label_posy = QLabel(f"pos y :", content_widget)
            label_posy.setObjectName(f"label_posy_{i}")
            label_posy.setMinimumSize(50, 70)
            label_posy.setMaximumSize(70, 70)
            label_posy.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            horizontal_layout.addWidget(label_posy)

            label_posy_nbr = QLabel(f"?", content_widget)
            label_posy_nbr.setObjectName(f"label_posy_nbr_{i}")
            label_posy_nbr.setMinimumSize(50, 70)
            label_posy_nbr.setMaximumSize(70, 70)
            horizontal_layout.addWidget(label_posy_nbr)
            
            # Ajoutez ce QHBoxLayout à la mise en page principale
            content_layout.addLayout(horizontal_layout)

            # Ajouter une ligne séparatrice
            separator = QFrame(content_widget)
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            content_layout.addWidget(separator)

        # Définir le widget contenu dans la scroll area
        self.scrollArea.setWidget(content_widget)
        self.scrollArea.setWidgetResizable(True)  # Permet au contenu de se redimensionner si nécessaire

    def on_display_grid_checkbox_changed(self, state):
        if self.algorithm_analysis:
            self.algorithm_analysis.set_show_grid(self.checkBox_DisplayGrid.isChecked())
        else:
            # Si pas d'acquisition, on utilise le comportement statique existant
            self.refresh_projector_background()

    def connect_checkbox_to_algorithm(self):
        if self.algorithm_analysis:
            self.checkBox_Visu_Cam.stateChanged.connect(self.algorithm_analysis.state_popUpCamera_changed)