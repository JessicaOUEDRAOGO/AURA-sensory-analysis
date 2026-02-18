from src.core.utils.paths import gui_path
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtGui import QPainter, QPixmap, QColor
from PyQt6 import uic, QtWidgets
import json

from src.ui.controllers.key_handler import KeyHandler
from src.ui.widgets.graphics_scene import GraphicsScene
from src.ui.widgets.layer import CircleLayer, LineLayer, TextLayer

class RealityAugementedWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()

        uic.loadUi(gui_path("RealityAugemented_Menu.ui"), self)
        # Labels à gauche (Rond / Trait / Text)
        self.label_Circle.setProperty("leftLabel", "true")
        self.label_trait.setProperty("leftLabel", "true")
        self.label_text.setProperty("leftLabel", "true")

        # Important : forcer le refresh du style
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

        # --- Force la lisibilité des 3 labels à gauche (Rond / Trait / Text) ---
        for lbl in (self.label_Circle, self.label_trait, self.label_text):
            lbl.setStyleSheet("QLabel { color: rgba(20,20,20,230); font-weight: 700; }")

        # Si ton fond est sombre, utilise plutôt :
        # for lbl in (self.label_Circle, self.label_trait, self.label_text):
        #     lbl.setStyleSheet("QLabel { color: rgba(255,255,255,240); font-weight: 700; }")

        # Rendre les labels de gauche lisibles
        self.label_Circle.setObjectName("leftLabelCircle")
        self.label_trait.setObjectName("leftLabelTrait")
        self.label_text.setObjectName("leftLabelText")

        # --- rendre lisibles Rond / Trait / Text ---
        self.label_Circle.setProperty("leftLabel", True)
        self.label_trait.setProperty("leftLabel", True)
        self.label_text.setProperty("leftLabel", True)

        # force Qt à ré-appliquer le style
        for w in (self.label_Circle, self.label_trait, self.label_text):
            w.style().unpolish(w)
            w.style().polish(w)

        self.parent = parent  # Référence à MainApp

        self.grid_xmin = -251
        self.grid_xmax = 349 
        self.grid_ymin = -286
        self.grid_ymax = 314

        self.y_TagPos = None
        self.x_TagPos = None
        self.scale_factor = 0.6
        self.marker_size_graphic = 300 * self.scale_factor

        self.scene = GraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.layer_counter = 0  # Unique ID for layers

        # Dictionnaire pour stocker les couches par ArUco Marker
        self.layers_by_marker = {}
        self.previous_marker = '0'  # Variable pour mémoriser le marqueur précédent
      
        # Initialize combo boxes
        self.comboBox_ChooseDict.addItems([
            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
            "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
            "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"
        ])

        self.hide_layout_widgets(self.verticalLayout_settings)
        self.grid_size = 600
        self.setup_graphics_view(self.grid_size)
        self.update_tag_combo_box("DICT_4X4_50")
        self.comboBox_ChooseDict.currentTextChanged.connect(self.update_tag_combo_box)
        self.comboBox_ChooseTag.currentTextChanged.connect(self.change_marker)

        # Connect buttons
        self.pushButton_AddCircle.clicked.connect(self.add_circle_layer)
        self.pushButton_AddTrait.clicked.connect(self.add_line_layer)
        self.pushButton_AddText.clicked.connect(self.add_text_layer)
        self.pushButton_ToutSuppr.clicked.connect(self.remove_all_layers)
        self.pushButton_SaveRAParams.clicked.connect(self.save_ra_params)  # Connecter le bouton

        self.pushButton_LoadRAconfig.clicked.connect(self.load_ra_params)

        self.pushButton_return.clicked.connect(self.go_to_main)

        # Ensure the scroll area has a proper child widget and layout
        self.scrollArea_Calques.setWidget(QtWidgets.QWidget())  # Set a child widget
        self.scrollArea_Calques.widget().setLayout(QtWidgets.QVBoxLayout())  # Assign a layout to the child widget

        self.horizontalSlider_red.setMaximum(255)
        self.horizontalSlider_green.setMaximum(255)
        self.horizontalSlider_blue.setMaximum(255)

        self.key_handler = KeyHandler(self)

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)

    def change_marker(self):
        """
        Change le marqueur actuel et charge les couches associées.
        """
        # Sauvegarder les couches actuelles
        current_marker = self.comboBox_ChooseTag.currentText()
        self.save_current_layers(self.previous_marker)

        self.previous_marker = current_marker

        # Effacer la scène et l'interface utilisateur
        self.remove_all_layers()

        # Mettre à jour l'image de fond pour le nouveau marqueur
        self.update_tag_displayed()

        # Charger les couches pour le nouveau marqueur
        new_marker = self.comboBox_ChooseTag.currentText()
        self.load_layers_for_marker(new_marker)

    def save_current_layers(self, marker_id):
        """
        Sauvegarde les couches actuelles dans le dictionnaire pour l'ID de marqueur donné
        et supprime les widgets associés de l'interface utilisateur.
        """
        # Sauvegarder les calques actuels
        self.layers_by_marker[marker_id] = list(self.scene.layers.values())

    def load_layers_for_marker(self, marker_id):
        """
        Charge les couches pour l'ID de marqueur donné depuis le dictionnaire.
        """
        if marker_id in self.layers_by_marker:
            for layer in self.layers_by_marker[marker_id]:
                self.scene.add_layer(layer)
                self.add_layer_to_ui(layer)

    def add_circle_layer(self):
        self.layer_counter += 1
        layer = CircleLayer(self.layer_counter, x=100, y=100, radius=50, color="red")
        self.scene.add_layer(layer)
        self.add_layer_to_ui(layer)

    def add_line_layer(self):
        self.layer_counter += 1
        layer = LineLayer(self.layer_counter, x1=250, y1=250, x2=50, y2=50, color="blue", thickness=2)
        self.scene.add_layer(layer)
        self.add_layer_to_ui(layer)

    def add_text_layer(self):
        self.layer_counter += 1
        layer = TextLayer(self.layer_counter, x=200, y=200, text="Hello", color="green", font_size=12)
        self.scene.add_layer(layer)
        self.add_layer_to_ui(layer)
    
    def add_layer_to_ui(self, layer):
        """
        Ajoute un widget correspondant au calque dans l'interface utilisateur.
        """
        # Get the widget inside the scroll area
        scroll_area_widget = self.scrollArea_Calques.widget()

        # Ensure the widget has a layout
        if scroll_area_widget.layout() is None:
            scroll_area_widget.setLayout(QtWidgets.QVBoxLayout())

        # Get the layout
        layout = scroll_area_widget.layout()

        # Create a new widget for the layer
        layer_widget = QtWidgets.QWidget()
        layer_layout = QtWidgets.QHBoxLayout(layer_widget)
        layer_label = QtWidgets.QLabel(f"{layer.layer_type.capitalize()} {layer.layer_id}")
        delete_button = QtWidgets.QPushButton("Supprimer")
        select_button = QtWidgets.QPushButton("Sélectionner")

        # Connect the delete button
        delete_button.clicked.connect(lambda: self.remove_layer(layer.layer_id))

        # Connect the select button
        select_button.clicked.connect(lambda: self.display_layer_settings(layer))
        self.display_layer_settings(layer)

        # Add widgets to the layer layout
        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(select_button)
        layer_layout.addWidget(delete_button)

        # Add the layer widget to the layout
        layout.addWidget(layer_widget)

    def display_layer_settings(self, layer):
        """
        Display the settings for the selected layer.
        Clears the current settings and shows the appropriate settings layout.
        """
        # Clear the current settings
        self.hide_layout_widgets(self.verticalLayout_settings)

        if layer.layer_type in ["circle", "line", "text"]:
            self.show_layout_widgets(self.horizontalLayout_Couleur)

            # Synchroniser les sliders et les QLineEdit avec la couleur du layer
            color = QColor(layer.color)
            self.horizontalSlider_red.blockSignals(True)
            self.horizontalSlider_green.blockSignals(True)
            self.horizontalSlider_blue.blockSignals(True)
            self.lineEdit_Rvalue.blockSignals(True)
            self.lineEdit_Gvalue.blockSignals(True)
            self.lineEdit_Bvalue.blockSignals(True)

            self.horizontalSlider_red.setValue(color.red())
            self.horizontalSlider_green.setValue(color.green())
            self.horizontalSlider_blue.setValue(color.blue())

            self.lineEdit_Rvalue.setText(str(color.red()))
            self.lineEdit_Gvalue.setText(str(color.green()))
            self.lineEdit_Bvalue.setText(str(color.blue()))

            self.horizontalSlider_red.blockSignals(False)
            self.horizontalSlider_green.blockSignals(False)
            self.horizontalSlider_blue.blockSignals(False)
            self.lineEdit_Rvalue.blockSignals(False)
            self.lineEdit_Gvalue.blockSignals(False)
            self.lineEdit_Bvalue.blockSignals(False)

            # Mettre à jour l'aperçu de la couleur
            self.update_color_preview(color)

            # Déconnecter les signaux existants
            try:
                self.horizontalSlider_red.valueChanged.disconnect()
                self.horizontalSlider_green.valueChanged.disconnect()
                self.horizontalSlider_blue.valueChanged.disconnect()
                self.lineEdit_Rvalue.editingFinished.disconnect()
                self.lineEdit_Gvalue.editingFinished.disconnect()
                self.lineEdit_Bvalue.editingFinished.disconnect()
            except TypeError:
                pass

            # Reconnecter les signaux pour les sliders
            self.horizontalSlider_red.valueChanged.connect(lambda value: self.update_color_from_slider(layer, "red", value))
            self.horizontalSlider_green.valueChanged.connect(lambda value: self.update_color_from_slider(layer, "green", value))
            self.horizontalSlider_blue.valueChanged.connect(lambda value: self.update_color_from_slider(layer, "blue", value))

            # Reconnecter les signaux pour les QLineEdit
            self.lineEdit_Rvalue.editingFinished.connect(lambda: self.update_color_from_line_edit(layer, "red"))
            self.lineEdit_Gvalue.editingFinished.connect(lambda: self.update_color_from_line_edit(layer, "green"))
            self.lineEdit_Bvalue.editingFinished.connect(lambda: self.update_color_from_line_edit(layer, "blue"))

        if layer.layer_type == "text":
            self.show_layout_widgets(self.horizontalLayout_Rotation)
            self.line_rotation.show()

            # Set the current rotation value in the QLineEdit
            self.lineEdit_rotation.setText(str(layer.rotation))

            # Connect signals for rotation
            try:
                self.lineEdit_rotation.editingFinished.disconnect()
            except TypeError:
                pass  # Signal was not connected
            self.lineEdit_rotation.editingFinished.connect(lambda: self.update_layer_rotation(layer))

            try:
                self.pushButton_Angle_P.clicked.disconnect()
            except TypeError:
                pass  # Signal was not connected
            self.pushButton_Angle_P.clicked.connect(lambda: self.increment_rotation(layer, 1))

            try:
                self.pushButton_Angle_M.clicked.disconnect()
            except TypeError:
                pass  # Signal was not connected
            self.pushButton_Angle_M.clicked.connect(lambda: self.increment_rotation(layer, -1))

            self.show_layout_widgets(self.horizontalLayout_setTEXT)

            # Définir les valeurs actuelles dans les QLineEdit
            self.lineEdit.setText(layer.text)  # Texte principal
            self.lineEdit_2.setText(str(layer.font_size))  # Taille de la police

            # Connecter les signaux pour le texte
            try:
                self.lineEdit.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit.editingFinished.connect(lambda: self.update_layer_text(layer))

            try:
                self.lineEdit_2.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_2.editingFinished.connect(lambda: self.update_layer_font_size(layer))

            # Connecter les boutons pour augmenter/diminuer la taille de la police
            try:
                self.pushButton.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton.clicked.connect(lambda: self.increment_font_size(layer, 1))

            try:
                self.pushButton_2.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_2.clicked.connect(lambda: self.increment_font_size(layer, -1))

            # Déconnecter les signaux des cases à cocher
            try:
                self.checkBox_Gras.stateChanged.disconnect()
            except TypeError:
                pass
            try:
                self.checkBox_Italique.stateChanged.disconnect()
            except TypeError:
                pass
            try:
                self.checkBox_Souligner.stateChanged.disconnect()
            except TypeError:
                pass

            # Mettre à jour l'état des cases à cocher en fonction des propriétés du layer
            self.checkBox_Gras.setChecked(layer.bold)
            self.checkBox_Italique.setChecked(layer.italic)
            self.checkBox_Souligner.setChecked(layer.underline)

            # Reconnecter les signaux des cases à cocher
            self.checkBox_Gras.stateChanged.connect(lambda state: self.toggle_text_style(layer, "bold", state))
            self.checkBox_Italique.stateChanged.connect(lambda state: self.toggle_text_style(layer, "italic", state))
            self.checkBox_Souligner.stateChanged.connect(lambda state: self.toggle_text_style(layer, "underline", state))

        if layer.layer_type == "circle":
            self.show_layout_widgets(self.horizontalLayout_Rayon)
            self.line_Rayon.show()
            self.lineEdit_Rayon.setText(str(layer.radius))

            # Mettre à jour l'état de la case à cocher en fonction de la propriété du layer
            self.checkBox_remplir.blockSignals(True)
            self.checkBox_remplir.setChecked(layer.fill)
            self.checkBox_remplir.blockSignals(False)

            # Connecter la case à cocher pour remplir ou non le cercle
            try:
                self.checkBox_remplir.stateChanged.disconnect()
            except TypeError:
                pass
            self.checkBox_remplir.stateChanged.connect(lambda state: self.toggle_circle_fill(layer, state))

            # Connecter les signaux pour le rayon
            try:
                self.lineEdit_Rayon.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_Rayon.editingFinished.connect(lambda: self.update_layer_radius(layer))
            try:
                self.pushButton_Rayon_P.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_Rayon_P.clicked.connect(lambda: self.increment_radius(layer, 1))
            try:
                self.pushButton_Rayon_M.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_Rayon_M.clicked.connect(lambda: self.increment_radius(layer, -1))

        if layer.layer_type == "line":
            self.show_layout_widgets(self.verticalLayout__MultiPts)
            self.line_MultiPos.show()

            # Synchroniser les valeurs des QLineEdit avec les coordonnées des points A et B
            self.lineEdit_ptA_X.setText(str(layer.x1))
            self.lineEdit_ptA_Y.setText(str(layer.y1))
            self.lineEdit_ptB_X.setText(str(layer.x2))
            self.lineEdit_ptB_Y.setText(str(layer.y2))

            # Mettre à jour l'état de la case à cocher pour lier les points
            self.checkBox_linkPoints.blockSignals(True)
            self.checkBox_linkPoints.setChecked(False)  # Par défaut, décoché
            self.checkBox_linkPoints.blockSignals(False)

            # Connecter les signaux pour les points A et B
            try:
                self.lineEdit_ptA_X.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_ptA_X.editingFinished.connect(lambda: self.update_line_point(layer, "A", "x"))

            try:
                self.lineEdit_ptA_Y.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_ptA_Y.editingFinished.connect(lambda: self.update_line_point(layer, "A", "y"))

            try:
                self.lineEdit_ptB_X.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_ptB_X.editingFinished.connect(lambda: self.update_line_point(layer, "B", "x"))

            try:
                self.lineEdit_ptB_Y.editingFinished.disconnect()
            except TypeError:
                pass
            self.lineEdit_ptB_Y.editingFinished.connect(lambda: self.update_line_point(layer, "B", "y"))

            # Connecter les boutons pour incrémenter/décrémenter les coordonnées des points A et B
            try:
                self.pushButton_A_PosX_P.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_A_PosX_P.clicked.connect(lambda: self.increment_line_point(layer, "A", "x", 1))

            try:
                self.pushButton_A_PosX_M.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_A_PosX_M.clicked.connect(lambda: self.increment_line_point(layer, "A", "x", -1))

            try:
                self.pushButton_A_PosY_P.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_A_PosY_P.clicked.connect(lambda: self.increment_line_point(layer, "A", "y", 1))

            try:
                self.pushButton_A_PosY_M.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_A_PosY_M.clicked.connect(lambda: self.increment_line_point(layer, "A", "y", -1))

            try:
                self.pushButton_B_PosX_P.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_B_PosX_P.clicked.connect(lambda: self.increment_line_point(layer, "B", "x", 1))

            try:
                self.pushButton_B_PosX_M.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_B_PosX_M.clicked.connect(lambda: self.increment_line_point(layer, "B", "x", -1))

            try:
                self.pushButton_B_PosY_P.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_B_PosY_P.clicked.connect(lambda: self.increment_line_point(layer, "B", "y", 1))

            try:
                self.pushButton_B_PosY_M.clicked.disconnect()
            except TypeError:
                pass
            self.pushButton_B_PosY_M.clicked.connect(lambda: self.increment_line_point(layer, "B", "y", -1))

        # Show the settings based on the layer type
        if layer.layer_type in ["circle", "text"]:
            self.show_layout_widgets(self.horizontalLayout_Position)
            self.line_Pos.show()
            self.lineEdit_PosX.setText(str(layer.x))
            self.lineEdit_PosY.setText(str(layer.y))

            # Connecter les signaux pour la position
            
            try:
                self.lineEdit_PosX.editingFinished.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.lineEdit_PosX.editingFinished.connect(lambda: self.update_layer_position(layer, "x"))

            try:
                self.pushButton_PosX_P.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_PosX_P.clicked.connect(lambda: self.increment_position(layer, "x", 1))

            try:
                self.pushButton_PosY_P.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_PosY_P.clicked.connect(lambda: self.increment_position(layer, "y", 1))
            try:
                self.lineEdit_PosY.editingFinished.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.lineEdit_PosY.editingFinished.connect(lambda: self.update_layer_position(layer, "y"))
            try:
                self.pushButton_PosX_M.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_PosX_M.clicked.connect(lambda: self.increment_position(layer, "x", -1))
            try:
                self.pushButton_PosY_M.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_PosY_M.clicked.connect(lambda: self.increment_position(layer, "y", -1))

        if layer.layer_type in ["circle", "line"]:
            self.show_layout_widgets(self.horizontalLayout_Epaisseur)
            self.line_epaisseur.show()
            self.lineEdit_Epaisseur.setText(str(layer.thickness))

            # Connecter les signaux pour l'épaisseur
            try:
                self.lineEdit_Epaisseur.editingFinished.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.lineEdit_Epaisseur.editingFinished.connect(lambda: self.update_layer_thickness(layer))
            try:
                self.pushButton_epaisseurP.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_epaisseurP.clicked.connect(lambda: self.increment_thickness(layer, 1))
            try:
                self.pushButton_epaisseurM.clicked.disconnect()
            except TypeError:
                pass  # Le signal n'était pas connecté
            self.pushButton_epaisseurM.clicked.connect(lambda: self.increment_thickness(layer, -1))

    def remove_layer(self, layer_id):

        """
        Supprime un calque spécifique de la scène et de l'interface utilisateur.
        """
        # Supprimer le calque de la scène
        if layer_id in self.scene.layers:
            layer = self.scene.layers.pop(layer_id)
            if layer.graphics_item:  # Vérifie si l'objet graphique existe
                self.scene.removeItem(layer.graphics_item)  # Supprime l'objet graphique de la scène

        # Supprimer le widget correspondant dans l'interface utilisateur
        layout = self.scrollArea_Calques.widget().layout()
        for i in range(layout.count()):
            item = layout.itemAt(i).widget()
            if item and f"{layer_id}" in item.findChild(QtWidgets.QLabel).text():
                layout.removeWidget(item)
                item.deleteLater()
                break

    def remove_all_layers(self):
        """
        Supprime tous les calques de la scène et de l'interface utilisateur.
        """
        # Supprimer tous les calques de la scène
        for layer_id in list(self.scene.layers.keys()):
            self.scene.remove_layer(layer_id)

        # Supprimer tous les widgets de la liste des calques dans l'interface utilisateur
        layout = self.scrollArea_Calques.widget().layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_tag_combo_box(self, selected_dict):
        """
        Updates the self.comboBox_ChooseTag with the IDs corresponding to the selected dictionary.
        """
        # Dictionary mapping to the number of tags
        aruco_dicts = {
            "DICT_4X4_50": 50, "DICT_4X4_100": 100, "DICT_4X4_250": 250, "DICT_4X4_1000": 1000,
            "DICT_5X5_50": 50, "DICT_5X5_100": 100, "DICT_5X5_250": 250, "DICT_5X5_1000": 1000,
            "DICT_6X6_50": 50, "DICT_6X6_100": 100, "DICT_6X6_250": 250, "DICT_6X6_1000": 1000,
            "DICT_7X7_50": 50, "DICT_7X7_100": 100, "DICT_7X7_250": 250, "DICT_7X7_1000": 1000,
            "DICT_ARUCO_ORIGINAL": 1024, "DICT_APRILTAG_16h5": 30, "DICT_APRILTAG_25h9": 35,
            "DICT_APRILTAG_36h10": 2320, "DICT_APRILTAG_36h11": 587
        }

        # Clear the tag combo box
        self.comboBox_ChooseTag.clear()
        # Populate with IDs based on the selected dictionary
        if selected_dict in aruco_dicts:
            num_tags = aruco_dicts[selected_dict]
            self.comboBox_ChooseTag.addItems([str(i) for i in range(num_tags)])
        self.update_tag_displayed()

    def update_tag_displayed(self):
        """
        Met à jour l'image de fond affichée en fonction du dictionnaire et de l'ID de marqueur sélectionnés.
        """
        selected_dict = self.comboBox_ChooseDict.currentText()
        selected_tag = self.comboBox_ChooseTag.currentText()

        # Construire le chemin de l'image pour le marqueur sélectionné
        image_path = f"./assets/aruco_tags/{selected_dict}/{selected_tag}.png"

        # Mettre à jour l'image de fond dans la scène
        if selected_tag != "":
            self.scene.display_image(image_path, x=self.y_TagPos, y=self.x_TagPos, scale_factor=self.scale_factor)

    def setup_graphics_view(self, grid_size):
        """Configure la scène et la vue graphique avec une grille."""
        size = self.graphicsView.viewport().size()
        self.scene = GraphicsScene(grid_size=grid_size, status_mathsElement = False, grid_xmin = self.grid_xmin, grid_xmax = self.grid_xmax, grid_ymin = self.grid_ymin, grid_ymax = self.grid_ymax)  # Taille de la grille
        self.scene.setSceneRect(0, 0, size.width(), size.height())  # Définir la taille de la scène
        selected_dict = self.comboBox_ChooseDict.currentText()
        self.scene.display_image("./assets/aruco_tags/DICT_4x4_50/0.png", x=self.y_TagPos, y=self.x_TagPos, scale_factor=self.scale_factor)

        self.graphicsView.setScene(self.scene)  # Assigner la scène au QGraphicsView
        # Options d'affichage
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)

    def hide_layout_widgets(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item, QtWidgets.QLayout):  # If the item is a layout, recursively hide its widgets
                self.hide_layout_widgets(item)
            else:
                widget = item.widget()
                if widget:
                    widget.hide()

    def show_layout_widgets(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item, QtWidgets.QLayout):  # If the item is a layout, recursively show its widgets
                self.show_layout_widgets(item)
            else:
                widget = item.widget()
                if widget:
                    widget.show()

    def update_layer_position(self, layer, axis):
        """
        Met à jour la position du layer en fonction des valeurs des QLineEdit.
        :param layer: Le layer sélectionné.
        :param axis: 'x' ou 'y' pour indiquer l'axe à mettre à jour.
        """
        try:
            if axis == "x":
                new_x = float(self.lineEdit_PosX.text())
                layer.update_properties(x=new_x)
            elif axis == "y":
                new_y = float(self.lineEdit_PosY.text())
                layer.update_properties(y=new_y)
            self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_position(self, layer, axis, increment):
        """
        Incrémente ou décrémente la position du layer en fonction des boutons + et -.
        :param layer: Le layer sélectionné.
        :param axis: 'x' ou 'y' pour indiquer l'axe à modifier.
        :param increment: La valeur d'incrémentation (positive ou négative).
        """
        multiplier = 10 if (axis == "x" and self.checkBox_PosX_x10.isChecked()) or \
                            (axis == "y" and self.checkBox_PosY_x10.isChecked()) else 1
        increment *= multiplier

        if axis == "x":
            new_x = layer.x + increment
            layer.update_properties(x=new_x)
            self.lineEdit_PosX.setText(str(new_x))
        elif axis == "y":
            new_y = layer.y + increment
            layer.update_properties(y=new_y)
            self.lineEdit_PosY.setText(str(new_y))
        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

    def update_layer_radius(self, layer):
        """
        Met à jour le rayon du layer en fonction de la valeur de QLineEdit.
        :param layer: Le layer sélectionné.
        """
        try:
            new_radius = float(self.lineEdit_Rayon.text())
            if new_radius > 0:  # Vérifie que le rayon est positif
                layer.update_properties(radius=new_radius)
                self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", "Le rayon doit être supérieur à 0.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_radius(self, layer, increment):
        """
        Incrémente ou décrémente le rayon du layer en fonction des boutons + et -.
        :param layer: Le layer sélectionné.
        :param increment: La valeur d'incrémentation (positive ou négative).
        """
        multiplier = 10 if self.checkBox_Rayon_x10.isChecked() else 1
        increment *= multiplier

        new_radius = layer.radius + increment
        if new_radius > 0:  # Vérifie que le rayon reste positif
            layer.update_properties(radius=new_radius)
            self.lineEdit_Rayon.setText(str(new_radius))
            self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Le rayon doit être supérieur à 0.")

    def update_layer_thickness(self, layer):
        """
        Met à jour l'épaisseur du layer en fonction de la valeur de QLineEdit.
        :param layer: Le layer sélectionné.
        """
        try:
            new_thickness = float(self.lineEdit_Epaisseur.text())
            if new_thickness > 0:  # Vérifie que l'épaisseur est positive
                layer.update_properties(thickness=new_thickness)
                self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", "L'épaisseur doit être supérieure à 0.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_thickness(self, layer, increment):
        """
        Incrémente ou décrémente l'épaisseur du layer en fonction des boutons + et -.
        :param layer: Le layer sélectionné.
        :param increment: La valeur d'incrémentation (positive ou négative).
        """
        # Vérifier si la case à cocher x10 est activée
        multiplier = 10 if self.checkBox_Epaisseur_x10.isChecked() else 1
        increment *= multiplier

        try:
            new_thickness = layer.thickness + increment
            if new_thickness > 0:  # Vérifie que l'épaisseur reste positive
                layer.update_properties(thickness=new_thickness)
                self.lineEdit_Epaisseur.setText(str(new_thickness))
                self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", "L'épaisseur doit être supérieure à 0.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def update_line_point(self, layer, point, axis):
        """
        Met à jour les coordonnées d'un point (A ou B) du segment en fonction des valeurs des QLineEdit.
        :param layer: Le layer sélectionné.
        :param point: 'A' ou 'B' pour indiquer le point à modifier.
        :param axis: 'x' ou 'y' pour indiquer l'axe à modifier.
        """
        try:
            if point == "A":
                if axis == "x":
                    new_x = float(self.lineEdit_ptA_X.text())
                    layer.update_properties(x1=new_x)
                elif axis == "y":
                    new_y = float(self.lineEdit_ptA_Y.text())
                    layer.update_properties(y1=new_y)
            elif point == "B":
                if axis == "x":
                    new_x = float(self.lineEdit_ptB_X.text())
                    layer.update_properties(x2=new_x)
                elif axis == "y":
                    new_y = float(self.lineEdit_ptB_Y.text())
                    layer.update_properties(y2=new_y)
            self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_line_point(self, layer, point, axis, increment):
        """
        Incrémente ou décrémente les coordonnées d'un point (A ou B) du segment.
        :param layer: Le layer sélectionné.
        :param point: 'A' ou 'B' pour indiquer le point à modifier.
        :param axis: 'x' ou 'y' pour indiquer l'axe à modifier.
        :param increment: La valeur d'incrémentation (positive ou négative).
        """
        multiplier = 10 if self.checkBox_PosMltPts_x10.isChecked() else 1
        increment *= multiplier
        if self.checkBox_linkPoints.isChecked():
            if axis == "x":
                increment_x = increment
                layer.update_properties(x1=layer.x1 + increment_x, x2=layer.x2 + increment_x)
                self.lineEdit_ptA_X.setText(str(layer.x1))
                self.lineEdit_ptB_X.setText(str(layer.x2))
            elif axis == "y":
                increment_y = increment
                layer.update_properties(y1=layer.y1 + increment_y, y2=layer.y2 + increment_y)
                self.lineEdit_ptA_Y.setText(str(layer.y1))
                self.lineEdit_ptB_Y.setText(str(layer.y2))
        else:
            if point == "A":
                if axis == "x":
                    new_x = layer.x1 + increment
                    layer.update_properties(x1=new_x)
                    self.lineEdit_ptA_X.setText(str(new_x))
                elif axis == "y":
                    new_y = layer.y1 + increment
                    layer.update_properties(y1=new_y)
                    self.lineEdit_ptA_Y.setText(str(new_y))
            elif point == "B":
                if axis == "x":
                    new_x = layer.x2 + increment
                    layer.update_properties(x2=new_x)
                    self.lineEdit_ptB_X.setText(str(new_x))
                elif axis == "y":
                    new_y = layer.y2 + increment
                    layer.update_properties(y2=new_y)
                    self.lineEdit_ptB_Y.setText(str(new_y))

        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

    def update_layer_rotation(self, layer):
        """
        Updates the rotation of the layer based on the value in QLineEdit.
        :param layer: The selected layer.
        """
        try:
            new_rotation = float(self.lineEdit_rotation.text())
            layer.update_properties(rotation=new_rotation)
            self.scene.update_layer(layer.layer_id)  # Redraw the layer in the scene
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_rotation(self, layer, increment):
        """
        Increments or decrements the rotation of the layer based on the + and - buttons.
        :param layer: The selected layer.
        :param increment: The increment value (positive or negative).
        """
        multiplier = 10 if self.checkBox_Rotation_x10.isChecked() else 1
        increment *= multiplier

        new_rotation = layer.rotation + increment
        layer.update_properties(rotation=new_rotation)
        self.lineEdit_rotation.setText(str(new_rotation))
        self.scene.update_layer(layer.layer_id)  # Redraw the layer in the scene

    def update_layer_text(self, layer):
        """
        Met à jour le texte du layer en fonction de la valeur dans QLineEdit.
        """
        new_text = self.lineEdit.text()
        layer.update_properties(text=new_text)
        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

    def update_layer_font_size(self, layer):
        """
        Met à jour la taille de la police du layer en fonction de la valeur dans QLineEdit.

        """
        try:
            new_font_size = int(self.lineEdit_2.text())
            if new_font_size > 0:
                layer.update_properties(font_size=new_font_size)
                self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", "La taille de la police doit être supérieure à 0.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur numérique valide.")

    def increment_font_size(self, layer, increment):
        """
        Incrémente ou décrémente la taille de la police du layer.
        """
        new_font_size = layer.font_size + increment
        if new_font_size > 0:
            layer.update_properties(font_size=new_font_size)
            self.lineEdit_2.setText(str(new_font_size))
            self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "La taille de la police doit être supérieure à 0.")

    def toggle_text_style(self, layer, style, state):
        """
        Active ou désactive un style de texte (gras, italique, souligné).
        """
        if style == "bold":
            layer.update_properties(bold=bool(state))
        elif style == "italic":
            layer.update_properties(italic=bool(state))
        elif style == "underline":
            layer.update_properties(underline=bool(state))
        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

    def update_color_from_slider(self, layer, channel, value):
        """
        Met à jour la couleur du layer en fonction des sliders.
        :param layer: Le layer sélectionné.
        :param channel: Le canal de couleur ("red", "green", "blue").
        :param value: La valeur du slider.
        """
        color = QColor(layer.color)
        if channel == "red":
            color.setRed(value)
            self.lineEdit_Rvalue.blockSignals(True)
            self.lineEdit_Rvalue.setText(str(value))
            self.lineEdit_Rvalue.blockSignals(False)
        elif channel == "green":
            color.setGreen(value)
            self.lineEdit_Gvalue.blockSignals(True)
            self.lineEdit_Gvalue.setText(str(value))
            self.lineEdit_Gvalue.blockSignals(False)
        elif channel == "blue":
            color.setBlue(value)
            self.lineEdit_Bvalue.blockSignals(True)
            self.lineEdit_Bvalue.setText(str(value))
            self.lineEdit_Bvalue.blockSignals(False)

        # Mettre à jour la couleur du layer
        layer.update_properties(color=color.name())
        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

        # Mettre à jour l'aperçu de la couleur
        self.update_color_preview(color)

    def update_color_from_line_edit(self, layer, channel):
        """
        Met à jour la couleur du layer en fonction des QLineEdit.
        :param layer: Le layer sélectionné.
        :param channel: Le canal de couleur ("red", "green", "blue").
        """
        try:
            value = int(getattr(self, f"lineEdit_{channel[0].upper()}value").text())
            if 0 <= value <= 255:
                slider = getattr(self, f"horizontalSlider_{channel}")
                slider.blockSignals(True)
                slider.setValue(value)  # Met à jour le slider correspondant
                slider.blockSignals(False)

                # Mettre à jour la couleur du layer
                color = QColor(layer.color)
                if channel == "red":
                    color.setRed(value)
                elif channel == "green":
                    color.setGreen(value)
                elif channel == "blue":
                    color.setBlue(value)

                layer.update_properties(color=color.name())
                self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

                # Mettre à jour l'aperçu de la couleur
                self.update_color_preview(color)
            else:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer une valeur entre 0 et 255.")

    def update_color_preview(self, color):
        """
        Met à jour l'aperçu de la couleur dans le label `label_ImgCouleurChange`.
        :param color: La couleur à afficher (QColor).
        """
        pixmap = QPixmap(100, 100)
        pixmap.fill(color)
        self.label_ImgCouleurChange.setPixmap(pixmap)

    def toggle_circle_fill(self, layer, state):
        """
        Active ou désactive le remplissage du cercle en fonction de l'état de la case à cocher.
        :param layer: Le layer sélectionné.
        :param state: L'état de la case à cocher (0 ou 2).
        """
        layer.update_properties(fill=bool(state))
        self.scene.update_layer(layer.layer_id)  # Redessiner le layer dans la scène

    def save_ra_params(self):
        """
        Sauvegarde les paramètres de réalité augmentée dans un fichier JSON.
        """
        self.save_current_layers(self.comboBox_ChooseTag.currentText())

        config = {}

        # Taille du marqueur dans la zone graphique (300 pixels)
        marker_size_graphic = self.marker_size_graphic

        center_x = 49
        center_y = 14

        # Taille de l'écran projeté (2160 pixels)
        projector_size = 1080

        # Facteur d'échelle pour ajuster les tailles relatives
        scale_factor = projector_size / self.grid_size

        for marker_id, layers in self.layers_by_marker.items():
            marker_config = []
            for layer in layers:
                color_hex = QColor(layer.color).name()  # Convert named color to hex
                if layer.layer_type == "circle":
                    marker_config.append({
                        "type": "circle",
                        "relative_position": {
                            "x":  (center_x - layer.x)/marker_size_graphic,
                            "y":  (center_y - layer.y)/marker_size_graphic
                        },
                        "relative_size": {
                            "radius": (layer.radius / marker_size_graphic),
                            "thickness" : layer.thickness / marker_size_graphic
                        },
                        "color": color_hex,  # Store color as hex
                        "fill": layer.fill
                    })
                elif layer.layer_type == "line":
                    marker_config.append({
                        "type": "line",
                        "relative_position": {
                            "x1": (center_x - layer.x1)/marker_size_graphic,
                            "y1": (center_y - layer.y1)/marker_size_graphic,
                            "x2": (center_x - layer.x2)/marker_size_graphic,
                            "y2": (center_y - layer.y2)/marker_size_graphic
                        },
                        "color": color_hex,  # Store color as hex
                        "thickness": (layer.thickness / marker_size_graphic)
                    })
                elif layer.layer_type == "text":
                    marker_config.append({
                        "type": "text",
                        "relative_position": {
                            "x": (center_x - layer.x)/marker_size_graphic,
                            "y": (center_y - layer.y)/marker_size_graphic
                        },
                        "text": layer.text,
                        "font_size": (layer.font_size / marker_size_graphic),
                        "color": color_hex,  # Store color as hex
                        "rotation": layer.rotation,
                        "bold": layer.bold,
                        "italic": layer.italic,
                        "underline": layer.underline
                    })
            config[f"marker_{marker_id}"] = marker_config

        # Sauvegarder dans un fichier JSON
        with open("./config/ra_config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        QtWidgets.QMessageBox.information(self, "Succès", "Configuration sauvegardée dans 'ra_config.json'.")
        
    def load_ra_params(self):
        """
        Charge les paramètres de réalité augmentée depuis un fichier JSON.
        """
        try:
            with open("./config/ra_config.json", "r") as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Le fichier 'ra_config.json' est introuvable.")
            return
        except json.JSONDecodeError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Le fichier 'ra_config.json' est corrompu ou invalide.")
            return

        # Réinitialiser les layers_by_marker et la scène
        self.layers_by_marker = {}
        self.remove_all_layers()

        # Taille du marqueur dans la zone graphique (300 pixels)
        marker_size_graphic = self.marker_size_graphic

        center_x = 49
        center_y = 14

        for marker_id, layers in config.items():
            marker_id = marker_id.replace("marker_", "")  # Extraire l'ID du marqueur
            self.layers_by_marker[marker_id] = []

            for layer_data in layers:
                layer_type = layer_data["type"]
                color = layer_data["color"]

                if layer_type == "circle":
                    x = center_x - (layer_data["relative_position"]["x"] * marker_size_graphic)
                    y = center_y - (layer_data["relative_position"]["y"] * marker_size_graphic)
                    radius = layer_data["relative_size"]["radius"] * marker_size_graphic
                    thickness = layer_data["relative_size"]["thickness"] * marker_size_graphic
                    fill = layer_data["fill"]
                    layer = CircleLayer(len(self.layers_by_marker[marker_id]) + 1, x, y, radius, color, thickness=thickness, fill=fill)

                elif layer_type == "line":
                    x1 = center_x - (layer_data["relative_position"]["x1"] * marker_size_graphic)
                    y1 = center_y - (layer_data["relative_position"]["y1"] * marker_size_graphic)
                    x2 = center_x - (layer_data["relative_position"]["x2"] * marker_size_graphic)
                    y2 = center_y - (layer_data["relative_position"]["y2"] * marker_size_graphic)
                    thickness = layer_data["thickness"] * marker_size_graphic
                    layer = LineLayer(len(self.layers_by_marker[marker_id]) + 1, x1, y1, x2, y2, color, thickness)

                elif layer_type == "text":
                    x = center_x - (layer_data["relative_position"]["x"] * marker_size_graphic)
                    y = center_y - (layer_data["relative_position"]["y"] * marker_size_graphic)
                    text = layer_data["text"]
                    font_size = int(layer_data["font_size"] * marker_size_graphic)
                    rotation = layer_data["rotation"]
                    bold = layer_data["bold"]
                    italic = layer_data["italic"]
                    underline = layer_data["underline"]
                    layer = TextLayer(len(self.layers_by_marker[marker_id]) + 1, x, y, text, color, font_size, rotation, bold, italic, underline)

                else:
                    continue  # Ignorer les types inconnus

                self.layers_by_marker[marker_id].append(layer)

        # Charger les layers pour le marqueur actuellement sélectionné
        current_marker = self.comboBox_ChooseTag.currentText()
        self.load_layers_for_marker(current_marker)

        QtWidgets.QMessageBox.information(self, "Succès", "Configuration chargée avec succès.")

    def go_to_main(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)  # Revenir au menu principal