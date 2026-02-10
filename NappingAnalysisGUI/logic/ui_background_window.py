from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QGraphicsView, QFileDialog
from PyQt6.QtGui import QPainter, QPixmap, QImage
from PyQt6 import uic, QtWidgets
import numpy as np
import cv2
from datetime import datetime

from logic.key_handler import KeyHandler
from logic.graphics_scene import GraphicsScene

class BackgroundWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()

        uic.loadUi("./gui/Background_Menu.ui", self)

        self.parent = parent  # Référence à MainApp
        
        self.grid_xmin = -401
        self.grid_xmax = 499
        self.grid_ymin = -436
        self.grid_ymax = 464

        self.scene = GraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.layer_counter = 0  # Unique ID for layers

        self.grid_size = 900
        self.setup_graphics_view()

        self.cropped_pixmap = None  # Pour stocker le dernier crop

        self.pushButton_return.clicked.connect(self.go_to_main)
        self.pushButton_openExplo.clicked.connect(self.open_image_explorer)
        self.pushButton_activate.clicked.connect(self.save_cropped_background)  # Nouveau bouton
        
        self.key_handler = KeyHandler(self)

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)
        
    def setup_graphics_view(self):
        """Configure la scène et la vue graphique avec une grille."""
        size = self.graphicsView.viewport().size()
        self.scene = GraphicsScene(grid_size=self.grid_size, status_mathsElement = False, grid_xmin = self.grid_xmin, grid_xmax = self.grid_xmax, grid_ymin = self.grid_ymin, grid_ymax = self.grid_ymax)  # Taille de la grille
        self.scene.setSceneRect(0, 0, size.width(), size.height())  # Définir la taille de la scène

        self.graphicsView.setScene(self.scene)  # Assigner la scène au QGraphicsView
        # Options d'affichage
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)

    def open_image_explorer(self):
        """
        Ouvre un explorateur de fichiers pour sélectionner une image, la crop en carré,
        l'affiche dans la zone graphique, et stocke le QPixmap cropé.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                self.lineEdit_path.setText(image_path)
                cropped = self.crop_to_square(image_path)
                if cropped:
                    self.cropped_pixmap = cropped  # Stocke le crop pour l'activation
                    # Afficher dans la GUI
                    scene_rect = self.scene.sceneRect()
                    target_size = 900
                    scaled = cropped.scaled(target_size, target_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.scene.clear()
                    x = (scene_rect.width() - scaled.width()) / 2
                    y = (scene_rect.height() - scaled.height()) / 2
                    self.scene.addPixmap(scaled).setPos(x, y)

    def compose_background_with_crop(self):
        """
        Colle le QPixmap cropé (self.cropped_pixmap) sur le fond blanc (self.parent.image_background)
        et met à jour l'image_background dans RecordWindow, puis affiche sur l'écran 2.
        """
        if self.cropped_pixmap is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune image cropée à coller.")
            return
        img1 = self._compose_background(self.cropped_pixmap)
        # Mettre à jour l'image_background dans RecordWindow
        self.parent.record_window.image_background = img1
        self.parent.record_window.image_background_clean = img1.copy()  # <-- Ajout important
        self.parent.record_window.image_background_with_grid = img1.copy()
        # Afficher sur l'écran 2
        self.display_background_on_projector(img1)
        # Rafraîchir la grille projetée si besoin
        self.parent.record_window.refresh_projector_background()  # <-- Ajout important

    def display_background_on_projector(self, image):
        """
        Affiche l'image donnée sur l'écran 2 via DisplayManager.
        """
        self.parent.display_manager.display_image_on_projector_monitor(image)

    def _compose_background(self, pixmap):
        """
        Colle le QPixmap donné sur le fond blanc (self.parent.image_background) et retourne l'image OpenCV résultante.
        """
        default_bg = self.parent.image_background
        bg_width = default_bg.shape[1]
        bg_height = default_bg.shape[0]

        # Redimensionner le crop pour qu'il tienne dans le fond blanc
        scaled_cropped = pixmap.scaled(bg_height, bg_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        cropped_qimage = scaled_cropped.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w2, h2 = cropped_qimage.width(), cropped_qimage.height()
        ptr = cropped_qimage.bits()
        ptr.setsize(h2 * w2 * 3)
        img2 = np.array(ptr, dtype=np.uint8).reshape((h2, w2, 3))
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        img1 = default_bg.copy()
        x_offset = (bg_width - w2) // 2
        y_offset = (bg_height - h2) // 2
        img1[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2
        return img1

    def crop_to_square(self, image_path):
        """
        Charge l'image et la crop en carré (gauche ou haut selon l'orientation).
        Retourne un QPixmap carré ou None si erreur.
        """
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de charger l'image sélectionnée.")
            return None

        w, h = pixmap.width(), pixmap.height()
        if w > h:
            rect = QRectF(0, 0, h, h)
        else:
            rect = QRectF(0, 0, w, w)
        return pixmap.copy(rect.toRect())

    def save_cropped_background(self):
        """
        Sauvegarde l'image cropée en carré (pleine résolution) dans ./assets/textures/ avec un nom unique.
        Puis colle et affiche le background sur l'écran 2, et sauvegarde aussi l'image finale avec collage.
        """
        if self.cropped_pixmap is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune image à sauvegarder. Veuillez d'abord sélectionner une image.")
            return

        # Générer un nom unique (timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_crop = f"./assets/textures/background_{timestamp}_crop.png"
        self.cropped_pixmap.save(save_path_crop, "PNG")
        print(f"Background cropé sauvegardé : {save_path_crop}")

        # Coller et afficher sur l'écran 2
        self.compose_background_with_crop()

        # Sauvegarder aussi l'image finale avec collage
        img_final = self._compose_background(self.cropped_pixmap)
        save_path_final = f"./assets/textures/background_{timestamp}_final.png"
        cv2.imwrite(save_path_final, img_final)
        print(f"Background final (avec collage) sauvegardé : {save_path_final}")


    def go_to_main(self):
        # Reset the scene: clear all items except the default grid lines
        self.scene.clear()
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)  # Revenir au menu principal

