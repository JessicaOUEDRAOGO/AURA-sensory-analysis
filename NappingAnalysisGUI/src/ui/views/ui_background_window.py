# -*- coding: utf-8 -*-
import os
from datetime import datetime

import cv2
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QFileDialog, QGraphicsView

from src.core.utils.paths import gui_path, asset_path
from src.ui.controllers.key_handler import KeyHandler
from src.ui.widgets.graphics_scene import GraphicsScene


class BackgroundWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        uic.loadUi(gui_path("Background_Menu.ui"), self)

        self.parent = parent  # MainApp

        # Grille (si tu utilises GraphicsScene avec repères)
        self.grid_xmin = -401
        self.grid_xmax = 499
        self.grid_ymin = -436
        self.grid_ymax = 464

        self.grid_size = 900
        self.scene = None
        self.setup_graphics_view()

        self.cropped_pixmap = None  # dernier crop

        self.pushButton_return.clicked.connect(self.go_to_main)
        self.pushButton_openExplo.clicked.connect(self.open_image_explorer)
        self.pushButton_activate.clicked.connect(self.save_cropped_background)

        self.key_handler = KeyHandler(self)

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)

    # ---------------------------------------------------------------------
    # UI / SCENE
    # ---------------------------------------------------------------------
    def setup_graphics_view(self):
        size = self.graphicsView.viewport().size()

        self.scene = GraphicsScene(
            grid_size=self.grid_size,
            status_mathsElement=False,
            grid_xmin=self.grid_xmin,
            grid_xmax=self.grid_xmax,
            grid_ymin=self.grid_ymin,
            grid_ymax=self.grid_ymax,
        )
        self.scene.setSceneRect(0, 0, size.width(), size.height())
        self.graphicsView.setScene(self.scene)

        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)

    # ---------------------------------------------------------------------
    # IMAGE IMPORT + CROP
    # ---------------------------------------------------------------------
    def open_image_explorer(self):
        """
        Sélection image -> crop carré -> affichage preview -> stocke self.cropped_pixmap
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")

        if not file_dialog.exec():
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        image_path = selected_files[0]
        self.lineEdit_path.setText(image_path)

        cropped = self.crop_to_square(image_path)
        if cropped is None:
            return

        self.cropped_pixmap = cropped

        # Preview dans la scène
        scene_rect = self.scene.sceneRect()
        target_size = self.grid_size

        scaled = cropped.scaled(
            target_size, target_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.scene.clear()
        x = (scene_rect.width() - scaled.width()) / 2
        y = (scene_rect.height() - scaled.height()) / 2
        self.scene.addPixmap(scaled).setPos(x, y)

    def crop_to_square(self, image_path: str):
        """
        Crop carré CENTRÉ (meilleur rendu que coin haut/gauche).
        Retourne QPixmap ou None.
        """
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de charger l'image sélectionnée.")
            return None

        w, h = pixmap.width(), pixmap.height()
        side = min(w, h)

        x0 = (w - side) // 2
        y0 = (h - side) // 2

        rect = QRect(x0, y0, side, side)
        return pixmap.copy(rect)

    # ---------------------------------------------------------------------
    # COMPOSITION + PROJECTION
    # ---------------------------------------------------------------------
    def _compose_background(self, pixmap: QPixmap):
        """
        Colle un pixmap carré au centre du fond blanc (parent.image_background).
        Retourne une image OpenCV (BGR).
        """
        default_bg = self.parent.image_background
        if default_bg is None:
            raise RuntimeError("parent.image_background est None (fond blanc non chargé).")

        bg_h, bg_w = default_bg.shape[:2]

        # Scale crop pour tenir dans le carré central (hauteur=bg_h)
        scaled = pixmap.scaled(
            bg_h, bg_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        qimg = scaled.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w2, h2 = qimg.width(), qimg.height()

        ptr = qimg.bits()
        ptr.setsize(h2 * w2 * 3)
        img_rgb = np.array(ptr, dtype=np.uint8).reshape((h2, w2, 3))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        out = default_bg.copy()
        x_offset = (bg_w - w2) // 2
        y_offset = (bg_h - h2) // 2
        out[y_offset:y_offset + h2, x_offset:x_offset + w2] = img_bgr
        return out

    def compose_background_with_crop(self):
        """
        Compose + met à jour record_window + affiche sur projecteur.
        """
        if self.cropped_pixmap is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune image cropée à coller.")
            return

        img_final = self._compose_background(self.cropped_pixmap)

        # Mettre à jour RecordWindow
        if hasattr(self.parent, "record_window") and self.parent.record_window is not None:
            self.parent.record_window.image_background = img_final
            self.parent.record_window.image_background_clean = img_final.copy()
            self.parent.record_window.image_background_with_grid = img_final.copy()

        # Afficher sur projecteur (id depuis settings)
        self.display_background_on_projector(img_final)

        # Si grille activée, refresh
        if hasattr(self.parent, "record_window") and self.parent.record_window is not None:
            self.parent.record_window.refresh_projector_background()

    def display_background_on_projector(self, image):
        """
        Affiche sur le projecteur sélectionné dans settings.
        """
        screen_id = self.parent.settings.get("projector_screen_id", None)
        self.parent.display_manager.display_image_on_projector_monitor(image, screen_id=screen_id)

    # ---------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------
    def save_cropped_background(self):
        """
        Sauvegarde crop + sauvegarde final + applique.
        """
        if self.cropped_pixmap is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune image. Sélectionne une image d'abord.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        textures_dir = asset_path("textures")
        os.makedirs(textures_dir, exist_ok=True)

        # crop
        save_path_crop = os.path.join(textures_dir, f"background_{timestamp}_crop.png")
        self.cropped_pixmap.save(save_path_crop, "PNG")
        print(f"Background crop sauvegardé : {save_path_crop}")

        # appliquer (compose + projector + record_window)
        self.compose_background_with_crop()

        # final
        img_final = self._compose_background(self.cropped_pixmap)
        save_path_final = os.path.join(textures_dir, f"background_{timestamp}_final.png")
        cv2.imwrite(save_path_final, img_final)
        print(f" Background final sauvegardé : {save_path_final}")

    # ---------------------------------------------------------------------
    # NAV
    # ---------------------------------------------------------------------
    def go_to_main(self):
        if self.scene:
            self.scene.clear()
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)
