# -*- coding: utf-8 -*-
import cv2
import numpy as np
from screeninfo import get_monitors
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap


class DisplayManager:
    """
    - label: QLabel optionnel (preview caméra dans UI).
    - projector_screen_id: index écran pour projection (screeninfo).
    """
    def __init__(self, label: QLabel = None, projector_screen_id: int = 2):
        self.label = label
        self.projector_screen_id = projector_screen_id
        self.resolution = None  # optionnel: (w, h)
        self._projector_window_name = "Projector"

    def show_frame(self, frame: np.ndarray):
        """
        Affiche une frame OpenCV dans un QLabel (si présent).
        Gère BGR, RGB, GRAY.
        """
        if self.label is None:
            return

        if frame is None or frame.size == 0:
            return

        # Convertir en RGB pour Qt
        if len(frame.shape) == 2:
            # Grayscale
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(q_pixmap)

    def display_image_on_projector_monitor(self, image_to_display: np.ndarray, screen_id: int = None):
        """
        Affiche une image OpenCV sur l'écran projecteur (fullscreen).
        - applique flip horizontal (miroir) comme ton V1.
        - peut redimensionner si self.resolution est définie.
        """
        if image_to_display is None or image_to_display.size == 0:
            return

        monitors = get_monitors()

        if screen_id is None:
            screen_id = self.projector_screen_id

        # Sécurisation de l'ID écran
        if screen_id < 0 or screen_id >= len(monitors):
            print(f"[WARNING] screen_id {screen_id} invalide, fallback sur dernier écran disponible.")
            screen_id = len(monitors) - 1

        monitor = monitors[screen_id]

        img = image_to_display

        # Resize si une résolution est demandée
        if self.resolution is not None:
            try:
                w, h = self.resolution
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARNING] Impossible de resize vers {self.resolution}: {e}")

        # Flip horizontal (effet miroir)
        flipped_image = cv2.flip(img, 1)

        cv2.namedWindow(self._projector_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self._projector_window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(self._projector_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self._projector_window_name, flipped_image)
        cv2.waitKey(1)

    def close_display(self):
        """
        Nettoie l'affichage label + ferme la fenêtre projecteur si ouverte.
        """
        if self.label is not None:
            self.label.clear()

        try:
            cv2.destroyWindow(self._projector_window_name)
        except Exception:
            pass
