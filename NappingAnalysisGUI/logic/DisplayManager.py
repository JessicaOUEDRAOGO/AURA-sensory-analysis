import cv2
import numpy as np
from screeninfo import get_monitors
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap

class DisplayManager:
    def __init__(self, label: QLabel = None, projector_screen_id: int = 2):
        self.label = label
        self.projector_screen_id = projector_screen_id  # Identifiant d'écran par défaut

    def show_frame(self, frame):
        # Convertir l'image OpenCV en format compatible avec Qt
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img.rgbSwapped())
        
        # Afficher l'image dans le QLabel
        self.label.setPixmap(q_pixmap)

    def display_image_on_projector_monitor(self, image_to_display, screen_id=None):
        # Récupérer les informations des écrans
        monitors = get_monitors()
        # Utilise l'identifiant fourni ou celui par défaut
        if screen_id is None:
            screen_id = self.projector_screen_id
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError("screen_id invalide")
        monitor = monitors[screen_id]  # Écran numéro 3 (indice 2)
        # **Affichage de l'image sur l'écran 3**
        flipped_image = cv2.flip(image_to_display, 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Image", monitor.x, monitor.y)  # Déplacer l'image sur l'écran 3
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", flipped_image)  # Afficher l'image retournée sur l'écran 3

    def close_display(self):
        if self.label:
            self.label.clear()