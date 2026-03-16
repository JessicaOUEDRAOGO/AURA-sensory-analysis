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
        if image_to_display is None:
            print("[DISPLAY] image_to_display = None")
            return

        if not isinstance(image_to_display, np.ndarray):
            print(f"[DISPLAY] type invalide: {type(image_to_display)}")
            return

        # print("=== DISPLAY DEBUG ===")
        # print("Incoming image shape:", image_to_display.shape)
        # print("Incoming image dtype:", image_to_display.dtype)
        # print("=====================")

        # Rejeter tout ce qui n'est pas une vraie image
        if image_to_display.ndim not in (2, 3):
            print(f"[DISPLAY] ndim invalide: {image_to_display.ndim}")
            return

        if image_to_display.ndim == 2:
            h, w = image_to_display.shape
            if h <= 10 and w <= 10:
                print("[DISPLAY] matrice 2D trop petite -> probablement pas une image")
                return
            image_to_display = cv2.cvtColor(image_to_display, cv2.COLOR_GRAY2BGR)

        if image_to_display.ndim == 3:
            h, w, c = image_to_display.shape
            if h <= 10 and w <= 10:
                print("[DISPLAY] matrice 3D trop petite -> probablement pas une image")
                return
            if c != 3:
                print(f"[DISPLAY] nombre de canaux invalide: {c}")
                return

        monitors = get_monitors()

        if screen_id is None:
            screen_id = self.projector_screen_id

        # print("screen_id:", screen_id)
        # for idx, m in enumerate(monitors):
        #     print(f"Monitor {idx}: x={m.x}, y={m.y}, width={m.width}, height={m.height}")

        # if screen_id < 0 or screen_id >= len(monitors):
        #     print(f"[WARNING] screen_id {screen_id} invalide, fallback sur dernier écran disponible.")
        #     screen_id = len(monitors) - 1

        monitor = monitors[screen_id]
        img = image_to_display

        # Redimensionnement vers la vraie résolution de l'écran cible
        target_w = monitor.width
        target_h = monitor.height
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow(self._projector_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self._projector_window_name, monitor.x, monitor.y)
        cv2.setWindowProperty(
            self._projector_window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow(self._projector_window_name, img)
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
