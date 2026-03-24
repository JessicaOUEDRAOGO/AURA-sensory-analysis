# -*- coding: utf-8 -*-
import cv2
import numpy as np
from screeninfo import get_monitors
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap


class DisplayManager:
    """
    Gestion centralisée de l'affichage projecteur.

    IMPORTANT :
    Toute correction d'orientation d'affichage doit être définie ici,
    et uniquement ici.

    Le repère "projecteur logique" correspond au repère utilisé par le
    CoordinateMapper et par le runtime pour dessiner.

    Le repère "affichage réel" correspond à l'image effectivement envoyée
    à l'écran du projecteur après transformation de display.
    """

    def __init__(self, label: QLabel = None, projector_screen_id: int = 2):
        self.label = label
        self.projector_screen_id = projector_screen_id
        self.resolution = None
        self._projector_window_name = "Projector"

    def show_frame(self, frame: np.ndarray):
        if self.label is None:
            return

        if frame is None or frame.size == 0:
            return

        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(q_pixmap)

    def _get_target_monitor(self, screen_id=None):
        monitors = get_monitors()

        if screen_id is None:
            screen_id = self.projector_screen_id

        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide: {screen_id}, moniteurs disponibles: {len(monitors)}")

        return monitors[screen_id]

    def get_projection_size(self, screen_id=None):
        monitor = self._get_target_monitor(screen_id)
        return int(monitor.width), int(monitor.height)

    def get_display_transform_matrix(self, width: int, height: int) -> np.ndarray:
        """
        Matrice homogène du transform d'affichage appliqué au projecteur.

        Ici : flip vertical
            x' = x
            y' = height - 1 - y
        """
        return np.array([
            [1.0,  0.0, 0.0],
            [0.0, -1.0, height - 1],
            [0.0,  0.0, 1.0]
        ], dtype=np.float64)

    def transform_projector_point_to_display(self, pt, height: int):
        """
        Transforme un point du repère projecteur logique
        vers le repère d'affichage réel.
        """
        x = float(pt[0])
        y = float(pt[1])
        return np.array([x, height - 1 - y], dtype=np.float32)

    def transform_projector_homography_to_display(self, H: np.ndarray, height: int) -> np.ndarray:
        """
        Convertit une homographie qui produit des coordonnées dans le repère
        projecteur logique vers une homographie dans le repère d'affichage réel.
        """
        T = self.get_display_transform_matrix(width=1, height=height)
        H_corr = T @ H

        if abs(H_corr[2, 2]) > 1e-12:
            H_corr = H_corr / H_corr[2, 2]

        return H_corr

    def _prepare_for_projection(self, img: np.ndarray) -> np.ndarray:
        """
        Corrige l'image pour le point de vue utilisateur.
        Ici : flip vertical.
        """
        return cv2.flip(img, 0)

    def display_image_on_projector_monitor(self, image_to_display: np.ndarray, screen_id: int = None):
        if image_to_display is None:
            print("[DISPLAY] image_to_display = None")
            return

        if not isinstance(image_to_display, np.ndarray):
            print(f"[DISPLAY] type invalide: {type(image_to_display)}")
            return

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

        monitor = self._get_target_monitor(screen_id)

        img = self._prepare_for_projection(image_to_display)

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
        if self.label is not None:
            self.label.clear()

        try:
            cv2.destroyWindow(self._projector_window_name)
        except Exception:
            pass