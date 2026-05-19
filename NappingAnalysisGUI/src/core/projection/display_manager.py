# -*- coding: utf-8 -*-
import cv2
import numpy as np
from screeninfo import get_monitors
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap


class DisplayManager:
    """
    Gestion centralisée de l'affichage projecteur.

    Convention retenue pour le pipeline :
    - le runtime dessine directement dans le repère projecteur final
    - DisplayManager ne transforme pas les points
    - DisplayManager ne transforme pas les homographies
    - DisplayManager affiche simplement l'image reçue

    Rôle de cette classe :
    - fournir la taille de projection réelle
    - afficher un aperçu dans un QLabel si disponible
    - afficher l'image en plein écran sur le moniteur projecteur
    """

    def __init__(self, label: QLabel = None, projector_screen_id: int = 2):
        self.label = label
        self.projector_screen_id = int(projector_screen_id)
        self._projector_window_name = "Projector"
        self._window_initialized = False   # fenêtre créée une seule fois
        self._cached_monitor = None

    # ======================================================================
    # Moniteur cible
    # ======================================================================
    def _get_target_monitor(self, screen_id=None):
        monitors = get_monitors()

        if screen_id is None:
            screen_id = self.projector_screen_id

        screen_id = int(screen_id)

        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(
                f"screen_id invalide: {screen_id}, moniteurs disponibles: {len(monitors)}"
            )

        return monitors[screen_id]

    def get_projection_size(self, screen_id=None):
        """
        Retourne la taille réelle du moniteur projecteur ciblé.
        """
        monitor = self._get_target_monitor(screen_id)
        return int(monitor.width), int(monitor.height)

    # ======================================================================
    # Preview Qt
    # ======================================================================
    def show_frame(self, frame: np.ndarray):
        """
        Affiche une preview dans le QLabel fourni à l'initialisation.
        N'influe pas sur la projection réelle.
        """
        if self.label is None:
            return

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(q_pixmap)

    # ======================================================================
    # Préparation image avant projection
    # ======================================================================
    def _prepare_for_projection(self, img: np.ndarray) -> np.ndarray:
        """
        Préparation finale avant projection.

        IMPORTANT :
        Dans l'architecture nettoyée, cette fonction doit rester neutre,
        sauf décision explicite et unique de correction globale d'affichage.

        Pour l'instant : aucune transformation.
        """
        return img

    def _validate_image(self, image_to_display: np.ndarray):
        """
        Vérifie et normalise l'image avant affichage.
        Retourne une image BGR uint8 exploitable par OpenCV.
        """
        if image_to_display is None:
            raise ValueError("[DISPLAY] image_to_display = None")

        if not isinstance(image_to_display, np.ndarray):
            raise TypeError(f"[DISPLAY] type invalide: {type(image_to_display)}")

        if image_to_display.ndim not in (2, 3):
            raise ValueError(f"[DISPLAY] ndim invalide: {image_to_display.ndim}")

        img = image_to_display

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            h, w = img.shape
            if h <= 10 or w <= 10:
                raise ValueError("[DISPLAY] matrice 2D trop petite -> probablement pas une image")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif img.ndim == 3:
            h, w, c = img.shape
            if h <= 10 or w <= 10:
                raise ValueError("[DISPLAY] matrice 3D trop petite -> probablement pas une image")
            if c != 3:
                raise ValueError(f"[DISPLAY] nombre de canaux invalide: {c}")

        return img

    # ======================================================================
    # Initialisation fenêtre projecteur (une seule fois)
    # ======================================================================
    def _ensure_window(self, monitor) -> None:
        """
        Crée, positionne et passe en plein écran la fenêtre projecteur.
        N'est exécutée qu'une seule fois grâce au flag _window_initialized.
        Appels répétés à chaque frame = source du cercle fantôme et du flicker.
        """
        if self._window_initialized:
            return
        cv2.namedWindow(self._projector_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self._projector_window_name, int(monitor.x), int(monitor.y))
        cv2.setWindowProperty(
            self._projector_window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
        self._window_initialized = True

    # ======================================================================
    # Projection réelle
    # ======================================================================
    def display_image_on_projector_monitor(self, image_to_display: np.ndarray, screen_id: int = None):
        """
        Affiche l'image sur le moniteur projecteur ciblé.

        Hypothèse de travail :
        - l'image reçue est déjà dans le bon repère projecteur
        - si sa taille diffère de la résolution écran, elle est redimensionnée

        Note : cv2.waitKey() est intentionnellement absent ici.
        Il doit être appelé une seule fois par frame dans la boucle principale
        du pipeline (cup_tracking_pipeline.py), pas dans cette méthode.
        """
        try:
            img = self._validate_image(image_to_display)
        except Exception as e:
            print(str(e))
            return

        if self._cached_monitor is None or screen_id is not None:
            self._cached_monitor = self._get_target_monitor(screen_id)
        monitor = self._cached_monitor

        img = self._prepare_for_projection(img)

        target_w = int(monitor.width)
        target_h = int(monitor.height)

        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self._ensure_window(monitor)
        cv2.imshow(self._projector_window_name, img)

    # ======================================================================
    # Fermeture
    # ======================================================================
    def close_display(self):
        if self.label is not None:
            self.label.clear()

        try:
            cv2.destroyWindow(self._projector_window_name)
            self._window_initialized = False   # reset pour permettre réouverture propre
        except Exception:
            pass