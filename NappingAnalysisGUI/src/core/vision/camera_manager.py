# -*- coding: utf-8 -*-
import cv2


class CameraManager:
    from src.core.config.app_config import (
        CAMERA_ID,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        CAMERA_FPS,
    )
    def __init__(self, camera_index=CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    # --------------------------------------------------
    # OUVERTURE CAMERA
    # --------------------------------------------------
    def open_camera(self):
        """
        Ouvre la caméra en forçant une résolution unique.
        Vérifie la résolution réelle obtenue.
        """
        if self.cap is not None and self.cap.isOpened():
            print("Caméra déjà ouverte.")
            return True

        # Tentative Windows
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Fallback si échec
        if not cap.isOpened():
            print("CAP_DSHOW échoué, tentative fallback...")
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra (index={self.camera_index})")

        # Forcer la résolution AVANT lecture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Première lecture de validation
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("Impossible de lire une première frame caméra.")

        real_h, real_w = frame.shape[:2]
        print(f"Caméra {self.camera_index} ouverte ({real_w}x{real_h})")

        # Vérification stricte
        if (real_w, real_h) != (self.width, self.height):
            cap.release()
            raise RuntimeError(
                f"Résolution caméra incohérente : obtenue {real_w}x{real_h}, "
                f"attendue {self.width}x{self.height}"
            )

        self.cap = cap
        return True

    # --------------------------------------------------
    # CAPTURE FRAME
    # --------------------------------------------------
    def get_frame(self):
        """
        Retourne une frame BGR.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Impossible de lire une frame caméra.")
            return None

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            print(
                f"Frame ignorée : résolution inattendue {w}x{h} "
                f"(attendu {self.width}x{self.height})"
            )
            return None

        return frame

    # --------------------------------------------------
    # FERMETURE CAMERA
    # --------------------------------------------------
    def close_camera(self):
        """
        Ferme proprement la caméra.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Caméra fermée.")