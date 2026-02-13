# -*- coding: utf-8 -*-
import cv2


class CameraManager:
    def __init__(self, camera_index=0, width=1920, height=1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    # --------------------------------------------------
    # OUVERTURE CAMERA
    # --------------------------------------------------
    def open_camera(self):
        """
        Ouvre la caméra avec configuration résolution.
        Compatible Windows (CAP_DSHOW) et fallback si nécessaire.
        """

        if self.cap is not None:
            print("Caméra déjà ouverte.")
            return

        # Windows optimisation
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Fallback si échec
        if not self.cap.isOpened():
            print("CAP_DSHOW échoué, tentative fallback...")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra (index={self.camera_index})")

        # Configuration résolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print(f"Caméra {self.camera_index} ouverte ({self.width}x{self.height})")

    # --------------------------------------------------
    # CAPTURE FRAME
    # --------------------------------------------------
    def get_frame(self):
        """
        Retourne une frame BGR.
        """
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Impossible de lire une frame caméra.")
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
