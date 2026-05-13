# -*- coding: utf-8 -*-
import cv2
import numpy as np


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

        # UNDISTORT
        self.map1 = None
        self.map2 = None
        self.K = None
        self.dist = None

    # --------------------------------------------------
    # OUVERTURE CAMERA
    # --------------------------------------------------
    def open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            print("Caméra déjà ouverte.")
            return True

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("CAP_DSHOW échoué, tentative fallback...")
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra (index={self.camera_index})")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        for _ in range(8):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("Impossible de lire une première frame caméra.")

        real_h, real_w = frame.shape[:2]
        print(f"Caméra {self.camera_index} ouverte ({real_w}x{real_h})")

        if (real_w, real_h) != (self.width, self.height):
            cap.release()
            raise RuntimeError(
                f"Résolution caméra incohérente : obtenue {real_w}x{real_h}, "
                f"attendue {self.width}x{self.height}"
            )

        self.cap = cap
        return True

    # --------------------------------------------------
    # UNDISTORT CONFIG
    # --------------------------------------------------
    def set_undistort(self, K, dist):
        """
        Initialise la correction de distorsion fisheye.
        DOIT être appelée après calibration.
        """

        h = self.height
        w = self.width

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, (w, h), np.eye(3), balance=0.3
        )

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        self.K = new_K
        self.dist = dist

        print("[CameraManager] Undistortion activé")

    # --------------------------------------------------
    # CAPTURE FRAME
    # --------------------------------------------------
    def get_frame(self):

        if self.cap is None:
            return None

        if not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
        except Exception:
            return None

        if not ret or frame is None:
            return None

        try:
            h, w = frame.shape[:2]
        except Exception:
            return None

        if (w, h) != (self.width, self.height):
            return None

        if self.map1 is not None and self.map2 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return frame

    # --------------------------------------------------
    # FERMETURE CAMERA
    # --------------------------------------------------
    def close_camera(self):
        if self.cap is None:          # 1. déjà None → on sort
            return
        try:
            if self.cap.isOpened():   # 2. vérifie que le handle est encore valide
                self.cap.release()
        except Exception as e:        # 3. absorbe l'erreur OpenCV au lieu de planter
            print(...)
        finally:
            self.cap = None           # 4. toujours remis à None même en cas d'erreur