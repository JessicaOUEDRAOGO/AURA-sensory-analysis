# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.core.utils.paths import config_path


class HandTrackingThread(QThread):
    frame_signal = pyqtSignal(object)
    hands_signal = pyqtSignal(object)
    def __init__(self, camera_id=0):
        super().__init__()

        # =========================================================
        # CHEMINS
        # =========================================================
        BASE_DIR = Path(__file__).resolve().parent

        self.MODEL_PATH = BASE_DIR / "hand_landmarker.task"
        self.CALIB_PATH = Path("config/camera_calibration_top.json")
        self.POSE_PATH = Path("config/camtop_table_pose.json")
        

        self.OUTPUT_PATH = config_path("hands_positions.json")

        self.camera_id = camera_id
        self.running = True

        # =========================================================
        # PARAM
        # =========================================================
        self.TABLE_SIZE_MM = 580.0
        self.GRID_SIZE = 700
        
        # =========================================================
        # LOAD CALIB
        # =========================================================
        self.rvec, self.tvec, self.K = self.load_pose()

        # =========================================================
        # MEDIAPIPE
        # =========================================================
        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
    
        
    # =========================================================
    # LOAD
    # =========================================================
    def load_pose(self):
        data = json.load(open(self.POSE_PATH))

        rvec = np.array(data["rvec"], dtype=np.float64)
        tvec = np.array(data["tvec"], dtype=np.float64)
        K    = np.array(data["camera_matrix"], dtype=np.float64)

        return rvec, tvec, K

    # =========================================================
    # GEO
    # =========================================================
    def pixel_to_ray(self, u, v):
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        return ray / np.linalg.norm(ray)

    def intersect_ray_plane(self, ray):
        R, _ = cv2.Rodrigues(self.rvec)
        normal = R[:, 2]
        plane_origin = self.tvec.reshape(3)

        denom = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None

        t = np.dot(normal, plane_origin) / denom
        if t < 0:
            return None

        return t * ray

    def camera_to_table(self, pt_cam):
        R, _ = cv2.Rodrigues(self.rvec)
        pt = R.T @ (pt_cam - self.tvec.reshape(3))
        return pt[0], pt[1]

    def pixel_to_table(self, u, v):
        ray = self.pixel_to_ray(u, v)
        pt_cam = self.intersect_ray_plane(ray)
        if pt_cam is None:
            return None
        return self.camera_to_table(pt_cam)

    def flip_y(self, x_mm, y_mm):
        return x_mm, self.TABLE_SIZE_MM - y_mm

    def mm_to_graph(self, x_mm, y_mm):
        xg = (x_mm / self.TABLE_SIZE_MM) * self.GRID_SIZE
        yg = (y_mm / self.TABLE_SIZE_MM) * self.GRID_SIZE
        return xg, yg

    # =========================================================
    # GRIP
    # =========================================================
    def get_grip_point(self, hand, w, h):
        p4 = hand[4]
        p8 = hand[8]

        pt4 = np.array([p4.x * w, p4.y * h])
        pt8 = np.array([p8.x * w, p8.y * h])

        return (pt4 + pt8) / 2

    # =========================================================
    # THREAD LOOP
    # =========================================================
    def run(self):

        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        print(f"[HAND THREAD] Camera {self.camera_id} started")
        if not cap.isOpened():
            print("[HAND THREAD ERROR] Camera not opened")
            return
        while self.running:

            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.detector.detect(mp_image)

            hands_data = []

            if result.hand_landmarks:
                for hand_idx, hand in enumerate(result.hand_landmarks):

                    # =============================
                    # GRIP POINT
                    # =============================
                    grip = self.get_grip_point(hand, w, h)
                    u, v = grip

                    # =============================
                    # PIXEL → TABLE
                    # =============================
                    res = self.pixel_to_table(u, v)
                    if res is None:
                        continue

                    x_mm, y_mm = res

                    # =============================
                    # ALIGNEMENT
                    # =============================
                    x_mm, y_mm = self.flip_y(x_mm, y_mm)

                    # =============================
                    # GRAPH
                    # =============================
                    xg, yg = self.mm_to_graph(x_mm, y_mm)

                    # =============================
                    # SAVE
                    # =============================
                    hands_data.append({
                        "id": hand_idx,
                        "x": float(xg),
                        "y": float(yg)
                    })

            # EMIT HANDS DATA (TEMPS RÉEL)
            self.hands_signal.emit(hands_data)

            # EMIT FRAME (optionnel debug)
            self.frame_signal.emit(frame)
        cap.release()
        print("[HAND THREAD] stopped")