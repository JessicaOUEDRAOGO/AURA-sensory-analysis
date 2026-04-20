# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path
import time as pytime
from collections import deque
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

        BASE_DIR = Path(__file__).resolve().parent

        self.MODEL_PATH = BASE_DIR / "hand_landmarker.task"
        self.CALIB_PATH = Path("config/camera_calibration_top.json")
        self.POSE_PATH  = Path("config/camtop_table_pose.json")
        self.OUTPUT_PATH = config_path("hands_positions.json")

        self.camera_id = camera_id
        self.running = True

        self.TABLE_SIZE_MM = 580.0
        self.GRID_SIZE     = 700

        # ------------------------------------------------------------------
        # NOUVEAU : lissage temporel
        # Fenêtre glissante par main (id → deque de np.array [xg, yg])
        # ------------------------------------------------------------------
        self.SMOOTH_WINDOW = 5          # nombre de frames conservées
        self._smooth_buffers: dict[int, deque] = {}

        # ------------------------------------------------------------------
        # NOUVEAU : résultat du détecteur LIVE_STREAM (thread-safe via lock)
        # ------------------------------------------------------------------
        import threading
        self._result_lock   = threading.Lock()
        self._latest_result = None      # stocke le dernier HandLandmarkerResult

        self.rvec, self.tvec, self.K = self.load_pose()

        # ------------------------------------------------------------------
        # NOUVEAU : mode LIVE_STREAM  → callback asynchrone, pas de blocage
        # ------------------------------------------------------------------
        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.LIVE_STREAM,   # ← changé
            result_callback=self._on_result                 # ← callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    # ======================================================================
    # Callback LIVE_STREAM  (appelé dans le thread MediaPipe interne)
    # ======================================================================
    def _on_result(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._latest_result = result

    # ======================================================================
    # Chargement calibration
    # ======================================================================
    def load_pose(self):
        data = json.load(open(self.POSE_PATH))
        rvec = np.array(data["rvec"], dtype=np.float64)
        tvec = np.array(data["tvec"], dtype=np.float64)
        K    = np.array(data["camera_matrix"], dtype=np.float64)
        return rvec, tvec, K

    # ======================================================================
    # Géométrie
    # ======================================================================
    def pixel_to_ray(self, u, v):
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        return ray / np.linalg.norm(ray)

    def intersect_ray_plane(self, ray):
        R, _ = cv2.Rodrigues(self.rvec)
        normal       = R[:, 2]
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
        pt   = R.T @ (pt_cam - self.tvec.reshape(3))
        return pt[0], pt[1]

    def pixel_to_table(self, u, v):
        ray    = self.pixel_to_ray(u, v)
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

    # ======================================================================
    # NOUVEAU : point de grip amélioré
    # ======================================================================
    def get_grip_point(self, hand, w, h):
        """
        Calcule le point de préhension pour un objet cylindrique (tasse).

        Ancienne méthode : moyenne pouce (4) + index (8) → trop haut,
        sensible à l'écartement des doigts, instable en rotation.

        Nouvelle méthode : centroïde pondéré des bases de tous les doigts
        (metacarpo-phalangeal joints 5, 9, 13, 17) + palme (0).
        Ce cluster est stable quelle que soit la posture des doigts et
        correspond physiquement à l'endroit où la main enserre l'objet.

        Poids :
          - MCP index  (5)  × 1.2  — référence principale
          - MCP majeur (9)  × 1.5  — centre de masse de la prise
          - MCP annulaire(13)× 1.2
          - MCP auriculaire(17)× 0.8
          - Palme      (0)  × 0.8  — ancre basse (stabilité)
        """
        IDX   = [(5, 1.2), (9, 1.5), (13, 1.2), (17, 0.8), (0, 0.8)]
        total_w = sum(w_i for _, w_i in IDX)

        x_sum = y_sum = 0.0
        for lm_idx, w_i in IDX:
            lm  = hand[lm_idx]
            x_sum += lm.x * w * w_i
            y_sum += lm.y * h * w_i

        return np.array([x_sum / total_w, y_sum / total_w], dtype=np.float32)

    # ======================================================================
    # NOUVEAU : lissage temporel (moyenne glissante)
    # ======================================================================
    def _smooth(self, hand_id: int, point: np.ndarray) -> np.ndarray:
        """
        Conserve les N dernières positions et retourne leur moyenne.
        Élimine le tremblement sans introduire de délai perceptible.
        """
        buf = self._smooth_buffers.setdefault(
            hand_id, deque(maxlen=self.SMOOTH_WINDOW)
        )
        buf.append(point)
        return np.mean(buf, axis=0)

    def _purge_lost_hands(self, active_ids: set):
        """Supprime les buffers des mains qui ont disparu."""
        for hid in list(self._smooth_buffers.keys()):
            if hid not in active_ids:
                del self._smooth_buffers[hid]

    # ======================================================================
    # Boucle principale
    # ======================================================================
    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # NOUVEAU : désactive le buffer interne de VideoCapture (1 frame max)
        # → réduit la latence de lecture caméra de ~60–100 ms
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[HAND THREAD] Camera {self.camera_id} started (LIVE_STREAM mode)")
        if not cap.isOpened():
            print("[HAND THREAD ERROR] Camera not opened")
            return

        timestamp_ms = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # NOUVEAU : LIVE_STREAM → detect_async, non bloquant
            # MediaPipe appelle _on_result en callback dans son propre thread.
            timestamp_ms += 1          # doit être strictement croissant
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.detector.detect_async(mp_image, timestamp_ms)

            # Lire le dernier résultat disponible (peut être N-1 frames en arrière)
            with self._result_lock:
                result = self._latest_result

            hands_data   = []
            active_ids   = set()

            if result and result.hand_landmarks:
                for hand_idx, hand in enumerate(result.hand_landmarks):

                    # ---- point de grip amélioré ----
                    grip = self.get_grip_point(hand, w, h)
                    u, v = grip

                    # ---- projection pixel → table ----
                    res = self.pixel_to_table(u, v)
                    if res is None:
                        continue
                    x_mm, y_mm = res
                    x_mm, y_mm = self.flip_y(x_mm, y_mm)
                    xg,   yg   = self.mm_to_graph(x_mm, y_mm)

                    # ---- lissage temporel ----
                    smoothed  = self._smooth(hand_idx, np.array([xg, yg]))
                    active_ids.add(hand_idx)

                    hands_data.append({
                        "id": hand_idx,
                        "x":  float(smoothed[0]),
                        "y":  float(smoothed[1])
                    })

            self._purge_lost_hands(active_ids)

            self.hands_signal.emit(hands_data)
            self.frame_signal.emit(frame)

            # NOUVEAU : sleep supprimé — la latence est dictée par MediaPipe.
            # On laisse quand même 1 ms pour céder le CPU si besoin.
            pytime.sleep(0.001)

        cap.release()
        print("[HAND THREAD] stopped")
