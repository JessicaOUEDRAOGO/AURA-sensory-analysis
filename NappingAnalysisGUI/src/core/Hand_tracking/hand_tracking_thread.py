# -*- coding: utf-8 -*-
"""
hand_tracking_thread.py — VERSION v4
=====================================
Changements vs v3 :
  - Accepte un FrameBuffer en plus du HandStateBuffer
    → chaque frame cap.read() est poussée dans frame_buffer pour que
      Algorithm_Analysis puisse l'utiliser pour le tracker CSRT
  - Aucune autre logique modifiée : Kalman, HandSlotManager, HandStateBuffer
    restent identiques
  - hands_signal et hand_buffer.push() conservés — utilisés plus tard
    pour la feature dessin
"""

import cv2
import numpy as np
import json
from pathlib import Path
import time as pytime
from PyQt6.QtCore import QThread, pyqtSignal

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.core.utils.paths import config_path
from src.core.Hand_tracking.hand_state_buffer import HandStateBuffer
from src.core.Hand_tracking.frame_buffer import FrameBuffer          # ← NOUVEAU


# ---------------------------------------------------------------------------
# Filtre de Kalman 2D — position + vitesse (inchangé)
# ---------------------------------------------------------------------------
class KalmanHand:
    def __init__(self, x: float, y: float,
                 process_noise: float = 5e-2,
                 measure_noise: float = 3.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100.0
        self.kf.statePost = np.array([[x], [y], [0.], [0.]], dtype=np.float32)

        self._vel_clip      = 80.0
        self._missing_count = 0

    def update(self, x: float, y: float) -> np.ndarray:
        self._missing_count = 0
        self.kf.predict()
        c = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return np.array([c[0, 0], c[1, 0]], dtype=np.float32)

    def predict_only(self) -> np.ndarray:
        self._missing_count += 1
        decay = 0.70 ** self._missing_count
        p = self.kf.predict()
        self.kf.statePost[2, 0] = float(np.clip(
            self.kf.statePost[2, 0] * decay, -self._vel_clip, self._vel_clip
        ))
        self.kf.statePost[3, 0] = float(np.clip(
            self.kf.statePost[3, 0] * decay, -self._vel_clip, self._vel_clip
        ))
        return np.array([p[0, 0], p[1, 0]], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        s = self.kf.statePost
        return np.clip(
            np.array([s[2, 0], s[3, 0]], dtype=np.float32),
            -self._vel_clip, self._vel_clip,
        )

    @property
    def position(self) -> np.ndarray:
        s = self.kf.statePost
        return np.array([s[0, 0], s[1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Gestionnaire de slots de mains (inchangé)
# ---------------------------------------------------------------------------
class HandSlotManager:
    MAX_HANDS          = 2
    MAX_MISSING_FRAMES = 2
    MATCH_THRESHOLD    = 150
    VELOCITY_WEIGHT    = 0.4

    def __init__(self, process_noise: float, measure_noise: float):
        self._process_noise = process_noise
        self._measure_noise = measure_noise
        self._slots: dict[int, dict] = {}
        self._next_id = 0

    def _composite_score(self, slot: dict, xg: float, yg: float) -> float:
        pos       = slot["kf"].position
        vel       = slot["kf"].velocity
        det_pos   = np.array([xg, yg], dtype=np.float32)
        predicted = pos + vel
        dist_pos  = float(np.linalg.norm(pos - det_pos))
        dist_pred = float(np.linalg.norm(predicted - det_pos))
        if dist_pos > self.MATCH_THRESHOLD:
            return float("inf")
        return (1.0 - self.VELOCITY_WEIGHT) * dist_pos + self.VELOCITY_WEIGHT * dist_pred

    def update(self, raw_detections: list[tuple[float, float]]) -> list[dict]:
        unmatched_detections = list(range(len(raw_detections)))
        matched_slots        = set()

        if self._slots and raw_detections:
            slot_ids = list(self._slots.keys())
            scores   = np.full((len(slot_ids), len(raw_detections)), np.inf)
            for si, sid in enumerate(slot_ids):
                for di, (xg, yg) in enumerate(raw_detections):
                    scores[si, di] = self._composite_score(self._slots[sid], xg, yg)
            while True:
                if scores.size == 0 or np.all(np.isinf(scores)):
                    break
                si, di = np.unravel_index(np.argmin(scores), scores.shape)
                if np.isinf(scores[si, di]):
                    break
                sid = slot_ids[si]
                xg, yg = raw_detections[di]
                self._slots[sid]["kf"].update(xg, yg)
                self._slots[sid]["missing"] = 0
                matched_slots.add(sid)
                unmatched_detections.remove(di)
                scores[si, :]  = np.inf
                scores[:, di]  = np.inf

        for di in unmatched_detections:
            if len(self._slots) >= self.MAX_HANDS:
                break
            xg, yg = raw_detections[di]
            sid = self._next_id
            self._next_id += 1
            self._slots[sid] = {
                "kf":      KalmanHand(xg, yg, self._process_noise, self._measure_noise),
                "missing": 0,
            }
            self._slots[sid]["kf"].update(xg, yg)
            matched_slots.add(sid)

        to_delete = []
        for sid, slot in self._slots.items():
            if sid in matched_slots:
                continue
            slot["missing"] += 1
            if slot["missing"] > self.MAX_MISSING_FRAMES:
                to_delete.append(sid)
            else:
                slot["kf"].predict_only()
        for sid in to_delete:
            del self._slots[sid]

        result = []
        for sid, slot in self._slots.items():
            pos = slot["kf"].position
            vel = slot["kf"].velocity
            result.append({
                "id":      sid,
                "x":       float(pos[0]),
                "y":       float(pos[1]),
                "vx":      float(vel[0]),
                "vy":      float(vel[1]),
                "missing": slot["missing"],
            })
        return result


# ---------------------------------------------------------------------------
# Thread principal
# ---------------------------------------------------------------------------
class HandTrackingThread(QThread):
    frame_signal = pyqtSignal(object)
    hands_signal = pyqtSignal(object)   # conservé pour la future feature dessin

    def __init__(self, camera_id: int = 0,
                 kalman_process_noise: float = 5e-2,
                 kalman_measure_noise: float = 3.0,
                 hand_buffer: HandStateBuffer = None,
                 frame_buffer: FrameBuffer = None):          # ← NOUVEAU paramètre
        super().__init__()

        BASE_DIR = Path(__file__).resolve().parent
        self.MODEL_PATH  = BASE_DIR / "hand_landmarker.task"
        self.POSE_PATH   = Path("config/camtop_table_pose.json")
        self.OUTPUT_PATH = config_path("hands_positions.json")

        self.camera_id     = camera_id
        self.running       = True
        self.TABLE_SIZE_MM = 597.0
        self.GRID_SIZE     = 700

        self.hand_buffer  = hand_buffer  or HandStateBuffer(max_age_ms=200)
        self.frame_buffer = frame_buffer or FrameBuffer(max_age_ms=200)  # ← NOUVEAU

        self._slot_manager = HandSlotManager(
            process_noise=kalman_process_noise,
            measure_noise=kalman_measure_noise,
        )

        import threading
        self._result_lock   = threading.Lock()
        self._latest_result = None

        self.rvec, self.tvec, self.K = self._load_pose()

        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._latest_result = (result, timestamp_ms)

    def _load_pose(self):
        data = json.load(open(self.POSE_PATH))
        return (
            np.array(data["rvec"], dtype=np.float64),
            np.array(data["tvec"], dtype=np.float64),
            np.array(data["camera_matrix"], dtype=np.float64),
        )

    def _pixel_to_ray(self, u, v):
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        return ray / np.linalg.norm(ray)

    def _intersect_ray_plane(self, ray):
        R, _         = cv2.Rodrigues(self.rvec)
        normal       = R[:, 2]
        plane_origin = self.tvec.reshape(3)
        denom = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        return (t * ray) if t >= 0 else None

    def _camera_to_table(self, pt_cam):
        R, _ = cv2.Rodrigues(self.rvec)
        pt   = R.T @ (pt_cam - self.tvec.reshape(3))
        return pt[0], pt[1]

    def _pixel_to_table(self, u, v):
        ray    = self._pixel_to_ray(u, v)
        pt_cam = self._intersect_ray_plane(ray)
        return self._camera_to_table(pt_cam) if pt_cam is not None else None

    def _flip_y(self, x_mm, y_mm):
        return x_mm, self.TABLE_SIZE_MM - y_mm

    def _mm_to_graph(self, x_mm, y_mm):
        return (
            (x_mm / self.TABLE_SIZE_MM) * self.GRID_SIZE,
            (y_mm / self.TABLE_SIZE_MM) * self.GRID_SIZE,
        )

    def _get_grip_point(self, hand, w, h):
        IDX     = [(5, 1.2), (9, 1.5), (13, 1.2), (17, 0.8), (0, 0.8)]
        total_w = sum(wi for _, wi in IDX)
        x_sum = y_sum = 0.0
        for lm_idx, wi in IDX:
            lm     = hand[lm_idx]
            x_sum += lm.x * w * wi
            y_sum += lm.y * h * wi
        return np.array([x_sum / total_w, y_sum / total_w], dtype=np.float32)

    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        print(f"[HAND THREAD] Camera {self.camera_id} — v4 FrameBuffer active")
        if not cap.isOpened():
            print("[HAND THREAD ERROR] Camera not opened")
            return

        self._fps_counter_top = 0
        self._fps_time_top    = pytime.time()
        _last_result_ts       = -1

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── NOUVEAU : pousser la frame brute dans le FrameBuffer ──────
            # Algorithm_Analysis en a besoin pour le tracker CSRT.
            # On pousse AVANT la détection MediaPipe pour minimiser la latence.
            self.frame_buffer.push(frame)
            # ─────────────────────────────────────────────────────────────

            h, w, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(pytime.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.detector.detect_async(mp_image, timestamp_ms)

            with self._result_lock:
                result_payload = self._latest_result

            raw_detections: list[tuple[float, float]] = []

            if result_payload is not None:
                result, result_ts = result_payload

                if result_ts != _last_result_ts:
                    _last_result_ts = result_ts

                    if result and result.hand_landmarks:
                        for hand in result.hand_landmarks:
                            grip = self._get_grip_point(hand, w, h)
                            res  = self._pixel_to_table(grip[0], grip[1])
                            if res is None:
                                continue
                            x_mm, y_mm = self._flip_y(*res)
                            xg, yg     = self._mm_to_graph(x_mm, y_mm)
                            raw_detections.append((xg, yg))

                    hands_data = self._slot_manager.update(raw_detections)
                    for h_entry in hands_data:
                        h_entry["ts_ms"] = result_ts

                    # Écriture dans HandStateBuffer (pour la future feature dessin)
                    self.hand_buffer.push(hands_data)
                    self.hands_signal.emit(hands_data)

            self._fps_counter_top += 1
            now = pytime.time()
            if now - self._fps_time_top >= 1.0:
                print(f"[FPS] cam_top = {self._fps_counter_top} | "
                      f"frame_buffer_age = {self.frame_buffer.last_age_ms} ms")
                self._fps_counter_top = 0
                self._fps_time_top    = now

            self.frame_signal.emit(frame)

        cap.release()
        print("[HAND THREAD] stopped")
