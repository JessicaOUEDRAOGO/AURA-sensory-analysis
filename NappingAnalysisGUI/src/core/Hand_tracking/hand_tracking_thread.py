# -*- coding: utf-8 -*-
"""
hand_tracking_thread.py — VERSION CORRIGÉE
===========================================
Corrections appliquées :
  - Bug 1 : suppression du filtre ts_ms stale qui ignorait les mains à tort
  - Bug 3 : extrapolation bornée avec decay exponentiel en carrier lock
  - Kalman : velocity clampée, predict_only() cohérent avec update()
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


# ---------------------------------------------------------------------------
# Filtre de Kalman 2D — position + vitesse
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

        # Borne de vélocité : évite les spikes au démarrage ou lors d'occlusions
        self._vel_clip = 80.0

    def update(self, x: float, y: float) -> np.ndarray:
        self.kf.predict()
        c = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return np.array([c[0, 0], c[1, 0]], dtype=np.float32)

    def predict_only(self) -> np.ndarray:
        """Avance le modèle sans mesure — utilisé pendant les frames manquantes."""
        p = self.kf.predict()
        # Clipper la vélocité dans statePost après chaque prédiction libre
        # pour éviter la divergence sur MAX_MISSING_FRAMES consécutives.
        self.kf.statePost[2, 0] = float(np.clip(self.kf.statePost[2, 0],
                                                 -self._vel_clip, self._vel_clip))
        self.kf.statePost[3, 0] = float(np.clip(self.kf.statePost[3, 0],
                                                 -self._vel_clip, self._vel_clip))
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
# Gestionnaire de slots de mains
# ---------------------------------------------------------------------------
# Problème résolu : l'ID MediaPipe (index dans la liste) change aléatoirement
# quand une main entre/sort du champ. L'ID basé sur gauche/droite crée un
# saut quand la main traverse x=350.
#
# Solution : on maintient un pool de "slots" (max MAX_HANDS slots actifs).
# Chaque slot a une position Kalman. À chaque frame, on associe les détections
# MediaPipe aux slots existants par distance minimale (Hungarian simplifié).
# Un slot non alimenté pendant MAX_MISSING_FRAMES est supprimé.
# L'ID émis est l'index du slot → stable tant que la main reste dans le champ.
# ---------------------------------------------------------------------------
class HandSlotManager:
    MAX_HANDS          = 2
    MAX_MISSING_FRAMES = 8    # frames avant suppression du slot
    MATCH_THRESHOLD    = 150  # pixels graphe — au-delà on crée un nouveau slot

    def __init__(self, process_noise: float, measure_noise: float):
        self._process_noise = process_noise
        self._measure_noise = measure_noise
        # slot_id → { "kf": KalmanHand, "missing": int }
        self._slots: dict[int, dict] = {}
        self._next_id = 0

    def update(self, raw_detections: list[tuple[float, float]]) -> list[dict]:
        """
        raw_detections : liste de (xg, yg) mesurés ce frame.
        Retourne la liste des slots mis à jour (ou prédits).

        Correction Bug 1 appliquée ici : pas de filtre ts_ms côté slot,
        le timestamp est simplement propagé tel quel depuis le thread.
        """
        unmatched_detections = list(range(len(raw_detections)))
        matched_slots        = set()

        if self._slots and raw_detections:
            slot_ids = list(self._slots.keys())
            costs = np.full((len(slot_ids), len(raw_detections)), np.inf)

            for si, sid in enumerate(slot_ids):
                slot_pos = self._slots[sid]["kf"].position
                for di, (xg, yg) in enumerate(raw_detections):
                    det_pos = np.array([xg, yg], dtype=np.float32)
                    costs[si, di] = float(np.linalg.norm(slot_pos - det_pos))

            # Greedy matching (suffisant pour MAX_HANDS = 2)
            while True:
                if costs.size == 0:
                    break
                si, di = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[si, di] > self.MATCH_THRESHOLD:
                    break
                sid = slot_ids[si]
                xg, yg = raw_detections[di]
                self._slots[sid]["kf"].update(xg, yg)
                self._slots[sid]["missing"] = 0
                matched_slots.add(sid)
                unmatched_detections.remove(di)
                costs[si, :]  = np.inf
                costs[:, di]  = np.inf

        # Créer un slot pour chaque détection non associée
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

        # Prédire / supprimer les slots non alimentés
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

        # Construire la sortie
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
    """
    Thread de suivi des mains (caméra top).

    Chaque entrée émise dans hands_signal :
        {
            "id":      int,    ID de slot stable
            "x":       float,  position filtrée Kalman en pixels graphe
            "y":       float,
            "vx":      float,  vélocité Kalman (pixels graphe / frame top)
            "vy":      float,
            "missing": int,    0 si mesure réelle, >0 si prédiction
            "ts_ms":   int,    timestamp monotone du thread (toujours présent)
        }
    """
    frame_signal = pyqtSignal(object)
    hands_signal = pyqtSignal(object)

    def __init__(self, camera_id: int = 0,
                 kalman_process_noise: float = 5e-2,
                 kalman_measure_noise: float = 3.0):
        super().__init__()

        BASE_DIR = Path(__file__).resolve().parent
        self.MODEL_PATH  = BASE_DIR / "hand_landmarker.task"
        self.POSE_PATH   = Path("config/camtop_table_pose.json")
        self.OUTPUT_PATH = config_path("hands_positions.json")

        self.camera_id     = camera_id
        self.running       = True
        self.TABLE_SIZE_MM = 597.0
        self.GRID_SIZE     = 700

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

    # --- Callback MediaPipe ---
    def _on_result(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._latest_result = result

    # --- Calibration ---
    def _load_pose(self):
        data = json.load(open(self.POSE_PATH))
        return (
            np.array(data["rvec"], dtype=np.float64),
            np.array(data["tvec"], dtype=np.float64),
            np.array(data["camera_matrix"], dtype=np.float64),
        )

    # --- Géométrie ---
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

    # --- Point de grip (centroïde MCP + palme, plus stable que pouce+index) ---
    def _get_grip_point(self, hand, w, h):
        IDX     = [(5, 1.2), (9, 1.5), (13, 1.2), (17, 0.8), (0, 0.8)]
        total_w = sum(wi for _, wi in IDX)
        x_sum = y_sum = 0.0
        for lm_idx, wi in IDX:
            lm     = hand[lm_idx]
            x_sum += lm.x * w * wi
            y_sum += lm.y * h * wi
        return np.array([x_sum / total_w, y_sum / total_w], dtype=np.float32)

    # --- Boucle principale ---
    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        print(f"[HAND THREAD] Camera {self.camera_id} – SlotManager + Kalman active")
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

            timestamp_ms += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.detector.detect_async(mp_image, timestamp_ms)

            with self._result_lock:
                result = self._latest_result

            raw_detections: list[tuple[float, float]] = []
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

            # ts_ms toujours présent et incrémental — Bug 1 fix :
            # on ne filtre JAMAIS côté consommateur sur ce timestamp.
            for h_entry in hands_data:
                h_entry["ts_ms"] = timestamp_ms

            self.hands_signal.emit(hands_data)
            self.frame_signal.emit(frame)

            pytime.sleep(0.001)

        cap.release()
        print("[HAND THREAD] stopped")
