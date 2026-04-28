# -*- coding: utf-8 -*-
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
# Filtre de Kalman 2D pour le suivi d'une main
# ---------------------------------------------------------------------------
# Modélise l'état [x, y, vx, vy] — position + vitesse en pixels graphe.
#
# Avantages vs la moyenne glissante adaptative précédente :
#   • Sépare automatiquement le bruit de mesure du mouvement réel.
#   • Pas de délai fixe : la fenêtre effective s'adapte à l'incertitude.
#   • Fournit la vélocité estimée directement, sans recalcul dans Algorithm_Analysis.
#   • Peut prédire la position quand MediaPipe manque un frame.
#
# Réglage rapide :
#   process_noise (Q) : augmenter si le suivi décroche en accélération rapide.
#   measure_noise (R) : augmenter si le point reste tremblant.
#   Valeurs de départ raisonnables : Q=1e-2, R=5.0
# ---------------------------------------------------------------------------

class KalmanHand:
    """Filtre de Kalman 2D pour une main (position + vitesse)."""

    def __init__(self, x: float, y: float,
                 process_noise: float = 1e-2,
                 measure_noise: float = 5.0):
        self.kf = cv2.KalmanFilter(4, 2)

        # Modèle vitesse constante (dt = 1 frame)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # On observe uniquement x et y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100.0

        self.kf.statePost = np.array(
            [[x], [y], [0.0], [0.0]], dtype=np.float32
        )

    def update(self, x: float, y: float) -> np.ndarray:
        """Intègre une mesure → retourne position filtrée [x, y]."""
        self.kf.predict()
        corrected = self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        return np.array([corrected[0, 0], corrected[1, 0]], dtype=np.float32)

    def predict_only(self) -> np.ndarray:
        """
        Prédit la position au frame suivant SANS mesure.
        Utile lors des sauts de détection MediaPipe (1–2 frames).
        """
        predicted = self.kf.predict()
        return np.array([predicted[0, 0], predicted[1, 0]], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        """Vélocité courante estimée [vx, vy] en pixels graphe / frame."""
        s = self.kf.statePost
        return np.array([s[2, 0], s[3, 0]], dtype=np.float32)


class HandTrackingThread(QThread):
    """
    Thread de suivi des mains (caméra top).

    Chaque entrée émise dans hands_signal :
        {
            "id":    int,    identifiant MediaPipe (0 ou 1)
            "x":     float,  position filtrée Kalman en pixels graphe
            "y":     float,
            "vx":    float,  vélocité Kalman (pixels graphe / frame)
            "vy":    float,
            "ts_ms": int,    timestamp monotone — permet à Algorithm_Analysis
                             de détecter si la donnée est trop ancienne
        }
    """

    frame_signal = pyqtSignal(object)
    hands_signal = pyqtSignal(object)

    # Frames consécutifs sans détection avant suppression du filtre Kalman.
    # Valeur haute → moins de clignotements sur les occultations brèves.
    MAX_MISSING_FRAMES = 8

    def __init__(self, camera_id: int = 0,
                 kalman_process_noise: float = 1e-2,
                 kalman_measure_noise: float = 5.0):
        super().__init__()

        BASE_DIR = Path(__file__).resolve().parent

        self.MODEL_PATH  = BASE_DIR / "hand_landmarker.task"
        self.POSE_PATH   = Path("config/camtop_table_pose.json")
        self.OUTPUT_PATH = config_path("hands_positions.json")

        self.camera_id = camera_id
        self.running   = True

        self.TABLE_SIZE_MM = 597.0
        self.GRID_SIZE     = 700

        self._kf_process_noise = kalman_process_noise
        self._kf_measure_noise = kalman_measure_noise

        # Un filtre Kalman + compteur de frames manquants par main
        self._kalman_filters: dict[int, KalmanHand] = {}
        self._missing_frames: dict[int, int]        = {}

        # Résultat MediaPipe LIVE_STREAM (thread-safe)
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

    # ======================================================================
    # Callback MediaPipe
    # ======================================================================
    def _on_result(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._latest_result = result

    # ======================================================================
    # Calibration
    # ======================================================================
    def _load_pose(self):
        data = json.load(open(self.POSE_PATH))
        return (
            np.array(data["rvec"], dtype=np.float64),
            np.array(data["tvec"], dtype=np.float64),
            np.array(data["camera_matrix"], dtype=np.float64),
        )

    # ======================================================================
    # Géométrie pixel → espace graphe
    # ======================================================================
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

    # ======================================================================
    # Point de grip (centroïde pondéré MCP + palme)
    # Plus stable que pouce+index lors des mouvements rapides.
    # ======================================================================
    def _get_grip_point(self, hand, w, h):
        IDX     = [(5, 1.2), (9, 1.5), (13, 1.2), (17, 0.8), (0, 0.8)]
        total_w = sum(wi for _, wi in IDX)
        x_sum = y_sum = 0.0
        for lm_idx, wi in IDX:
            lm     = hand[lm_idx]
            x_sum += lm.x * w * wi
            y_sum += lm.y * h * wi
        return np.array([x_sum / total_w, y_sum / total_w], dtype=np.float32)

    # ======================================================================
    # Gestion des filtres Kalman
    # ======================================================================
    def _get_or_create_kalman(self, hand_id: int, x: float, y: float) -> KalmanHand:
        if hand_id not in self._kalman_filters:
            self._kalman_filters[hand_id] = KalmanHand(
                x, y,
                process_noise=self._kf_process_noise,
                measure_noise=self._kf_measure_noise,
            )
            self._missing_frames[hand_id] = 0
        return self._kalman_filters[hand_id]

    def _handle_missing_hands(self, detected_ids: set) -> list:
        """
        Pour chaque main non détectée ce frame :
          - si <= 2 frames manquants → prédiction Kalman (continuité)
          - au-delà de MAX_MISSING_FRAMES → suppression du filtre
        Retourne la liste des entrées prédites à ajouter à hands_data.
        """
        predicted_entries = []
        for hid in list(self._kalman_filters.keys()):
            if hid in detected_ids:
                continue
            self._missing_frames[hid] = self._missing_frames.get(hid, 0) + 1
            n_miss = self._missing_frames[hid]

            if n_miss <= 2:
                # Court saut de détection → on prédit pour éviter le clignotement
                kf   = self._kalman_filters[hid]
                pred = kf.predict_only()
                vel  = kf.velocity
                predicted_entries.append({
                    "id":        hid,
                    "x":         float(pred[0]),
                    "y":         float(pred[1]),
                    "vx":        float(vel[0]),
                    "vy":        float(vel[1]),
                    "ts_ms":     -1,          # -1 = position prédite, pas mesurée
                    "predicted": True,
                })
            elif n_miss > self.MAX_MISSING_FRAMES:
                del self._kalman_filters[hid]
                del self._missing_frames[hid]

        return predicted_entries

    # ======================================================================
    # Boucle principale
    # ======================================================================
    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)    # latence minimale

        print(
            f"[HAND THREAD] Camera {self.camera_id} – Kalman "
            f"(Q={self._kf_process_noise}, R={self._kf_measure_noise})"
        )
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

            hands_data   = []
            detected_ids = set()

            if result and result.hand_landmarks:
                for hand_idx, hand in enumerate(result.hand_landmarks):
                    grip = self._get_grip_point(hand, w, h)
                    res  = self._pixel_to_table(grip[0], grip[1])
                    if res is None:
                        continue

                    x_mm, y_mm = self._flip_y(*res)
                    xg, yg     = self._mm_to_graph(x_mm, y_mm)

                    kf       = self._get_or_create_kalman(hand_idx, xg, yg)
                    filtered = kf.update(xg, yg)
                    vel      = kf.velocity

                    detected_ids.add(hand_idx)
                    self._missing_frames[hand_idx] = 0

                    hands_data.append({
                        "id":    hand_idx,
                        "x":     float(filtered[0]),
                        "y":     float(filtered[1]),
                        "vx":    float(vel[0]),
                        "vy":    float(vel[1]),
                        "ts_ms": timestamp_ms,
                    })

            # Prédiction pour les mains brièvement perdues
            hands_data.extend(self._handle_missing_hands(detected_ids))

            self.hands_signal.emit(hands_data)
            self.frame_signal.emit(frame)

            pytime.sleep(0.001)

        cap.release()
        print("[HAND THREAD] stopped")