# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hand_landmarker.task"
CALIB_PATH = Path("config/camera_calibration_top.json")
H_PATH = Path("config/camera_top_mapping.json")

# =========================================================
# CHARGEMENT CALIBRATION
# =========================================================
def load_camera_top_calibration():
    data = json.load(open(CALIB_PATH))

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    return K, dist

def load_H_top():
    data = json.load(open(H_PATH))
    return np.array(data["H_top_to_table"], dtype=np.float64)

# =========================================================
# OUTILS GEO
# =========================================================
def undistort_points(pts, K, dist):
    pts = pts.reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1,2)

def apply_H(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    out = H @ p
    out /= out[2]
    return out[:2]

# =========================================================
# MEDIAPIPE
# =========================================================
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# =========================================================
# INIT
# =========================================================
K, dist = load_camera_top_calibration()
H_top_to_table = load_H_top()

cap = cv2.VideoCapture(0)

# =========================================================
# LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_idx, hand in enumerate(result.hand_landmarks):
            for lm_idx, lm in enumerate(hand):

                # coordonnées pixel caméra
                pt = np.array([
                    lm.x * frame.shape[1],
                    lm.y * frame.shape[0]
                ], dtype=np.float32)

                # 1. undistortion
                # pt_undist = undistort_points(pt.reshape(1,2), K, dist)[0]

                # 2. projection table
                pt_table = apply_H(H_top_to_table, pt)

                # affichage caméra
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, (0,255,0), -1)

                # affichage coordonnées table
                print(f"Point détecté sur la main {hand_idx + 1}, landmark {lm_idx}:")
                print(f"  Coordonnées caméra: ({x}, {y})")
                print(f"  Coordonnées table: ({int(pt_table[0])}, {int(pt_table[1])})")

    cv2.imshow("Hand Tracking → Table", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()