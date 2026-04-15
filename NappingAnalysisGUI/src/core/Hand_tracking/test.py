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

POSE_PATH = Path("config/camtop_table_pose.json")

def load_pose():
    data = json.load(open(POSE_PATH))

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    K    = np.array(data["camera_matrix"], dtype=np.float64)

    return rvec, tvec, K

rvec, tvec, K = load_pose()
# =========================================================
# OUTILS GEO
# =========================================================

def pixel_to_ray(u, v, K):
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    return ray / np.linalg.norm(ray)

def intersect_ray_plane(ray, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]
    plane_origin = tvec.reshape(3)

    denom = np.dot(normal, ray)
    if abs(denom) < 1e-9:
        return None

    t = np.dot(normal, plane_origin) / denom
    if t < 0:
        return None

    return t * ray

def camera_to_table(pt_cam, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    pt = R.T @ (pt_cam - tvec.reshape(3))
    return pt[0], pt[1]

def pixel_to_table(u, v, rvec, tvec, K):
    ray = pixel_to_ray(u, v, K)
    pt_cam = intersect_ray_plane(ray, rvec, tvec)
    if pt_cam is None:
        return None
    return camera_to_table(pt_cam, rvec, tvec)
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


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

                res = pixel_to_table(pt[0], pt[1], rvec, tvec, K)

                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, (0,255,0), -1)

                if res is not None:
                    x_mm, y_mm = res

                    print(f"Main {hand_idx+1}, landmark {lm_idx}")
                    print(f"  Cam: ({x}, {y})")
                    print(f"  Table: ({int(x_mm)}, {int(y_mm)})")

    cv2.imshow("Hand Tracking → Table", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()