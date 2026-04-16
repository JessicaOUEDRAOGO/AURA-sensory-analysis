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

TABLE_SIZE_MM = 580.0
GRID_SIZE = 700

def get_grip_point(hand, w, h):
    # landmarks utiles
    p4 = hand[4]   # pouce
    p8 = hand[8]   # index

    pt4 = np.array([p4.x * w, p4.y * h])
    pt8 = np.array([p8.x * w, p8.y * h])

    grip = (pt4 + pt8) / 2
    return grip


def flip_y(x_mm, y_mm):
    return x_mm, TABLE_SIZE_MM - y_mm


def mm_to_graph(x_mm, y_mm):
    xg = (x_mm / TABLE_SIZE_MM) * GRID_SIZE
    yg = (y_mm / TABLE_SIZE_MM) * GRID_SIZE
    return xg, yg
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


cap = cv2.VideoCapture(1)

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

            # =====================================================
            # 1. POINT DE PRÉHENSION
            # =====================================================
            grip = get_grip_point(hand, w, h)
            u, v = grip

            cv2.circle(frame, (int(u), int(v)), 8, (0,0,255), -1)

            # =====================================================
            # 2. PIXEL → MM
            # =====================================================
            res = pixel_to_table(u, v, rvec, tvec, K)

            if res is None:
                continue

            x_mm, y_mm = res

            # =====================================================
            # 3. ALIGNEMENT avec CAM1
            # =====================================================
            x_mm, y_mm = flip_y(x_mm, y_mm)

            # =====================================================
            # 4. CONVERSION GRAPH
            # =====================================================
            xg, yg = mm_to_graph(x_mm, y_mm)

            # =====================================================
            # DEBUG
            # =====================================================
            print(f"\nMain {hand_idx+1}")
            print(f"  Pixel: ({int(u)}, {int(v)})")
            print(f"  Table mm: ({int(x_mm)}, {int(y_mm)})")
            print(f"  Graph: ({int(xg)}, {int(yg)})")

            cv2.putText(frame,
                f"G({int(xg)},{int(yg)})",
                (int(u)+10, int(v)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,0,255), 2)

    cv2.imshow("Hand Tracking → Table", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()