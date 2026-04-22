# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMS
# =========================================================
CAM_TOP_ID = 1
CAM_BOTTOM_ID = 0
TEST_TAG_ID = 7

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

POSE_TOP_PATH = CONFIG_DIR / "camtop_table_pose.json"
POSE_BOTTOM_PATH = CONFIG_DIR / "cambottom_table_pose.json"

CALIB_TOP_PATH = CONFIG_DIR / "camera_calibration_top.json"
CALIB_BOTTOM_PATH = CONFIG_DIR / "camera_calibration.json"

# =========================================================
# LOAD
# =========================================================
def load_pose(path):
    data = json.load(open(path))
    return (
        np.array(data["rvec"], dtype=np.float64),
        np.array(data["tvec"], dtype=np.float64),
        np.array(data["camera_matrix"], dtype=np.float64)
    )

def load_calib(path):
    data = json.load(open(path))
    return (
        np.array(data["camera_matrix"], dtype=np.float64),
        np.array(data["dist_coeffs"], dtype=np.float64)
    )

# =========================================================
# GEOMETRY
# =========================================================
def pixel_to_ray(u, v, K):
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    return ray / np.linalg.norm(ray)

def intersect(ray, rvec, tvec):
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

def cam_to_table(pt, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    pt_table = R.T @ (pt - tvec.reshape(3))
    return float(pt_table[0]), float(pt_table[1])

def pixel_to_table(u, v, rvec, tvec, K):
    ray = pixel_to_ray(u, v, K)
    pt = intersect(ray, rvec, tvec)
    if pt is None:
        return None
    return cam_to_table(pt, rvec, tvec)

# =========================================================
# UTIL
# =========================================================
def get_center(pts):
    return np.mean(pts, axis=0)

# =========================================================
# INIT
# =========================================================
rvec_top, tvec_top, newK_top = load_pose(POSE_TOP_PATH)
rvec_bot, tvec_bot, newK_bot = load_pose(POSE_BOTTOM_PATH)

K_top, dist_top = load_calib(CALIB_TOP_PATH)
K_bot, dist_bot = load_calib(CALIB_BOTTOM_PATH)

cap_top = cv2.VideoCapture(CAM_TOP_ID)
cap_bot = cv2.VideoCapture(CAM_BOTTOM_ID)

cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

print("\n[INFO] Comparaison temps réel TOP vs BOTTOM (tag 7)\n")

# =========================================================
# LOOP
# =========================================================
while True:

    ret1, frame_top = cap_top.read()
    ret2, frame_bot = cap_bot.read()

    if not ret1 or not ret2:
        break

    # UNDISTORT
    undist_top = cv2.undistort(frame_top, K_top, dist_top, None, newK_top)
    undist_bot = cv2.undistort(frame_bot, K_bot, dist_bot, None, newK_bot)

    gray_top = cv2.cvtColor(undist_top, cv2.COLOR_BGR2GRAY)
    gray_bot = cv2.cvtColor(undist_bot, cv2.COLOR_BGR2GRAY)

    corners_t, ids_t, _ = detector.detectMarkers(gray_top)
    corners_b, ids_b, _ = detector.detectMarkers(gray_bot)

    pos_top = None
    pos_bot = None

    # -------- TOP --------
    if ids_t is not None:
        for i, mid in enumerate(ids_t.flatten()):
            if mid == TEST_TAG_ID:
                cx, cy = get_center(corners_t[i][0])
                pos_top = pixel_to_table(cx, cy, rvec_top, tvec_top, newK_top)
                cv2.circle(undist_top, (int(cx), int(cy)), 6, (0,0,255), -1)

    # -------- BOTTOM --------
    if ids_b is not None:
        for i, mid in enumerate(ids_b.flatten()):
            if mid == TEST_TAG_ID:
                cx, cy = get_center(corners_b[i][0])
                pos_bot = pixel_to_table(cx, cy, rvec_bot, tvec_bot, newK_bot)
                cv2.circle(undist_bot, (int(cx), int(cy)), 6, (0,0,255), -1)

    # -------- AFFICHAGE --------
    if pos_top is not None:
        cv2.putText(undist_top,
            f"TOP: X={pos_top[0]:.1f} Y={pos_top[1]:.1f}",
            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if pos_bot is not None:
        cv2.putText(undist_bot,
            f"BOT: X={pos_bot[0]:.1f} Y={pos_bot[1]:.1f}",
            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if pos_top is not None and pos_bot is not None:
        dx = pos_bot[0] - pos_top[0]
        dy = pos_bot[1] - pos_top[1]

        print(f"TOP {pos_top} | BOT {pos_bot} | Δ=({dx:.2f}, {dy:.2f})")

    cv2.imshow("CAM TOP", undist_top)
    cv2.imshow("CAM BOTTOM", undist_bot)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_top.release()
cap_bot.release()
cv2.destroyAllWindows()