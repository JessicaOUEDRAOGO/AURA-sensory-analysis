# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path
import json

# =========================================================
# IMPORTS (TES SCRIPTS)
# =========================================================
from src.core.calibration.pose_cam_bottom_V2 import (
    get_marker_centers,
    pixel_to_table_mm,
    load_camera_calibration as load_cam1_calib,
)

from src.core.calibration.pose_top_cam_pixel_to_mm import (
    pixel_to_table_mm as pixel_to_table_mm_top,
    load_camera_calibration as load_cam2_calib,
)

# =========================================================
# PARAMÈTRES
# =========================================================
CAM1_ID = 0
CAM2_ID = 1

GRID_SIZE = 700
CALIB_TAGS = {41, 40, 43, 42}

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# CAM1
CAM1_CALIB = CONFIG_DIR / "camera_calibration_fisheye.json"
CAM1_POSE  = CONFIG_DIR / "cambottom_table_pose.json"

# CAM2
CAM2_CALIB = CONFIG_DIR / "camera_calibration_top.json"
CAM2_POSE  = CONFIG_DIR / "camtop_table_pose.json"


# =========================================================
# LOAD POSES
# =========================================================
def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)

    return (
        np.array(data["rvec"], dtype=np.float64),
        np.array(data["tvec"], dtype=np.float64),
        float(data["table_size_mm"])
    )


# =========================================================
# UTILS
# =========================================================
def mm_to_graph(x, y, table_size):
    return (x / table_size) * GRID_SIZE, (y / table_size) * GRID_SIZE


def create_grid():
    img = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) * 255
    step = 100

    for i in range(0, GRID_SIZE, step):
        cv2.line(img, (i, 0), (i, GRID_SIZE), (200, 200, 200), 1)
        cv2.line(img, (0, i), (GRID_SIZE, i), (200, 200, 200), 1)

    return img


# =========================================================
# MAIN
# =========================================================
def main():

    # ================= CAM1 (BOTTOM) =================
    K1, D1 = load_cam1_calib(CAM1_CALIB)
    rvec1, tvec1, table_size = load_pose(CAM1_POSE)

    cap1 = cv2.VideoCapture(CAM1_ID)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Undistortion fisheye
    ret, frame = cap1.read()
    h, w = frame.shape[:2]

    new_K1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K1, D1, (w, h), np.eye(3), balance=0.3
    )
    map1_1, map1_2 = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, np.eye(3), new_K1, (w, h), cv2.CV_16SC2
    )

    # ================= CAM2 (TOP) =================
    K2, D2 = load_cam2_calib(CAM2_CALIB)
    rvec2, tvec2, _ = load_pose(CAM2_POSE)

    cap2 = cv2.VideoCapture(CAM2_ID)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    new_K2, _ = cv2.getOptimalNewCameraMatrix(K2, D2, (w, h), 1, (w, h))

    # ================= ARUCO =================
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Comparaison CAM1 vs CAM2")

    while True:

        grid = create_grid()

        # ================= CAM1 =================
        ret1, frame1 = cap1.read()
        graph_cam1 = {}

        if ret1:
            frame1 = cv2.remap(frame1, map1_1, map1_2, cv2.INTER_LINEAR)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            corners1, ids1, _ = detector.detectMarkers(gray1)

            if ids1 is not None:
                centers1 = get_marker_centers(corners1, ids1)

                for marker_id in ids1.flatten():
                    marker_id = int(marker_id)

                    if marker_id in CALIB_TAGS:
                        continue

                    u, v = centers1[marker_id]

                    result = pixel_to_table_mm(u, v, rvec1, tvec1, new_K1)
                    if result is None:
                        continue

                    x_mm, y_mm = result
                    xg, yg = mm_to_graph(x_mm, y_mm, table_size)

                    graph_cam1[marker_id] = (xg, yg)

        # ================= CAM2 =================
        ret2, frame2 = cap2.read()
        graph_cam2 = {}

        if ret2:
            frame2 = cv2.undistort(frame2, K2, D2, None, new_K2)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            corners2, ids2, _ = detector.detectMarkers(gray2)

            if ids2 is not None:
                centers2 = get_marker_centers(corners2, ids2)

                for marker_id in ids2.flatten():
                    marker_id = int(marker_id)

                    if marker_id in CALIB_TAGS:
                        continue

                    u, v = centers2[marker_id]

                    result = pixel_to_table_mm_top(u, v, rvec2, tvec2, new_K2)
                    if result is None:
                        continue

                    x_mm, y_mm = result

                    # IMPORTANT : flip pour aligner avec cam1
                    y_mm = table_size - y_mm

                    xg, yg = mm_to_graph(x_mm, y_mm, table_size)

                    graph_cam2[marker_id] = (xg, yg)

        # ================= COMPARAISON =================
        all_ids = set(graph_cam1.keys()) | set(graph_cam2.keys())

        for marker_id in all_ids:

            if marker_id in graph_cam1:
                x1, y1 = graph_cam1[marker_id]
                cv2.circle(grid, (int(x1), int(y1)), 8, (0, 0, 255), -1)

            if marker_id in graph_cam2:
                x2, y2 = graph_cam2[marker_id]
                cv2.circle(grid, (int(x2), int(y2)), 8, (255, 0, 0), -1)

            if marker_id in graph_cam1 and marker_id in graph_cam2:
                dx = x1 - x2
                dy = y1 - y2

                print(f"ID {marker_id} | Δ=({dx:.2f}, {dy:.2f})")

        cv2.imshow("CAM1 vs CAM2", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()