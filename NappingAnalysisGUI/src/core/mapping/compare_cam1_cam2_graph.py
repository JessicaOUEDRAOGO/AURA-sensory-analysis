# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path
import json

# =========================================================
# IMPORTS CAM1 (runtime existant)
# =========================================================
from src.core.mapping.coordinate_mapper import CoordinateMapper

# =========================================================
# IMPORTS CAM2 (ton script)
# =========================================================
from src.core.calibration.pose_top_cam_pixel_to_mm import (
    get_marker_centers,
    pixel_to_table_mm,
    load_camera_calibration,
)

# =========================================================
# PARAMÈTRES
# =========================================================
CAM1_ID = 0   # caméra bottom
CAM2_ID = 1   # caméra top

GRID_SIZE = 700
TABLE_SIZE_MM = 580.0

CALIB_TAGS = {41, 40, 43, 42}

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAM2_CALIB = CONFIG_DIR / "camera_calibration_top.json"
POSE_PATH = CONFIG_DIR / "camtop_table_pose.json"

# =========================================================
# UTILS CAM2
# =========================================================
def load_top_pose(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["rvec"]), np.array(data["tvec"])

def flip_y(x, y):
    return x, TABLE_SIZE_MM - y

def mm_to_graph(x, y):
    return (x / TABLE_SIZE_MM) * GRID_SIZE, (y / TABLE_SIZE_MM) * GRID_SIZE


# =========================================================
# GRILLE VISUELLE
# =========================================================
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

    # -------- CAM1 (bottom)
    mapper = CoordinateMapper()
    mapper.load()

    cap1 = cv2.VideoCapture(CAM1_ID)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # -------- CAM2 (top)
    K2, dist2 = load_camera_calibration(CAM2_CALIB)
    rvec, tvec = load_top_pose(POSE_PATH)

    cap2 = cv2.VideoCapture(CAM2_ID)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # new_K cam2
    ret, frame = cap2.read()
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K2, dist2, (w, h), 1, (w, h))

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Appuie sur q pour quitter")

    while True:

        grid = create_grid()

        # =====================================================
        # CAM1
        # =====================================================
        ret1, frame1 = cap1.read()
        graph_cam1 = {}

        if ret1:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            corners1, ids1, _ = detector.detectMarkers(gray1)

            if ids1 is not None:
                centers1 = get_marker_centers(corners1, ids1)

                for marker_id in ids1.flatten():
                    marker_id = int(marker_id)

                    if marker_id in CALIB_TAGS:
                        continue

                    u, v = centers1[marker_id]

                    graph = mapper.camera_raw_to_graph(np.array([u, v]))
                    graph_cam1[marker_id] = graph

        # =====================================================
        # CAM2
        # =====================================================
        ret2, frame2 = cap2.read()
        graph_cam2 = {}

        if ret2:
            frame2_undist = cv2.undistort(frame2, K2, dist2, None, new_K)

            gray2 = cv2.cvtColor(frame2_undist, cv2.COLOR_BGR2GRAY)
            corners2, ids2, _ = detector.detectMarkers(gray2)

            if ids2 is not None:
                centers2 = get_marker_centers(corners2, ids2)

                for marker_id in ids2.flatten():
                    marker_id = int(marker_id)

                    if marker_id in CALIB_TAGS:
                        continue

                    u, v = centers2[marker_id]

                    result = pixel_to_table_mm(u, v, rvec, tvec, new_K)
                    if result is None:
                        continue

                    x_mm, y_mm = result
                    x_mm, y_mm = flip_y(x_mm, y_mm)

                    xg, yg = mm_to_graph(x_mm, y_mm)

                    graph_cam2[marker_id] = (xg, yg)

        # =====================================================
        # COMPARAISON + AFFICHAGE
        # =====================================================
        all_ids = set(graph_cam1.keys()) | set(graph_cam2.keys())

        for marker_id in all_ids:

            if marker_id in graph_cam1:
                x1, y1 = graph_cam1[marker_id]
                cv2.circle(grid, (int(x1), int(y1)), 8, (0, 0, 255), -1)  # rouge

            if marker_id in graph_cam2:
                x2, y2 = graph_cam2[marker_id]
                cv2.circle(grid, (int(x2), int(y2)), 8, (255, 0, 0), -1)  # bleu

            if marker_id in graph_cam1 and marker_id in graph_cam2:
                dx = x1 - x2
                dy = y1 - y2

                print(f"ID {marker_id} | cam1=({x1:.1f},{y1:.1f}) cam2=({x2:.1f},{y2:.1f}) Δ=({dx:.1f},{dy:.1f})")

        cv2.imshow("Comparaison CAM1 vs CAM2 (graph 700x700)", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()