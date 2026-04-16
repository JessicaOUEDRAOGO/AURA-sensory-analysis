# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

# =========================================================
# IMPORT DES FONCTIONS EXISTANTES (TON SCRIPT)
# =========================================================
from src.core.calibration.pose_top_cam_pixel_to_mm import (
    get_marker_centers,
    pixel_to_table_mm,
    load_camera_calibration,
)

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0

GRID_SIZE = 700
TABLE_SIZE_MM = 580.0

CALIBRATION_TAG_IDS = {41, 40, 43, 42}

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration_top.json"
POSE_PATH = CONFIG_DIR / "camtop_table_pose.json"


# =========================================================
# CHARGEMENT POSE
# =========================================================
def load_top_pose(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)

    return rvec, tvec


# =========================================================
# TRANSFORMATIONS
# =========================================================
def flip_y_top_to_bottom(x_mm, y_mm):
    """
    Aligne repère cam2 (top) vers repère cam1 (bottom)
    """
    return x_mm, TABLE_SIZE_MM - y_mm


def mm_to_graph(x_mm, y_mm):
    xg = (x_mm / TABLE_SIZE_MM) * GRID_SIZE
    yg = (y_mm / TABLE_SIZE_MM) * GRID_SIZE
    return xg, yg


# =========================================================
# MAIN
# =========================================================
def main():

    # --- calibration caméra top (origine)
    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)

    # --- ouvrir caméra
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] caméra non ouverte")
        return

    # --- lire une frame pour new_K
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] lecture frame")
        return

    h, w = frame.shape[:2]

    # IMPORTANT : new_K (cohérent avec ton script original)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    print("[INFO] new_K utilisé")

    # --- charger pose
    rvec, tvec = load_top_pose(POSE_PATH)

    # --- ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] détection en cours... (q pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =====================================================
        # 1. UNDISTORT (CRUCIAL)
        # =====================================================
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)

        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()

        if ids is not None and len(ids) > 0:

            centers = get_marker_centers(corners, ids)

            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for marker_id in ids.flatten():

                marker_id = int(marker_id)

                # ignorer tags de calibration
                if marker_id in CALIBRATION_TAG_IDS:
                    continue

                if marker_id not in centers:
                    continue

                u, v = centers[marker_id]

                # =====================================================
                # 2. PIXEL -> MM (REPÈRE TOP)
                # =====================================================
                result = pixel_to_table_mm(u, v, rvec, tvec, new_K)

                if result is None:
                    continue

                x_top, y_top = result

                # =====================================================
                # 3. FLIP Y (REPÈRE CAM1)
                # =====================================================
                x_bottom, y_bottom = flip_y_top_to_bottom(x_top, y_top)

                # =====================================================
                # 4. MM -> GRAPH 700x700
                # =====================================================
                xg, yg = mm_to_graph(x_bottom, y_bottom)

                # =====================================================
                # DEBUG VISUEL
                # =====================================================
                cv2.circle(display, (int(u), int(v)), 6, (0, 0, 255), -1)

                txt = f"ID {marker_id}"
                txt2 = f"graph=({xg:.1f},{yg:.1f})"

                cv2.putText(display, txt,
                            (int(u)+10, int(v)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

                cv2.putText(display, txt2,
                            (int(u)+10, int(v)+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,165,255), 1)

                print(f"ID {marker_id} -> graph ({xg:.2f}, {yg:.2f})")

        cv2.imshow("Top -> Graph 700x700", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()