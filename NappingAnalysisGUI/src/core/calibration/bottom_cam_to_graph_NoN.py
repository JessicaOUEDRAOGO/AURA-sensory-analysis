# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

# =========================================================
# IMPORT DES FONCTIONS EXISTANTES
# =========================================================
from src.core.calibration.pose_cam_bottom_V2 import (
    get_marker_centers,
    pixel_to_table_mm,
    load_camera_calibration,
)

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0

# Taille du repère graphique de sortie (pixels).
# NE PAS confondre avec TABLE_SIZE_MM : c'est indépendant.
GRID_SIZE = 700

# IDs des tags de calibration à ignorer lors de la détection
CALIBRATION_TAG_IDS = {42, 43, 40, 41}

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR   = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration_fisheye.json"
POSE_PATH         = CONFIG_DIR / "cambottom_table_pose.json"


# =========================================================
# CHARGEMENT POSE
# =========================================================
def load_bottom_pose(path: Path):
    """
    Charge rvec, tvec, TABLE_SIZE_MM et la matrice caméra rectifiée
    depuis le JSON de pose.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    table_size_mm = float(data["table_size_mm"])
    rectified_K = np.array(data["camera_matrix"], dtype=np.float64)

    return rvec, tvec, table_size_mm, rectified_K

# =========================================================
# TRANSFORMATIONS
# =========================================================
def mm_to_graph(x_mm: float, y_mm: float, table_size_mm: float) -> tuple[float, float]:
    """
    Convertit des coordonnées en mm dans le repère table
    (origine = coin TL, X→droite, Y→bas)
    en coordonnées dans le repère graphique [0, GRID_SIZE].

    La normalisation utilise TABLE_SIZE_MM chargée depuis le JSON,
    pas une constante hardcodée.
    """
    xg = (x_mm / table_size_mm) * GRID_SIZE
    yg = (y_mm / table_size_mm) * GRID_SIZE
    return xg, yg


# =========================================================
# MAIN
# =========================================================
def main():

    # --- Calibration caméra brute fisheye ---
    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)

    # --- Ouvrir caméra ---
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] Caméra non ouverte")
        return

    # Lire une frame pour fixer la résolution
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Résolution : {w}x{h}")

    # --- Charger pose + matrice rectifiée EXACTE utilisée pour la pose ---
    rvec, tvec, table_size_mm, new_K = load_bottom_pose(POSE_PATH)

    print(f"[INFO] new_K (depuis pose JSON) : fx={new_K[0,0]:.1f}, fy={new_K[1,1]:.1f}, "
          f"cx={new_K[0,2]:.1f}, cy={new_K[1,2]:.1f}")
    print(f"[INFO] TABLE_SIZE_MM (depuis JSON) : {table_size_mm} mm")
    print(f"[INFO] GRID_SIZE : {GRID_SIZE} px")
    print(f"[INFO] Résolution graphique : {GRID_SIZE / table_size_mm:.3f} px/mm")

    # --- Maps fisheye cohérentes avec pose.py ---
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    # --- ArUco ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Détection en cours... (q pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Rectification fisheye cohérente avec la pose ---
        frame_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()

        if ids is not None and len(ids) > 0:
            centers = get_marker_centers(corners, ids)
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for marker_id in ids.flatten():
                marker_id = int(marker_id)

                if marker_id in CALIBRATION_TAG_IDS:
                    continue

                if marker_id not in centers:
                    continue

                u, v = centers[marker_id]

                # --- Pixel rectifié -> mm avec la MEME new_K que celle de pose.py ---
                result = pixel_to_table_mm(u, v, rvec, tvec, new_K)
                if result is None:
                    continue

                x_mm, y_mm = result

                in_table = (0.0 <= x_mm <= table_size_mm) and \
                           (0.0 <= y_mm <= table_size_mm)

                xg, yg = mm_to_graph(x_mm, y_mm, table_size_mm)

                dot_color = (0, 255, 0) if in_table else (0, 0, 255)
                cv2.circle(display, (int(u), int(v)), 6, dot_color, -1)

                cv2.putText(display,
                            f"ID {marker_id}",
                            (int(u) + 10, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display,
                            f"mm=({x_mm:.1f}, {y_mm:.1f})",
                            (int(u) + 10, int(v) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.putText(display,
                            f"graph=({xg:.1f}, {yg:.1f})",
                            (int(u) + 10, int(v) + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                print(f"ID {marker_id:3d} | "
                      f"mm=({x_mm:7.2f}, {y_mm:7.2f}) | "
                      f"graph=({xg:7.2f}, {yg:7.2f})"
                      + ("" if in_table else "  [HORS TABLE]"))

        cv2.imshow("Bottom -> Graph", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()