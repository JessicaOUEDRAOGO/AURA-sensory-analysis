# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# IMPORTS TON PIPELINE
# =========================================================
from src.core.calibration.pose_cam_bottom_V2 import (
    pixel_to_table_mm,
    load_camera_calibration
)

# =========================================================
# PARAMETRES
# =========================================================
CAMERA_ID = 0

PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

TABLE_SIZE_MM = 580.0

# grille de points projetés (plus = plus précis)
GRID_COLS = 6
GRID_ROWS = 6

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

POSE_PATH = CONFIG_DIR / "cambottom_table_pose.json"
CAM_CALIB_PATH = CONFIG_DIR / "camera_calibration_fisheye.json"
OUTPUT_PATH = CONFIG_DIR / "homography_table_to_projector.json"

# =========================================================
# LOAD POSE CAM BOTTOM
# =========================================================
def load_pose():
    data = json.load(open(POSE_PATH))
    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    return rvec, tvec

# =========================================================
# GENERER POINTS PROJECTEUR
# =========================================================
def generate_projector_points():
    MARGIN_X = 400
    MARGIN_Y = 300

    usable_w = PROJECTOR_WIDTH - 2*MARGIN_X
    usable_h = PROJECTOR_HEIGHT - 2*MARGIN_Y

    pts = []
    for i in range(GRID_COLS):
        for j in range(GRID_ROWS):
            x = MARGIN_X + (i + 0.5) * usable_w / GRID_COLS
            y = MARGIN_Y + (j + 0.5) * usable_h / GRID_ROWS
            pts.append((x, y))

    return pts
# =========================================================
# AFFICHAGE PROJECTEUR
# =========================================================
def create_projection(points):
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), 15, (255, 255, 255), -1)

    return img

# =========================================================
# DETECTION CENTROÏDES
# =========================================================
def detect_blobs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))

    return centers

# =========================================================
# ASSOCIATION POINTS
# =========================================================
def match_points(cam_pts, proj_pts):
    """
    Associe chaque point caméra au point projecteur le plus proche (tri spatial simple)
    """
    cam_pts = sorted(cam_pts, key=lambda p: (p[1], p[0]))
    proj_pts = sorted(proj_pts, key=lambda p: (p[1], p[0]))

    if len(cam_pts) != len(proj_pts):
        print("[ERREUR] Nombre de points différent")
        return None, None

    return cam_pts, proj_pts

# =========================================================
# MAIN
# =========================================================
def main():

    # --- caméra ---
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Erreur caméra")
        return

    # --- calibration ---
    K, dist = load_camera_calibration(CAM_CALIB_PATH)
    rvec, tvec = load_pose()

    # --- points projecteur ---
    proj_points = generate_projector_points()

    print(f"[INFO] {len(proj_points)} points projetés")

    print("[INFO] Appuie sur 'c' pour capturer")

    while True:
        proj_img = create_projection(proj_points)

        cv2.imshow("PROJECTOR", proj_img)

        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        cv2.imshow("CAMERA", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):

            print("[INFO] Capture...")

            # --- detect blobs ---
            cam_pixels = detect_blobs(frame)

            if len(cam_pixels) < 10:
                print("[ERREUR] Pas assez de points détectés")
                continue

            # --- pixel -> table mm ---
            table_pts = []
            for (u, v) in cam_pixels:
                res = pixel_to_table_mm(u, v, rvec, tvec, K)
                if res is None:
                    continue
                table_pts.append(res)

            if len(table_pts) != len(proj_points):
                print("[ERREUR] mismatch points")
                continue

            # --- association ---
            table_pts, proj_pts = match_points(table_pts, proj_points)

            table_pts = np.array(table_pts, dtype=np.float32)
            proj_pts = np.array(proj_pts, dtype=np.float32)

            # --- HOMOGRAPHIE ---
            H, mask = cv2.findHomography(table_pts, proj_pts, cv2.RANSAC, 3.0)

            print("\n=== HOMOGRAPHIE ===")
            print(H)

            # --- sauvegarde ---
            data = {
                "H_table_to_proj": H.tolist(),
                "TABLE_SIZE_MM": TABLE_SIZE_MM,
                "PROJECTOR_WIDTH": PROJECTOR_WIDTH,
                "PROJECTOR_HEIGHT": PROJECTOR_HEIGHT
            }

            with open(OUTPUT_PATH, "w") as f:
                json.dump(data, f, indent=4)

            print(f"[OK] Sauvegardé → {OUTPUT_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()