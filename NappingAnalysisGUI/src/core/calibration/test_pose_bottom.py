# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

POSE_PATH = CONFIG_DIR / "cambottom_table_pose.json"
CALIB_PATH = CONFIG_DIR / "camera_calibration.json"

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0
TEST_TAG_ID = 7
EXPECTED_MM = (200.0, 400.0)

TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41
}

# =========================================================
# LOAD
# =========================================================
def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    new_K = np.array(data["camera_matrix"], dtype=np.float64)

    return rvec, tvec, new_K

def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    return K, dist

# =========================================================
# GEOMETRIE (IDENTIQUE TOP)
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
    return float(pt[0]), float(pt[1])

def pixel_to_table(u, v, rvec, tvec, K):
    ray = pixel_to_ray(u, v, K)
    pt_cam = intersect_ray_plane(ray, rvec, tvec)
    if pt_cam is None:
        return None
    return camera_to_table(pt_cam, rvec, tvec)

# =========================================================
# OUTILS
# =========================================================
def get_marker_centers(corners, ids):
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centers[int(marker_id)] = (cx, cy)
    return centers

# =========================================================
# TEST
# =========================================================
def run_test(centers, rvec, tvec, K):

    print("\n=== TEST COINS ===")

    for name, tag_id in TAG_IDS.items():
        if tag_id in centers:
            cx, cy = centers[tag_id]
            x_mm, y_mm = pixel_to_table(cx, cy, rvec, tvec, K)
            print(f"{name} -> ({x_mm:.2f}, {y_mm:.2f}) mm")
        else:
            print(f"{name} -> non détecté")

    print("\n=== TEST TAG ===")

    if TEST_TAG_ID in centers:
        cx, cy = centers[TEST_TAG_ID]
        x_mm, y_mm = pixel_to_table(cx, cy, rvec, tvec, K)

        error_x = x_mm - EXPECTED_MM[0]
        error_y = y_mm - EXPECTED_MM[1]
        error_total = np.sqrt(error_x**2 + error_y**2)

        print(f"ID = {TEST_TAG_ID}")
        print(f"Mesuré (mm) : ({x_mm:.2f}, {y_mm:.2f})")
        print(f"Attendu (mm) : {EXPECTED_MM}")
        print(f"Erreur X : {error_x:.2f} mm")
        print(f"Erreur Y : {error_y:.2f} mm")
        print(f"Erreur totale : {error_total:.2f} mm")
    else:
        print(f"Tag {TEST_TAG_ID} non détecté")

# =========================================================
# MAIN
# =========================================================
def main():

    print("=== CHARGEMENT ===")
    print("Pose :", POSE_PATH)

    rvec, tvec, new_K = load_pose(POSE_PATH)
    K, dist = load_calibration(CALIB_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("\n[INFO] Test automatique - appuie sur 'q' pour quitter")

    tested = False

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =================================================
        # UNDISTORT (OBLIGATOIRE)
        # =================================================
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        display = frame_undist.copy()

        centers = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            centers = get_marker_centers(corners, ids)

            for marker_id, (cx, cy) in centers.items():
                cv2.circle(display, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(display, str(marker_id),
                            (int(cx)+5, int(cy)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

        if all(tag_id in centers for tag_id in TAG_IDS.values()):
            if not tested:
                run_test(centers, rvec, tvec, new_K)
                tested = True
        else:
            tested = False

        cv2.imshow("TEST CAM BOTTOM (PnP)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
if __name__ == "__main__":
    main()