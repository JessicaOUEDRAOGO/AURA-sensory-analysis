# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0

TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

TABLE_SIZE_MM = 580.0

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
OUTPUT_JSON = CONFIG_DIR / "cambottom_table_pose.json"

# =========================================================
# CHARGEMENT
# =========================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_camera_calibration(path):
    data = load_json(path)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist

# =========================================================
# GEOMETRIE (IDENTIQUE TOP)
# =========================================================
def estimate_table_pose(centers, K):

    object_points = np.array([
        [0,              TABLE_SIZE_MM,  0],   # TL devient bas gauche
        [TABLE_SIZE_MM,  TABLE_SIZE_MM,  0],   # TR devient bas droit
        [TABLE_SIZE_MM,  0,              0],   # BR devient haut droit
        [0,              0,              0],   # BL devient haut gauche
    ], dtype=np.float64)

    image_points = np.array([
        centers[TAG_IDS["TL"]],
        centers[TAG_IDS["TR"]],
        centers[TAG_IDS["BR"]],
        centers[TAG_IDS["BL"]],
    ], dtype=np.float64)

    dist_zero = np.zeros((5,1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_zero,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        K,
        dist_zero,
        rvec,
        tvec
    )

    return rvec, tvec

# =========================================================
# CONVERSION PIXEL → MM
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
        cx = float(np.mean(pts[:,0]))
        cy = float(np.mean(pts[:,1]))
        centers[int(marker_id)] = (cx, cy)
    return centers

def save_pose(rvec, tvec, K):
    data = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "camera_matrix": K.tolist()
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[OK] Sauvegardé -> {OUTPUT_JSON}")

# =========================================================
# VALIDATION
# =========================================================
def run_validation(centers, rvec, tvec, K):

    print("\n=== VALIDATION COINS ===")

    expected = {
        "TL": (0, 0),
        "TR": (TABLE_SIZE_MM, 0),
        "BR": (TABLE_SIZE_MM, TABLE_SIZE_MM),
        "BL": (0, TABLE_SIZE_MM),
    }

    for name, tag_id in TAG_IDS.items():
        cx, cy = centers[tag_id]
        x, y = pixel_to_table(cx, cy, rvec, tvec, K)
        print(f"{name} -> ({x:.2f}, {y:.2f}) mm")

    print("\n=== TEST CENTRE IMAGE ===")
    x, y = pixel_to_table(960, 540, rvec, tvec, K)
    print(f"Centre image -> ({x:.2f}, {y:.2f}) mm")

# =========================================================
# MAIN
# =========================================================
def main():

    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] appuie sur 'c' pour capturer")

    while True:

        ret, frame = cap.read()
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        display = frame_undist.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            centers = get_marker_centers(corners, ids)

            if all(t in centers for t in TAG_IDS.values()):
                cv2.putText(display, "OK - press C",
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0),2)

        cv2.imshow("POSE CAM BOTTOM", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):

            if ids is None:
                continue

            centers = get_marker_centers(corners, ids)

            if not all(t in centers for t in TAG_IDS.values()):
                print("tags manquants")
                continue

            rvec, tvec = estimate_table_pose(centers, new_K)

            print("rvec =", rvec.flatten())
            print("tvec =", tvec.flatten())

            run_validation(centers, rvec, tvec, new_K)

            save_pose(rvec, tvec, new_K)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()