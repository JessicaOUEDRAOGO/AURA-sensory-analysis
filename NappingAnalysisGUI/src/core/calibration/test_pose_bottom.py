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

# =========================================================
# LOAD POSE
# =========================================================
def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    K    = np.array(data["camera_matrix"], dtype=np.float64)

    return rvec, tvec, K

# =========================================================
# GEOMETRY
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
# CENTER
# =========================================================
def get_center(pts):
    return np.mean(pts, axis=0)

# =========================================================
# MAIN
# =========================================================
def main():

    rvec, tvec, K = load_pose(POSE_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("\n[TEST] Bouge le tag 7 (pose vs levé)\n")

    last_xy = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame.copy()

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):

                if marker_id == TEST_TAG_ID:

                    pts = corners[i][0]
                    cx, cy = get_center(pts)

                    res = pixel_to_table(cx, cy, rvec, tvec, K)

                    if res is not None:
                        x_mm, y_mm = res

                        cv2.circle(display, (int(cx), int(cy)), 6, (0,0,255), -1)

                        text = f"X={x_mm:.1f} mm  Y={y_mm:.1f} mm"
                        cv2.putText(display, text, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        # stabilité
                        if last_xy is not None:
                            dx = x_mm - last_xy[0]
                            dy = y_mm - last_xy[1]
                            drift = np.sqrt(dx*dx + dy*dy)

                            cv2.putText(display,
                                f"Drift={drift:.2f} mm",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                        last_xy = (x_mm, y_mm)

        cv2.imshow("TEST Z != 0", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()