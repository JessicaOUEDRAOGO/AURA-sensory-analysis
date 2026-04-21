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

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
HOMOGRAPHY_PATH = CONFIG_DIR / "homography_cam_bottom_to_table.json"

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
# CHARGEMENT
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_camera_calibration(path):
    data = load_json(path)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist

def load_homography(path):
    data = load_json(path)
    H = np.array(data["H_pixel_to_mm"], dtype=np.float64)
    return H

# =========================================================
# OUTILS
# =========================================================
def pixel_to_mm(pt, H):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    p_mm = H @ p
    p_mm /= p_mm[2]
    return float(p_mm[0]), float(p_mm[1])

def get_marker_centers(corners, ids):
    """
    EXACTEMENT la même logique que le script de pose :
    centre = moyenne des 4 coins
    """
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
def run_test(centers, H):
    print("\n=== TEST COINS (H chargée) ===")

    for name, tag_id in TAG_IDS.items():
        if tag_id in centers:
            cx, cy = centers[tag_id]
            x_mm, y_mm = pixel_to_mm((cx, cy), H)
            print(f"{name} -> ({x_mm:.2f}, {y_mm:.2f}) mm")
        else:
            print(f"{name} -> non détecté")

    print("\n=== TEST TAG ===")

    if TEST_TAG_ID in centers:
        cx, cy = centers[TEST_TAG_ID]
        x_mm, y_mm = pixel_to_mm((cx, cy), H)

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
    print("Calibration :", CAMERA_CALIB_PATH)
    print("Homographie :", HOMOGRAPHY_PATH)

    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)
    H = load_homography(HOMOGRAPHY_PATH)

    print("\nH utilisée =\n", np.round(H, 6))

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] caméra")
        return

    h, w = frame.shape[:2]
    print("frame size =", w, h)

    # EXACTEMENT même paramètre que pose
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("\n[INFO] Test automatique - appuie sur 'q' pour quitter")

    tested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------------------------
        # IMPORTANT : même pipeline que pose
        # -------------------------------------------------
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        display = frame_undist.copy()

        centers = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            centers = get_marker_centers(corners, ids)

            # affichage
            for marker_id, (cx, cy) in centers.items():
                cv2.circle(display, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(display, str(marker_id),
                            (int(cx)+5, int(cy)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

        # test automatique si 4 tags visibles
        if all(tag_id in centers for tag_id in TAG_IDS.values()):
            if not tested:
                run_test(centers, H)
                tested = True
        else:
            tested = False

        cv2.imshow("TEST HOMOGRAPHIE (UNDIST)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
if __name__ == "__main__":
    main()