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
    "TL": 41,
    "TR": 40,
    "BR": 43,
    "BL": 42
}

TABLE_SIZE_MM = 580.0
# =========================================================
# CHEMINS PROJET
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration_top.json"
OUTPUT_JSON = CONFIG_DIR / "homography_camtop_to_table.json"


# =========================================================
# CALIBRATION CAMERA
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_camera_calibration(path: Path):
    data = load_json(path)

    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])

    return K, dist
# =========================================================
# OUTILS
# =========================================================
def compute_center_diagonals(pts):
    p1, p2, p3, p4 = pts

    def line(p, q):
        A = q[1] - p[1]
        B = p[0] - q[0]
        C = A*p[0] + B*p[1]
        return A, B, C

    A1, B1, C1 = line(p1, p3)
    A2, B2, C2 = line(p2, p4)

    det = A1*B2 - A2*B1

    if abs(det) < 1e-6:
        return None

    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det

    return (x, y)


def get_marker_centers(corners, ids):
    centers = {}

    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]

        center = compute_center_diagonals(pts)

        if center is not None:
            cx, cy = center
        else:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

        centers[int(marker_id)] = (cx, cy)

    return centers


def build_homography(centers):
    pts_img = np.array([
        centers[TAG_IDS["TL"]],
        centers[TAG_IDS["TR"]],
        centers[TAG_IDS["BR"]],
        centers[TAG_IDS["BL"]],
    ], dtype=np.float32)

    pts_mm = np.array([
        [0, 0],
        [TABLE_SIZE_MM, 0],
        [TABLE_SIZE_MM, TABLE_SIZE_MM],
        [0, TABLE_SIZE_MM],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_img, pts_mm, cv2.RANSAC)

    return H, pts_img, pts_mm


def pixel_to_mm(pt, H):
    p = np.array([pt[0], pt[1], 1.0])
    p_mm = H @ p
    p_mm /= p_mm[2]
    return p_mm[0], p_mm[1]


def save_json(H, pts_img, pts_mm):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "H_pixel_to_mm": H.tolist(),
        "points_image_px": pts_img.tolist(),
        "points_table_mm": pts_mm.tolist(),
        "table_size_mm": TABLE_SIZE_MM,
        "tag_ids": TAG_IDS
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Sauvegardé dans {OUTPUT_JSON}")


# =========================================================
# MAIN
# =========================================================
def main():

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra")
        return

    # Charger calibration
    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)
    
    # Lire une frame pour init new_K
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra")
        return

    h, w = frame.shape[:2]
    print("frame size =", w, h)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Appuie sur 'c' pour capturer et calculer l'homographie")
    # TEST_TAG_ID = 7
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # UNDISTORTION
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)

        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            centers = get_marker_centers(corners, ids)

            if all(tag_id in centers for tag_id in TAG_IDS.values()):

                for name, tag_id in TAG_IDS.items():
                    cx, cy = centers[tag_id]

                    cv2.circle(display, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                    cv2.putText(display, name, (int(cx)+5, int(cy)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(display, "4 TAGS OK - press 'c'",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # # Affichage du tag test
            # if TEST_TAG_ID in centers:
            #     cx, cy = centers[TEST_TAG_ID]

            #     cv2.circle(display, (int(cx), int(cy)), 8, (255, 0, 0), -1)
            #     cv2.putText(display, "TEST 7", (int(cx)+10, int(cy)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Detection camera top", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):

            if ids is None:
                print("[ERREUR] Aucun tag détecté")
                continue

            centers = get_marker_centers(corners, ids)

            if not all(tag_id in centers for tag_id in TAG_IDS.values()):
                print("[ERREUR] Les 4 tags ne sont pas visibles")
                continue

            H, pts_img, pts_mm = build_homography(centers)

            print("\n=== MATRICE H (pixel -> mm) ===")
            print(H)

            # TEST CENTRE IMAGE (UNDISTORDUE)
            h, w = frame_undist.shape[:2]
            test_pt = (w//2, h//2)

            x_mm, y_mm = pixel_to_mm(test_pt, H)

            print(f"\nTest point centre image -> ({x_mm:.1f} mm, {y_mm:.1f} mm)")
            # # =========================================================
            # # TEST TAG ID = 7 (position connue)
            # # =========================================================
            # TEST_TAG_ID = 7
            # EXPECTED_MM = (400.0, 300.0)

            # if TEST_TAG_ID in centers:
            #     cx, cy = centers[TEST_TAG_ID]

            #     x_mm, y_mm = pixel_to_mm((cx, cy), H)

            #     print("\n=== TEST TAG ID 7 ===")
            #     print(f"Position détectée (mm) : ({x_mm:.2f}, {y_mm:.2f})")
            #     print(f"Position attendue (mm) : ({EXPECTED_MM[0]}, {EXPECTED_MM[1]})")

            #     error_x = x_mm - EXPECTED_MM[0]
            #     error_y = y_mm - EXPECTED_MM[1]
            #     error_total = np.sqrt(error_x**2 + error_y**2)

            #     print(f"Erreur X : {error_x:.2f} mm")
            #     print(f"Erreur Y : {error_y:.2f} mm")
            #     print(f"Erreur totale : {error_total:.2f} mm")

            # else:
            #     print("\n[ERREUR] Tag ID 7 non détecté")
            # TEST COINS (TRÈS IMPORTANT)
            print("\n=== TEST COINS ===")
            for name, tag_id in TAG_IDS.items():
                cx, cy = centers[tag_id]
                x_mm, y_mm = pixel_to_mm((cx, cy), H)
                print(f"{name} -> ({x_mm:.1f}, {y_mm:.1f}) mm")

            save_json(H, pts_img, pts_mm)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()