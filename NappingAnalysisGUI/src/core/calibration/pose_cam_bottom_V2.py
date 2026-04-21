# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

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

# Taille extérieure de la table (mm)
TABLE_SIZE_MM = 580.0

# Taille physique d'un tag ArUco (mm)
# Comme on utilise les CENTRES des tags, les centres sont décalés
# de half = TAG_SIZE_MM / 2 vers l'intérieur par rapport aux bords.
TAG_SIZE_MM = 40.0

# Nombre max de frames gardées pour la moyenne
N_FRAMES_AVERAGE = 30

# Nombre minimal de frames avant d'autoriser une capture moyennée
MIN_FRAMES_FOR_CAPTURE = 10

# Critères pour cornerSubPix
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)
SUBPIX_WIN = (5, 5)
SUBPIX_ZERO = (-1, -1)

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
OUTPUT_JSON = CONFIG_DIR / "homography_cam_bottom_to_table.json"

# =========================================================
# CHARGEMENT JSON / CALIBRATION
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_camera_calibration(path: Path):
    data = load_json(path)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist

# =========================================================
# OUTILS GÉOMÉTRIQUES
# =========================================================
def refine_corners(gray: np.ndarray, corners: list) -> list:
    """
    Raffine les coins ArUco au sous-pixel.
    """
    refined = []
    for c in corners:
        pts = c[0].astype(np.float32).copy()  # shape (4,2)
        refined_pts = cv2.cornerSubPix(
            gray, pts, SUBPIX_WIN, SUBPIX_ZERO, SUBPIX_CRITERIA
        )
        refined.append(refined_pts.reshape(1, 4, 2))
    return refined

def get_marker_centers(corners: list, ids: np.ndarray) -> dict:
    """
    Centre = moyenne des 4 coins.
    On garde exactement la même logique partout :
    pose, test, runtime si possible.
    """
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]  # (4,2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centers[int(marker_id)] = (cx, cy)
    return centers

def pixel_to_mm(pt, H: np.ndarray):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    p_mm = H @ p
    p_mm /= p_mm[2]
    return float(p_mm[0]), float(p_mm[1])

# =========================================================
# ACCUMULATION TEMPORELLE
# =========================================================
def accumulate_centers(
    accumulator: dict,
    centers: dict,
    required_ids: set,
    max_frames: int
) -> dict:
    """
    Accumule uniquement les frames où les 4 tags requis sont visibles.
    Renvoie les centres moyennés si on a assez d'échantillons.
    """
    if not required_ids.issubset(centers.keys()):
        return {}

    for tag_id in required_ids:
        accumulator[tag_id].append(centers[tag_id])
        if len(accumulator[tag_id]) > max_frames:
            accumulator[tag_id].pop(0)

    if all(len(accumulator[tid]) >= MIN_FRAMES_FOR_CAPTURE for tid in required_ids):
        averaged = {}
        for tid in required_ids:
            xs = [p[0] for p in accumulator[tid]]
            ys = [p[1] for p in accumulator[tid]]
            averaged[tid] = (float(np.mean(xs)), float(np.mean(ys)))
        return averaged

    return {}

def get_accumulated_count(accumulator: dict, required_ids: set) -> int:
    if not accumulator:
        return 0
    counts = [len(accumulator.get(tid, [])) for tid in required_ids]
    return min(counts) if counts else 0

# =========================================================
# HOMOGRAPHIE
# =========================================================
def build_homography(centers: dict):
    """
    H : pixel_undist -> mm

    Référentiel mm :
      - origine au coin extérieur haut-gauche de la table
      - les centres des tags sont à half du bord
    """
    half = TAG_SIZE_MM / 2.0
    S = TABLE_SIZE_MM

    pts_img = np.array([
        centers[TAG_IDS["TL"]],
        centers[TAG_IDS["TR"]],
        centers[TAG_IDS["BR"]],
        centers[TAG_IDS["BL"]],
    ], dtype=np.float64)

    pts_mm = np.array([
        [half,     half],
        [S - half, half],
        [S - half, S - half],
        [half,     S - half],
    ], dtype=np.float64)

    H, _ = cv2.findHomography(pts_img, pts_mm, method=0)

    if H is None:
        raise RuntimeError("Échec du calcul de l'homographie.")

    return H, pts_img, pts_mm

# =========================================================
# VALIDATION
# =========================================================
def validate_homography(H: np.ndarray, centers: dict, verbose: bool = True):
    """
    Validation interne :
    - coins de référence
    - tag test ID=7 si visible

    Note :
    la validation coins vérifie surtout la cohérence interne.
    La vraie validation utile est le tag test indépendant.
    """
    print("\n=== VALIDATION COINS ===")

    half = TAG_SIZE_MM / 2.0
    S = TABLE_SIZE_MM

    expected = {
        TAG_IDS["TL"]: (half,     half),
        TAG_IDS["TR"]: (S - half, half),
        TAG_IDS["BR"]: (S - half, S - half),
        TAG_IDS["BL"]: (half,     S - half),
    }

    errors = []
    for name, tag_id in TAG_IDS.items():
        cx, cy = centers[tag_id]
        x_mm, y_mm = pixel_to_mm((cx, cy), H)
        ex, ey = expected[tag_id]
        err = np.hypot(x_mm - ex, y_mm - ey)
        errors.append(err)

        if verbose:
            print(
                f"  {name} (id={tag_id}) -> mesuré ({x_mm:.2f}, {y_mm:.2f}) mm  "
                f"| attendu ({ex:.2f}, {ey:.2f}) mm  | erreur {err:.2f} mm"
            )

    rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    print(f"\n  RMS erreur coins : {rms:.2f} mm")

    TEST_TAG_ID = 7
    EXPECTED_MM = (200.0, 200.0)

    if TEST_TAG_ID in centers:
        cx, cy = centers[TEST_TAG_ID]
        x_mm, y_mm = pixel_to_mm((cx, cy), H)

        err_x = x_mm - EXPECTED_MM[0]
        err_y = y_mm - EXPECTED_MM[1]
        err_total = np.hypot(err_x, err_y)

        print(f"\n=== TEST TAG ID={TEST_TAG_ID} ===")
        print(f"  Mesuré  : ({x_mm:.2f}, {y_mm:.2f}) mm")
        print(f"  Attendu : ({EXPECTED_MM[0]:.1f}, {EXPECTED_MM[1]:.1f}) mm")
        print(f"  Erreur X : {err_x:+.2f} mm  |  Erreur Y : {err_y:+.2f} mm  |  Totale : {err_total:.2f} mm")
    else:
        print(f"\n  [INFO] Tag ID={TEST_TAG_ID} non visible, test ignoré.")

    return rms

# =========================================================
# SAUVEGARDE
# =========================================================
def save_json(H: np.ndarray, pts_img: np.ndarray, pts_mm: np.ndarray):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "H_pixel_to_mm": H.tolist(),
        "points_image_px": pts_img.tolist(),
        "points_table_mm": pts_mm.tolist(),
        "table_size_mm": TABLE_SIZE_MM,
        "tag_size_mm": TAG_SIZE_MM,
        "tag_ids": TAG_IDS,
        "space": "pixel_undist_to_table_mm",
        "calibration_file": str(CAMERA_CALIB_PATH),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\n[OK] Homographie sauvegardée -> {OUTPUT_JSON}")

# =========================================================
# AFFICHAGE
# =========================================================
def _draw_mm_polyline(img, pts_mm, H, color):
    H_inv = np.linalg.inv(H)
    pts_px = []

    for pt in pts_mm:
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        px = H_inv @ p
        px /= px[2]
        pts_px.append((int(round(px[0])), int(round(px[1]))))

    for i in range(len(pts_px) - 1):
        cv2.line(img, pts_px[i], pts_px[i + 1], color, 1, cv2.LINE_AA)

def draw_hud(display: np.ndarray, centers: dict, accumulator: dict, required_ids: set, H=None):
    all_present = required_ids.issubset(centers.keys())

    for name, tag_id in TAG_IDS.items():
        if tag_id in centers:
            cx, cy = int(round(centers[tag_id][0])), int(round(centers[tag_id][1]))
            color = (0, 255, 0) if all_present else (0, 165, 255)
            cv2.circle(display, (cx, cy), 6, color, -1)
            cv2.putText(
                display, name, (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    TEST_TAG_ID = 7
    if TEST_TAG_ID in centers:
        cx, cy = int(round(centers[TEST_TAG_ID][0])), int(round(centers[TEST_TAG_ID][1]))
        cv2.circle(display, (cx, cy), 8, (255, 80, 0), -1)
        cv2.putText(
            display, "TEST-7", (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2
        )

    n_acc = get_accumulated_count(accumulator, required_ids)
    bar_w = int(300 * min(n_acc, N_FRAMES_AVERAGE) / N_FRAMES_AVERAGE)

    cv2.rectangle(display, (20, 60), (320, 85), (60, 60, 60), -1)
    cv2.rectangle(display, (20, 60), (20 + bar_w, 85), (0, 220, 100), -1)
    cv2.putText(
        display, f"Stabilisation : {n_acc}/{N_FRAMES_AVERAGE}",
        (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
    )

    if all_present:
        msg = "4 TAGS OK - appuie sur 'c' pour capturer"
        col = (0, 255, 0)
    else:
        missing = [k for k, v in TAG_IDS.items() if v not in centers]
        msg = f"Tags manquants : {', '.join(missing)}"
        col = (0, 50, 255)

    cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    if H is not None and all_present:
        step = TABLE_SIZE_MM / 4.0
        for i in range(5):
            pts_row = np.array(
                [[j * TABLE_SIZE_MM / 20.0, i * step] for j in range(21)],
                dtype=np.float64
            )
            pts_col = np.array(
                [[i * step, j * TABLE_SIZE_MM / 20.0] for j in range(21)],
                dtype=np.float64
            )
            _draw_mm_polyline(display, pts_row, H, (180, 180, 50))
            _draw_mm_polyline(display, pts_col, H, (180, 180, 50))

# =========================================================
# MAIN
# =========================================================
def main():
    print("=== CHARGEMENT CALIBRATION ===")
    print("Calibration :", CAMERA_CALIB_PATH)

    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra.")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Résolution : {w} x {h}")

    # Même logique que top : on définit new_K une seule fois
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    required_ids = set(TAG_IDS.values())
    accumulator = defaultdict(list)
    H_current = None

    print("[INFO] 'c' = capturer  |  'r' = reset accumulateur  |  'q' = quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------------------------
        # IMPORTANT : toute la pose est faite en UNDISTORTED
        # -------------------------------------------------
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()
        centers = {}
        averaged = {}

        if ids is not None:
            corners = refine_corners(gray, corners)
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            centers = get_marker_centers(corners, ids)
            averaged = accumulate_centers(accumulator, centers, required_ids, N_FRAMES_AVERAGE)

        draw_hud(display, centers, accumulator, required_ids, H_current)
        cv2.imshow("Pose homographie - camera bottom", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            accumulator.clear()
            H_current = None
            print("[INFO] Accumulateur réinitialisé.")

        elif key == ord('c'):
            # Priorité aux centres moyennés si disponibles
            use_centers = averaged if averaged else centers

            if not required_ids.issubset(use_centers.keys()):
                print("[ERREUR] Les 4 tags ne sont pas visibles ou la stabilisation est insuffisante.")
                continue

            try:
                H, pts_img, pts_mm = build_homography(use_centers)
            except RuntimeError as e:
                print(f"[ERREUR] {e}")
                continue

            print("\n=== MATRICE H (pixel_undist -> mm) ===")
            print(np.round(H, 6))

            rms = validate_homography(H, use_centers)

            if rms > 5.0:
                print(f"\n[AVERTISSEMENT] RMS={rms:.2f} mm")
            else:
                print(f"\n[OK] Précision interne acceptable (RMS={rms:.2f} mm)")

            save_json(H, pts_img, pts_mm)
            H_current = H

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()