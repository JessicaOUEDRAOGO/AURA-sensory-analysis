# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 1

TAG_IDS = {
    "TL": 41,
    "TR": 40,
    "BR": 43,
    "BL": 42
}

TABLE_SIZE_MM = 597.0

# Distance entre le bord extérieur de la table et le centre du tag (mm).
# Tags de 20×20 mm posés à ras du coin → centre à 10 mm du bord.
TAG_OFFSET_MM = 10.0

# =========================================================
# CHEMINS PROJET
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration_top.json"
OUTPUT_JSON = CONFIG_DIR / "camtop_table_pose.json"


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
# GÉOMÉTRIE 3D TABLE
# =========================================================

def get_tag_object_points():
    """
    Retourne les coordonnées 3D RÉELLES des centres des 4 tags dans le
    repère table (Z = 0, origine = coin TL de la table).

    Les tags (20×20 mm) sont posés à ras des coins → leur centre est à
    TAG_OFFSET_MM du bord, donc :

        TL : (TAG_OFFSET_MM,                TAG_OFFSET_MM,               0)
        TR : (TABLE_SIZE_MM - TAG_OFFSET_MM, TAG_OFFSET_MM,               0)
        BR : (TABLE_SIZE_MM - TAG_OFFSET_MM, TABLE_SIZE_MM - TAG_OFFSET_MM, 0)
        BL : (TAG_OFFSET_MM,                TABLE_SIZE_MM - TAG_OFFSET_MM, 0)

    L'ordre doit correspondre exactement à l'ordre des image_points.
    """
    o = TAG_OFFSET_MM
    s = TABLE_SIZE_MM
    return np.array([
        [o,     o,     0],   # TL
        [s - o, o,     0],   # TR
        [s - o, s - o, 0],   # BR
        [o,     s - o, 0],   # BL
    ], dtype=np.float64)


def estimate_table_pose(centers, K):
    """
    Estime la pose de la table via solvePnP.

    On passe dist=zeros car l'image est déjà undistordue (new_K utilisé).
    Les object_points correspondent aux centres réels des tags dans le
    repère table (pas aux coins de la table).

    Retourne (rvec, tvec) ou (None, None) si échec.
    """
    object_points = get_tag_object_points()

    image_points = np.array([
        centers[TAG_IDS["TL"]],
        centers[TAG_IDS["TR"]],
        centers[TAG_IDS["BR"]],
        centers[TAG_IDS["BL"]],
    ], dtype=np.float64)

    dist_zero = np.zeros((5, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_zero,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    # Raffiner avec solvePnPRefineLM pour plus de précision
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        K,
        dist_zero,
        rvec,
        tvec
    )

    return rvec, tvec


def pixel_to_ray(u, v, K):
    """Calcule le rayon unitaire passant par le pixel (u, v) dans le repère caméra."""
    K_inv = np.linalg.inv(K)
    pt = np.array([u, v, 1.0], dtype=np.float64)
    ray = K_inv @ pt
    ray = ray / np.linalg.norm(ray)
    return ray


def intersect_ray_with_table_plane(ray, rvec, tvec):
    """
    Intersecte le rayon (depuis l'origine caméra) avec le plan de la table
    (Z = 0 dans le repère table).

    Le plan est défini par :
      - sa normale : axe Z de la table exprimé dans le repère caméra (3e colonne de R)
      - un point appartenant au plan : tvec (origine de la table dans le repère caméra)

    Retourne le point d'intersection dans le repère caméra, ou None.
    """
    R, _ = cv2.Rodrigues(rvec)
    normal      = R[:, 2]
    plane_origin = tvec.reshape(3)
    ray_origin   = np.zeros(3, dtype=np.float64)

    denom = np.dot(normal, ray)
    if abs(denom) < 1e-9:
        return None  # Rayon parallèle au plan

    t = np.dot(normal, plane_origin - ray_origin) / denom
    if t < 0:
        return None  # Intersection derrière la caméra

    return ray_origin + t * ray


def camera_to_table(point_cam, rvec, tvec):
    """
    Transforme un point du repère caméra vers le repère table.
    Retourne (x_mm, y_mm) dans le plan de la table (Z ignoré, proche de 0).
    """
    R, _ = cv2.Rodrigues(rvec)
    point_table = R.T @ (point_cam - tvec.reshape(3))
    return float(point_table[0]), float(point_table[1])


def pixel_to_table_mm(u, v, rvec, tvec, K):
    """
    Convertit un pixel (u, v) en coordonnées millimètriques dans le repère
    table (origine = coin TL de la table, X vers la droite, Y vers le bas).

    Retourne (x_mm, y_mm) ou None si impossible.
    """
    ray      = pixel_to_ray(u, v, K)
    point_cam = intersect_ray_with_table_plane(ray, rvec, tvec)
    if point_cam is None:
        return None
    return camera_to_table(point_cam, rvec, tvec)


# =========================================================
# OUTILS
# =========================================================

def compute_center_diagonals(pts):
    """Calcule le centre d'un quadrilatère par intersection de ses diagonales."""
    p1, p2, p3, p4 = pts

    def line(p, q):
        A = q[1] - p[1]
        B = p[0] - q[0]
        C = A * p[0] + B * p[1]
        return A, B, C

    A1, B1, C1 = line(p1, p3)
    A2, B2, C2 = line(p2, p4)

    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return (x, y)


def get_marker_centers(corners, ids):
    """Retourne un dict {marker_id: (cx, cy)} pour chaque tag détecté."""
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]
        center = compute_center_diagonals(pts)
        if center is not None:
            cx, cy = center
        else:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
        centers[int(marker_id)] = (cx, cy)
    return centers


def save_pose(rvec, tvec, new_K):
    """
    Sauvegarde la pose de la table.

    new_K est la matrice après getOptimalNewCameraMatrix ; c'est elle qui
    doit être utilisée lors des appels ultérieurs à pixel_to_table_mm
    (avec dist = zeros, image préalablement undistordue).
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "camera_matrix": new_K.tolist(),
        "dist_coeffs": np.zeros((5, 1)).tolist(),
        "table_size_mm": TABLE_SIZE_MM,
        "tag_offset_mm": TAG_OFFSET_MM,
        "tag_ids": TAG_IDS,
        "note": (
            "camera_matrix = new_K (après getOptimalNewCameraMatrix). "
            "dist_coeffs = 0 car l'image est préalablement undistordue. "
            "Le repère table a son origine au coin TL de la table (pas au centre du tag TL). "
            f"Les centres des tags sont à TAG_OFFSET_MM={TAG_OFFSET_MM} mm des bords."
        )
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Pose sauvegardée dans {OUTPUT_JSON}")


def run_validation_tests(centers, rvec, tvec, new_K, frame_undist):
    """
    Tests de validation après capture.

    On vérifie que les centres des tags sont bien projetés aux positions
    attendues (TAG_OFFSET_MM des bords), et que les coins de la table
    (0,0), (580,0), (580,580), (0,580) sont cohérents par projection inverse.
    """
    o = TAG_OFFSET_MM
    s = TABLE_SIZE_MM
    dist_zero = np.zeros((5, 1), dtype=np.float64)

    # ------------------------------------------------------------------
    print("\n=== TEST CENTRES DES TAGS (doivent être proches de ±TAG_OFFSET_MM) ===")
    expected_tag_mm = {
        "TL": (o,     o    ),
        "TR": (s - o, o    ),
        "BR": (s - o, s - o),
        "BL": (o,     s - o),
    }
    errors = []
    for name, tag_id in TAG_IDS.items():
        cx, cy = centers[tag_id]
        result = pixel_to_table_mm(cx, cy, rvec, tvec, new_K)
        if result is None:
            print(f"  {name} -> ERREUR intersection")
            continue
        x_mm, y_mm = result
        ex, ey = expected_tag_mm[name]
        err = np.sqrt((x_mm - ex) ** 2 + (y_mm - ey) ** 2)
        errors.append(err)
        print(f"  {name} : pixel ({cx:.1f}, {cy:.1f}) "
              f"-> ({x_mm:7.2f}, {y_mm:7.2f}) mm  "
              f"[attendu ({ex:.1f}, {ey:.1f})]  erreur = {err:.2f} mm")

    if errors:
        print(f"  Erreur moyenne : {np.mean(errors):.2f} mm | max : {np.max(errors):.2f} mm")

    # ------------------------------------------------------------------
    print("\n=== TEST COINS DE LA TABLE (projection inverse) ===")
    corner_pts_3d = {
        "TL": [0, 0,     0],
        "TR": [s, 0,     0],
        "BR": [s, s,     0],
        "BL": [0, s,     0],
    }
    for name, pt in corner_pts_3d.items():
        pt_3d = np.array([pt], dtype=np.float64)
        projected, _ = cv2.projectPoints(pt_3d, rvec, tvec, new_K, dist_zero)
        px, py = projected[0][0]
        print(f"  Coin {name} ({pt[0]:.0f}, {pt[1]:.0f}) mm -> pixel ({px:.1f}, {py:.1f})")

    # ------------------------------------------------------------------
    print("\n=== TEST CENTRE IMAGE ===")
    h, w = frame_undist.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0
    result = pixel_to_table_mm(cx_img, cy_img, rvec, tvec, new_K)
    if result is not None:
        x_mm, y_mm = result
        print(f"  Pixel ({cx_img:.0f}, {cy_img:.0f}) -> ({x_mm:.1f}, {y_mm:.1f}) mm")
        print(f"  (Centre géométrique table = {s/2:.1f}, {s/2:.1f} mm)")
        print(f"  Décalage caméra/centre table : "
              f"Δx={x_mm - s/2:.1f} mm, Δy={y_mm - s/2:.1f} mm")
    else:
        print("  Impossible de calculer l'intersection")

    # ------------------------------------------------------------------
    print("\n=== TEST CENTRE TABLE (290, 290) → pixel (projection inverse) ===")
    pt_3d = np.array([[s / 2, s / 2, 0.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(pt_3d, rvec, tvec, new_K, dist_zero)
    px, py = projected[0][0]
    print(f"  Centre table ({s/2:.0f}, {s/2:.0f}) mm -> pixel ({px:.1f}, {py:.1f})")


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

    # Lire une frame pour initialiser new_K
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Résolution : {w}x{h}")

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    print(f"[INFO] new_K diagonal : fx={new_K[0,0]:.1f}, fy={new_K[1,1]:.1f}, "
          f"cx={new_K[0,2]:.1f}, cy={new_K[1,2]:.1f}")

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Appuie sur 'c' pour capturer et calculer la pose | 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistortion avec new_K
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
                    cv2.putText(display, name, (int(cx) + 5, int(cy) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(display, "4 TAGS OK - appuie sur 'c'",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                detected = [int(i) for i in ids.flatten()]
                cv2.putText(display, f"Tags detectes: {detected}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

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
                missing = [name for name, tid in TAG_IDS.items() if tid not in centers]
                print(f"[ERREUR] Tags manquants : {missing}")
                continue

            print("\n[INFO] Calcul de la pose en cours...")
            rvec, tvec = estimate_table_pose(centers, new_K)

            if rvec is None or tvec is None:
                print("[ERREUR] solvePnP a échoué")
                continue

            print(f"  rvec : {rvec.flatten()}")
            print(f"  tvec : {tvec.flatten()} mm")

            run_validation_tests(centers, rvec, tvec, new_K, frame_undist)
            save_pose(rvec, tvec, new_K)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()