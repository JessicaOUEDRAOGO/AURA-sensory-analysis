# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMS
# =========================================================
CAMERA_ID  = 0
TEST_TAG_ID = 7

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR   = PROJECT_ROOT / "config"

POSE_PATH = CONFIG_DIR / "cambottom_table_pose.json"

# =========================================================
# LOAD POSE
# =========================================================
def load_pose(path: Path):
    """
    Charge rvec, tvec, K (new_K) et dist depuis le JSON de pose.
    dist est zéro car le script de calibration sauvegarde new_K +
    dist=zeros (image préalablement undistordue).
    On charge aussi la calibration brute (K_raw, dist_raw) pour
    effectuer l'undistortion dans ce script.
    """
    with open(path, "r") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"],           dtype=np.float64)
    tvec = np.array(data["tvec"],           dtype=np.float64)
    K    = np.array(data["camera_matrix"],  dtype=np.float64)  # new_K
    dist = np.array(data["dist_coeffs"],    dtype=np.float64)  # zeros

    return rvec, tvec, K, dist


def load_camera_calibration(config_dir: Path):
    """
    Charge la calibration caméra brute (K_raw, dist_raw) nécessaire
    pour undistordre les frames avant détection.
    """
    calib_path = config_dir / "camera_calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration introuvable : {calib_path}")
    with open(calib_path, "r") as f:
        data = json.load(f)
    K_raw    = np.array(data["camera_matrix"], dtype=np.float64)
    dist_raw = np.array(data["dist_coeffs"],   dtype=np.float64)
    return K_raw, dist_raw


# =========================================================
# GÉOMÉTRIE
# =========================================================
def pixel_to_ray(u, v, K):
    """Rayon unitaire passant par le pixel (u, v) dans le repère caméra."""
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def intersect_ray_plane(ray, rvec, tvec):
    """
    Intersecte le rayon (origine = centre optique caméra) avec le plan
    de la table (Z = 0 dans le repère table).

    Retourne le point d'intersection dans le repère caméra, ou None.
    """
    R, _         = cv2.Rodrigues(rvec)
    normal       = R[:, 2]            # axe Z table dans repère caméra
    plane_origin = tvec.reshape(3)    # origine table dans repère caméra
    ray_origin   = np.zeros(3)        # centre optique caméra

    denom = np.dot(normal, ray)
    if abs(denom) < 1e-9:
        return None  # rayon parallèle au plan

    t = np.dot(normal, plane_origin - ray_origin) / denom
    if t < 0:
        return None  # intersection derrière la caméra

    return ray_origin + t * ray


def camera_to_table(pt_cam, rvec, tvec):
    """
    Transforme un point du repère caméra vers le repère table.
    Retourne (x_mm, y_mm) avec origine = coin TL de la table.
    """
    R, _ = cv2.Rodrigues(rvec)
    pt   = R.T @ (pt_cam - tvec.reshape(3))
    return float(pt[0]), float(pt[1])


def pixel_to_table(u, v, rvec, tvec, K):
    """
    Convertit un pixel (u, v) — dans l'image undistordue — en
    coordonnées millimètriques dans le repère table.
    Retourne (x_mm, y_mm) ou None.
    """
    ray    = pixel_to_ray(u, v, K)
    pt_cam = intersect_ray_plane(ray, rvec, tvec)
    if pt_cam is None:
        return None
    return camera_to_table(pt_cam, rvec, tvec)


# =========================================================
# UTILITAIRES
# =========================================================
def get_center(pts):
    """Centre d'un quadrilatère par intersection des diagonales (robuste)."""
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
        # Fallback sur la moyenne
        return np.mean(pts, axis=0)
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return np.array([x, y])


# =========================================================
# MAIN
# =========================================================
def main():
    # --- Chargement -------------------------------------------------------
    rvec, tvec, new_K, _ = load_pose(POSE_PATH)

    # Calibration brute nécessaire pour l'undistortion
    try:
        K_raw, dist_raw = load_camera_calibration(CONFIG_DIR)
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        return

    # --- Caméra -----------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra")
        return

    # Calculer new_K une seule fois à partir de la résolution réelle
    ret, frame0 = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra")
        cap.release()
        return

    h, w = frame0.shape[:2]
    # new_K doit être cohérent avec celui sauvegardé lors de la calibration.
    # On le recalcule pour être sûr (même alpha=1, même taille).
    new_K_check, _ = cv2.getOptimalNewCameraMatrix(K_raw, dist_raw, (w, h), 1, (w, h))

    # Vérification rapide que new_K chargé ≈ new_K recalculé
    diff = np.max(np.abs(new_K - new_K_check))
    if diff > 1.0:
        print(f"[AVERTISSEMENT] new_K chargé et new_K recalculé diffèrent de {diff:.2f} px. "
              f"Vérifiez la résolution caméra utilisée lors de la calibration.")
    else:
        print(f"[OK] new_K cohérent (écart max = {diff:.4f} px)")

    # --- ArUco ------------------------------------------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    print(f"\n[TEST] Bouge le tag {TEST_TAG_ID} au-dessus de la table.\n"
          f"       Appuie sur 'q' pour quitter.\n")

    last_xy = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Undistortion (OBLIGATOIRE, cohérent avec la calibration) -----
        frame_undist = cv2.undistort(frame, K_raw, dist_raw, None, new_K)

        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != TEST_TAG_ID:
                    continue

                pts       = corners[i][0]
                cx, cy    = get_center(pts)

                result = pixel_to_table(cx, cy, rvec, tvec, new_K)

                if result is not None:
                    x_mm, y_mm = result

                    # Marqueur visuel sur le centre du tag
                    cv2.circle(display, (int(cx), int(cy)), 6, (0, 0, 255), -1)

                    # Coordonnées table
                    cv2.putText(display,
                                f"X={x_mm:.1f} mm   Y={y_mm:.1f} mm",
                                (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Drift (stabilité inter-frame)
                    if last_xy is not None:
                        dx    = x_mm - last_xy[0]
                        dy    = y_mm - last_xy[1]
                        drift = np.hypot(dx, dy)
                        color = (0, 255, 0) if drift < 1.0 else (0, 165, 255) if drift < 5.0 else (0, 0, 255)
                        cv2.putText(display,
                                    f"Drift={drift:.2f} mm",
                                    (50, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                    last_xy = (x_mm, y_mm)
                else:
                    cv2.putText(display, "Hors plan table",
                                (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Test pose table", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()