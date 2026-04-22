# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

# =========================================================
# IMPORT DES FONCTIONS EXISTANTES
# =========================================================
from src.core.calibration.pose_top_cam_pixel_to_mm import (
    get_marker_centers,
    pixel_to_table_mm,
    load_camera_calibration,
)

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 1

# Taille du repère graphique de sortie (pixels).
GRID_SIZE = 700

# IDs des tags de calibration à ignorer lors de la détection
CALIBRATION_TAG_IDS = {41, 40, 43, 42}

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR   = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration_top.json"
POSE_PATH         = CONFIG_DIR / "camtop_table_pose.json"


# =========================================================
# CHARGEMENT POSE
# =========================================================
def load_top_pose(path: Path):
    """
    Charge rvec, tvec et table_size_mm depuis le JSON de pose.
    Toutes les valeurs géométriques viennent du fichier —
    aucune constante hardcodée.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec          = np.array(data["rvec"], dtype=np.float64)
    tvec          = np.array(data["tvec"], dtype=np.float64)
    table_size_mm = float(data["table_size_mm"])

    return rvec, tvec, table_size_mm


# =========================================================
# TRANSFORMATIONS
# =========================================================

def top_to_bottom_frame(x_top: float, y_top: float,
                         table_size_mm: float) -> tuple[float, float]:
    """
    Convertit des coordonnées mm du repère caméra TOP vers le repère
    caméra BOTTOM, d'après la correspondance physique observée :

        Top  TL (0, 0) → Bottom BL (0, S)
        Top  TR (S, 0) → Bottom BR (S, S)
        Top  BR (S, S) → Bottom TR (S, 0)
        Top  BL (0, S) → Bottom TL (0, 0)

    La transformation est un flip Y pur :
        x_bottom = x_top
        y_bottom = S - y_top

    Les deux repères partagent le même axe X (gauche→droite identique).
    Seul l'axe Y est inversé car les caméras se font face verticalement.
    """
    return x_top, table_size_mm - y_top


def mm_to_graph(x_mm: float, y_mm: float,
                table_size_mm: float) -> tuple[float, float]:
    """
    Convertit des coordonnées mm dans le repère bottom (ou tout repère
    normalisé sur [0, table_size_mm]) en coordonnées graphiques
    [0, GRID_SIZE].
    """
    xg = (x_mm / table_size_mm) * GRID_SIZE
    yg = (y_mm / table_size_mm) * GRID_SIZE
    return xg, yg


def top_pixel_to_graph(u: float, v: float,
                        rvec, tvec,
                        new_K,
                        table_size_mm: float):
    """
    Pipeline complet : pixel caméra top → coordonnées graphiques
    dans le repère caméra bottom.

        1. pixel (u,v)  →  (x_top, y_top) mm   [repère table top]
        2. flip Y       →  (x_bot, y_bot) mm    [repère table bottom]
        3. normalisation → (xg, yg)             [repère graphique 0..GRID_SIZE]

    Retourne (xg, yg, x_top, y_top, x_bot, y_bot) ou None si impossible.
    """
    result = pixel_to_table_mm(u, v, rvec, tvec, new_K)
    if result is None:
        return None

    x_top, y_top = result
    x_bot, y_bot = top_to_bottom_frame(x_top, y_top, table_size_mm)
    xg,    yg    = mm_to_graph(x_bot, y_bot, table_size_mm)

    return xg, yg, x_top, y_top, x_bot, y_bot


# =========================================================
# MAIN
# =========================================================
def main():

    # --- Calibration caméra brute (pour undistortion) ---
    K, dist = load_camera_calibration(CAMERA_CALIB_PATH)

    # --- Ouvrir caméra ---
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] Caméra non ouverte")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra")
        cap.release()
        return

    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    print(f"[INFO] Résolution : {w}x{h}")
    print(f"[INFO] new_K : fx={new_K[0,0]:.1f}, fy={new_K[1,1]:.1f}, "
          f"cx={new_K[0,2]:.1f}, cy={new_K[1,2]:.1f}")

    # --- Charger pose ---
    rvec, tvec, table_size_mm = load_top_pose(POSE_PATH)
    print(f"[INFO] TABLE_SIZE_MM (JSON)  : {table_size_mm} mm")
    print(f"[INFO] GRID_SIZE             : {GRID_SIZE} px")
    print(f"[INFO] Résolution graphique  : {GRID_SIZE / table_size_mm:.3f} px/mm")

    # --- Vérification de la transformation sur les 4 coins au démarrage ---
    print("\n[CHECK] Correspondance coins top → bottom → graph :")
    s = table_size_mm
    corners_check = [
        ("TL", 0, 0, "BL", 0,   s  ),
        ("TR", s, 0, "BR", s,   s  ),
        ("BR", s, s, "TR", s,   0  ),
        ("BL", 0, s, "TL", 0,   0  ),
    ]
    all_ok = True
    for top_name, xt, yt, expected_bot_name, ex_xb, ex_yb in corners_check:
        xb, yb = top_to_bottom_frame(xt, yt, s)
        xg, yg = mm_to_graph(xb, yb, s)
        ok = abs(xb - ex_xb) < 1e-6 and abs(yb - ex_yb) < 1e-6
        all_ok = all_ok and ok
        status = "OK" if ok else "ERREUR"
        print(f"  [{status}] Top {top_name:2s} ({xt:5.1f},{yt:5.1f}) "
              f"→ bottom ({xb:5.1f},{yb:5.1f}) mm "
              f"→ graph ({xg:6.1f},{yg:6.1f}) px "
              f"[attendu bottom {expected_bot_name} ({ex_xb:.1f},{ex_yb:.1f})]")
    if all_ok:
        print("  → Transformation correcte.\n")
    else:
        print("  → PROBLÈME dans la transformation, vérifier top_to_bottom_frame().\n")

    # --- ArUco ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    print("[INFO] Détection en cours... (q pour quitter)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistortion obligatoire
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)
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

                # Pipeline complet
                res = top_pixel_to_graph(u, v, rvec, tvec, new_K, table_size_mm)
                if res is None:
                    continue

                xg, yg, x_top, y_top, x_bot, y_bot = res

                in_table = (0.0 <= x_bot <= table_size_mm) and \
                           (0.0 <= y_bot <= table_size_mm)

                dot_color = (0, 255, 0) if in_table else (0, 0, 255)
                cv2.circle(display, (int(u), int(v)), 6, dot_color, -1)

                cv2.putText(display,
                            f"ID {marker_id}",
                            (int(u)+10, int(v)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display,
                            f"top=({x_top:.1f}, {y_top:.1f}) mm",
                            (int(u)+10, int(v)+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(display,
                            f"bot=({x_bot:.1f}, {y_bot:.1f}) mm",
                            (int(u)+10, int(v)+32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                cv2.putText(display,
                            f"graph=({xg:.1f}, {yg:.1f})",
                            (int(u)+10, int(v)+49),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

                print(f"ID {marker_id:3d} | "
                      f"top=({x_top:7.2f},{y_top:7.2f}) mm | "
                      f"bot=({x_bot:7.2f},{y_bot:7.2f}) mm | "
                      f"graph=({xg:7.2f},{yg:7.2f})"
                      + ("" if in_table else "  [HORS TABLE]"))

        cv2.imshow("Top → Bottom frame → Graph", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
