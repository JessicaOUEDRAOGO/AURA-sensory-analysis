# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0

# Mets ici la vraie résolution du projecteur
PROJECTOR_RES = (3840, 2160)

# Rectangle "sûr" DANS l'image projecteur qui tombe sur la surface utile
# A AJUSTER si besoin après la prévisualisation
PROJ_SAFE_RECT = (1400, 870, 2500, 1380)   # (x_min, y_min, x_max, y_max)

GRID_ROWS = 3
GRID_COLS = 3

CALIB_TAG_ID = 7

# Balance fisheye : doit être cohérente avec celle utilisée pour la pose bottom
FISHEYE_BALANCE = 0.3

# Taille visuelle de la croix projetée
CROSS_RADIUS = 22
CROSS_HALF = 34
CROSS_THICKNESS = 3

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

POSE_PATH = CONFIG_DIR / "cambottom_table_pose.json"
FISHEYE_CALIB_PATH = CONFIG_DIR / "camera_calibration_fisheye.json"
OUTPUT_PATH = CONFIG_DIR / "H_table_to_proj.json"

# =========================================================
# OUTILS JSON
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================
# CHARGEMENT CALIBRATIONS
# =========================================================
def load_pose():
    data = load_json(POSE_PATH)
    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)
    new_K_pose = np.array(data["camera_matrix"], dtype=np.float64)
    return rvec, tvec, new_K_pose

def load_fisheye_calibration():
    data = load_json(FISHEYE_CALIB_PATH)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, D

# =========================================================
# GÉNÉRATION DES POINTS PROJECTEUR
# =========================================================
def generate_projector_grid(rect, rows=3, cols=3):
    x_min, y_min, x_max, y_max = rect

    if not (0 <= x_min < x_max <= PROJECTOR_RES[0] and 0 <= y_min < y_max <= PROJECTOR_RES[1]):
        raise ValueError(
            f"PROJ_SAFE_RECT={rect} invalide pour PROJECTOR_RES={PROJECTOR_RES}"
        )

    xs = np.linspace(x_min, x_max, cols)
    ys = np.linspace(y_min, y_max, rows)

    pts = []
    for y in ys:
        for x in xs:
            pts.append((int(round(x)), int(round(y))))
    return pts

# =========================================================
# GÉOMÉTRIE PIXEL -> TABLE (avec new_K sur image rectifiée)
# =========================================================
def pixel_to_ray(u, v, K):
    K_inv = np.linalg.inv(K)
    pt = np.array([u, v, 1.0], dtype=np.float64)
    ray = K_inv @ pt
    return ray / np.linalg.norm(ray)

def intersect_ray_with_table_plane(ray, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]                 # axe Z table dans le repère caméra
    plane_origin = tvec.reshape(3)   # origine table dans le repère caméra
    ray_origin = np.zeros(3, dtype=np.float64)

    denom = np.dot(normal, ray)
    if abs(denom) < 1e-9:
        return None

    t = np.dot(normal, plane_origin - ray_origin) / denom
    if t < 0:
        return None

    return ray_origin + t * ray

def camera_to_table(point_cam, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    point_table = R.T @ (point_cam - tvec.reshape(3))
    return float(point_table[0]), float(point_table[1])

def pixel_to_table_mm(u, v, rvec, tvec, K):
    ray = pixel_to_ray(u, v, K)
    point_cam = intersect_ray_with_table_plane(ray, rvec, tvec)
    if point_cam is None:
        return None
    return camera_to_table(point_cam, rvec, tvec)

# =========================================================
# ARUCO
# =========================================================
def compute_center_diagonals(pts):
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
    return float(x), float(y)

def get_marker_center(marker_corners):
    pts = marker_corners[0]  # shape (4,2)
    center = compute_center_diagonals(pts)
    if center is not None:
        return center
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

# =========================================================
# AFFICHAGE PROJECTEUR
# =========================================================
def draw_cross(img, point, color=(255, 255, 255), label=None):
    x, y = int(point[0]), int(point[1])
    cv2.circle(img, (x, y), CROSS_RADIUS, color, CROSS_THICKNESS)
    cv2.line(img, (x - CROSS_HALF, y), (x + CROSS_HALF, y), color, CROSS_THICKNESS)
    cv2.line(img, (x, y - CROSS_HALF), (x, y + CROSS_HALF), color, CROSS_THICKNESS)
    if label is not None:
        cv2.putText(
            img, str(label), (x + 24, y - 24),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
        )

def create_projection_image_single(point, idx, total):
    img = np.zeros((PROJECTOR_RES[1], PROJECTOR_RES[0], 3), dtype=np.uint8)
    draw_cross(img, point, color=(255, 255, 255), label=idx + 1)
    txt = f"Point {idx+1}/{total}  |  c / Espace / Entree = capturer   |   q / ESC = quitter"
    cv2.putText(img, txt, (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def create_projection_image_all(points):
    img = np.zeros((PROJECTOR_RES[1], PROJECTOR_RES[0], 3), dtype=np.uint8)
    for i, p in enumerate(points):
        draw_cross(img, p, color=(255, 255, 255), label=i + 1)
    cv2.putText(
        img,
        "Previsualisation grille - appuie sur une touche dans la fenetre CAMERA pour commencer",
        (80, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )
    return img

# =========================================================
# FENÊTRES
# =========================================================
def setup_windows():
    cv2.namedWindow("PROJECTOR", cv2.WINDOW_NORMAL)
    cv2.namedWindow("CAMERA", cv2.WINDOW_NORMAL)

    # Essaie de forcer la fenêtre projecteur en plein écran.
    cv2.setWindowProperty("PROJECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def key_is_capture(key):
    return key in (ord('c'), ord('C'), 13, 32)  # c, C, Entrée, Espace

def key_is_quit(key):
    return key in (ord('q'), ord('Q'), 27)      # q, Q, ESC

# =========================================================
# MAIN
# =========================================================
def main():
    # ---------- chargements ----------
    rvec, tvec, new_K_pose = load_pose()
    K_fish, D_fish = load_fisheye_calibration()
    projector_points = generate_projector_grid(PROJ_SAFE_RECT, GRID_ROWS, GRID_COLS)

    print("PROJECTOR_POINTS générés :")
    for i, p in enumerate(projector_points, start=1):
        print(f"  {i:02d}: {p}")

    # ---------- caméra ----------
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERREUR] Impossible de lire une frame caméra")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Résolution caméra : {w}x{h}")

    # ---------- maps fisheye ----------
    # On recalcule new_K runtime avec le même balance que la pose.
    new_K_runtime = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_fish, D_fish, (w, h), np.eye(3), balance=FISHEYE_BALANCE
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_fish, D_fish, np.eye(3), new_K_runtime, (w, h), cv2.CV_16SC2
    )

    # Vérification cohérence
    diff = np.abs(new_K_runtime - new_K_pose)
    print(f"[INFO] new_K runtime fx={new_K_runtime[0,0]:.3f}, fy={new_K_runtime[1,1]:.3f}, "
          f"cx={new_K_runtime[0,2]:.3f}, cy={new_K_runtime[1,2]:.3f}")
    print(f"[INFO] max |new_K_runtime - new_K_pose| = {diff.max():.6f}")

    # ---------- aruco ----------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    # ---------- fenêtres ----------
    setup_windows()

    # ---------- prévisualisation ----------
    preview = create_projection_image_all(projector_points)
    cv2.imshow("PROJECTOR", preview)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            for j, marker_id in enumerate(ids.flatten()):
                cx, cy = get_marker_center(corners[j])
                color = (0, 0, 255) if int(marker_id) == CALIB_TAG_ID else (0, 255, 255)
                cv2.circle(display, (int(round(cx)), int(round(cy))), 6, color, -1)
                cv2.putText(display, f"ID {int(marker_id)}", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.putText(display,
                    "Verifier que les croix tombent sur la surface utile. Appuie dans cette fenetre pour commencer.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("CAMERA", display)

        key = cv2.waitKey(30) & 0xFF
        if key != 255:
            break

    if key_is_quit(key):
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Calibration annulée.")
        return

    print("\n=== CALIBRATION H_table_to_proj ===")
    print("Place le tag au centre de chaque croix.")
    print("Capture : c / Espace / Entree | Quitter : q / ESC\n")

    table_pts = []
    proj_pts = []

    total = len(projector_points)

    for i, proj_pt in enumerate(projector_points):
        while True:
            proj_img = create_projection_image_single(proj_pt, i, total)
            cv2.imshow("PROJECTOR", proj_img)

            ret, frame = cap.read()
            if not ret:
                continue

            # IMPORTANT : rectification fisheye avant détection
            frame_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            display = frame_undist.copy()

            tag_found = False
            tag_center = None

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                for j, marker_id in enumerate(ids.flatten()):
                    cx, cy = get_marker_center(corners[j])

                    if int(marker_id) == CALIB_TAG_ID:
                        tag_found = True
                        tag_center = (cx, cy)
                        cv2.circle(display, (int(round(cx)), int(round(cy))), 7, (0, 0, 255), -1)
                        mm = pixel_to_table_mm(cx, cy, rvec, tvec, new_K_runtime)
                        if mm is not None:
                            txt = f"ID {CALIB_TAG_ID}  ->  ({mm[0]:.1f}, {mm[1]:.1f}) mm"
                        else:
                            txt = f"ID {CALIB_TAG_ID}  ->  conversion impossible"
                        cv2.putText(display, txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(display, (int(round(cx)), int(round(cy))), 5, (0, 255, 255), -1)

            cv2.putText(display,
                        f"Point {i+1}/{total} | tag attendu ID={CALIB_TAG_ID}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        "c/Espace/Entree = capturer | q/ESC = quitter",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            status = "TAG OK" if tag_found else "TAG NON DETECTE"
            status_color = (0, 255, 0) if tag_found else (0, 0, 255)
            cv2.putText(display, status, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

            cv2.imshow("CAMERA", display)

            key = cv2.waitKey(30) & 0xFF

            if key_is_quit(key):
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Calibration annulée.")
                return

            if key_is_capture(key):
                if not tag_found or tag_center is None:
                    print(f"[Point {i+1}/{total}] ❌ Tag ID={CALIB_TAG_ID} non détecté")
                    continue

                cx, cy = tag_center
                mm = pixel_to_table_mm(cx, cy, rvec, tvec, new_K_runtime)

                if mm is None:
                    print(f"[Point {i+1}/{total}] ❌ Conversion pixel->table impossible")
                    continue

                x_mm, y_mm = mm
                print(f"[Point {i+1}/{total}] ✔ table=({x_mm:.2f}, {y_mm:.2f}) mm <-> proj={proj_pt}")

                table_pts.append([x_mm, y_mm])
                proj_pts.append([float(proj_pt[0]), float(proj_pt[1])])
                break

    cap.release()
    cv2.destroyAllWindows()

    # ---------- calcul homographie ----------
    table_pts = np.array(table_pts, dtype=np.float32)
    proj_pts = np.array(proj_pts, dtype=np.float32)

    if len(table_pts) < 4:
        print("[ERREUR] Pas assez de points pour calculer une homographie")
        return

    H, mask = cv2.findHomography(table_pts, proj_pts, method=0)

    if H is None:
        print("[ERREUR] findHomography a échoué")
        return

    print("\n=== MATRICE H_table_to_proj ===")
    print(H)

    # ---------- erreur de reprojection ----------
    reproj = cv2.perspectiveTransform(table_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = np.linalg.norm(reproj - proj_pts, axis=1)

    print("\n=== ERREURS DE REPROJECTION ===")
    for i, e in enumerate(errs, start=1):
        print(f"  Point {i:02d} : {e:.2f} px")
    print(f"  Moyenne : {errs.mean():.2f} px")
    print(f"  Max     : {errs.max():.2f} px")

    # ---------- sauvegarde ----------
    data = {
        "H_table_to_proj": H.tolist(),
        "points_table_mm": table_pts.tolist(),
        "points_projector_px": proj_pts.tolist(),
        "projector_resolution": list(PROJECTOR_RES),
        "proj_safe_rect": list(PROJ_SAFE_RECT),
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "calib_tag_id": CALIB_TAG_ID,
        "fisheye_balance": FISHEYE_BALANCE,
        "mean_reprojection_error_px": float(errs.mean()),
        "max_reprojection_error_px": float(errs.max()),
        "note": (
            "Homographie calculee a partir de correspondances reelles "
            "table(mm) -> projecteur(px). "
            "Detection ArUco effectuee sur image fisheye rectifiee."
        )
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\n[OK] Sauvegardé dans {OUTPUT_PATH}")

if __name__ == "__main__":
    main()