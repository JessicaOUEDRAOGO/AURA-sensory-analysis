# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time

import cv2
import numpy as np
from screeninfo import get_monitors


# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_SCREEN_ID = 1

PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# dimensions physiques utiles de l'écran diffusant
SCREEN_WIDTH_MM = 600.0
SCREEN_HEIGHT_MM = 600.0

# taille de la grille logique de ton appli
GRID_SIZE = 700

# rectangle blanc affiché pour aider visuellement
DISPLAY_RECT_WIDTH = 2200
DISPLAY_RECT_HEIGHT = 2200
DISPLAY_BG_LEVEL = 220
DISPLAY_BORDER_THICKNESS = 8


# =========================================================
# POSITIONS PHYSIQUES DES POINTS DE REFERENCE (en mm)
# Ces points doivent correspondre EXACTEMENT aux coins de tags
# que tu sélectionnes dans detect_corner_tags().
# Repère écran :
#   origine = coin haut-gauche de la surface utile
#   x vers la droite, y vers le bas
# =========================================================
REF_TL_MM = (10.0, 10.0)
REF_TR_MM = (590.0, 10.0)
REF_BR_MM = (580.0, 580.0)
REF_BL_MM = (10.0, 580.0)
# IDs ArUco des 4 coins de la surface utile
CORNER_TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}
AUTO_SEARCH_NUM_FRAMES = 120
AUTO_SEARCH_FLUSH = 1

# chemins
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
PROJECTOR_CALIB_PATH = CONFIG_DIR / "projector_calibration_moreno_refined.json"
STEREO_CALIB_PATH = CONFIG_DIR / "stereo_camera_projector_calibration.json"

OUTPUT_CALIB_PATH = CONFIG_DIR / "calibration_data.json"
OUTPUT_POSE_PATH = CONFIG_DIR / "screen_pose_manual.json"
OUTPUT_DEBUG_PATH = CONFIG_DIR / "manual_pose_debug.png"
RUNTIME_MAPPING_PATH = CONFIG_DIR / "runtime_mapping.json"


# =========================================================
# OUTILS JSON / CALIB
# =========================================================
def ensure_output_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Cette fonction normalise les coefficients de distorsion (dist) en un format cohérent pour OpenCV.
def flatten_dist(dist):
    arr = np.array(dist, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.T
    return arr

# Cette fonction charge les données de calibration depuis des fichiers JSON 
# retourne les matrices nécessaires pour la pose estimation.
def load_calibrations():
    cam_data = load_json(CAMERA_CALIB_PATH)
    proj_data = load_json(PROJECTOR_CALIB_PATH)
    stereo_data = load_json(STEREO_CALIB_PATH)
    # Matrice intrinsèque caméra (K_cam) : contient les paramètres de la caméra (longueurs focales fx, fy et centre optique cx, cy).
    K_cam = np.array(cam_data["camera_matrix"], dtype=np.float64)
    # coefficients de distorsion caméra (dist_cam) : décrivent la distorsion optique de la caméra, nécessaire pour corriger les points détectés.
    dist_cam = flatten_dist(cam_data["dist_coeffs"])
    # Matrice intrinsèque projecteur (K_proj) : contient les paramètres du projecteur, similaire à K_cam mais pour le projecteur.
    K_proj = np.array(proj_data["projector_matrix"], dtype=np.float64)
    if "projector_dist_coeffs" in proj_data:
        dist_proj = flatten_dist(proj_data["projector_dist_coeffs"])
    elif "dist_coeffs" in proj_data:
        dist_proj = flatten_dist(proj_data["dist_coeffs"])
    else:
        raise KeyError("Aucune distorsion projecteur trouvée.")
    # stereo_data : Rotation (R_cp) et translation (T_cp) caméra ↔ projecteur
    R_cp = np.array(stereo_data["R"], dtype=np.float64)
    T_cp = np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1)

    return K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp


# =========================================================
# PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectorManualPoseAruco"

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    def show(self, image: np.ndarray):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def build_display_pattern():
    """
    Mire visuelle uniquement pour aider à positionner les tags de coin.
    """
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    x0 = (PROJECTOR_WIDTH - DISPLAY_RECT_WIDTH) // 2
    y0 = (PROJECTOR_HEIGHT - DISPLAY_RECT_HEIGHT) // 2
    x1 = x0 + DISPLAY_RECT_WIDTH - 1
    y1 = y0 + DISPLAY_RECT_HEIGHT - 1

    cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        (DISPLAY_BG_LEVEL, DISPLAY_BG_LEVEL, DISPLAY_BG_LEVEL),
        thickness=-1
    )
    cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        (255, 255, 255),
        thickness=DISPLAY_BORDER_THICKNESS
    )

    for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        cv2.circle(img, p, 16, (0, 0, 255), -1)

    return img


# =========================================================
# CAMERA
# =========================================================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    for _ in range(8):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Impossible de lire une première frame caméra.")

    h, w = frame.shape[:2]
    if (w, h) != (CAMERA_WIDTH, CAMERA_HEIGHT):
        cap.release()
        raise RuntimeError(
            f"Résolution caméra incohérente : obtenu {w}x{h}, attendu {CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )

    return cap


def grab_frame(cap, n_flush=3):
    for _ in range(n_flush):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Erreur capture caméra.")

    return frame


def average_frames(cap, n_frames=8, n_flush_each=1):
    frames = []
    for _ in range(n_frames):
        fr = grab_frame(cap, n_flush=n_flush_each)
        frames.append(fr.astype(np.float32))
    return np.mean(frames, axis=0).astype(np.uint8)


# =========================================================
# OUTILS GEOMETRIE
# =========================================================
def undistort_pixel_points(points_px, K, dist):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float32)


def pose_to_homography(K, R, t):
    return K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))


def compute_reprojection_error(objpoints, imgpoints, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
    proj_2d = proj.reshape(-1, 2)
    img = np.array(imgpoints, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(img - proj_2d, axis=1)
    return float(np.mean(d)), d.tolist(), proj

def build_reference_points():
    screen_points_mm = np.array([
        [REF_TL_MM[0], REF_TL_MM[1]],
        [REF_TR_MM[0], REF_TR_MM[1]],
        [REF_BR_MM[0], REF_BR_MM[1]],
        [REF_BL_MM[0], REF_BL_MM[1]],
    ], dtype=np.float32)

    object_points_m = np.array([
        [REF_TL_MM[0] / 1000.0, REF_TL_MM[1] / 1000.0, 0.0],
        [REF_TR_MM[0] / 1000.0, REF_TR_MM[1] / 1000.0, 0.0],
        [REF_BR_MM[0] / 1000.0, REF_BR_MM[1] / 1000.0, 0.0],
        [REF_BL_MM[0] / 1000.0, REF_BL_MM[1] / 1000.0, 0.0],
    ], dtype=np.float32)

    return screen_points_mm, object_points_m
def estimate_best_pose_from_frame(frame, detector, K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp):
    # Détecte les 4 tags de coin
    detected, cam_points_raw, gray = detect_corner_tags(frame, detector)

    if cam_points_raw is None:
        return None
    # Corrige la distorsion des points détectés
    cam_points_undist = undistort_pixel_points(cam_points_raw, K_cam, dist_cam)

    screen_points_mm, object_points_m = build_reference_points()

    graph_points = np.array([
        [0, 0],
        [GRID_SIZE - 1, 0],
        [GRID_SIZE - 1, GRID_SIZE - 1],
        [0, GRID_SIZE - 1],
    ], dtype=np.float32)

    candidates = []

    for flag_name, flag in [
        ("IPPE", cv2.SOLVEPNP_IPPE),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ]:
        ok, rvec, tvec = cv2.solvePnP(
            object_points_m,
            cam_points_raw,
            K_cam,
            dist_cam,
            flags=flag
        )
        if ok:
            mean_err, per_point_errs, proj_pts = compute_reprojection_error(
                object_points_m,
                cam_points_raw,
                rvec,
                tvec,
                K_cam,
                dist_cam
            )
            candidates.append((flag_name, rvec, tvec, mean_err, per_point_errs, proj_pts))

    if not candidates:
        return None
    # Vecteur rotation écran→caméra (rvec_sc) et translation (tvec_sc) avec la meilleure précision de reprojection (mean_err) parmi les méthodes testées.
    best_flag, rvec_sc, tvec_sc, pnp_mean_err, pnp_per_point_errs, pnp_proj_pts = min(
        candidates, key=lambda x: x[3]
    )
    # R_sc : Matrice rotation écran→caméra (3×3)
    # R_sp : Matrice rotation écran→projecteur
    R_sc, _ = cv2.Rodrigues(rvec_sc)
    
    R_sp = R_cp @ R_sc
    T_sp = R_cp @ tvec_sc + T_cp

    H_screen_to_cam = pose_to_homography(K_cam, R_sc, tvec_sc)
    H_screen_to_proj = pose_to_homography(K_proj, R_sp, T_sp)

    #  Homographie caméra non-distordue → projecteur

    H_proj = H_screen_to_proj @ np.linalg.inv(H_screen_to_cam)
    H_proj = H_proj / H_proj[2, 2]
    H_inv_proj = np.linalg.inv(H_proj)

    H_graph, _ = cv2.findHomography(cam_points_undist, graph_points)
    H_graph = H_graph / H_graph[2, 2]
    H_inv_graph = np.linalg.inv(H_graph)

    H_graph_to_proj = H_proj @ H_inv_graph
    H_graph_to_proj = H_graph_to_proj / H_graph_to_proj[2, 2]

    H_proj_to_graph = np.linalg.inv(H_graph_to_proj)
    H_proj_to_graph = H_proj_to_graph / H_proj_to_graph[2, 2]

    H_cam_to_screen_mm, _ = cv2.findHomography(cam_points_undist, screen_points_mm)
    H_cam_to_screen_mm = H_cam_to_screen_mm / H_cam_to_screen_mm[2, 2]
    H_screen_mm_to_cam = np.linalg.inv(H_cam_to_screen_mm)

    return {
        "frame": frame.copy(),
        "gray": gray.copy(),
        "detected": detected,
        "cam_points_raw": cam_points_raw.copy(),
        "cam_points_undist": cam_points_undist.copy(),
        "screen_points_mm": screen_points_mm.copy(),
        "graph_points": graph_points.copy(),
        "best_flag": best_flag,
        "rvec_sc": rvec_sc.copy(),
        "tvec_sc": tvec_sc.copy(),
        "R_sc": R_sc.copy(),
        "R_sp": R_sp.copy(),
        "T_sp": T_sp.copy(),
        "pnp_mean_err": float(pnp_mean_err),
        "pnp_per_point_errs": list(pnp_per_point_errs),
        "pnp_proj_pts": pnp_proj_pts.copy(),
        "H_proj": H_proj.copy(),
        "H_inv_proj": H_inv_proj.copy(),
        "H_graph": H_graph.copy(),
        "H_inv_graph": H_inv_graph.copy(),
        "H_graph_to_proj": H_graph_to_proj.copy(),
        "H_proj_to_graph": H_proj_to_graph.copy(),
        "H_cam_to_screen_mm": H_cam_to_screen_mm.copy(),
        "H_screen_mm_to_cam": H_screen_mm_to_cam.copy(),
    }
# =========================================================
# ARUCO
# =========================================================
def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # réglages conservateurs
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def detect_corner_tags(frame, detector):
    """
    Détecte les 4 tags de coin et retourne :
    - detected : dict par ID
    - cam_points_raw : points dans l'ordre TL, TR, BR, BL
    - gray : image niveau de gris
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return {}, None, gray

    ids = ids.flatten().tolist()
    detected = {}

    for i, marker_id in enumerate(ids):
        marker_id = int(marker_id)
        if marker_id in CORNER_TAG_IDS.values():
            pts = corners[i][0].astype(np.float32)  # shape (4,2)
            center = np.mean(pts, axis=0)
            detected[marker_id] = {
                "corners": pts,
                "center": center
            }

    required_ids = [
        CORNER_TAG_IDS["TL"],
        CORNER_TAG_IDS["TR"],
        CORNER_TAG_IDS["BR"],
        CORNER_TAG_IDS["BL"],
    ]

    if not all(tag_id in detected for tag_id in required_ids):
        return detected, None, gray

    cam_points_raw = np.array([
        select_physical_corner(detected[CORNER_TAG_IDS["TL"]]["corners"], "TL"),
        select_physical_corner(detected[CORNER_TAG_IDS["TR"]]["corners"], "TR"),
        select_physical_corner(detected[CORNER_TAG_IDS["BR"]]["corners"], "BR"),
        select_physical_corner(detected[CORNER_TAG_IDS["BL"]]["corners"], "BL"),
    ], dtype=np.float32)

    return detected, cam_points_raw, gray


def draw_detected_corner_tags(img, detected, cam_points_raw=None):
    out = img.copy()

    for marker_id, data in detected.items():
        pts = data["corners"].astype(np.int32)
        center = data["center"]

        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.circle(out, tuple(np.round(center).astype(int)), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"ID {marker_id}",
            tuple(np.round(center).astype(int) + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

    if detected and all(tag_id in detected for tag_id in CORNER_TAG_IDS.values()):
        chosen = {
            "TL": detected[CORNER_TAG_IDS["TL"]]["corners"][0],
            "TR": detected[CORNER_TAG_IDS["TR"]]["corners"][1],
            "BR": detected[CORNER_TAG_IDS["BR"]]["corners"][2],
            "BL": detected[CORNER_TAG_IDS["BL"]]["corners"][3],
        }

        for label, p in chosen.items():
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(out, (x, y), 9, (0, 255, 255), 2)
            cv2.putText(
                out,
                f"{label}_sel",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    if cam_points_raw is not None and len(cam_points_raw) == 4:
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(cam_points_raw):
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(out, (x, y), 8, (255, 255, 0), 2)
            cv2.putText(
                out,
                labels[i],
                (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        cv2.polylines(out, [np.round(cam_points_raw).astype(np.int32)], True, (255, 255, 0), 2)

    help_lines = [
        "Pose ecran par ArUco",
        "Convention : TL=42, TR=43, BR=40, BL=41",
        "s: sauvegarder | r: rafraichir | q: quitter",
    ]

    y = 30
    for line in help_lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 30

    return out
# test pose camera : affiche les points détectés et reprojetés pour vérifier la précision de solvePnP.
# def show_camera_pose_debug(frame, cam_points_raw, pnp_proj_pts, title="Camera Pose Debug"):
#     """
#     Affiche dans l'image caméra :
#     - jaune  : points détectés utilisés pour solvePnP
#     - vert   : points reprojectés par solvePnP
#     - cyan   : segments entre détecté et reprojeté
#     """
#     debug = frame.copy()

#     labels = ["TL", "TR", "BR", "BL"]

#     detected_pts = np.array(cam_points_raw, dtype=np.float32).reshape(-1, 2)
#     reproj_pts = np.array(pnp_proj_pts, dtype=np.float32).reshape(-1, 2)

#     for i, (pd, pr) in enumerate(zip(detected_pts, reproj_pts)):
#         xd, yd = int(round(float(pd[0]))), int(round(float(pd[1])))
#         xr, yr = int(round(float(pr[0]))), int(round(float(pr[1])))

#         # point détecté
#         cv2.circle(debug, (xd, yd), 8, (0, 255, 255), -1)   # jaune
#         cv2.putText(
#             debug,
#             f"{labels[i]}_det",
#             (xd + 10, yd - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.65,
#             (0, 255, 255),
#             2
#         )

#         # point reprojeté
#         cv2.circle(debug, (xr, yr), 8, (0, 255, 0), 2)      # vert
#         cv2.putText(
#             debug,
#             f"{labels[i]}_rep",
#             (xr + 10, yr + 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.65,
#             (0, 255, 0),
#             2
#         )

#         # segment erreur
#         cv2.line(debug, (xd, yd), (xr, yr), (255, 255, 0), 2)

#         err = np.linalg.norm(pd - pr)
#         cv2.putText(
#             debug,
#             f"{err:.2f}px",
#             ((xd + xr) // 2 + 8, (yd + yr) // 2),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (255, 255, 0),
#             2
#         )

#     cv2.putText(
#         debug,
#         "JAUNE=detecte | VERT=reprojete solvePnP",
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (255, 255, 255),
#         2
#     )

#     return debug
# test : projection des points de calibration dans le projecteur pour vérifier qu'ils tombent sur les coins physiques correspondants.
def show_projector_pose_test_B(projector, H_proj, cam_points_raw, detector, cap, K_cam, dist_cam):
    """
    Test B :
    - rouge : projection des 4 points exacts utilisés pour la pose
    - bleu  : projection des centres des 4 tags de coin
    """

    print("\n=== TEST B : VALIDATION CAMERA -> PROJECTEUR ===")
    print("ROUGE = coins de calibration")
    print("BLEU  = centres des tags")
    print("ESC pour quitter")

    labels = ["TL", "TR", "BR", "BL"]

    while True:
        frame = grab_frame(cap, n_flush=1)
        detected, _, _ = detect_corner_tags(frame, detector)

        img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        # -------------------------------------------------
        # 1) Rouge = coins de calibration
        # -------------------------------------------------
        cam_points_undist = undistort_pixel_points(cam_points_raw, K_cam, dist_cam)

        for i, p_und in enumerate(cam_points_undist):
            proj_pt = apply_homography_to_point(H_proj, p_und)

            if proj_pt is not None:
                print(f"{labels[i]} projected = ({proj_pt[0]:.2f}, {proj_pt[1]:.2f})")

            draw_projected_point(
                img,
                proj_pt,
                color=(0, 0, 255),
                radius=16,
                label=labels[i]
            )

        # -------------------------------------------------
        # 2) Bleu = centres des tags
        # -------------------------------------------------
        ordered_ids = [
            CORNER_TAG_IDS["TL"],
            CORNER_TAG_IDS["TR"],
            CORNER_TAG_IDS["BR"],
            CORNER_TAG_IDS["BL"],
        ]

        centers_raw = []
        center_labels = []

        for marker_id in ordered_ids:
            if marker_id in detected:
                centers_raw.append(detected[marker_id]["center"])
                center_labels.append(f"ID{marker_id}")

        if len(centers_raw) > 0:
            centers_raw = np.array(centers_raw, dtype=np.float32)
            centers_undist = undistort_pixel_points(centers_raw, K_cam, dist_cam)

            for i, p_und in enumerate(centers_undist):
                proj_pt = apply_homography_to_point(H_proj, p_und)
                draw_projected_point(
                    img,
                    proj_pt,
                    color=(255, 0, 0),
                    radius=10,
                    label=center_labels[i]
                )

        cv2.putText(
            img,
            "ROUGE=coins calibration | BLEU=centres tags",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        projector.show(img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

def show_projected_validation(projector, H_graph_to_proj):
    grid_img = build_validation_grid_image()
    # applique une transformation de perspective à une image via une homographie.
    # grille déformée pour s'adapter à l'écran projecteur, affichée pour validation visuelle de la pose. 
    # Si la grille s'aligne bien, la pose est correcte.
    warped = cv2.warpPerspective(
        grid_img,
        H_graph_to_proj,
        (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    )

    while True:
        projector.show(warped)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("y"):
            return True
        elif key == ord("n") or key == ord("q") or key == 27:
            return False
# Cette fonction crée une image grille de validation de taille GRID_SIZE × GRID_SIZE (700×700) pour tester la pose estimée.
def build_validation_grid_image():
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

    # fond noir
    img[:] = (0, 0, 0)

    # contour complet de la grille logique
    cv2.rectangle(
        img,
        (0, 0),
        (GRID_SIZE - 1, GRID_SIZE - 1),
        (255, 255, 255),
        4
    )

    # grille verte
    n_div = 14
    xs = np.linspace(0, GRID_SIZE - 1, n_div + 1).astype(int)
    ys = np.linspace(0, GRID_SIZE - 1, n_div + 1).astype(int)

    for x in xs:
        cv2.line(img, (x, 0), (x, GRID_SIZE - 1), (0, 255, 0), 2)

    for y in ys:
        cv2.line(img, (0, y), (GRID_SIZE - 1, y), (0, 255, 0), 2)

    # centre rouge + label, pour vérifier que le centre de la grille est bien aligné sur l'écran projeté
    c = (GRID_SIZE // 2, GRID_SIZE // 2)
    cv2.circle(img, c, 12, (0, 0, 255), -1)
    cv2.putText(
        img,
        "CENTER",
        (c[0] - 45, c[1] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # coins
    # corners = [
    #     ((10, 10), "TL"),
    #     ((GRID_SIZE - 11, 10), "TR"),
    #     ((GRID_SIZE - 11, GRID_SIZE - 11), "BR"),
    #     ((10, GRID_SIZE - 11), "BL"),
    # ]
    # coins
    corners = [
        ((0, 0), "TL"),
        ((GRID_SIZE - 1, 0), "TR"),
        ((GRID_SIZE - 1, GRID_SIZE - 1), "BR"),
        ((0, GRID_SIZE - 1), "BL"),
    ]

    for (x, y), label in corners:
        cv2.circle(img, (x, y), 10, (255, 0, 255), -1)

        if label == "TL":
            tx, ty = x + 12, y + 28
        elif label == "TR":
            tx, ty = x - 55, y + 28
        elif label == "BR":
            tx, ty = x - 55, y - 12
        else:
            tx, ty = x + 12, y - 12

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

    return img
# test : projection d'un point rouge au centre des tag de calibration
def apply_homography_to_point(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    out = H @ p
    if abs(out[2]) < 1e-12:
        return None
    out /= out[2]
    return out[:2].astype(np.float32)


def draw_projected_point(img, pt_proj, color, radius=16, label=None):
    if pt_proj is None:
        return

    x = int(round(float(pt_proj[0])))
    y = int(round(float(pt_proj[1])))

    # On ne rejette que si le point est vraiment loin hors image
    if x < -radius or x >= PROJECTOR_WIDTH + radius or y < -radius or y >= PROJECTOR_HEIGHT + radius:
        return

    cv2.circle(img, (x, y), radius, color, -1)

    if label is not None:
        tx = x + 18
        ty = y - 12

        if ty < 25:
            ty = y + 28
        if tx > PROJECTOR_WIDTH - 80:
            tx = x - 60
        if tx < 5:
            tx = 5

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


def show_pose_debug_projection(projector, H_proj, cam_points_raw, detector, cap, K_cam, dist_cam):
    """
    Debug projection :
    - rouge  : les 4 points de calibration sélectionnés (TL, TR, BR, BL)
    - bleu   : les centres des 4 tags de coin détectés (42, 43, 40, 41)
    IMPORTANT :
    H_proj = homographie camera UNDISTORDUE -> projecteur
    donc tous les points doivent être undistordus avant projection.
    """

    print("\n[TEST] Debug projection pose")
    print("Rouge = points calibration sélectionnés")
    print("Bleu  = centres des tags de coin")
    print("ESC pour quitter")

    calib_labels = ["TL", "TR", "BR", "BL"]

    while True:
        frame = grab_frame(cap, n_flush=1)

        detected, _, _ = detect_corner_tags(frame, detector)

        img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        # -------------------------------------------------
        # 1) Points de calibration sélectionnés
        # -------------------------------------------------
        cam_points_undist = undistort_pixel_points(cam_points_raw, K_cam, dist_cam)

        for i, p_und in enumerate(cam_points_undist):
            proj_pt = apply_homography_to_point(H_proj, p_und)
            draw_projected_point(
                img,
                proj_pt,
                color=(0, 0, 255),   # rouge
                radius=16,
                label=f"CAL_{calib_labels[i]}"
            )

        # -------------------------------------------------
        # 2) Centres des 4 tags de coin détectés
        # -------------------------------------------------
        ordered_ids = [
            CORNER_TAG_IDS["TL"],
            CORNER_TAG_IDS["TR"],
            CORNER_TAG_IDS["BR"],
            CORNER_TAG_IDS["BL"],
        ]

        centers_raw = []
        center_labels = []

        for marker_id in ordered_ids:
            if marker_id in detected:
                centers_raw.append(detected[marker_id]["center"])
                center_labels.append(f"ID{marker_id}")

        if len(centers_raw) > 0:
            centers_raw = np.array(centers_raw, dtype=np.float32)
            centers_undist = undistort_pixel_points(centers_raw, K_cam, dist_cam)

            for i, p_und in enumerate(centers_undist):
                proj_pt = apply_homography_to_point(H_proj, p_und)
                draw_projected_point(
                    img,
                    proj_pt,
                    color=(255, 0, 0),   # bleu
                    radius=11,
                    label=center_labels[i]
                )

        # -------------------------------------------------
        # 3) Texte d'aide
        # -------------------------------------------------
        cv2.putText(
            img,
            "ROUGE = calibration points | BLEU = tag centers",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        projector.show(img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        # fin test
    
def select_physical_corner(corners, position):
    # corners shape (4,2)

    if position == "TL":
        return corners[np.argmin(corners[:,0] + corners[:,1])]

    elif position == "TR":
        return corners[np.argmin(-corners[:,0] + corners[:,1])]

    elif position == "BR":
        return corners[np.argmax(corners[:,0] + corners[:,1])]

    elif position == "BL":
        return corners[np.argmax(-corners[:,0] + corners[:,1])]    

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    print("=== POSE ECRAN PAR ARUCO ===")
    print("Convention utilisee :")
    print(f"  TL = {CORNER_TAG_IDS['TL']}")
    print(f"  TR = {CORNER_TAG_IDS['TR']}")
    print(f"  BR = {CORNER_TAG_IDS['BR']}")
    print(f"  BL = {CORNER_TAG_IDS['BL']}")

    K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp = load_calibrations()

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    projector.show(build_display_pattern())

    cap = open_camera()
    detector = create_aruco_detector()

    try:
        time.sleep(1.0)

        print(f"\nRecherche automatique sur {AUTO_SEARCH_NUM_FRAMES} frames...")
        cv2.namedWindow("Manual Screen Pose", cv2.WINDOW_NORMAL)

        best_result = None
        valid_count = 0

        for idx in range(AUTO_SEARCH_NUM_FRAMES):
            frame = grab_frame(cap, n_flush=AUTO_SEARCH_FLUSH)

            detected, current_cam_points_raw, _ = detect_corner_tags(frame, detector)
            preview = draw_detected_corner_tags(frame, detected, current_cam_points_raw)
            cv2.putText(
                preview,
                f"Frame {idx + 1}/{AUTO_SEARCH_NUM_FRAMES}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if best_result is not None:
                cv2.putText(
                    preview,
                    f"Best err: {best_result['pnp_mean_err']:.3f}px",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Manual Screen Pose", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Sortie demandee.")
                return

            result = estimate_best_pose_from_frame(
                frame, detector, K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp
            )

            if result is None:
                continue

            valid_count += 1

            if best_result is None or result["pnp_mean_err"] < best_result["pnp_mean_err"]:
                best_result = result

        if best_result is None:
            raise RuntimeError("Aucune frame valide avec les 4 tags detectes.")

        print(f"\nFrames valides : {valid_count}/{AUTO_SEARCH_NUM_FRAMES}")

        frozen = best_result["frame"]
        frozen_gray = best_result["gray"]
        detected = best_result["detected"]
        cam_points_raw = best_result["cam_points_raw"]
        cam_points_undist = best_result["cam_points_undist"]
        screen_points_mm = best_result["screen_points_mm"]
        graph_points = best_result["graph_points"]

        best_flag = best_result["best_flag"]
        rvec_sc = best_result["rvec_sc"]
        tvec_sc = best_result["tvec_sc"]
        R_sc = best_result["R_sc"]
        R_sp = best_result["R_sp"]
        T_sp = best_result["T_sp"]
        pnp_mean_err = best_result["pnp_mean_err"]
        pnp_per_point_errs = best_result["pnp_per_point_errs"]
        pnp_proj_pts = best_result["pnp_proj_pts"]

        # print("\n=== TEST A : VERIFICATION POSE ECRAN -> CAMERA ===")
        # print("Ferme la fenetre ou appuie sur une touche pour continuer.")

        # camera_debug = show_camera_pose_debug(frozen, cam_points_raw, pnp_proj_pts)
        # cv2.imshow("Camera Pose Debug", camera_debug)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Camera Pose Debug")

        
        H_proj = best_result["H_proj"]
        H_inv_proj = best_result["H_inv_proj"]
        H_graph = best_result["H_graph"]
        H_inv_graph = best_result["H_inv_graph"]
        H_graph_to_proj = best_result["H_graph_to_proj"]
        H_proj_to_graph = best_result["H_proj_to_graph"]
        H_cam_to_screen_mm = best_result["H_cam_to_screen_mm"]
        H_screen_mm_to_cam = best_result["H_screen_mm_to_cam"]
        show_projector_pose_test_B(
            projector,
            H_proj,
            cam_points_raw,
            detector,
            cap,
            K_cam,
            dist_cam
        )
        print("\nValidation visuelle de la projection...")
        print("Touches : y = accepter / n = rejeter")

        accepted = show_projected_validation(projector, H_graph_to_proj)

        if not accepted:
            print("Pose rejetee.")
            return

        show_pose_debug_projection(
            projector,
            H_proj,
            cam_points_raw,
            detector,
            cap,
            K_cam,
            dist_cam
        ) 

        print("\nPoints camera bruts detectes (TL, TR, BR, BL) :")
        print(cam_points_raw)

        print("\n=== RESULTATS ===")
        print("Methode solvePnP choisie :", best_flag)
        print("Erreur reprojection solvePnP (px) :", pnp_mean_err)
        print("Erreurs par point :", pnp_per_point_errs)

        # debug image
        debug = draw_detected_corner_tags(frozen, detected, cam_points_raw)
        for p in pnp_proj_pts.reshape(-1, 2):
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(debug, (x, y), 6, (0, 255, 0), 2)

        cv2.imwrite(str(OUTPUT_DEBUG_PATH), debug)

        calib_data = {
            "H_proj": H_proj.tolist(),
            "H_inv_proj": H_inv_proj.tolist(),
            "H_graph": H_graph.tolist(),
            "H_inv_graph": H_inv_graph.tolist()
        }

        with open(OUTPUT_CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4)

        pose_data = {
            "method": "aruco_screen_reference_points_solvepnp_best_frame",
            "screen_width_mm": SCREEN_WIDTH_MM,
            "screen_height_mm": SCREEN_HEIGHT_MM,
            "corner_tag_ids": CORNER_TAG_IDS,
            "camera_points_raw_TL_TR_BR_BL": cam_points_raw.tolist(),
            "camera_points_undist_TL_TR_BR_BL": cam_points_undist.tolist(),
            "screen_points_mm_TL_TR_BR_BL": screen_points_mm.tolist(),
            "graph_points_TL_TR_BR_BL": graph_points.tolist(),
            "camera_matrix": K_cam.tolist(),
            "camera_dist_coeffs": dist_cam.tolist(),
            "projector_matrix": K_proj.tolist(),
            "projector_dist_coeffs": dist_proj.tolist(),
            "R_camera_to_projector": R_cp.tolist(),
            "T_camera_to_projector": T_cp.tolist(),
            "solvepnp_method_selected": best_flag,
            "solvepnp_mean_reprojection_error_px": pnp_mean_err,
            "solvepnp_per_point_errors_px": pnp_per_point_errs,
            "rvec_screen_to_camera": rvec_sc.tolist(),
            "tvec_screen_to_camera": tvec_sc.tolist(),
            "R_screen_to_camera": R_sc.tolist(),
            "R_screen_to_projector": R_sp.tolist(),
            "T_screen_to_projector": T_sp.tolist(),
            "H_proj": H_proj.tolist(),
            "H_inv_proj": H_inv_proj.tolist(),
            "H_graph": H_graph.tolist(),
            "H_inv_graph": H_inv_graph.tolist(),
            "H_cam_to_screen_mm": H_cam_to_screen_mm.tolist(),
            "H_screen_mm_to_cam": H_screen_mm_to_cam.tolist()
        }

        with open(OUTPUT_POSE_PATH, "w", encoding="utf-8") as f:
            json.dump(pose_data, f, indent=4)

        runtime_mapping_data = {
            "metadata": {
                "grid_size": GRID_SIZE,
                "projector_width": PROJECTOR_WIDTH,
                "projector_height": PROJECTOR_HEIGHT,
                "screen_width_mm": SCREEN_WIDTH_MM,
                "screen_height_mm": SCREEN_HEIGHT_MM,
                "source_pose_method": "aruco_screen_reference_points_solvepnp_best_frame"
            },
            "screen_pose": {
                "rvec_screen_to_camera": rvec_sc.tolist(),
                "tvec_screen_to_camera": tvec_sc.tolist(),
                "R_screen_to_camera": R_sc.tolist(),
                "R_screen_to_projector": R_sp.tolist(),
                "T_screen_to_projector": T_sp.tolist()
            },
            "homographies": {
                "H_cam_undist_to_proj": H_proj.tolist(),
                "H_proj_to_cam_undist": H_inv_proj.tolist(),
                "H_cam_undist_to_graph": H_graph.tolist(),
                "H_graph_to_cam_undist": H_inv_graph.tolist(),
                "H_graph_to_proj": H_graph_to_proj.tolist(),
                "H_proj_to_graph": H_proj_to_graph.tolist()
            },
            "reference_points": {
                "corner_tag_ids": CORNER_TAG_IDS,
                "camera_points_raw_TL_TR_BR_BL": cam_points_raw.tolist(),
                "camera_points_undist_TL_TR_BR_BL": cam_points_undist.tolist(),
                "screen_points_mm_TL_TR_BR_BL": screen_points_mm.tolist(),
                "graph_points_TL_TR_BR_BL": graph_points.tolist()
            }
        }

        with open(RUNTIME_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(runtime_mapping_data, f, indent=4)

        print("-", RUNTIME_MAPPING_PATH)
        print("\nFichiers sauvegardes :")
        print("-", OUTPUT_CALIB_PATH)
        print("-", OUTPUT_POSE_PATH)
        print("-", OUTPUT_DEBUG_PATH)

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()