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

GRID_SIZE = 700  # repere logique table

# IDs des 4 tags physiques poses sur la table
PHYSICAL_TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# IDs reserves a la grille projetee
# IMPORTANT : ne pas utiliser 40,41,42,43 ici
PROJECTED_TAG_IDS = list(range(0, 16))   # 16 tags => grille 4x4

# Parametres de la grille projetee
GRID_ROWS = 4
GRID_COLS = 4
TAG_SIZE_PX = 220
TAG_GAP_PX = 80
MARGIN_PX = 180
BACKGROUND_LEVEL = 255

# Attente avant capture
SETTLE_TIME_SEC = 1.2

# Nombre de frames valides a moyenner pour stabiliser
NB_VALID_FRAMES = 20

# Sauvegarde
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
OUTPUT_JSON_PATH = CONFIG_DIR / "projected_grid_table_calibration.json"
OUTPUT_GRID_IMAGE_PATH = CONFIG_DIR / "projected_aruco_grid.png"

# =========================================================
# JSON
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# =========================================================
# CAMERA CALIB
# =========================================================
def load_camera_calibration():
    cam = load_json(CAMERA_CALIB_PATH)
    K_cam = np.array(cam["camera_matrix"], dtype=np.float64)
    dist_cam = np.array(cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K_cam, dist_cam

# =========================================================
# CAMERA
# =========================================================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera ID={CAMERA_ID}")

    return cap

# =========================================================
# PROJECTOR WINDOW
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id=1):
        monitors = get_monitors()
        if screen_id >= len(monitors):
            raise RuntimeError(
                f"Moniteur projecteur index {screen_id} introuvable. "
                f"Moniteurs detectes: {len(monitors)}"
            )

        m = monitors[screen_id]
        self.name = "projector"

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, m.x, m.y)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, img):
        cv2.imshow(self.name, img)
        cv2.waitKey(1)

# =========================================================
# ARUCO
# =========================================================
def build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    return dictionary, cv2.aruco.ArucoDetector(dictionary, params)

def detect_markers(gray, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []

    ids = ids.flatten()
    detections = []
    for i, marker_id in enumerate(ids):
        pts = corners[i][0].astype(np.float32)   # 4x2
        center = np.mean(pts, axis=0)
        detections.append({
            "id": int(marker_id),
            "corners": pts,
            "center": center,
        })
    return detections

# =========================================================
# GEOMETRIE
# =========================================================
def undistort_points(points, K, dist):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)

def select_outer_corner(corners, pos):
    """
    Recupere le coin externe du tag physique.
    corners est un tableau 4x2.
    """
    if pos == "TL":
        return corners[np.argmin(corners[:, 0] + corners[:, 1])]
    if pos == "TR":
        return corners[np.argmin(-corners[:, 0] + corners[:, 1])]
    if pos == "BR":
        return corners[np.argmax(corners[:, 0] + corners[:, 1])]
    if pos == "BL":
        return corners[np.argmax(-corners[:, 0] + corners[:, 1])]
    raise ValueError(f"Position invalide: {pos}")

def apply_homography(H, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)

def mean_homographies(H_list):
    """
    Moyenne simple puis renormalisation.
    Suffisant ici pour stabiliser un peu.
    """
    H = np.mean(np.stack(H_list, axis=0), axis=0)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H

# =========================================================
# GRILLE PROJETEE
# =========================================================
def build_projected_aruco_grid(dictionary):
    """
    Cree une grande image projecteur contenant une grille de tags ArUco.
    Retourne :
    - canvas (image BGR a projeter)
    - src_points_by_id : dict[id] = coins 4x2 dans l'image projetee
    """
    canvas = np.full(
        (PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3),
        BACKGROUND_LEVEL,
        dtype=np.uint8
    )

    total_w = GRID_COLS * TAG_SIZE_PX + (GRID_COLS - 1) * TAG_GAP_PX
    total_h = GRID_ROWS * TAG_SIZE_PX + (GRID_ROWS - 1) * TAG_GAP_PX

    # On centre la grille dans le projecteur
    x_start = (PROJECTOR_WIDTH - total_w) // 2
    y_start = (PROJECTOR_HEIGHT - total_h) // 2

    src_points_by_id = {}

    tag_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            marker_id = PROJECTED_TAG_IDS[tag_idx]
            tag_idx += 1

            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, TAG_SIZE_PX)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

            x0 = x_start + c * (TAG_SIZE_PX + TAG_GAP_PX)
            y0 = y_start + r * (TAG_SIZE_PX + TAG_GAP_PX)
            x1 = x0 + TAG_SIZE_PX
            y1 = y0 + TAG_SIZE_PX

            canvas[y0:y1, x0:x1] = marker_bgr

            # Coins connus DANS L'IMAGE PROJETEE
            src_points_by_id[marker_id] = np.array([
                [x0, y0],   # TL
                [x1, y0],   # TR
                [x1, y1],   # BR
                [x0, y1],   # BL
            ], dtype=np.float32)

    cv2.imwrite(str(OUTPUT_GRID_IMAGE_PATH), canvas)
    return canvas, src_points_by_id

# =========================================================
# CORRESPONDANCES
# =========================================================
def extract_physical_table_corners(detections):
    """
    Retourne les 4 coins du quadrilatere utile detectes dans la camera :
    ordre TL, TR, BR, BL
    """
    by_id = {d["id"]: d for d in detections}

    required = [PHYSICAL_TAG_IDS["TL"], PHYSICAL_TAG_IDS["TR"],
                PHYSICAL_TAG_IDS["BR"], PHYSICAL_TAG_IDS["BL"]]

    for rid in required:
        if rid not in by_id:
            return None

    cam_pts = np.array([
        select_outer_corner(by_id[PHYSICAL_TAG_IDS["TL"]]["corners"], "TL"),
        select_outer_corner(by_id[PHYSICAL_TAG_IDS["TR"]]["corners"], "TR"),
        select_outer_corner(by_id[PHYSICAL_TAG_IDS["BR"]]["corners"], "BR"),
        select_outer_corner(by_id[PHYSICAL_TAG_IDS["BL"]]["corners"], "BL"),
    ], dtype=np.float32)

    return cam_pts

def extract_projected_grid_correspondences(detections, src_points_by_id, K_cam, dist_cam):
    """
    Recupere les correspondances :
    image projetee connue -> image camera
    en utilisant tous les tags projetes detectes.
    """
    img_pts = []
    cam_pts = []

    for det in detections:
        marker_id = det["id"]

        if marker_id not in src_points_by_id:
            continue

        # Coins connus dans l'image projetee
        src_corners = src_points_by_id[marker_id]  # 4x2

        # Coins detectes dans l'image camera
        dst_corners_raw = det["corners"]           # 4x2
        dst_corners_und = undistort_points(dst_corners_raw, K_cam, dist_cam)

        img_pts.extend(src_corners.tolist())
        cam_pts.extend(dst_corners_und.tolist())

    if len(img_pts) < 8:
        return None, None

    img_pts = np.array(img_pts, dtype=np.float32)
    cam_pts = np.array(cam_pts, dtype=np.float32)
    return img_pts, cam_pts

# =========================================================
# VISU DEBUG
# =========================================================
def draw_debug(frame, detections, quad=None, info_text=None):
    vis = frame.copy()

    for det in detections:
        corners = det["corners"].astype(int)
        center = det["center"].astype(int)
        marker_id = det["id"]

        color = (0, 255, 0) if marker_id in PROJECTED_TAG_IDS else (0, 0, 255)
        cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, color, 2)
        cv2.circle(vis, tuple(center), 4, (255, 0, 0), -1)
        cv2.putText(
            vis, str(marker_id), tuple(center + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    if quad is not None:
        q = quad.astype(int)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (255, 255, 0), 3)
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(q):
            cv2.circle(vis, tuple(p), 8, (255, 255, 0), -1)
            cv2.putText(
                vis, labels[i], tuple(p + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

    if info_text:
        y = 35
        for txt in info_text:
            cv2.putText(
                vis, txt, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            y += 30

    return vis

# =========================================================
# MAIN
# =========================================================
def main():
    print("Chargement calibration camera...")
    K_cam, dist_cam = load_camera_calibration()

    print("Ouverture camera...")
    cap = open_camera()

    print("Creation fenetre projecteur...")
    proj = ProjectorWindow(PROJECTOR_SCREEN_ID)

    print("Creation detecteur ArUco...")
    dictionary, detector = build_aruco_detector()

    print("Generation grille projetee...")
    projected_grid_img, src_points_by_id = build_projected_aruco_grid(dictionary)

    # Affiche d'abord un fond blanc pour aider un peu la scene
    white = np.full((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), 255, np.uint8)
    proj.show(white)
    time.sleep(0.5)

    print("Projection de la grille...")
    proj.show(projected_grid_img)
    time.sleep(SETTLE_TIME_SEC)

    table_pts = np.array([
        [0, 0],
        [GRID_SIZE, 0],
        [GRID_SIZE, GRID_SIZE],
        [0, GRID_SIZE]
    ], dtype=np.float32)

    valid_H_cam_to_table = []
    valid_H_img_to_cam = []
    valid_H_img_to_table = []

    valid_count = 0
    frame_count = 0

    print("Acquisition en cours... Appuie sur ESC pour quitter.")
    print("Conditions de validite :")
    print("- les 4 tags physiques doivent etre detectes")
    print("- plusieurs tags projetes doivent etre detectes")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_markers(gray, detector)

        # -------- 1) Tags physiques -> quadrilatere utile camera
        cam_table_quad_raw = extract_physical_table_corners(detections)

        info_lines = [
            f"Frames valides: {valid_count}/{NB_VALID_FRAMES}",
            f"Detections totales: {len(detections)}"
        ]

        H_cam_to_table = None
        H_img_to_cam = None
        H_img_to_table = None

        if cam_table_quad_raw is not None:
            cam_table_quad_und = undistort_points(cam_table_quad_raw, K_cam, dist_cam)

            H_cam_to_table, mask_ct = cv2.findHomography(
                cam_table_quad_und,
                table_pts,
                method=0
            )

            info_lines.append("4 tags physiques: OK")
        else:
            info_lines.append("4 tags physiques: NON")

        # -------- 2) Grille projetee -> camera
        img_pts, cam_pts = extract_projected_grid_correspondences(
            detections, src_points_by_id, K_cam, dist_cam
        )

        if img_pts is not None and cam_pts is not None:
            H_img_to_cam, mask_ic = cv2.findHomography(
                img_pts,
                cam_pts,
                cv2.RANSAC,
                3.0
            )
            info_lines.append(f"Points grille utilises: {len(img_pts)}")
        else:
            info_lines.append("Points grille utilises: insuffisants")

        # -------- 3) Composition
        if H_cam_to_table is not None and H_img_to_cam is not None:
            H_img_to_table = H_cam_to_table @ H_img_to_cam
            if abs(H_img_to_table[2, 2]) > 1e-12:
                H_img_to_table = H_img_to_table / H_img_to_table[2, 2]

            valid_H_cam_to_table.append(H_cam_to_table)
            valid_H_img_to_cam.append(H_img_to_cam)
            valid_H_img_to_table.append(H_img_to_table)
            valid_count += 1

            info_lines.append("Homographies: OK")
        else:
            info_lines.append("Homographies: NON")

        # Visu debug
        debug_quad = cam_table_quad_raw if cam_table_quad_raw is not None else None
        vis = draw_debug(frame, detections, quad=debug_quad, info_text=info_lines)
        cv2.imshow("camera_debug", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Interruption utilisateur.")
            break

        if valid_count >= NB_VALID_FRAMES:
            print(f"{NB_VALID_FRAMES} frames valides obtenues.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if valid_count == 0:
        print("[ERREUR] Aucune frame valide. Rien a sauvegarder.")
        return

    # Moyenne des homographies
    H_cam_to_table_mean = mean_homographies(valid_H_cam_to_table)
    H_img_to_cam_mean = mean_homographies(valid_H_img_to_cam)
    H_img_to_table_mean = mean_homographies(valid_H_img_to_table)
    H_table_to_img_mean = np.linalg.inv(H_img_to_table_mean)
    H_cam_to_table_inv_mean = np.linalg.inv(H_cam_to_table_mean)
    H_cam_to_img_mean = np.linalg.inv(H_img_to_cam_mean)

    # Test de coherence simple :
    # les coins de l'image utile projetee (grille) transformes dans la table
    img_corners = np.array([
        [0, 0],
        [PROJECTOR_WIDTH - 1, 0],
        [PROJECTOR_WIDTH - 1, PROJECTOR_HEIGHT - 1],
        [0, PROJECTOR_HEIGHT - 1]
    ], dtype=np.float32)

    projected_img_on_table = apply_homography(H_img_to_table_mean, img_corners)

    result = {
        "projector_width": PROJECTOR_WIDTH,
        "projector_height": PROJECTOR_HEIGHT,
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "grid_size": GRID_SIZE,
        "physical_tag_ids": PHYSICAL_TAG_IDS,
        "projected_tag_ids": PROJECTED_TAG_IDS,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "tag_size_px": TAG_SIZE_PX,
        "tag_gap_px": TAG_GAP_PX,
        "nb_valid_frames": valid_count,

        "H_cam_to_table": H_cam_to_table_mean.tolist(),
        "H_table_to_cam": H_cam_to_table_inv_mean.tolist(),

        "H_img_to_cam": H_img_to_cam_mean.tolist(),
        "H_cam_to_img": H_cam_to_img_mean.tolist(),

        "H_img_to_table": H_img_to_table_mean.tolist(),
        "H_table_to_img": H_table_to_img_mean.tolist(),

        "projected_image_corners_mapped_to_table": projected_img_on_table.tolist(),

        "notes": [
            "H_cam_to_table : camera undistorted -> table logical space",
            "H_img_to_cam   : projected image pixels -> camera undistorted",
            "H_img_to_table : projected image pixels -> table logical space",
            "H_table_to_img : table logical space -> projected image pixels",
            "Pour projeter un point de table dans l'image projecteur corrigee, utiliser H_table_to_img."
        ]
    }

    save_json(OUTPUT_JSON_PATH, result)

    print("\nCalibration terminee.")
    print(f"JSON sauvegarde : {OUTPUT_JSON_PATH}")
    print(f"Image grille sauvegardee : {OUTPUT_GRID_IMAGE_PATH}")

    print("\nMatrice H_img_to_table :")
    print(H_img_to_table_mean)

    print("\nMatrice H_table_to_img :")
    print(H_table_to_img_mean)

if __name__ == "__main__":
    main()