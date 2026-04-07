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

GRID_SIZE = 700

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
OUTPUT_JSON_PATH = CONFIG_DIR / "projected_board_homography.json"

# image du motif à projeter
BOARD_IMAGE_PATH = PROJECT_ROOT / "aruco_a3.jpg"

# IDs physiques aux coins de la table
CORNER_TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# taille relative de l'image projetée par rapport au projecteur
# augmente à 0.75 si tu veux plus grand
PROJECTED_BOARD_SCALE = 0.62

# fond blanc pour aider la détection
WHITE_BACKGROUND_LEVEL = 255

# ROI centrale pour garder seulement les tags du motif projeté
CENTER_ROI_RATIO = 0.70

# nombre minimal de corners du motif projeté pour calculer H_img_to_cam
MIN_PROJECTED_CORNERS = 16

AUTO_SEARCH_NUM_FRAMES = 200
AUTO_SEARCH_FLUSH = 2


# =========================================================
# JSON / CALIB
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dist(dist):
    arr = np.array(dist, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.T
    return arr


def load_camera_calibration():
    cam_data = load_json(CAMERA_CALIB_PATH)
    K_cam = np.array(cam_data["camera_matrix"], dtype=np.float64)
    dist_cam = flatten_dist(cam_data["dist_coeffs"])
    return K_cam, dist_cam


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
        raise RuntimeError("Impossible de lire une image caméra.")

    return cap


def grab_frame(cap, n_flush=2):
    for _ in range(n_flush):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Erreur capture caméra.")
    return frame


# =========================================================
# PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectorPose2"

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


# =========================================================
# GEOMETRIE
# =========================================================
def undistort_pixel_points(points_px, K, dist):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float32)


def select_physical_corner(corners, position):
    corners = np.array(corners, dtype=np.float32)

    if position == "TL":
        return corners[np.argmin(corners[:, 0] + corners[:, 1])]
    elif position == "TR":
        return corners[np.argmin(-corners[:, 0] + corners[:, 1])]
    elif position == "BR":
        return corners[np.argmax(corners[:, 0] + corners[:, 1])]
    elif position == "BL":
        return corners[np.argmax(-corners[:, 0] + corners[:, 1])]
    else:
        raise ValueError(f"Position inconnue: {position}")


def corner_distance_to_expected(corners, expected_xy):
    center = np.mean(corners, axis=0)
    return float(np.linalg.norm(center - np.array(expected_xy, dtype=np.float32)))


# =========================================================
# PREPROCESS IMAGE PROJETEE
# =========================================================
def preprocess_board_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Rend l'image plus nette pour la projection :
    - passage en gris
    - binarisation franche
    - reconversion BGR
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return out


def build_projection_canvas(board_img_bgr: np.ndarray):
    """
    Fond blanc plein écran + motif centré et proportionné.
    Retourne :
      canvas : image finale envoyée au projecteur
      board_rect_proj_pts : rectangle du motif dans le repère projecteur (TL,TR,BR,BL)
      board_resized : motif redimensionné
      placement_info : dict avec scale, x0, y0, w, h
    """
    canvas = np.full(
        (PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3),
        WHITE_BACKGROUND_LEVEL,
        dtype=np.uint8
    )

    src_h, src_w = board_img_bgr.shape[:2]

    max_w = int(PROJECTOR_WIDTH * PROJECTED_BOARD_SCALE)
    max_h = int(PROJECTOR_HEIGHT * PROJECTED_BOARD_SCALE)

    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    board_resized = cv2.resize(
        board_img_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    x0 = (PROJECTOR_WIDTH - new_w) // 2
    y0 = (PROJECTOR_HEIGHT - new_h) // 2
    x1 = x0 + new_w
    y1 = y0 + new_h

    canvas[y0:y1, x0:x1] = board_resized

    # contour rouge de debug
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 4)

    board_rect_proj_pts = np.array([
        [x0, y0],
        [x1 - 1, y0],
        [x1 - 1, y1 - 1],
        [x0, y1 - 1],
    ], dtype=np.float32)

    placement_info = {
        "scale": float(scale),
        "x0": int(x0),
        "y0": int(y0),
        "width": int(new_w),
        "height": int(new_h),
    }

    return canvas, board_rect_proj_pts, board_resized, placement_info


# =========================================================
# DETECTEUR ARUCO
# =========================================================
def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 8

    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0

    parameters.polygonalApproxAccuracyRate = 0.05

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def preprocess_for_detection(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def detect_all_markers(gray, detector):
    """
    Retourne une liste de détections :
    [{"id": int, "corners": (4,2), "center": (2,)}]
    """
    corners, ids, _ = detector.detectMarkers(gray)

    detections = []
    if ids is None or len(ids) == 0:
        return detections

    ids = ids.flatten().tolist()

    for i, marker_id in enumerate(ids):
        pts = corners[i][0].astype(np.float32)
        center = np.mean(pts, axis=0)
        detections.append({
            "id": int(marker_id),
            "corners": pts,
            "center": center
        })

    return detections


# =========================================================
# DETECTION TAGS PHYSIQUES
# =========================================================
def select_physical_tags(detections, image_shape):
    """
    Pour chaque ID physique, on choisit la détection la plus proche du coin attendu de l'image.
    Cela évite de confondre avec un tag projeté ayant le même ID.
    """
    h, w = image_shape[:2]

    expected_positions = {
        CORNER_TAG_IDS["TL"]: (0, 0),
        CORNER_TAG_IDS["TR"]: (w - 1, 0),
        CORNER_TAG_IDS["BR"]: (w - 1, h - 1),
        CORNER_TAG_IDS["BL"]: (0, h - 1),
    }

    selected = {}

    for marker_id, expected_xy in expected_positions.items():
        candidates = [d for d in detections if d["id"] == marker_id]
        if not candidates:
            return None

        best = min(
            candidates,
            key=lambda d: corner_distance_to_expected(d["corners"], expected_xy)
        )
        selected[marker_id] = best

    return selected


def get_physical_table_points_raw(selected_physical):
    cam_points_raw = np.array([
        select_physical_corner(selected_physical[CORNER_TAG_IDS["TL"]]["corners"], "TL"),
        select_physical_corner(selected_physical[CORNER_TAG_IDS["TR"]]["corners"], "TR"),
        select_physical_corner(selected_physical[CORNER_TAG_IDS["BR"]]["corners"], "BR"),
        select_physical_corner(selected_physical[CORNER_TAG_IDS["BL"]]["corners"], "BL"),
    ], dtype=np.float32)
    return cam_points_raw


# =========================================================
# DETECTION DU MOTIF PROJETE
# =========================================================
def build_source_board_marker_database(board_img_bgr, detector):
    """
    Détecte les markers dans l'image source (fichier aruco_a3.jpg).
    On crée une base : id -> corners dans l'image source.
    On ignore les IDs physiques pour éviter toute ambiguïté.
    """
    gray = cv2.cvtColor(board_img_bgr, cv2.COLOR_BGR2GRAY)
    detections = detect_all_markers(gray, detector)

    db = {}
    for d in detections:
        marker_id = d["id"]
        if marker_id in CORNER_TAG_IDS.values():
            continue
        if marker_id not in db:
            db[marker_id] = d["corners"].copy()

    if len(db) == 0:
        raise RuntimeError(
            "Aucun tag détecté dans l'image source projetée. "
            "Vérifie BOARD_IMAGE_PATH et le contenu de l'image."
        )

    return db


def select_projected_board_detections(detections, source_db, image_shape):
    """
    Sélectionne les tags du motif projeté :
    - présents dans la base source
    - hors IDs physiques
    - de préférence au centre de l'image
    En cas de doublon d'ID, on garde la détection la plus proche du centre.
    """
    h, w = image_shape[:2]
    cx = w * 0.5
    cy = h * 0.5

    roi_w = w * CENTER_ROI_RATIO
    roi_h = h * CENTER_ROI_RATIO
    x_min = cx - roi_w * 0.5
    x_max = cx + roi_w * 0.5
    y_min = cy - roi_h * 0.5
    y_max = cy + roi_h * 0.5

    grouped = {}

    for d in detections:
        marker_id = d["id"]

        if marker_id in CORNER_TAG_IDS.values():
            continue
        if marker_id not in source_db:
            continue

        x, y = d["center"]

        # on préfère les tags centraux, donc ceux du motif projeté
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        dist_to_center = float(np.linalg.norm(d["center"] - np.array([cx, cy], dtype=np.float32)))

        if marker_id not in grouped:
            grouped[marker_id] = (dist_to_center, d)
        else:
            if dist_to_center < grouped[marker_id][0]:
                grouped[marker_id] = (dist_to_center, d)

    selected = {marker_id: item[1] for marker_id, item in grouped.items()}
    return selected


def build_projected_correspondences(source_db, selected_projected, K_cam, dist_cam):
    """
    Construit des correspondances multi-points :
    source image pixel corners -> camera undistorted pixel corners
    en utilisant les tags du motif projeté.
    """
    src_pts = []
    cam_pts = []

    for marker_id, det in selected_projected.items():
        if marker_id not in source_db:
            continue

        src_corners = np.array(source_db[marker_id], dtype=np.float32)    # 4x2
        cam_corners_raw = np.array(det["corners"], dtype=np.float32)      # 4x2
        cam_corners_und = undistort_pixel_points(cam_corners_raw, K_cam, dist_cam)

        for k in range(4):
            src_pts.append(src_corners[k])
            cam_pts.append(cam_corners_und[k])

    if len(src_pts) < MIN_PROJECTED_CORNERS:
        return None, None

    src_pts = np.array(src_pts, dtype=np.float32)
    cam_pts = np.array(cam_pts, dtype=np.float32)
    return src_pts, cam_pts


# =========================================================
# DEBUG VISUEL
# =========================================================
def draw_debug_view(frame, selected_physical, selected_projected, status_text):
    out = frame.copy()

    if selected_projected is not None:
        for marker_id, d in selected_projected.items():
            pts = d["corners"].astype(np.int32)
            center = d["center"]
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
            cv2.putText(
                out,
                f"P{id}",
                tuple(np.round(center).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

    if selected_physical is not None:
        for marker_id, d in selected_physical.items():
            pts = d["corners"].astype(np.int32)
            center = d["center"]
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

    y = 30
    for line in status_text:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 28

    return out


def draw_projector_debug(canvas, board_rect_proj_pts):
    out = canvas.copy()
    labels = ["IMG_TL", "IMG_TR", "IMG_BR", "IMG_BL"]
    for i, p in enumerate(board_rect_proj_pts):
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(out, (x, y), 14, (0, 255, 0), -1)
        cv2.putText(
            out,
            labels[i],
            (x + 16, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 150, 0),
            2
        )
    return out


# =========================================================
# CALCUL HOMOGRAPHIES
# =========================================================
def estimate_result_from_frame(
    frame,
    detector,
    source_db,
    K_cam,
    dist_cam
):
    gray = preprocess_for_detection(frame)
    detections = detect_all_markers(gray, detector)

    selected_physical = select_physical_tags(detections, frame.shape)
    if selected_physical is None:
        return None, gray, detections, None, None

    selected_projected = select_projected_board_detections(detections, source_db, frame.shape)

    cam_table_raw = get_physical_table_points_raw(selected_physical)
    cam_table_und = undistort_pixel_points(cam_table_raw, K_cam, dist_cam)

    table_grid_pts = np.array([
        [0, 0],
        [GRID_SIZE - 1, 0],
        [GRID_SIZE - 1, GRID_SIZE - 1],
        [0, GRID_SIZE - 1],
    ], dtype=np.float32)

    H_cam_to_table, _ = cv2.findHomography(cam_table_und, table_grid_pts)
    if H_cam_to_table is None:
        return None, gray, detections, selected_physical, selected_projected

    H_cam_to_table = H_cam_to_table / H_cam_to_table[2, 2]

    src_pts, cam_proj_pts = build_projected_correspondences(
        source_db,
        selected_projected,
        K_cam,
        dist_cam
    )

    if src_pts is None or cam_proj_pts is None:
        return None, gray, detections, selected_physical, selected_projected

    H_img_to_cam, mask = cv2.findHomography(src_pts, cam_proj_pts, cv2.RANSAC, 4.0)
    if H_img_to_cam is None:
        return None, gray, detections, selected_physical, selected_projected

    H_img_to_cam = H_img_to_cam / H_img_to_cam[2, 2]
    H_img_to_table = H_cam_to_table @ H_img_to_cam
    H_img_to_table = H_img_to_table / H_img_to_table[2, 2]

    result = {
        "frame": frame.copy(),
        "gray": gray.copy(),
        "selected_physical": selected_physical,
        "selected_projected": selected_projected,
        "cam_table_raw": cam_table_raw.copy(),
        "cam_table_und": cam_table_und.copy(),
        "table_grid_pts": table_grid_pts.copy(),
        "src_pts": src_pts.copy(),
        "cam_proj_pts_und": cam_proj_pts.copy(),
        "H_cam_to_table": H_cam_to_table.copy(),
        "H_img_to_cam": H_img_to_cam.copy(),
        "H_img_to_table": H_img_to_table.copy(),
        "ransac_inliers": int(mask.sum()) if mask is not None else 0,
        "num_projected_markers": len(selected_projected),
        "num_projected_corners": int(len(src_pts)),
    }

    return result, gray, detections, selected_physical, selected_projected


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== FOND BLANC + MOTIF CENTRE + DETECTION TAGS PHYSIQUES ET PROJETES ===")

    K_cam, dist_cam = load_camera_calibration()
    detector = create_aruco_detector()

    board_img = cv2.imread(str(BOARD_IMAGE_PATH), cv2.IMREAD_COLOR)
    if board_img is None:
        raise FileNotFoundError(f"Image introuvable : {BOARD_IMAGE_PATH}")

    board_img = preprocess_board_image(board_img)

    # base des tags présents dans l'image source
    source_db = build_source_board_marker_database(board_img, detector)
    print(f"Tags détectés dans l'image source : {len(source_db)}")

    projection_canvas, board_rect_proj_pts, board_resized, placement_info = build_projection_canvas(board_img)
    projector_canvas_debug = draw_projector_debug(projection_canvas, board_rect_proj_pts)

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    cap = open_camera()

    cv2.namedWindow("Camera Debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detection Gray", cv2.WINDOW_NORMAL)

    try:
        projector.show(projector_canvas_debug)
        time.sleep(1.0)

        best_result = None

        for idx in range(AUTO_SEARCH_NUM_FRAMES):
            frame = grab_frame(cap, n_flush=AUTO_SEARCH_FLUSH)

            result, gray, detections, selected_physical, selected_projected = estimate_result_from_frame(
                frame=frame,
                detector=detector,
                source_db=source_db,
                K_cam=K_cam,
                dist_cam=dist_cam
            )
            if result is not None:

                # =====================================
                # PROJECTION CORRIGÉE SUR LA TABLE
                # =====================================

                cam_table_und = result["cam_table_und"]
                H_cam_to_table = result["H_cam_to_table"]

                h_img, w_img = board_resized.shape[:2]

                src_rect = np.array([
                    [0, 0],
                    [w_img-1, 0],
                    [w_img-1, h_img-1],
                    [0, h_img-1]
                ], dtype=np.float32)

                # projection des coins table vers projecteur
                table_proj_pts = cv2.perspectiveTransform(
                    cam_table_und.reshape(-1,1,2),
                    np.linalg.inv(H_cam_to_table)
                ).reshape(-1,2)

                # homographie image -> projecteur
                H_img_to_proj, _ = cv2.findHomography(src_rect, table_proj_pts)

                warp = cv2.warpPerspective(
                    board_resized,
                    H_img_to_proj,
                    (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
                )

                canvas = np.full((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), 255, dtype=np.uint8)

                mask = (warp > 0)
                canvas[mask] = warp[mask]

                projector.show(canvas)

                best_result = result
                break
            status_text = [
                "Fond blanc projete + motif centre",
                f"Frame {idx + 1}/{AUTO_SEARCH_NUM_FRAMES}",
                f"Tags physiques detectes : {0 if selected_physical is None else len(selected_physical)} / 4",
                f"Tags projetes retenus : {0 if selected_projected is None else len(selected_projected)}",
                "s: sauvegarder | q: quitter"
            ]

            preview = draw_debug_view(frame, selected_physical, selected_projected, status_text)

            cv2.imshow("Camera Debug", preview)
            cv2.imshow("Detection Gray", gray)
            # projector.show(projector_canvas_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Sortie demandee.")
                return

            if result is not None:
                best_result = result
                break

        if best_result is None:
            raise RuntimeError(
                "Aucune frame valide. "
                "Soit les 4 tags physiques ne sont pas tous detectes, "
                "soit les tags du motif projete ne sont pas detectes en nombre suffisant."
            )

        final_frame = best_result["frame"]
        final_gray = best_result["gray"]
        selected_physical = best_result["selected_physical"]
        selected_projected = best_result["selected_projected"]

        status_text = [
            "DETECTION OK",
            f"Tags projetes : {best_result['num_projected_markers']}",
            f"Corners projetes utilises : {best_result['num_projected_corners']}",
            f"Inliers RANSAC : {best_result['ransac_inliers']}",
            "Appuie sur s pour sauvegarder, q pour quitter"
        ]
        final_preview = draw_debug_view(final_frame, selected_physical, selected_projected, status_text)

        cv2.imshow("Camera Debug", final_preview)
        cv2.imshow("Detection Gray", final_gray)
        # projector.show(projector_canvas_debug)

        print("\nDETECTION OK")
        print("cam_table_raw:")
        print(best_result["cam_table_raw"])
        print("\ncam_table_und:")
        print(best_result["cam_table_und"])
        print("\nH_cam_to_table:")
        print(best_result["H_cam_to_table"])
        print("\nH_img_to_cam:")
        print(best_result["H_img_to_cam"])
        print("\nH_img_to_table:")
        print(best_result["H_img_to_table"])

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)

                data = {
                    "metadata": {
                        "method": "white_background_centered_projected_board_and_physical_corner_tags",
                        "board_image_path": str(BOARD_IMAGE_PATH),
                        "projector_width": PROJECTOR_WIDTH,
                        "projector_height": PROJECTOR_HEIGHT,
                        "camera_width": CAMERA_WIDTH,
                        "camera_height": CAMERA_HEIGHT,
                        "grid_size": GRID_SIZE,
                        "projected_board_scale": PROJECTED_BOARD_SCALE,
                        "white_background_level": WHITE_BACKGROUND_LEVEL,
                        "center_roi_ratio": CENTER_ROI_RATIO,
                        "corner_tag_ids": CORNER_TAG_IDS,
                        "placement_info": placement_info,
                    },
                    "camera": {
                        "camera_matrix": K_cam.tolist(),
                        "camera_dist_coeffs": dist_cam.tolist(),
                    },
                    "board_source_database": {
                        str(marker_id): source_db[marker_id].tolist()
                        for marker_id in sorted(source_db.keys())
                    },
                    "reference_points": {
                        "cam_table_raw_TL_TR_BR_BL": best_result["cam_table_raw"].tolist(),
                        "cam_table_und_TL_TR_BR_BL": best_result["cam_table_und"].tolist(),
                        "table_grid_pts_TL_TR_BR_BL": best_result["table_grid_pts"].tolist(),
                        "src_pts_used_for_H_img_to_cam": best_result["src_pts"].tolist(),
                        "cam_proj_pts_und_used_for_H_img_to_cam": best_result["cam_proj_pts_und"].tolist(),
                    },
                    "homographies": {
                        "H_cam_to_table": best_result["H_cam_to_table"].tolist(),
                        "H_img_to_cam": best_result["H_img_to_cam"].tolist(),
                        "H_img_to_table": best_result["H_img_to_table"].tolist(),
                    },
                    "quality": {
                        "num_projected_markers": best_result["num_projected_markers"],
                        "num_projected_corners": best_result["num_projected_corners"],
                        "ransac_inliers": best_result["ransac_inliers"],
                    }
                }

                with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

                print(f"\nSauvegarde OK : {OUTPUT_JSON_PATH}")
                break

            elif key == ord("q") or key == 27:
                print("Sortie sans sauvegarde.")
                break

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()