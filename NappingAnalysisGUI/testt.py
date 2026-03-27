# -*- coding: utf-8 -*-
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from screeninfo import get_monitors

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.utils.paths import config_path


# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_SCREEN_ID = 1
CAMERA_ID = 0

WINDOW_NAME_PROJECTOR = "ProjectorCorrectionDebug"
WINDOW_NAME_CAMERA = "ProjectorCorrectionDebugCamera"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
EXCLUDED_TAG_IDS = {40, 41, 42, 43}


# =========================================================
# AFFICHAGE PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = WINDOW_NAME_PROJECTOR

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    def show(self, image: np.ndarray):
        cv2.imshow(self.window_name, image)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


# =========================================================
# OUTILS
# =========================================================
def apply_homography(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("Coordonnée homogène nulle")
    q /= q[2]
    return q[:2].astype(np.float32)


def load_projector_correction():
    path = Path(config_path("projector_correction.json"))
    if not path.exists():
        print("[DEBUG] projector_correction.json absent -> identité")
        return np.eye(3, dtype=np.float64)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = np.array(data["H_proj_correction"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H_proj_correction invalide : {H.shape}")

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    print("[DEBUG] H_proj_correction chargée :")
    print(H)
    return H


def build_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def detect_markers(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def filter_useful_markers(corners, ids):
    if ids is None or len(ids) == 0:
        return [], None

    filtered_corners = []
    filtered_ids = []

    for i, marker_id_arr in enumerate(ids):
        marker_id = int(marker_id_arr[0])
        if marker_id in EXCLUDED_TAG_IDS:
            continue

        filtered_corners.append(corners[i])
        filtered_ids.append([marker_id])

    if len(filtered_ids) == 0:
        return [], None

    return filtered_corners, np.array(filtered_ids, dtype=np.int32)


def draw_camera_preview(frame, corners, ids):
    preview = frame.copy()

    if ids is not None and len(ids) > 0:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            marker_id = int(ids[i][0])

            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])
                cv2.line(preview, pt1, pt2, (0, 255, 0), 2)

            center = tuple(np.mean(pts, axis=0).astype(int))
            cv2.circle(preview, center, 6, (0, 0, 255), -1)

            cv2.putText(
                preview,
                f"ID {marker_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

    return preview


def build_nominal_background(mapper: CoordinateMapper):
    proj_w = int(mapper.projector_width)
    proj_h = int(mapper.projector_height)

    graph_bg = np.full((mapper.grid_size, mapper.grid_size, 3), 255, dtype=np.uint8)
    H_graph_to_proj = mapper.get_graph_to_projector_homography()

    projector_bg = cv2.warpPerspective(
        graph_bg,
        H_graph_to_proj,
        (proj_w, proj_h)
    )
    return projector_bg


def draw_debug_overlay(img, marker_id, center_nominal, center_corrected):
    x_nom, y_nom = int(round(center_nominal[0])), int(round(center_nominal[1]))
    x_cor, y_cor = int(round(center_corrected[0])), int(round(center_corrected[1]))

    # nominal = vert
    cv2.circle(img, (x_nom, y_nom), 16, (0, 255, 0), 3)

    # corrigé = rouge
    cv2.circle(img, (x_cor, y_cor), 12, (0, 0, 255), 3)

    # relier les deux
    cv2.line(img, (x_nom, y_nom), (x_cor, y_cor), (255, 0, 255), 2)

    # label
    cv2.putText(
        img,
        f"ID {marker_id}",
        (x_nom + 15, y_nom - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # distance
    dist = float(np.linalg.norm(np.array(center_corrected) - np.array(center_nominal)))
    cv2.putText(
        img,
        f"{dist:.1f}px",
        (x_cor + 15, y_cor + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== DEBUG CORRECTION HORS RUNTIME ===")
    print("Vert = centre nominal")
    print("Rouge = centre corrigé")
    print("Ligne violette = déplacement induit par la correction")
    print("Q ou Echap pour quitter")

    mapper = CoordinateMapper()
    try:
        mapper.load(load_projector_correction=False)
    except TypeError:
        # si load() n'accepte pas encore le paramètre
        mapper.load()
        if hasattr(mapper, "reset_projector_correction_homography"):
            mapper.reset_projector_correction_homography()

    H_corr = load_projector_correction()

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    detector = build_aruco_detector()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("[WARNING] Frame caméra invalide.")
                continue

            bg = build_nominal_background(mapper)

            corners_all, ids_all = detect_markers(detector, frame)
            corners, ids = filter_useful_markers(corners_all, ids_all)

            preview = draw_camera_preview(frame, corners, ids)

            if ids is not None and len(ids) > 0:
                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])

                    camera_center = np.mean(corner[0], axis=0).astype(np.float32)

                    center_nominal = mapper.camera_raw_to_projector_nominal(camera_center)
                    center_corrected = apply_homography(H_corr, center_nominal)

                    draw_debug_overlay(bg, marker_id, center_nominal, center_corrected)

                    print(
                        f"ID {marker_id} | nominal=({center_nominal[0]:.1f},{center_nominal[1]:.1f}) "
                        f"| corrected=({center_corrected[0]:.1f},{center_corrected[1]:.1f})"
                    )

            projector.show(bg)
            cv2.imshow(WINDOW_NAME_CAMERA, cv2.resize(preview, (1080, 720)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()