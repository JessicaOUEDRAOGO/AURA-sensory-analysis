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
MARKER_RADIUS = 22
MARKER_THICKNESS = -1

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CALIB_PATH = CONFIG_DIR / "calibration_data.json"
CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"

OUTPUT_REFINED_CALIB = CONFIG_DIR / "calibration_data_refined.json"
OUTPUT_DEBUG_IMG = CONFIG_DIR / "refine_hproj_debug.png"


# =========================================================
# OUTILS JSON
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


# =========================================================
# PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "RefineProjectorHomography"

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
        raise RuntimeError("Impossible de lire une frame caméra.")

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


# =========================================================
# GEOMETRIE
# =========================================================
def undistort_pixel_points(points_px, K, dist):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float32)


def apply_homography(H, points_2d):
    pts = np.array(points_2d, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


# =========================================================
# INTERACTION
# =========================================================
clicked_points = []
frozen_gray = None


def refine_click(gray, x, y, roi_half_size=20):
    h, w = gray.shape[:2]

    x0 = max(0, x - roi_half_size)
    y0 = max(0, y - roi_half_size)
    x1 = min(w, x + roi_half_size + 1)
    y1 = min(h, y + roi_half_size + 1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return float(x), float(y)

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5
    )

    if corners is None:
        return float(x), float(y)

    corners = corners.reshape(-1, 2)
    target = np.array([x - x0, y - y0], dtype=np.float32)
    d2 = np.sum((corners - target) ** 2, axis=1)
    best = corners[np.argmin(d2)]

    return float(best[0] + x0), float(best[1] + y0)


def mouse_callback(event, x, y, flags, param):
    global clicked_points, frozen_gray

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        rx, ry = refine_click(frozen_gray, x, y, roi_half_size=20)
        clicked_points.append([rx, ry])


def draw_points(img, pts):
    labels = ["TL", "TR", "BR", "BL"]
    out = img.copy()

    for i, p in enumerate(pts):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            out,
            labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    if len(pts) == 4:
        cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)

    return out


# =========================================================
# MOTIF DE REFINEMENT
# =========================================================
def build_refinement_pattern(projector_control_points):
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)
    labels = ["TL", "TR", "BR", "BL"]

    for i, p in enumerate(projector_control_points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), MARKER_RADIUS + 6, (0, 255, 0), 2)
        cv2.circle(img, (x, y), MARKER_RADIUS, (255, 255, 255), MARKER_THICKNESS)
        cv2.putText(
            img,
            labels[i],
            (x + 16, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    return img


# =========================================================
# MAIN
# =========================================================
def main():
    global clicked_points, frozen_gray

    print("=== RAFFINEMENT EMPIRIQUE DE H_proj ===")

    calib = load_json(CALIB_PATH)
    cam_calib = load_json(CAMERA_CALIB_PATH)

    required = ["H_proj", "H_inv_graph", "H_graph"]
    for key in required:
        if key not in calib:
            raise KeyError(f"Clé absente dans calibration_data.json : {key}")

    H_proj = np.array(calib["H_proj"], dtype=np.float64)
    H_inv_graph = np.array(calib["H_inv_graph"], dtype=np.float64)
    H_graph = np.array(calib["H_graph"], dtype=np.float64)

    K_cam = np.array(cam_calib["camera_matrix"], dtype=np.float64)
    dist_cam = flatten_dist(cam_calib["dist_coeffs"])

    # graph -> projector courant
    H_graph_to_proj = H_proj @ H_inv_graph
    H_graph_to_proj /= H_graph_to_proj[2, 2]

    # 4 coins logiques de la grille
    graph_corners = np.array([
        [0.0, 0.0],
        [GRID_SIZE - 1, 0.0],
        [GRID_SIZE - 1, GRID_SIZE - 1],
        [0.0, GRID_SIZE - 1],
    ], dtype=np.float32)

    # positions projecteur actuellement prévues pour ces coins
    projector_control_points = apply_homography(H_graph_to_proj, graph_corners)

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    cap = open_camera()

    try:
        pattern = build_refinement_pattern(projector_control_points)
        projector.show(pattern)

        time.sleep(1.0)
        frame = grab_frame(cap)
        frozen = frame.copy()
        frozen_gray = cv2.cvtColor(frozen, cv2.COLOR_BGR2GRAY)

        clicked_points = []

        print("\nClique les 4 marqueurs projetés dans cet ordre :")
        print("1. TL")
        print("2. TR")
        print("3. BR")
        print("4. BL")
        print("\nTouches :")
        print("  r = reset")
        print("  s = sauvegarder")
        print("  q = quitter")

        cv2.namedWindow("Refine H_proj", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Refine H_proj", mouse_callback)

        while True:
            display = draw_points(frozen, clicked_points)
            cv2.imshow("Refine H_proj", display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                clicked_points = []

            elif key == ord("q"):
                return

            elif key == ord("s"):
                if len(clicked_points) != 4:
                    print("Il faut exactement 4 points.")
                    continue

                cam_points_raw = np.array(clicked_points, dtype=np.float32)
                cam_points_undist = undistort_pixel_points(cam_points_raw, K_cam, dist_cam)

                # Nouvelle homographie empirique caméra undist -> projecteur
                H_proj_refined, _ = cv2.findHomography(
                    cam_points_undist,
                    projector_control_points.astype(np.float32),
                    method=0
                )
                H_proj_refined = H_proj_refined / H_proj_refined[2, 2]
                H_inv_proj_refined = np.linalg.inv(H_proj_refined)

                # on sauvegarde une copie raffinée
                calib_refined = dict(calib)
                calib_refined["H_proj"] = H_proj_refined.tolist()
                calib_refined["H_inv_proj"] = H_inv_proj_refined.tolist()

                with open(OUTPUT_REFINED_CALIB, "w", encoding="utf-8") as f:
                    json.dump(calib_refined, f, indent=4)

                debug = draw_points(frozen, clicked_points)
                cv2.imwrite(str(OUTPUT_DEBUG_IMG), debug)

                print("\n=== RESULTATS ===")
                print("H_proj raffiné =")
                print(H_proj_refined)
                print("\nFichiers sauvegardés :")
                print("-", OUTPUT_REFINED_CALIB)
                print("-", OUTPUT_DEBUG_IMG)
                return

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()