# -*- coding: utf-8 -*-

from pathlib import Path
import json
import time
import cv2
import numpy as np
from screeninfo import get_monitors

# ============================
# PARAMETRES
# ============================

PROJECTOR_SCREEN_ID = 1

PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

POINT_RADIUS = 60

DIFF_THRESHOLD = 20
MIN_AREA = 200
MAX_AREA = 50000

CAMERA_WARMUP = 10
POINT_DELAY = 0.6

# ============================
# CHEMINS
# ============================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CALIB_FILE = CONFIG_DIR / "calibration_data.json"

# ============================
# PROJECTEUR
# ============================

class Projector:

    def __init__(self):

        monitors = get_monitors()
        monitor = monitors[PROJECTOR_SCREEN_ID]

        self.name = "RuntimeCalibration"

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, monitor.x, monitor.y)

        cv2.setWindowProperty(
            self.name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    def show(self, img):
        cv2.imshow(self.name, img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.name)


def black_image():
    return np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)


def draw_point(pt):

    img = black_image()

    x = int(pt[0])
    y = int(pt[1])

    cv2.circle(img, (x, y), POINT_RADIUS, (255, 255, 255), -1)

    return img


# ============================
# CAMERA
# ============================

def open_camera():

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    for _ in range(CAMERA_WARMUP):
        cap.read()

    return cap


def grab(cap):

    cap.read()
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Camera capture failed")

    return frame


# ============================
# BLOB DETECTION
# ============================

def detect_blob(frame_black, frame_point):

    g1 = cv2.cvtColor(frame_black, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame_point, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(g2, g1)

    _, th = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:

        area = cv2.contourArea(c)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        if area > best_area:
            best = c
            best_area = area

    if best is None:
        return None

    M = cv2.moments(best)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return np.array([cx, cy], dtype=np.float32)


# ============================
# UTILS
# ============================

def perspective_transform(pts, H):

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    out = cv2.perspectiveTransform(pts, H)

    return out.reshape(-1, 2)


# ============================
# MAIN
# ============================

def main():

    print("=== AUTO RECALAGE RUNTIME ===")

    if not CALIB_FILE.exists():
        raise RuntimeError("calibration_data.json manquant")

    with open(CALIB_FILE) as f:
        calib = json.load(f)

    H_proj = np.array(calib["H_proj"])
    H_inv_graph = np.array(calib["H_inv_graph"])

    H_graph_to_proj = H_proj @ H_inv_graph

    # 4 points dans l'espace logique

    graph_pts = np.array([
        [120, 120],
        [580, 120],
        [580, 580],
        [120, 580]
    ], dtype=np.float32)

    # positions projecteur attendues

    proj_pts = perspective_transform(graph_pts, H_graph_to_proj)

    projector = Projector()
    cap = open_camera()

    try:

        projector.show(black_image())
        time.sleep(0.8)

        frame_black = grab(cap)

        cam_pts = []

        for pt in proj_pts:

            img = draw_point(pt)

            projector.show(img)

            time.sleep(POINT_DELAY)

            frame = grab(cap)

            blob = detect_blob(frame_black, frame)

            if blob is None:
                raise RuntimeError("blob non detecte")

            cam_pts.append(blob)

        cam_pts = np.array(cam_pts, dtype=np.float32)

        # recalage

        H_graph_runtime, _ = cv2.findHomography(cam_pts, graph_pts)
        H_proj_runtime, _ = cv2.findHomography(cam_pts, proj_pts)

        print("\n=== RECALAGE TERMINE ===")

        print("H_graph_runtime =")
        print(H_graph_runtime)

        print("\nH_proj_runtime =")
        print(H_proj_runtime)

        runtime = {
            "H_graph_runtime": H_graph_runtime.tolist(),
            "H_proj_runtime": H_proj_runtime.tolist()
        }

        with open(CONFIG_DIR / "runtime_alignment.json", "w") as f:
            json.dump(runtime, f, indent=4)

        print("\nFichier sauvegarde : runtime_alignment.json")

    finally:

        projector.show(black_image())

        cap.release()
        projector.close()


if __name__ == "__main__":
    main()