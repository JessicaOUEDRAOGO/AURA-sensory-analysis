# -*- coding: utf-8 -*-
from pathlib import Path
import json

import cv2
import numpy as np
from screeninfo import get_monitors


# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_SCREEN_ID = 1
PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

GRID_SIZE = 700
GRID_STEP = 50
LINE_THICKNESS = 2
BORDER_THICKNESS = 4
POINT_RADIUS = 8
SHOW_LABELS = True

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CALIB_PATH = CONFIG_DIR / "calibration_data.json"
OUTPUT_TEST_PATH = CONFIG_DIR / "projected_grid_test_correct.png"


# =========================================================
# PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectedGridCorrect"

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
def load_calibration_data():
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CALIB_PATH}")

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["H_proj", "H_inv_graph"]
    for key in required:
        if key not in data:
            raise KeyError(f"Clé absente dans calibration_data.json : {key}")

    H_proj = np.array(data["H_proj"], dtype=np.float64)
    H_inv_graph = np.array(data["H_inv_graph"], dtype=np.float64)

    # IMPORTANT :
    # H_proj : camera undistorted -> projector
    # H_inv_graph : graph -> camera undistorted
    # donc graph -> projector
    H_graph_to_proj = H_proj @ H_inv_graph
    H_graph_to_proj /= H_graph_to_proj[2, 2]

    return H_graph_to_proj


def build_grid_image():
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

    # fond noir
    img[:] = (0, 0, 0)

    # lignes verticales
    for x in range(0, GRID_SIZE, GRID_STEP):
        cv2.line(img, (x, 0), (x, GRID_SIZE - 1), (0, 255, 0), LINE_THICKNESS)

    # lignes horizontales
    for y in range(0, GRID_SIZE, GRID_STEP):
        cv2.line(img, (0, y), (GRID_SIZE - 1, y), (0, 255, 0), LINE_THICKNESS)

    # contour principal
    cv2.rectangle(
        img,
        (0, 0),
        (GRID_SIZE - 1, GRID_SIZE - 1),
        (255, 255, 255),
        BORDER_THICKNESS
    )

    # centre
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    cv2.circle(img, center, 12, (0, 0, 255), -1)

    # coins
    corners = [
        ((0, 0), "TL"),
        ((GRID_SIZE - 1, 0), "TR"),
        ((GRID_SIZE - 1, GRID_SIZE - 1), "BR"),
        ((0, GRID_SIZE - 1), "BL"),
    ]

    for (x, y), label in corners:
        cv2.circle(img, (x, y), POINT_RADIUS, (255, 0, 255), -1)

        if SHOW_LABELS:
            tx = x + 12 if x < GRID_SIZE // 2 else x - 70
            ty = y + 28 if y < GRID_SIZE // 2 else y - 12
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2
            )

    if SHOW_LABELS:
        cv2.putText(img, "CENTER", (center[0] - 45, center[1] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


def warp_grid_to_projector(grid_img, H_graph_to_proj):
    warped = cv2.warpPerspective(
        grid_img,
        H_graph_to_proj,
        (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    )
    return warped


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== TEST CORRECT DE PROJECTION DE GRILLE ===")

    H_graph_to_proj = load_calibration_data()
    print("Homographie graph -> projector =")
    print(H_graph_to_proj)

    grid_img = build_grid_image()
    warped = warp_grid_to_projector(grid_img, H_graph_to_proj)

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)

    try:
        print("\nTouches :")
        print("  q ou Echap = quitter")
        print("  s = sauvegarder l'image projetée")

        while True:
            projector.show(warped)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("s"):
                cv2.imwrite(str(OUTPUT_TEST_PATH), warped)
                print(f"Image sauvegardée : {OUTPUT_TEST_PATH}")

    finally:
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()