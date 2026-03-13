
# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path

import cv2
import numpy as np
from screeninfo import get_monitors


# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_SCREEN_ID = 1          # a verifier
PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

OUTPUT_ROOT = Path("projector_calibration_data")
POSE_NAME = "pose_screen_01"

# temporisations
SETTLE_TIME_SEC = 0.35           # temps pour stabiliser affichage/projecteur
CAPTURE_DELAY_SEC = 0.10         # petite marge avant lecture

# aperçu caméra
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720


# =========================================================
# OUTILS AFFICHAGE PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        self.screen_id = screen_id
        self.window_name = "ProjectorGrayCode"

        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide: {screen_id}")

        self.monitor = monitors[screen_id]

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
        raise RuntimeError("Impossible d'ouvrir la camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # on vide un peu le buffer
    for _ in range(5):
        cap.read()

    return cap


def grab_frame(cap, n_flush=2):
    frame = None
    for _ in range(n_flush):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Erreur capture camera.")
    return frame


# =========================================================
# GRAY CODE
# =========================================================
def create_graycode_pattern():
    if not hasattr(cv2, "structured_light_GrayCodePattern"):
        raise RuntimeError(
            "Le module structured_light n'est pas disponible. "
            "Installe opencv-contrib-python."
        )

    pattern = cv2.structured_light_GrayCodePattern.create(
        PROJECTOR_WIDTH,
        PROJECTOR_HEIGHT
    )

    ok, pattern_images = pattern.generate()
    if not ok:
        raise RuntimeError("Impossible de generer les motifs Gray code.")

    return pattern, pattern_images


# =========================================================
# SAUVEGARDE
# =========================================================
def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# =========================================================
# MAIN
# =========================================================
def main():
    pose_dir = OUTPUT_ROOT / POSE_NAME
    patterns_dir = pose_dir / "projected_patterns"
    captures_dir = pose_dir / "camera_captures"
    preview_dir = pose_dir / "preview"

    pose_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("Ouverture camera...")
    cap = open_camera()

    print("Creation des motifs Gray code...")
    graycode, pattern_images = create_graycode_pattern()

    print(f"Nombre de motifs Gray code : {len(pattern_images)}")

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)

    # images tout blanc / tout noir utiles pour le decodage
    white_img = np.full((PROJECTOR_HEIGHT, PROJECTOR_WIDTH), 255, dtype=np.uint8)
    black_img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH), dtype=np.uint8)

    try:
        print("\nPlace le damier dans UNE pose fixe.")
        print("Puis appuie sur ENTER dans le terminal pour lancer l'acquisition.")
        input()

        # -------------------------------------------------
        # capture image blanche
        # -------------------------------------------------
        print("Capture image blanche...")
        projector.show(white_img)
        time.sleep(SETTLE_TIME_SEC)
        time.sleep(CAPTURE_DELAY_SEC)
        frame_white = grab_frame(cap)
        save_image(captures_dir / "white.png", frame_white)
        save_image(patterns_dir / "white.png", white_img)

        # -------------------------------------------------
        # capture image noire
        # -------------------------------------------------
        print("Capture image noire...")
        projector.show(black_img)
        time.sleep(SETTLE_TIME_SEC)
        time.sleep(CAPTURE_DELAY_SEC)
        frame_black = grab_frame(cap)
        save_image(captures_dir / "black.png", frame_black)
        save_image(patterns_dir / "black.png", black_img)

        # -------------------------------------------------
        # capture sequence Gray code
        # -------------------------------------------------
        print("Capture des motifs Gray code...")
        for i, pat in enumerate(pattern_images):
            # pattern peut arriver en 1 canal ; on le force en uint8 1 canal
            pat_u8 = pat.astype(np.uint8)

            projector.show(pat_u8)
            time.sleep(SETTLE_TIME_SEC)
            time.sleep(CAPTURE_DELAY_SEC)

            frame = grab_frame(cap)

            save_image(patterns_dir / f"pattern_{i:03d}.png", pat_u8)
            save_image(captures_dir / f"capture_{i:03d}.png", frame)

            # aperçu léger
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.putText(
                preview,
                f"Capture pattern {i+1}/{len(pattern_images)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Camera Preview", preview)
            cv2.waitKey(1)

            print(f"  motif {i+1}/{len(pattern_images)} capture")

        # image blanche finale
        projector.show(white_img)

        print(f"\nAcquisition terminee pour {POSE_NAME}")
        print(f"Dossier : {pose_dir}")

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()