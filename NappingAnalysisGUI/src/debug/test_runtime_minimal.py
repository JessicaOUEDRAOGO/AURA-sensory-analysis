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

CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "config" / "projector_useful_area.json"

STEP_NORMAL = 5
POINT_RADIUS = 18


# =========================================================
# FENETRE PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectorUsefulAreaCalibration"

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
        raise RuntimeError("Impossible de lire une première frame caméra.")

    return cap


def grab_frame(cap, n_flush=1):
    for _ in range(n_flush):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Erreur capture caméra.")

    return frame


def build_camera_preview(frame, active_label):
    out = frame.copy()

    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # croix centrale
    cv2.line(out, (cx - 40, cy), (cx + 40, cy), (0, 255, 255), 2)
    cv2.line(out, (cx, cy - 40), (cx, cy + 40), (0, 255, 255), 2)
    cv2.circle(out, (cx, cy), 8, (0, 0, 255), -1)

    help_lines = [
        "Vue camera",
        f"Coin actif: {active_label}",
        "Regle les points sur le projecteur en regardant cette vue",
        "1=TL  2=TR  3=BR  4=BL",
        "Fleches = deplacer | S = sauvegarder | Q = quitter",
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


# =========================================================
# AFFICHAGE PROJECTEUR
# =========================================================
def build_image(points, active_label):
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    ordered_labels = ["TL", "TR", "BR", "BL"]

    poly = np.array([points[k] for k in ordered_labels], dtype=np.int32)
    cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

    center = np.mean(np.array([points[k] for k in ordered_labels], dtype=np.float32), axis=0)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(img, (cx, cy), 14, (0, 0, 255), -1)
    cv2.putText(
        img,
        "CENTER",
        (cx + 20, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    for label in ordered_labels:
        x, y = points[label]

        if label == active_label:
            color = (0, 255, 0)
            thickness = -1
        else:
            color = (255, 0, 255)
            thickness = 2

        cv2.circle(img, (x, y), POINT_RADIUS, color, thickness)

        tx = x + 20
        ty = y - 20

        if label == "TR":
            tx = x - 90
            ty = y + 35
        elif label == "BR":
            tx = x - 90
            ty = y - 20
        elif label == "BL":
            tx = x + 20
            ty = y - 20
        elif label == "TL":
            tx = x + 20
            ty = y + 35

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

    help_lines = [
        "Calibration zone utile projecteur",
        "1=TL  2=TR  3=BR  4=BL",
        "Fleches = deplacer le point actif",
        "S = sauvegarder | Q ou ESC = quitter",
    ]

    y0 = 50
    for line in help_lines:
        cv2.putText(
            img,
            line,
            (40, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
        y0 += 38

    return img


# =========================================================
# SAUVEGARDE
# =========================================================
def save_points(points):
    data = {
        "projector_width": PROJECTOR_WIDTH,
        "projector_height": PROJECTOR_HEIGHT,
        "useful_area_points_TL_TR_BR_BL": [
            [int(points["TL"][0]), int(points["TL"][1])],
            [int(points["TR"][0]), int(points["TR"][1])],
            [int(points["BR"][0]), int(points["BR"][1])],
            [int(points["BL"][0]), int(points["BL"][1])],
        ]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Sauvegarde OK : {OUTPUT_PATH}")
    print(json.dumps(data, indent=4))


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== CALIBRATION ZONE UTILE PROJECTEUR ===")
    print("1=TL  2=TR  3=BR  4=BL")
    print("Fleches = deplacement")
    print("S = sauvegarder")
    print("Q ou ESC = quitter")

    points = {
        "TL": [300, 250],
        "TR": [PROJECTOR_WIDTH - 300, 250],
        "BR": [PROJECTOR_WIDTH - 300, PROJECTOR_HEIGHT - 250],
        "BL": [300, PROJECTOR_HEIGHT - 250],
    }

    active_label = "TL"

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    cap = open_camera()

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            proj_img = build_image(points, active_label)
            projector.show(proj_img)

            frame = grab_frame(cap, n_flush=1)
            cam_preview = build_camera_preview(frame, active_label)
            cam_preview = cv2.resize(cam_preview, (1200, 700), interpolation=cv2.INTER_AREA)
            cv2.imshow("Camera Preview", cam_preview)

            key = cv2.waitKeyEx(30)

            if key in (27, ord("q"), ord("Q")):
                print("Sortie sans sauvegarde.")
                break

            elif key == ord("1"):
                active_label = "TL"
            elif key == ord("2"):
                active_label = "TR"
            elif key == ord("3"):
                active_label = "BR"
            elif key == ord("4"):
                active_label = "BL"

            elif key in (ord("s"), ord("S")):
                save_points(points)

            elif key == 2424832:      # LEFT
                points[active_label][0] -= STEP_NORMAL
            elif key == 2555904:      # RIGHT
                points[active_label][0] += STEP_NORMAL
            elif key == 2490368:      # UP
                points[active_label][1] -= STEP_NORMAL
            elif key == 2621440:      # DOWN
                points[active_label][1] += STEP_NORMAL

            points[active_label][0] = int(np.clip(points[active_label][0], 0, PROJECTOR_WIDTH - 1))
            points[active_label][1] = int(np.clip(points[active_label][1], 0, PROJECTOR_HEIGHT - 1))

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()