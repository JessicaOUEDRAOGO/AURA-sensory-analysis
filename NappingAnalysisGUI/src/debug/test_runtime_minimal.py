# -*- coding: utf-8 -*-
import cv2
import numpy as np
from screeninfo import get_monitors


PROJECTOR_SCREEN_ID = 1
PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160


class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectorRawGridTest"

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


def build_test_pattern():
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # contour image complet
    cv2.rectangle(
        img,
        (0, 0),
        (PROJECTOR_WIDTH - 1, PROJECTOR_HEIGHT - 1),
        (255, 255, 255),
        6
    )

    # grille
    n_cols = 12
    n_rows = 8

    xs = np.linspace(0, PROJECTOR_WIDTH - 1, n_cols).astype(int)
    ys = np.linspace(0, PROJECTOR_HEIGHT - 1, n_rows).astype(int)

    for x in xs:
        cv2.line(img, (x, 0), (x, PROJECTOR_HEIGHT - 1), (0, 255, 0), 2)

    for y in ys:
        cv2.line(img, (0, y), (PROJECTOR_WIDTH - 1, y), (0, 255, 0), 2)

    # centre
    cx = PROJECTOR_WIDTH // 2
    cy = PROJECTOR_HEIGHT // 2
    cv2.circle(img, (cx, cy), 16, (0, 0, 255), -1)
    cv2.putText(
        img,
        "CENTER",
        (cx + 20, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    # coins image
    corners = [
        ((0, 0), "TL"),
        ((PROJECTOR_WIDTH - 1, 0), "TR"),
        ((PROJECTOR_WIDTH - 1, PROJECTOR_HEIGHT - 1), "BR"),
        ((0, PROJECTOR_HEIGHT - 1), "BL"),
    ]

    for (x, y), label in corners:
        cv2.circle(img, (x, y), 18, (255, 0, 255), -1)

        if label == "TL":
            tx, ty = x + 25, y + 40
        elif label == "TR":
            tx, ty = x - 90, y + 40
        elif label == "BR":
            tx, ty = x - 90, y - 20
        else:
            tx, ty = x + 25, y - 20

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

    return img


def main():
    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    img = build_test_pattern()

    print("=== TEST MIRE PROJECTEUR BRUTE ===")
    print("ESC ou q pour quitter")

    try:
        while True:
            projector.show(img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()