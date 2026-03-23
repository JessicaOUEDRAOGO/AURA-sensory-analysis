# -*- coding: utf-8 -*-
import cv2


CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Erreur lecture caméra.")
            break

        # Rotation 180°
        rotated = cv2.rotate(frame, cv2.ROTATE_180)

        cv2.imshow("Camera - rotation 180", rotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()