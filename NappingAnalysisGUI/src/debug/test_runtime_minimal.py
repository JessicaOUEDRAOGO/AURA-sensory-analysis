# -*- coding: utf-8 -*-
import cv2
import numpy as np

from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.projection.display_manager import DisplayManager
from src.core.calibration.pose_aruco import create_aruco_detector

CAMERA_ID = 0


def select_physical_corner(corners):
    # même logique que pose_aruco (TL)
    return corners[np.argmin(corners[:, 0] + corners[:, 1])]


def main():
    print("=== TEST MAPPER UNIQUEMENT ===")

    mapper = CoordinateMapper()
    mapper.load()

    display = DisplayManager(projector_screen_id=1)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Camera non ouverte")

    # warmup
    for _ in range(10):
        cap.read()

    detector = create_aruco_detector()

    print("Démarrage test...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        proj_w = mapper.projector_width
        proj_h = mapper.projector_height

        projector_img = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])

                # point anchor
                pts = corner[0].astype(np.float32)
                anchor = select_physical_corner(pts)

                # mapping
                proj_pt = mapper.camera_raw_to_projector_nominal(anchor)

                x = int(proj_pt[0])
                y = int(proj_pt[1])

                # draw
                cv2.circle(projector_img, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(
                    projector_img,
                    f"ID {marker_id}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

        display.display_image_on_projector_monitor(projector_img)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    display.close_display()


if __name__ == "__main__":
    main()