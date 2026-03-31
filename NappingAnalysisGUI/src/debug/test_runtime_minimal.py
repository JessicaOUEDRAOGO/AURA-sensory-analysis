# -*- coding: utf-8 -*-
import cv2
import numpy as np

from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.projection.display_manager import DisplayManager
from src.core.calibration.pose_aruco import create_aruco_detector

CAMERA_ID = 0

def select_physical_corner(corners, position):
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

def get_anchor_raw(corner):
    pts = corner[0].astype(np.float32)
    return select_physical_corner(pts, "TL")

def main():
    mapper = CoordinateMapper()
    mapper.load()

    display = DisplayManager(projector_screen_id=1)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    for _ in range(10):
        cap.read()

    detector = create_aruco_detector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            proj_w = int(mapper.projector_width)
            proj_h = int(mapper.projector_height)
            img = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])

                    # Ignore les tags de calibration si tu veux
                    if marker_id in [40, 41, 42, 43]:
                        continue

                    p_raw = get_anchor_raw(corner)

                    # chemin A : direct
                    p_proj_direct = mapper.camera_raw_to_projector_nominal(p_raw)

                    # chemin B : via graph
                    p_graph = mapper.camera_raw_to_graph(p_raw)
                    H_graph_to_proj = mapper.get_graph_to_projector_homography()
                    p_proj_via_graph = mapper._apply_homography(H_graph_to_proj, p_graph)

                    # draw direct = rouge plein
                    xd, yd = int(round(float(p_proj_direct[0]))), int(round(float(p_proj_direct[1])))
                    cv2.circle(img, (xd, yd), 10, (0, 0, 255), -1)
                    cv2.putText(
                        img,
                        f"D {marker_id}",
                        (xd + 12, yd - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    # draw via graph = vert contour
                    xg, yg = int(round(float(p_proj_via_graph[0]))), int(round(float(p_proj_via_graph[1])))
                    cv2.circle(img, (xg, yg), 12, (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"G {marker_id}",
                        (xg + 12, yg + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    err = np.linalg.norm(p_proj_direct - p_proj_via_graph)
                    print(f"ID {marker_id} | err direct-via-graph = {err:.3f} px")

            display.display_image_on_projector_monitor(img)
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        display.close_display()

if __name__ == "__main__":
    main()