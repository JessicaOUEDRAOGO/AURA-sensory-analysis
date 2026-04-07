# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np

from src.core.mapping.coordinate_mapper import CoordinateMapper
from src.core.projection.display_manager import DisplayManager
from src.core.utils.paths import config_path


CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

PROJECTOR_SCREEN_ID = 1

CALIBRATION_TAG_IDS = {40, 41, 42, 43}


def load_projector_useful_homography(grid_size: int):
    path = config_path("projector_useful_area.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    useful_pts = np.array(
        data["useful_area_points_TL_TR_BR_BL"],
        dtype=np.float32
    )

    graph_pts = np.array([
        [0, 0],
        [grid_size - 1, 0],
        [grid_size - 1, grid_size - 1],
        [0, grid_size - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(graph_pts, useful_pts)
    H = H / H[2, 2]
    return H, useful_pts


def apply_homography_to_point(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    out = H @ p
    if abs(out[2]) < 1e-12:
        return None
    out /= out[2]
    return out[:2].astype(np.float32)


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


def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)


def select_physical_corner_runtime(pts: np.ndarray, position: str) -> np.ndarray:
    if position == "TL":
        return pts[np.argmin(pts[:, 0] + pts[:, 1])].astype(np.float32)
    elif position == "TR":
        return pts[np.argmin(-pts[:, 0] + pts[:, 1])].astype(np.float32)
    elif position == "BR":
        return pts[np.argmax(pts[:, 0] + pts[:, 1])].astype(np.float32)
    elif position == "BL":
        return pts[np.argmax(-pts[:, 0] + pts[:, 1])].astype(np.float32)
    else:
        raise ValueError(f"Position inconnue: {position}")


def get_marker_anchor_raw(corner):
    pts = corner[0].astype(np.float32)
    return select_physical_corner_runtime(pts, "TL")


def draw_useful_area_outline(img, useful_pts):
    poly = np.round(useful_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, (0, 255, 255), 2)

    labels = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(useful_pts):
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(img, (x, y), 10, (255, 0, 255), 2)
        cv2.putText(
            img,
            labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


def show_camera_window(frame, corners=None, ids=None):
    preview = frame.copy()

    if corners is not None and ids is not None:
        for i, marker_corners in enumerate(corners):
            pts = marker_corners[0].astype(int)

            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])
                cv2.line(preview, pt1, pt2, (0, 255, 0), 2)

            center = tuple(np.mean(pts, axis=0).astype(int))
            marker_id = int(ids[i][0])
            cv2.circle(preview, center, 6, (0, 0, 255), -1)
            cv2.putText(
                preview,
                f"ID {marker_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            anchor = get_marker_anchor_raw(marker_corners)
            ax, ay = int(round(float(anchor[0]))), int(round(float(anchor[1])))
            cv2.circle(preview, (ax, ay), 8, (255, 255, 0), 2)
            cv2.putText(
                preview,
                "A",
                (ax + 8, ay - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    cv2.putText(
        preview,
        "ESC = quitter",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    resized = cv2.resize(preview, (1200, 700), interpolation=cv2.INTER_AREA)
    cv2.imshow("Camera", resized)


def main():
    print("=== TEST RUNTIME MINIMAL SURFACE UTILE ===")

    mapper = CoordinateMapper()
    mapper.load()
    print("[OK] CoordinateMapper chargé")

    H_graph_to_useful, useful_pts = load_projector_useful_homography(mapper.grid_size)
    print("[OK] projector_useful_area.json chargé")

    display = DisplayManager(projector_screen_id=PROJECTOR_SCREEN_ID)
    cap = open_camera()
    detector = create_aruco_detector()

    try:
        while True:
            frame = grab_frame(cap, n_flush=1)

            proj_w = int(mapper.projector_width)
            proj_h = int(mapper.projector_height)

            current_image_background = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
            draw_useful_area_outline(current_image_background, useful_pts)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                valid_indices = [
                    i for i in range(len(ids))
                    if int(ids[i][0]) not in CALIBRATION_TAG_IDS
                ]

                if len(valid_indices) > 0:
                    ids = ids[valid_indices]
                    corners = [corners[i] for i in valid_indices]
                else:
                    ids = None
                    corners = []

            else:
                ids = None
                corners = []

            if ids is not None and len(corners) > 0:
                for i, corner in enumerate(corners):
                    marker_id_int = int(ids[i][0])

                    anchor_raw = get_marker_anchor_raw(corner)

                    # camera_raw -> graph
                    graph_anchor = mapper.camera_raw_to_graph(anchor_raw)

                    # graph -> useful projector area
                    projector_anchor = apply_homography_to_point(H_graph_to_useful, graph_anchor)

                    if projector_anchor is None:
                        continue

                    projector_x = int(round(float(projector_anchor[0])))
                    projector_y = int(round(float(projector_anchor[1])))

                    cv2.circle(current_image_background, (projector_x, projector_y), 10, (0, 0, 255), -1)
                    cv2.putText(
                        current_image_background,
                        f"ID {marker_id_int}",
                        (projector_x + 12, projector_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    # debug console
                    print(
                        f"ID {marker_id_int} | raw=({anchor_raw[0]:.1f},{anchor_raw[1]:.1f}) "
                        f"| graph=({graph_anchor[0]:.1f},{graph_anchor[1]:.1f}) "
                        f"| useful=({projector_anchor[0]:.1f},{projector_anchor[1]:.1f})"
                    )

            display.display_image_on_projector_monitor(current_image_background)
            show_camera_window(frame, corners=corners if ids is not None else None, ids=ids)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        display.close_display()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()