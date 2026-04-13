# -*- coding: utf-8 -*-
from math import dist

import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMETRES
# =========================================================
CAMERA_ID = 0   # ⚠️ caméra TOP
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

GRID_SIZE = 700

CONFIG_DIR = Path("config")
OUTPUT_PATH = CONFIG_DIR / "camera_top_mapping.json"

# IDs des coins (identiques à bottom)
CORNER_TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# =========================================================
# CAMERA
# =========================================================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera TOP non ouverte")

    return cap

def grab_frame(cap):
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Erreur capture")
    return frame

def load_camera_top_calibration():
    data = json.load(open("config/camera_calibration_top.json"))

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    return K, dist

def undistort_points(pts, K, dist):
    pts = pts.reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1,2)

# =========================================================
# ARUCO
# =========================================================
def create_detector():
    dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dict_aruco, params)

def select_corner(corners, pos):
    if pos == "TL":
        return corners[np.argmax(-corners[:,0] + corners[:,1])]
    elif pos == "TR":
        return corners[np.argmax(corners[:,0] + corners[:,1])]
    elif pos == "BR":
        return corners[np.argmin(-corners[:,0] + corners[:,1])]
    elif pos == "BL":
        return corners[np.argmin(corners[:,0] + corners[:,1])]

def detect_tags(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    ids = ids.flatten()
    detected = {}

    # -------------------------------------------------
    # 1) Stockage des tags détectés
    # -------------------------------------------------
    for i, id_ in enumerate(ids):
        id_ = int(id_)
        if id_ in CORNER_TAG_IDS.values():
            pts = corners[i][0].astype(np.float32)  # (4,2)

            # centre du tag (robuste)
            center = np.mean(pts, axis=0)

            detected[id_] = {
                "corners": pts,
                "center": center
            }

    # -------------------------------------------------
    # 2) Vérifie que les 4 coins sont présents
    # -------------------------------------------------
    required_ids = [
        CORNER_TAG_IDS["TL"],  # 42
        CORNER_TAG_IDS["TR"],  # 43
        CORNER_TAG_IDS["BR"],  # 40
        CORNER_TAG_IDS["BL"],  # 41
    ]

    if not all(tag_id in detected for tag_id in required_ids):
        return detected, None

    # -------------------------------------------------
    # 3) Construction des points à partir des coins des tags
    # -------------------------------------------------
    pts = np.array([
        detected[CORNER_TAG_IDS["TL"]]["center"],
        detected[CORNER_TAG_IDS["TR"]]["center"],
        detected[CORNER_TAG_IDS["BR"]]["center"],
        detected[CORNER_TAG_IDS["BL"]]["center"],
    ], dtype=np.float32)

    return detected, pts

# =========================================================
# HOMOGRAPHIE TOP -> GRAPH
# =========================================================
def compute_H_top(pts, K, dist):

    pts_undist = undistort_points(pts, K, dist)

    table_pts= np.array([
        [10, 580],   # TL_top → BL_table
        [580, 580],  # TR_top → BR_table
        [590, 10],   # BR_top → TR_table
        [10, 10],    # BL_top → TL_table
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_undist, table_pts)

    return H, pts_undist

def apply_H(H, pt):
    p = np.array([pt[0], pt[1], 1.0])
    out = H @ p
    out /= out[2]
    return out[:2]

def line_intersection(p1, p2, p3, p4):
    """
    Intersection de deux droites 2D :
    (p1,p2) et (p3,p4)
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-12:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py], dtype=np.float32)
# =========================================================
# DEBUG VISUEL
# =========================================================
def draw_debug(frame, detected, pts, H, pts_undist=None):
    out = frame.copy()

    # dessin des tags
    if detected:
        for id_, data in detected.items():
            corners = data["corners"]
            center = data["center"]

            cv2.polylines(out, [corners.astype(int)], True, (0,255,0), 2)
            cv2.circle(out, (int(center[0]), int(center[1])), 5, (0,0,255), -1)
            cv2.putText(out, str(id_), (int(center[0]), int(center[1])), 0, 1, (255,0,0), 2)

    if pts is not None:
        labels = ["TL","TR","BR","BL"]

        for i, p in enumerate(pts):
            x,y = int(p[0]), int(p[1])
            cv2.circle(out, (x,y), 8, (255,255,0), 2)
            cv2.putText(out, labels[i], (x+10,y), 0, 0.8, (255,255,0),2)

        # centre projeté
        center_cam = line_intersection(
            pts_undist[0], pts_undist[2],
            pts_undist[1], pts_undist[3]
        )

        if center_cam is not None:
            center_graph = apply_H(H, center_cam)

        cv2.putText(
            out,
            f"GRAPH: {center_graph[0]:.1f},{center_graph[1]:.1f}",
            (50,50),
            0,1,(0,255,255),2
        )

    return out

# =========================================================
# GRILLE DEBUG
# =========================================================
def draw_graph_grid():
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

    for i in range(0, GRID_SIZE, 50):
        cv2.line(img, (i,0), (i,GRID_SIZE), (0,255,0),1)
        cv2.line(img, (0,i), (GRID_SIZE,i), (0,255,0),1)

    cv2.circle(img, (GRID_SIZE//2, GRID_SIZE//2), 10, (0,0,255), -1)

    return img

# =========================================================
# MAIN
# =========================================================
def main():
    CONFIG_DIR.mkdir(exist_ok=True)

    cap = open_camera()
    detector = create_detector()

    H_final = None

    while True:
        frame = grab_frame(cap)

        detected, pts = detect_tags(frame, detector)
        K, dist = load_camera_top_calibration()
        if pts is not None:

            H, pts_undist = compute_H_top(pts, K, dist)
            H_final = H

            debug = draw_debug(frame, detected, pts, H, pts_undist)

            # test projection centre
            center_cam = line_intersection(
                pts_undist[0], pts_undist[2],   # TL -> BR
                pts_undist[1], pts_undist[3]    # TR -> BL
            )

            if center_cam is not None:
                proj = apply_H(H, center_cam)
                print(f"CENTER TABLE: {proj}")

        else:
            debug = frame

        cv2.imshow("Camera TOP Debug", debug)

        if H_final is not None:
            grid = draw_graph_grid()
            cv2.imshow("Graph View", grid)

        key = cv2.waitKey(1)

        if key == ord('s') and H_final is not None:
            with open(OUTPUT_PATH, "w") as f:
                json.dump({"H_top_to_graph": H_final.tolist()}, f, indent=4)
            print("✅ Sauvegarde OK")

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()