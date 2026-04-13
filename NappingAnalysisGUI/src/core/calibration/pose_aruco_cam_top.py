# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMETRES
# =========================================================
CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

CONFIG_DIR = Path("config")
OUTPUT_PATH = CONFIG_DIR / "camera_top_mapping.json"

# on garde les 4 IDs utiles, mais sans leur imposer un rôle spatial
CORNER_TAG_IDS = [42, 43, 40, 41]

# repère cible défini PAR RAPPORT A L'IMAGE CAM_TOP
TABLE_W = 600
TABLE_H = 600
MARGIN = 10

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


# =========================================================
# CALIB
# =========================================================
def load_camera_top_calibration():
    with open(CONFIG_DIR / "camera_calibration_top.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def undistort_points(pts, K, dist):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


# =========================================================
# ARUCO
# =========================================================
def create_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def select_outer_corner(corners):
    """
    corners : (4,2) d'un tag ArUco
    Retourne le coin du tag le plus éloigné du centre global de l'image,
    ce qui correspond bien au coin 'extérieur' du tag placé au bord de la table.
    Cette stratégie est indépendante du nom TL/TR/BR/BL hérité de la bottom.
    """
    center_img = np.array([CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0], dtype=np.float32)
    dists = np.linalg.norm(corners - center_img, axis=1)
    return corners[np.argmax(dists)]


def order_points_image_top(pts):
    """
    Trie 4 points selon LE REPERE DE L'IMAGE CAM_TOP :
    retourne [TL, TR, BR, BL]
    """
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_tags(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}, None, None, gray

    ids = ids.flatten()
    detected = {}

    for i, marker_id in enumerate(ids):
        marker_id = int(marker_id)
        if marker_id in CORNER_TAG_IDS:
            pts = corners[i][0].astype(np.float32)
            center = np.mean(pts, axis=0)

            detected[marker_id] = {
                "corners": pts,
                "center": center,
                "selected_outer_corner": select_outer_corner(pts),
            }

    if len(detected) != 4:
        return detected, None, None, gray

    raw_selected = np.array(
        [detected[mid]["selected_outer_corner"] for mid in detected.keys()],
        dtype=np.float32
    )

    ordered_pts = order_points_image_top(raw_selected)

    return detected, raw_selected, ordered_pts, gray


# =========================================================
# HOMOGRAPHIE TOP -> TABLE (repère cam_top)
# =========================================================
def compute_H_top(ordered_pts, K, dist):
    pts_undist = undistort_points(ordered_pts, K, dist)

    table_pts = np.array([
        [MARGIN, MARGIN],                              # TL dans l'image top
        [TABLE_W - MARGIN, MARGIN],                    # TR
        [TABLE_W - MARGIN, TABLE_H - MARGIN],          # BR
        [MARGIN, TABLE_H - MARGIN],                    # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_undist, table_pts)
    return H, pts_undist, table_pts


def apply_H(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    out = H @ p
    if abs(out[2]) < 1e-12:
        return None
    out /= out[2]
    return out[:2]


def line_intersection(p1, p2, p3, p4):
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
# DEBUG
# =========================================================
def draw_debug(frame, detected, raw_selected, ordered_pts, H, pts_undist=None):
    out = frame.copy()

    for marker_id, data in detected.items():
        corners = data["corners"]
        center = data["center"]
        sel = data["selected_outer_corner"]

        cv2.polylines(out, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(out, tuple(np.round(center).astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(out, str(marker_id), tuple(np.round(center).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.circle(out, tuple(np.round(sel).astype(int)), 9, (0, 255, 255), 2)

    if raw_selected is not None:
        for p in raw_selected:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(out, (x, y), 6, (255, 0, 255), -1)

    if ordered_pts is not None:
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(ordered_pts):
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(out, (x, y), 8, (255, 255, 0), 2)
            cv2.putText(out, labels[i], (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.polylines(out, [np.round(ordered_pts).astype(np.int32)], True, (255, 255, 0), 2)

        if pts_undist is not None and H is not None:
            center_cam = line_intersection(
                pts_undist[0], pts_undist[2],
                pts_undist[1], pts_undist[3]
            )
            if center_cam is not None:
                center_table = apply_H(H, center_cam)
                if center_table is not None:
                    cv2.putText(
                        out,
                        f"TABLE: {center_table[0]:.1f},{center_table[1]:.1f}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

    return out


def draw_table_grid():
    img = np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8)

    for i in range(0, TABLE_W, 50):
        cv2.line(img, (i, 0), (i, TABLE_H - 1), (0, 255, 0), 1)
    for j in range(0, TABLE_H, 50):
        cv2.line(img, (0, j), (TABLE_W - 1, j), (0, 255, 0), 1)

    cv2.circle(img, (TABLE_W // 2, TABLE_H // 2), 10, (0, 0, 255), -1)
    cv2.putText(img, "CENTER", (TABLE_W // 2 - 35, TABLE_H // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


# =========================================================
# MAIN
# =========================================================
def main():
    CONFIG_DIR.mkdir(exist_ok=True)

    cap = open_camera()
    detector = create_detector()
    K, dist = load_camera_top_calibration()

    H_final = None

    while True:
        frame = grab_frame(cap)

        detected, raw_selected, ordered_pts, gray = detect_tags(frame, detector)

        if ordered_pts is not None:
            H, pts_undist, table_pts = compute_H_top(ordered_pts, K, dist)
            H_final = H

            # DEBUG : affiche les points ordonnés
            labels = ["TL", "TR", "BR", "BL"]
            for i, (cam_pt, undist_pt, table_pt) in enumerate(zip(ordered_pts, pts_undist, table_pts)):
                print(f"{labels[i]}: cam={cam_pt} → undist={undist_pt} → table_target={table_pt}")
            
            # Affiche aussi sur img
            print(f"\nImage size: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            print(f"Cam corners bounds: x=[{ordered_pts[:,0].min():.0f}, {ordered_pts[:,0].max():.0f}], y=[{ordered_pts[:,1].min():.0f}, {ordered_pts[:,1].max():.0f}]")

            debug = draw_debug(frame, detected, raw_selected, ordered_pts, H, pts_undist)

            center_cam = line_intersection(
                pts_undist[0], pts_undist[2],
                pts_undist[1], pts_undist[3]
            )
            if center_cam is not None:
                proj = apply_H(H, center_cam)
                if proj is not None:
                    print(f"CENTER TABLE: {proj}")

        else:
            debug = frame

        cv2.imshow("Camera TOP Debug", debug)

        if H_final is not None:
            cv2.imshow("Table View", draw_table_grid())

        key = cv2.waitKey(1)

        if key == ord("s") and H_final is not None:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "H_top_to_table": H_final.tolist(),
                    "table_width": TABLE_W,
                    "table_height": TABLE_H,
                    "margin": MARGIN,
                    "point_order_in_cam_top": ["TL", "TR", "BR", "BL"]
                }, f, indent=4)
            print("Sauvegarde OK")

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()