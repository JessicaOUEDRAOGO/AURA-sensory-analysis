# -*- coding: utf-8 -*-
import cv2
import numpy as np

# =========================================================
# PARAMETRES
# =========================================================
CAM_TOP_ID = 1
CAM_BOTTOM_ID = 0

WIDTH = 640
HEIGHT = 480

REF_TL = (10.0, 10.0)
REF_TR = (590.0, 10.0)
REF_BR = (580.0, 580.0)
REF_BL = (10.0, 580.0)

TABLE_POINTS = np.array([REF_TL, REF_TR, REF_BR, REF_BL], dtype=np.float32)

CORNER_TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# =========================================================
# ARUCO
# =========================================================
def create_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def select_corner(corners, position):
    if position == "TL":
        return corners[np.argmin(corners[:, 0] + corners[:, 1])]
    elif position == "TR":
        return corners[np.argmin(-corners[:, 0] + corners[:, 1])]
    elif position == "BR":
        return corners[np.argmax(corners[:, 0] + corners[:, 1])]
    elif position == "BL":
        return corners[np.argmax(-corners[:, 0] + corners[:, 1])]


def detect_4_points(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    ids = ids.flatten().tolist()
    found = {}

    for i, marker_id in enumerate(ids):
        if int(marker_id) in CORNER_TAG_IDS.values():
            pts = corners[i][0].astype(np.float32)
            found[int(marker_id)] = pts

    required = [
        CORNER_TAG_IDS["TL"],
        CORNER_TAG_IDS["TR"],
        CORNER_TAG_IDS["BR"],
        CORNER_TAG_IDS["BL"],
    ]

    if not all(k in found for k in required):
        return None

    ordered = np.array([
        select_corner(found[CORNER_TAG_IDS["TL"]], "TL"),
        select_corner(found[CORNER_TAG_IDS["TR"]], "TR"),
        select_corner(found[CORNER_TAG_IDS["BR"]], "BR"),
        select_corner(found[CORNER_TAG_IDS["BL"]], "BL"),
    ], dtype=np.float32)

    return ordered


# =========================================================
# HOMOGRAPHIE
# =========================================================
def compute_H(image_pts):
    H, _ = cv2.findHomography(image_pts, TABLE_POINTS)
    return H


def map_point(H, pt):
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, H)
    return mapped[0, 0]


# =========================================================
# CAMERA
# =========================================================
def open_cam(cam_id):
    print(f"Ouverture caméra {cam_id}...")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra ID={cam_id}"
        )

    # CRITIQUE : réduire la charge USB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # TRÈS IMPORTANT
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # buffer minimal
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # flush
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Caméra {cam_id} ouverte mais aucune frame")

    h, w = frame.shape[:2]
    print(f"Caméra {cam_id} OK → {w}x{h}")

    return cap

def grab_frame(cap, n_flush=1):
    for _ in range(n_flush):
        cap.grab()
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


# =========================================================
# AFFICHAGE
# =========================================================
def draw_overlay(frame, pts, H, cam_label):
    out = frame.copy()

    # Points détectés
    if pts is not None:
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(pts):
            cv2.circle(out, tuple(p.astype(int)), 8, (0, 255, 0), -1)
            cv2.putText(out, labels[i],
                        tuple((p + np.array([10, -10])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Contour
        cv2.polylines(out, [pts.astype(np.int32)], True, (0, 255, 255), 2)

    # Statut homographie
    status = f"{cam_label} | H: {'OK' if H is not None else 'non calibre'}"
    color = (0, 255, 0) if H is not None else (0, 0, 255)
    cv2.putText(out, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Nombre de tags détectés
    n = len(pts) if pts is not None else 0
    cv2.putText(out, f"Tags: {n}/4", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return out


# =========================================================
# MAIN
# =========================================================
def main():
    print("Ouverture des caméras...")

    try:
        cap_top = open_cam(CAM_TOP_ID)
    except RuntimeError as e:
        print(f"[ERREUR] {e}")
        return

    try:
        cap_bot = open_cam(CAM_BOTTOM_ID)
    except RuntimeError as e:
        print(f"[ERREUR] {e}")
        cap_top.release()
        return
    print("\n--- DEBUG CAM ---")
    print(f"TOP ID = {CAM_TOP_ID}")
    print(f"BOTTOM ID = {CAM_BOTTOM_ID}")
    detector = create_detector()

    H_top = None
    H_bot = None

    cv2.namedWindow("TOP", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BOTTOM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TOP", 960, 540)
    cv2.resizeWindow("BOTTOM", 960, 540)

    print("\nCommandes :")
    print("  t = calibrer caméra TOP")
    print("  b = calibrer caméra BOTTOM")
    print("  q = quitter")

    while True:
        frame_top = grab_frame(cap_top, n_flush=1)
        frame_bot = grab_frame(cap_bot, n_flush=1)

        if frame_top is None or frame_bot is None:
            print("[WARN] Frame manquante, on continue...")
            cv2.waitKey(30)
            continue

        pts_top = detect_4_points(frame_top, detector)
        pts_bot = detect_4_points(frame_bot, detector)

        display_top = draw_overlay(frame_top, pts_top, H_top, "TOP")
        display_bot = draw_overlay(frame_bot, pts_bot, H_bot, "BOTTOM")

        # Test en temps réel si les deux H sont calibrés
        if H_top is not None and H_bot is not None:
            center = np.array([WIDTH / 2.0, HEIGHT / 2.0])
            pt_top = map_point(H_top, center)
            pt_bot = map_point(H_bot, center)

            info_top = f"Centre -> table: ({pt_top[0]:.1f}, {pt_top[1]:.1f}) mm"
            info_bot = f"Centre -> table: ({pt_bot[0]:.1f}, {pt_bot[1]:.1f}) mm"

            cv2.putText(display_top, info_top, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_bot, info_bot, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("TOP", display_top)
        cv2.imshow("BOTTOM", display_bot)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("t"):
            if pts_top is not None:
                H_top = compute_H(pts_top)
                print("\n[TOP] Calibration OK")
                print("H_top_to_table:")
                print(H_top)
            else:
                print("[TOP] 4 tags non détectés, calibration impossible.")

        elif key == ord("b"):
            if pts_bot is not None:
                H_bot = compute_H(pts_bot)
                print("\n[BOTTOM] Calibration OK")
                print("H_bottom_to_table:")
                print(H_bot)
            else:
                print("[BOTTOM] 4 tags non détectés, calibration impossible.")

        elif key == ord("q") or key == 27:
            print("Sortie.")
            break

    cap_top.release()
    cap_bot.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
