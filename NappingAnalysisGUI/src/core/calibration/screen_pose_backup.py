# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time

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

# dimensions physiques utiles de l'écran diffusant
SCREEN_WIDTH_MM = 590.0
SCREEN_HEIGHT_MM = 590.0

# taille de la grille logique de ton appli
GRID_SIZE = 700

# rectangle blanc affiché pour aider au clic
DISPLAY_RECT_WIDTH = 2200
DISPLAY_RECT_HEIGHT = 2200
DISPLAY_BG_LEVEL = 220
DISPLAY_BORDER_THICKNESS = 8

# raffinement local du clic
REFINE_ROI_HALF_SIZE = 25

# chemins
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
PROJECTOR_CALIB_PATH = CONFIG_DIR / "projector_calibration_moreno_refined.json"
STEREO_CALIB_PATH = CONFIG_DIR / "stereo_camera_projector_calibration.json"

OUTPUT_CALIB_PATH = CONFIG_DIR / "calibration_data.json"
OUTPUT_POSE_PATH = CONFIG_DIR / "screen_pose_manual.json"
OUTPUT_DEBUG_PATH = CONFIG_DIR / "manual_pose_debug.png"


# =========================================================
# OUTILS JSON / CALIB
# =========================================================
def ensure_output_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dist(dist):
    arr = np.array(dist, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.T
    return arr


def load_calibrations():
    cam_data = load_json(CAMERA_CALIB_PATH)
    proj_data = load_json(PROJECTOR_CALIB_PATH)
    stereo_data = load_json(STEREO_CALIB_PATH)

    K_cam = np.array(cam_data["camera_matrix"], dtype=np.float64)
    dist_cam = flatten_dist(cam_data["dist_coeffs"])

    K_proj = np.array(proj_data["projector_matrix"], dtype=np.float64)
    if "projector_dist_coeffs" in proj_data:
        dist_proj = flatten_dist(proj_data["projector_dist_coeffs"])
    elif "dist_coeffs" in proj_data:
        dist_proj = flatten_dist(proj_data["dist_coeffs"])
    else:
        raise KeyError("Aucune distorsion projecteur trouvée.")

    R_cp = np.array(stereo_data["R"], dtype=np.float64)
    T_cp = np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1)

    return K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp


# =========================================================
# PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = "ProjectorManualPose"

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


def build_display_pattern():
    """
    Mire visuelle uniquement pour aider au clic.
    Les calculs utilisent les coins réels de l'écran cliqués dans l'image caméra.
    """
    img = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    x0 = (PROJECTOR_WIDTH - DISPLAY_RECT_WIDTH) // 2
    y0 = (PROJECTOR_HEIGHT - DISPLAY_RECT_HEIGHT) // 2
    x1 = x0 + DISPLAY_RECT_WIDTH - 1
    y1 = y0 + DISPLAY_RECT_HEIGHT - 1

    cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        (DISPLAY_BG_LEVEL, DISPLAY_BG_LEVEL, DISPLAY_BG_LEVEL),
        thickness=-1
    )
    cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        (255, 255, 255),
        thickness=DISPLAY_BORDER_THICKNESS
    )

    for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        cv2.circle(img, p, 16, (0, 0, 255), -1)

    return img


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

    h, w = frame.shape[:2]
    if (w, h) != (CAMERA_WIDTH, CAMERA_HEIGHT):
        cap.release()
        raise RuntimeError(
            f"Résolution caméra incohérente : obtenu {w}x{h}, attendu {CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )

    return cap


def grab_frame(cap, n_flush=3):
    for _ in range(n_flush):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Erreur capture caméra.")

    return frame


# =========================================================
# OUTILS GEOMETRIE
# =========================================================
def undistort_pixel_points(points_px, K, dist):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float32)


def pose_to_homography(K, R, t):
    return K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))


def compute_reprojection_error(objpoints, imgpoints, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    img = np.array(imgpoints, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(img - proj, axis=1)
    return float(np.mean(d)), d.tolist(), proj


def refine_click_to_corner(gray, x, y, roi_half_size=25):
    """
    Raffine le clic en cherchant un coin fort localement.
    Si rien n'est trouvé, retourne le point brut.
    """
    h, w = gray.shape[:2]

    x0 = max(0, x - roi_half_size)
    y0 = max(0, y - roi_half_size)
    x1 = min(w, x + roi_half_size + 1)
    y1 = min(h, y + roi_half_size + 1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return float(x), float(y)

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5
    )

    if corners is None:
        return float(x), float(y)

    corners = corners.reshape(-1, 2)
    target = np.array([x - x0, y - y0], dtype=np.float32)
    d2 = np.sum((corners - target) ** 2, axis=1)
    best = corners[np.argmin(d2)]

    return float(best[0] + x0), float(best[1] + y0)


# =========================================================
# INTERACTION
# =========================================================
clicked_points = []
frozen_gray = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, frozen_gray

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        rx, ry = refine_click_to_corner(
            frozen_gray,
            x,
            y,
            roi_half_size=REFINE_ROI_HALF_SIZE
        )
        clicked_points.append([rx, ry])


def draw_points(img, pts):
    out = img.copy()
    labels = ["TL", "TR", "BR", "BL"]

    for i, p in enumerate(pts):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            out,
            labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    if len(pts) == 4:
        cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)

    return out


# =========================================================
# MAIN
# =========================================================
def main():
    global clicked_points, frozen_gray

    ensure_output_dir()

    print("=== POSE ECRAN MANUELLE ROBUSTE ===")

    K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp = load_calibrations()

    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)
    projector.show(build_display_pattern())

    cap = open_camera()

    try:
        time.sleep(1.0)
        frame = grab_frame(cap)
        frozen = frame.copy()
        frozen_gray = cv2.cvtColor(frozen, cv2.COLOR_BGR2GRAY)

        print("\nClique exactement les 4 coins internes réels de la plaque écran dans cet ordre :")
        print("1. haut gauche")
        print("2. haut droit")
        print("3. bas droit")
        print("4. bas gauche")
        print("\nTouches :")
        print("  r = reset")
        print("  s = sauvegarder")
        print("  q = quitter")

        cv2.namedWindow("Manual Screen Pose", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Manual Screen Pose", mouse_callback)

        while True:
            display = draw_points(frozen, clicked_points)
            cv2.imshow("Manual Screen Pose", display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                clicked_points = []

            elif key == ord("q"):
                return

            elif key == ord("s"):
                if len(clicked_points) != 4:
                    print("Il faut exactement 4 points.")
                    continue

                cam_points_raw = np.array(clicked_points, dtype=np.float32)
                cam_points_undist = undistort_pixel_points(cam_points_raw, K_cam, dist_cam)

                # coins physiques de la plaque écran
                object_points_m = np.array([
                    [0.0, 0.0, 0.0],
                    [SCREEN_WIDTH_MM / 1000.0, 0.0, 0.0],
                    [SCREEN_WIDTH_MM / 1000.0, SCREEN_HEIGHT_MM / 1000.0, 0.0],
                    [0.0, SCREEN_HEIGHT_MM / 1000.0, 0.0],
                ], dtype=np.float32)

                graph_points = np.array([
                    [0, 0],
                    [GRID_SIZE - 1, 0],
                    [GRID_SIZE - 1, GRID_SIZE - 1],
                    [0, GRID_SIZE - 1],
                ], dtype=np.float32)

                screen_points_mm = np.array([
                    [0.0, 0.0],
                    [SCREEN_WIDTH_MM, 0.0],
                    [SCREEN_WIDTH_MM, SCREEN_HEIGHT_MM],
                    [0.0, SCREEN_HEIGHT_MM],
                ], dtype=np.float32)

                # -------------------------------------------------
                # estimation pose écran -> caméra
                # on teste IPPE puis ITERATIVE, on garde la meilleure
                # -------------------------------------------------
                candidates = []

                for flag_name, flag in [
                    ("IPPE", cv2.SOLVEPNP_IPPE),
                    ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
                ]:
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points_m,
                        cam_points_raw,
                        K_cam,
                        dist_cam,
                        flags=flag
                    )
                    if ok:
                        mean_err, per_point_errs, proj_pts = compute_reprojection_error(
                            object_points_m,
                            cam_points_raw,
                            rvec,
                            tvec,
                            K_cam,
                            dist_cam
                        )
                        candidates.append((flag_name, rvec, tvec, mean_err, per_point_errs, proj_pts))

                if not candidates:
                    raise RuntimeError("solvePnP a échoué pour toutes les méthodes.")

                best = min(candidates, key=lambda x: x[3])
                best_flag, rvec_sc, tvec_sc, pnp_mean_err, pnp_per_point_errs, pnp_proj_pts = best

                R_sc, _ = cv2.Rodrigues(rvec_sc)

                # écran -> projecteur via stéréo caméra -> projecteur
                R_sp = R_cp @ R_sc
                T_sp = R_cp @ tvec_sc + T_cp

                # homographies analytiques
                H_screen_to_cam = pose_to_homography(K_cam, R_sc, tvec_sc)
                H_screen_to_proj = pose_to_homography(K_proj, R_sp, T_sp)

                H_proj = H_screen_to_proj @ np.linalg.inv(H_screen_to_cam)
                H_proj = H_proj / H_proj[2, 2]
                H_inv_proj = np.linalg.inv(H_proj)

                # caméra undistordue -> grille logique
                H_graph, _ = cv2.findHomography(cam_points_undist, graph_points)
                H_graph = H_graph / H_graph[2, 2]
                H_inv_graph = np.linalg.inv(H_graph)

                # caméra undistordue -> mm écran
                H_cam_to_screen_mm, _ = cv2.findHomography(cam_points_undist, screen_points_mm)
                H_cam_to_screen_mm = H_cam_to_screen_mm / H_cam_to_screen_mm[2, 2]
                H_screen_mm_to_cam = np.linalg.inv(H_cam_to_screen_mm)

                # image debug
                debug = draw_points(frozen, clicked_points)
                for p in pnp_proj_pts:
                    x, y = int(round(p[0])), int(round(p[1]))
                    cv2.circle(debug, (x, y), 6, (0, 255, 0), 2)

                cv2.imwrite(str(OUTPUT_DEBUG_PATH), debug)

                calib_data = {
                    "H_proj": H_proj.tolist(),
                    "H_inv_proj": H_inv_proj.tolist(),
                    "H_graph": H_graph.tolist(),
                    "H_inv_graph": H_inv_graph.tolist()
                }

                with open(OUTPUT_CALIB_PATH, "w", encoding="utf-8") as f:
                    json.dump(calib_data, f, indent=4)

                pose_data = {
                    "method": "manual_screen_corners_solvepnp",
                    "screen_width_mm": SCREEN_WIDTH_MM,
                    "screen_height_mm": SCREEN_HEIGHT_MM,
                    "camera_points_raw_TL_TR_BR_BL": cam_points_raw.tolist(),
                    "camera_points_undist_TL_TR_BR_BL": cam_points_undist.tolist(),
                    "screen_points_mm_TL_TR_BR_BL": screen_points_mm.tolist(),
                    "graph_points_TL_TR_BR_BL": graph_points.tolist(),
                    "camera_matrix": K_cam.tolist(),
                    "camera_dist_coeffs": dist_cam.tolist(),
                    "projector_matrix": K_proj.tolist(),
                    "projector_dist_coeffs": dist_proj.tolist(),
                    "R_camera_to_projector": R_cp.tolist(),
                    "T_camera_to_projector": T_cp.tolist(),
                    "solvepnp_method_selected": best_flag,
                    "solvepnp_mean_reprojection_error_px": pnp_mean_err,
                    "solvepnp_per_point_errors_px": pnp_per_point_errs,
                    "rvec_screen_to_camera": rvec_sc.tolist(),
                    "tvec_screen_to_camera": tvec_sc.tolist(),
                    "R_screen_to_camera": R_sc.tolist(),
                    "R_screen_to_projector": R_sp.tolist(),
                    "T_screen_to_projector": T_sp.tolist(),
                    "H_proj": H_proj.tolist(),
                    "H_inv_proj": H_inv_proj.tolist(),
                    "H_graph": H_graph.tolist(),
                    "H_inv_graph": H_inv_graph.tolist(),
                    "H_cam_to_screen_mm": H_cam_to_screen_mm.tolist(),
                    "H_screen_mm_to_cam": H_screen_mm_to_cam.tolist()
                }

                with open(OUTPUT_POSE_PATH, "w", encoding="utf-8") as f:
                    json.dump(pose_data, f, indent=4)

                print("\n=== RESULTATS ===")
                print("Méthode solvePnP choisie :", best_flag)
                print("Erreur reprojection solvePnP (px) :", pnp_mean_err)
                print("R_screen_to_camera =\n", R_sc)
                print("t_screen_to_camera =\n", tvec_sc)
                print("\nFichiers sauvegardés :")
                print("-", OUTPUT_CALIB_PATH)
                print("-", OUTPUT_POSE_PATH)
                print("-", OUTPUT_DEBUG_PATH)
                return

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()