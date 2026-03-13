# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np


# =========================================================
# PARAMETRES
# =========================================================
PATTERN_SIZE = (9, 6)           # coins internes (colonnes, lignes)
SQUARE_SIZE_M = 0.024           # taille case damier en mètres
MIN_VALID_POSES = 6

PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

VALID_POSES = [
    "pose_01", "pose_02", "pose_03", "pose_04",
    "pose_06", "pose_09", "pose_10", "pose_11", "pose_12"
]

# ---------------------------------------------------------
# chemins
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]   # si script dans src/core/calibration/
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_ROOT = BASE_DIR / "projector_calibration_data"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
PROJECTOR_JSON_PATH = CONFIG_DIR / "projector_calibration_moreno_refined.json"
PROJECTOR_MATRIX_NPY_PATH = CONFIG_DIR / "proj_mint_moreno_refined.npy"
PROJECTOR_DIST_NPY_PATH = CONFIG_DIR / "proj_distcoef_moreno_refined.npy"
DEBUG_JSON_PATH = CONFIG_DIR / "projector_calibration_debug.json"

# ---------------------------------------------------------
# paramètres Moreno renforcés
# ---------------------------------------------------------
SEARCH_RADIUS = 35
MIN_WHITE_BLACK_DIFF = 40
MIN_LOCAL_POINTS = 50
RANSAC_REPROJ_THRESH = 2.0
MIN_INLIERS = 35

# ---------------------------------------------------------
# calibration projecteur contrainte
# ---------------------------------------------------------
CALIB_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS |
    cv2.CALIB_FIX_PRINCIPAL_POINT |
    cv2.CALIB_ZERO_TANGENT_DIST |
    cv2.CALIB_FIX_K3 |
    cv2.CALIB_FIX_K4 |
    cv2.CALIB_FIX_K5 |
    cv2.CALIB_FIX_K6
)

TERM_CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-10
)

# ---------------------------------------------------------
# matrice initiale plausible
# ---------------------------------------------------------
INITIAL_FX = 4000.0
INITIAL_FY = 4000.0
INITIAL_CX = PROJECTOR_WIDTH / 2.0
INITIAL_CY = PROJECTOR_HEIGHT / 2.0


# =========================================================
# OUTILS
# =========================================================
def ensure_output_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def build_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def flatten_dist(dist):
    arr = np.array(dist, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.T
    return arr


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_calibration(json_path: Path):
    data = load_json(json_path)

    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise KeyError(f"Format invalide dans {json_path}")

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = flatten_dist(data["dist_coeffs"])
    return camera_matrix, dist_coeffs


def load_graycode_pattern(width, height):
    if not hasattr(cv2, "structured_light_GrayCodePattern"):
        raise RuntimeError(
            "opencv-contrib-python est requis pour structured_light_GrayCodePattern."
        )
    return cv2.structured_light_GrayCodePattern.create(width, height)


def detect_checker_corners(image_bgr, pattern_size):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if found and corners is not None:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.astype(np.float32)


def load_pose_images(pose_dir: Path):
    captures_dir = pose_dir / "camera_captures"

    if not captures_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {captures_dir}")

    white = cv2.imread(str(captures_dir / "white.png"), cv2.IMREAD_GRAYSCALE)
    black = cv2.imread(str(captures_dir / "black.png"), cv2.IMREAD_GRAYSCALE)

    if white is None or black is None:
        raise RuntimeError(f"white.png ou black.png manquante dans {captures_dir}")

    if white.shape != black.shape:
        raise RuntimeError(f"white/black de tailles différentes dans {captures_dir}")

    capture_files = sorted(captures_dir.glob("capture_*.png"))
    if not capture_files:
        raise RuntimeError(f"Aucune capture Gray code trouvée dans {captures_dir}")

    captures = []
    for p in capture_files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Capture introuvable ou illisible : {p}")
        if img.shape != white.shape:
            raise RuntimeError(
                f"Taille incohérente pour {p.name} : {img.shape} vs {white.shape}"
            )
        captures.append(img)

    return white, black, captures


def decode_projector_pixel(graycode, captures, x, y):
    try:
        ok, proj_pix = graycode.getProjPixel(captures, int(x), int(y))
    except Exception:
        return None

    if not ok:
        return None

    px, py = proj_pix
    if px < 0 or px >= PROJECTOR_WIDTH or py < 0 or py >= PROJECTOR_HEIGHT:
        return None

    return float(px), float(py)


def collect_local_correspondences(graycode, captures, white, black, corner_xy, radius):
    h, w = white.shape[:2]
    cx, cy = corner_xy
    cx = int(round(cx))
    cy = int(round(cy))

    cam_pts = []
    proj_pts = []

    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)

    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if int(white[yy, xx]) - int(black[yy, xx]) < MIN_WHITE_BLACK_DIFF:
                continue

            proj = decode_projector_pixel(graycode, captures, xx, yy)
            if proj is None:
                continue

            cam_pts.append([float(xx), float(yy)])
            proj_pts.append([proj[0], proj[1]])

    if len(cam_pts) < MIN_LOCAL_POINTS:
        return None, None

    cam_pts = np.array(cam_pts, dtype=np.float32)
    proj_pts = np.array(proj_pts, dtype=np.float32)
    return cam_pts, proj_pts


def estimate_projector_corner_local_homography(graycode, captures, white, black, corner_xy, radius):
    cam_pts, proj_pts = collect_local_correspondences(
        graycode, captures, white, black, corner_xy, radius
    )

    if cam_pts is None or proj_pts is None:
        return None, 0

    H, mask = cv2.findHomography(
        cam_pts,
        proj_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH
    )

    if H is None or mask is None:
        return None, 0

    inliers = int(mask.sum())
    if inliers < MIN_INLIERS:
        return None, inliers

    inlier_cam = cam_pts[mask.ravel() == 1]
    inlier_proj = proj_pts[mask.ravel() == 1]

    if len(inlier_cam) < MIN_INLIERS:
        return None, inliers

    H_refined, _ = cv2.findHomography(inlier_cam, inlier_proj, method=0)
    if H_refined is None:
        return None, inliers

    corner = np.array([[[corner_xy[0], corner_xy[1]]]], dtype=np.float32)
    proj_corner = cv2.perspectiveTransform(corner, H_refined)[0, 0]

    px, py = float(proj_corner[0]), float(proj_corner[1])
    if px < 0 or px >= PROJECTOR_WIDTH or py < 0 or py >= PROJECTOR_HEIGHT:
        return None, inliers

    return np.array([px, py], dtype=np.float32), inliers


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    per_view = []
    total_err = 0.0
    total_n = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err_l2 = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        err_per_point = err_l2 / len(projected)

        per_view.append(float(err_per_point))
        total_err += err_l2
        total_n += len(projected)

    mean_err = total_err / total_n if total_n > 0 else float("inf")
    return per_view, float(mean_err)


def is_projector_calibration_plausible(K, dist):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if fx <= 0 or fy <= 0:
        return False, "fx/fy non positifs"
    if not (0 <= cx <= PROJECTOR_WIDTH):
        return False, "cx hors image"
    if not (0 <= cy <= PROJECTOR_HEIGHT):
        return False, "cy hors image"

    flat_dist = np.array(dist).reshape(-1)
    if np.any(~np.isfinite(flat_dist)):
        return False, "distorsion non finie"

    return True, "OK"


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    print("=== CALIBRATION PROJECTEUR (Moreno raffiné) ===")
    print(f"Dossier poses : {DATA_ROOT}")
    print(f"Fichier calibration caméra : {CAMERA_CALIB_PATH}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dossier introuvable : {DATA_ROOT}")

    camera_matrix, dist_coeffs = load_camera_calibration(CAMERA_CALIB_PATH)
    graycode = load_graycode_pattern(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

    objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    objpoints = []
    proj_imgpoints = []

    successful_poses = []
    failed_poses = []
    debug_poses = []

    for pose_name in VALID_POSES:
        pose_dir = DATA_ROOT / pose_name
        print(f"\nTraitement {pose_name}...")

        pose_debug = {
            "pose": pose_name,
            "status": "failed",
            "reason": "",
            "num_detected_checker_corners": 0,
            "num_projector_corners": 0,
            "mean_inliers": None
        }

        try:
            white_bgr = cv2.imread(str(pose_dir / "camera_captures" / "white.png"))
            if white_bgr is None:
                pose_debug["reason"] = "white.png introuvable ou illisible"
                print("  white.png introuvable")
                failed_poses.append(pose_name)
                debug_poses.append(pose_debug)
                continue

            found, corners = detect_checker_corners(white_bgr, PATTERN_SIZE)
            if not found or corners is None:
                pose_debug["reason"] = "damier non détecté"
                print("  damier non détecté")
                failed_poses.append(pose_name)
                debug_poses.append(pose_debug)
                continue

            pose_debug["num_detected_checker_corners"] = int(len(corners))

            white, black, captures = load_pose_images(pose_dir)

            projector_corners = []
            inliers_stats = []
            ok_pose = True

            for c in corners.reshape(-1, 2):
                proj_pt, n_inliers = estimate_projector_corner_local_homography(
                    graycode=graycode,
                    captures=captures,
                    white=white,
                    black=black,
                    corner_xy=c,
                    radius=SEARCH_RADIUS
                )

                if proj_pt is None:
                    ok_pose = False
                    break

                projector_corners.append(proj_pt)
                inliers_stats.append(int(n_inliers))

            if not ok_pose or len(projector_corners) != len(corners):
                pose_debug["reason"] = "estimation Moreno insuffisante"
                print("  estimation Moreno insuffisante")
                failed_poses.append(pose_name)
                debug_poses.append(pose_debug)
                continue

            mean_inliers = float(np.mean(inliers_stats)) if inliers_stats else 0.0
            pose_debug["mean_inliers"] = mean_inliers
            pose_debug["num_projector_corners"] = int(len(projector_corners))

            if mean_inliers < MIN_INLIERS + 5:
                pose_debug["reason"] = f"inliers moyens trop faibles ({mean_inliers:.1f})"
                print(f"  pose rejetée (inliers moyens trop faibles: {mean_inliers:.1f})")
                failed_poses.append(pose_name)
                debug_poses.append(pose_debug)
                continue

            projector_corners = np.array(projector_corners, dtype=np.float32).reshape(-1, 1, 2)

            objpoints.append(objp.copy())
            proj_imgpoints.append(projector_corners)
            successful_poses.append(pose_name)

            pose_debug["status"] = "accepted"
            pose_debug["reason"] = "OK"
            debug_poses.append(pose_debug)

            print(f"  OK ({len(projector_corners)} coins projecteur, inliers moyens={mean_inliers:.1f})")

        except Exception as e:
            pose_debug["reason"] = str(e)
            print(f"  erreur : {e}")
            failed_poses.append(pose_name)
            debug_poses.append(pose_debug)
            continue

    print("\n=== BILAN PROJECTEUR ===")
    print("Poses retenues :", successful_poses)
    print("Poses rejetées :", failed_poses)
    print("Nb retenues :", len(successful_poses))

    if len(successful_poses) < MIN_VALID_POSES:
        raise RuntimeError(
            f"Pas assez de poses exploitables pour calibrer le projecteur : "
            f"{len(successful_poses)} (minimum requis : {MIN_VALID_POSES})"
        )

    proj_matrix_init = np.array([
        [INITIAL_FX, 0.0, INITIAL_CX],
        [0.0, INITIAL_FY, INITIAL_CY],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    proj_dist_init = np.zeros((8, 1), dtype=np.float64)

    rms, proj_matrix, proj_dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        proj_imgpoints,
        (PROJECTOR_WIDTH, PROJECTOR_HEIGHT),
        proj_matrix_init,
        proj_dist_init,
        flags=CALIB_FLAGS,
        criteria=TERM_CRIT
    )

    per_view_errors, mean_error = compute_reprojection_errors(
        objpoints,
        proj_imgpoints,
        rvecs,
        tvecs,
        proj_matrix,
        proj_dist
    )

    is_ok, plausibility_msg = is_projector_calibration_plausible(proj_matrix, proj_dist)

    print("\n=== RESULTATS PROJECTEUR ===")
    print("Erreur RMS :", rms)
    print("Projector matrix :\n", proj_matrix)
    print("Projector dist coeffs :\n", proj_dist.ravel())
    print("Erreur moyenne de reprojection (px) :", mean_error)
    print("Erreurs par pose (px) :", per_view_errors)
    print("Contrôle de plausibilité :", plausibility_msg)

    out = {
        "projector_width": PROJECTOR_WIDTH,
        "projector_height": PROJECTOR_HEIGHT,
        "pattern_size": [PATTERN_SIZE[0], PATTERN_SIZE[1]],
        "square_size_m": SQUARE_SIZE_M,
        "num_valid_poses": len(successful_poses),
        "used_poses": successful_poses,
        "rms_error": float(rms),
        "mean_reprojection_error_px": float(mean_error),
        "per_view_reprojection_errors_px": per_view_errors,
        "projector_matrix": proj_matrix.tolist(),
        "projector_dist_coeffs": proj_dist.tolist(),
        "plausibility_check": {
            "is_ok": bool(is_ok),
            "message": plausibility_msg
        }
    }

    with open(PROJECTOR_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    np.save(PROJECTOR_MATRIX_NPY_PATH, proj_matrix)
    np.save(PROJECTOR_DIST_NPY_PATH, proj_dist)

    debug_data = {
        "data_root": str(DATA_ROOT),
        "camera_calibration_path": str(CAMERA_CALIB_PATH),
        "valid_poses_requested": VALID_POSES,
        "accepted_poses": successful_poses,
        "rejected_poses": failed_poses,
        "pose_details": debug_poses
    }

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=4)

    print("\n=== FICHIERS SAUVEGARDES ===")
    print(f"- {PROJECTOR_JSON_PATH}")
    print(f"- {PROJECTOR_MATRIX_NPY_PATH}")
    print(f"- {PROJECTOR_DIST_NPY_PATH}")
    print(f"- {DEBUG_JSON_PATH}")


if __name__ == "__main__":
    main()
# Methode 2# # -*- coding: utf-8 -*-
# import json
# from pathlib import Path

# import cv2
# import numpy as np


# # =========================================================
# # PARAMETRES
# # =========================================================
# ROOT = Path("projector_calibration_data")

# VALID_POSES = [
#     "pose_01", "pose_02", "pose_03", "pose_04",
#     "pose_06", "pose_09", "pose_10",
#     "pose_11", "pose_12"
# ]

# PATTERN_SIZE = (9, 6)          # coins internes
# SQUARE_SIZE_M = 0.024          # adapte si besoin

# PROJECTOR_WIDTH = 3840
# PROJECTOR_HEIGHT = 2160

# CAMERA_CALIB_PATH = Path("camera_calibration.json")

# # -----------------------------
# # paramètres Moreno renforcés
# # -----------------------------
# SEARCH_RADIUS = 35
# MIN_WHITE_BLACK_DIFF = 40
# MIN_LOCAL_POINTS = 50
# RANSAC_REPROJ_THRESH = 2.0
# MIN_INLIERS = 35

# # calibration projecteur contrainte
# CALIB_FLAGS = (
#     cv2.CALIB_USE_INTRINSIC_GUESS |
#     cv2.CALIB_FIX_PRINCIPAL_POINT |
#     cv2.CALIB_ZERO_TANGENT_DIST |
#     cv2.CALIB_FIX_K3 |
#     cv2.CALIB_FIX_K4 |
#     cv2.CALIB_FIX_K5 |
#     cv2.CALIB_FIX_K6
# )

# TERM_CRIT = (
#     cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
#     200,
#     1e-10
# )

# # matrice initiale plausible
# INITIAL_FX = 4000.0
# INITIAL_FY = 4000.0
# INITIAL_CX = PROJECTOR_WIDTH / 2.0
# INITIAL_CY = PROJECTOR_HEIGHT / 2.0


# # =========================================================
# # OUTILS
# # =========================================================
# def build_object_points(pattern_size, square_size):
#     objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
#     objp *= square_size
#     return objp


# def load_camera_calibration(json_path: Path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
#     dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
#     return camera_matrix, dist_coeffs


# def load_graycode_pattern(width, height):
#     if not hasattr(cv2, "structured_light_GrayCodePattern"):
#         raise RuntimeError("opencv-contrib-python est requis.")
#     return cv2.structured_light_GrayCodePattern.create(width, height)


# def detect_checker_corners(image_bgr, pattern_size):
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
#     if not found:
#         return False, None

#     return True, corners.astype(np.float32)


# def load_pose_images(pose_dir: Path):
#     captures_dir = pose_dir / "camera_captures"

#     white = cv2.imread(str(captures_dir / "white.png"), cv2.IMREAD_GRAYSCALE)
#     black = cv2.imread(str(captures_dir / "black.png"), cv2.IMREAD_GRAYSCALE)

#     if white is None or black is None:
#         raise RuntimeError(f"white/black manquantes dans {pose_dir}")

#     capture_files = sorted(captures_dir.glob("capture_*.png"))
#     captures = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in capture_files]

#     if any(c is None for c in captures):
#         raise RuntimeError(f"Une ou plusieurs captures introuvables dans {pose_dir}")

#     return white, black, captures


# def decode_projector_pixel(graycode, captures, x, y):
#     try:
#         ok, proj_pix = graycode.getProjPixel(captures, int(x), int(y))
#     except Exception:
#         return None

#     if not ok:
#         return None

#     px, py = proj_pix
#     if px < 0 or px >= PROJECTOR_WIDTH or py < 0 or py >= PROJECTOR_HEIGHT:
#         return None

#     return float(px), float(py)


# def collect_local_correspondences(graycode, captures, white, black, corner_xy, radius):
#     h, w = white.shape[:2]
#     cx, cy = corner_xy
#     cx = int(round(cx))
#     cy = int(round(cy))

#     cam_pts = []
#     proj_pts = []

#     x0 = max(0, cx - radius)
#     x1 = min(w - 1, cx + radius)
#     y0 = max(0, cy - radius)
#     y1 = min(h - 1, cy + radius)

#     for yy in range(y0, y1 + 1):
#         for xx in range(x0, x1 + 1):
#             if int(white[yy, xx]) - int(black[yy, xx]) < MIN_WHITE_BLACK_DIFF:
#                 continue

#             proj = decode_projector_pixel(graycode, captures, xx, yy)
#             if proj is None:
#                 continue

#             cam_pts.append([float(xx), float(yy)])
#             proj_pts.append([proj[0], proj[1]])

#     if len(cam_pts) < MIN_LOCAL_POINTS:
#         return None, None

#     cam_pts = np.array(cam_pts, dtype=np.float32)
#     proj_pts = np.array(proj_pts, dtype=np.float32)
#     return cam_pts, proj_pts


# def estimate_projector_corner_local_homography(graycode, captures, white, black, corner_xy, radius):
#     cam_pts, proj_pts = collect_local_correspondences(
#         graycode, captures, white, black, corner_xy, radius
#     )
#     if cam_pts is None:
#         return None, 0

#     H, mask = cv2.findHomography(
#         cam_pts,
#         proj_pts,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=RANSAC_REPROJ_THRESH
#     )
#     if H is None or mask is None:
#         return None, 0

#     inliers = int(mask.sum())
#     if inliers < MIN_INLIERS:
#         return None, inliers

#     # raffinement sur inliers uniquement
#     inlier_cam = cam_pts[mask.ravel() == 1]
#     inlier_proj = proj_pts[mask.ravel() == 1]

#     if len(inlier_cam) < MIN_INLIERS:
#         return None, inliers

#     H_refined, _ = cv2.findHomography(inlier_cam, inlier_proj, method=0)
#     if H_refined is None:
#         return None, inliers

#     corner = np.array([[[corner_xy[0], corner_xy[1]]]], dtype=np.float32)
#     proj_corner = cv2.perspectiveTransform(corner, H_refined)[0, 0]

#     px, py = float(proj_corner[0]), float(proj_corner[1])
#     if px < 0 or px >= PROJECTOR_WIDTH or py < 0 or py >= PROJECTOR_HEIGHT:
#         return None, inliers

#     return np.array([px, py], dtype=np.float32), inliers


# def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
#     per_view = []
#     total_err = 0.0
#     total_n = 0

#     for i in range(len(objpoints)):
#         projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
#         err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
#         per_view.append(float(err))

#         total_err += cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
#         total_n += len(projected)

#     mean_err = total_err / total_n if total_n > 0 else float("inf")
#     return per_view, float(mean_err)


# # =========================================================
# # MAIN
# # =========================================================
# def main():
#     camera_matrix, dist_coeffs = load_camera_calibration(CAMERA_CALIB_PATH)
#     graycode = load_graycode_pattern(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

#     objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

#     objpoints = []
#     proj_imgpoints = []

#     successful_poses = []
#     failed_poses = []

#     for pose_name in VALID_POSES:
#         pose_dir = ROOT / pose_name
#         print(f"\nTraitement {pose_name}...")

#         white_bgr = cv2.imread(str(pose_dir / "camera_captures" / "white.png"))
#         if white_bgr is None:
#             print("  white.png introuvable")
#             failed_poses.append(pose_name)
#             continue

#         found, corners = detect_checker_corners(white_bgr, PATTERN_SIZE)
#         if not found:
#             print("  damier non détecté")
#             failed_poses.append(pose_name)
#             continue

#         white, black, captures = load_pose_images(pose_dir)

#         projector_corners = []
#         ok_pose = True
#         inliers_stats = []

#         for c in corners.reshape(-1, 2):
#             proj_pt, n_inliers = estimate_projector_corner_local_homography(
#                 graycode=graycode,
#                 captures=captures,
#                 white=white,
#                 black=black,
#                 corner_xy=c,
#                 radius=SEARCH_RADIUS
#             )

#             if proj_pt is None:
#                 ok_pose = False
#                 break

#             projector_corners.append(proj_pt)
#             inliers_stats.append(n_inliers)

#         if not ok_pose or len(projector_corners) != len(corners):
#             print("  estimation Moreno insuffisante")
#             failed_poses.append(pose_name)
#             continue

#         # contrôle simple sur la qualité moyenne de la pose
#         mean_inliers = float(np.mean(inliers_stats))
#         if mean_inliers < MIN_INLIERS + 5:
#             print(f"  pose rejetée (inliers moyens trop faibles: {mean_inliers:.1f})")
#             failed_poses.append(pose_name)
#             continue

#         projector_corners = np.array(projector_corners, dtype=np.float32).reshape(-1, 1, 2)

#         objpoints.append(objp.copy())
#         proj_imgpoints.append(projector_corners)
#         successful_poses.append(pose_name)

#         print(f"  OK ({len(projector_corners)} coins projecteur, inliers moyens={mean_inliers:.1f})")

#     print("\n=== BILAN PROJECTEUR ===")
#     print("Poses retenues :", successful_poses)
#     print("Poses rejetées :", failed_poses)
#     print("Nb retenues :", len(successful_poses))

#     if len(successful_poses) < 6:
#         raise RuntimeError("Pas assez de poses exploitables pour calibrer le projecteur.")

#     # matrice initiale plausible
#     proj_matrix_init = np.array([
#         [INITIAL_FX, 0.0, INITIAL_CX],
#         [0.0, INITIAL_FY, INITIAL_CY],
#         [0.0, 0.0, 1.0]
#     ], dtype=np.float64)

#     proj_dist_init = np.zeros((8, 1), dtype=np.float64)

#     ret, proj_matrix, proj_dist, rvecs, tvecs = cv2.calibrateCamera(
#         objpoints,
#         proj_imgpoints,
#         (PROJECTOR_WIDTH, PROJECTOR_HEIGHT),
#         proj_matrix_init,
#         proj_dist_init,
#         flags=CALIB_FLAGS,
#         criteria=TERM_CRIT
#     )

#     per_view_errors, mean_error = compute_reprojection_errors(
#         objpoints, proj_imgpoints, rvecs, tvecs, proj_matrix, proj_dist
#     )

#     print("\n=== RESULTATS PROJECTEUR ===")
#     print("Erreur RMS :", ret)
#     print("Projector matrix :\n", proj_matrix)
#     print("Projector dist coeffs :\n", proj_dist.ravel())
#     print("Erreur moyenne de reprojection (px) :", mean_error)
#     print("Erreurs par pose (px) :", per_view_errors)

#     out = {
#         "projector_width": PROJECTOR_WIDTH,
#         "projector_height": PROJECTOR_HEIGHT,
#         "pattern_size": [PATTERN_SIZE[0], PATTERN_SIZE[1]],
#         "square_size_m": SQUARE_SIZE_M,
#         "used_poses": successful_poses,
#         "rms_error": float(ret),
#         "mean_reprojection_error_px": float(mean_error),
#         "per_view_reprojection_errors_px": per_view_errors,
#         "projector_matrix": proj_matrix.tolist(),
#         "projector_dist_coeffs": proj_dist.tolist(),
#     }

#     with open("projector_calibration_moreno_refined.json", "w", encoding="utf-8") as f:
#         json.dump(out, f, indent=4)

#     np.save("proj_mint_moreno_refined.npy", proj_matrix)
#     np.save("proj_distcoef_moreno_refined.npy", proj_dist)

#     print("\nSauvegardé dans :")
#     print("- projector_calibration_moreno_refined.json")
#     print("- proj_mint_moreno_refined.npy")
#     print("- proj_distcoef_moreno_refined.npy")


# if __name__ == "__main__":
#     main()