# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np


# =========================================================
# PARAMETRES
# =========================================================
PATTERN_SIZE = (9, 6)           # coins internes (colonnes, lignes)
SQUARE_SIZE_M = 0.024
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
PROJECTOR_CALIB_PATH = CONFIG_DIR / "projector_calibration_moreno_refined.json"

STEREO_JSON_PATH = CONFIG_DIR / "stereo_camera_projector_calibration.json"
STEREO_DEBUG_JSON_PATH = CONFIG_DIR / "stereo_camera_projector_calibration_debug.json"

# ---------------------------------------------------------
# paramètres Moreno locaux
# ---------------------------------------------------------
SEARCH_RADIUS = 35
MIN_WHITE_BLACK_DIFF = 40
MIN_LOCAL_POINTS = 50
RANSAC_REPROJ_THRESH = 2.0
MIN_INLIERS = 35

# ---------------------------------------------------------
# stereo calibrate
# ---------------------------------------------------------
STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC

STEREO_TERM_CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-10
)


# =========================================================
# OUTILS GENERAUX
# =========================================================
def ensure_output_dir() -> None:
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


def build_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


# =========================================================
# CHARGEMENT CALIBRATIONS
# =========================================================
def load_camera_calibration(path: Path):
    data = load_json(path)

    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise KeyError(f"Format invalide dans {path}")

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = flatten_dist(data["dist_coeffs"])
    return K, dist


def load_projector_calibration(path: Path):
    data = load_json(path)

    if "projector_matrix" not in data:
        raise KeyError(f"Clé 'projector_matrix' absente dans {path}")

    # compatibilité avec anciennes et nouvelles versions
    if "projector_dist_coeffs" in data:
        dist_key = "projector_dist_coeffs"
    elif "dist_coeffs" in data:
        dist_key = "dist_coeffs"
    else:
        raise KeyError(f"Aucune clé de distorsion projecteur trouvée dans {path}")

    K = np.array(data["projector_matrix"], dtype=np.float64)
    dist = flatten_dist(data[dist_key])
    return K, dist


# =========================================================
# DETECTION DAMIER
# =========================================================
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


# =========================================================
# GRAY CODE / MORENO
# =========================================================
def load_graycode_pattern(width, height):
    if not hasattr(cv2, "structured_light_GrayCodePattern"):
        raise RuntimeError(
            "opencv-contrib-python est requis pour structured_light_GrayCodePattern."
        )
    return cv2.structured_light_GrayCodePattern.create(width, height)


def load_pose_images(pose_dir: Path):
    captures_dir = pose_dir / "camera_captures"
    if not captures_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {captures_dir}")

    white = cv2.imread(str(captures_dir / "white.png"), cv2.IMREAD_GRAYSCALE)
    black = cv2.imread(str(captures_dir / "black.png"), cv2.IMREAD_GRAYSCALE)

    if white is None or black is None:
        raise RuntimeError(f"white.png ou black.png manquante(s) dans {captures_dir}")

    if white.shape != black.shape:
        raise RuntimeError(f"white et black n'ont pas la même taille dans {captures_dir}")

    capture_files = sorted(captures_dir.glob("capture_*.png"))
    if not capture_files:
        raise RuntimeError(f"Aucune image Gray code trouvée dans {captures_dir}")

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


# =========================================================
# METRIQUES
# =========================================================
def compute_mono_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
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


def compute_relative_pose_from_two_pnps(objpoints, cam_imgpts, proj_imgpts, K_cam, dist_cam, K_proj, dist_proj):
    ok_cam, rvec_c, tvec_c = cv2.solvePnP(
        objpoints, cam_imgpts, K_cam, dist_cam, flags=cv2.SOLVEPNP_ITERATIVE
    )
    ok_proj, rvec_p, tvec_p = cv2.solvePnP(
        objpoints, proj_imgpts, K_proj, dist_proj, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok_cam or not ok_proj:
        return None

    R_c, _ = cv2.Rodrigues(rvec_c)
    R_p, _ = cv2.Rodrigues(rvec_p)

    # X_cam = R_c X_obj + t_c
    # X_proj = R_p X_obj + t_p
    # donc X_proj = R_cp X_cam + T_cp
    R_cp = R_p @ R_c.T
    T_cp = tvec_p - R_cp @ tvec_c

    return {
        "rvec_camera": rvec_c,
        "tvec_camera": tvec_c,
        "R_camera": R_c,
        "rvec_projector": rvec_p,
        "tvec_projector": tvec_p,
        "R_projector": R_p,
        "R_cp": R_cp,
        "T_cp": T_cp
    }


def rotation_angle_deg(R_a, R_b):
    R_delta = R_a @ R_b.T
    trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(trace_val)
    return float(np.degrees(angle_rad))


def translation_distance(a, b):
    return float(np.linalg.norm(np.array(a).reshape(3) - np.array(b).reshape(3)))


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    print("=== STEREOCALIBRATION CAMERA-PROJECTEUR ===")
    print(f"Dossier poses : {DATA_ROOT}")
    print(f"Calibration caméra : {CAMERA_CALIB_PATH}")
    print(f"Calibration projecteur : {PROJECTOR_CALIB_PATH}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dossier introuvable : {DATA_ROOT}")

    K_cam, dist_cam = load_camera_calibration(CAMERA_CALIB_PATH)
    K_proj, dist_proj = load_projector_calibration(PROJECTOR_CALIB_PATH)
    graycode = load_graycode_pattern(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

    objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    object_points = []
    cam_imgpoints = []
    proj_imgpoints = []

    accepted_poses = []
    rejected_poses = []
    pose_debug_list = []

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
                rejected_poses.append(pose_name)
                pose_debug_list.append(pose_debug)
                continue

            found, corners_cam = detect_checker_corners(white_bgr, PATTERN_SIZE)
            if not found or corners_cam is None:
                pose_debug["reason"] = "damier non détecté"
                print("  damier non détecté")
                rejected_poses.append(pose_name)
                pose_debug_list.append(pose_debug)
                continue

            pose_debug["num_detected_checker_corners"] = int(len(corners_cam))

            white, black, captures = load_pose_images(pose_dir)

            projector_corners = []
            inliers_stats = []
            ok_pose = True

            for c in corners_cam.reshape(-1, 2):
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

            if not ok_pose or len(projector_corners) != len(corners_cam):
                pose_debug["reason"] = "estimation Moreno insuffisante"
                print("  estimation Moreno insuffisante")
                rejected_poses.append(pose_name)
                pose_debug_list.append(pose_debug)
                continue

            mean_inliers = float(np.mean(inliers_stats)) if inliers_stats else 0.0
            pose_debug["mean_inliers"] = mean_inliers
            pose_debug["num_projector_corners"] = int(len(projector_corners))

            if mean_inliers < MIN_INLIERS + 5:
                pose_debug["reason"] = f"inliers moyens trop faibles ({mean_inliers:.1f})"
                print(f"  pose rejetée (inliers moyens trop faibles: {mean_inliers:.1f})")
                rejected_poses.append(pose_name)
                pose_debug_list.append(pose_debug)
                continue

            corners_proj = np.array(projector_corners, dtype=np.float32).reshape(-1, 1, 2)

            object_points.append(objp.copy())
            cam_imgpoints.append(corners_cam.astype(np.float32))
            proj_imgpoints.append(corners_proj.astype(np.float32))
            accepted_poses.append(pose_name)

            pose_debug["status"] = "accepted"
            pose_debug["reason"] = "OK"
            pose_debug_list.append(pose_debug)

            print(f"  OK ({len(corners_proj)} coins projecteur, inliers moyens={mean_inliers:.1f})")

        except Exception as e:
            pose_debug["reason"] = str(e)
            print(f"  erreur : {e}")
            rejected_poses.append(pose_name)
            pose_debug_list.append(pose_debug)
            continue

    print("\n=== BILAN STEREO ===")
    print("Poses retenues :", accepted_poses)
    print("Poses rejetées :", rejected_poses)
    print("Nb retenues :", len(accepted_poses))

    if len(accepted_poses) < MIN_VALID_POSES:
        raise RuntimeError(
            f"Pas assez de poses exploitables pour la stéréocalibration : "
            f"{len(accepted_poses)} (minimum requis : {MIN_VALID_POSES})"
        )

    # -----------------------------------------------------
    # Stéréocalibration avec intrinsèques figées
    # -----------------------------------------------------
    rms, K_cam_out, dist_cam_out, K_proj_out, dist_proj_out, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=object_points,
        imagePoints1=cam_imgpoints,
        imagePoints2=proj_imgpoints,
        cameraMatrix1=K_cam.copy(),
        distCoeffs1=dist_cam.copy(),
        cameraMatrix2=K_proj.copy(),
        distCoeffs2=dist_proj.copy(),
        imageSize=(PROJECTOR_WIDTH, PROJECTOR_HEIGHT),
        criteria=STEREO_TERM_CRIT,
        flags=STEREO_FLAGS
    )

    # -----------------------------------------------------
    # Erreurs mono par pose après estimation PnP
    # -----------------------------------------------------
    cam_rvecs = []
    cam_tvecs = []
    proj_rvecs = []
    proj_tvecs = []

    for i in range(len(object_points)):
        ok_cam, rvec_c, tvec_c = cv2.solvePnP(
            object_points[i], cam_imgpoints[i], K_cam_out, dist_cam_out, flags=cv2.SOLVEPNP_ITERATIVE
        )
        ok_proj, rvec_p, tvec_p = cv2.solvePnP(
            object_points[i], proj_imgpoints[i], K_proj_out, dist_proj_out, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok_cam or not ok_proj:
            raise RuntimeError(f"solvePnP a échoué pour la pose {accepted_poses[i]}")

        cam_rvecs.append(rvec_c)
        cam_tvecs.append(tvec_c)
        proj_rvecs.append(rvec_p)
        proj_tvecs.append(tvec_p)

    cam_per_view_errors, cam_mean_error = compute_mono_reprojection_errors(
        object_points, cam_imgpoints, cam_rvecs, cam_tvecs, K_cam_out, dist_cam_out
    )
    proj_per_view_errors, proj_mean_error = compute_mono_reprojection_errors(
        object_points, proj_imgpoints, proj_rvecs, proj_tvecs, K_proj_out, dist_proj_out
    )

    # -----------------------------------------------------
    # Contrôle de cohérence par pose
    # -----------------------------------------------------
    per_pose_consistency = []

    for i in range(len(object_points)):
        rel = compute_relative_pose_from_two_pnps(
            object_points[i],
            cam_imgpoints[i],
            proj_imgpoints[i],
            K_cam_out,
            dist_cam_out,
            K_proj_out,
            dist_proj_out
        )

        if rel is None:
            per_pose_consistency.append({
                "pose": accepted_poses[i],
                "status": "pnp_failed"
            })
            continue

        angle_deg = rotation_angle_deg(rel["R_cp"], R)
        t_dist = translation_distance(rel["T_cp"], T)

        per_pose_consistency.append({
            "pose": accepted_poses[i],
            "status": "ok",
            "camera_reprojection_error_px": cam_per_view_errors[i],
            "projector_reprojection_error_px": proj_per_view_errors[i],
            "relative_rotation_error_deg": angle_deg,
            "relative_translation_error_m": t_dist,
            "R_cp_from_pose": rel["R_cp"].tolist(),
            "T_cp_from_pose": rel["T_cp"].reshape(3, 1).tolist()
        })

    valid_rot = [p["relative_rotation_error_deg"] for p in per_pose_consistency if p["status"] == "ok"]
    valid_trans = [p["relative_translation_error_m"] for p in per_pose_consistency if p["status"] == "ok"]

    mean_rot_err = float(np.mean(valid_rot)) if valid_rot else None
    mean_trans_err = float(np.mean(valid_trans)) if valid_trans else None

    print("\n=== RESULTATS STEREO ===")
    print("Erreur RMS stereo :", rms)
    print("R (camera -> projector) =\n", R)
    print("T (camera -> projector) =\n", T.reshape(3, 1))
    print("Erreur moyenne reprojection camera (px) :", cam_mean_error)
    print("Erreur moyenne reprojection projecteur (px) :", proj_mean_error)
    print("Erreur moyenne rotation relative par pose (deg) :", mean_rot_err)
    print("Erreur moyenne translation relative par pose (m) :", mean_trans_err)

    # -----------------------------------------------------
    # Sauvegarde principale
    # -----------------------------------------------------
    stereo_data = {
        "pattern_size": [PATTERN_SIZE[0], PATTERN_SIZE[1]],
        "square_size_m": SQUARE_SIZE_M,
        "projector_width": PROJECTOR_WIDTH,
        "projector_height": PROJECTOR_HEIGHT,
        "num_valid_poses": len(accepted_poses),
        "used_poses": accepted_poses,
        "stereo_rms_error": float(rms),
        "camera_matrix": K_cam_out.tolist(),
        "camera_dist_coeffs": dist_cam_out.tolist(),
        "projector_matrix": K_proj_out.tolist(),
        "projector_dist_coeffs": dist_proj_out.tolist(),
        "R": R.tolist(),
        "T": T.reshape(3, 1).tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "camera_mean_reprojection_error_px": float(cam_mean_error),
        "projector_mean_reprojection_error_px": float(proj_mean_error),
        "camera_per_view_reprojection_errors_px": cam_per_view_errors,
        "projector_per_view_reprojection_errors_px": proj_per_view_errors,
        "mean_relative_rotation_error_deg": mean_rot_err,
        "mean_relative_translation_error_m": mean_trans_err
    }

    with open(STEREO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stereo_data, f, indent=4)

    # -----------------------------------------------------
    # Sauvegarde debug détaillée
    # -----------------------------------------------------
    debug_data = {
        "data_root": str(DATA_ROOT),
        "camera_calibration_path": str(CAMERA_CALIB_PATH),
        "projector_calibration_path": str(PROJECTOR_CALIB_PATH),
        "valid_poses_requested": VALID_POSES,
        "accepted_poses": accepted_poses,
        "rejected_poses": rejected_poses,
        "pose_detection_details": pose_debug_list,
        "per_pose_consistency": per_pose_consistency
    }

    with open(STEREO_DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=4)

    print("\n=== FICHIERS SAUVEGARDES ===")
    print(f"- {STEREO_JSON_PATH}")
    print(f"- {STEREO_DEBUG_JSON_PATH}")


if __name__ == "__main__":
    main()