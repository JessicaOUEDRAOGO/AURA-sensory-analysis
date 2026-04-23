# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np


# =========================================================
# PARAMETRES
# =========================================================
PATTERN_SIZE   = (9, 6)       # nb de coins internes (colonnes, lignes)
SQUARE_SIZE_M  = 0.024        # taille d'une case en mètres
MIN_VALID_IMAGES = 15         # fisheye a besoin de plus d'images que le modèle standard

# Dossiers
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR   = PROJECT_ROOT / "config"
IMAGES_DIR   = BASE_DIR / "img_checker"
IMAGES_GLOB  = "*.jpg"

# Fichiers de sortie
CAMERA_JSON_PATH       = CONFIG_DIR / "camera_calibration_fisheye.json"
CAMERA_MATRIX_NPY_PATH = CONFIG_DIR / "cam_mint_fisheye.npy"
CAMERA_DIST_NPY_PATH   = CONFIG_DIR / "cam_distcoef_fisheye.npy"
DEBUG_JSON_PATH        = CONFIG_DIR / "camera_calibration_fisheye_debug.json"

# Sauvegarde des images annotées
SAVE_DETECTED_IMAGES = True
DETECTED_DIR         = BASE_DIR / "detected_checkerboards_fisheye"

# Critères de raffinement subpix
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Prévisualisation
SHOW_PREVIEW    = True
PREVIEW_WIDTH   = 1280
PREVIEW_HEIGHT  = 720
PREVIEW_DELAY_MS = 300


# =========================================================
# OUTILS
# =========================================================
def ensure_output_dirs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_DETECTED_IMAGES:
        DETECTED_DIR.mkdir(parents=True, exist_ok=True)


def build_object_points(pattern_size, square_size_m):
    """Format imposé par cv2.fisheye.calibrate : (N, 1, 3) float64"""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def detect_checkerboard(gray, pattern_size):
    """Détection robuste : SB d'abord, fallback classique + subpix."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if found and corners is not None:
            return True, corners.reshape(-1, 1, 2).astype(np.float64)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return True, corners.reshape(-1, 1, 2).astype(np.float64)


def compute_reprojection_errors(object_points, image_points, rvecs, tvecs, K, D):
    per_image = []
    total_err, total_pts = 0.0, 0

    for i in range(len(object_points)):
        proj, _ = cv2.fisheye.projectPoints(object_points[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(image_points[i], proj, cv2.NORM_L2)
        per_pt = err / len(proj)
        per_image.append(float(per_pt))
        total_err += err
        total_pts  += len(proj)

    mean_err = total_err / total_pts if total_pts > 0 else float("inf")
    return per_image, float(mean_err)


def check_pose_diversity(rvecs):
    """
    Vérifie que les poses du damier sont suffisamment variées.
    Avertit si toutes les rotations sont trop similaires (cas caméra+damier fixes).
    """
    angles = [np.linalg.norm(r) * 180 / np.pi for r in rvecs]
    angle_std = float(np.std(angles))
    print(f"\n[DIVERSITE] Std des angles de rotation : {angle_std:.1f}°")
    if angle_std < 5.0:
        print("  ⚠️  ATTENTION : très faible diversité de poses !")
        print("     → Les images semblent toutes prises sous le même angle.")
        print("     → Recalibrer avec le damier incliné/déplacé dans le champ.")
    else:
        print("  ✓  Diversité de poses correcte.")
    return angle_std


def validate_intrinsics(K, img_size):
    """Sanity-check des intrinsèques : fx/fy ratio et position du point principal."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w,  h  = img_size

    ratio = fx / fy
    cx_rel = cx / w
    cy_rel = cy / h

    print("\n[VALIDATION INTRINSEQUES]")
    print(f"  fx={fx:.1f}  fy={fy:.1f}  ratio fx/fy={ratio:.3f}  (idéal : 0.95-1.05)")
    print(f"  cx={cx:.1f} ({cx_rel*100:.1f}% de {w}px)  cy={cy:.1f} ({cy_rel*100:.1f}% de {h}px)")
    print(f"  (idéal : cx/cy proches de 50% de la résolution)")

    warnings = []
    if not (0.85 < ratio < 1.15):
        warnings.append(f"⚠️  Ratio fx/fy={ratio:.3f} anormal (hors [0.85, 1.15])")
    if not (0.35 < cx_rel < 0.65):
        warnings.append(f"⚠️  cx={cx:.1f} anormal ({cx_rel*100:.1f}% ≠ ~50%)")
    if not (0.35 < cy_rel < 0.65):
        warnings.append(f"⚠️  cy={cy:.1f} anormal ({cy_rel*100:.1f}% ≠ ~50%)")

    if warnings:
        for w_ in warnings:
            print(" ", w_)
        print("  → Calibration suspecte : augmenter la diversité des poses du damier.")
    else:
        print("  ✓  Intrinsèques cohérentes.")

    return len(warnings) == 0


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dirs()

    images = sorted(IMAGES_DIR.glob(IMAGES_GLOB))
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans : {IMAGES_DIR}")

    print("=== CALIBRATION CAMERA FISHEYE ===")
    print(f"Images trouvées : {len(images)}")

    objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    object_points = []
    image_points  = []
    used_images   = []
    rejected      = []
    img_size      = None

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            rejected.append({"file": img_path.name, "reason": "illisible"})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])

        if img_size is None:
            img_size = size
        elif size != img_size:
            rejected.append({"file": img_path.name, "reason": f"taille {size} ≠ {img_size}"})
            continue

        found, corners = detect_checkerboard(gray, PATTERN_SIZE)

        if not found:
            print(f"  [SKIP] Damier non détecté : {img_path.name}")
            rejected.append({"file": img_path.name, "reason": "damier non détecté"})
            continue

        object_points.append(objp.copy())
        image_points.append(corners)
        used_images.append(img_path.name)
        print(f"  [OK]   {img_path.name}")

        if SAVE_DETECTED_IMAGES:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, PATTERN_SIZE,
                                      corners.reshape(-1, 1, 2).astype(np.float32), found)
            cv2.imwrite(str(DETECTED_DIR / img_path.name), vis)

        if SHOW_PREVIEW:
            vis2 = img.copy()
            cv2.drawChessboardCorners(vis2, PATTERN_SIZE,
                                      corners.reshape(-1, 1, 2).astype(np.float32), found)
            cv2.imshow("Calibration fisheye", cv2.resize(vis2, (PREVIEW_WIDTH, PREVIEW_HEIGHT)))
            cv2.waitKey(PREVIEW_DELAY_MS)

    cv2.destroyAllWindows()

    valid_count = len(object_points)
    print(f"\nImages valides : {valid_count} / {len(images)}")

    if valid_count < MIN_VALID_IMAGES:
        raise RuntimeError(
            f"Pas assez d'images valides : {valid_count} < {MIN_VALID_IMAGES}\n"
            "→ Reprendre la calibration avec plus de poses variées du damier."
        )

    # ------------------------------------------------------------------
    # CALIBRATION FISHEYE
    #
    # Flags choisis :
    #   CALIB_RECOMPUTE_EXTRINSIC  : réestime les extrinsèques à chaque itération
    #   CALIB_FIX_SKEW             : force skew=0 (objectif standard)
    #   CALIB_FIX_PRINCIPAL_POINT  : fixe cx/cy au centre de l'image
    #                                → empêche cy=982 sur 1080px
    #                                → à retirer si vraiment décentré
    #
    # NOTE : on N'utilise PAS CALIB_CHECK_COND car il rejette parfois des
    # images valides quand la diversité est limite.
    # ------------------------------------------------------------------
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    # Initialiser cx/cy au centre pour aider la convergence
    K[0, 2] = img_size[0] / 2.0
    K[1, 2] = img_size[1] / 2.0

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT   # cx/cy fixés au centre
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            img_size,
            K,
            D,
            None,
            None,
            flags,
            criteria
        )
    except cv2.error as e:
        print(f"\n[ERREUR] Calibration échouée : {e}")
        print("→ Le flag CALIB_CHECK_COND a probablement rejeté des images.")
        print("→ Relance sans CALIB_CHECK_COND (déjà le cas ici).")
        print("→ Si l'erreur persiste : ajouter plus de poses variées du damier.")
        return

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    per_image_errors, mean_err = compute_reprojection_errors(
        object_points, image_points, rvecs, tvecs, K, D
    )

    print("\n=== RESULTATS ===")
    print(f"RMS             : {rms:.4f}")
    print(f"Erreur moyenne  : {mean_err:.4f} px")
    print(f"Camera matrix   :\n{K}")
    print(f"Dist coeffs (k1,k2,k3,k4) : {D.ravel()}")

    check_pose_diversity(rvecs)
    intrinsics_ok = validate_intrinsics(K, img_size)

    if rms > 1.0:
        print(f"\n⚠️  RMS={rms:.3f} > 1.0 px — calibration de mauvaise qualité.")
        print("   → Ajouter des images avec damier très incliné dans les coins.")
    elif rms > 0.5:
        print(f"\n⚠️  RMS={rms:.3f} — calibration acceptable mais perfectible.")
    else:
        print(f"\n✓  RMS={rms:.3f} — bonne calibration.")

    # ------------------------------------------------------------------
    # SAUVEGARDE
    # ------------------------------------------------------------------
    np.save(CAMERA_MATRIX_NPY_PATH, K)
    np.save(CAMERA_DIST_NPY_PATH,   D)

    data = {
        "model":                      "fisheye",
        "pattern_size":               list(PATTERN_SIZE),
        "square_size_m":              SQUARE_SIZE_M,
        "image_size_px":              list(img_size),
        "num_input_images":           len(images),
        "num_valid_images":           valid_count,
        "rms_error":                  float(rms),
        "mean_reprojection_error_px": float(mean_err),
        "intrinsics_valid":           intrinsics_ok,
        "camera_matrix":              K.tolist(),
        "dist_coeffs":                D.tolist(),
        "flags_used": [
            "CALIB_RECOMPUTE_EXTRINSIC",
            "CALIB_FIX_SKEW",
            "CALIB_FIX_PRINCIPAL_POINT",
        ]
    }

    with open(CAMERA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    debug = {
        "model":        "fisheye",
        "used_images":  used_images,
        "rejected":     rejected,
        "per_image_reprojection_error_px": [
            {"file": used_images[i], "error_px": per_image_errors[i]}
            for i in range(valid_count)
        ]
    }

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=4)

    print(f"\n[SAVED] {CAMERA_JSON_PATH}")
    print(f"[SAVED] {DEBUG_JSON_PATH}")

    # ------------------------------------------------------------------
    # CONSEIL UNDISTORT pour la pose
    # ------------------------------------------------------------------
    print("\n=== UTILISATION POUR LA POSE ===")
    print("Avec le modèle fisheye, utiliser cv2.fisheye.undistortImage :")
    print("""
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, np.eye(3), balance=0.5
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, img_size, cv2.CV_16SC2
    )
    frame_undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    # Puis détection ArUco sur frame_undist avec new_K et dist=zeros
    """)


if __name__ == "__main__":
    main()