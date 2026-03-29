# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np


# =========================================================
# PARAMETRES
# =========================================================
PATTERN_SIZE = (9, 6)          # nb de coins internes (colonnes, lignes)
SQUARE_SIZE_M = 0.024          # taille d'une case en mètres
MIN_VALID_IMAGES = 8


# Dossiers
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]   # adapte si besoin selon l'emplacement réel du script
CONFIG_DIR = PROJECT_ROOT / "config"
IMAGES_DIR = BASE_DIR / "img_checker"
IMAGES_GLOB = "*.jpg"

# Fichiers de sortie
CAMERA_JSON_PATH = CONFIG_DIR / "camera_calibration.json"
CAMERA_MATRIX_NPY_PATH = CONFIG_DIR / "cam_mint.npy"
CAMERA_DIST_NPY_PATH = CONFIG_DIR / "cam_distcoef.npy"
DEBUG_JSON_PATH = CONFIG_DIR / "camera_calibration_debug.json"

# Sauvegarde des images avec damier détecté
SAVE_DETECTED_IMAGES = True
DETECTED_DIR = BASE_DIR / "detected_checkerboards"
SAVE_RAW_DETECTED_IMAGES = False   # True si tu veux aussi sauvegarder l'image brute
RAW_DETECTED_DIR = BASE_DIR / "detected_checkerboards_raw"

# Critères de raffinement
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Prévisualisation
SHOW_PREVIEW = True
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
PREVIEW_DELAY_MS = 200


# =========================================================
# OUTILS
# =========================================================
def ensure_output_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_DETECTED_IMAGES:
        DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_RAW_DETECTED_IMAGES:
        RAW_DETECTED_DIR.mkdir(parents=True, exist_ok=True)

def save_detected_checkerboard_image(image, image_path: Path, pattern_size, corners, found):
    """
    Sauvegarde une image annotée avec les coins du damier détectés.
    """
    vis = image.copy()
    # Dessine les coins du damier détectés sur l'image 
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)

    output_path = DETECTED_DIR / image_path.name
    ok = cv2.imwrite(str(output_path), vis)

    if not ok:
        print(f"[WARN] Impossible d'enregistrer l'image détectée : {output_path}")
    else:
        print(f"[SAVE] Image détectée sauvegardée : {output_path.name}")

def save_raw_detected_image(image, image_path: Path):
    """
    Sauvegarde l'image brute pour laquelle le damier a été détecté.
    """
    output_path = RAW_DETECTED_DIR / image_path.name
    ok = cv2.imwrite(str(output_path), image)

    if not ok:
        print(f"[WARN] Impossible d'enregistrer l'image brute : {output_path}")

def build_object_points(pattern_size, square_size_m):
    # Crée une matrice de zéros avec 9×6 = 54 lignes et 3 colonnes (x, y, z)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)               
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def find_images(images_dir: Path, pattern: str):
    images = sorted(images_dir.glob(pattern))
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans : {images_dir}")
    return images


def detect_checkerboard(gray, pattern_size):
    """
    Détection robuste du damier :
    1) findChessboardCornersSB si dispo
    2) fallback sur findChessboardCorners + cornerSubPix
    """
    # Méthode moderne et souvent plus robuste
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if found and corners is not None:
            return True, corners.astype(np.float32)

    # Fallback classique
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not found or corners is None:
        return False, None

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        SUBPIX_CRITERIA
    )
    return True, corners_refined.astype(np.float32)


def compute_reprojection_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    per_image_errors = []
    total_error = 0.0
    total_points = 0
# cv2.projectPoints : Projette les points 3D (object_points[i]) vers 2D en utilisant les paramètres calibrés (rvecs[i], tvecs[i], camera_matrix, dist_coeffs).
    for i in range(len(object_points)):
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
# Distance euclidienne totale entre points détectés et projetés.
        err_l2 = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
        err_per_point = err_l2 / len(projected_points)

        per_image_errors.append(float(err_per_point))
        total_error += err_l2
        total_points += len(projected_points)

    mean_error = total_error / total_points if total_points > 0 else float("inf")
    return per_image_errors, float(mean_error)


def resize_for_preview(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def flatten_dist_coeffs(dist_coeffs):
    arr = np.array(dist_coeffs, dtype=np.float64)
    return arr.reshape(-1).tolist()


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_output_dir()

    images = find_images(IMAGES_DIR, IMAGES_GLOB)
    objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    object_points = []
    image_points = []

    used_images = []
    rejected_images = []

    img_size = None

    print("=== CALIBRATION CAMERA ===")
    print(f"Dossier images : {IMAGES_DIR}")
    print(f"Nombre d'images trouvées : {len(images)}")

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Image illisible : {image_path.name}")
            rejected_images.append({
                "file": image_path.name,
                "reason": "image illisible"
            })
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]

        if img_size is None:
            img_size = current_size
        elif current_size != img_size:
            print(f"[WARN] Taille incohérente : {image_path.name} -> {current_size}, attendu {img_size}")
            rejected_images.append({
                "file": image_path.name,
                "reason": f"taille incohérente {current_size} vs {img_size}"
            })
            continue

        found, corners = detect_checkerboard(gray, PATTERN_SIZE)

        if not found:
            print(f"Damier non détecté : {image_path.name}")
            rejected_images.append({
                "file": image_path.name,
                "reason": "damier non détecté"
            })
            continue

        object_points.append(objp.copy())
        image_points.append(corners)
        used_images.append(image_path.name)
        if SAVE_DETECTED_IMAGES:
            save_detected_checkerboard_image(
                image=image,
                image_path=image_path,
                pattern_size=PATTERN_SIZE,
                corners=corners,
                found=found
            )

        if SAVE_RAW_DETECTED_IMAGES:
            save_raw_detected_image(image, image_path)
        if SHOW_PREVIEW:
            vis = image.copy()
            cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, found)
            preview = resize_for_preview(vis, PREVIEW_WIDTH, PREVIEW_HEIGHT)
            cv2.imshow("Detection damier", preview)
            cv2.waitKey(PREVIEW_DELAY_MS)

    cv2.destroyAllWindows()

    valid_count = len(object_points)

    print("\n=== BILAN DETECTION ===")
    print("Images valides :", valid_count)
    print("Images rejetées :", len(rejected_images))

    if valid_count < MIN_VALID_IMAGES:
        raise RuntimeError(
            f"Pas assez d'images valides pour calibrer correctement : {valid_count} "
            f"(minimum recommandé : {MIN_VALID_IMAGES})"
        )

    if img_size is None:
        raise RuntimeError("Aucune taille d'image valide détectée.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        img_size,
        None,
        None
    )

    per_image_errors, mean_reprojection_error = compute_reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )

    print("\n=== RESULTATS CAMERA ===")
    print("Nombre d'images valides :", valid_count)
    print("Erreur RMS :", rms)
    print("Camera matrix :\n", camera_matrix)
    print("Dist coeffs :\n", dist_coeffs.ravel())
    print("Erreur moyenne de reprojection (px) :", mean_reprojection_error)

    # Sauvegarde binaire
    np.save(CAMERA_MATRIX_NPY_PATH, camera_matrix)
    np.save(CAMERA_DIST_NPY_PATH, dist_coeffs)

    # Sauvegarde principale
    camera_data = {
        "pattern_size": [PATTERN_SIZE[0], PATTERN_SIZE[1]],
        "square_size_m": SQUARE_SIZE_M,
        "image_size_px": [int(img_size[0]), int(img_size[1])],
        "num_input_images": len(images),
        "num_valid_images": valid_count,
        "rms_error": float(rms),
        "mean_reprojection_error_px": float(mean_reprojection_error),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist()
    }

    with open(CAMERA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(camera_data, f, indent=4)

    # Sauvegarde debug détaillée
    debug_data = {
        "used_images": used_images,
        "rejected_images": rejected_images,
        "per_image_reprojection_error_px": [
            {"file": used_images[i], "error_px": per_image_errors[i]}
            for i in range(len(used_images))
        ]
    }

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=4)

    print("\n=== FICHIERS SAUVEGARDES ===")
    print(f"- {CAMERA_JSON_PATH}")
    print(f"- {CAMERA_MATRIX_NPY_PATH}")
    print(f"- {CAMERA_DIST_NPY_PATH}")
    print(f"- {DEBUG_JSON_PATH}")


if __name__ == "__main__":
    main()
