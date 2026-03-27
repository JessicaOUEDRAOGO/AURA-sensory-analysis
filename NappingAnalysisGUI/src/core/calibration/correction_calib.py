# -*- coding: utf-8 -*-
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from screeninfo import get_monitors

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.mapping.coordinate_mapper import CoordinateMapper


# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_SCREEN_ID = 1
CAMERA_ID = 0

WINDOW_NAME_PROJECTOR = "ProjectorCalibration9Points"
WINDOW_NAME_CAMERA = "CameraCalibration9Points"

GRAPH_SIZE = 700
MARGIN_GRAPH = 90

SQUARE_SIZE_GRAPH = 70
CENTER_RADIUS_GRAPH = 8

ARUCO_DICT = cv2.aruco.DICT_4X4_50
EXPECTED_IDS = list(range(9))
EXCLUDED_TAG_IDS = {40, 41, 42, 43}

OUTPUT_PATH = ROOT / "config" / "projector_correction.json"

# nombre de frames à moyenner après appui sur S
AVERAGE_FRAMES = 30


# =========================================================
# AFFICHAGE PROJECTEUR
# =========================================================
class ProjectorWindow:
    def __init__(self, screen_id: int):
        monitors = get_monitors()
        if screen_id < 0 or screen_id >= len(monitors):
            raise ValueError(f"screen_id invalide : {screen_id}")

        self.monitor = monitors[screen_id]
        self.window_name = WINDOW_NAME_PROJECTOR

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    def show(self, image: np.ndarray):
        cv2.imshow(self.window_name, image)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


# =========================================================
# OUTILS HOMOGRAPHIE
# =========================================================
def apply_homography(H, pt):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("Coordonnée homogène nulle")
    q /= q[2]
    return q[:2].astype(np.float32)


def affine_2x3_to_homography(A):
    if A is None:
        raise ValueError("Affine nulle")
    if A.shape != (2, 3):
        raise ValueError(f"Affine attendue en 2x3, reçu {A.shape}")

    H = np.eye(3, dtype=np.float64)
    H[0:2, :] = A
    return H


def evaluate_mapping(H, src_pts, dst_pts):
    pred_pts = np.array([apply_homography(H, p) for p in src_pts], dtype=np.float64)
    errors = np.linalg.norm(pred_pts - dst_pts, axis=1)
    return pred_pts, errors


def evaluate_nominal(src_pts, dst_pts):
    errors = np.linalg.norm(src_pts - dst_pts, axis=1)
    return errors


# =========================================================
# CIBLES DANS LE REPERE GRAPH
# =========================================================
def build_graph_points():
    xs = np.linspace(MARGIN_GRAPH, GRAPH_SIZE - MARGIN_GRAPH, 3)
    ys = np.linspace(MARGIN_GRAPH, GRAPH_SIZE - MARGIN_GRAPH, 3)

    pts = []
    for y in ys:
        pts.append([xs[2], y])  # droite
        pts.append([xs[1], y])  # centre
        pts.append([xs[0], y])  # gauche

    return np.array(pts, dtype=np.float32)


def draw_target_square_graph(img, center_xy, label):
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = SQUARE_SIZE_GRAPH // 2

    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    cv2.circle(img, (cx, cy), CENTER_RADIUS_GRAPH, (0, 0, 255), -1)

    cv2.putText(
        img,
        str(label),
        (cx - 14, cy - half - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def build_graph_pattern(graph_pts):
    img = np.full((GRAPH_SIZE, GRAPH_SIZE, 3), 255, dtype=np.uint8)

    for idx, pt in enumerate(graph_pts):
        draw_target_square_graph(img, pt, idx)

    cv2.putText(
        img,
        f"Place tag IDs 0..8 on targets, then press S ({AVERAGE_FRAMES} frames avg)",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    return img


# =========================================================
# DETECTION ARUCO
# =========================================================
def build_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def detect_markers(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def filter_useful_markers(corners, ids):
    if ids is None or len(ids) == 0:
        return [], None

    filtered_corners = []
    filtered_ids = []

    for i, marker_id_arr in enumerate(ids):
        marker_id = int(marker_id_arr[0])

        if marker_id in EXCLUDED_TAG_IDS:
            continue

        if marker_id not in EXPECTED_IDS:
            continue

        filtered_corners.append(corners[i])
        filtered_ids.append([marker_id])

    if len(filtered_ids) == 0:
        return [], None

    return filtered_corners, np.array(filtered_ids, dtype=np.int32)


def draw_camera_preview(frame, corners, ids):
    preview = frame.copy()

    if ids is not None and len(ids) > 0:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            marker_id = int(ids[i][0])

            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])
                cv2.line(preview, pt1, pt2, (0, 255, 0), 2)

            center = tuple(np.mean(pts, axis=0).astype(int))
            cv2.circle(preview, center, 6, (0, 0, 255), -1)

            cv2.putText(
                preview,
                str(marker_id),
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

    return preview


# =========================================================
# CENTRES PROJECTEUR CALCULES ACTUELS
# =========================================================
def compute_projector_centers(mapper: CoordinateMapper, corners, ids):
    centers = {}

    if ids is None or len(ids) == 0:
        return centers

    for i, corner in enumerate(corners):
        marker_id = int(ids[i][0])

        projector_corners = np.array(
            [mapper.camera_raw_to_projector(corner[0][j]) for j in range(4)],
            dtype=np.float32
        )
        projector_center = np.mean(projector_corners, axis=0)
        centers[marker_id] = projector_center.astype(np.float32)

    return centers


def average_projector_centers_over_frames(cap, detector, mapper, projector, projector_pattern,
                                          num_frames=AVERAGE_FRAMES):
    accum = {i: [] for i in EXPECTED_IDS}
    kept_frames = 0
    attempts = 0
    max_attempts = max(num_frames * 4, 60)

    while kept_frames < num_frames and attempts < max_attempts:
        attempts += 1

        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue

        corners_all, ids_all = detect_markers(detector, frame)
        useful_corners, useful_ids = filter_useful_markers(corners_all, ids_all)
        centers_dict = compute_projector_centers(mapper, useful_corners, useful_ids)

        projector.show(projector_pattern)
        preview = draw_camera_preview(frame, useful_corners, useful_ids)

        status_text = f"Sampling {kept_frames + 1}/{num_frames}"
        cv2.putText(
            preview,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow(WINDOW_NAME_CAMERA, cv2.resize(preview, (1080, 720)))
        cv2.waitKey(1)

        if all(i in centers_dict for i in EXPECTED_IDS):
            for i in EXPECTED_IDS:
                accum[i].append(centers_dict[i])
            kept_frames += 1

    missing_counts = {i: len(accum[i]) for i in EXPECTED_IDS}
    if not all(len(accum[i]) > 0 for i in EXPECTED_IDS):
        raise RuntimeError(
            f"Impossible de moyenner tous les tags. Comptes par ID: {missing_counts}"
        )

    mean_pts = np.array(
        [np.mean(np.array(accum[i], dtype=np.float32), axis=0) for i in EXPECTED_IDS],
        dtype=np.float32
    )

    stds = np.array(
        [np.std(np.array(accum[i], dtype=np.float32), axis=0) for i in EXPECTED_IDS],
        dtype=np.float32
    )

    return mean_pts, stds, kept_frames, attempts


# =========================================================
# SAUVEGARDE
# =========================================================
def save_projector_correction(H_corr, src_pts, dst_pts, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "H_proj_correction": np.array(H_corr, dtype=np.float64).tolist(),
        "src_pts": np.array(src_pts, dtype=np.float32).tolist(),
        "dst_pts": np.array(dst_pts, dtype=np.float32).tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Correction sauvegardée dans : {output_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== CALIBRATION PROJECTOR CORRECTION ===")
    print("Instructions :")
    print("1. Pose les tags IDs 0..8 sur les carrés projetés correspondants.")
    print("2. Les tags 40, 41, 42, 43 sont exclus proprement.")
    print(f"3. Quand les 9 tags sont visibles, appuie sur S (moyenne sur {AVERAGE_FRAMES} frames).")
    print("4. Q ou Echap pour quitter.")

    mapper = CoordinateMapper()

    try:
        mapper.load(load_projector_correction=False)
    except TypeError:
        correction_file = OUTPUT_PATH
        temp_backup = None

        if correction_file.exists():
            temp_backup = correction_file.with_name(
                correction_file.stem + "_backup_before_recalib.json"
            )
            correction_file.rename(temp_backup)

        try:
            mapper.load()
        finally:
            if temp_backup is not None and temp_backup.exists() and not correction_file.exists():
                temp_backup.rename(correction_file)

    proj_w = int(mapper.projector_width)
    proj_h = int(mapper.projector_height)

    H_graph_to_proj = mapper.get_graph_to_projector_homography()

    graph_pts = build_graph_points()
    dst_pts = np.array(
        [apply_homography(H_graph_to_proj, pt) for pt in graph_pts],
        dtype=np.float32
    )

    graph_pattern = build_graph_pattern(graph_pts)
    projector_pattern = cv2.warpPerspective(
        graph_pattern,
        H_graph_to_proj,
        (proj_w, proj_h)
    )

    detector = build_aruco_detector()
    projector = ProjectorWindow(PROJECTOR_SCREEN_ID)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("[WARNING] Frame caméra invalide.")
                continue

            corners_all, ids_all = detect_markers(detector, frame)
            useful_corners, useful_ids = filter_useful_markers(corners_all, ids_all)

            if useful_ids is not None and len(useful_ids) > 0:
                print("IDs utiles détectés :", [int(x[0]) for x in useful_ids])
            else:
                print("Aucun ID utile détecté")

            preview = draw_camera_preview(frame, useful_corners, useful_ids)

            projector.show(projector_pattern)
            cv2.imshow(WINDOW_NAME_CAMERA, cv2.resize(preview, (1080, 720)))

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("s"):
                try:
                    src_pts, stds, kept_frames, attempts = average_projector_centers_over_frames(
                        cap=cap,
                        detector=detector,
                        mapper=mapper,
                        projector=projector,
                        projector_pattern=projector_pattern,
                        num_frames=AVERAGE_FRAMES
                    )
                except RuntimeError as e:
                    print(f"[ERREUR] {e}")
                    continue

                print("\n===== STABILITE DES CENTRES =====")
                print(f"Frames valides gardées : {kept_frames} / tentatives : {attempts}")
                for idx, marker_id in enumerate(EXPECTED_IDS):
                    sx, sy = stds[idx]
                    print(f"ID {marker_id} -> std_x = {float(sx):.3f} px ; std_y = {float(sy):.3f} px")
                print("=================================\n")

                nominal_errors = evaluate_nominal(src_pts, dst_pts)

                # Méthode 1 : affine complète
                A_full, _ = cv2.estimateAffine2D(
                    src_pts, dst_pts,
                    method=cv2.LMEDS,
                    refineIters=10
                )

                # Méthode 2 : affine partielle
                A_partial, _ = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.LMEDS,
                    refineIters=10
                )

                candidates = []

                if A_full is not None:
                    H_full = affine_2x3_to_homography(A_full)
                    _, errors_full = evaluate_mapping(H_full, src_pts, dst_pts)
                    candidates.append(("affine_complete", H_full, errors_full))

                if A_partial is not None:
                    H_partial = affine_2x3_to_homography(A_partial)
                    _, errors_partial = evaluate_mapping(H_partial, src_pts, dst_pts)
                    candidates.append(("affine_partielle", H_partial, errors_partial))

                if not candidates:
                    print("[ERREUR] Impossible de calculer une correction affine.")
                    continue

                best_name, H_corr, errors = min(
                    candidates,
                    key=lambda item: float(np.mean(item[2]))
                )

                print("\n===== RESULTATS CANDIDATS =====")
                print(
                    f"nominal -> erreur moyenne = {float(np.mean(nominal_errors)):.3f} px ; "
                    f"erreur max = {float(np.max(nominal_errors)):.3f} px"
                )

                for name, _, err in candidates:
                    print(
                        f"{name} -> erreur moyenne = {float(np.mean(err)):.3f} px ; "
                        f"erreur max = {float(np.max(err)):.3f} px"
                    )

                print("Méthode retenue :", best_name)
                print("================================\n")

                print("===== ERREURS PAR ID =====")
                for idx, marker_id in enumerate(EXPECTED_IDS):
                    print(
                        f"ID {marker_id} -> nominal = {float(nominal_errors[idx]):.3f} px ; "
                        f"corrigé = {float(errors[idx]):.3f} px"
                    )
                print("==========================\n")

                print("[OK] H_proj_correction calculée :")
                print(H_corr)

                save_projector_correction(H_corr, src_pts, dst_pts, OUTPUT_PATH)
                print("[OK] Calibration terminée.")
                break

    finally:
        cap.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()