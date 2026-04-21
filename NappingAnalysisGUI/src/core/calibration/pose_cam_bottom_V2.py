# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# =========================================================
# PARAMÈTRES
# =========================================================
CAMERA_ID = 0

TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# Taille réelle de la table mesurée bord extérieur à bord extérieur (en mm)
TABLE_SIZE_MM = 580.0

# Taille physique d'un marker ArUco (côté du carré imprimé, en mm)
# IMPORTANT : H mappe les CENTRES des tags.
# Les centres sont décalés de ½ TAG_SIZE_MM par rapport aux bords extérieurs.
# Mesure TAG_SIZE_MM avec un pied à coulisse pour une précision maximale.
TAG_SIZE_MM = 40.0   # ← ajuste selon tes markers imprimés

# Distance réelle centre-à-centre entre les tags opposés
_CENTER_TO_CENTER = TABLE_SIZE_MM - TAG_SIZE_MM

# Nombre de frames à accumuler pour stabiliser la détection avant capture
N_FRAMES_AVERAGE = 30

# Critères de sous-pixel pour raffinement des coins ArUco
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SUBPIX_WIN = (5, 5)
SUBPIX_ZERO = (-1, -1)

# =========================================================
# CHEMINS PROJET
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

OUTPUT_JSON = CONFIG_DIR / "homography_cam_bottom_to_table.json"

# =========================================================
# UTILS — CENTRES DE MARKERS
# =========================================================

def refine_corners(gray: np.ndarray, corners: list) -> list:
    """
    Affine les coins de chaque marker au niveau sous-pixel.
    Retourne une liste de corners avec la même structure qu'en entrée.
    """
    refined = []
    for c in corners:
        pts = c[0].copy()  # shape (4, 2)
        refined_pts = cv2.cornerSubPix(gray, pts, SUBPIX_WIN, SUBPIX_ZERO, SUBPIX_CRITERIA)
        refined.append(refined_pts.reshape(1, 4, 2))
    return refined


def get_marker_centers(corners: list, ids: np.ndarray) -> dict:
    """
    Calcule le centre de chaque marker comme moyenne de ses 4 coins.
    (Plus robuste que l'intersection des diagonales en présence de bruit.)
    """
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]          # shape (4, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        centers[int(marker_id)] = (cx, cy)
    return centers


# =========================================================
# ACCUMULATION TEMPORELLE
# =========================================================

def accumulate_centers(
    accumulator: dict,          # {tag_id: [(cx, cy), ...]}
    centers: dict,
    required_ids: set,
    max_frames: int
) -> dict:
    """
    Ajoute les centres détectés dans l'accumulateur.
    Garde uniquement les max_frames dernières mesures.
    Retourne les centres moyennés si tous les tags requis sont présents.
    """
    if not required_ids.issubset(centers.keys()):
        return {}                  # Pas tous les tags visibles → on ignore cette frame

    for tag_id, center in centers.items():
        accumulator[tag_id].append(center)
        if len(accumulator[tag_id]) > max_frames:
            accumulator[tag_id].pop(0)

    # On ne renvoie les moyennes que si on a assez de mesures pour les 4 coins
    if all(len(accumulator[tid]) >= 5 for tid in required_ids):
        averaged = {
            tid: (
                float(np.mean([p[0] for p in accumulator[tid]])),
                float(np.mean([p[1] for p in accumulator[tid]])),
            )
            for tid in accumulator
        }
        return averaged

    return {}


# =========================================================
# HOMOGRAPHIE
# =========================================================

def build_homography(centers: dict):
    """
    Calcule H : pixel → mm à partir des 4 centres de tags.

    Le référentiel mm est ancré sur les BORDS EXTÉRIEURS de la table :
        (0, 0)              = coin ext. TL
        (TABLE_SIZE_MM, 0)  = coin ext. TR
        etc.

    Les centres des tags sont décalés de ½ TAG_SIZE_MM vers l'intérieur
    par rapport aux coins extérieurs.
    """
    half = TAG_SIZE_MM / 2.0
    S    = TABLE_SIZE_MM

    pts_img = np.array([
        centers[TAG_IDS["TL"]],
        centers[TAG_IDS["TR"]],
        centers[TAG_IDS["BR"]],
        centers[TAG_IDS["BL"]],
    ], dtype=np.float64)

    pts_mm = np.array([
        [half,       half      ],   # TL : décalé de ½ tag vers l'intérieur
        [S - half,   half      ],   # TR
        [S - half,   S - half  ],   # BR
        [half,       S - half  ],   # BL
    ], dtype=np.float64)

    # Avec exactement 4 points, RANSAC n'apporte rien → méthode exacte (0)
    H, _ = cv2.findHomography(pts_img, pts_mm, method=0)

    if H is None:
        raise RuntimeError("Échec du calcul de l'homographie.")

    return H, pts_img, pts_mm


def pixel_to_mm(pt, H: np.ndarray):
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    p_mm = H @ p
    p_mm /= p_mm[2]
    return float(p_mm[0]), float(p_mm[1])


# =========================================================
# VALIDATION
# =========================================================

def validate_homography(H: np.ndarray, centers: dict, verbose: bool = True):
    """
    Re-projette les 4 coins et un tag de test (ID 7 si présent).
    Affiche les erreurs et retourne l'erreur RMS sur les 4 coins.
    """
    print("\n=== VALIDATION COINS ===")
    half = TAG_SIZE_MM / 2.0
    S    = TABLE_SIZE_MM
    expected = {
        TAG_IDS["TL"]: (half,     half    ),
        TAG_IDS["TR"]: (S - half, half    ),
        TAG_IDS["BR"]: (S - half, S - half),
        TAG_IDS["BL"]: (half,     S - half),
    }
    errors = []
    for name, tag_id in TAG_IDS.items():
        cx, cy = centers[tag_id]
        x_mm, y_mm = pixel_to_mm((cx, cy), H)
        ex, ey = expected[tag_id]
        err = np.hypot(x_mm - ex, y_mm - ey)
        errors.append(err)
        if verbose:
            print(f"  {name} (id={tag_id}) → mesuré ({x_mm:.2f}, {y_mm:.2f}) mm  "
                  f"| attendu ({ex:.0f}, {ey:.0f}) mm  | erreur {err:.2f} mm")

    rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    print(f"\n  RMS erreur coins : {rms:.2f} mm")

    # Tag de test optionnel (id=7)
    TEST_TAG_ID = 7
    EXPECTED_MM  = (200.0, 200.0)
    if TEST_TAG_ID in centers:
        cx, cy = centers[TEST_TAG_ID]
        x_mm, y_mm = pixel_to_mm((cx, cy), H)
        err_x = x_mm - EXPECTED_MM[0]
        err_y = y_mm - EXPECTED_MM[1]
        err_total = np.hypot(err_x, err_y)
        print(f"\n=== TEST TAG ID={TEST_TAG_ID} ===")
        print(f"  Mesuré  : ({x_mm:.2f}, {y_mm:.2f}) mm")
        print(f"  Attendu : ({EXPECTED_MM[0]:.1f}, {EXPECTED_MM[1]:.1f}) mm")
        print(f"  Erreur X : {err_x:+.2f} mm  |  Erreur Y : {err_y:+.2f} mm  |  Totale : {err_total:.2f} mm")
    else:
        print(f"\n  [INFO] Tag ID={TEST_TAG_ID} non visible, test ignoré.")

    return rms


# =========================================================
# SAUVEGARDE
# =========================================================

def save_json(H: np.ndarray, pts_img: np.ndarray, pts_mm: np.ndarray):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "H_pixel_to_mm": H.tolist(),
        "points_image_px": pts_img.tolist(),
        "points_table_mm": pts_mm.tolist(),
        "table_size_mm": TABLE_SIZE_MM,
        "tag_ids": TAG_IDS,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"\n[OK] Homographie sauvegardée → {OUTPUT_JSON}")


# =========================================================
# AFFICHAGE HUD
# =========================================================

def draw_hud(display: np.ndarray, centers: dict, accumulator: dict,
             required_ids: set, H=None):
    """Dessine les centres détectés, le statut et la grille si H est disponible."""

    # Centres des 4 tags principaux
    all_present = required_ids.issubset(centers.keys())
    for name, tag_id in TAG_IDS.items():
        if tag_id in centers:
            cx, cy = int(centers[tag_id][0]), int(centers[tag_id][1])
            color = (0, 255, 0) if all_present else (0, 165, 255)
            cv2.circle(display, (cx, cy), 7, color, -1)
            cv2.putText(display, name, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Tag de test (id=7)
    TEST_TAG_ID = 7
    if TEST_TAG_ID in centers:
        cx, cy = int(centers[TEST_TAG_ID][0]), int(centers[TEST_TAG_ID][1])
        cv2.circle(display, (cx, cy), 9, (255, 80, 0), -1)
        cv2.putText(display, "TEST-7", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

    # Nombre de frames accumulées
    n_acc = min(len(accumulator.get(TAG_IDS["TL"], [])), N_FRAMES_AVERAGE)
    bar_w = int(300 * n_acc / N_FRAMES_AVERAGE)
    cv2.rectangle(display, (20, 60), (320, 85), (60, 60, 60), -1)
    cv2.rectangle(display, (20, 60), (20 + bar_w, 85), (0, 220, 100), -1)
    cv2.putText(display, f"Stabilisation : {n_acc}/{N_FRAMES_AVERAGE}",
                (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Statut général
    if all_present:
        msg = "4 TAGS OK — Appuie sur 'c' pour capturer"
        col = (0, 255, 0)
    else:
        missing = [k for k, v in TAG_IDS.items() if v not in centers]
        msg = f"Tags manquants : {', '.join(missing)}"
        col = (0, 50, 255)
    cv2.putText(display, msg, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Grille repère si H disponible
    if H is not None and all_present:
        step = TABLE_SIZE_MM / 4
        for i in range(5):
            # Lignes horizontales (y = i*step)
            pts_row = np.array([[j * TABLE_SIZE_MM / 20, i * step] for j in range(21)],
                               dtype=np.float64)
            _draw_mm_polyline(display, pts_row, H, (180, 180, 50))
            # Lignes verticales
            pts_col = np.array([[i * step, j * TABLE_SIZE_MM / 20] for j in range(21)],
                               dtype=np.float64)
            _draw_mm_polyline(display, pts_col, H, (180, 180, 50))


def _draw_mm_polyline(img, pts_mm, H, color):
    """Projette une polyligne mm → pixel et la dessine."""
    H_inv = np.linalg.inv(H)
    pts_px = []
    for pt in pts_mm:
        p = np.array([pt[0], pt[1], 1.0])
        px = H_inv @ p
        px /= px[2]
        pts_px.append((int(px[0]), int(px[1])))
    for i in range(len(pts_px) - 1):
        cv2.line(img, pts_px[i], pts_px[i + 1], color, 1, cv2.LINE_AA)


# =========================================================
# MAIN
# =========================================================

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # Désactive l'autofocus si supporté

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la caméra.")
        return
    print(f"[INFO] Résolution : {frame.shape[1]}×{frame.shape[0]}")

    # Détecteur ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    required_ids = set(TAG_IDS.values())
    accumulator  = defaultdict(list)   # {tag_id: [(cx,cy), ...]}
    H_current    = None                # Dernière homographie calculée

    print("[INFO] 'c' = capturer  |  'r' = réinitialiser accumulateur  |  'q' = quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display  = frame.copy()
        centers  = {}

        if ids is not None:
            # Raffinement sous-pixel
            corners = refine_corners(gray, corners)
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            centers = get_marker_centers(corners, ids)

            # Accumulation temporelle
            averaged = accumulate_centers(accumulator, centers, required_ids, N_FRAMES_AVERAGE)
        else:
            averaged = {}

        draw_hud(display, centers, accumulator, required_ids, H_current)
        cv2.imshow("Homographie — caméra bas", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            accumulator.clear()
            H_current = None
            print("[INFO] Accumulateur réinitialisé.")

        elif key == ord('c'):
            # On utilise les centres moyennés si dispo, sinon les centres courants
            use_centers = averaged if averaged else centers

            if not required_ids.issubset(use_centers.keys()):
                print("[ERREUR] Les 4 tags ne sont pas visibles (ou pas assez de frames accumulées).")
                continue

            try:
                H, pts_img, pts_mm = build_homography(use_centers)
            except RuntimeError as e:
                print(f"[ERREUR] {e}")
                continue

            print("\n=== MATRICE H (pixel → mm) ===")
            print(np.round(H, 6))

            rms = validate_homography(H, use_centers)

            if rms > 5.0:
                print(f"\n[AVERTISSEMENT] RMS={rms:.1f} mm — vérifiez le placement des tags.")
            else:
                print(f"\n[OK] Précision acceptable (RMS={rms:.2f} mm).")

            save_json(H, pts_img, pts_mm)
            H_current = H

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()