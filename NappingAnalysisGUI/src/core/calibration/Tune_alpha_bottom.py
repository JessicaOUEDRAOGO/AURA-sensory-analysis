# -*- coding: utf-8 -*-
"""
Outil de diagnostic : ajuste alpha en temps réel via trackbar
pour trouver la valeur qui garde les 4 tags de coin visibles
après undistortion.

Caméra : bottom (vue de dessous)
Tags   : TL=42, TR=43, BR=40, BL=41  (mêmes que top)
"""

import cv2
import numpy as np
import json
from pathlib import Path

# =========================================================
# PARAMÈTRES — adapte ces chemins
# =========================================================
CAMERA_ID = 0  # change si besoin (0 = top, 1 = bottom probablement)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CALIB_PATH  = CONFIG_DIR / "camera_calibration_top.json"  # même calibration réutilisée

TAG_IDS = {
    "TL": 42,
    "TR": 43,
    "BR": 40,
    "BL": 41,
}

# =========================================================
# CALIBRATION
# =========================================================
def load_calibration(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Calibration introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K    = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"],   dtype=np.float64)
    return K, dist


# =========================================================
# MAIN
# =========================================================
def main():
    K, dist = load_calibration(CALIB_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra {CAMERA_ID}")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire une frame")
        cap.release()
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Résolution : {w}x{h}")
    print("[INFO] Trackbar ALPHA : 0 = max recadré | 100 = max champ (zones noires)")
    print("[INFO] Appuie sur 's' pour sauvegarder l'alpha courant | 'q' pour quitter")

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    # Fenêtre + trackbar
    win = "Tune alpha - bottom camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.createTrackbar("alpha x100", win, 0, 100, lambda x: None)

    best_alpha = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lire alpha depuis trackbar
        alpha = cv2.getTrackbarPos("alpha x100", win) / 100.0

        # Calculer new_K pour cet alpha
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))

        # Undistort
        frame_undist = cv2.undistort(frame, K, dist, None, new_K)

        # Détecter ArUco
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = frame_undist.copy()

        # Afficher les tags détectés
        detected_names = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            detected_ids = [int(i) for i in ids.flatten()]

            for name, tid in TAG_IDS.items():
                if tid in detected_ids:
                    detected_names.append(name)

        all_ok = len(detected_names) == 4

        # HUD
        color_alpha = (0, 255, 0) if all_ok else (0, 165, 255)
        cv2.putText(display, f"alpha = {alpha:.2f}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_alpha, 2)

        status = "4/4 TAGS OK !" if all_ok else f"Tags coin visibles : {detected_names} ({len(detected_names)}/4)"
        cv2.putText(display, status, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_alpha, 2)

        # Dessiner les bords du ROI valide (zone sans pixels noirs)
        x_roi, y_roi, w_roi, h_roi = roi
        cv2.rectangle(display, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi),
                      (255, 255, 0), 2)
        cv2.putText(display, f"ROI valide : {w_roi}x{h_roi} px",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Conseil
        conseil = (
            "Augmente alpha si des tags sont hors champ"
            if not all_ok else
            "Diminue alpha jusqu'a la limite ou les 4 tags restent visibles"
        )
        cv2.putText(display, conseil, (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            best_alpha = alpha
            print(f"\n[SAUVEGARDE] alpha optimal = {best_alpha:.2f}")
            print(f"  new_K diagonal : fx={new_K[0,0]:.1f}, fy={new_K[1,1]:.1f}, "
                  f"cx={new_K[0,2]:.1f}, cy={new_K[1,2]:.1f}")
            print(f"  ROI valide     : {roi}")
            print(f"\n  --> Utilise cette valeur dans ton script de pose :")
            print(f"      new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), {best_alpha:.2f}, (w, h))")

            # Sauvegarder aussi dans un JSON pour référence
            out = {
                "alpha":          best_alpha,
                "new_K":          new_K.tolist(),
                "roi":            list(roi),
                "dist_coeffs":    dist.tolist(),
                "camera_id":      CAMERA_ID,
                "resolution":     [w, h],
            }
            out_path = CONFIG_DIR / "alpha_bottom_tuned.json"
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=4)
            print(f"  --> Sauvegardé dans {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()