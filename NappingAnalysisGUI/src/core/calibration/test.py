# -*- coding: utf-8 -*-
"""
test_height_correction.py
=========================
Mesure l'erreur de projection due à la hauteur de la tasse.
Place le tag 7 AU SOL, note les coordonnées.
Pose la tasse dessus, note la différence.
Calcule la correction nécessaire.
"""

import cv2
import numpy as np
import json
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR   = PROJECT_ROOT / "config"

POSE_TOP_PATH  = CONFIG_DIR / "camtop_table_pose.json"
CALIB_TOP_PATH = CONFIG_DIR / "camera_calibration_top.json"

CAM_TOP_ID    = 1
TABLE_SIZE_MM = 597.0
TAG_ID        = 7


def load_pose(path):
    d = json.load(open(path))
    return (np.array(d["rvec"], dtype=np.float64),
            np.array(d["tvec"], dtype=np.float64),
            np.array(d["camera_matrix"], dtype=np.float64))

def load_calib(path):
    d = json.load(open(path))
    return (np.array(d["camera_matrix"], dtype=np.float64),
            np.array(d["dist_coeffs"],   dtype=np.float64))

def pixel_to_mm_at_z(u, v, rvec, tvec, K, z_table=0.0):
    """
    Conversion pixel → mm en supposant que l'objet est à Z=z_table
    dans le repère table (0 = plan table, positif = au dessus).
    """
    R, _  = cv2.Rodrigues(rvec)
    K_inv = np.linalg.inv(K)
    ray   = K_inv @ np.array([u, v, 1.0])
    ray  /= np.linalg.norm(ray)

    # Plan à Z=z_table dans le repère table
    # → normal = axe Z table = R[:,2] dans repère caméra
    # → point du plan = tvec + R @ [0,0,z_table]
    normal      = R[:, 2]
    plane_point = tvec.reshape(3) + R @ np.array([0, 0, z_table])

    denom = np.dot(normal, ray)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(normal, plane_point) / denom
    if t < 0:
        return None

    pt_cam   = ray * t
    pt_table = R.T @ (pt_cam - tvec.reshape(3))
    return float(pt_table[0]), TABLE_SIZE_MM - float(pt_table[1])


def main():
    rvec, tvec, new_K = load_pose(POSE_TOP_PATH)
    K, dist           = load_calib(CALIB_TOP_PATH)

    cap = cv2.VideoCapture(CAM_TOP_ID, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    new_K_full, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (1920, 1080), 1, (1920, 1080))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K_full, (1920, 1080), cv2.CV_16SC2)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict)

    # Mesures accumulées
    pos_sol   = []   # tag posé sur la table
    pos_tasse = []   # tag posé sur la tasse

    mode = "sol"     # "sol" ou "tasse"
    cup_height = 0.0

    print("\n=== TEST CORRECTION HAUTEUR ===")
    print("1. Place le tag 7 À PLAT sur la table (sans tasse)")
    print("   Appuie sur S pour enregistrer la position sol")
    print("2. Pose la tasse sur le tag, le tag doit rester visible")
    print("   Entre la hauteur de la tasse en mm quand demandé")
    print("   Appuie sur T pour enregistrer la position tasse")
    print("q = quitter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_u = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        gray    = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        display = cv2.resize(frame_u, (960, 540))
        sx, sy  = 960/1920, 540/1080

        pos_z0 = None
        cx_raw = cy_raw = None

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid != TAG_ID:
                    continue
                pts   = corners[i][0]
                cx_raw = float(np.mean(pts[:, 0]))
                cy_raw = float(np.mean(pts[:, 1]))

                # Position en supposant Z=0 (plan table)
                pos_z0 = pixel_to_mm_at_z(
                    cx_raw, cy_raw, rvec, tvec, new_K)

                # Affichage
                pcx = int(cx_raw * sx)
                pcy = int(cy_raw * sy)
                cv2.circle(display, (pcx, pcy), 8, (0, 0, 255), -1)

                if pos_z0:
                    cv2.putText(display,
                        f"Z=0: ({pos_z0[0]:.1f}, {pos_z0[1]:.1f})mm",
                        (pcx+10, pcy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Si on a aussi la hauteur, afficher la correction
                    if cup_height > 0:
                        pos_zh = pixel_to_mm_at_z(
                            cx_raw, cy_raw, rvec, tvec, new_K, cup_height)
                        if pos_zh:
                            cv2.putText(display,
                                f"Z={cup_height:.0f}: ({pos_zh[0]:.1f}, {pos_zh[1]:.1f})mm",
                                (pcx+10, pcy+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Stats
        status_lines = [
            f"Mode: {mode.upper()}  |  tag7: {'VU' if pos_z0 else 'NON VU'}",
            f"Mesures sol: {len(pos_sol)}  |  tasse: {len(pos_tasse)}",
        ]
        if pos_sol and pos_tasse:
            mean_sol   = np.mean(pos_sol,   axis=0)
            mean_tasse = np.mean(pos_tasse, axis=0)
            dx = mean_tasse[0] - mean_sol[0]
            dy = mean_tasse[1] - mean_sol[1]
            dist_err = (dx**2 + dy**2)**0.5
            status_lines.append(
                f"ERREUR sol→tasse: Δx={dx:.1f} Δy={dy:.1f} dist={dist_err:.1f}mm")
            status_lines.append(
                f"Hauteur tasse = {cup_height:.0f}mm")

        for i, txt in enumerate(status_lines):
            cv2.putText(display, txt, (10, 20+20*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

        cv2.putText(display, "S=sol  T=tasse  H=hauteur  Q=quitter",
                    (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.imshow("Test hauteur", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            if pos_z0:
                pos_sol.append(pos_z0)
                print(f"[Sol]   ({pos_z0[0]:.1f}, {pos_z0[1]:.1f})mm  "
                      f"(n={len(pos_sol)})")
            else:
                print("[Sol] Tag non visible")

        elif key == ord('t'):
            if pos_z0:
                pos_tasse.append(pos_z0)
                print(f"[Tasse] ({pos_z0[0]:.1f}, {pos_z0[1]:.1f})mm  "
                      f"(n={len(pos_tasse)})")
            else:
                print("[Tasse] Tag non visible")

        elif key == ord('h'):
            cap.release()
            cv2.destroyAllWindows()
            h = input("Hauteur de la tasse en mm (bord supérieur depuis table) : ")
            try:
                cup_height = float(h)
                print(f"Hauteur = {cup_height}mm")
            except ValueError:
                print("Valeur invalide")
            cap = cv2.VideoCapture(CAM_TOP_ID, cv2.CAP_MSMF)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # ── Résultats finaux ──────────────────────────────────────────────
    if pos_sol and pos_tasse:
        mean_sol   = np.mean(pos_sol,   axis=0)
        mean_tasse = np.mean(pos_tasse, axis=0)
        dx = mean_tasse[0] - mean_sol[0]
        dy = mean_tasse[1] - mean_sol[1]
        dist_err = (dx**2 + dy**2)**0.5

        print(f"\n{'='*50}")
        print(f"Position sol   : ({mean_sol[0]:.2f}, {mean_sol[1]:.2f})mm")
        print(f"Position tasse : ({mean_tasse[0]:.2f}, {mean_tasse[1]:.2f})mm")
        print(f"Erreur due à la hauteur :")
        print(f"  Δx = {dx:+.2f}mm")
        print(f"  Δy = {dy:+.2f}mm")
        print(f"  dist = {dist_err:.2f}mm")
        if cup_height > 0:
            print(f"  Hauteur tasse = {cup_height:.0f}mm")
            print(f"  Erreur/mm de hauteur : {dist_err/cup_height:.3f}mm/mm")
        print(f"{'='*50}")

        # Sauvegarder pour usage dans cam_top_thread
        result = {
            "cup_height_mm":      cup_height,
            "error_at_height_mm": dist_err,
            "delta_x_mm":         float(dx),
            "delta_y_mm":         float(dy),
            "correction_factor":  float(dist_err/cup_height) if cup_height > 0 else 0,
        }
        out = CONFIG_DIR / "cup_height_correction.json"
        json.dump(result, open(out, "w"), indent=2)
        print(f"\n[OK] Résultats sauvegardés → {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()