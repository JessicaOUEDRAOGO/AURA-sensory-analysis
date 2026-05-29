
# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager

CAM_BOTTOM_INDEX = 1
CAM_TOP_INDEX    = 0
TABLE_SIZE_MM    = 597.0
PROJ_W, PROJ_H   = 3840, 2160
PROJ_SCREEN_ID   = 1
TAG_ID           = 7

TEST_POINTS_MM = [
    (100, 100), (300, 100), (500, 100),
    (100, 300), (300, 300), (500, 300),
    (100, 500), (300, 500), (500, 500),
]


def pixel_to_mm(cx, cy, K, rvec, tvec):
    K_inv = np.linalg.inv(K)
    ray   = K_inv @ np.array([cx, cy, 1.0])
    ray  /= np.linalg.norm(ray)
    R, _  = cv2.Rodrigues(rvec)
    n, o  = R[:,2], tvec.reshape(3)
    d     = np.dot(n, ray)
    if abs(d) < 1e-9: return None
    t = np.dot(n, o) / d
    if t < 0: return None
    pt = R.T @ (ray*t - o)
    return float(pt[0]), float(pt[1])


def detect_tag(frame, detector, tag_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None: return None
    for i, mid in enumerate(ids.flatten()):
        if mid != tag_id: continue
        pts = corners[i][0]
        return float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))
    return None


def main():
    # ── Projecteur ───────────────────────────────────────────────────
    H = np.array(
        json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
        dtype=np.float32)

    dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
    dm.resolution = (PROJ_W, PROJ_H)
    proj = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

    for (x_mm, y_mm) in TEST_POINTS_MM:
        pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H)
        px, py = int(res[0,0,0]), int(res[0,0,1])
        if 0 < px < PROJ_W and 0 < py < PROJ_H:
            cv2.drawMarker(proj, (px,py), (200,0,0),
                           cv2.MARKER_CROSS, 150, 10)
            cv2.putText(proj, f"({x_mm},{y_mm})",
                        (px+40, py-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (200,0,0), 4)

    dm.display_image_on_projector_monitor(proj)
    print("[OK] Grille projetée")

    # ── Poses ────────────────────────────────────────────────────────
    def load_pose(name):
        d = json.load(open(config_path(name)))
        return (np.array(d["camera_matrix"], dtype=np.float64),
                np.array(d["rvec"],          dtype=np.float64),
                np.array(d["tvec"],          dtype=np.float64))

    K_bot, rvec_bot, tvec_bot = load_pose("cambottom_table_pose.json")
    K_top, rvec_top, tvec_top = load_pose("camtop_table_pose.json")

    # ── Undistort cam_bottom (fisheye) ───────────────────────────────
    d     = json.load(open(config_path("camera_calibration_fisheye.json")))
    K_f   = np.array(d["camera_matrix"], dtype=np.float64)
    dist_f= np.array(d["dist_coeffs"],   dtype=np.float64).reshape(4, 1)
    nK_f  = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_f, dist_f, (1920,1080), np.eye(3), balance=1.0)
    map1_bot, map2_bot = cv2.fisheye.initUndistortRectifyMap(
        K_f, dist_f, np.eye(3), nK_f, (1920,1080), cv2.CV_16SC2)
    print("[OK] Undistort fisheye bot")

    # ── Undistort cam_top (pinhole) ──────────────────────────────────
    d     = json.load(open(config_path("camera_calibration_top.json")))
    K_t   = np.array(d["camera_matrix"], dtype=np.float64)
    dist_t= np.array(d["dist_coeffs"],   dtype=np.float64)
    nK_t, _ = cv2.getOptimalNewCameraMatrix(
        K_t, dist_t, (1920,1080), 1, (1920,1080))
    map1_top, map2_top = cv2.initUndistortRectifyMap(
        K_t, dist_t, None, nK_t, (1920,1080), cv2.CV_16SC2)
    print("[OK] Undistort pinhole top")

    # ── Caméras — exactement comme test_cross_calibration.py ─────────
    cap_bot = cv2.VideoCapture(CAM_BOTTOM_INDEX, cv2.CAP_DSHOW)
    cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"[OK] cam_bottom ouverte={cap_bot.isOpened()}")

    cap_top = cv2.VideoCapture(CAM_TOP_INDEX, cv2.CAP_DSHOW)
    cap_top.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_top.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"[OK] cam_top ouverte={cap_top.isOpened()}")

    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    results = []
    S = TABLE_SIZE_MM

    print("\nPlace tag7 sur une croix, ESPACE=capturer, Q=quitter\n")

    while True:
        ret_b, frame_bot = cap_bot.read()
        ret_t, frame_top = cap_top.read()

        if not ret_b or frame_bot is None: continue
        if not ret_t or frame_top is None: continue

        frame_bot = cv2.remap(frame_bot, map1_bot, map2_bot, cv2.INTER_LINEAR)
        frame_top = cv2.remap(frame_top, map1_top, map2_top, cv2.INTER_LINEAR)

        tag_bot    = detect_tag(frame_bot, det, TAG_ID)
        tag_top    = detect_tag(frame_top, det, TAG_ID)
        mm_bot     = pixel_to_mm(*tag_bot, K_bot, rvec_bot, tvec_bot) if tag_bot else None
        mm_top_raw = pixel_to_mm(*tag_top, K_top, rvec_top, tvec_top) if tag_top else None

        prev_bot = cv2.resize(frame_bot, (640, 360))
        prev_top = cv2.resize(frame_top, (640, 360))

        cv2.putText(prev_bot,
                    f"BOT: ({mm_bot[0]:.0f},{mm_bot[1]:.0f})mm" if mm_bot else "BOT: non vu",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if mm_bot else (0,0,255), 2)

        if mm_top_raw:
            x, y = mm_top_raw
            for j, (lbl, xv, yv) in enumerate([
                ("raw",    x,     y    ),
                ("inv_y",  x,     S-y  ),
                ("inv_x",  S-x,   y    ),
                ("inv_xy", S-x,   S-y  ),
            ]):
                cv2.putText(prev_top,
                            f"{lbl}: ({xv:.0f},{yv:.0f})",
                            (10, 25+22*j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        else:
            cv2.putText(prev_top, "TOP: non vu",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(prev_bot,
                    f"n={len(results)}  ESPACE=capture  Q=fin",
                    (10,350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

        cv2.imshow("CAM BOTTOM", prev_bot)
        cv2.imshow("CAM TOP",    prev_top)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            cap_bot.read(); _, fb2 = cap_bot.read()
            cap_top.read(); _, ft2 = cap_top.read()
            if fb2 is None or ft2 is None:
                print("  ❌ frame nulle"); continue

            fb2 = cv2.remap(fb2, map1_bot, map2_bot, cv2.INTER_LINEAR)
            ft2 = cv2.remap(ft2, map1_top, map2_top, cv2.INTER_LINEAR)

            tb = detect_tag(fb2, det, TAG_ID)
            tt = detect_tag(ft2, det, TAG_ID)

            if tb is None: print("  ❌ tag non vu bot"); continue
            if tt is None: print("  ❌ tag non vu top"); continue

            mb = pixel_to_mm(*tb, K_bot, rvec_bot, tvec_bot)
            mt = pixel_to_mm(*tt, K_top, rvec_top, tvec_top)
            if mb is None or mt is None:
                print("  ❌ conversion échouée"); continue

            x, y = mt
            results.append((mb, {"raw":(x,y),"inv_y":(x,S-y),
                                  "inv_x":(S-x,y),"inv_xy":(S-x,S-y)}))

            print(f"\n── #{len(results)} ─────────────────────────────────")
            print(f"  BOT     = ({mb[0]:6.1f}, {mb[1]:6.1f})")
            print(f"  raw     = ({x:6.1f}, {y:6.1f})")
            print(f"  inv_y   = ({x:6.1f}, {S-y:6.1f})")
            print(f"  inv_x   = ({S-x:6.1f}, {y:6.1f})")
            print(f"  inv_xy  = ({S-x:6.1f}, {S-y:6.1f})")

    # ── Bilan ────────────────────────────────────────────────────────
    if len(results) >= 2:
        errs = {v: 0.0 for v in ["raw","inv_y","inv_x","inv_xy"]}
        for (bot, top) in results:
            for v in errs:
                tx, ty = top[v]
                errs[v] += ((bot[0]-tx)**2+(bot[1]-ty)**2)**0.5

        best_v = min(errs, key=errs.get)
        print(f"\n{'='*50}\nBILAN {len(results)} mesures\n{'='*50}")
        for v, e in sorted(errs.items(), key=lambda x: x[1]):
            print(f"  {v:8s}: {e/len(results):6.1f}mm"
                  + (" ← MEILLEURE" if v==best_v else ""))

        print(f"\nDétail '{best_v}':")
        for i,(bot,top) in enumerate(results):
            tx,ty = top[best_v]
            err = ((bot[0]-tx)**2+(bot[1]-ty)**2)**0.5
            print(f"  #{i+1}: BOT=({bot[0]:.1f},{bot[1]:.1f}) "
                  f"TOP=({tx:.1f},{ty:.1f}) err={err:.1f}mm")
        print(f"\n→ Utilise '{best_v}' dans cam_top_thread.py")
        # ── Calcul homographie top → bottom ──────────────────────────
        pts_top = np.array([[r[1]["raw"][0], r[1]["raw"][1]] for r in results],
                            dtype=np.float32)
        pts_bot = np.array([[r[0][0], r[0][1]] for r in results],
                            dtype=np.float32)

        H_t2b, _ = cv2.findHomography(pts_top, pts_bot, cv2.RANSAC)
        print(f"\nH_top_to_bottom:\n{H_t2b}")

        print("\nVérification reprojection :")
        for i, (bot, top) in enumerate(results):
            pt  = np.array([[[top['raw'][0], top['raw'][1]]]], dtype=np.float32)
            res = cv2.perspectiveTransform(pt, H_t2b)
            rx, ry = res[0,0,0], res[0,0,1]
            err = ((bot[0]-rx)**2+(bot[1]-ry)**2)**0.5
            print(f"  #{i+1}: BOT=({bot[0]:.1f},{bot[1]:.1f})  "
                  f"H→({rx:.1f},{ry:.1f})  err={err:.1f}mm")

        out = {"H_top_to_bottom": H_t2b.tolist()}
        json.dump(out, open(config_path("H_top_to_bottom.json"), "w"), indent=2)
        print(f"\n✅ Sauvegardé → H_top_to_bottom.json")

    proj[:] = 0
    dm.display_image_on_projector_monitor(proj)
    cap_bot.release()
    cap_top.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()