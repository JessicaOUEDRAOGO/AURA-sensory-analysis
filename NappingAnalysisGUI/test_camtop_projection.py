# # -*- coding: utf-8 -*-
# """
# test_projection_blanc.py
# ========================
# - Projecteur : fond blanc + anneaux verts sur les tasses
# - Détection : HsvCupDetector (même algo que cup_tracking_pipeline.py)
# - Touches : q=quitter  d=masque  +/-=seuil  r=reset
# """

# import cv2
# import json
# import numpy as np
# import time

# from src.core.utils.paths import config_path
# from src.core.projection.display_manager import DisplayManager

# # ─────────────────────────────────────────────────────────────────────
# # PARAMÈTRES — modifie ici
# # ─────────────────────────────────────────────────────────────────────
# CAMERA_INDEX   = 1
# CAPTURE_W      = 1920
# CAPTURE_H      = 1080
# PROCESS_W      = 640
# PROCESS_H      = 360
# TABLE_SIZE_MM  = 597.0

# PROJ_W         = 3840
# PROJ_H         = 2160
# PROJ_SCREEN_ID = 1
# RING_RADIUS    = 100
# RING_THICKNESS = 10
# RING_COLOR     = (0, 200, 0)    # vert sur fond blanc

# # ── Détection — AJUSTE CES VALEURS ───────────────────────────────────
# V_THRESHOLD  = 110   # +/- avec les touches  (sombre=bas, clair=haut)
# AREA_MIN     = 800   # aire min blob (espace 640×360)
# AREA_MAX     = 25000 # aire max blob
# CIRC_MIN     = 0.20  # circularité min (0=n'importe quelle forme, 1=cercle)
# MARGIN       = 20    # marge bbox en pixels

# SCALE = PROCESS_W / CAPTURE_W   # 0.333


# # ─────────────────────────────────────────────────────────────────────
# # PoseConverter — coords natives → mm
# # ─────────────────────────────────────────────────────────────────────
# class PoseConverter:
#     def __init__(self, pose_path):
#         d          = json.load(open(pose_path, "r", encoding="utf-8"))
#         self._rvec = np.array(d["rvec"],          dtype=np.float64)
#         self._tvec = np.array(d["tvec"],          dtype=np.float64)
#         self._K    = np.array(d["camera_matrix"], dtype=np.float64)
#         print(f"[Pose] fx={self._K[0,0]:.1f} cx={self._K[0,2]:.1f}")

#     def pixel_to_mm(self, u, v):
#         K_inv = np.linalg.inv(self._K)
#         ray   = K_inv @ np.array([u, v, 1.0])
#         ray  /= np.linalg.norm(ray)
#         R, _  = cv2.Rodrigues(self._rvec)
#         n, o  = R[:,2], self._tvec.reshape(3)
#         d     = np.dot(n, ray)
#         if abs(d) < 1e-9: return None
#         t = np.dot(n, o) / d
#         if t < 0: return None
#         pt = R.T @ (ray*t - o)
#         return float(pt[0]), TABLE_SIZE_MM - float(pt[1])


# # ─────────────────────────────────────────────────────────────────────
# # Détecteur — copié de cup_tracking_pipeline.py
# # ─────────────────────────────────────────────────────────────────────
# class CupDetector:
#     def __init__(self):
#         self.v_threshold = V_THRESHOLD
#         self._kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#         self._ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#     def detect(self, frame):
#         mask = self._mask(frame)
#         return self._bboxes(frame, mask), mask

#     def _mask(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, mask = cv2.threshold(
#             gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kc, iterations=1)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._ko, iterations=1)
#         return mask

#     def _bboxes(self, frame, mask):
#         fh, fw = frame.shape[:2]
#         cnts, _ = cv2.findContours(
#             mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         out = []
#         for cnt in cnts:
#             area = cv2.contourArea(cnt)
#             if not (AREA_MIN <= area <= AREA_MAX):
#                 continue
#             perim = cv2.arcLength(cnt, True)
#             if perim < 1 or 4*np.pi*area/perim**2 < CIRC_MIN:
#                 continue
#             x, y, w, h = cv2.boundingRect(cnt)
#             # Centre de masse
#             M = cv2.moments(cnt)
#             if M["m00"] < 1: continue
#             cx = M["m10"] / M["m00"]
#             cy = M["m01"] / M["m00"]
#             x1 = max(0, x-MARGIN);   y1 = max(0, y-MARGIN)
#             x2 = min(fw, x+w+MARGIN); y2 = min(fh, y+h+MARGIN)
#             out.append((cx, cy, x2-x1, y2-y1))
#         return out

#     @staticmethod
#     def _remove_border(mask):
#         """Supprime uniquement les blobs collés au bord — pas ceux qui sont juste proches."""
#         h, w   = mask.shape
#         # Eroder le masque de 5px — les blobs qui touchent le bord disparaissent
#         # mais ceux qui sont juste proches restent
#         kernel = np.ones((5, 5), np.uint8)
#         eroded = cv2.erode(mask, kernel, iterations=1)
#         # Reconstruire depuis l'érodé — récupère les blobs intérieurs complets
#         # en supprimant ce qui a disparu à l'érosion sur les bords
#         border_mask = np.zeros((h+2, w+2), np.uint8)
#         flood = eroded.copy()
#         # Flood depuis les 4 coins seulement (pas tout le bord)
#         for pt in [(0,0),(0,h-1),(w-1,0),(w-1,h-1)]:
#             if flood[pt[1], pt[0]]:
#                 cv2.floodFill(flood, border_mask, pt, 0)
#         return flood


# # ─────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────
# def main():
#     H_proj = np.array(
#         json.load(open(config_path("H_table_to_proj.json")))["H_table_to_proj"],
#         dtype=np.float32)

#     converter = PoseConverter(str(config_path("camtop_table_pose.json")))
#     detector  = CupDetector()

#     # Undistort
#     map1 = map2 = None
#     try:
#         c    = json.load(open(config_path("camera_calibration_top.json")))
#         K    = np.array(c["camera_matrix"], dtype=np.float64)
#         dist = np.array(c["dist_coeffs"],   dtype=np.float64)
#         nK, _ = cv2.getOptimalNewCameraMatrix(
#             K, dist, (CAPTURE_W, CAPTURE_H), 1, (CAPTURE_W, CAPTURE_H))
#         map1, map2 = cv2.initUndistortRectifyMap(
#             K, dist, None, nK, (CAPTURE_W, CAPTURE_H), cv2.CV_16SC2)
#         print("[Test] Undistort OK")
#     except FileNotFoundError:
#         print("[Test] Pas de calibration — frames brutes")

#     dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
#     dm.resolution = (PROJ_W, PROJ_H)

#     # Buffer projecteur — fond BLANC
#     proj_frame = np.ones((PROJ_H, PROJ_W, 3), dtype=np.uint8) * 255

#     cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

#     if not cap.isOpened():
#         print(f"[Test] ERREUR cam {CAMERA_INDEX}")
#         return

#     print("q=quitter  d=masque  +/-=seuil  r=reset seuil")
#     show_mask = False
#     fps_t0    = time.monotonic()
#     fps_count = 0
#     fps       = 0.0

#     while True:
#         ret, frame_native = cap.read()
#         if not ret or frame_native is None:
#             continue

#         if map1 is not None:
#             frame_native = cv2.remap(frame_native, map1, map2, cv2.INTER_LINEAR)

#         frame_small = cv2.resize(frame_native, (PROCESS_W, PROCESS_H),
#                                   interpolation=cv2.INTER_LINEAR)

#         bboxes, mask = detector.detect(frame_small)

#         # ── Fond blanc + anneaux ─────────────────────────────────────
#         proj_frame[:] = 255

#         results = []
#         for (cx_s, cy_s, bw, bh) in bboxes:
#             cx_n = cx_s / SCALE
#             cy_n = cy_s / SCALE
#             mm   = converter.pixel_to_mm(cx_n, cy_n)
#             if mm is None:
#                 continue
#             x_mm, y_mm = mm
#             pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
#             pxy = cv2.perspectiveTransform(pt, H_proj)
#             px  = int(pxy[0,0,0]);  py = int(pxy[0,0,1])
#             results.append((cx_s, cy_s, x_mm, y_mm, px, py))

#             m = RING_RADIUS + RING_THICKNESS
#             if m <= px <= PROJ_W-m and m <= py <= PROJ_H-m:
#                 cv2.circle(proj_frame, (px, py),
#                            RING_RADIUS, RING_COLOR, RING_THICKNESS,
#                            lineType=cv2.LINE_AA)

#         dm.display_image_on_projector_monitor(proj_frame)

#         # ── Preview ──────────────────────────────────────────────────
#         preview = cv2.resize(frame_small, (960, 540))
#         ps = 960 / PROCESS_W

#         for i, (cx_s, cy_s, x_mm, y_mm, px, py) in enumerate(results):
#             pcx = int(cx_s * ps);  pcy = int(cy_s * ps)
#             cv2.circle(preview, (pcx, pcy), 8, (0, 0, 255), -1)
#             cv2.putText(preview,
#                         f"c{i} ({x_mm:.0f},{y_mm:.0f})mm",
#                         (pcx+5, pcy-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,80), 1)

#         fps_count += 1
#         now = time.monotonic()
#         if now - fps_t0 >= 1.0:
#             fps    = fps_count / (now - fps_t0)
#             fps_count = 0
#             fps_t0    = now
#             print(f"[Test] FPS={fps:.1f}  det={len(bboxes)}"
#                   f"  V_seuil={detector.v_threshold}")
#             for i, (cx_s, cy_s, x_mm, y_mm, px, py) in enumerate(results):
#                 print(f"  c{i}  mm=({x_mm:.1f},{y_mm:.1f})"
#                       f"  proj=({px},{py})")

#         cv2.putText(preview,
#                     f"FPS:{fps:.1f}  det:{len(bboxes)}"
#                     f"  seuil:{detector.v_threshold}  (+/-)",
#                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
#         cv2.imshow("CamTop preview", preview)

#         if show_mask:
#             cv2.imshow("Masque HSV",
#                        cv2.resize(mask, (960, 540)))

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('d'):
#             show_mask = not show_mask
#             if not show_mask:
#                 cv2.destroyWindow("Masque HSV")
#         elif key in (ord('+'), ord('=')):
#             detector.v_threshold = min(245, detector.v_threshold + 5)
#             print(f"V_seuil → {detector.v_threshold}")
#         elif key == ord('-'):
#             detector.v_threshold = max(10, detector.v_threshold - 5)
#             print(f"V_seuil → {detector.v_threshold}")
#         elif key == ord('r'):
#             detector.v_threshold = V_THRESHOLD
#             print(f"V_seuil reset → {V_THRESHOLD}")

#     cap.release()
#     cv2.destroyAllWindows()
#     proj_frame[:] = 0
#     dm.display_image_on_projector_monitor(proj_frame)
#     print("[Test] Terminé")


# if __name__ == "__main__":
#     main()
# test_cross_calibration.py
# Compare les coordonnées mm ArUco (cam_bottom) vs PoseConverter (cam_top)
# pour les mêmes tasses vues des deux caméras simultanément

# import cv2
# import json
# import numpy as np
# from src.core.utils.paths import config_path

# CAM_BOTTOM_INDEX = 1
# CAM_TOP_INDEX    = 0
# CAM_BOTTOM_W     = 1920
# CAM_BOTTOM_H     = 1080
# CAM_TOP_NATIVE_W = 1920
# CAM_TOP_NATIVE_H = 1080
# CAP_W = 640
# CAP_H = 360
# TABLE_SIZE_MM = 597.0

# # ── ArUco (cam_bottom) ───────────────────────────────────────────────
# def make_aruco_detector():
#     d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#     p = cv2.aruco.DetectorParameters()
#     return cv2.aruco.ArucoDetector(d, p)

# def aruco_detect(frame, detector, K, rvec, tvec):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     corners, ids, _ = detector.detectMarkers(gray)
#     if ids is None:
#         return {}
#     result = {}
#     K_inv = np.linalg.inv(K)
#     R, _  = cv2.Rodrigues(rvec)
#     for i, mid in enumerate(ids.flatten()):
#         pts = corners[i][0]
#         cx  = float(np.mean(pts[:, 0]))
#         cy  = float(np.mean(pts[:, 1]))
#         ray = K_inv @ np.array([cx, cy, 1.0])
#         ray = ray / np.linalg.norm(ray)
#         n   = R[:, 2]
#         orig = tvec.reshape(3)
#         d   = np.dot(n, ray)
#         if abs(d) < 1e-9:
#             continue
#         t   = np.dot(n, orig) / d
#         if t < 0:
#             continue
#         pt  = R.T @ (ray * t - orig)
#         result[int(mid)] = (float(pt[0]), float(pt[1]))
#     return result

# # ── PoseConverter (cam_top) ──────────────────────────────────────────
# def make_top_converter():
#     data  = json.load(open(config_path("camtop_table_pose.json"), "r"))
#     rvec  = np.array(data["rvec"], dtype=np.float64)
#     tvec  = np.array(data["tvec"], dtype=np.float64)
#     K     = np.array(data["camera_matrix"], dtype=np.float64)

#     # Scale K pour résolution de capture
#     sx = CAP_W / CAM_TOP_NATIVE_W
#     sy = CAP_H / CAM_TOP_NATIVE_H
#     K[0,0]*=sx; K[1,1]*=sy; K[0,2]*=sx; K[1,2]*=sy

#     print(f"[CamTop] K après scaling: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
#           f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
#     return rvec, tvec, K

# def top_pixel_to_mm(u, v, rvec, tvec, K):
#     K_inv = np.linalg.inv(K)
#     ray   = K_inv @ np.array([u, v, 1.0])
#     ray  /= np.linalg.norm(ray)
#     R, _  = cv2.Rodrigues(rvec)
#     n, o  = R[:,2], tvec.reshape(3)
#     d     = np.dot(n, ray)
#     if abs(d) < 1e-9: return None
#     t = np.dot(n, o) / d
#     if t < 0: return None
#     pt = R.T @ (ray*t - o)
#     # Nouvelle orientation cam_top : pas d'inversion Y
#     return float(pt[0]), float(pt[1])

# def main():
#     # Charger poses cam_bottom
#     b = json.load(open(config_path("cambottom_table_pose.json"), "r"))
#     K_bot   = np.array(b["camera_matrix"], dtype=np.float64)
#     rvec_bot = np.array(b["rvec"], dtype=np.float64)
#     tvec_bot = np.array(b["tvec"], dtype=np.float64)
#     aruco_det = make_aruco_detector()

#     # Charger pose cam_top
#     rvec_top, tvec_top, K_top = make_top_converter()

#     # Ouvrir les deux caméras
#     cap_bot = cv2.VideoCapture(CAM_BOTTOM_INDEX, cv2.CAP_DSHOW)
#     cap_bot.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_BOTTOM_W)
#     cap_bot.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_BOTTOM_H)

#     cap_top = cv2.VideoCapture(CAM_TOP_INDEX, cv2.CAP_DSHOW)
#     cap_top.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
#     cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

#     print("\n[Calib] Place les tasses sur les croix projetées")
#     print("[Calib] Appuie sur ESPACE pour capturer, Q pour quitter\n")

#     while True:
#         ret_b, frame_bot = cap_bot.read()
#         ret_t, frame_top = cap_top.read()
#         if not ret_b or not ret_t:
#             continue

#         cv2.imshow("CamBottom", cv2.resize(frame_bot, (640, 360)))
#         cv2.imshow("CamTop",    frame_top)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord(' '):
#             # ── Capture et comparaison ────────────────────────────
#             aruco_pos = aruco_detect(frame_bot, aruco_det, K_bot, rvec_bot, tvec_bot)

#             print("\n" + "="*60)
#             print("COMPARAISON ArUco (cam_bottom) vs PoseConverter (cam_top)")
#             print("="*60)

#             # Pour chaque tag ArUco détecté, cherche la tasse correspondante
#             # dans cam_top par proximité spatiale
#             # D'abord : détecte les centres dans cam_top via seuillage simple
#             gray_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)
#             _, mask  = cv2.threshold(gray_top, 110, 255, cv2.THRESH_BINARY_INV)
#             kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
#             cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_SIMPLE)

#             top_centers_mm = []
#             for cnt in cnts:
#                 area = cv2.contourArea(cnt)
#                 if not (800 <= area <= 25000):
#                     continue
#                 M = cv2.moments(cnt)
#                 if M["m00"] < 1:
#                     continue
#                 cx = M["m10"] / M["m00"]
#                 cy = M["m01"] / M["m00"]
#                 mm = top_pixel_to_mm(cx, cy, rvec_top, tvec_top, K_top)
#                 if mm:
#                     top_centers_mm.append((cx, cy, mm[0], mm[1]))

#             print(f"\nArUco détecte {len(aruco_pos)} tags :")
#             for mid, (x, y) in sorted(aruco_pos.items()):
#                 print(f"  tag{mid:2d} → ArUco mm = ({x:6.1f}, {y:6.1f})")

#             print(f"\nCamTop détecte {len(top_centers_mm)} centres :")
#             for (cpx, cpy, xmm, ymm) in top_centers_mm:
#                 print(f"  cam_px=({cpx:5.1f},{cpy:5.1f}) → Top mm = ({xmm:6.1f}, {ymm:6.1f})")

#             # Appariement par proximité
#             print(f"\nAPPARIEMENT (tolérance 80mm) :")
#             for mid, (ax, ay) in sorted(aruco_pos.items()):
#                 best_d = float("inf")
#                 best   = None
#                 for (cpx, cpy, tx, ty) in top_centers_mm:
#                     d = ((ax-tx)**2 + (ay-ty)**2)**0.5
#                     if d < best_d:
#                         best_d = d
#                         best   = (tx, ty)
#                 if best and best_d < 80:
#                     err_x = best[0] - ax
#                     err_y = best[1] - ay
#                     print(f"  tag{mid:2d}: ArUco=({ax:6.1f},{ay:6.1f})  "
#                           f"Top=({best[0]:6.1f},{best[1]:6.1f})  "
#                           f"ERREUR=({err_x:+.1f},{err_y:+.1f})mm  dist={best_d:.1f}mm")
#                 else:
#                     print(f"  tag{mid:2d}: ArUco=({ax:6.1f},{ay:6.1f})  "
#                           f"Top= PAS DE MATCH (dist={best_d:.0f}mm)")

#         elif key == ord('q'):
#             break

#     cap_bot.release()
#     cap_top.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
# test_camtop_projection.py — MODE CALIBRATION
# Place une tasse à une position connue, note les coordonnées cam_top
# et vérifie que la projection tombe bien dessus

# import cv2
# import json
# import numpy as np
# from src.core.utils.paths import config_path
# from src.core.projection.display_manager import DisplayManager

# PROJ_W         = 3840
# PROJ_H         = 2160
# PROJ_SCREEN_ID = 1
# TABLE_SIZE_MM  = 597.0

# def main():
#     h_data = json.load(open(config_path("H_table_to_proj.json"), "r"))
#     H = np.array(h_data["H_table_to_proj"], dtype=np.float32)

#     dm = DisplayManager(projector_screen_id=PROJ_SCREEN_ID)
#     dm.resolution = (PROJ_W, PROJ_H)
#     proj_frame = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)

#     # ── Grille de points de test en mm ───────────────────────────────
#     # Ces points couvrent la table (0-597mm x 0-597mm)
#     test_points_mm = [
#         (100, 100), (300, 100), (500, 100),
#         (100, 300), (300, 300), (500, 300),
#         (100, 500), (300, 500), (500, 500),
#     ]

#     for (x_mm, y_mm) in test_points_mm:
#         pt   = np.array([[[x_mm, y_mm]]], dtype=np.float32)
#         proj = cv2.perspectiveTransform(pt, H)
#         px   = int(proj[0, 0, 0])
#         py   = int(proj[0, 0, 1])

#         if 0 < px < PROJ_W and 0 < py < PROJ_H:
#             # Croix + coordonnées mm
#             cv2.drawMarker(proj_frame, (px, py), (0, 255, 0),
#                           cv2.MARKER_CROSS, 40, 3)
#             cv2.putText(proj_frame, f"({x_mm},{y_mm})",
#                        (px+15, py-15),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#     dm.display_image_on_projector_monitor(proj_frame)
#     print("[Calib] Grille projetée — appuie sur ENTREE pour quitter")
#     input()

#     proj_frame[:] = 0
#     dm.display_image_on_projector_monitor(proj_frame)

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
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