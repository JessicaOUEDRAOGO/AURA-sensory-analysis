# -*- coding: utf-8 -*-
"""
cam_top_thread.py
=================
Thread dédié à la caméra du haut (KCF tracking).

Responsabilités :
  - Lire la cam_top en continu à ~30fps
  - Interroger CupStateBuffer pour savoir si une tasse est SOULEVEE
  - Initialiser un tracker KCF quand une tasse passe SOULEVEE
  - Écrire la position KCF dans CupStateBuffer via update_from_top()
  - Arrêter le tracker quand la tasse repose (state ≠ SOULEVEE)

Ce thread ne fait RIEN d'autre. Pas d'ArUco, pas de projection, pas de mains.

Pourquoi KCF et non CSRT ?
  CSRT ≈ 40ms/frame → 25fps max (et souvent 15fps avec l'overhead)
  KCF  ≈  5ms/frame → 30fps stable

Architecture interne :
  - HsvCupDetector : détection initiale par seuillage (pour init bbox KCF)
  - TrackerKCF     : tracking léger entre les détections
  - Réinitialisation périodique KCF sur détection HSV (corrige la dérive)
"""

import cv2
import json
import numpy as np
import time
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional, Tuple

from src.core.utils.paths import config_path


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres KCF
# ─────────────────────────────────────────────────────────────────────────────

# Résolution interne de traitement (frame réduite pour le KCF)
# 640px : KCF ~5ms  |  960px : ~10ms  |  1280px : ~20ms
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Intervalle entre deux réinitialisations KCF depuis détection HSV (secondes)
DETECT_REINIT_INTERVAL_S = 0.20   # ~5 réinits/sec pour corriger la dérive

# Taille de la bbox initiale en pixels (espace PROCESS_WIDTH)
# Ajuster selon la hauteur de ta cam_top et le diamètre de la tasse
CUP_BBOX_PROC_PX = 50

# Seuil de luminosité pour détection HSV des tasses sombres
V_THRESHOLD = 110

# Drift max acceptable entre deux frames (fraction de la diagonale bbox)
MAX_DRIFT_RATIO = 0.8
MAX_KCF_ARUCO_DIST_MM = 250.0  # distance max KCF↔ArUco en mm avant d'invalider

# ─────────────────────────────────────────────────────────────────────────────
# Détecteur HSV (seuillage luminosité → contours → bbox)
# ─────────────────────────────────────────────────────────────────────────────

class _HsvDetector:
    """Détecte les tasses sombres sur fond blanc rétro-éclairé."""

    AREA_MIN        = 800
    AREA_MAX        = 25_000
    CIRCULARITY_MIN = 0.20
    MARGIN          = 15

    def __init__(self, v_threshold: int = V_THRESHOLD):
        self.v_threshold = v_threshold
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> list:
        """Retourne [(x, y, w, h), ...] dans l'espace frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.v_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
        mask = self._remove_border(mask)

        fh, fw = frame.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (self.AREA_MIN <= area <= self.AREA_MAX):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1.0:
                continue
            if 4.0 * np.pi * area / (perim ** 2) < self.CIRCULARITY_MIN:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(0, x - self.MARGIN)
            y1 = max(0, y - self.MARGIN)
            x2 = min(fw, x + w + self.MARGIN)
            y2 = min(fh, y + h + self.MARGIN)
            out.append((x1, y1, x2 - x1, y2 - y1))
        return out

    @staticmethod
    def _remove_border(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        flood = mask.copy()
        for x in range(w):
            if flood[0,   x]: cv2.floodFill(flood, None, (x, 0),   0)
            if flood[h-1, x]: cv2.floodFill(flood, None, (x, h-1), 0)
        for y in range(h):
            if flood[y,   0]: cv2.floodFill(flood, None, (0,   y), 0)
            if flood[y, w-1]: cv2.floodFill(flood, None, (w-1, y), 0)
        return flood


# ─────────────────────────────────────────────────────────────────────────────
# Conversion pixel cam_top → mm table
# ─────────────────────────────────────────────────────────────────────────────

class _PoseConverter:
    """Charge la pose cam_top et expose pixel→mm."""

    def __init__(self, pose_path: str):
        data       = json.load(open(pose_path, "r", encoding="utf-8"))
        self._rvec = np.array(data["rvec"],          dtype=np.float64)
        self._tvec = np.array(data["tvec"],          dtype=np.float64)
        self._K    = np.array(data["camera_matrix"], dtype=np.float64)

    # Taille de la table en mm — nécessaire pour la correction d'axe Y
    TABLE_SIZE_MM = 597.0

    def pixel_to_mm(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Pixel cam_top (résolution native) → (x_mm, y_mm) repère table commun.

        Correction axe Y : cam_top et cam_bottom partagent le même X mais
        ont un Y miroir (TL_bottom = BL_top, BL_bottom = TL_top).
        → y_mm = TABLE_SIZE_MM - y_raw pour ramener dans le repère cam_bottom.
        """
        K_inv = np.linalg.inv(self._K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray   = ray / np.linalg.norm(ray)

        R, _         = cv2.Rodrigues(self._rvec)
        normal       = R[:, 2]
        plane_origin = self._tvec.reshape(3)
        denom        = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        if t < 0:
            return None
        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - self._tvec.reshape(3))

        x_mm =       float(pt_table[0])          # X identique entre les deux caméras
        y_mm = self.TABLE_SIZE_MM - float(pt_table[1])  # Y miroir → repère cam_bottom
        return x_mm, y_mm

    def mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        (x_mm, y_mm) repère cam_bottom → pixel cam_top (résolution native).

        Avant de projeter, on inverse la correction Y pour repasser dans
        le repère propre à cam_top.
        """
        x_top = x_mm
        y_top = self.TABLE_SIZE_MM - y_mm   # inverse de la correction pixel_to_mm

        pt_3d = np.array([[x_top, y_top, 0.0]], dtype=np.float64)
        dist  = np.zeros((5, 1), dtype=np.float64)
        proj, _ = cv2.projectPoints(pt_3d, self._rvec, self._tvec, self._K, dist)
        return float(proj[0][0][0]), float(proj[0][0][1])


# ─────────────────────────────────────────────────────────────────────────────
# Thread principal
# ─────────────────────────────────────────────────────────────────────────────

class CamTopThread(QThread):
    """
    Thread caméra haute — KCF tracking uniquement.

    Signaux :
      fps_signal(float)   : FPS mesuré (émis chaque seconde)
      pos_signal(int, float, float) : (marker_id, x_mm, y_mm) à chaque update KCF
    """

    fps_signal = pyqtSignal(float)
    pos_signal = pyqtSignal(int, float, float)   # optionnel, pour debug

    def __init__(
        self,
        cup_state_buffer,           # CupStateBuffer partagé
        camera_index: int,
        pose_path: str,
        cam_width:  int = 1920,
        cam_height: int = 1080,
        show_preview: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.cup_state_buffer = cup_state_buffer
        self.camera_index     = camera_index
        self.pose_path        = pose_path
        self.cam_width        = cam_width
        self.cam_height       = cam_height
        self.show_preview     = show_preview
        self.running          = False

        # ── Vérification préalable des fichiers critiques ─────────────────
        import os
        if not os.path.isfile(pose_path):
            raise FileNotFoundError(
                f"[CamTop] ERREUR FATALE : pose_path introuvable → {pose_path}\n"
                "Vérifie que camtop_table_pose.json existe dans config/."
            )
        print(f"[CamTop] pose_path OK : {pose_path}")

        try:
            self._converter = _PoseConverter(pose_path)
            print("[CamTop] _PoseConverter initialisé OK")
        except Exception as e:
            raise RuntimeError(f"[CamTop] Échec _PoseConverter : {e}") from e

        self._detector = _HsvDetector()

        # ── Undistort cam_top (pinhole standard, cv2.calibrateCamera) ──────────
        # camera_calibration_top.json contient K et dist bruts (pas new_K).
        # On calcule les maps ici une seule fois → cv2.remap() dans run().
        # camtop_table_pose.json stocke new_K + dist=zeros : la conversion
        # pixel→mm attend donc une frame déjà undistordue.
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._load_undistort(PROCESS_WIDTH, PROCESS_HEIGHT)
        # Adapter la matrice K du converter à la résolution de capture réelle
        scale_x = PROCESS_WIDTH  / self.cam_width   # 640/1920
        scale_y = PROCESS_HEIGHT / self.cam_height  # 360/1080
        K_scaled = self._converter._K.copy()
        K_scaled[0, 0] *= scale_x   # fx
        K_scaled[1, 1] *= scale_y   # fy
        K_scaled[0, 2] *= scale_x   # cx
        K_scaled[1, 2] *= scale_y   # cy
        self._converter._K = K_scaled
        # _scale = 1.0 donc cx_full = cx_proc dans _tick — cohérent
        
        self._scale  = 1.0                    # on capture directement en 640×360
        self._proc_h = PROCESS_HEIGHT

        # État KCF courant
        self._kcf_tracker: Optional[cv2.TrackerKCF] = None
        self._kcf_cup_id:  Optional[int]  = None
        self._kcf_bbox:    Optional[tuple] = None
        self._kcf_active:  bool = False

        # Timestamp dernier réinit HSV
        self._last_hsv_t: float = 0.0

        # FPS
        self._fps_count = 0
        self._fps_t0    = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Boucle principale
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running    = True
        self._fps_count = 0
        self._fps_t0    = time.monotonic()

        print(f"[CamTop] run() entré — ouverture cam index={self.camera_index}")

        try:
            # CAP_DSHOW obligatoire sur Windows pour éviter le blocage
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PROCESS_WIDTH)   # 640 directement
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)  # 360 directement
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[CamTop] ERREUR : cam {self.camera_index} inaccessible "
                    f"(index 0=top, 1=bottom — vérifie branchement USB)")
                return

            # Test lecture initiale avec timeout
            t_open = time.monotonic()
            ret, frame = cap.read()
            print(f"[CamTop] première frame : ret={ret}  "
                f"délai={time.monotonic()-t_open:.2f}s")
            if not ret or frame is None:
                print("[CamTop] ERREUR : première frame échouée")
                cap.release()
                return

            print(f"[CamTop] Caméra {self.camera_index} ouverte  "
                f"shape={frame.shape}  "
                f"scale={self._scale:.3f}  proc={PROCESS_WIDTH}x{self._proc_h}")

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                if self._map1 is not None:
                    frame = cv2.remap(frame, self._map1, self._map2,
                                    interpolation=cv2.INTER_LINEAR)

                proc = frame
                
                self._tick(proc, frame)

                self._fps_count += 1
                now = time.monotonic()
                if now - self._fps_t0 >= 1.0:
                    fps = self._fps_count / (now - self._fps_t0)
                    self.fps_signal.emit(fps)
                    print(f"[CamTop] FPS={fps:.1f}  kcf_active={self._kcf_active}  "
                        f"cup_id={self._kcf_cup_id}")
                    self._fps_count = 0
                    self._fps_t0    = now

                if self.show_preview:
                    self._draw_preview(frame)

        except Exception as e:
            import traceback
            print(f"[CamTop] EXCEPTION dans run() : {e}")
            traceback.print_exc()
        finally:
            cap.release()
            print("[CamTop] Thread arrêté")
            if self.show_preview:
                cv2.destroyWindow("CamTop Preview")

    def stop(self) -> None:
        self.running = False

    def _load_undistort(self, cam_width: int, cam_height: int) -> None:
        """
        Charge camera_calibration_top.json et précalcule les maps
        pour cv2.remap (modèle pinhole standard, cv2.calibrateCamera).

        Si le fichier est absent, l'undistort est désactivé silencieusement
        (les frames seront utilisées brutes — résultat moins précis).
        """
        try:
            calib_path = config_path("camera_calibration_top.json")
            data  = json.load(open(calib_path, "r", encoding="utf-8"))
            K     = np.array(data["camera_matrix"], dtype=np.float64)
            dist  = np.array(data["dist_coeffs"],   dtype=np.float64)

            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K, dist, (cam_width, cam_height), 1, (cam_width, cam_height)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                K, dist, None, new_K,
                (cam_width, cam_height), cv2.CV_16SC2
            )
            print(f"[CamTop] Undistort pinhole activé  "
                  f"new_K fx={new_K[0,0]:.1f} fy={new_K[1,1]:.1f}")
        except FileNotFoundError:
            print("[CamTop] ⚠ camera_calibration_top.json introuvable — "
                  "undistort désactivé (décalage possible)")
        except Exception as e:
            print(f"[CamTop] ⚠ Erreur chargement calibration top : {e} — "
                  "undistort désactivé")



    def _tick(self, proc: np.ndarray, frame_full: np.ndarray) -> None:
        """Appelé chaque frame. proc = frame réduite, frame_full = résolution native."""

        # ── 1. Vérifier si on doit démarrer un tracker ─────────────────
        if not self._kcf_active:
            todo = self.cup_state_buffer.get_cup_to_track()
            if todo is not None:
                marker_id, last_pos_mm = todo
                self._start_kcf(proc, frame_full, marker_id, last_pos_mm)

        # ── 2. Mettre à jour le tracker actif ──────────────────────────
        if self._kcf_active:
            # Vérifier si la tasse est encore SOULEVEE
            if not self.cup_state_buffer.is_kcf_needed(self._kcf_cup_id):
                self._stop_kcf()
                return

            # Update KCF sur frame proc (~5ms)
            ok, raw = self._kcf_tracker.update(proc)

            if ok and raw is not None:
                x, y, w, h     = raw
                new_bbox       = (int(x), int(y), max(4, int(w)), max(4, int(h)))

                # Contrôle de dérive
                if self._kcf_bbox and not self._drift_ok(self._kcf_bbox, new_bbox):
                    print(f"[CamTop] KCF dérive trop → stop cup_id={self._kcf_cup_id}")
                    self._stop_kcf()
                    return

                self._kcf_bbox = new_bbox

                # Centre en coords proc → coords native → mm
                cx_proc = x + w / 2.0
                cy_proc = y + h / 2.0
                cx_full = cx_proc / self._scale
                cy_full = cy_proc / self._scale

                pos_mm = self._converter.pixel_to_mm(cx_full, cy_full)
                if pos_mm is not None:
                    # Validation cohérence géométrique avec dernière pos ArUco
                    last_aruco = self.cup_state_buffer.get_last_aruco_pos(self._kcf_cup_id)
                    if last_aruco is not None:
                        dx   = pos_mm[0] - last_aruco[0]
                        dy   = pos_mm[1] - last_aruco[1]
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist > MAX_KCF_ARUCO_DIST_MM:
                            print(f"[CamTop] KCF dérive géo  dist={dist:.0f}mm "
                                f"> {MAX_KCF_ARUCO_DIST_MM}mm → stop cup_id={self._kcf_cup_id}")
                            self._stop_kcf()
                            return

                    self.cup_state_buffer.update_from_top(self._kcf_cup_id, list(pos_mm))
                    self.pos_signal.emit(self._kcf_cup_id, pos_mm[0], pos_mm[1])

            else:
                print(f"[CamTop] KCF perdu cup_id={self._kcf_cup_id}")
                self._stop_kcf()
                return

            # ── 3. Réinitialisation périodique depuis HSV ────────────────
            now = time.monotonic()
            if now - self._last_hsv_t >= DETECT_REINIT_INTERVAL_S:
                self._last_hsv_t = now
                self._reinit_from_hsv(proc, frame_full)

    # ──────────────────────────────────────────────────────────────────
    # Gestion KCF
    # ──────────────────────────────────────────────────────────────────

    def _start_kcf(
        self,
        proc: np.ndarray,
        frame_full: np.ndarray,
        marker_id: int,
        last_pos_mm: list,
    ) -> None:
        """
        Initialise le tracker KCF.
        1. Projette last_pos_mm → pixel native → pixel proc
        2. Construit la bbox proc
        3. Essaie de raffiner avec HSV si une bbox HSV est proche
        """
        # Projeter la dernière pos connue (ArUco) → pixel native → proc
        px_full, py_full = self._converter.mm_to_pixel(last_pos_mm[0], last_pos_mm[1])
        px_proc = px_full * self._scale
        py_proc = py_full * self._scale

        half = CUP_BBOX_PROC_PX // 2
        fh, fw = proc.shape[:2]
        x = max(0, int(px_proc - half))
        y = max(0, int(py_proc - half))
        w = max(4, min(CUP_BBOX_PROC_PX, fw - x))
        h = max(4, min(CUP_BBOX_PROC_PX, fh - y))

        # Essayer de raffiner avec une détection HSV
        hsv_bboxes = self._detector.detect(proc)
        best = self._best_hsv_match((x, y, w, h), hsv_bboxes)
        if best is not None:
            x, y, w, h = best
            print(f"[CamTop] KCF init raffiné par HSV cup_id={marker_id} "
                  f"bbox=({x},{y},{w},{h})")
        else:
            print(f"[CamTop] KCF init depuis ArUco cup_id={marker_id} "
                  f"bbox=({x},{y},{w},{h})")

        tracker = cv2.TrackerKCF_create()
        try:
            tracker.init(proc, (x, y, w, h))
        except Exception as e:
            print(f"[CamTop] KCF init ERREUR cup_id={marker_id}: {e}")
            return

        self._kcf_tracker = tracker
        self._kcf_cup_id  = marker_id
        self._kcf_bbox    = (x, y, w, h)
        self._kcf_active  = True
        self._last_hsv_t  = time.monotonic()

        self.cup_state_buffer.set_kcf_active(marker_id, True)
        print(f"[CamTop] KCF démarré cup_id={marker_id}")

    def _stop_kcf(self) -> None:
        if self._kcf_cup_id is not None:
            self.cup_state_buffer.set_kcf_active(self._kcf_cup_id, False)
            self.cup_state_buffer.kcf_stopped(self._kcf_cup_id)
            print(f"[CamTop] KCF stoppé cup_id={self._kcf_cup_id}")
        self._kcf_tracker = None
        self._kcf_cup_id  = None
        self._kcf_bbox    = None
        self._kcf_active  = False

    def _reinit_from_hsv(self, proc: np.ndarray, frame_full: np.ndarray) -> None:
        """Réinitialise le KCF sur la bbox HSV la plus proche (corrige la dérive)."""
        if not self._kcf_active or self._kcf_bbox is None:
            return
        hsv_bboxes = self._detector.detect(proc)
        best = self._best_hsv_match(self._kcf_bbox, hsv_bboxes)
        if best is None:
            return
        x, y, w, h = best
        fh, fw = proc.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(4, min(w, fw - x))
        h = max(4, min(h, fh - y))
        try:
            new_tracker = cv2.TrackerKCF_create()
            new_tracker.init(proc, (x, y, w, h))
            self._kcf_tracker = new_tracker
            self._kcf_bbox    = (x, y, w, h)
        except Exception as e:
            print(f"[CamTop] KCF réinit HSV ERREUR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Helpers géométriques
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _center(bbox: tuple) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _dist(b1: tuple, b2: tuple) -> float:
        cx1, cy1 = CamTopThread._center(b1)
        cx2, cy2 = CamTopThread._center(b2)
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

    def _drift_ok(self, prev: tuple, new: tuple) -> bool:
        x, y, w, h = prev
        diag = max(np.sqrt(w ** 2 + h ** 2), 1.0)
        return self._dist(prev, new) < MAX_DRIFT_RATIO * diag

    def _best_hsv_match(self, ref_bbox: tuple, candidates: list) -> Optional[tuple]:
        """Retourne la bbox HSV la plus proche de ref_bbox (si assez proche)."""
        if not candidates:
            return None
        ref_cx, ref_cy = self._center(ref_bbox)
        ref_w, ref_h   = ref_bbox[2], ref_bbox[3]
        max_dist       = max(ref_w, ref_h) * 1.5   # tolérance raisonnable

        best_d   = float("inf")
        best_box = None
        for b in candidates:
            d = self._dist(ref_bbox, b)
            if d < best_d and d < max_dist:
                best_d   = d
                best_box = b
        return best_box

    # ──────────────────────────────────────────────────────────────────
    # Preview debug
    # ──────────────────────────────────────────────────────────────────

    def _draw_preview(self, frame_full: np.ndarray) -> None:
        preview = frame_full.copy()
        if self._kcf_active and self._kcf_bbox is not None:
            inv = 1.0 / self._scale
            x, y, w, h = self._kcf_bbox
            x2, y2, w2, h2 = int(x*inv), int(y*inv), int(w*inv), int(h*inv)
            cv2.rectangle(preview, (x2, y2), (x2+w2, y2+h2), (0, 200, 80), 3)
            cv2.putText(
                preview,
                f"KCF cup_id={self._kcf_cup_id}",
                (x2, max(20, y2 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2
            )
        cv2.imshow("CamTop Preview",
                   cv2.resize(preview, (960, 540), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)
