# -*- coding: utf-8 -*-
"""
cup_top_tracker.py — Tracker CSRT pour la tasse vue depuis la cam_top
======================================================================
VERSION PATCHEE : update() retourne (ok, pos_mm, bbox) au lieu de (ok, pos_mm)
pour eviter le double appel CSRT qui divisait le FPS par 2.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class CupTopTracker:
    """
    Encapsule un tracker CSRT OpenCV pour une tasse unique.

    - start()  : initialise le tracker sur la frame et la bbox donnees
    - update() : retourne (ok, pos_mm, bbox_px) — UN SEUL appel CSRT par frame
    - stop()   : libere le tracker
    """

    def __init__(self, pose_path: str):
        """
        pose_path : chemin vers camtop_table_pose.json
                    (contient rvec, tvec, camera_matrix)
        """
        self._tracker: Optional[cv2.TrackerCSRT] = None
        self._active:  bool = False
        self._cup_id:  Optional[int] = None

        data       = json.load(open(pose_path, "r", encoding="utf-8"))
        self._rvec = np.array(data["rvec"],          dtype=np.float64)
        self._tvec = np.array(data["tvec"],          dtype=np.float64)
        self._K    = np.array(data["camera_matrix"], dtype=np.float64)
        self._dist = np.zeros((5, 1), dtype=np.float64)

    # ── Proprietes ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def cup_id(self) -> Optional[int]:
        return self._cup_id

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self, frame: np.ndarray,
              bbox_pixel: Tuple[int, int, int, int],
              cup_id: int) -> bool:
        self.stop()

        x, y, w, h = bbox_pixel
        fh, fw = frame.shape[:2]

        print(f"[CupTopTracker] init — cup_id={cup_id} "
              f"bbox_raw=({x},{y},{w},{h}) frame=({fw}x{fh})")

        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(4, min(w, fw - x))
        h = max(4, min(h, fh - y))

        self._tracker = cv2.TrackerCSRT_create()
        try:
            self._tracker.init(frame, (x, y, w, h))
            init_ok = True

            debug = frame.copy()
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(debug, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
            cv2.imwrite(f"debug_csrt_init_cup{cup_id}.jpg", debug)
            cv2.imwrite(f"debug_csrt_crop_cup{cup_id}.jpg", frame[y:y+h, x:x+w])

        except Exception as e:
            print(f"[CupTopTracker] Exception init : {e}")
            init_ok = False

        if init_ok:
            self._active = True
            self._cup_id = cup_id
            print(f"[CupTopTracker] START OK — cup_id={cup_id} bbox=({x},{y},{w},{h})")
        else:
            self._tracker = None
            print(f"[CupTopTracker] ERREUR init — cup_id={cup_id}")

        return init_ok

    def update(self, frame: np.ndarray
               ) -> Tuple[bool, Optional[Tuple[float, float]], Optional[Tuple[int, int, int, int]]]:
        """
        Met a jour le tracker sur la nouvelle frame.

        Retourne :
            (True,  (x_mm, y_mm), (x, y, w, h))  si succes
            (False, None,         None)            si perdu ou non actif

        IMPORTANT : UN SEUL appel a _tracker.update() par frame.
        Ne pas rappeler _tracker.update() en dehors de cette methode.
        """
        if not self._active or self._tracker is None:
            return False, None, None

        ok, raw_bbox = self._tracker.update(frame)

        if not ok:
            print(f"[CupTopTracker] Tracker perdu — cup_id={self._cup_id}")
            return False, None, None

        x, y, w, h = raw_bbox
        bbox_int = (int(x), int(y), max(4, int(w)), max(4, int(h)))

        cx = x + w / 2.0
        cy = y + h / 2.0

        pos_mm = self._pixel_to_table_mm(cx, cy)
        if pos_mm is None:
            return False, None, None

        return True, pos_mm, bbox_int

    def stop(self) -> None:
        if self._active:
            print(f"[CupTopTracker] STOP — cup_id={self._cup_id}")
        self._tracker = None
        self._active  = False
        self._cup_id  = None

    # ── Conversion pixel -> mm ────────────────────────────────────────────────

    def _pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        K_inv = np.linalg.inv(self._K)
        ray   = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def _intersect_ray_plane(self, ray: np.ndarray) -> Optional[np.ndarray]:
        R, _         = cv2.Rodrigues(self._rvec)
        normal       = R[:, 2]
        plane_origin = self._tvec.reshape(3)
        denom        = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        return t * ray if t >= 0 else None

    def _camera_to_table(self, pt_cam: np.ndarray) -> Tuple[float, float]:
        R, _ = cv2.Rodrigues(self._rvec)
        pt   = R.T @ (pt_cam - self._tvec.reshape(3))
        return float(pt[0]), float(pt[1])

    def _pixel_to_table_mm(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        ray    = self._pixel_to_ray(u, v)
        pt_cam = self._intersect_ray_plane(ray)
        if pt_cam is None:
            return None
        return self._camera_to_table(pt_cam)

    def table_mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        pt_3d = np.array([[x_mm, y_mm, 0.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(pt_3d, self._rvec, self._tvec,
                                         self._K, self._dist)
        px, py = projected[0][0]
        return float(px), float(py)