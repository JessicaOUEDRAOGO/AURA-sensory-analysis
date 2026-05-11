# -*- coding: utf-8 -*-
"""
projection_loop.py
==================
Logique de projection IDENTIQUE à test_projection_blanc.py.

Source de position : cam_top_thread.positions
  → dict {marker_id: (x_mm, y_mm)} partagé, thread-safe
  → une seule source, jamais de fantôme

Pas d'EMA ici — déjà appliqué dans TrackedCup.update_mm().
"""

import cv2
import json
import numpy as np
import time
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


PROJ_WIDTH        = 3840
PROJ_HEIGHT       = 2160
TARGET_FPS        = 30.0
RING_RADIUS_PX    = 100
RING_THICKNESS_PX = 10
RING_COLOR        = (0, 200, 0)   # vert — identique à test_projection_blanc


class ProjectionLoop(QThread):

    data_signal = pyqtSignal(dict)
    fps_signal  = pyqtSignal(float)

    def __init__(
        self,
        cam_top_thread,           # référence directe → self.positions
        display_manager:  DisplayManager,
        image_background: np.ndarray,
        show_grid:        bool = False,
        record_window=None,
        grid_size:        int  = 700,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_top         = cam_top_thread
        self.display_manager  = display_manager
        self.image_background = image_background
        self.show_grid        = show_grid
        self.record_window    = record_window
        self.grid_size        = grid_size
        self.running          = False

        h_data  = json.load(open(config_path("H_table_to_proj.json"), "r"))
        self._H = np.array(h_data["H_table_to_proj"], dtype=np.float32)

        # Frame pré-allouée — identique à test_projection_blanc
        self._proj_frame = np.ones(
            (PROJ_HEIGHT, PROJ_WIDTH, 3), dtype=np.uint8) * 255

        self._fps_count = 0
        self._fps_t0    = 0.0

    def set_show_grid(self, show: bool) -> None:
        self.show_grid = bool(show)

    def update_background(self, new_bg: np.ndarray) -> None:
        self.image_background = new_bg

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.running    = True
        self._fps_count = 0
        self._fps_t0    = time.monotonic()
        frame_duration  = 1.0 / TARGET_FPS

        print(f"[Projection] démarré  {PROJ_WIDTH}x{PROJ_HEIGHT}")

        while self.running:
            t_start = time.monotonic()

            # ── Lire positions KCF — source unique ────────────────────
            with self._cam_top._pos_lock:
                positions = dict(self._cam_top.positions)

            # ── Rendu — identique à test_projection_blanc ─────────────
            self._proj_frame[:] = 255

            data_out = []
            for marker_id, (x_mm, y_mm) in positions.items():
                pt  = np.array([[[x_mm, y_mm]]], dtype=np.float32)
                pxy = cv2.perspectiveTransform(pt, self._H)
                px  = int(pxy[0, 0, 0])
                py  = int(pxy[0, 0, 1])

                m = RING_RADIUS_PX + RING_THICKNESS_PX
                if m <= px <= PROJ_WIDTH - m and m <= py <= PROJ_HEIGHT - m:
                    cv2.circle(self._proj_frame, (px, py),
                               RING_RADIUS_PX, RING_COLOR, RING_THICKNESS_PX,
                               lineType=cv2.LINE_AA)

                data_out.append((marker_id, [x_mm, y_mm]))

            # ── Grille optionnelle ────────────────────────────────────
            if self.show_grid and self.record_window is not None:
                grid = self._build_warped_grid(positions)
                if grid is not None:
                    cv2.addWeighted(grid, 1.0, self._proj_frame, 1.0, 0,
                                    self._proj_frame)

            # ── Envoi projecteur — identique à test_projection_blanc ──
            self.display_manager.display_image_on_projector_monitor(
                self._proj_frame)

            self.data_signal.emit({"data": data_out})

            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                fps = self._fps_count / (now - self._fps_t0)
                self.fps_signal.emit(fps)
                print(f"[Projection] FPS={fps:.1f}  cups={len(positions)}")
                self._fps_count = 0
                self._fps_t0    = now

            elapsed = time.monotonic() - t_start
            sleep   = frame_duration - elapsed
            if sleep > 0:
                time.sleep(sleep)

        self._proj_frame[:] = 0
        self.display_manager.display_image_on_projector_monitor(
            self._proj_frame)
        print("[Projection] arrêté")

    def _build_warped_grid(self, positions) -> Optional[np.ndarray]:
        try:
            from src.core.projection.draw_utils import DrawUtils
            bounds = self.record_window.get_bounds_from_inputs()
            if bounds is None: return None
            x_min,x_max,y_min,y_max,x_leg,y_leg = bounds
            TABLE_SIZE_MM = 597.0
            graph = np.full((self.grid_size,self.grid_size,3),255,dtype=np.uint8)
            graph = DrawUtils.draw_math_grid_on_image(
                graph,x_min,x_max,y_min,y_max,x_leg,y_leg,self.grid_size)
            for mid,(x_mm,y_mm) in positions.items():
                xg=int((x_mm/TABLE_SIZE_MM)*self.grid_size)
                yg=int((y_mm/TABLE_SIZE_MM)*self.grid_size)
                if 0<=xg<self.grid_size and 0<=yg<self.grid_size:
                    cv2.circle(graph,(xg,yg),14,RING_COLOR,-1)
            return cv2.warpPerspective(graph,self._H,(PROJ_WIDTH,PROJ_HEIGHT))
        except Exception as e:
            print(f"[Projection] grille erreur: {e}"); return None