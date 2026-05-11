# -*- coding: utf-8 -*-
"""
projection_loop.py
==================
Boucle de projection — lit CupStateBuffer et affiche sur projecteur.

Améliorations vs ancienne version :
  - EMA sur les positions mm avant projection (α=0.35) → plus de tremblements
  - Frame pré-allouée réutilisée → pas de fantômes
  - Pas de thread display séparé → pas de saccades
  - Même chaîne H_table_to_proj que test_projection_blanc.py

Couleurs :
  POSEE              → Vert   (0, 200, 0)
  PEUT_ETRE_SOULEVEE → Orange (0, 165, 255)
  SOULEVEE           → Bleu   (255, 100, 0)
"""

import cv2
import json
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

PROJ_WIDTH        = 3840
PROJ_HEIGHT       = 2160
TARGET_FPS        = 30.0

RING_RADIUS_PX    = 100
RING_THICKNESS_PX = 10

COLOR_POSEE              = (0,   200,   0)
COLOR_PEUT_ETRE_SOULEVEE = (0,   165, 255)
COLOR_SOULEVEE           = (255, 100,   0)

# EMA sur les positions mm — même valeur que test_projection_blanc
EMA_ALPHA     = 0.35
EMA_MAX_JUMP  = 200.0


# ══════════════════════════════════════════════════════════════════════════════
#  EMA Filter par tasse
# ══════════════════════════════════════════════════════════════════════════════

class _EMAFilter:
    def __init__(self, alpha: float = EMA_ALPHA,
                 max_jump: float = EMA_MAX_JUMP):
        self._a        = alpha
        self._max_jump = max_jump
        self._val: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._val is None:
            self._val = (x, y)
            return self._val
        dx, dy = x - self._val[0], y - self._val[1]
        if (dx * dx + dy * dy) ** 0.5 > self._max_jump:
            self._val = (x, y)
            return self._val
        self._val = (
            self._a * x + (1 - self._a) * self._val[0],
            self._a * y + (1 - self._a) * self._val[1],
        )
        return self._val

    def reset(self) -> None:
        self._val = None


# ══════════════════════════════════════════════════════════════════════════════
#  ProjectionLoop
# ══════════════════════════════════════════════════════════════════════════════

class ProjectionLoop(QThread):
    """
    Thread de projection.

    Signaux :
      data_signal(dict)  : {"data": [(marker_id, pos, state), ...]}
      fps_signal(float)
    """

    data_signal = pyqtSignal(dict)
    fps_signal  = pyqtSignal(float)

    def __init__(
        self,
        cup_state_buffer,
        display_manager:  DisplayManager,
        image_background: np.ndarray,
        show_grid:        bool = False,
        record_window=None,
        grid_size:        int  = 700,
        parent=None,
    ):
        super().__init__(parent)
        self.cup_state_buffer = cup_state_buffer
        self.display_manager  = display_manager
        self.image_background = image_background
        self.show_grid        = show_grid
        self.record_window    = record_window
        self.grid_size        = grid_size
        self.running          = False

        # Homographie table → projecteur (même fichier que test_projection_blanc)
        h_data   = json.load(open(config_path("H_table_to_proj.json"), "r"))
        self._H  = np.array(h_data["H_table_to_proj"], dtype=np.float32)

        # Frame pré-allouée — réutilisée chaque frame, jamais recréée
        self._proj_frame = np.ones(
            (PROJ_HEIGHT, PROJ_WIDTH, 3), dtype=np.uint8) * 255

        # EMA par marker_id — créé à la première apparition
        self._ema_filters: Dict[int, _EMAFilter] = {}

        self._fps_count = 0
        self._fps_t0    = 0.0

    # ── API publique ──────────────────────────────────────────────────────────

    def set_show_grid(self, show: bool) -> None:
        self.show_grid = bool(show)

    def update_background(self, new_bg: np.ndarray) -> None:
        self.image_background = new_bg

    def stop(self) -> None:
        self.running = False

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self) -> None:
        self.running    = True
        self._fps_count = 0
        self._fps_t0    = time.monotonic()
        frame_duration  = 1.0 / TARGET_FPS

        print(f"[Projection] Thread démarré  cible={TARGET_FPS}fps  "
              f"{PROJ_WIDTH}x{PROJ_HEIGHT}")

        while self.running:
            t_start = time.monotonic()
            cups    = self.cup_state_buffer.get_all()

            # ── Rendu — frame réinitialisée blanc ────────────────────
            self._proj_frame[:] = 255

            data_out = []
            for marker_id, cup in cups.items():
                pos_raw = cup["last_pos"]

                # EMA sur les mm — supprime les micro-tremblements
                if marker_id not in self._ema_filters:
                    self._ema_filters[marker_id] = _EMAFilter()
                pos_smooth = self._ema_filters[marker_id].update(
                    pos_raw[0], pos_raw[1])

                # mm → pixel projecteur via H_table_to_proj
                proj_x, proj_y = self._mm_to_proj(pos_smooth)

                state = cup.get("state", "POSEE")
                self._draw_ring(self._proj_frame, proj_x, proj_y, state)

                data_out.append((marker_id, list(pos_smooth), state))

            # ── Grille optionnelle ────────────────────────────────────
            if self.show_grid and self.record_window is not None:
                grid = self._build_warped_grid(cups)
                if grid is not None:
                    cv2.addWeighted(grid, 1.0, self._proj_frame, 1.0, 0,
                                    self._proj_frame)

            # ── Envoi projecteur (bloquant ici — pas de thread séparé)
            self.display_manager.display_image_on_projector_monitor(
                self._proj_frame)

            # ── Nettoyage EMA pour tasses disparues ──────────────────
            active_ids = set(cups.keys())
            for mid in list(self._ema_filters.keys()):
                if mid not in active_ids:
                    del self._ema_filters[mid]

            # ── Signal UI ────────────────────────────────────────────
            self.data_signal.emit({"data": data_out})

            # ── FPS ──────────────────────────────────────────────────
            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                fps = self._fps_count / (now - self._fps_t0)
                self.fps_signal.emit(fps)
                print(f"[Projection] FPS={fps:.1f}  cups={len(cups)}")
                self._fps_count = 0
                self._fps_t0    = now

            # ── Throttle ─────────────────────────────────────────────
            elapsed = time.monotonic() - t_start
            sleep   = frame_duration - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # Éteindre le projecteur proprement
        self._proj_frame[:] = 0
        self.display_manager.display_image_on_projector_monitor(
            self._proj_frame)
        print("[Projection] Thread arrêté")

    # ── Géométrie ─────────────────────────────────────────────────────────────

    def _mm_to_proj(self, pos_mm: Tuple[float, float]) -> Tuple[int, int]:
        """
        (x_mm, y_mm) repère table → pixel projecteur.
        Exactement comme test_projection_blanc.py.
        """
        x_mm, y_mm = pos_mm
        pt   = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, self._H)
        return int(proj[0, 0, 0]), int(proj[0, 0, 1])

    # ── Dessin ────────────────────────────────────────────────────────────────

    def _draw_ring(self, img: np.ndarray,
                   proj_x: int, proj_y: int,
                   state: str) -> None:
        color = (COLOR_SOULEVEE           if state == "SOULEVEE"
                 else COLOR_PEUT_ETRE_SOULEVEE if state == "PEUT_ETRE_SOULEVEE"
                 else COLOR_POSEE)
        m = RING_RADIUS_PX + RING_THICKNESS_PX
        if m <= proj_x <= PROJ_WIDTH - m and m <= proj_y <= PROJ_HEIGHT - m:
            cv2.circle(img, (proj_x, proj_y),
                       RING_RADIUS_PX, color, RING_THICKNESS_PX,
                       lineType=cv2.LINE_AA)

    # ── Grille optionnelle ────────────────────────────────────────────────────

    def _build_warped_grid(self, cups: dict) -> Optional[np.ndarray]:
        try:
            from src.core.projection.draw_utils import DrawUtils
            bounds = self.record_window.get_bounds_from_inputs()
            if bounds is None:
                return None
            x_min, x_max, y_min, y_max, x_leg, y_leg = bounds
            TABLE_SIZE_MM = 597.0
            graph = np.full(
                (self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)
            graph = DrawUtils.draw_math_grid_on_image(
                graph, x_min, x_max, y_min, y_max, x_leg, y_leg,
                self.grid_size)
            for marker_id, cup in cups.items():
                x_mm, y_mm = cup["last_pos"]
                xg = int((x_mm / TABLE_SIZE_MM) * self.grid_size)
                yg = int((y_mm / TABLE_SIZE_MM) * self.grid_size)
                if 0 <= xg < self.grid_size and 0 <= yg < self.grid_size:
                    state = cup.get("state", "POSEE")
                    color = (COLOR_SOULEVEE if state == "SOULEVEE"
                             else COLOR_PEUT_ETRE_SOULEVEE
                             if state == "PEUT_ETRE_SOULEVEE"
                             else COLOR_POSEE)
                    cv2.circle(graph, (xg, yg), 14, color, -1)
            return cv2.warpPerspective(
                graph, self._H, (PROJ_WIDTH, PROJ_HEIGHT))
        except Exception as e:
            print(f"[Projection] Erreur grille: {e}")
            return None