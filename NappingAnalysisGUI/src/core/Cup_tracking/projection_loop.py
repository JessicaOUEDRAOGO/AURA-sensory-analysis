# -*- coding: utf-8 -*-
"""
projection_loop.py
==================
Boucle dédiée à la projection des cercles sur le projecteur.

Responsabilités :
  - Lire CupStateBuffer (états des tasses)
  - Calculer les positions projecteur via H_table_to_proj
  - Dessiner les cercles colorés selon l'état de chaque tasse
  - Envoyer l'image au DisplayManager
  - Émettre les données pour l'UI (signal data_signal)

Ce module ne sait rien des caméras. Il lit uniquement CupStateBuffer.

Couleurs :
  POSEE             → Rouge   (0, 0, 255)
  PEUT_ETRE_SOULEVEE → Orange  (0, 165, 255)
  SOULEVEE          → Bleu    (255, 0, 0)

FPS cible : ~30fps (limité par le DisplayManager / projecteur, pas par les caméras)
"""

import cv2
import json
import numpy as np
import time
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.utils.paths import config_path
from src.core.projection.display_manager import DisplayManager


# ─────────────────────────────────────────────────────────────────────────────
# Constantes visuelles
# ─────────────────────────────────────────────────────────────────────────────

COLOR_POSEE             = (0,   0, 255)    # Rouge
COLOR_PEUT_ETRE_SOULEVEE = (0, 165, 255)   # Orange
COLOR_SOULEVEE          = (255,  0,   0)   # Bleu

RING_RADIUS_PX    = 100      # Rayon du cercle projeté en pixels projecteur
RING_THICKNESS_PX = 10      # Épaisseur du cercle
RING_ALPHA        = 0.7     # Opacité du cercle (blend avec background)

PROJ_WIDTH  = 3840
PROJ_HEIGHT = 2160

TARGET_FPS = 30.0   # FPS cible de la boucle de projection
TARGET_DISPLAY_FPS = 15.0  # FPS max du DisplayManager / projecteur (limite réelle observée)

class ProjectionLoop(QThread):
    """
    Thread de projection — lit CupStateBuffer et affiche sur projecteur.

    Signaux :
      data_signal(dict)     : {"data": [(marker_id, pos, state), ...]}
                              émis chaque frame pour l'UI (RecordWindow)
      fps_signal(float)     : FPS de la boucle projection
    """

    data_signal = pyqtSignal(dict)
    fps_signal  = pyqtSignal(float)

    def __init__(
        self,
        cup_state_buffer,
        display_manager: DisplayManager,
        image_background: np.ndarray,
        show_grid: bool = False,
        record_window=None,    # pour _build_warped_grid si grille activée
        grid_size: int = 700,
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

        # Chargement homographie table → projecteur
        h_data = json.load(open(config_path("H_table_to_proj.json"), "r"))
        self._H = np.array(h_data["H_table_to_proj"], dtype=np.float32)

        self._fps_count = 0
        self._fps_t0    = 0.0

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def set_show_grid(self, show: bool) -> None:
        self.show_grid = bool(show)

    def update_background(self, new_bg: np.ndarray) -> None:
        self.image_background = new_bg

    def stop(self) -> None:
        self.running = False

    # ──────────────────────────────────────────────────────────────────
    # Boucle principale
    # ──────────────────────────────────────────────────────────────────


    def run(self) -> None:
        self.running        = True
        self._fps_count     = 0
        self._fps_t0        = time.monotonic()
        self._last_display  = 0.0          # timestamp dernier affichage projecteur
        frame_duration      = 1.0 / TARGET_FPS
        display_interval    = 1.0 / TARGET_DISPLAY_FPS

        # Buffer pré-alloué
        self._proj_frame = np.ones((PROJ_HEIGHT, PROJ_WIDTH, 3), dtype=np.uint8) * 255

        print(f"[Projection] Thread démarré  data={TARGET_FPS}fps  "
            f"display={TARGET_DISPLAY_FPS}fps  {PROJ_WIDTH}x{PROJ_HEIGHT}")

        while self.running:
            t_start = time.monotonic()
            cups    = self.cup_state_buffer.get_all()

            # ── Toujours émettre data_signal à 30fps (pour l'UI Qt) ──────
            data_out = [
                (mid, cup["last_pos"], cup.get("state", "POSEE"))
                for mid, cup in cups.items()
            ]
            self.data_signal.emit({"data": data_out})

            # ── Affichage projecteur throttlé à 15fps ────────────────────
            now = time.monotonic()
            if now - self._last_display >= display_interval:
                self._last_display = now
                self._proj_frame[:] = 255

                for marker_id, cup in cups.items():
                    proj_x, proj_y = self._table_to_projector(cup["last_pos"])
                    self._draw_cup_ring(self._proj_frame, proj_x, proj_y, cup)

                if self.show_grid and self.record_window is not None:
                    grid = self._build_warped_grid(cups)
                    if grid is not None:
                        cv2.addWeighted(grid, 1.0, self._proj_frame, 1.0, 0,
                                        self._proj_frame)

                self.display_manager.display_image_on_projector_monitor(self._proj_frame)

            # ── FPS data (pas display) ────────────────────────────────────
            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_t0 >= 1.0:
                fps = self._fps_count / (now - self._fps_t0)
                self.fps_signal.emit(fps)
                print(f"[Projection] data_fps={fps:.1f}  cups={len(cups)}")
                self._fps_count = 0
                self._fps_t0    = now

            # ── Throttle boucle data ──────────────────────────────────────
            elapsed = time.monotonic() - t_start
            sleep   = frame_duration - elapsed
            if sleep > 0:
                time.sleep(sleep)

        print("[Projection] Thread arrêté")

    # ──────────────────────────────────────────────────────────────────
    # Géométrie
    # ──────────────────────────────────────────────────────────────────

    def _table_to_projector(self, pos_mm: list) -> tuple:
        """Convertit (x_mm, y_mm) repère table → pixel projecteur."""
        x_mm, y_mm = pos_mm
        pt   = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, self._H)
        return int(proj[0, 0, 0]), int(proj[0, 0, 1])

    # ──────────────────────────────────────────────────────────────────
    # Dessin
    # ──────────────────────────────────────────────────────────────────

    def _draw_cup_ring(self, img, proj_x, proj_y, cup):
        state = cup.get("state", "POSEE")
        color = (COLOR_SOULEVEE if state == "SOULEVEE"
                else COLOR_PEUT_ETRE_SOULEVEE if state == "PEUT_ETRE_SOULEVEE"
                else COLOR_POSEE)
        cv2.circle(img, (proj_x, proj_y),
                RING_RADIUS_PX, color, RING_THICKNESS_PX,
                lineType=cv2.LINE_AA)
    # ──────────────────────────────────────────────────────────────────
    # Grille optionnelle
    # ──────────────────────────────────────────────────────────────────

    def _build_warped_grid(self, cups: dict) -> np.ndarray:
        """Construit la grille mathématique et la warpe vers l'espace projecteur."""
        try:
            from src.core.projection.draw_utils import DrawUtils

            bounds = self.record_window.get_bounds_from_inputs()
            if bounds is None:
                return None
            x_min, x_max, y_min, y_max, x_leg, y_leg = bounds

            TABLE_SIZE_MM = 597.0
            graph = np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)
            graph = DrawUtils.draw_math_grid_on_image(
                graph, x_min, x_max, y_min, y_max, x_leg, y_leg, self.grid_size
            )

            # Dessiner les tasses sur la grille
            RADIUS_GRAPH = 14
            for marker_id, cup in cups.items():
                x_mm, y_mm = cup["last_pos"]
                xg = int((x_mm / TABLE_SIZE_MM) * self.grid_size)
                yg = int((y_mm / TABLE_SIZE_MM) * self.grid_size)
                if 0 <= xg < self.grid_size and 0 <= yg < self.grid_size:
                    state = cup.get("state", "POSEE")
                    color = (COLOR_SOULEVEE if state == "SOULEVEE"
                             else COLOR_PEUT_ETRE_SOULEVEE if state == "PEUT_ETRE_SOULEVEE"
                             else COLOR_POSEE)
                    cv2.circle(graph, (xg, yg), RADIUS_GRAPH, color, -1)

            return cv2.warpPerspective(graph, self._H, (PROJ_WIDTH, PROJ_HEIGHT))

        except Exception as e:
            print(f"[Projection] Erreur grille: {e}")
            return None
