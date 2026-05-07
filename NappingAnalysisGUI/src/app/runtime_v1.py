# -*- coding: utf-8 -*-
"""
algorithm_analysis.py — VERSION v5
====================================
Refonte complète — architecture 3 threads séparés.

Supprimé vs v4 :
  - Toute logique CSRT / mediapipe / mains
  - HandStateBuffer, FrameBuffer, HandTrackingThread
  - detect_and_process() monolithique (une seule boucle = 5fps)
  - CupTopTracker (CSRT)

Nouveau :
  - CupStateBuffer  : état partagé thread-safe
  - CamBottomThread : détection ArUco uniquement (~30fps)
  - CamTopThread    : KCF tracking uniquement (~30fps)
  - ProjectionLoop  : dessin + projecteur uniquement (~30fps)

Chaque thread tourne indépendamment et communique UNIQUEMENT via CupStateBuffer.
L'Algorithm_Analysis devient un simple orchestrateur : il démarre/arrête les threads.

Données émises vers l'UI :
  data_signal(dict) → relayé depuis ProjectionLoop.data_signal
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import time as pytime
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal, QThread

from src.core.utils.paths import config_path, data_path
from src.core.projection.display_manager import DisplayManager

# Nouveaux modules
from src.core.cup_tracking.cup_state_buffer  import CupStateBuffer
from src.core.cup_tracking.cam_bottom_thread import CamBottomThread
from src.core.cup_tracking.cam_top_thread    import CamTopThread
from src.core.cup_tracking.projection_loop   import ProjectionLoop


class Algorithm_Analysis(QObject):
    """
    Orchestrateur des 3 threads de traitement.

    Interface publique inchangée pour RecordWindow :
      - start() / stop()
      - data_signal
      - finished_signal
      - set_show_grid()
      - update_background_image()
      - state_popUpCamera_changed()
    """

    data_signal     = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_manager: DisplayManager,
        image_background: np.ndarray,
        record_window=None,
        output_dir: str = None,
        output_name: str = "data",
        modules_enabled: dict = None,
        assets=None,
        timeline_steps=None,
        protocol=None,
        grid_size: int = 700,
        **kwargs,   # absorbe hand_buffer / frame_buffer pour compatibilité
    ):
        super().__init__()

        self.parent           = parent
        self.display_manager  = display_manager
        self.record_window    = record_window
        self.grid_size        = int(grid_size)
        self.modules_enabled  = modules_enabled or {}
        self.assets           = assets or []
        self.timeline_steps   = timeline_steps or []
        self.protocol         = protocol
        self.running          = False

        # Validation background
        self.image_background = self._validate_bg(image_background)

        # Output CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            os.makedirs(data_path(), exist_ok=True)
            output_dir = data_path()
        else:
            os.makedirs(output_dir, exist_ok=True)
        self.output_csv  = os.path.join(output_dir, f"{output_name}_{timestamp}.csv")
        self.data_buffer = []

        # Buffer partagé entre les 3 threads
        self._cup_buffer = CupStateBuffer()

        # Threads (créés dans start())
        self._cam_bottom: CamBottomThread | None = None
        self._cam_top:    CamTopThread    | None = None
        self._projection: ProjectionLoop  | None = None

        # Caméra bottom (déjà gérée par parent.camera_manager)
        self._camera_manager = getattr(parent, "camera_manager", None)

        # Config
        self._pose_bottom_path = str(config_path("cambottom_table_pose.json"))
        self._pose_top_path    = str(config_path("camtop_table_pose.json"))

        # État preview caméra (ancienne feature conservée)
        self.status_popUpCamera = False
        self._camera_was_active = False

        print("[Algo v5] Initialisé")

    # ──────────────────────────────────────────────────────────────────
    # Compatibilité RecordWindow (no-op)
    # ──────────────────────────────────────────────────────────────────

    def set_hands_provider(self, func):
        """Conservé pour compatibilité. Sans effet en v5."""
        pass

    def on_consigne_key_pressed(self):
        pass

    def state_popUpCamera_changed(self):
        self.status_popUpCamera = not self.status_popUpCamera
        if self._cam_bottom:
            self._cam_bottom.show_preview = self.status_popUpCamera
        if self._cam_top:
            self._cam_top.show_preview = self.status_popUpCamera

    # ──────────────────────────────────────────────────────────────────
    # API principale
    # ──────────────────────────────────────────────────────────────────

    def detect_and_process(self) -> None:
        """
        Point d'entrée appelé par RecordWindow dans un QThread.
        Démarre les 3 threads fils et attend leur fin.
        """
        self.running = True
        print("[Algo v5] Démarrage des 3 threads")

        # ── CamBottomThread ──────────────────────────────────────────
        self._cam_bottom = CamBottomThread(
            cup_state_buffer=self._cup_buffer,
            camera_manager=self._camera_manager,
            pose_path=self._pose_bottom_path,
            show_preview=self.status_popUpCamera,
        )
        # Surcharger K si undistort actif
        runtime_K = getattr(self.parent, "runtime_K", None)
        if runtime_K is not None:
            self._cam_bottom.set_camera_matrix(runtime_K)

        # ── CamTopThread ─────────────────────────────────────────────
        cam_top_id = getattr(
            getattr(self.parent, "parent", None), "camera_top_id", 0
        )
        self._cam_top = CamTopThread(
            cup_state_buffer=self._cup_buffer,
            camera_index=cam_top_id,
            pose_path=self._pose_top_path,
            show_preview=self.status_popUpCamera,
        )

        # ── ProjectionLoop ───────────────────────────────────────────
        self._projection = ProjectionLoop(
            cup_state_buffer=self._cup_buffer,
            display_manager=self.display_manager,
            image_background=self.image_background,
            show_grid=False,
            record_window=self.record_window,
            grid_size=self.grid_size,
        )

        # Relayer data_signal depuis ProjectionLoop → UI
        self._projection.data_signal.connect(self._on_projection_data)

        # ── Démarrage ────────────────────────────────────────────────
        self._cam_bottom.start()
        self._cam_top.start()
        self._projection.start()

        print("[Algo v5] 3 threads démarrés")

        # ── Boucle de garde (très légère) ────────────────────────────
        # Ce thread reste vivant pour pouvoir recevoir stop()
        # et sauvegarder le CSV à la fin.
        while self.running:
            pytime.sleep(0.1)

        # ── Arrêt ordonné ────────────────────────────────────────────
        print("[Algo v5] Arrêt en cours...")

        if self._projection:
            self._projection.stop()
            self._projection.wait(2000)

        if self._cam_bottom:
            self._cam_bottom.stop()
            self._cam_bottom.wait(2000)

        if self._cam_top:
            self._cam_top.stop()
            self._cam_top.wait(2000)

        self._save_csv()
        print("[Algo v5] Tous les threads arrêtés")
        self.finished_signal.emit()

    def stop(self) -> None:
        print("[Algo v5] STOP demandé")
        self.running = False

    # ──────────────────────────────────────────────────────────────────
    # Réception données depuis ProjectionLoop
    # ──────────────────────────────────────────────────────────────────

    def _on_projection_data(self, data: dict) -> None:
        """Relaye les données vers RecordWindow ET sauvegarde dans le buffer CSV."""
        self.data_signal.emit(data)
        self._save_to_buffer(data.get("data", []))

    # ──────────────────────────────────────────────────────────────────
    # Grille / background
    # ──────────────────────────────────────────────────────────────────

    def set_show_grid(self, show: bool) -> None:
        if self._projection:
            self._projection.set_show_grid(show)

    def update_background_image(self, new_bg: np.ndarray) -> None:
        new_bg = self._validate_bg(new_bg)
        self.image_background = new_bg
        if self._projection:
            self._projection.update_background(new_bg)

    # ──────────────────────────────────────────────────────────────────
    # CSV
    # ──────────────────────────────────────────────────────────────────

    def _save_to_buffer(self, graph_coords: list) -> None:
        frame_data = {
            "frame":     len(self.data_buffer) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for entry in graph_coords:
            marker_id, pos, *_ = entry
            frame_data[f"ID_{marker_id}_x"] = float(pos[0])
            frame_data[f"ID_{marker_id}_y"] = float(pos[1])
        self.data_buffer.append(frame_data)

    def _save_csv(self) -> None:
        if not self.data_buffer:
            return
        pd.DataFrame(self.data_buffer).to_csv(self.output_csv, index=False)
        print(f"[Algo v5] CSV sauvegardé : {self.output_csv}")

    # ──────────────────────────────────────────────────────────────────
    # Utilitaire
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_bg(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image_background invalide : type={type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image_background invalide : shape={image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def is_enabled(self, key: str, default: bool = False) -> bool:
        v = self.modules_enabled.get(key, default)
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
