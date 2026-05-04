# -*- coding: utf-8 -*-
"""
frame_buffer.py — Buffer partagé thread-safe pour les frames de la cam_top
===========================================================================

Même pattern que HandStateBuffer : single-slot, seule la frame la plus
récente est conservée.

Usage dans HandTrackingThread.run() :
    self.frame_buffer.push(frame)   # après cap.read()

Usage dans Algorithm_Analysis (detect_and_process) :
    frame_top = self.frame_buffer.get_latest()
    if frame_top is not None:
        ok, bbox, pos_mm = self.cup_tracker.update(frame_top)
"""

import threading
import time as pytime
from typing import Optional

import numpy as np


class FrameBuffer:
    """
    Buffer single-slot thread-safe pour les frames NumPy (BGR).

    - HandTrackingThread écrit via push() à chaque cap.read()
    - Algorithm_Analysis lit via get_latest() sans consommer
      (on veut toujours la frame la plus fraîche, pas une gate ts_ms)
    - max_age_ms : si la frame a plus de N ms, get_latest() retourne None
      → évite d'utiliser une frame périmée si la cam_top a crashé
    """

    def __init__(self, max_age_ms: int = 200):
        self._lock      = threading.Lock()
        self._frame:    Optional[np.ndarray] = None
        self._wall_ms:  int = 0
        self.max_age_ms = max_age_ms

    # ------------------------------------------------------------------
    # Écriture (thread HandTracking)
    # ------------------------------------------------------------------
    def push(self, frame: np.ndarray) -> None:
        """
        Appelé par HandTrackingThread après chaque cap.read() réussi.
        On copie la frame pour éviter que le thread de lecture ne lise
        une frame en cours d'écriture (le copy() est rapide, ~1 ms pour 1080p).
        """
        frame_copy = frame.copy()
        now = int(pytime.time() * 1000)
        with self._lock:
            self._frame   = frame_copy
            self._wall_ms = now

    # ------------------------------------------------------------------
    # Lecture (thread Algorithm_Analysis)
    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[np.ndarray]:
        """
        Retourne la frame la plus récente si son âge < max_age_ms.
        Retourne None si aucune frame disponible ou trop ancienne.

        Pas de gate ts_ms ici — on veut toujours la frame la plus fraîche
        disponible au moment de l'appel (CSRT.update() en a besoin à chaque
        frame de cam_bottom, pas seulement quand cam_top a produit du nouveau).
        """
        now = int(pytime.time() * 1000)
        with self._lock:
            if self._frame is None:
                return None
            if now - self._wall_ms > self.max_age_ms:
                return None
            return self._frame   # référence OK — on ne modifie pas la frame en dehors

    @property
    def last_age_ms(self) -> int:
        """Âge de la frame courante en ms (9999 si aucune)."""
        now = int(pytime.time() * 1000)
        with self._lock:
            if self._frame is None:
                return 9999
            return now - self._wall_ms

    def clear(self) -> None:
        with self._lock:
            self._frame   = None
            self._wall_ms = 0
