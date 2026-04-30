# -*- coding: utf-8 -*-
"""
hand_state_buffer.py — Buffer partagé thread-safe pour les données de mains
============================================================================

Remplace la communication directe entre HandTrackingThread et Algorithm_Analysis.

Principe :
  - HandTrackingThread écrit dans HandStateBuffer à chaque nouvelle détection
  - Algorithm_Analysis lit le dernier snapshot via get_fresh_snapshot()
  - L'association ne se déclenche QUE si ts_ms a changé depuis la dernière lecture
  - Un snapshot est "valide" si son âge < MAX_AGE_MS (configurable, défaut 200 ms)
    → tolérant aux bursts de latence MediaPipe sans bloquer sur chaque frame

Usage dans HandTrackingThread.run() :
    self.buffer.push(hands_data)          # remplace self.hands_signal.emit(...)

Usage dans Algorithm_Analysis :
    snapshot = self.hand_buffer.get_fresh_snapshot()
    if snapshot is not None:
        self.associate_hands_to_cups(snapshot.hands)
"""

import threading
import time as pytime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HandSnapshot:
    hands:   list          # liste de dicts {id, x, y, vx, vy, ts_ms}
    ts_ms:   int           # timestamp de la détection MediaPipe
    wall_ms: int           # timestamp d'écriture dans le buffer (monotone)


class HandStateBuffer:
    """
    Buffer single-slot thread-safe.

    Seul le snapshot le plus récent est conservé — pas de file d'attente.
    Algorithm_Analysis ne "rate" jamais une frame utile car cam_bottom
    tourne à la même fréquence que cam_top.
    """

    def __init__(self, max_age_ms: int = 200):
        """
        max_age_ms : âge maximal accepté pour un snapshot (ms).
                     Au-delà, get_fresh_snapshot() retourne None et les
                     locks sont relâchés proprement dans Algorithm_Analysis.
        """
        self._lock       = threading.Lock()
        self._snapshot:  Optional[HandSnapshot] = None
        self._last_read_ts: int = -1        # ts_ms du dernier snapshot consommé
        self.max_age_ms  = max_age_ms

    # ------------------------------------------------------------------
    # Écriture (thread HandTracking)
    # ------------------------------------------------------------------
    def push(self, hands: list) -> None:
        """
        Appelé par HandTrackingThread à chaque frame.
        hands : liste produite par HandSlotManager.update()  +  ts_ms injecté
        """
        if not hands:
            ts = int(pytime.time() * 1000)
            snap = HandSnapshot(hands=[], ts_ms=ts, wall_ms=ts)
        else:
            ts   = max(h.get("ts_ms", 0) for h in hands)
            snap = HandSnapshot(hands=hands, ts_ms=ts, wall_ms=int(pytime.time() * 1000))

        with self._lock:
            self._snapshot = snap

    # ------------------------------------------------------------------
    # Lecture (thread Algorithm_Analysis)
    # ------------------------------------------------------------------
    def get_fresh_snapshot(self) -> Optional[HandSnapshot]:
        """
        Retourne le snapshot si :
          1. Il existe
          2. Son ts_ms est différent du dernier consommé  ← gate principale
          3. Son âge (wall_ms) est < max_age_ms           ← garde fraîcheur

        Retourne None sinon → Algorithm_Analysis ne doit PAS appeler associate().
        Marque automatiquement le snapshot comme consommé.
        """
        now = int(pytime.time() * 1000)
        with self._lock:
            snap = self._snapshot
            if snap is None:
                return None
            age = now - snap.wall_ms
            if age > self.max_age_ms:
                return None
            if snap.ts_ms == self._last_read_ts:
                return None          # déjà traité
            self._last_read_ts = snap.ts_ms
            return snap

    def get_latest(self) -> Optional[HandSnapshot]:
        """
        Retourne le snapshot le plus récent SANS vérifier s'il a déjà été lu.
        Utile pour l'affichage (pas pour l'association).
        """
        with self._lock:
            return self._snapshot

    def clear(self) -> None:
        with self._lock:
            self._snapshot      = None
            self._last_read_ts  = -1

    @property
    def last_age_ms(self) -> int:
        """Âge du snapshot courant en ms (0 si aucun)."""
        now = int(pytime.time() * 1000)
        with self._lock:
            if self._snapshot is None:
                return 9999
            return now - self._snapshot.wall_ms
