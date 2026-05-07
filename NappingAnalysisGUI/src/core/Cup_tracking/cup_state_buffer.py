# -*- coding: utf-8 -*-
"""
cup_state_buffer.py
===================
Buffer thread-safe partagé entre :
  - CamBottomThread  → écrit les états ArUco détectés
  - CamTopThread     → écrit les positions KCF quand la tasse est levée
  - ProjectionLoop   → lit les états pour projeter les cercles

Structure d'une cup :
  {
    "state":         "POSEE" | "PEUT_ETRE_SOULEVEE" | "SOULEVEE",
    "last_pos":      [x_mm, y_mm],   # repère table
    "pos_is_top_mm": bool,           # True si la pos vient du KCF cam_top
    "lost_frames":   int,
    "lift_frames":   int,
    "pose_frames":   int,
  }
"""

import threading
import copy
from typing import Dict, List, Optional


class CupStateBuffer:
    """
    Accès thread-safe aux états des tasses.
    Toutes les méthodes sont protégées par un verrou.
    """

    N_LIFT_CONFIRM = 3   # frames sans ArUco avant → SOULEVEE
    N_POSE_CONFIRM = 6   # frames avec ArUco avant → retour POSEE

    def __init__(self):
        self._lock = threading.Lock()
        self._cups: Dict[int, dict] = {}

    # ──────────────────────────────────────────────────────────────────
    # API cam_bottom  (appelé depuis CamBottomThread)
    # ──────────────────────────────────────────────────────────────────

    def update_from_bottom(self, detected: Dict[int, List[float]]) -> None:
        """
        detected : {marker_id: [x_mm, y_mm]}
        Met à jour les états des tasses depuis les détections ArUco.
        """
        with self._lock:
            # Marqueurs détectés cette frame
            for marker_id, pos in detected.items():
                if marker_id not in self._cups:
                    self._cups[marker_id] = self._new_cup(pos)
                    continue

                cup = self._cups[marker_id]
                cup["lost_frames"] = 0

                if cup["state"] in ("SOULEVEE", "PEUT_ETRE_SOULEVEE"):
                    # ArUco réapparaît : on compte les frames de confirmation
                    cup["pose_frames"] = cup.get("pose_frames", 0) + 1
                    cup["last_pos"]    = list(pos)
                    cup["pos_is_top_mm"] = False
                    if cup["pose_frames"] >= self.N_POSE_CONFIRM:
                        # Retour confirmé au sol
                        cup["state"]       = "POSEE"
                        cup["lift_frames"] = 0
                        cup["pose_frames"] = 0
                else:
                    cup["last_pos"]      = list(pos)
                    cup["lift_frames"]   = 0
                    cup["pose_frames"]   = 0
                    cup["state"]         = "POSEE"
                    cup["pos_is_top_mm"] = False

            # Marqueurs non détectés cette frame
            for marker_id, cup in self._cups.items():
                if marker_id in detected:
                    continue
                cup["pose_frames"] = 0
                cup["lost_frames"] += 1

                if cup["state"] == "POSEE":
                    cup["state"]       = "PEUT_ETRE_SOULEVEE"
                    cup["lift_frames"] = 1
                elif cup["state"] == "PEUT_ETRE_SOULEVEE":
                    cup["lift_frames"] += 1
                    if cup["lift_frames"] >= self.N_LIFT_CONFIRM:
                        cup["state"] = "SOULEVEE"
                # Si déjà SOULEVEE, le KCF prend le relais → on ne touche pas

            # Règle : max 1 tasse soulevée à la fois
            soulevees = [mid for mid, c in self._cups.items()
                         if c["state"] == "SOULEVEE"]
            if len(soulevees) > 1:
                soulevees.sort(
                    key=lambda mid: self._cups[mid].get("lift_frames", 0),
                    reverse=True
                )
                for mid in soulevees[1:]:
                    self._cups[mid]["state"]        = "POSEE"
                    self._cups[mid]["pos_is_top_mm"] = False

    # ──────────────────────────────────────────────────────────────────
    # API cam_top  (appelé depuis CamTopThread)
    # ──────────────────────────────────────────────────────────────────

    def get_cup_to_track(self) -> Optional[tuple]:
        """
        Retourne (marker_id, last_pos_mm) de la tasse SOULEVEE
        dont le KCF doit être initialisé, ou None si aucune.
        Utilisé par CamTopThread pour savoir quand démarrer le tracking.
        """
        with self._lock:
            for marker_id, cup in self._cups.items():
                if (cup["state"] == "SOULEVEE"
                        and cup.get("lost_frames", 0) >= self.N_POSE_CONFIRM
                        and not cup.get("kcf_active", False)):
                    return marker_id, list(cup["last_pos"])
            return None

    def set_kcf_active(self, marker_id: int, active: bool) -> None:
        """CamTopThread appelle set_kcf_active(id, True) quand le tracker démarre."""
        with self._lock:
            if marker_id in self._cups:
                self._cups[marker_id]["kcf_active"] = active

    def update_from_top(self, marker_id: int, pos_mm: List[float]) -> None:
        """CamTopThread pousse la position KCF dans le buffer."""
        with self._lock:
            if marker_id in self._cups:
                cup = self._cups[marker_id]
                if cup["state"] == "SOULEVEE":
                    cup["last_pos"]      = list(pos_mm)
                    cup["pos_is_top_mm"] = True

    def is_kcf_needed(self, marker_id: int) -> bool:
        """Retourne True si la tasse est encore SOULEVEE (KCF doit continuer)."""
        with self._lock:
            cup = self._cups.get(marker_id)
            if cup is None:
                return False
            return cup["state"] == "SOULEVEE"

    def kcf_stopped(self, marker_id: int) -> None:
        """CamTopThread appelle ceci quand le tracker s'arrête."""
        with self._lock:
            if marker_id in self._cups:
                self._cups[marker_id]["kcf_active"] = False

    # ──────────────────────────────────────────────────────────────────
    # API lecture  (appelé depuis ProjectionLoop)
    # ──────────────────────────────────────────────────────────────────

    def get_all(self) -> Dict[int, dict]:
        """Retourne une copie profonde de tous les états tasses."""
        with self._lock:
            return copy.deepcopy(self._cups)

    def get_cup(self, marker_id: int) -> Optional[dict]:
        with self._lock:
            cup = self._cups.get(marker_id)
            return copy.deepcopy(cup) if cup else None

    # ──────────────────────────────────────────────────────────────────
    # Interne
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _new_cup(pos: List[float]) -> dict:
        return {
            "state":        "POSEE",
            "last_pos":     list(pos),
            "pos_is_top_mm": False,
            "lost_frames":  0,
            "lift_frames":  0,
            "pose_frames":  0,
            "kcf_active":   False,
        }
