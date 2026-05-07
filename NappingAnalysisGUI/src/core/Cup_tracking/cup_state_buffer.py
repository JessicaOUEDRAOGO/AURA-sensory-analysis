# -*- coding: utf-8 -*-
import threading
import copy
from typing import Dict, List, Optional


class CupStateBuffer:

    N_LIFT_CONFIRM = 8   # frames sans ArUco → SOULEVEE (~320ms à 25fps)
    N_POSE_CONFIRM = 5   # frames avec ArUco → retour POSEE (~200ms)

    def __init__(self):
        self._lock = threading.Lock()
        self._cups: Dict[int, dict] = {}

    def update_from_bottom(self, detected: Dict[int, List[float]]) -> None:
        with self._lock:

            # ── Marqueurs détectés cette frame ───────────────────────
            for marker_id, pos in detected.items():
                if marker_id not in self._cups:
                    self._cups[marker_id] = self._new_cup(pos)
                    continue

                cup = self._cups[marker_id]
                cup["lost_frames"] = 0

                if cup["state"] == "SOULEVEE":
                    # KCF actif → ArUco voit la tasse revenir
                    # On bloque le KCF et on compte les frames de confirmation
                    cup["kcf_active"]  = True   # stoppe get_cup_to_track()
                    cup["pose_frames"] = cup.get("pose_frames", 0) + 1
                    cup["lift_frames"] = 0      # ← reset lift pour prochain soulèvement
                    self._smooth_pos(cup, pos)

                    if cup["pose_frames"] >= self.N_POSE_CONFIRM:
                        # Retour au sol confirmé — reset complet
                        cup["state"]         = "POSEE"
                        cup["pose_frames"]   = 0
                        cup["kcf_active"]    = False
                        cup["pos_is_top_mm"] = False
                        cup["last_pos"]      = list(pos)   # snap propre

                elif cup["state"] == "PEUT_ETRE_SOULEVEE":
                    # Fausse alerte — la tasse n'était pas vraiment soulevée
                    # Retour immédiat à POSEE, pas besoin de confirmation
                    cup["state"]         = "POSEE"
                    cup["lift_frames"]   = 0
                    cup["pose_frames"]   = 0
                    cup["kcf_active"]    = False   # pas de KCF actif dans cet état
                    cup["pos_is_top_mm"] = False
                    self._smooth_pos(cup, pos)

                else:
                    # POSEE normale
                    cup["lift_frames"]   = 0
                    cup["pose_frames"]   = 0
                    cup["pos_is_top_mm"] = False
                    self._smooth_pos(cup, pos)

            # ── Marqueurs non détectés cette frame ───────────────────
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
                        cup["state"]      = "SOULEVEE"
                        cup["kcf_active"] = False   # KCF peut démarrer

                # SOULEVEE → KCF gère, on ne touche rien

            # ── Max 1 tasse soulevée à la fois ────────────────────────
            soulevees = [mid for mid, c in self._cups.items()
                         if c["state"] == "SOULEVEE"]
            if len(soulevees) > 1:
                soulevees.sort(
                    key=lambda mid: self._cups[mid].get("lift_frames", 0),
                    reverse=True
                )
                for mid in soulevees[1:]:
                    self._cups[mid]["state"]         = "POSEE"
                    self._cups[mid]["kcf_active"]    = False
                    self._cups[mid]["pos_is_top_mm"] = False

    @staticmethod
    def _smooth_pos(cup: dict, new_pos: List[float]) -> None:
        """Lissage EMA — évite les sauts visuels sur la projection."""
        alpha = 0.4
        old = cup["last_pos"]
        cup["last_pos"] = [
            alpha * new_pos[0] + (1 - alpha) * old[0],
            alpha * new_pos[1] + (1 - alpha) * old[1],
        ]
        cup["last_aruco_pos"] = list(new_pos)   # position brute pour init KCF

    # ──────────────────────────────────────────────────────────────────
    # API cam_top
    # ──────────────────────────────────────────────────────────────────

    def get_cup_to_track(self) -> Optional[tuple]:
        with self._lock:
            for marker_id, cup in self._cups.items():
                if (cup["state"] == "SOULEVEE"
                        and not cup.get("kcf_active", False)):
                    init_pos = cup.get("last_aruco_pos", cup["last_pos"])
                    return marker_id, list(init_pos)
            return None

    def get_last_aruco_pos(self, marker_id: int) -> Optional[List[float]]:
        with self._lock:
            cup = self._cups.get(marker_id)
            if cup is None:
                return None
            return list(cup.get("last_aruco_pos", cup["last_pos"]))

    def set_kcf_active(self, marker_id: int, active: bool) -> None:
        with self._lock:
            if marker_id in self._cups:
                self._cups[marker_id]["kcf_active"] = active

    def update_from_top(self, marker_id: int, pos_mm: List[float]) -> None:
        with self._lock:
            if marker_id in self._cups:
                cup = self._cups[marker_id]
                if cup["state"] == "SOULEVEE":
                    cup["last_pos"]      = list(pos_mm)
                    cup["pos_is_top_mm"] = True

    def is_kcf_needed(self, marker_id: int) -> bool:
        with self._lock:
            cup = self._cups.get(marker_id)
            if cup is None:
                return False
            return cup["state"] == "SOULEVEE"

    def kcf_stopped(self, marker_id: int) -> None:
        with self._lock:
            if marker_id in self._cups:
                self._cups[marker_id]["kcf_active"] = False

    # ──────────────────────────────────────────────────────────────────
    # API lecture
    # ──────────────────────────────────────────────────────────────────

    def get_all(self) -> Dict[int, dict]:
        with self._lock:
            return copy.deepcopy(self._cups)

    def get_cup(self, marker_id: int) -> Optional[dict]:
        with self._lock:
            cup = self._cups.get(marker_id)
            return copy.deepcopy(cup) if cup else None

    @staticmethod
    def _new_cup(pos: List[float]) -> dict:
        return {
            "state":          "POSEE",
            "last_pos":       list(pos),
            "last_aruco_pos": list(pos),
            "pos_is_top_mm":  False,
            "lost_frames":    0,
            "lift_frames":    0,
            "pose_frames":    0,
            "kcf_active":     False,
        }