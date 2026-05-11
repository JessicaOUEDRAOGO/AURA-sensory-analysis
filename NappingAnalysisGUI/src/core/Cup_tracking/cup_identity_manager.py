# -*- coding: utf-8 -*-
"""
cup_identity_manager.py
=======================
Gère l'association marker_id ArUco ↔ TrackedCup KCF.

Rôle :
  - Initialisation : associe chaque bbox HSV au marker_id ArUco le plus proche
  - Surveillance : détecte les tags absents (N frames consécutives)
  - Réassociation : quand un tag réapparaît, vérifie que c'est le bon
    (une seule tasse bougée à la fois → le tag qui réapparaît = celui qui avait disparu)

Thread-safe : toutes les méthodes publiques sont protégées par un Lock.
"""

import threading
import time
from typing import Dict, Optional, Set, Tuple


# Nombre de frames consécutives sans détection ArUco avant de déclarer un tag absent
ABSENT_FRAMES_THRESHOLD = 15   # ~600ms à 25fps


class CupIdentityManager:
    """
    Maintient la table d'association {marker_id: cup_id_KCF}.

    cup_id_KCF : ID interne du TrackingManager (entier ≥ 0)
    marker_id  : ID ArUco (entier ≥ 0)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Association principale
        self._marker_to_cup:  Dict[int, int] = {}   # marker_id → cup_id KCF
        self._cup_to_marker:  Dict[int, int] = {}   # cup_id KCF → marker_id

        # Surveillance absences
        self._absent_frames:  Dict[int, int] = {}   # marker_id → nb frames absent
        self._absent_markers: Set[int]        = set()  # tags déclarés absents

        # Dernières positions ArUco connues (mm)
        self._last_aruco_pos: Dict[int, Tuple[float, float]] = {}

        print("[Identity] CupIdentityManager initialisé")

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self,
                   aruco_positions: Dict[int, Tuple[float, float]],
                   cup_positions_mm: Dict[int, Tuple[float, float]]) -> None:
        """
        Association initiale ArUco ↔ cups KCF par proximité mm.

        aruco_positions  : {marker_id: (x_mm, y_mm)} depuis cam_bottom
        cup_positions_mm : {cup_id: (x_mm, y_mm)}    depuis cam_top KCF
        """
        with self._lock:
            self._marker_to_cup.clear()
            self._cup_to_marker.clear()
            self._absent_frames.clear()
            self._absent_markers.clear()

            used_cups = set()
            for marker_id, (ax, ay) in aruco_positions.items():
                self._last_aruco_pos[marker_id] = (ax, ay)
                best_cup  = None
                best_dist = float('inf')
                for cup_id, (cx, cy) in cup_positions_mm.items():
                    if cup_id in used_cups:
                        continue
                    d = ((ax - cx)**2 + (ay - cy)**2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_cup  = cup_id

                if best_cup is not None and best_dist < 100.0:
                    self._marker_to_cup[marker_id] = best_cup
                    self._cup_to_marker[best_cup]  = marker_id
                    used_cups.add(best_cup)
                    print(f"[Identity] #{marker_id} ↔ cup_{best_cup} "
                          f"dist={best_dist:.0f}mm")
                else:
                    print(f"[Identity] #{marker_id} — aucune tasse proche "
                          f"(dist={best_dist:.0f}mm)")

    # ── Mise à jour ArUco ─────────────────────────────────────────────────────

    def update_aruco(self,
                     detected: Dict[int, Tuple[float, float]]) -> None:
        """
        Appelé à chaque frame de cam_bottom.
        Met à jour les positions et gère les absences.

        detected : {marker_id: (x_mm, y_mm)}
        """
        with self._lock:
            all_known = set(self._marker_to_cup.keys())

            for marker_id, pos in detected.items():
                self._last_aruco_pos[marker_id] = pos
                # Tag présent → reset compteur absence
                self._absent_frames[marker_id] = 0
                if marker_id in self._absent_markers:
                    self._absent_markers.discard(marker_id)

            # Tags non détectés cette frame
            for marker_id in all_known:
                if marker_id not in detected:
                    count = self._absent_frames.get(marker_id, 0) + 1
                    self._absent_frames[marker_id] = count
                    if count >= ABSENT_FRAMES_THRESHOLD:
                        self._absent_markers.add(marker_id)

            # Réassociation si un seul tag était absent et réapparaît
            reappeared = {mid for mid in detected
                          if mid in self._absent_markers}
            if len(reappeared) == 1 and len(self._absent_markers) == 1:
                marker_id = next(iter(reappeared))
                self._absent_markers.discard(marker_id)
                self._absent_frames[marker_id] = 0
                print(f"[Identity] #{marker_id} réapparu — association maintenue")

    # ── Lecture ───────────────────────────────────────────────────────────────

    def get_marker_for_cup(self, cup_id: int) -> Optional[int]:
        with self._lock:
            return self._cup_to_marker.get(cup_id)

    def get_cup_for_marker(self, marker_id: int) -> Optional[int]:
        with self._lock:
            return self._marker_to_cup.get(marker_id)

    def get_last_aruco_pos(self, marker_id: int) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._last_aruco_pos.get(marker_id)

    def get_all_aruco_pos(self) -> Dict[int, Tuple[float, float]]:
        with self._lock:
            return dict(self._last_aruco_pos)

    def is_absent(self, marker_id: int) -> bool:
        with self._lock:
            return marker_id in self._absent_markers

    def get_associations(self) -> Dict[int, int]:
        """Retourne {marker_id: cup_id}."""
        with self._lock:
            return dict(self._marker_to_cup)

    def reset(self) -> None:
        with self._lock:
            self._marker_to_cup.clear()
            self._cup_to_marker.clear()
            self._absent_frames.clear()
            self._absent_markers.clear()
            self._last_aruco_pos.clear()
