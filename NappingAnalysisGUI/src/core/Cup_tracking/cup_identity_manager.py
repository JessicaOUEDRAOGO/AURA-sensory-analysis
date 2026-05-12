# -*- coding: utf-8 -*-
"""
cup_identity_manager.py
========================
Gestionnaire d'identité des tasses.

Responsabilité unique : faire correspondre les IDs ArUco (cam_bottom)
aux trackers visuels KCF (cam_top) et conserver l'identité pendant
les phases de mouvement où la cam_bottom perd le tag.

ÉTATS D'UNE TASSE :
  MATCHED   → tag ArUco visible + tracker KCF actif + identité confirmée
  AIRBORNE  → tag ArUco perdu (tasse levée), tracker KCF conserve l'identité
  LOST      → tracker KCF perdu ET tag ArUco absent → identité inconnue
  PENDING   → tag ArUco visible mais pas encore associé à un tracker

MACHINE D'ÉTATS :
  MATCHED  ──[tag disparu]──► AIRBORNE  ──[tag réapparu + même zone]──► MATCHED
                                         ──[tag réapparu + mauvaise zone]──► MATCHED (correction ID)
                                         ──[tracker KCF perdu]──────────► LOST
  PENDING  ──[match position < seuil]──► MATCHED

THREAD SAFETY :
  update_aruco() est appelé depuis CamBottomThread (thread séparé).
  update_trackers() est appelé depuis CamTopThread (thread principal).
  Un verrou RLock protège toutes les structures internes.

PARAMETRES CLÉS :
  MATCH_DIST_MM   : distance max (mm) pour associer un tag ArUco à un tracker
  REPOSE_DIST_MM  : distance max pour valider la repose d'une tasse
  TAG_LOST_DELAY  : frames sans tag avant de passer en mode AIRBORNE
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Paramètres
# ──────────────────────────────────────────────────────────────────────────────

MATCH_DIST_MM   = 60.0    # Distance max pour le match initial tag ↔ tracker
REPOSE_DIST_MM  = 80.0    # Distance max pour valider la repose
TAG_LOST_DELAY  = 6       # Frames sans tag avant de passer AIRBORNE
TAG_FOUND_CONF  = 3       # Frames consécutives avec tag pour confirmer présence
ORPHAN_TTL_S    = 5.0     # Durée (s) avant suppression d'une identité orpheline


# ──────────────────────────────────────────────────────────────────────────────
#  États
# ──────────────────────────────────────────────────────────────────────────────

class CupState(Enum):
    MATCHED  = auto()   # tag visible + tracker actif, identité certaine
    AIRBORNE = auto()   # tasse levée, tag non visible, KCF conserve l'ID
    LOST     = auto()   # tracker perdu ET tag absent
    PENDING  = auto()   # tag visible mais tracker non encore associé


# ──────────────────────────────────────────────────────────────────────────────
#  Entrée d'identité
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CupIdentity:
    aruco_id:       int
    tracker_id:     Optional[int]          = None   # cup_id dans TrackingManager
    state:          CupState               = CupState.PENDING
    pos_mm:         Optional[Tuple[float, float]] = None   # dernière pos connue
    tag_last_seen:  float                  = field(default_factory=time.monotonic)
    tag_lost_count: int                    = 0      # frames consécutives sans tag
    tag_conf_count: int                    = 0      # frames consécutives avec tag
    created_at:     float                  = field(default_factory=time.monotonic)
    # Historique pour détection de mouvement
    airborne_origin: Optional[Tuple[float, float]] = None  # pos au décollage

    @property
    def label(self) -> str:
        return f"Cup#{self.aruco_id}"


# ──────────────────────────────────────────────────────────────────────────────
#  Manager principal
# ──────────────────────────────────────────────────────────────────────────────

class CupIdentityManager:
    """
    Gère la correspondance ArUco ↔ tracker KCF.

    Utilisation typique :
        manager = CupIdentityManager()

        # Dans CamBottomThread, à chaque frame :
        manager.update_aruco({1: (120.5, 340.2), 3: (400.1, 200.8)})

        # Dans CamTopThread / TrackingManager, à chaque frame :
        manager.update_trackers({0: (125.0, 338.0), 1: (398.5, 202.1)})

        # Pour afficher :
        labels = manager.get_labels()
        # → {tracker_id: "Cup#1", ...}
    """

    def __init__(
        self,
        match_dist_mm:  float = MATCH_DIST_MM,
        repose_dist_mm: float = REPOSE_DIST_MM,
        tag_lost_delay: int   = TAG_LOST_DELAY,
        tag_found_conf: int   = TAG_FOUND_CONF,
    ):
        self._lock = threading.RLock()

        self._match_dist  = match_dist_mm
        self._repose_dist = repose_dist_mm
        self._lost_delay  = tag_lost_delay
        self._found_conf  = tag_found_conf

        # aruco_id → CupIdentity
        self._identities: Dict[int, CupIdentity] = {}

        # tracker_id → aruco_id  (lookup rapide)
        self._tracker_to_aruco: Dict[int, int] = {}

        # Dernières positions ArUco brutes (depuis cam_bottom)
        self._last_aruco: Dict[int, Tuple[float, float]] = {}

        # Dernières positions trackers (depuis cam_top)
        self._last_trackers: Dict[int, Tuple[float, float]] = {}

        self._frame_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface publique
    # ──────────────────────────────────────────────────────────────────────────

    def update_aruco(self, aruco_positions: Dict[int, Tuple[float, float]]) -> None:
        """
        Appelé par CamBottomThread à chaque frame.
        aruco_positions : {aruco_id: (x_mm, y_mm)}
        """
        with self._lock:
            self._last_aruco = dict(aruco_positions)
            self._process_aruco_frame(aruco_positions)

    def update_trackers(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        """
        Appelé par CamTopThread / TrackingManager à chaque frame.
        tracker_positions : {tracker_id: (x_mm, y_mm)}
        """
        with self._lock:
            self._last_trackers = dict(tracker_positions)
            self._frame_count  += 1
            self._process_tracker_frame(tracker_positions)

    def get_labels(self) -> Dict[int, str]:
        """
        Retourne {tracker_id: label} pour l'affichage.
        Un tracker sans identité reçoit le label "Cup#?".
        """
        with self._lock:
            result = {}
            for aruco_id, ident in self._identities.items():
                if ident.tracker_id is not None:
                    result[ident.tracker_id] = ident.label
            return result

    def get_identity(self, tracker_id: int) -> Optional[CupIdentity]:
        """Retourne l'identité associée à un tracker_id, ou None."""
        with self._lock:
            aruco_id = self._tracker_to_aruco.get(tracker_id)
            if aruco_id is None:
                return None
            return self._identities.get(aruco_id)

    def get_all_identities(self) -> Dict[int, CupIdentity]:
        """Retourne une copie du dict {aruco_id: CupIdentity}."""
        with self._lock:
            return dict(self._identities)

    def get_state(self, tracker_id: int) -> Optional[CupState]:
        ident = self.get_identity(tracker_id)
        return ident.state if ident else None

    def reset(self) -> None:
        """Remet tout à zéro."""
        with self._lock:
            self._identities.clear()
            self._tracker_to_aruco.clear()
            self._last_aruco.clear()
            self._last_trackers.clear()
            self._frame_count = 0
        print("[IdentityManager] Reset complet")

    # ──────────────────────────────────────────────────────────────────────────
    #  Logique interne — traitement ArUco
    # ──────────────────────────────────────────────────────────────────────────

    def _process_aruco_frame(
        self,
        aruco_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        now = time.monotonic()

        # 1. Mettre à jour les identités existantes
        for aruco_id, ident in list(self._identities.items()):
            if aruco_id in aruco_positions:
                pos = aruco_positions[aruco_id]
                ident.pos_mm          = pos
                ident.tag_last_seen   = now
                ident.tag_lost_count  = 0
                ident.tag_conf_count  = min(
                    ident.tag_conf_count + 1, self._found_conf + 1)

                if ident.state == CupState.AIRBORNE:
                    # Tag retrouvé → tentative de confirmation de repose
                    self._try_confirm_repose(ident, pos)
                elif ident.state == CupState.LOST:
                    if ident.tag_conf_count >= self._found_conf:
                        ident.state = CupState.PENDING
                        print(f"[Identity] {ident.label} : LOST → PENDING "
                              f"(tag retrouvé à {pos[0]:.0f},{pos[1]:.0f}mm)")
                elif ident.state == CupState.MATCHED:
                    pass  # normal, pas de changement d'état

            else:
                # Tag absent cette frame
                ident.tag_lost_count += 1
                ident.tag_conf_count  = 0

                if (ident.state == CupState.MATCHED
                        and ident.tag_lost_count >= self._lost_delay):
                    ident.state          = CupState.AIRBORNE
                    ident.airborne_origin = ident.pos_mm
                    print(f"[Identity] {ident.label} → AIRBORNE "
                          f"(origine {ident.airborne_origin})")

        # 2. Créer de nouvelles identités pour les nouveaux tags
        for aruco_id, pos in aruco_positions.items():
            if aruco_id not in self._identities:
                ident = CupIdentity(
                    aruco_id=aruco_id,
                    pos_mm=pos,
                    state=CupState.PENDING,
                )
                self._identities[aruco_id] = ident
                print(f"[Identity] Nouveau tag ArUco #{aruco_id} "
                      f"à ({pos[0]:.0f},{pos[1]:.0f})mm")

    # ──────────────────────────────────────────────────────────────────────────
    #  Logique interne — traitement trackers
    # ──────────────────────────────────────────────────────────────────────────

    def _process_tracker_frame(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        # Supprimer les associations dont le tracker a disparu
        for aruco_id, ident in self._identities.items():
            if (ident.tracker_id is not None
                    and ident.tracker_id not in tracker_positions):
                old_tid = ident.tracker_id
                self._tracker_to_aruco.pop(old_tid, None)
                ident.tracker_id = None
                if ident.state == CupState.MATCHED:
                    ident.state = CupState.LOST
                    print(f"[Identity] {ident.label} → LOST "
                          f"(tracker {old_tid} disparu)")

        # Associer les trackers PENDING aux identités PENDING
        self._match_pending(tracker_positions)

    def _match_pending(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        """
        Associe les identités PENDING aux trackers non encore liés.
        Stratégie greedy sur la distance mm.
        """
        # Trackers déjà liés
        bound_trackers = set(self._tracker_to_aruco.keys())
        free_trackers  = {tid: pos for tid, pos in tracker_positions.items()
                          if tid not in bound_trackers}

        if not free_trackers:
            return

        pending = [ident for ident in self._identities.values()
                   if ident.state == CupState.PENDING
                   and ident.pos_mm is not None
                   and ident.tracker_id is None]

        if not pending:
            return

        # Matrice distances
        tid_list    = list(free_trackers.keys())
        ident_list  = pending

        dist_matrix = np.full((len(ident_list), len(tid_list)), np.inf)
        for i, ident in enumerate(ident_list):
            for j, tid in enumerate(tid_list):
                ix, iy = ident.pos_mm
                tx, ty = free_trackers[tid]
                dist_matrix[i, j] = np.sqrt((ix-tx)**2 + (iy-ty)**2)

        # Greedy : meilleur score en premier
        used_i, used_j = set(), set()
        while True:
            mask = np.full(dist_matrix.shape, np.inf)
            for i in range(len(ident_list)):
                for j in range(len(tid_list)):
                    if i not in used_i and j not in used_j:
                        mask[i, j] = dist_matrix[i, j]
            if mask.min() > self._match_dist:
                break
            idx = np.argmin(mask)
            i, j = divmod(int(idx), len(tid_list))
            self._bind(ident_list[i], tid_list[j])
            used_i.add(i)
            used_j.add(j)

    def _bind(self, ident: CupIdentity, tracker_id: int) -> None:
        ident.tracker_id = tracker_id
        ident.state      = CupState.MATCHED
        self._tracker_to_aruco[tracker_id] = ident.aruco_id
        pos_str = (f"({ident.pos_mm[0]:.0f},{ident.pos_mm[1]:.0f})mm"
                   if ident.pos_mm else "?")
        print(f"[Identity] MATCH : {ident.label} ↔ tracker#{tracker_id} "
              f"@ {pos_str}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Confirmation de repose
    # ──────────────────────────────────────────────────────────────────────────

    def _try_confirm_repose(
        self,
        ident: CupIdentity,
        tag_pos: Tuple[float, float],
    ) -> None:
        """
        Appelé quand un tag AIRBORNE réapparaît.
        Cherche le tracker le plus proche pour confirmer la repose.
        """
        if not self._last_trackers:
            return

        best_tid  = None
        best_dist = np.inf
        for tid, tpos in self._last_trackers.items():
            d = np.sqrt((tpos[0]-tag_pos[0])**2 + (tpos[1]-tag_pos[1])**2)
            if d < best_dist:
                best_dist = d
                best_tid  = tid

        if best_tid is None or best_dist > self._repose_dist:
            # Tag apparu loin de tout tracker → on attend
            return

        # ── Cas 1 : ce tracker était déjà lié à cette identité ───────────────
        if best_tid == ident.tracker_id:
            ident.state = CupState.MATCHED
            print(f"[Identity] {ident.label} → MATCHED "
                  f"(repose confirmée, tracker#{best_tid}, dist={best_dist:.0f}mm)")
            return

        # ── Cas 2 : ce tracker appartient à une autre identité ───────────────
        other_aruco = self._tracker_to_aruco.get(best_tid)
        if other_aruco is not None and other_aruco != ident.aruco_id:
            other_ident = self._identities.get(other_aruco)
            print(f"[Identity] ⚠ Conflit repose : {ident.label} "
                  f"↔ tracker#{best_tid} (anciennement {other_ident.label if other_ident else '?'})"
                  f" — correction ID appliquée")
            # On libère l'autre identité de ce tracker
            if other_ident is not None:
                other_ident.tracker_id = None
                other_ident.state      = CupState.LOST
            self._tracker_to_aruco.pop(best_tid, None)

        # ── Liaison du tracker à cette identité ──────────────────────────────
        # Libérer l'ancien tracker si nécessaire
        if ident.tracker_id is not None:
            self._tracker_to_aruco.pop(ident.tracker_id, None)

        self._bind(ident, best_tid)

    # ──────────────────────────────────────────────────────────────────────────
    #  Debug
    # ──────────────────────────────────────────────────────────────────────────

    def debug_summary(self) -> str:
        with self._lock:
            lines = [f"[IdentityManager] frame={self._frame_count}"]
            for aid, ident in sorted(self._identities.items()):
                pos = (f"({ident.pos_mm[0]:.0f},{ident.pos_mm[1]:.0f})"
                       if ident.pos_mm else "None")
                lines.append(
                    f"  ArUco#{aid:2d} → tracker={ident.tracker_id!s:>4}  "
                    f"state={ident.state.name:<10}  pos={pos}mm"
                )
            return "\n".join(lines)
