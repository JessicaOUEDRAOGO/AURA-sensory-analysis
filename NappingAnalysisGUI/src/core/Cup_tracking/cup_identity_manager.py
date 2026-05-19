# -*- coding: utf-8 -*-
"""
cup_identity_manager.py
========================
Gestionnaire d'identité des tasses.

Fix v3 — cercles fantômes :
  Le cycle rapide AIRBORNE↔MATCHED visible dans les logs est causé par
  _try_confirm_repose() qui confirmait une repose sur un seul tag vu
  pendant une seule frame ArUco. Or les tags scintillent naturellement
  (détectés/perdus alternativement à ~22 fps). La solution : n'accepter
  la repose que si le tag est vu REPOSE_CONF_FRAMES fois consécutives
  à distance < REPOSE_DIST_MM du tracker. Le compteur est réinitialisé
  si le tag disparaît ou si la distance dépasse le seuil.

  Avant (version précédente) :
    tag vu 1 frame → MATCHED immédiat → tag perdu 1 frame → AIRBORNE
    → tag vu 1 frame → MATCHED → ... (10+ cycles par seconde)

  Après :
    tag doit être vu REPOSE_CONF_FRAMES=3 frames consécutives et stables
    avant que la repose soit confirmée → les scintillements ArUco normaux
    ne déclenchent plus de faux MATCHED.

Ajout :
  purge_stale_identities() — appelée par CupTrackingPipeline toutes les
  PURGE_EVERY_FRAMES frames pour éviter l'accumulation sur longue session.
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

MATCH_DIST_MM      = 60.0
REPOSE_DIST_MM     = 40.0
TAG_LOST_DELAY     = 15
TAG_FOUND_CONF     = 5
ORPHAN_TTL_S       = 5.0

# Nombre de frames consécutives où le tag doit être visible et stable
# avant de confirmer une repose (anti-scintillement ArUco)
REPOSE_CONF_FRAMES = 3


# ──────────────────────────────────────────────────────────────────────────────
#  États
# ──────────────────────────────────────────────────────────────────────────────

class CupState(Enum):
    MATCHED  = auto()
    AIRBORNE = auto()
    LOST     = auto()
    PENDING  = auto()


# ──────────────────────────────────────────────────────────────────────────────
#  Entrée d'identité
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CupIdentity:
    aruco_id:        int
    tracker_id:      Optional[int]                = None
    state:           CupState                     = CupState.PENDING
    pos_mm:          Optional[Tuple[float, float]] = None
    tag_last_seen:   float = field(default_factory=time.monotonic)
    tag_lost_count:  int   = 0
    tag_conf_count:  int   = 0
    created_at:      float = field(default_factory=time.monotonic)
    airborne_origin: Optional[Tuple[float, float]] = None

    # Compteur de confirmation de repose (anti-scintillement)
    # Incrémenté à chaque frame où le tag est vu proche du tracker,
    # remis à 0 si le tag disparaît ou s'éloigne.
    repose_conf_count: int = 0

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

        # Toutes les ~300 frames :
        manager.purge_stale_identities(max_age_s=60.0)
    """

    def __init__(
        self,
        match_dist_mm:    float = MATCH_DIST_MM,
        repose_dist_mm:   float = REPOSE_DIST_MM,
        tag_lost_delay:   int   = TAG_LOST_DELAY,
        tag_found_conf:   int   = TAG_FOUND_CONF,
        repose_conf_frames: int = REPOSE_CONF_FRAMES,
    ):
        self._lock = threading.RLock()

        self._match_dist        = match_dist_mm
        self._repose_dist       = repose_dist_mm
        self._lost_delay        = tag_lost_delay
        self._found_conf        = tag_found_conf
        self._repose_conf_frames = repose_conf_frames

        self._identities:       Dict[int, CupIdentity]        = {}
        self._tracker_to_aruco: Dict[int, int]                = {}
        self._last_aruco:       Dict[int, Tuple[float, float]] = {}
        self._last_trackers:    Dict[int, Tuple[float, float]] = {}
        self._frame_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface publique
    # ──────────────────────────────────────────────────────────────────────────

    def update_aruco(self, aruco_positions: Dict[int, Tuple[float, float]]) -> None:
        with self._lock:
            self._last_aruco = dict(aruco_positions)
            self._process_aruco_frame(aruco_positions)

    def update_trackers(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        with self._lock:
            self._last_trackers = dict(tracker_positions)
            self._frame_count  += 1
            self._process_tracker_frame(tracker_positions)

    def get_labels(self) -> Dict[int, str]:
        with self._lock:
            return {
                ident.tracker_id: ident.label
                for ident in self._identities.values()
                if ident.tracker_id is not None
            }

    def get_identity(self, tracker_id: int) -> Optional[CupIdentity]:
        with self._lock:
            aruco_id = self._tracker_to_aruco.get(tracker_id)
            if aruco_id is None:
                return None
            return self._identities.get(aruco_id)

    def get_all_identities(self) -> Dict[int, CupIdentity]:
        with self._lock:
            return dict(self._identities)

    def get_state(self, tracker_id: int) -> Optional[CupState]:
        ident = self.get_identity(tracker_id)
        return ident.state if ident else None

    def reset(self) -> None:
        with self._lock:
            self._identities.clear()
            self._tracker_to_aruco.clear()
            self._last_aruco.clear()
            self._last_trackers.clear()
            self._frame_count = 0
        print("[IdentityManager] Reset complet")

    def purge_stale_identities(self, max_age_s: float = 60.0) -> None:
        """
        Supprime les identités LOST sans tracker depuis plus de max_age_s secondes.
        Appelée périodiquement par CupTrackingPipeline pour éviter l'accumulation
        de fantômes sur des sessions longues.
        """
        now = time.monotonic()
        with self._lock:
            to_delete = [
                aid for aid, ident in self._identities.items()
                if ident.state == CupState.LOST
                and ident.tracker_id is None
                and (now - ident.tag_last_seen) > max_age_s
            ]
            for aid in to_delete:
                del self._identities[aid]
                print(f"[Identity] Purge identité stale ArUco#{aid}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Logique interne — traitement ArUco
    # ──────────────────────────────────────────────────────────────────────────

    def _process_aruco_frame(
        self,
        aruco_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        now = time.monotonic()

        for aruco_id, ident in list(self._identities.items()):
            if aruco_id in aruco_positions:
                pos = aruco_positions[aruco_id]
                ident.pos_mm         = pos
                ident.tag_last_seen  = now
                ident.tag_lost_count = 0
                ident.tag_conf_count = min(
                    ident.tag_conf_count + 1, self._found_conf + 1)

                if ident.state == CupState.AIRBORNE:
                    # Tag retrouvé → tentative de confirmation de repose
                    # (avec compteur anti-scintillement)
                    self._try_confirm_repose(ident, pos)
                elif ident.state == CupState.LOST:
                    if ident.tag_conf_count >= self._found_conf:
                        ident.state = CupState.PENDING
                        ident.repose_conf_count = 0
                        print(f"[Identity] {ident.label} : LOST → PENDING "
                              f"(tag retrouvé à {pos[0]:.0f},{pos[1]:.0f}mm)")
            else:
                # Tag absent cette frame
                ident.tag_lost_count += 1
                ident.tag_conf_count  = 0
                # Réinitialiser le compteur de repose si le tag disparaît
                if ident.state == CupState.AIRBORNE:
                    ident.repose_conf_count = 0

                if (ident.state == CupState.MATCHED
                        and ident.tag_lost_count >= self._lost_delay):
                    ident.state           = CupState.AIRBORNE
                    ident.airborne_origin = ident.pos_mm
                    ident.repose_conf_count = 0
                    print(f"[Identity] {ident.label} → AIRBORNE "
                          f"(origine {ident.airborne_origin})")

        for aruco_id, pos in aruco_positions.items():
            if aruco_id not in self._identities:
                self._identities[aruco_id] = CupIdentity(
                    aruco_id=aruco_id, pos_mm=pos, state=CupState.PENDING)
                print(f"[Identity] Nouveau tag ArUco #{aruco_id} "
                      f"à ({pos[0]:.0f},{pos[1]:.0f})mm")

    # ──────────────────────────────────────────────────────────────────────────
    #  Logique interne — traitement trackers
    # ──────────────────────────────────────────────────────────────────────────

    def _process_tracker_frame(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
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

        self._match_pending(tracker_positions)

    def _match_pending(
        self,
        tracker_positions: Dict[int, Tuple[float, float]],
    ) -> None:
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

        tid_list   = list(free_trackers.keys())
        ident_list = pending

        dist_matrix = np.full((len(ident_list), len(tid_list)), np.inf)
        for i, ident in enumerate(ident_list):
            for j, tid in enumerate(tid_list):
                ix, iy = ident.pos_mm
                tx, ty = free_trackers[tid]
                dist_matrix[i, j] = np.sqrt((ix-tx)**2 + (iy-ty)**2)

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
        ident.tracker_id       = tracker_id
        ident.state            = CupState.MATCHED
        ident.repose_conf_count = 0
        self._tracker_to_aruco[tracker_id] = ident.aruco_id
        pos_str = (f"({ident.pos_mm[0]:.0f},{ident.pos_mm[1]:.0f})mm"
                   if ident.pos_mm else "?")
        print(f"[Identity] MATCH : {ident.label} ↔ tracker#{tracker_id} @ {pos_str}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Confirmation de repose — avec compteur anti-scintillement
    # ──────────────────────────────────────────────────────────────────────────

    def _try_confirm_repose(
        self,
        ident: CupIdentity,
        tag_pos: Tuple[float, float],
    ) -> None:
        """
        Appelé quand un tag AIRBORNE réapparaît.
        Cherche le tracker le plus proche. Si la distance est inférieure à
        REPOSE_DIST_MM, incrémente repose_conf_count. La repose n'est confirmée
        que lorsque repose_conf_count atteint REPOSE_CONF_FRAMES.

        Cela évite le cycle AIRBORNE↔MATCHED rapide causé par le scintillement
        naturel des tags ArUco (détectés/perdus alternativement à ~22 fps).
        """
        if not self._last_trackers:
            ident.repose_conf_count = 0
            return

        best_tid  = None
        best_dist = np.inf
        for tid, tpos in self._last_trackers.items():
            d = np.sqrt((tpos[0]-tag_pos[0])**2 + (tpos[1]-tag_pos[1])**2)
            if d < best_dist:
                best_dist = d
                best_tid  = tid

        if best_tid is None or best_dist > self._repose_dist:
            # Tag trop loin ou pas de tracker → réinitialiser le compteur
            ident.repose_conf_count = 0
            return

        # Le tag est proche du tracker → incrémenter le compteur
        ident.repose_conf_count += 1

        if ident.repose_conf_count < self._repose_conf_frames:
            # Pas encore assez de frames consécutives stables → attendre
            return

        # Compteur atteint — confirmer la repose
        ident.repose_conf_count = 0

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
                  f"↔ tracker#{best_tid} "
                  f"(anciennement {other_ident.label if other_ident else '?'})"
                  f" — correction ID appliquée")
            if other_ident is not None:
                other_ident.tracker_id      = None
                other_ident.state           = CupState.LOST
                other_ident.repose_conf_count = 0
            self._tracker_to_aruco.pop(best_tid, None)

        # ── Liaison du tracker à cette identité ──────────────────────────────
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
                    f"state={ident.state.name:<10}  pos={pos}mm  "
                    f"repose_conf={ident.repose_conf_count}"
                )
            return "\n".join(lines)