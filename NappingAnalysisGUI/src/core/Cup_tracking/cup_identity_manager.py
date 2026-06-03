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

Ajout v4 — export enrichi :
  - get_raw_aruco_positions()   : positions brutes cam_bottom ce frame
  - get_raw_tracker_positions() : positions brutes trackers KCF ce frame
  - get_tracker_positions_by_aruco() : positions trackers indexées par aruco_id
    (retourne None si le tracker n'est pas encore matchéà un tag)
  - get_association_log()       : historique complet des liaisons tracker↔tag
    pour export JSON en fin de session
  Ces méthodes sont thread-safe et n'altèrent aucune logique interne.

Ajout :
  purge_stale_identities() — appelée par CupTrackingPipeline toutes les
  PURGE_EVERY_FRAMES frames pour éviter l'accumulation sur longue session.

1. AssociationEvent — champ extra: Optional[dict] = None
     Permet de stocker des données supplémentaires (drift_mm, positions, etc.)
     sur les événements de type "hijack_detected".

  2. log_hijack_event() — nouvelle méthode publique
     Appelée depuis la boucle principale de napping_lite.py quand un hijacking
     est confirmé par drift ArUco. Enregistre un AssociationEvent avec
     event="hijack_detected" et les infos de drift dans extra.

  3. get_association_log() — enrichi pour inclure extra
     Les dicts retournés incluent maintenant les champs de extra quand présents.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

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

# Nouveau paramètre — après ORPHAN_TTL_S
ORPHAN_FRAMES_BEFORE_BOOTSTRAP = 20   # frames en PENDING sans tracker avant tentative
SPAWN_CONF_FRAMES               = 5   # frames de stabilité post-spawn pour valider
SPAWN_VALID_MM                  = 50  # distance max ArUco↔tracker pendant validation
AMBIGUITY_RADIUS_MM             = 120 # si 2 ArUcos connus à moins de ça → pas de spawn
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

    # ── Bootstrap recovery ───────────────────────────────────────────────────
    orphan_frames:    int          = 0     # frames consécutives PENDING sans tracker
    spawn_tracker_id: Optional[int] = None  # tracker en cours de validation post-spawn
    spawn_conf_count: int          = 0     # frames stables depuis le spawn

    @property
    def label(self) -> str:
        return f"Cup#{self.aruco_id}"


# ──────────────────────────────────────────────────────────────────────────────
#  Enregistrement des associations — pour export JSON
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AssociationEvent:
    """Un événement de liaison ou déliaison tracker↔tag."""
    aruco_id:   int
    tracker_id: int
    event:      str   # "bind" ou "unbind"
    timestamp:  float = field(default_factory=time.monotonic)
    frame:      int   = 0
    extra:      Optional[dict] = None


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

        # Export CSV — positions brutes de chaque source :
        aruco_pos   = manager.get_raw_aruco_positions()
        tracker_pos = manager.get_raw_tracker_positions()
        by_aruco    = manager.get_tracker_positions_by_aruco()

        # Toutes les ~300 frames :
        manager.purge_stale_identities(max_age_s=60.0)

        # En fin de session :
        log = manager.get_association_log()
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

        # Historique des associations pour export JSON
        self._association_log: List[AssociationEvent] = []

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface publique — mise à jour
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface publique — lecture (logique existante)
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface publique — positions brutes pour CSV enrichi (NOUVEAU)
    # ──────────────────────────────────────────────────────────────────────────

    def get_raw_aruco_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Retourne les positions ArUco brutes de la dernière frame cam_bottom,
        indexées par aruco_id.

        Utilisé pour alimenter les colonnes ID_X_x_bottom / ID_X_y_bottom du CSV.
        Retourne une copie thread-safe — ne pas modifier le dict retourné.
        """
        with self._lock:
            return dict(self._last_aruco)

    def get_raw_tracker_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Retourne les positions de tous les trackers KCF actifs de la dernière
        frame cam_top, indexées par tracker_id interne.

        Utilisé pour les colonnes ID_X_x_tracker / ID_X_y_tracker du CSV,
        APRÈS résolution tracker_id → aruco_id via get_tracker_positions_by_aruco().
        """
        with self._lock:
            return dict(self._last_trackers)

    def get_tracker_positions_by_aruco(self) -> Dict[int, Tuple[float, float]]:
        """
        Retourne les positions des trackers KCF indexées par aruco_id.

        Ne retourne que les trackers actuellement liés à un tag ArUco connu.
        Les trackers sans identité (pas encore matchés) sont ignorés ici — leurs
        coordonnées apparaîtront quand même dans les colonnes _tracker dès qu'un
        match est établi.

        Exemple de retour :
            {6: (234.5, 178.2), 8: (401.0, 320.7)}
        """
        with self._lock:
            result: Dict[int, Tuple[float, float]] = {}
            for tracker_id, pos in self._last_trackers.items():
                aruco_id = self._tracker_to_aruco.get(tracker_id)
                if aruco_id is not None:
                    result[aruco_id] = pos
            return result

    def get_association_log(self) -> List[dict]:
        with self._lock:
            result = []
            for ev in self._association_log:
                d = {
                    "aruco_id":   ev.aruco_id,
                    "tracker_id": ev.tracker_id,
                    "event":      ev.event,
                    "timestamp":  round(ev.timestamp, 3),
                    "frame":      ev.frame,
                }
                if ev.extra:
                    d.update(ev.extra)   # ajoute drift_mm, positions, etc.
                result.append(d)
            return result

    def get_tracker_history_by_aruco(self) -> Dict[int, List[int]]:
        """
        Retourne, pour chaque aruco_id, la liste de tous les tracker_id
        qui lui ont été associés durant la session (sans doublons, en ordre
        chronologique).

        Format :
            {6: [0, 3, 5], 8: [1, 4]}

        Utilisé pour construire le résumé compact dans associations.json.
        """
        with self._lock:
            history: Dict[int, List[int]] = {}
            for ev in self._association_log:
                if ev.event == "bind":
                    lst = history.setdefault(ev.aruco_id, [])
                    if ev.tracker_id not in lst:
                        lst.append(ev.tracker_id)
            return history

    # Dans CupIdentityManager — nouvelle méthode publique
    def get_orphan_aruco_ids(
        self,
        min_frames: int = ORPHAN_FRAMES_BEFORE_BOOTSTRAP,
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Retourne les ArUco en PENDING sans tracker depuis min_frames frames.
        Format : [(aruco_id, pos_mm), ...]
        Utilisé par napping_lite pour décider de bootstrapper un tracker.
        """
        with self._lock:
            result = []
            for aid, ident in self._identities.items():
                if (ident.state == CupState.PENDING
                        and ident.tracker_id is None
                        and ident.pos_mm is not None
                        and ident.orphan_frames >= min_frames):
                    result.append((aid, ident.pos_mm))
            return result

    def tick_orphan_frames(self) -> None:
        """
        À appeler une fois par frame depuis la boucle principale.
        Incrémente orphan_frames pour les PENDING sans tracker,
        remet à 0 pour les autres.
        """
        with self._lock:
            for ident in self._identities.values():
                if ident.state == CupState.PENDING and ident.tracker_id is None:
                    ident.orphan_frames += 1
                else:
                    ident.orphan_frames = 0

    def validate_spawn(
        self,
        aruco_id: int,
        tracker_id: int,
        current_aruco_pos: Optional[Tuple[float, float]],
    ) -> str:
        """
        À appeler chaque frame après un spawn pour valider ou rollback.
        Retourne : 'confirmed', 'pending', ou 'rollback'
        """
        with self._lock:
            ident = self._identities.get(aruco_id)
            if ident is None:
                return 'rollback'

            if current_aruco_pos is None:
                ident.spawn_conf_count = 0
                return 'pending'

            if ident.pos_mm is None:
                return 'rollback'

            dx = ident.pos_mm[0] - current_aruco_pos[0]
            dy = ident.pos_mm[1] - current_aruco_pos[1]
            # On compare le tracker (via pos_mm de l'identité mise à jour)
            # avec la position ArUco courante
            dist = (dx*dx + dy*dy)**0.5

            if dist > SPAWN_VALID_MM:
                ident.spawn_conf_count = 0
                ident.spawn_tracker_id = None
                return 'rollback'

            ident.spawn_conf_count += 1
            if ident.spawn_conf_count >= SPAWN_CONF_FRAMES:
                ident.spawn_conf_count = 0
                ident.spawn_tracker_id = None
                return 'confirmed'

            return 'pending'
    # ──────────────────────────────────────────────────────────────────────────
    #  Reset et purge
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._identities.clear()
            self._tracker_to_aruco.clear()
            self._last_aruco.clear()
            self._last_trackers.clear()
            self._frame_count = 0
            # On ne vide PAS _association_log pour garder l'historique complet
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
    
    def log_hijack_event(
        self,
        aruco_id:    int,
        tracker_id:  int,
        frame:       int,
        drift_event: dict,
    ) -> None:
        """
        Enregistre un événement de hijacking détecté par drift ArUco.

        Paramètres :
          aruco_id    — aruco_id de la tasse dont le tracker a dérivé
          tracker_id  — tracker_id interne invalidé
          frame       — numéro de frame courant
          drift_event — dict avec drift_mm, drift_frames,
                        tracker_pos_mm, aruco_pos_mm

        L'événement apparaît dans get_association_log() avec
        event="hijack_detected" et les champs de drift_event à plat.
        """
        with self._lock:
            self._association_log.append(AssociationEvent(
                aruco_id=aruco_id,
                tracker_id=tracker_id,
                event="hijack_detected",
                frame=frame,
                extra=drift_event,
            ))

    # ──────────────────────────────────────────────────────────────────────────
    #  Logique interne — traitement ArUco (inchangée)
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
                    self._try_confirm_repose(ident, pos)
                elif ident.state == CupState.LOST:
                    if ident.tag_conf_count >= self._found_conf:
                        ident.state = CupState.PENDING
                        ident.repose_conf_count = 0
                        print(f"[Identity] {ident.label} : LOST → PENDING "
                              f"(tag retrouvé à {pos[0]:.0f},{pos[1]:.0f}mm)")
            else:
                ident.tag_lost_count += 1
                ident.tag_conf_count  = 0
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
    #  Logique interne — traitement trackers (inchangée)
    # ──────────────────────────────────────────────────────────────────────────

    def _process_tracker_frame(self, tracker_positions):
        for aruco_id, ident in self._identities.items():
            if (ident.tracker_id is not None
                    and ident.tracker_id not in tracker_positions):
                old_tid = ident.tracker_id
                self._tracker_to_aruco.pop(old_tid, None)
                ident.tracker_id = None
                if ident.state == CupState.MATCHED:
                    ident.state = CupState.LOST
                    self._association_log.append(AssociationEvent(
                        aruco_id=aruco_id, tracker_id=old_tid,
                        event="unbind", frame=self._frame_count))
                    print(f"[Identity] {ident.label} → LOST (tracker {old_tid} disparu)")
            # PENDING sans tracker → incrémenter orphan_frames
            if ident.state == CupState.PENDING and ident.tracker_id is None:
                ident.orphan_frames += 1
            elif ident.tracker_id is not None:
                ident.orphan_frames = 0

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
        # Log liaison
        self._association_log.append(AssociationEvent(
            aruco_id=ident.aruco_id,
            tracker_id=tracker_id,
            event="bind",
            frame=self._frame_count,
        ))
        print(f"[Identity] MATCH : {ident.label} ↔ tracker#{tracker_id} @ {pos_str}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Confirmation de repose — avec compteur anti-scintillement (inchangée)
    # ──────────────────────────────────────────────────────────────────────────

    def _try_confirm_repose(
        self,
        ident: CupIdentity,
        tag_pos: Tuple[float, float],
    ) -> None:
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
            ident.repose_conf_count = 0
            return

        ident.repose_conf_count += 1

        if ident.repose_conf_count < self._repose_conf_frames:
            return

        ident.repose_conf_count = 0

        if best_tid == ident.tracker_id:
            ident.state = CupState.MATCHED
            print(f"[Identity] {ident.label} → MATCHED "
                  f"(repose confirmée, tracker#{best_tid}, dist={best_dist:.0f}mm)")
            return

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
                # Log déliaison suite à conflit
                self._association_log.append(AssociationEvent(
                    aruco_id=other_aruco,
                    tracker_id=best_tid,
                    event="unbind",
                    frame=self._frame_count,
                ))
            self._tracker_to_aruco.pop(best_tid, None)

        if ident.tracker_id is not None:
            self._tracker_to_aruco.pop(ident.tracker_id, None)

        self._bind(ident, best_tid)

    # ──────────────────────────────────────────────────────────────────────────
    #  Debug (inchangé)
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