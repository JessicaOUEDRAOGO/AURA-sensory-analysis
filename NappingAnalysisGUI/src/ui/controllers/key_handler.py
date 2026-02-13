# -*- coding: utf-8 -*-
from PyQt6.QtCore import Qt


class KeyHandler:
    """
    Gestion centralisée des raccourcis clavier.

    - handle_key(): utilisé AVANT le démarrage (ex: consignes)
    - handle_key_for_program_started(): utilisé PENDANT l'acquisition
    """

    def __init__(self, parent, algorithm_analysis=None, debug: bool = False):
        self.parent = parent
        self.algorithm_analysis = algorithm_analysis
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(msg)

    # ------------------------------------------------------------
    # AVANT LANCEMENT (consignes, menus, etc.)
    # ------------------------------------------------------------
    def handle_key(self, event):
        key = event.key()

        # S : valider / passer consigne
        if key == Qt.Key.Key_S:
            if self.algorithm_analysis and getattr(self.algorithm_analysis, "waiting_for_consigne_key", False):
                self._log("[KeyHandler] Consigne validée (S).")
                self.algorithm_analysis.on_consigne_key_pressed()
            return

        # autres touches (optionnel)
        if key in (Qt.Key.Key_U, Qt.Key.Key_D, Qt.Key.Key_L, Qt.Key.Key_R):
            self._log(f"[KeyHandler] Touche {key} pressée (hors acquisition).")
            return

    # ------------------------------------------------------------
    # PENDANT L'ACQUISITION
    # ------------------------------------------------------------
    def handle_key_for_program_started(self, event):
        key = event.key()
        aa = self.algorithm_analysis
        if aa is None:
            self._log("[KeyHandler] Aucun Algorithm_Analysis attaché.")
            return

        # --- Toggle mode multidim ---
        if key == Qt.Key.Key_M:
            aa.mode_multidim = not aa.mode_multidim
            self._log(f"[KeyHandler] mode_multidim = {aa.mode_multidim}")
            return

        # --- Navigation tag courant ---
        # L / Left : précédent
        if key in (Qt.Key.Key_L, Qt.Key.Key_Left):
            aa.select_next_marker(-1)
            self._log("[KeyHandler] Tag précédent.")
            return

        # R / Right : suivant
        if key in (Qt.Key.Key_R, Qt.Key.Key_Right):
            aa.select_next_marker(1)
            self._log("[KeyHandler] Tag suivant.")
            return

        # --- Ajuster dimension ---
        # U / Up : +0.5
        if key in (Qt.Key.Key_U, Qt.Key.Key_Up):
            aa.modify_current_marker_dimension(+0.5)
            self._log("[KeyHandler] Dim +0.5")
            return

        # D / Down : -0.5
        if key in (Qt.Key.Key_D, Qt.Key.Key_Down):
            aa.modify_current_marker_dimension(-0.5)
            self._log("[KeyHandler] Dim -0.5")
            return

        self._log(f"[KeyHandler] Touche ignorée: {key}")
