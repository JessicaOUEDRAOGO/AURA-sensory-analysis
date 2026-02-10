from PyQt6.QtCore import Qt

class KeyHandler:
    def __init__(self, parent, algorithm_analysis=None):
        self.parent = parent  # Référence à la fenêtre ou à la scène
        self.algorithm_analysis = algorithm_analysis

    def handle_key(self, event):
        key = event.key()
        if key == Qt.Key.Key_S:
            if self.algorithm_analysis and getattr(self.algorithm_analysis, "waiting_for_consigne_key", False):
                self.algorithm_analysis.on_consigne_key_pressed()
        elif key == Qt.Key.Key_U:
            print("Touche U pressée")
        elif key == Qt.Key.Key_D:
            print("Touche D pressée")
        elif key == Qt.Key.Key_L:
            print("Touche L pressée")
        elif key == Qt.Key.Key_R:
            print("Touche R pressée")
        # Ajoute ici d'autres actions

    def handle_key_for_program_started(self, event):
        key = event.key()
        # Basculer entre mode 2D et multidim (par exemple touche "M")
        if key == Qt.Key.Key_M:
            if self.algorithm_analysis:
                self.algorithm_analysis.mode_multidim = not self.algorithm_analysis.mode_multidim
        # Sélection du tag courant
        elif key == Qt.Key.Key_L:
            if self.algorithm_analysis:
                self.algorithm_analysis.select_next_marker(-1)
        elif key == Qt.Key.Key_R:
            if self.algorithm_analysis:
                self.algorithm_analysis.select_next_marker(1)
        # Variation de la dimension supplémentaire
        elif key == Qt.Key.Key_U:
            if self.algorithm_analysis:
                self.algorithm_analysis.modify_current_marker_dimension(0.5)
        elif key == Qt.Key.Key_D:
            if self.algorithm_analysis:
                self.algorithm_analysis.modify_current_marker_dimension(-0.5)
        else:
            print(f"Touche {key} pressée, aucune action définie.")