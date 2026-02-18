# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.utils.paths import asset_path
from src.ui.widgets.background_widget import BackgroundWidget


from src.ui.views.ui_background_window import BackgroundWindow  


class ProjectionBackgroundWindowWithBG(BackgroundWidget):
    """
    Wrapper qui :
    - met un fond (image)
    - ajoute un scroll pour éviter que la barre des tâches cache les boutons en bas
    - garde ta fenêtre actuelle intacte (inner)
    """
    def __init__(self, parent):
        super().__init__(parent, bg_path=asset_path("images", "backgrounds", "record_bg_dark.jpg"))
        self.parent = parent

        # ----- Layout principal -----
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ----- ScrollArea : évite les boutons cachés en bas -----
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container interne (pour ajouter un padding bas)
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)

        # IMPORTANT : padding bas (réglable)
        # => même si la fenêtre est trop petite / barre des tâches, tu peux scroller.
        lay.setContentsMargins(20, 20, 20, 90)   # 90px en bas = safe
        lay.setSpacing(0)

        # ----- Ta vraie fenêtre (inchangée) -----
        self.inner = BackgroundWindow(parent)
        self.inner.setParent(container)

        lay.addWidget(self.inner)
        lay.addStretch(0)

        self.scroll.setWidget(container)
        root.addWidget(self.scroll)

        # (optionnel) style scroll invisible
        self.scroll.setStyleSheet("""
            QScrollArea { background: transparent; }
            QWidget { background: transparent; }
        """)
