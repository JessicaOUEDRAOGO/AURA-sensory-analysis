# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.utils.paths import asset_path
from src.ui.widgets.background_widget import BackgroundWidget

from src.ui.views.ui_reality_augmented_window import RealityAugementedWindow  # adapte le chemin


class RealityAugementedWindowWithBG(BackgroundWidget):
    def __init__(self, parent):
        super().__init__(parent, bg_path=asset_path("images", "backgrounds", "record_bg_dark.jpg"))
        self.parent = parent

        # Layout principal centré + marges "safe"
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(30, 20, 30, 40)   # <-- + marge bas pour éviter la barre des tâches
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Card
        self.card = QtWidgets.QFrame()
        self.card.setObjectName("arCard")
        self.card.setMaximumWidth(1500)
        self.card.setMinimumWidth(1000)

        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(0)

        # ScrollArea : permet d'accéder au bas si l'écran est trop petit
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Ton widget existant (inchangé)
        self.inner = RealityAugementedWindow(parent)
        self.scroll.setWidget(self.inner)

        card_layout.addWidget(self.scroll)
        root.addWidget(self.card)

        # Style lisibilité + scroll discret
        self.setStyleSheet("""
            #arCard {
                background-color: rgba(15, 18, 22, 185);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 60);
            }

            #arCard QLabel {
                color: rgba(255,255,255,235);
                font-weight: 600;
            }
            QLabel#leftLabelCircle,
            QLabel#leftLabelTrait,
            QLabel#leftLabelText {
                color: rgba(255,255,255,240);   /* noir foncé */
                font-weight: 700;
            }

            #arCard QLineEdit, #arCard QComboBox, #arCard QSpinBox, #arCard QDoubleSpinBox {
                background-color: rgba(255,255,255,230);
                border-radius: 10px;
                padding: 6px 10px;
                border: 1px solid rgba(0,0,0,70);
                color: rgba(10,10,10,240);
            }

            #arCard QPushButton {
                background-color: rgba(255, 214, 120, 45);
                border: 1px solid rgba(255, 180, 60, 200);
                border-radius: 12px;
                padding: 8px 14px;
                font-weight: 800;
                color: rgba(20,20,20,235);
            }
            #arCard QPushButton:hover {
                background-color: rgba(255, 214, 120, 75);
            }
            QLabel[leftLabel="true"]{
                color: rgba(20,20,20,230);
                font-weight: 700;
            }
            /* Scrollbar discrète */
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,120);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
            #arCard QLabel {
            color: rgba(240,240,240,240);
            font-weight: 700;
            }
            /* Labels spécifiques à gauche (Rond / Trait / Text) */
            #arCard QLabel[leftLabel="true"] {
                color: rgba(20, 20, 20, 235);
                font-weight: 800;
            }

        """)

    # Bonus : limite la hauteur de la card selon la fenêtre (évite de passer sous la barre des tâches)
    def resizeEvent(self, event):
        super().resizeEvent(event)

        # on laisse un “safe area” en bas
        safe_bottom = 70
        safe_top = 40
        max_h = max(300, self.height() - (safe_top + safe_bottom))

        self.card.setMaximumHeight(max_h)
