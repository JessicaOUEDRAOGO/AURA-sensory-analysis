# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect

from src.core.utils.paths import asset_path
from src.ui.widgets.background_widget import BackgroundWidget

from src.ui.views.ui_background_window import BackgroundWindow  # <-- À ADAPTER


class ProjectionBackgroundWindowWithBG(BackgroundWidget):
    def __init__(self, parent):
        # même image de fond que tes autres pages
        super().__init__(parent, bg_path=asset_path("images", "backgrounds", "record_bg_dark.jpg"))
        self.parent = parent

        # ---- ton widget existant (NE PAS CASSER) ----
        self.inner = BackgroundWindow(parent)

        self._build()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # wrapper centré
        wrapper = QtWidgets.QWidget(self)
        wrap_lay = QtWidgets.QHBoxLayout(wrapper)
        wrap_lay.setContentsMargins(40, 30, 40, 30)     # marges écran
        wrap_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # card
        card = QtWidgets.QFrame(wrapper)
        card.setObjectName("card")
        card.setMaximumWidth(1300)                      # ajuste si tu veux plus large
        card.setMinimumWidth(900)
        card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # ombre (optionnel mais fait “premium”)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(35)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 180))# fallback si besoin
        # NOTE: QColorDialog().currentColor() peut être noir selon contexte,
        # si tu veux stable, remplace les 2 lignes ci-dessus par:
        # from PyQt6.QtGui import QColor
        # shadow.setColor(QColor(0, 0, 0, 180))
        card.setGraphicsEffect(shadow)

        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(18, 18, 18, 18)
        card_lay.setSpacing(12)

        # scroll pour éviter que la taskbar cache les boutons
        scroll = QtWidgets.QScrollArea(card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # IMPORTANT: l'inner doit être dans un container
        container = QtWidgets.QWidget()
        cont_lay = QtWidgets.QVBoxLayout(container)
        cont_lay.setContentsMargins(0, 0, 0, 0)
        cont_lay.setSpacing(0)

        # marge basse extra => même sans scroller un peu, boutons restent accessibles
        # (sinon tu peux réduire)
        cont_lay.addWidget(self.inner)
        cont_lay.addSpacing(20)

        scroll.setWidget(container)
        card_lay.addWidget(scroll)

        wrap_lay.addWidget(card)
        root.addWidget(wrapper)

        # style identique à AR
        self.setStyleSheet("""
            #card {
                background-color: rgba(15, 18, 22, 185);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 60);
            }

            /* Texte */
            #card QLabel {
                color: rgba(240,240,240,240);
                font-weight: 700;
            }

            /* Inputs (si tu as des lineEdit en haut) */
            #card QLineEdit, #card QComboBox, #card QSpinBox, #card QDoubleSpinBox {
                background-color: rgba(255,255,255,230);
                border-radius: 10px;
                padding: 6px 10px;
                border: 1px solid rgba(0,0,0,70);
                color: rgba(10,10,10,240);
            }

            /* BOUTONS AMBRE (identique RA) */
            #card QPushButton {
                background-color: rgba(255, 214, 120, 45);
                border: 1px solid rgba(255, 180, 60, 200);
                border-radius: 12px;
                padding: 8px 14px;
                font-weight: 800;
                color: rgba(20,20,20,235);
            }
            #card QPushButton:hover {
                background-color: rgba(255, 214, 120, 75);
            }
            #card QPushButton:pressed {
                background-color: rgba(255, 214, 120, 95);
            }

            /* Scrollbar propre */
            QScrollBar:vertical {
                background: rgba(0,0,0,20);
                width: 10px;
                margin: 6px 3px 6px 3px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 214, 120, 120);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 214, 120, 180);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

