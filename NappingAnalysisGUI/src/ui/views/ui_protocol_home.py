# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.protocol.service import ProtocolService
from src.core.utils.paths import asset_path
from src.ui.widgets.background_widget import BackgroundWidget


class ProtocolHomeWindow(BackgroundWidget):
    def __init__(self, parent, protocol_service: ProtocolService):
        super().__init__(parent, bg_path=asset_path("images", "backgrounds", "record_bg_dark.jpg"))
        self.parent = parent
        self.protocol_service = protocol_service

        self._build_ui()
        self.refresh_list()

    # ---------------- UI ----------------
    def _build_ui(self):
        # =========================
        # Layout principal (UN SEUL)
        # =========================
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # =========================
        # Wrapper centré
        # =========================
        wrapper = QtWidgets.QWidget(self)
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 40, 0, 40)
        wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # =========================
        # Card (container)
        # =========================
        self.card = QtWidgets.QFrame(wrapper)
        self.card.setObjectName("protocolCard")
        self.card.setMinimumWidth(850)
        self.card.setMaximumWidth(1100)

        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(22)

        wrapper_layout.addWidget(self.card)
        root.addWidget(wrapper, 1)

        # =========================
        # UI (on ajoute DIRECTEMENT dans la card)
        # =========================
        title = QtWidgets.QLabel("Protocole")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        card_layout.addWidget(title)

        # ---- Create box ----
        box_create = QtWidgets.QGroupBox("Créer un nouveau protocole")
        lay_create = QtWidgets.QGridLayout(box_create)

        self.input_new_name = QtWidgets.QLineEdit()
        self.input_new_name.setPlaceholderText("Nom du protocole (unique)")

        self.combo_instruction = QtWidgets.QComboBox()
        self.combo_instruction.addItems(["image", "audio", "video"])

        self.btn_create = QtWidgets.QPushButton("Créer")
        self.btn_create.clicked.connect(self.on_create_clicked)

        lay_create.addWidget(QtWidgets.QLabel("Nom"), 0, 0)
        lay_create.addWidget(self.input_new_name, 0, 1)

        lay_create.addWidget(QtWidgets.QLabel("Type consigne"), 1, 0)
        lay_create.addWidget(self.combo_instruction, 1, 1)

        lay_create.addWidget(self.btn_create, 2, 1)
        card_layout.addWidget(box_create)

        # ---- Existing box ----
        box_existing = QtWidgets.QGroupBox("Utiliser un protocole existant")
        lay_existing = QtWidgets.QVBoxLayout(box_existing)

        search_row = QtWidgets.QHBoxLayout()
        self.input_search = QtWidgets.QLineEdit()
        self.input_search.setPlaceholderText("Rechercher par nom...")
        self.input_search.textChanged.connect(self.refresh_list)

        self.btn_refresh = QtWidgets.QPushButton("Rafraîchir")
        self.btn_refresh.clicked.connect(self.refresh_list)

        search_row.addWidget(self.input_search)
        search_row.addWidget(self.btn_refresh)
        lay_existing.addLayout(search_row)

        self.list_protocols = QtWidgets.QListWidget()
        self.list_protocols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        lay_existing.addWidget(self.list_protocols)

        actions_row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Ouvrir (lecture seule)")
        self.btn_open.clicked.connect(self.on_open_clicked)

        self.btn_duplicate = QtWidgets.QPushButton("Dupliquer")
        self.btn_duplicate.clicked.connect(self.on_duplicate_clicked)

        self.btn_back = QtWidgets.QPushButton("Retour menu")
        self.btn_back.clicked.connect(self.go_to_main_menu)

        actions_row.addWidget(self.btn_open)
        actions_row.addWidget(self.btn_duplicate)
        actions_row.addStretch(1)
        actions_row.addWidget(self.btn_back)

        lay_existing.addLayout(actions_row)
        card_layout.addWidget(box_existing)

        self.label_status = QtWidgets.QLabel("")
        self.label_status.setStyleSheet("color: rgba(255,255,255,180);")
        card_layout.addWidget(self.label_status)

        card_layout.addStretch(1)

        # =========================
        # Style premium (ciblé sur la card)
        # =========================
        self.setStyleSheet("""
            /* =========================
            CARD (fond sombre)
            ========================== */
            #protocolCard {
                background-color: rgba(15, 18, 22, 190);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 60);
            }

            /* Texte général (sur la card sombre) */
            #protocolCard QLabel {
                color: rgba(255,255,255,235);
            }

            /* =========================
            GROUPBOX (fond clair)
            /* GroupBox clair */
            #protocolCard QGroupBox {
                background-color: rgba(255,255,255,235);
                border-radius: 12px;
                border: 1px solid rgba(0,0,0,55);
                margin-top: 18px;          /* laisse de la place au titre */
                padding: 18px 14px 14px 14px;
                color: rgba(20,20,20,235);
                font-weight: 600;
            }

            /* Titre en "badge" (lisible) */
            #protocolCard QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 0px;
                padding: 4px 10px;
                background-color: rgba(255,255,255,245);  /* badge clair */
                color: rgba(15,15,15,245);                /* texte noir */
                border-radius: 8px;
                font-weight: 800;
            }

            /* Labels dans les groupbox */
            #protocolCard QGroupBox QLabel {
                color: rgba(25,25,25,235);
                font-weight: 600;
            }

            /* Champs */
            #protocolCard QLineEdit, #protocolCard QComboBox {
                background-color: rgba(255,255,255,245);
                border-radius: 10px;
                padding: 8px 12px;
                border: 1px solid rgba(0,0,0,70);
                color: rgba(15,15,15,240);
            }

            /* Liste */
            #protocolCard QListWidget {
                background-color: rgba(255,255,255,245);
                border-radius: 10px;
                border: 1px solid rgba(0,0,0,70);
                color: rgba(15,15,15,240);
            }

            /* =========================
            BOUTONS (ambre lisible)
            ========================== */
            #protocolCard QPushButton {
                background-color: rgba(255, 214, 120, 55);
                border: 1px solid rgba(255, 180, 60, 200);
                border-radius: 12px;
                padding: 10px 18px;
                font-weight: 800;
                color: rgba(20,20,20,235);
            }

            #protocolCard QPushButton:hover {
                background-color: rgba(255, 214, 120, 85);
                border: 1px solid rgba(255, 180, 60, 255);
            }

            #protocolCard QPushButton:pressed {
                background-color: rgba(255, 214, 120, 110);
            }
            
        """)
    # ---------------- Helpers ----------------
    def refresh_list(self):
        self.list_protocols.clear()
        search = (self.input_search.text() or "").strip()
        protocols = self.protocol_service.repo.list(search=search)

        for p in protocols:
            locked_txt = "🔒" if p.locked else "✏️"
            item = QtWidgets.QListWidgetItem(f"{locked_txt} {p.name}   (v{p.version})")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.list_protocols.addItem(item)

        self.label_status.setText(f"{len(protocols)} protocole(s) trouvé(s).")

    def _get_selected_protocol(self):
        item = self.list_protocols.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _apply_protocol_to_record(self, p):
        if hasattr(self.parent, "record_window") and self.parent.record_window is not None:
            self.parent.record_window.active_protocol_id = p.id
            if not getattr(self.parent.record_window, "active_participant_id", None):
                self.parent.record_window.active_participant_id = "P001"

    # ---------------- Actions ----------------
    def on_create_clicked(self):
        name = (self.input_new_name.text() or "").strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Entre un nom de protocole.")
            return

        instruction_type = self.combo_instruction.currentText()

        try:
            p = self.protocol_service.create_new(name=name, instruction_type=instruction_type)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Protocole", str(e))
            return
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Protocole", f"Erreur création protocole : {e}")
            return

        self.parent.current_protocol = p
        self._apply_protocol_to_record(p)

        try:
            from src.ui.views.ui_protocol_editor import ProtocolEditorWizard
            dlg = ProtocolEditorWizard(self.parent, p)
            res = dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Protocole", f"Impossible d’ouvrir l’éditeur : {e}")
            res = None

        self.input_new_name.clear()
        self.refresh_list()

        if res == QtWidgets.QDialog.DialogCode.Accepted and hasattr(self.parent, "record_window"):
            QtWidgets.QMessageBox.information(self, "Protocole", f"Protocole créé : {p.name}")
            self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)
        else:
            self.label_status.setText(f"Protocole créé mais édition annulée : {p.name}")

    def on_open_clicked(self):
        p = self._get_selected_protocol()
        if not p:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Sélectionne un protocole dans la liste.")
            return

        try:
            opened = self.protocol_service.open_existing_readonly(p.name)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Protocole", str(e))
            return

        self.parent.current_protocol = opened
        self._apply_protocol_to_record(opened)
        self.refresh_list()

        QtWidgets.QMessageBox.information(self, "Protocole", f"Ouvré en lecture seule : {opened.name}")
        if hasattr(self.parent, "record_window"):
            self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)

    def on_duplicate_clicked(self):
        p = self._get_selected_protocol()
        if not p:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Sélectionne un protocole dans la liste.")
            return

        new_name, ok = QtWidgets.QInputDialog.getText(self, "Dupliquer protocole", "Nouveau nom (unique) :", text=f"{p.name}_COPY")
        if not ok:
            return

        new_name = (new_name or "").strip()
        if not new_name:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Nom invalide.")
            return

        try:
            dup = self.protocol_service.duplicate(source_name=p.name, new_name=new_name)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Protocole", str(e))
            return

        self.parent.current_protocol = dup
        self._apply_protocol_to_record(dup)
        self.refresh_list()

        QtWidgets.QMessageBox.information(self, "Protocole", f"Protocole dupliqué : {dup.name}")
        if hasattr(self.parent, "record_window"):
            self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)

    def go_to_main_menu(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)
