# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.protocol.service import ProtocolService


class ProtocolHomeWindow(QtWidgets.QWidget):
    """
    Home protocol:
    - Create new protocol (name uniqueness)
    - Search existing protocols
    - Open (locked read-only)
    - Duplicate (editable)
    """

    def __init__(self, parent, protocol_service: ProtocolService):
        super().__init__(parent)
        self.parent = parent
        self.protocol_service = protocol_service

        self._build_ui()
        self.refresh_list()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Protocole")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        # --- Create section
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
        root.addWidget(box_create)

        # --- Existing section
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
        root.addWidget(box_existing)

        self.label_status = QtWidgets.QLabel("")
        self.label_status.setStyleSheet("color: #444;")
        root.addWidget(self.label_status)

        root.addStretch(1)

    # ---------------- Helpers ----------------
    def refresh_list(self):
        self.list_protocols.clear()
        search = (self.input_search.text() or "").strip()

        # repo.list is in repository, so we need access via service.repo
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
        """
        Injecte le protocole sélectionné dans RecordWindow
        pour éviter d'utiliser l'ancien PROTO_TEST.
        """
        if hasattr(self.parent, "record_window") and self.parent.record_window is not None:
            self.parent.record_window.active_protocol_id = p.id

            # fallback participant si non défini
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

        self.parent.current_protocol = p
        self._apply_protocol_to_record(p)
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)
        self.input_new_name.clear()
        self.refresh_list()

        QtWidgets.QMessageBox.information(self, "Protocole", f"Protocole créé : {p.name}")
        # Ici on peut basculer plus tard vers l'éditeur (wizard). Pour l'instant -> Record
        if hasattr(self.parent, "record_window"):
            self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)

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

        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Dupliquer protocole",
            "Nouveau nom (unique) :",
            text=f"{p.name}_COPY"
        )
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
        if hasattr(self.parent, "main_menu"):
            self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)
