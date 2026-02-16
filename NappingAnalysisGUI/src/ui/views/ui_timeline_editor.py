# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.protocol.models import TimelineStep, InstructionAsset
from src.core.protocol.timeline_repository import TimelineRepository, new_step_id
from src.core.protocol.asset_repository import InstructionAssetRepository

DEFAULT_DURATION_S = 10.0

class TimelineEditorPage(QtWidgets.QWidget):
    """
    Étape 3 — Timeline
    - Liste ordonnée des étapes
    - Durée + label
    - Réordonner / dupliquer / supprimer
    - Ajouter Pause
    - Sauvegarde dans timeline_steps
    """

    def __init__(self, parent, protocol_id: str):
        super().__init__(parent)
        self.parent = parent
        self.protocol_id = protocol_id

        self.timeline_repo = TimelineRepository()
        self.asset_repo = InstructionAssetRepository()

        self.assets: List[InstructionAsset] = []
        self.steps: List[TimelineStep] = []

        self._build_ui()
        self.load_from_db_or_assets()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Étape 3 — Organisation des étapes (Timeline)")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        self.table = QtWidgets.QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["#", "Label", "Type", "Fichier", "Durée (s)", "Pause"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table)

        btns = QtWidgets.QHBoxLayout()

        self.btn_up = QtWidgets.QPushButton("↑")
        self.btn_down = QtWidgets.QPushButton("↓")
        self.btn_duplicate = QtWidgets.QPushButton("Dupliquer")
        self.btn_delete = QtWidgets.QPushButton("Supprimer")
        self.btn_add_pause = QtWidgets.QPushButton("Ajouter pause")
        self.btn_save = QtWidgets.QPushButton("Valider timeline")

        self.btn_up.clicked.connect(lambda: self.move_selected(-1))
        self.btn_down.clicked.connect(lambda: self.move_selected(+1))
        self.btn_duplicate.clicked.connect(self.duplicate_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_add_pause.clicked.connect(self.add_pause)
        self.btn_save.clicked.connect(self.save_to_db)

        btns.addWidget(self.btn_up)
        btns.addWidget(self.btn_down)
        btns.addWidget(self.btn_duplicate)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        btns.addWidget(self.btn_add_pause)
        btns.addWidget(self.btn_save)

        root.addLayout(btns)

        self.label_status = QtWidgets.QLabel("")
        root.addWidget(self.label_status)

    # ---------------- Loading ----------------
    def load_from_db_or_assets(self):
        self.steps = []
        existing = self.timeline_repo.list(self.protocol_id)
        if existing:
            self.steps = existing
            self.refresh_table()
            self.label_status.setText(f"Timeline chargée ({len(self.steps)} étape(s)).")
            return

        # Sinon : on initialise la timeline depuis les assets importés (2B)
        self.assets = self.asset_repo.list(self.protocol_id)
        if not self.assets:
            self.label_status.setText("Aucun asset importé. Reviens à l’étape 2B.")
            return

        self.steps = []
        for idx, a in enumerate(self.assets):
            self.steps.append(
                TimelineStep(
                    id=new_step_id(),
                    protocol_id=self.protocol_id,
                    order_index=idx,
                    asset_ref=a.id,
                    duration_s=DEFAULT_DURATION_S,
                    label=f"Étape {idx+1}",
                    repeat=None,
                    pause=False,
                    trigger=None,
                )
            )
        self.refresh_table()
        self.label_status.setText("Timeline initialisée depuis les assets (pense à valider).")

    # ---------------- Table helpers ----------------
    def refresh_table(self):
        self.table.setRowCount(0)

        # re-index
        for i, s in enumerate(self.steps):
            self.steps[i] = TimelineStep(**{**s.__dict__, "order_index": i})

        # cache assets by id
        assets = {a.id: a for a in self.asset_repo.list(self.protocol_id)}

        for i, s in enumerate(self.steps):
            self.table.insertRow(i)

            # # (ordre)
            item_idx = QtWidgets.QTableWidgetItem(str(i + 1))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item_idx)

            # label (editable)
            item_label = QtWidgets.QTableWidgetItem(s.label)
            self.table.setItem(i, 1, item_label)

            # type (read-only)
            a = assets.get(s.asset_ref) if s.asset_ref else None
            typ = "pause" if (s.pause or (s.asset_ref is None)) else (a.asset_type if a else "unknown")
            item_type = QtWidgets.QTableWidgetItem(typ)
            item_type.setFlags(item_type.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, item_type)

            # fichier (read-only)
            fname = ""
            if a:
                fname = a.path.split("/")[-1].split("\\")[-1]
            item_file = QtWidgets.QTableWidgetItem(fname)
            item_file.setFlags(item_file.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 3, item_file)

            # durée (editable)
            item_dur = QtWidgets.QTableWidgetItem(str(s.duration_s))
            self.table.setItem(i, 4, item_dur)

            # pause (checkbox)
            cb = QtWidgets.QCheckBox()
            cb.setChecked(bool(s.pause))
            cb.setEnabled(True)
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(cb)
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(i, 5, w)

    def selected_row(self) -> Optional[int]:
        items = self.table.selectedItems()
        if not items:
            return None
        return items[0].row()

    # ---------------- Actions ----------------
    def move_selected(self, delta: int):
        r = self.selected_row()
        if r is None:
            return
        new_r = r + delta
        if new_r < 0 or new_r >= len(self.steps):
            return
        self.steps[r], self.steps[new_r] = self.steps[new_r], self.steps[r]
        self.refresh_table()
        self.table.selectRow(new_r)

    def duplicate_selected(self):
        r = self.selected_row()
        if r is None:
            return
        src = self.steps[r]
        dup = TimelineStep(
            id=new_step_id(),
            protocol_id=self.protocol_id,
            order_index=r + 1,
            asset_ref=src.asset_ref,
            duration_s=src.duration_s,
            label=f"{src.label} (copy)",
            repeat=src.repeat,
            pause=src.pause,
            trigger=src.trigger,
        )
        self.steps.insert(r + 1, dup)
        self.refresh_table()
        self.table.selectRow(r + 1)

    def delete_selected(self):
        r = self.selected_row()
        if r is None:
            return
        del self.steps[r]
        self.refresh_table()
        if self.steps:
            self.table.selectRow(min(r, len(self.steps) - 1))

    def add_pause(self):
        r = self.selected_row()
        insert_at = (r + 1) if r is not None else len(self.steps)
        pause_step = TimelineStep(
            id=new_step_id(),
            protocol_id=self.protocol_id,
            order_index=insert_at,
            asset_ref=None,
            duration_s=5.0,
            label="Pause",
            repeat=None,
            pause=True,
            trigger=None,
        )
        self.steps.insert(insert_at, pause_step)
        self.refresh_table()
        self.table.selectRow(insert_at)

    def save_to_db(self):
        # lire table -> steps
        for i in range(self.table.rowCount()):
            label = (self.table.item(i, 1).text() if self.table.item(i, 1) else "").strip() or f"Étape {i+1}"

            # durée
            dur_txt = (self.table.item(i, 4).text() if self.table.item(i, 4) else "").strip()
            try:
                dur = float(dur_txt)
                if dur <= 0:
                    raise ValueError()
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Timeline", f"Durée invalide à la ligne {i+1}.")
                return

            # pause checkbox
            cell = self.table.cellWidget(i, 5)
            cb = cell.findChild(QtWidgets.QCheckBox) if cell else None
            is_pause = bool(cb.isChecked()) if cb else False

            s = self.steps[i]
            self.steps[i] = TimelineStep(**{
                **s.__dict__,
                "order_index": i,
                "label": label,
                "duration_s": dur,
                "pause": is_pause
            })

        self.timeline_repo.replace_all(self.protocol_id, self.steps)
        self.label_status.setText("Timeline sauvegardée.")
        QtWidgets.QMessageBox.information(self, "Timeline", "Timeline sauvegardée.")
