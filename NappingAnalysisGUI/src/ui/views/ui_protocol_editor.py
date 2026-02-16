# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.asset_repository import InstructionAssetRepository
from src.core.protocol.models import InstructionAsset
from src.core.utils.paths import data_path
from src.ui.views.ui_timeline_editor import TimelineEditorPage

def _sanitize_name(s: str) -> str:
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip().replace(" ", "_")


class ProtocolEditorWizard(QtWidgets.QDialog):
    """
    Wizard minimal:
    - Page 2A: goal + hypotheses
    - Page 2B: instruction_type (radio) + import assets
    """

    def __init__(self, parent, protocol):
        super().__init__(parent)
        self.parent = parent
        self.protocol = protocol  # Protocol dataclass (frozen) mais on ne modifie pas l’objet direct
        self.setWindowTitle(f"Éditeur protocole — {protocol.name}")
        self.resize(900, 600)

        self.repo = ProtocolRepository()
        self.asset_repo = InstructionAssetRepository()

        self.stack = QtWidgets.QStackedWidget()

        # --- pages
        self.page_infos = self._build_page_infos()
        self.page_assets = self._build_page_assets()
        self.page_timeline = TimelineEditorPage(self, protocol_id=self.protocol.id)

        self.stack.addWidget(self.page_infos)
        self.stack.addWidget(self.page_assets)
        self.stack.addWidget(self.page_timeline)


        # --- footer buttons
        self.btn_prev = QtWidgets.QPushButton("Précédent")
        self.btn_next = QtWidgets.QPushButton("Suivant")
        self.btn_finish = QtWidgets.QPushButton("Terminer")
        self.btn_cancel = QtWidgets.QPushButton("Annuler")

        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)
        self.btn_finish.clicked.connect(self.finish)
        self.btn_cancel.clicked.connect(self.reject)

        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.btn_prev)
        footer.addWidget(self.btn_next)
        footer.addStretch(1)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_finish)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.stack)
        root.addLayout(footer)

        self._refresh_buttons()
        self._load_existing_values()
        self._refresh_asset_list()

    # ---------------- Pages ----------------
    def _build_page_infos(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        title = QtWidgets.QLabel("Informations générales")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        lay.addWidget(title)

        self.lbl_name = QtWidgets.QLabel(f"Nom : {self.protocol.name}")
        lay.addWidget(self.lbl_name)

        lay.addWidget(QtWidgets.QLabel("Objectif (goal)"))
        self.input_goal = QtWidgets.QPlainTextEdit()
        lay.addWidget(self.input_goal)

        lay.addWidget(QtWidgets.QLabel("Hypothèses"))
        self.input_hyp = QtWidgets.QPlainTextEdit()
        lay.addWidget(self.input_hyp)

        lay.addStretch(1)
        return w

    def _build_page_assets(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        title = QtWidgets.QLabel("Consignes (type unique) + import")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        lay.addWidget(title)

        # type selection
        group = QtWidgets.QGroupBox("Type de consigne (un seul choix)")
        g = QtWidgets.QHBoxLayout(group)

        self.rb_image = QtWidgets.QRadioButton("image")
        self.rb_audio = QtWidgets.QRadioButton("audio")
        self.rb_video = QtWidgets.QRadioButton("video")
        g.addWidget(self.rb_image)
        g.addWidget(self.rb_audio)
        g.addWidget(self.rb_video)
        g.addStretch(1)

        lay.addWidget(group)

        # import button
        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Importer des fichiers…")
        self.btn_remove = QtWidgets.QPushButton("Supprimer l’asset sélectionné")
        row.addWidget(self.btn_import)
        row.addWidget(self.btn_remove)
        row.addStretch(1)

        self.btn_import.clicked.connect(self.import_assets)
        self.btn_remove.clicked.connect(self.remove_selected_asset)

        lay.addLayout(row)

        self.list_assets = QtWidgets.QListWidget()
        lay.addWidget(self.list_assets)

        hint = QtWidgets.QLabel("Les fichiers sont copiés dans data/protocols/<PROTO_NAME>/assets/")
        hint.setStyleSheet("color: #555;")
        lay.addWidget(hint)

        lay.addStretch(1)
        return w

    # ---------------- Navigation ----------------
    def _refresh_buttons(self):
        idx = self.stack.currentIndex()
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < self.stack.count() - 1)
        self.btn_finish.setEnabled(True)

    def prev_page(self):
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._refresh_buttons()

    def next_page(self):
        current = self.stack.currentWidget()

        # Si on quitte page_infos → sauvegarder
        if current == self.page_infos:
            if not self._save_protocol_fields():
                return

        # Si on quitte page_assets → recharger timeline
        if current == self.page_assets:
            self.page_timeline.load_from_db_or_assets()

        self.stack.setCurrentIndex(min(self.stack.count() - 1, self.stack.currentIndex() + 1))
        self._refresh_buttons()


    # ---------------- Data load/save ----------------
    def _load_existing_values(self):
        # goal/hypotheses depuis protocol (si déjà en DB tu peux les avoir vides)
        self.input_goal.setPlainText(self.protocol.goal or "")
        self.input_hyp.setPlainText(self.protocol.hypotheses or "")

        t = (self.protocol.instruction_type or "image").lower()
        if t == "audio":
            self.rb_audio.setChecked(True)
        elif t == "video":
            self.rb_video.setChecked(True)
        else:
            self.rb_image.setChecked(True)

    def _selected_instruction_type(self) -> str:
        if self.rb_audio.isChecked():
            return "audio"
        if self.rb_video.isChecked():
            return "video"
        return "image"

    def _save_protocol_fields(self) -> bool:
        # si protocole verrouillé, on empêche (sécurité)
        if getattr(self.protocol, "locked", False):
            QtWidgets.QMessageBox.warning(self, "Protocole", "Ce protocole est en lecture seule (locked).")
            return False

        goal = (self.input_goal.toPlainText() or "").strip()
        hyp = (self.input_hyp.toPlainText() or "").strip()
        itype = self._selected_instruction_type()

        try:
            self.repo.update_fields(self.protocol.id, goal, hyp, itype)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Impossible de sauvegarder le protocole : {e}")
            return False

    # ---------------- Assets ----------------
    def _protocol_asset_dir(self) -> str:
        safe_proto = _sanitize_name(self.protocol.name)
        return data_path("protocols", safe_proto, "assets")

    def _refresh_asset_list(self):
        self.list_assets.clear()
        assets = self.asset_repo.list_by_protocol(self.protocol.id)
        for a in assets:
            item = QtWidgets.QListWidgetItem(f"[{a.asset_type}] {os.path.basename(a.path)}")
            item.setData(Qt.ItemDataRole.UserRole, a)
            self.list_assets.addItem(item)

    def import_assets(self):
        if getattr(self.protocol, "locked", False):
            QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked). Duplique pour modifier.")
            return

        itype = self._selected_instruction_type()

        if itype == "image":
            flt = "Images (*.png *.jpg *.jpeg)"
        elif itype == "audio":
            flt = "Audio (*.wav *.mp3)"
        else:
            flt = "Vidéos (*.mp4)"

        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Importer", "", flt)
        if not paths:
            return

        os.makedirs(self._protocol_asset_dir(), exist_ok=True)

        for src in paths:
            try:
                base = os.path.basename(src)
                dst = os.path.join(self._protocol_asset_dir(), base)

                # évite écrasement : si existe, on suffixe
                if os.path.exists(dst):
                    name, ext = os.path.splitext(base)
                    dst = os.path.join(self._protocol_asset_dir(), f"{name}_{uuid.uuid4().hex[:6]}{ext}")

                shutil.copy2(src, dst)

                a = InstructionAsset(
                    id=str(uuid.uuid4()),
                    protocol_id=self.protocol.id,
                    asset_type=itype,
                    path=dst,
                    meta={"original_name": os.path.basename(src)},
                    created_at=datetime.now().isoformat(timespec="seconds")
                )
                self.asset_repo.add(a)

            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Import", f"Import échoué pour {src}\n{e}")

        # on sauvegarde aussi instruction_type (au cas où l’utilisateur a changé)
        self._save_protocol_fields()
        self._refresh_asset_list()

    def remove_selected_asset(self):
        item = self.list_assets.currentItem()
        if not item:
            return
        a = item.data(Qt.ItemDataRole.UserRole)

        if getattr(self.protocol, "locked", False):
            QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked).")
            return

        ok = QtWidgets.QMessageBox.question(self, "Supprimer", "Supprimer cet asset ?")
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            # (option) supprimer le fichier disque aussi
            try:
                if a.path and os.path.exists(a.path):
                    os.remove(a.path)
            except Exception:
                pass

            self.asset_repo.delete(a.id)
            self._refresh_asset_list()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Suppression impossible : {e}")

    # ---------------- Finish ----------------
    def finish(self):
        # sauvegarde champs protocole
        if not self._save_protocol_fields():
            return

        # si on est sur la page timeline → sauvegarder
        if hasattr(self, "page_timeline"):
            try:
                self.page_timeline.save_to_db()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Timeline", f"Erreur sauvegarde timeline : {e}")
                return

        self.accept()

