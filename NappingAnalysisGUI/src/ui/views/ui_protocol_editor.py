# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import uuid
import csv
from datetime import datetime

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.asset_repository import InstructionAssetRepository
from src.core.protocol.models import InstructionAsset
from src.core.utils.paths import data_path
from src.ui.views.ui_timeline_editor import TimelineEditorPage
from src.core.storage.db import connect


def _sanitize_name(s: str) -> str:
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip().replace(" ", "_")


# ----------------------------
# Page Step 4: Modules + Export
# ----------------------------
class ProtocolModulesPage(QtWidgets.QWidget):
    def __init__(self, parent, protocol_id: str, locked: bool):
        super().__init__(parent)
        self.parent = parent
        self.protocol_id = protocol_id
        self.locked = locked

        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Étape 4 — Configuration du protocole")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        # ---- Modules
        gb_modules = QtWidgets.QGroupBox("Modules activés")
        lay_m = QtWidgets.QVBoxLayout(gb_modules)

        self.cb_drawing = QtWidgets.QCheckBox("Dessin")
        self.cb_grouping = QtWidgets.QCheckBox("Regroupement")
        self.cb_projection_media = QtWidgets.QCheckBox("Projection média")
        self.cb_annotations = QtWidgets.QCheckBox("Annotations")
        self.cb_overlay_ra = QtWidgets.QCheckBox("Overlay RA")
        self.cb_advanced_logs = QtWidgets.QCheckBox("Logs avancés")

        for cb in [
            self.cb_drawing, self.cb_grouping, self.cb_projection_media,
            self.cb_annotations, self.cb_overlay_ra, self.cb_advanced_logs
        ]:
            cb.setEnabled(not locked)
            lay_m.addWidget(cb)

        # ---- Data export
        gb_export = QtWidgets.QGroupBox("Données à exporter")
        lay_e = QtWidgets.QVBoxLayout(gb_export)

        self.cb_events = QtWidgets.QCheckBox("Events")
        self.cb_trajectories = QtWidgets.QCheckBox("Trajectoires")
        self.cb_step_durations = QtWidgets.QCheckBox("Durée par étape")
        self.cb_logs = QtWidgets.QCheckBox("Logs")
        self.cb_snapshots = QtWidgets.QCheckBox("Snapshots")

        for cb in [
            self.cb_events, self.cb_trajectories, self.cb_step_durations,
            self.cb_logs, self.cb_snapshots
        ]:
            cb.setEnabled(not locked)
            lay_e.addWidget(cb)

        root.addWidget(gb_modules)
        root.addWidget(gb_export)

        hint = QtWidgets.QLabel("Ces paramètres sont sauvegardés dans la table protocols (JSON).")
        hint.setStyleSheet("color: #555;")
        root.addWidget(hint)

        root.addStretch(1)

        # valeurs par défaut utiles
        if not locked:
            self.cb_events.setChecked(True)
            self.cb_trajectories.setChecked(True)

    def get_values(self) -> tuple[dict, dict]:
        modules_enabled = {
            "drawing": self.cb_drawing.isChecked(),
            "grouping": self.cb_grouping.isChecked(),
            "projection_media": self.cb_projection_media.isChecked(),
            "annotations": self.cb_annotations.isChecked(),
            "overlay_ra": self.cb_overlay_ra.isChecked(),
            "advanced_logs": self.cb_advanced_logs.isChecked(),
        }
        data_to_export = {
            "events": self.cb_events.isChecked(),
            "trajectories": self.cb_trajectories.isChecked(),
            "step_durations": self.cb_step_durations.isChecked(),
            "logs": self.cb_logs.isChecked(),
            "snapshots": self.cb_snapshots.isChecked(),
        }
        return modules_enabled, data_to_export

    def set_values(self, modules_enabled: dict, data_to_export: dict) -> None:
        modules_enabled = modules_enabled or {}
        data_to_export = data_to_export or {}

        self.cb_drawing.setChecked(bool(modules_enabled.get("drawing", False)))
        self.cb_grouping.setChecked(bool(modules_enabled.get("grouping", False)))
        self.cb_projection_media.setChecked(bool(modules_enabled.get("projection_media", False)))
        self.cb_annotations.setChecked(bool(modules_enabled.get("annotations", False)))
        self.cb_overlay_ra.setChecked(bool(modules_enabled.get("overlay_ra", False)))
        self.cb_advanced_logs.setChecked(bool(modules_enabled.get("advanced_logs", False)))

        # exports
        self.cb_events.setChecked(bool(data_to_export.get("events", True)))
        self.cb_trajectories.setChecked(bool(data_to_export.get("trajectories", True)))
        self.cb_step_durations.setChecked(bool(data_to_export.get("step_durations", False)))
        self.cb_logs.setChecked(bool(data_to_export.get("logs", False)))
        self.cb_snapshots.setChecked(bool(data_to_export.get("snapshots", False)))


# ----------------------------
# Page Step 5: Participants
# ----------------------------
class ProtocolParticipantsPage(QtWidgets.QWidget):
    def __init__(self, parent, protocol_id: str, locked: bool):
        super().__init__(parent)
        self.parent = parent
        self.protocol_id = protocol_id
        self.locked = locked

        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Étape 5 — Participants")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText("1 ID par ligne\nEx:\nP001\nP002\nP003")
        self.text.setEnabled(not locked)
        root.addWidget(self.text)

        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Importer CSV")
        self.btn_import.setEnabled(not locked)
        self.btn_import.clicked.connect(self.import_csv)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#555;")
        row.addWidget(self.btn_import)
        row.addWidget(self.lbl_info)
        row.addStretch(1)
        root.addLayout(row)

        hint = QtWidgets.QLabel("Stockage: table protocol_participants (1 ligne par participant).")
        hint.setStyleSheet("color: #555;")
        root.addWidget(hint)

        root.addStretch(1)

    def _parse_ids(self) -> tuple[list[str], list[str]]:
        raw = self.text.toPlainText().splitlines()
        ids = [s.strip() for s in raw if s.strip()]

        seen = set()
        dup = []
        out = []
        for pid in ids:
            if pid in seen:
                dup.append(pid)
            else:
                seen.add(pid)
                out.append(pid)
        return out, dup

    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir un CSV", "", "CSV (*.csv)")
        if not path:
            return

        imported: list[str] = []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    val = (row[0] or "").strip()
                    if val:
                        imported.append(val)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import CSV", f"Impossible de lire le CSV : {e}")
            return

        current, _ = self._parse_ids()
        merged = current + imported

        # unique + trim
        seen = set()
        final = []
        for pid in merged:
            if pid not in seen:
                seen.add(pid)
                final.append(pid)

        self.text.setPlainText("\n".join(final))
        self.lbl_info.setText(f"{len(imported)} importé(s)")

    def get_participants(self) -> list[str]:
        ids, _ = self._parse_ids()
        return ids

    def set_participants(self, participant_ids: list[str]) -> None:
        participant_ids = participant_ids or []
        self.text.setPlainText("\n".join(participant_ids))


# ----------------------------
# Wizard
# ----------------------------
class ProtocolEditorWizard(QtWidgets.QDialog):
    """
    Wizard V2:
    - Page 2A : goal + hypotheses
    - Page 2B : instruction_type + assets
    - Étape 4 : modules + export
    - Étape 5 : participants
    - Étape 3 : timeline
    """

    def __init__(self, parent, protocol):
        super().__init__(parent)
        self.parent = parent
        self.protocol = protocol  # Protocol dataclass
        self.setWindowTitle(f"Éditeur protocole — {protocol.name}")
        self.resize(900, 650)

        self.repo = ProtocolRepository()
        self.asset_repo = InstructionAssetRepository()

        self.stack = QtWidgets.QStackedWidget()

        # --- pages
        self.page_infos = self._build_page_infos()
        self.page_assets = self._build_page_assets()

        # step4 / step5
        self.page_modules = ProtocolModulesPage(self, protocol_id=self.protocol.id, locked=getattr(self.protocol, "locked", False))
        self.page_participants = ProtocolParticipantsPage(self, protocol_id=self.protocol.id, locked=getattr(self.protocol, "locked", False))

        # timeline (step3)
        self.page_timeline = TimelineEditorPage(self, protocol_id=self.protocol.id)

        self.stack.addWidget(self.page_infos)
        self.stack.addWidget(self.page_assets)
        self.stack.addWidget(self.page_modules)
        self.stack.addWidget(self.page_participants)
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
        self._load_existing_modules_and_participants()

        # si locked : désactiver save timeline (au minimum)
        if getattr(self.protocol, "locked", False):
            self.page_timeline.btn_save.setEnabled(False)
            self.page_timeline.btn_add_pause.setEnabled(False)
            self.page_timeline.btn_delete.setEnabled(False)
            self.page_timeline.btn_duplicate.setEnabled(False)
            self.page_timeline.btn_up.setEnabled(False)
            self.page_timeline.btn_down.setEnabled(False)

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
        self.input_goal.setEnabled(not getattr(self.protocol, "locked", False))
        lay.addWidget(self.input_goal)

        lay.addWidget(QtWidgets.QLabel("Hypothèses"))
        self.input_hyp = QtWidgets.QPlainTextEdit()
        self.input_hyp.setEnabled(not getattr(self.protocol, "locked", False))
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

        locked = getattr(self.protocol, "locked", False)
        self.rb_image.setEnabled(not locked)
        self.rb_audio.setEnabled(not locked)
        self.rb_video.setEnabled(not locked)

        g.addWidget(self.rb_image)
        g.addWidget(self.rb_audio)
        g.addWidget(self.rb_video)
        g.addStretch(1)

        lay.addWidget(group)

        # import button
        row = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Importer des fichiers…")
        self.btn_remove = QtWidgets.QPushButton("Supprimer l’asset sélectionné")

        self.btn_import.setEnabled(not locked)
        self.btn_remove.setEnabled(not locked)

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

        # Page infos -> save fields
        if current == self.page_infos:
            if not self._save_protocol_fields():
                return

        # Page assets -> refresh timeline init
        if current == self.page_assets:
            self.page_timeline.load_from_db_or_assets()

        # Page modules -> save config
        if current == self.page_modules:
            if not self._save_modules_and_export():
                return

        # Page participants -> save participants
        if current == self.page_participants:
            if not self._save_participants():
                return

        self.stack.setCurrentIndex(min(self.stack.count() - 1, self.stack.currentIndex() + 1))
        self._refresh_buttons()

    # ---------------- Load existing ----------------
    def _load_existing_values(self):
        self.input_goal.setPlainText(self.protocol.goal or "")
        self.input_hyp.setPlainText(self.protocol.hypotheses or "")

        t = (self.protocol.instruction_type or "image").lower()
        if t == "audio":
            self.rb_audio.setChecked(True)
        elif t == "video":
            self.rb_video.setChecked(True)
        else:
            self.rb_image.setChecked(True)

    def _load_existing_modules_and_participants(self):
        # Reload depuis DB (plus fiable que l'objet passé)
        p_db = self.repo.get_by_id(self.protocol.id)
        if p_db:
            # modules/export (dict JSON)
            self.page_modules.set_values(
                modules_enabled=getattr(p_db, "modules_enabled", {}) or {},
                data_to_export=getattr(p_db, "data_to_export", {}) or {}
            )

        # participants
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT participant_id FROM protocol_participants WHERE protocol_id = ? ORDER BY participant_id ASC",
                (self.protocol.id,)
            ).fetchall()
            ids = [r["participant_id"] for r in rows]
        finally:
            conn.close()

        self.page_participants.set_participants(ids)

    # ---------------- Data save ----------------
    def _selected_instruction_type(self) -> str:
        if self.rb_audio.isChecked():
            return "audio"
        if self.rb_video.isChecked():
            return "video"
        return "image"

    def _save_protocol_fields(self) -> bool:
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

    def _save_modules_and_export(self) -> bool:
        if getattr(self.protocol, "locked", False):
            QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked). Duplique pour modifier.")
            return False

        modules_enabled, data_to_export = self.page_modules.get_values()

        # Option : imposer au moins une donnée exportée
        if not any(data_to_export.values()):
            QtWidgets.QMessageBox.warning(self, "Export", "Coche au moins une donnée à exporter.")
            return False

        try:
            self.repo.update_config(self.protocol.id, modules_enabled, data_to_export)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Sauvegarde config impossible : {e}")
            return False

    def _save_participants(self) -> bool:
        if getattr(self.protocol, "locked", False):
            QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked).")
            return False

        ids = self.page_participants.get_participants()
        if not ids:
            QtWidgets.QMessageBox.warning(self, "Participants", "Liste vide : ajoute au moins 1 ID.")
            return False

        # vérif doublons
        if len(ids) != len(set(ids)):
            QtWidgets.QMessageBox.warning(self, "Participants", "Doublons détectés : corrige la liste.")
            return False

        conn = connect()
        try:
            conn.execute("DELETE FROM protocol_participants WHERE protocol_id = ?", (self.protocol.id,))
            conn.executemany(
                "INSERT OR IGNORE INTO protocol_participants(protocol_id, participant_id) VALUES (?, ?)",
                [(self.protocol.id, pid) for pid in ids]
            )
            conn.commit()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Participants", f"Sauvegarde impossible : {e}")
            return False
        finally:
            conn.close()

        return True

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

                # évite écrasement : suffixe
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

        # on sauvegarde aussi instruction_type si modifié
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
        # save fields
        if not self._save_protocol_fields():
            return

        # save modules/export
        if not self._save_modules_and_export():
            return

        # save participants
        if not self._save_participants():
            return

        # save timeline
        if hasattr(self, "page_timeline"):
            if getattr(self.protocol, "locked", False):
                # locked: ne pas sauvegarder timeline
                pass
            else:
                try:
                    self.page_timeline.save_to_db()
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Timeline", f"Erreur sauvegarde timeline : {e}")
                    return

        self.accept()

# # -*- coding: utf-8 -*-
# from __future__ import annotations

# import os
# import shutil
# import uuid
# from datetime import datetime

# from PyQt6 import QtWidgets
# from PyQt6.QtCore import Qt

# from src.core.protocol.repository import ProtocolRepository
# from src.core.protocol.asset_repository import InstructionAssetRepository
# from src.core.protocol.models import InstructionAsset
# from src.core.utils.paths import data_path
# from src.ui.views.ui_timeline_editor import TimelineEditorPage

# def _sanitize_name(s: str) -> str:
#     bad = '\\/:*?"<>|'
#     for c in bad:
#         s = s.replace(c, "_")
#     return s.strip().replace(" ", "_")


# class ProtocolEditorWizard(QtWidgets.QDialog):
#     """
#     Wizard minimal:
#     - Page 2A: goal + hypotheses
#     - Page 2B: instruction_type (radio) + import assets
#     """

#     def __init__(self, parent, protocol):
#         super().__init__(parent)
#         self.parent = parent
#         self.protocol = protocol  # Protocol dataclass (frozen) mais on ne modifie pas l’objet direct
#         self.setWindowTitle(f"Éditeur protocole — {protocol.name}")
#         self.resize(900, 600)

#         self.repo = ProtocolRepository()
#         self.asset_repo = InstructionAssetRepository()

#         self.stack = QtWidgets.QStackedWidget()

#         # --- pages
#         self.page_infos = self._build_page_infos()
#         self.page_assets = self._build_page_assets()
#         self.page_timeline = TimelineEditorPage(self, protocol_id=self.protocol.id)

#         self.stack.addWidget(self.page_infos)
#         self.stack.addWidget(self.page_assets)
#         self.stack.addWidget(self.page_timeline)


#         # --- footer buttons
#         self.btn_prev = QtWidgets.QPushButton("Précédent")
#         self.btn_next = QtWidgets.QPushButton("Suivant")
#         self.btn_finish = QtWidgets.QPushButton("Terminer")
#         self.btn_cancel = QtWidgets.QPushButton("Annuler")

#         self.btn_prev.clicked.connect(self.prev_page)
#         self.btn_next.clicked.connect(self.next_page)
#         self.btn_finish.clicked.connect(self.finish)
#         self.btn_cancel.clicked.connect(self.reject)

#         footer = QtWidgets.QHBoxLayout()
#         footer.addWidget(self.btn_prev)
#         footer.addWidget(self.btn_next)
#         footer.addStretch(1)
#         footer.addWidget(self.btn_cancel)
#         footer.addWidget(self.btn_finish)

#         root = QtWidgets.QVBoxLayout(self)
#         root.addWidget(self.stack)
#         root.addLayout(footer)

#         self._refresh_buttons()
#         self._load_existing_values()
#         self._refresh_asset_list()

#     # ---------------- Pages ----------------
#     def _build_page_infos(self):
#         w = QtWidgets.QWidget()
#         lay = QtWidgets.QVBoxLayout(w)

#         title = QtWidgets.QLabel("Informations générales")
#         title.setStyleSheet("font-size: 18px; font-weight: 600;")
#         lay.addWidget(title)

#         self.lbl_name = QtWidgets.QLabel(f"Nom : {self.protocol.name}")
#         lay.addWidget(self.lbl_name)

#         lay.addWidget(QtWidgets.QLabel("Objectif (goal)"))
#         self.input_goal = QtWidgets.QPlainTextEdit()
#         lay.addWidget(self.input_goal)

#         lay.addWidget(QtWidgets.QLabel("Hypothèses"))
#         self.input_hyp = QtWidgets.QPlainTextEdit()
#         lay.addWidget(self.input_hyp)

#         lay.addStretch(1)
#         return w

#     def _build_page_assets(self):
#         w = QtWidgets.QWidget()
#         lay = QtWidgets.QVBoxLayout(w)

#         title = QtWidgets.QLabel("Consignes (type unique) + import")
#         title.setStyleSheet("font-size: 18px; font-weight: 600;")
#         lay.addWidget(title)

#         # type selection
#         group = QtWidgets.QGroupBox("Type de consigne (un seul choix)")
#         g = QtWidgets.QHBoxLayout(group)

#         self.rb_image = QtWidgets.QRadioButton("image")
#         self.rb_audio = QtWidgets.QRadioButton("audio")
#         self.rb_video = QtWidgets.QRadioButton("video")
#         g.addWidget(self.rb_image)
#         g.addWidget(self.rb_audio)
#         g.addWidget(self.rb_video)
#         g.addStretch(1)

#         lay.addWidget(group)

#         # import button
#         row = QtWidgets.QHBoxLayout()
#         self.btn_import = QtWidgets.QPushButton("Importer des fichiers…")
#         self.btn_remove = QtWidgets.QPushButton("Supprimer l’asset sélectionné")
#         row.addWidget(self.btn_import)
#         row.addWidget(self.btn_remove)
#         row.addStretch(1)

#         self.btn_import.clicked.connect(self.import_assets)
#         self.btn_remove.clicked.connect(self.remove_selected_asset)

#         lay.addLayout(row)

#         self.list_assets = QtWidgets.QListWidget()
#         lay.addWidget(self.list_assets)

#         hint = QtWidgets.QLabel("Les fichiers sont copiés dans data/protocols/<PROTO_NAME>/assets/")
#         hint.setStyleSheet("color: #555;")
#         lay.addWidget(hint)

#         lay.addStretch(1)
#         return w

#     # ---------------- Navigation ----------------
#     def _refresh_buttons(self):
#         idx = self.stack.currentIndex()
#         self.btn_prev.setEnabled(idx > 0)
#         self.btn_next.setEnabled(idx < self.stack.count() - 1)
#         self.btn_finish.setEnabled(True)

#     def prev_page(self):
#         self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
#         self._refresh_buttons()

#     def next_page(self):
#         current = self.stack.currentWidget()

#         # Si on quitte page_infos → sauvegarder
#         if current == self.page_infos:
#             if not self._save_protocol_fields():
#                 return

#         # Si on quitte page_assets → recharger timeline
#         if current == self.page_assets:
#             self.page_timeline.load_from_db_or_assets()

#         self.stack.setCurrentIndex(min(self.stack.count() - 1, self.stack.currentIndex() + 1))
#         self._refresh_buttons()


#     # ---------------- Data load/save ----------------
#     def _load_existing_values(self):
#         # goal/hypotheses depuis protocol (si déjà en DB tu peux les avoir vides)
#         self.input_goal.setPlainText(self.protocol.goal or "")
#         self.input_hyp.setPlainText(self.protocol.hypotheses or "")

#         t = (self.protocol.instruction_type or "image").lower()
#         if t == "audio":
#             self.rb_audio.setChecked(True)
#         elif t == "video":
#             self.rb_video.setChecked(True)
#         else:
#             self.rb_image.setChecked(True)

#     def _selected_instruction_type(self) -> str:
#         if self.rb_audio.isChecked():
#             return "audio"
#         if self.rb_video.isChecked():
#             return "video"
#         return "image"

#     def _save_protocol_fields(self) -> bool:
#         # si protocole verrouillé, on empêche (sécurité)
#         if getattr(self.protocol, "locked", False):
#             QtWidgets.QMessageBox.warning(self, "Protocole", "Ce protocole est en lecture seule (locked).")
#             return False

#         goal = (self.input_goal.toPlainText() or "").strip()
#         hyp = (self.input_hyp.toPlainText() or "").strip()
#         itype = self._selected_instruction_type()

#         try:
#             self.repo.update_fields(self.protocol.id, goal, hyp, itype)
#             return True
#         except Exception as e:
#             QtWidgets.QMessageBox.warning(self, "Erreur", f"Impossible de sauvegarder le protocole : {e}")
#             return False

#     # ---------------- Assets ----------------
#     def _protocol_asset_dir(self) -> str:
#         safe_proto = _sanitize_name(self.protocol.name)
#         return data_path("protocols", safe_proto, "assets")

#     def _refresh_asset_list(self):
#         self.list_assets.clear()
#         assets = self.asset_repo.list_by_protocol(self.protocol.id)
#         for a in assets:
#             item = QtWidgets.QListWidgetItem(f"[{a.asset_type}] {os.path.basename(a.path)}")
#             item.setData(Qt.ItemDataRole.UserRole, a)
#             self.list_assets.addItem(item)

#     def import_assets(self):
#         if getattr(self.protocol, "locked", False):
#             QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked). Duplique pour modifier.")
#             return

#         itype = self._selected_instruction_type()

#         if itype == "image":
#             flt = "Images (*.png *.jpg *.jpeg)"
#         elif itype == "audio":
#             flt = "Audio (*.wav *.mp3)"
#         else:
#             flt = "Vidéos (*.mp4)"

#         paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Importer", "", flt)
#         if not paths:
#             return

#         os.makedirs(self._protocol_asset_dir(), exist_ok=True)

#         for src in paths:
#             try:
#                 base = os.path.basename(src)
#                 dst = os.path.join(self._protocol_asset_dir(), base)

#                 # évite écrasement : si existe, on suffixe
#                 if os.path.exists(dst):
#                     name, ext = os.path.splitext(base)
#                     dst = os.path.join(self._protocol_asset_dir(), f"{name}_{uuid.uuid4().hex[:6]}{ext}")

#                 shutil.copy2(src, dst)

#                 a = InstructionAsset(
#                     id=str(uuid.uuid4()),
#                     protocol_id=self.protocol.id,
#                     asset_type=itype,
#                     path=dst,
#                     meta={"original_name": os.path.basename(src)},
#                     created_at=datetime.now().isoformat(timespec="seconds")
#                 )
#                 self.asset_repo.add(a)

#             except Exception as e:
#                 QtWidgets.QMessageBox.warning(self, "Import", f"Import échoué pour {src}\n{e}")

#         # on sauvegarde aussi instruction_type (au cas où l’utilisateur a changé)
#         self._save_protocol_fields()
#         self._refresh_asset_list()

#     def remove_selected_asset(self):
#         item = self.list_assets.currentItem()
#         if not item:
#             return
#         a = item.data(Qt.ItemDataRole.UserRole)

#         if getattr(self.protocol, "locked", False):
#             QtWidgets.QMessageBox.warning(self, "Protocole", "Lecture seule (locked).")
#             return

#         ok = QtWidgets.QMessageBox.question(self, "Supprimer", "Supprimer cet asset ?")
#         if ok != QtWidgets.QMessageBox.StandardButton.Yes:
#             return

#         try:
#             # (option) supprimer le fichier disque aussi
#             try:
#                 if a.path and os.path.exists(a.path):
#                     os.remove(a.path)
#             except Exception:
#                 pass

#             self.asset_repo.delete(a.id)
#             self._refresh_asset_list()

#         except Exception as e:
#             QtWidgets.QMessageBox.warning(self, "Erreur", f"Suppression impossible : {e}")

#     # ---------------- Finish ----------------
#     def finish(self):
#         # sauvegarde champs protocole
#         if not self._save_protocol_fields():
#             return

#         # si on est sur la page timeline → sauvegarder
#         if hasattr(self, "page_timeline"):
#             try:
#                 self.page_timeline.save_to_db()
#             except Exception as e:
#                 QtWidgets.QMessageBox.warning(self, "Timeline", f"Erreur sauvegarde timeline : {e}")
#                 return

#         self.accept()

