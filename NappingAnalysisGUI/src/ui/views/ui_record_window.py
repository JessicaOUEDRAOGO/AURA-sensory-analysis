# -*- coding: utf-8 -*-
import os
import glob
import json
from src.core.session.session_service import SessionService
from src.core.session.event_store import EventStore
from src.core.session.export_service import ExportService
import numpy as np
from src.core.protocol.asset_repository import InstructionAssetRepository
from src.core.protocol.timeline_repository import TimelineRepository
from pathlib import Path
from src.core.utils.paths import config_path

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QTimer, QThread
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QFileDialog, QLineEdit, QLabel, QHBoxLayout,
    QGraphicsView, QFrame, QComboBox
)
from src.core.protocol.repository import ProtocolRepository
from src.core.utils.paths import gui_path, asset_path
from src.ui.controllers.key_handler import KeyHandler
from src.ui.widgets.graphics_scene import GraphicsScene
from src.core.vision.camera_manager import CameraManager
from src.app.runtime_v1 import Algorithm_Analysis


class RecordWindow(QtWidgets.QWidget):
    """
    Fenêtre d'enregistrement.

    Nouveau pipeline de calibration :
      Les matrices de projection sont chargées automatiquement par
      Algorithm_Analysis depuis les fichiers JSON :
        - cambottom_table_pose.json
        - H_table_to_proj.json
        - camera_calibration_bottom.json
      Il n'y a plus de bouton "Load Calibration" ni de calib_data dict.
    """
    def __init__(
        self,
        parent,
        nbr_Tag,
        image_background,
        display_manager=None,
        cam_width=3840,
        cam_height=2160,
        grid_size=700
    ):
        super().__init__()
        uic.loadUi(gui_path("Record_Menu.ui"), self)

        self.parent = parent
        self.nbr_Tag = nbr_Tag
        self.cam_width  = cam_width
        self.cam_height = cam_height
        self.grid_size  = grid_size

        self.display_manager         = display_manager
        self.image_background        = image_background
        self.image_background_clean  = image_background.copy()
        self.image_background_with_grid = image_background.copy()

        # Participant (combo)
        self.active_participant_id = None
        self.active_protocol_id    = None

        self._participant_label = QLabel("Participant ID:", self)
        self._participant_combo = QComboBox(self)
        self._participant_combo.currentTextChanged.connect(self._on_participant_changed)

        if hasattr(self, "verticalLayout"):
            self.verticalLayout.insertWidget(0, self._participant_label)
            self.verticalLayout.insertWidget(1, self._participant_combo)

        # Session
        self.session_id         = None
        self.session_output_dir = None
        self.session_service    = SessionService()
        self.event_store        = EventStore()
        self.export_service     = ExportService()

        # Camera
        self.camera_manager = CameraManager(
            camera_index=self.parent.settings["camera_id"],
            width=self.cam_width,
            height=self.cam_height
        )

        # Algo thread
        self.algorithm_thread:   QThread | None = None
        self.algorithm_analysis: Algorithm_Analysis | None = None

        # Key handler
        self.key_handler = KeyHandler(self)

        # UI state
        self.checkboxes = []
        self.pushButton_Stop.setEnabled(False)
        self.pushButton_Start.setEnabled(True)   # plus besoin d'attendre un "load"

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_label)
        self.elapsed_time  = 0
        self.timer_started = False

        # Frame counter
        self.frame_count = 0

        # Graphics scene
        self.scene       = None
        self.default_xmin = -10
        self.default_xmax =  10
        self.default_ymin = -10
        self.default_ymax =  10
        self.default_xleg = "x"
        self.default_yleg = "y"

        self.setup_graphics_view(
            self.default_xmin, self.default_xmax,
            self.default_ymin, self.default_ymax,
            self.default_xleg, self.default_yleg,
            self.grid_size
        )
        self.set_bounds_to_inputs()

        # Signals
        self.pushButton_return.clicked.connect(self.go_to_main)
        self.pushButton_UpdateChanges.clicked.connect(self.update_bounds)
        self.pushButton_UpdateChanges.clicked.connect(self.refresh_projector_background)

        self.pushButton_calibration.clicked.connect(self.start_calibration)

        self.pushButton_Start.clicked.connect(self.start_recording)
        self.pushButton_Stop.clicked.connect(self.stop_recording)

        self.checkBox_DisplayGrid.stateChanged.connect(self.on_display_grid_checkbox_changed)

        # Build tags UI
        self._init_scrollarea_container()
        self.create_tags()

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------
    def showEvent(self, event):
        p = getattr(self.parent, "current_protocol", None)
        if p and hasattr(self, "label_protocol"):
            self.label_protocol.setText(f"Protocole : {p.name} (v{p.version})")
        if p and hasattr(self, "_participant_combo"):
            self._load_participants_for_protocol(p.id)
        super().showEvent(event)

    def keyPressEvent(self, event):
        if not self.timer_started:
            self.key_handler.handle_key(event)
        else:
            self.key_handler.handle_key_for_program_started(event)
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------
    def _on_participant_changed(self, text: str):
        pid = (text or "").strip()
        if pid:
            self.active_participant_id = pid

    def _load_participants_for_protocol(self, protocol_id: str):
        from src.core.storage.db import connect
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT participant_id FROM protocol_participants "
                "WHERE protocol_id=? ORDER BY participant_id ASC",
                (protocol_id,)
            ).fetchall()
            ids = [r["participant_id"] for r in rows]
        finally:
            conn.close()

        self._participant_combo.blockSignals(True)
        self._participant_combo.clear()

        if ids:
            self._participant_combo.addItems(ids)
            self.active_participant_id = ids[0]
            self._participant_combo.setCurrentText(ids[0])
        else:
            self._participant_combo.addItem("P001")
            self.active_participant_id = "P001"
            self._participant_combo.setCurrentText("P001")

        self._participant_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Background / grid projection
    # ------------------------------------------------------------------
    def refresh_projector_background(self):
        if self.display_manager is None:
            return
        bounds = self.get_bounds_from_inputs()
        if bounds is None:
            return
        x_min, x_max, y_min, y_max, x_legend, y_legend = bounds

        if self.checkBox_DisplayGrid.isChecked():
            from src.core.projection.draw_utils import DrawUtils
            img_with_grid = DrawUtils.draw_math_grid_on_image(
                self.image_background_clean,
                x_min, x_max, y_min, y_max,
                x_legend, y_legend,
                self.grid_size
            )
            self.image_background_with_grid = img_with_grid
            self.display_manager.display_image_on_projector_monitor(img_with_grid)
        else:
            self.display_manager.display_image_on_projector_monitor(self.image_background_clean)

    def get_latest_background_path(self):
        files = glob.glob(asset_path("textures", "background_*_final.png"))
        if not files:
            files = glob.glob(asset_path("textures", "background_*.png"))
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def _sanitize_filename(self, s: str) -> str:
        bad = '\\/:*?"<>|'
        for c in bad:
            s = s.replace(c, "_")
        return s.strip().replace(" ", "_")

    def get_export_basename(self) -> str:
        p = getattr(self.parent, "current_protocol", None)
        proto_name     = p.name if p else "UNKNOWN_PROTOCOL"
        participant_id = getattr(self, "active_participant_id", "P001")
        return f"{self._sanitize_filename(proto_name)}_{self._sanitize_filename(participant_id)}"

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        repo     = ProtocolRepository()
        proto_id = None
        p_mem    = getattr(self.parent, "current_protocol", None)
        if p_mem:
            proto_id = p_mem.id

        p_db = repo.get_by_id(proto_id) if proto_id else None
        if not p_db:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Protocole introuvable en base.")
            return

        if self.display_manager is None:
            QtWidgets.QMessageBox.warning(self, "DisplayManager",
                                          "display_manager est None (non injecté depuis MainApp).")
            return

        participant_id = None
        if hasattr(self, "_participant_combo"):
            participant_id = (self._participant_combo.currentText() or "").strip()
        participant_id = participant_id or getattr(self, "active_participant_id", None) or "P001"

        self.active_protocol_id    = p_db.id
        self.active_participant_id = participant_id

        try:
            self.session_id, self.session_output_dir = self.session_service.start_session(
                protocol_id=self.active_protocol_id,
                participant_id=self.active_participant_id,
                protocol_name=p_db.name
            )
            self.event_store.log(self.session_id, "session_started", {
                "protocol_id":    self.active_protocol_id,
                "participant_id": self.active_participant_id
            })
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Session", f"Impossible de créer la session : {e}")
            return

        p = getattr(self.parent, "current_protocol", None)
        if not p:
            QtWidgets.QMessageBox.warning(self, "Protocole", "Sélectionne d'abord un protocole.")
            return

        asset_repo    = InstructionAssetRepository()
        timeline_repo = TimelineRepository()
        assets  = asset_repo.list_by_protocol(p_db.id)
        steps   = timeline_repo.list(p_db.id)

        print("=== START_RECORDING DEBUG ===")
        print("Protocol:", p_db.name, p_db.id)
        print("modules_enabled:", p_db.modules_enabled)
        print("assets:", len(assets), "steps:", len(steps))
        print("=============================")
        self.algorithm_analysis = Algorithm_Analysis(
                parent=self,
                display_manager=self.display_manager,
                image_background=self.image_background,
                record_window=self,
                output_dir=self.session_output_dir,
                output_name=self.get_export_basename(),
                modules_enabled=p_db.modules_enabled or {},
                assets=assets,
                timeline_steps=steps,
                protocol=p_db,
            )

        self.key_handler.algorithm_analysis = self.algorithm_analysis
        self.connect_checkbox_to_algorithm()

        # ── 1. Ouvrir la caméra bottom + undistort ───────────────────────
        self.camera_manager.open_camera()
        try:
            with open(config_path("camera_calibration_fisheye.json"), "r") as f:
                calib = json.load(f)
            K    = np.array(calib["camera_matrix"], dtype=np.float64)
            dist = np.array(calib["dist_coeffs"], dtype=np.float64).reshape(4, 1)
            self.camera_manager.set_undistort(K, dist)
            self.runtime_K = self.camera_manager.K
            print("✅ Undistortion fisheye activé")
        except Exception as e:
            print("❌ Erreur calibration undistort:", e)
            self.runtime_K = None

        # ── 2. Préparer les threads MAINTENANT (runtime_K disponible) ────
        # UNE SEULE fois — __init__ ne l'appelle plus
        self.algorithm_analysis._prepare_threads()

        # ── 3. Connecter et démarrer le thread Qt wrappeur ───────────────
        self.algorithm_thread = QThread()
        self.algorithm_analysis.moveToThread(self.algorithm_thread)

        self.algorithm_thread.started.connect(self.algorithm_analysis.detect_and_process)
        self.algorithm_analysis.data_signal.connect(self.update_ui)
        self.algorithm_analysis.data_signal.connect(self.start_timer_on_first_frame)
        self.algorithm_analysis.finished_signal.connect(self.algorithm_thread.quit)
        self.algorithm_analysis.finished_signal.connect(self.algorithm_analysis.deleteLater)
        self.algorithm_thread.finished.connect(self.algorithm_thread.deleteLater)
        self.algorithm_thread.finished.connect(self.on_thread_finished)

        self.frame_count   = 0
        self.elapsed_time  = 0
        self.timer_started = False
        self.label_timer.setText("Timer : 0 sec")
        self.update_frame_label()

        self.algorithm_thread.start()
        self.pushButton_Start.setEnabled(False)
        self.pushButton_Stop.setEnabled(True)

    def start_timer_on_first_frame(self, _data):
        if not self.timer_started:
            self.timer_started = True
            self.timer.start(1000)

    def on_thread_finished(self):
        print("[UI] thread réellement terminé")
        self.algorithm_thread   = None
        self.algorithm_analysis = None

    def stop_recording(self):
        print("[UI] STOP demandé")

        if self.algorithm_analysis:
            self.algorithm_analysis.stop()

        if self.camera_manager:
            self.camera_manager.close_camera()
            self.parent.stop_hand_tracking()

        if self.algorithm_thread is not None:
            try:
                if self.algorithm_thread.isRunning():
                    print("[UI] thread en cours d'arrêt...")
            except RuntimeError:
                print("[UI] thread déjà supprimé")
                self.algorithm_thread = None

        self.timer.stop()
        self.timer_started = False

        if self.session_id:
            try:
                self.event_store.log(self.session_id, "session_ended", {})
                self.session_service.end_session(self.session_id)
                self.export_service.export_session_minimal(self.session_id)
            except Exception as e:
                print(f"[WARNING] Fin session V2 échouée: {e}")

        self.session_id         = None
        self.session_output_dir = None

        self.pushButton_Start.setEnabled(True)
        self.pushButton_Stop.setEnabled(False)

    # ------------------------------------------------------------------
    # UI update from algo
    # ------------------------------------------------------------------
    def update_ui(self, data):
        graph_coords = data["data"]
        self.frame_count += 1
        self.update_frame_label()
        self.scene.clear_markers()
        detected_ids = set()

        # APRÈS
        for entry in graph_coords:
            marker_id, pos, *rest = entry          # rétrocompatible si state absent
            state = rest[0] if rest else "POSEE"
            x, y = pos
            detected_ids.add(marker_id)
            if 0 <= marker_id < len(self.checkboxes) and self.checkboxes[marker_id].isChecked():
                self.scene.add_marker(x, y, marker_id, state=state)
            label_posx_nbr = self.findChild(QLabel, f"label_posx_nbr_{marker_id}")
            label_posy_nbr = self.findChild(QLabel, f"label_posy_nbr_{marker_id}")
            if label_posx_nbr and label_posy_nbr:
                label_posx_nbr.setText(f"{self.scene.pixel_to_index_x(x):.2f}")
                label_posy_nbr.setText(f"{self.scene.pixel_to_index_y(y):.2f}")

        for marker_id in range(len(self.checkboxes)):
            if marker_id not in detected_ids:
                label_posx_nbr = self.findChild(QLabel, f"label_posx_nbr_{marker_id}")
                label_posy_nbr = self.findChild(QLabel, f"label_posy_nbr_{marker_id}")
                if label_posx_nbr and label_posy_nbr:
                    label_posx_nbr.setText("?")
                    label_posy_nbr.setText("?")

        self.graphicsView.viewport().update()

    def update_frame_label(self):
        self.label_nbr_frame.setText(str(self.frame_count))

    def update_timer_label(self):
        self.elapsed_time += 1
        self.label_timer.setText(f"Timer : {self.elapsed_time} sec")

    # ------------------------------------------------------------------
    # Navigation / calibration
    # ------------------------------------------------------------------
    def start_calibration(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.calibration_window)

    def go_to_main(self):
        self.stop_recording()
        self.parent.stacked_widget.setCurrentWidget(self.parent.main_menu)

    # ------------------------------------------------------------------
    # Bounds / grid inputs
    # ------------------------------------------------------------------
    def get_bounds_from_inputs(self):
        try:
            x_min    = float(self.lineEdit_xmin.text())
            x_max    = float(self.lineEdit_xmax.text())
            y_min    = float(self.lineEdit_ymin.text())
            y_max    = float(self.lineEdit_ymax.text())
            x_legend = self.lineEdit_legx.text()
            y_legend = self.lineEdit_legy.text()
            return x_min, x_max, y_min, y_max, x_legend, y_legend
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur",
                                          "Veuillez entrer des valeurs numériques valides.")
            return None

    def set_bounds_to_inputs(self):
        self.lineEdit_xmin.setText(str(self.default_xmin))
        self.lineEdit_xmax.setText(str(self.default_xmax))
        self.lineEdit_ymin.setText(str(self.default_ymin))
        self.lineEdit_ymax.setText(str(self.default_ymax))
        self.lineEdit_legx.setText(self.default_xleg)
        self.lineEdit_legy.setText(self.default_yleg)

    def update_bounds(self):
        bounds = self.get_bounds_from_inputs()
        if bounds is None:
            return
        x_min, x_max, y_min, y_max, x_legend, y_legend = bounds
        self.scene.update_bounds(x_min, x_max, y_min, y_max, x_legend, y_legend)

    def setup_graphics_view(self, x_min, x_max, y_min, y_max, x_leg, y_leg, grid_size):
        size = self.graphicsView.viewport().size()
        self.scene = GraphicsScene(
            grid_size=grid_size,
            status_mathsElement=True,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            x_legend=x_leg, y_legend=y_leg
        )
        self.scene.setSceneRect(0, 0, size.width(), size.height())
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )

    # ------------------------------------------------------------------
    # Tags in scrollArea
    # ------------------------------------------------------------------
    def _init_scrollarea_container(self):
        if getattr(self, "_tags_container", None) is None:
            self._tags_container = QWidget()
            self._tags_layout    = QVBoxLayout(self._tags_container)
            self._tags_layout.setContentsMargins(6, 6, 6, 6)
            self._tags_layout.setSpacing(4)
            self.scrollArea.setWidget(self._tags_container)
            self.scrollArea.setWidgetResizable(True)

    def create_tags(self):
        while self._tags_layout.count():
            item = self._tags_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.checkboxes.clear()

        for i in range(50):
            row        = QWidget(self._tags_container)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            checkBox = QCheckBox(f"TAG {i}", row)
            checkBox.setObjectName(f"checkBox_Tag{i}")
            checkBox.setChecked(True)
            row_layout.addWidget(checkBox)
            self.checkboxes.append(checkBox)

            for axis in ("x", "y"):
                lbl = QLabel(f"pos {axis} :", row)
                lbl.setObjectName(f"label_pos{axis}_{i}")
                lbl.setMinimumSize(50, 30)
                lbl.setMaximumSize(70, 30)
                lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                row_layout.addWidget(lbl)

                val = QLabel("?", row)
                val.setObjectName(f"label_pos{axis}_nbr_{i}")
                val.setMinimumSize(50, 30)
                val.setMaximumSize(70, 30)
                row_layout.addWidget(val)

            self._tags_layout.addWidget(row)

            sep = QFrame(self._tags_container)
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            self._tags_layout.addWidget(sep)

        self._tags_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Checkbox signals
    # ------------------------------------------------------------------
    def on_display_grid_checkbox_changed(self, _state):
        if self.algorithm_analysis:
            self.algorithm_analysis.set_show_grid(self.checkBox_DisplayGrid.isChecked())
        else:
            self.refresh_projector_background()

    def connect_checkbox_to_algorithm(self):
        if self.algorithm_analysis:
            self.checkBox_Visu_Cam.stateChanged.connect(
                self.algorithm_analysis.state_popUpCamera_changed
            )

    # ------------------------------------------------------------------
    # Export session
    # ------------------------------------------------------------------
    def export_session_summary(self, session_id: str):
        protocol_id    = getattr(self, "active_protocol_id", None)
        participant_id = getattr(self, "active_participant_id", "P001")
        protocol_name  = "UNKNOWN_PROTOCOL"

        if protocol_id:
            from src.core.storage.db import connect
            conn = connect()
            try:
                row = conn.execute(
                    "SELECT name FROM protocols WHERE id = ?", (protocol_id,)
                ).fetchone()
                if row:
                    protocol_name = row["name"]
            finally:
                conn.close()

        safe_proto = protocol_name.replace(" ", "_")
        safe_pid   = participant_id.replace(" ", "_")
        default_name = f"{safe_proto}_{safe_pid}.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la session",
            os.path.join(os.getcwd(), default_name),
            "JSON (*.json)"
        )
        if not path:
            return

        payload = {
            "session_id":     session_id,
            "protocol_id":    protocol_id,
            "protocol_name":  protocol_name,
            "participant_id": participant_id,
            "exported_at":    __import__("datetime").datetime.now().isoformat(timespec="seconds")
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        QtWidgets.QMessageBox.information(self, "Export", f"Export OK:\n{path}")