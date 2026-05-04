# -*- coding: utf-8 -*-
"""
algorithm_analysis.py — VERSION v4
====================================
Changements vs v3 :
  - associate_hands_to_cups() supprimée
  - Nouveau : logique CSRT dans detect_and_process()
      · quand une tasse passe SOULEVEE → CupTopTracker.start() via bbox projetée
      · chaque frame → CupTopTracker.update(frame_top) → pos_mm
      · quand ArUco réapparaît → CupTopTracker.stop()
  - Nouveau : FrameBuffer injecté (frame_top disponible à chaque frame)
  - HandStateBuffer conservé mais silencieux — prêt pour la feature dessin
  - _carrier_lock_frames, _assigned_hand_ids, CARRIER_LOCK_DURATION supprimés
  - Couleurs inchangées : Rouge=POSEE, Orange=INCERT, Bleu=SOULEVEE
"""

import os
import cv2
import json
import numpy as np
import time as pytime
import pandas as pd
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal

from src.core.utils.paths import config_path, data_path
from src.core.projection.display_manager import DisplayManager
from src.core.projection.draw_utils import DrawUtils
from src.core.Hand_tracking.hand_state_buffer import HandStateBuffer
from src.core.Hand_tracking.frame_buffer import FrameBuffer               # ← NOUVEAU
from src.core.Hand_tracking.cup_top_tracker import CupTopTracker          # ← NOUVEAU


COLOR_CUP_POSEE    = (0, 0, 255)
COLOR_CUP_SOULEVEE = (255, 0, 0)
COLOR_CUP_INCERT   = (0, 165, 255)

CIRCLE_RADIUS_GRAPH   = 14
RING_RADIUS_FACTOR    = 3.0
RING_THICKNESS_FACTOR = 0.18

TABLE_SIZE_MM = 597.0

# Taille de la bbox d'initialisation CSRT en pixels (cam_top 1920x1080)
# Une tasse de ~80mm de diamètre vue de 1m ≈ 80–120 px selon la hauteur caméra
# À ajuster selon ta configuration réelle
CUP_BBOX_PX = 60  # dessus interieur tasse uniquement (~50mm a 890mm hauteur)


class Algorithm_Analysis(QObject):
    data_signal     = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_manager: DisplayManager,
        image_background,
        H_projector=None,
        H_inv_projector=None,
        H_graph=None,
        H_inv_graph=None,
        grid_size=700,
        output_name="data",
        record_window=None,
        output_dir=None,
        modules_enabled=None,
        assets=None,
        timeline_steps=None,
        protocol=None,
        hand_buffer: HandStateBuffer = None,
        frame_buffer: FrameBuffer = None,                                  # ← NOUVEAU
    ):
        super().__init__()

        self.parent          = parent
        self.display_manager = display_manager
        self.grid_size       = int(grid_size)

        self.modules_enabled = modules_enabled or {}
        self.hand_tracking_enabled = self.is_enabled("hand_tracking", False)
        self.assets          = assets or []
        self.timeline_steps  = timeline_steps or []
        self.protocol        = protocol
        self.record_window   = record_window
        self.TABLE_SIZE_MM   = TABLE_SIZE_MM

        # Buffers partagés
        self.hand_buffer:  HandStateBuffer = hand_buffer  or HandStateBuffer(max_age_ms=250)
        self.frame_buffer: FrameBuffer     = frame_buffer or FrameBuffer(max_age_ms=200)  # ← NOUVEAU

        # Tracker CSRT — une seule tasse levée à la fois
        _pose_path = config_path("camtop_table_pose.json")
        self.cup_tracker = CupTopTracker(pose_path=str(_pose_path))        # ← NOUVEAU

        pose      = json.load(open(config_path("cambottom_table_pose.json")))
        self.rvec = np.array(pose["rvec"], dtype=np.float64)
        self.tvec = np.array(pose["tvec"], dtype=np.float64)
        self.K    = (
            parent.runtime_K
            if (hasattr(parent, "runtime_K") and parent.runtime_K is not None)
            else np.array(pose["camera_matrix"], dtype=np.float64)
        )

        H_data               = json.load(open(config_path("H_table_to_proj.json")))
        self.H_table_to_proj = np.array(H_data["H_table_to_proj"], dtype=np.float32)

        self.image_background               = self._validate_background_image(image_background)
        self.image_height, self.image_width = self.image_background.shape[:2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            os.makedirs(data_path(), exist_ok=True)
            output_dir = data_path()
        else:
            os.makedirs(output_dir, exist_ok=True)
        self.output_csv = os.path.join(output_dir, f"{output_name}_{timestamp}.csv")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector   = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.data_buffer              = []
        self.running                  = False
        self.status_popUpCamera       = False
        self.show_grid                = False
        self.waiting_for_consigne_key = False
        self.marker_states            = {}
        self.move_threshold           = 10.0
        self.stable_count_required    = 5
        self.extra_dimensions         = {}
        self.current_marker_id        = None
        self.mode_multidim            = True
        self.colormap                 = cv2.COLORMAP_COOL

        self.cups              = {}
        self.N_LIFT            = 5
        self.N_LIFT_CONFIRM    = 3
        self.N_POSE_CONFIRM    = 6
        self.get_hands         = None   # conservé pour compatibilité

        self.HANDS_MAX_AGE_FRAMES: int = 10
        self._hands_last_frame:    int = -999
        self._frame_counter:       int = 0

    # ======================================================================
    # Compatibilité v3 (no-op en v4)
    # ======================================================================
    def set_hands_provider(self, func):
        """Conservé pour compatibilité avec record_window. Sans effet en v4."""
        self.get_hands = func

    # ======================================================================
    # Géométrie
    # ======================================================================
    def table_to_projector(self, x_mm, y_mm):
        pt   = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, self.H_table_to_proj)
        return proj[0, 0]

    def pixel_to_table(self, u, v):
        K_inv = np.linalg.inv(self.K)
        ray   = K_inv @ np.array([u, v, 1.0])
        ray   = ray / np.linalg.norm(ray)
        R, _         = cv2.Rodrigues(self.rvec)
        normal       = R[:, 2]
        plane_origin = self.tvec.reshape(3)
        denom = np.dot(normal, ray)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(normal, plane_origin) / denom
        if t < 0:
            return None
        pt_cam   = ray * t
        pt_table = R.T @ (pt_cam - self.tvec.reshape(3))
        return float(pt_table[0]), float(pt_table[1])

    # ======================================================================
    # Couleur
    # ======================================================================
    def _cup_color_for_state(self, cup: dict) -> tuple:
        state = cup.get("state", "POSEE")
        if state == "POSEE":
            return COLOR_CUP_POSEE
        if state == "PEUT_ETRE_SOULEVEE":
            return COLOR_CUP_INCERT
        if state == "SOULEVEE":
            return COLOR_CUP_SOULEVEE
        return COLOR_CUP_POSEE

    # ======================================================================
    # Coords graphe
    # ======================================================================
    def _cup_to_graph_coords(self, cup: dict):
        x_raw, y_raw = cup["last_pos"]
        if cup.get("pos_is_top_mm", False):
            # pos_is_top_mm=True → last_pos est déjà en mm table
            xg = int((x_raw / self.TABLE_SIZE_MM) * self.grid_size)
            yg = int((y_raw / self.TABLE_SIZE_MM) * self.grid_size)
        else:
            xg = int((x_raw / self.TABLE_SIZE_MM) * self.grid_size)
            yg = int((y_raw / self.TABLE_SIZE_MM) * self.grid_size)
        if 0 <= xg < self.grid_size and 0 <= yg < self.grid_size:
            return xg, yg
        return None

    # ======================================================================
    # Grille projetée
    # ======================================================================
    def _build_warped_grid(self, proj_w, proj_h, cups=None, hands=None):
        x_min, x_max, y_min, y_max, x_legend, y_legend = \
            self.record_window.get_bounds_from_inputs()
        graph_grid = np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)
        graph_grid = DrawUtils.draw_math_grid_on_image(
            graph_grid, x_min, x_max, y_min, y_max, x_legend, y_legend, self.grid_size
        )
        if cups:
            for marker_id, cup in cups.items():
                coords = self._cup_to_graph_coords(cup)
                if coords is None:
                    continue
                xg, yg = coords
                color  = self._cup_color_for_state(cup)
                cv2.circle(graph_grid, (xg, yg), CIRCLE_RADIUS_GRAPH, color, -1)
                overlay = graph_grid.copy()
                cv2.circle(overlay, (xg, yg), CIRCLE_RADIUS_GRAPH + 4, color, 2)
                cv2.addWeighted(overlay, 0.4, graph_grid, 0.6, 0, graph_grid)
        return cv2.warpPerspective(graph_grid, self.H_table_to_proj, (proj_w, proj_h))

    def _draw_cup_ring_on_projector(self, img, projector_x, projector_y, marker_size, cup):
        color          = self._cup_color_for_state(cup)
        ring_radius    = int(marker_size * RING_RADIUS_FACTOR)
        ring_thickness = max(4, int(marker_size * RING_THICKNESS_FACTOR))
        overlay = img.copy()
        cv2.circle(overlay, (projector_x, projector_y), ring_radius, color, ring_thickness)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # ======================================================================
    # Helpers
    # ======================================================================
    def _validate_background_image(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image_background invalide : type={type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image_background invalide : shape={image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def is_enabled(self, key: str, default: bool = False) -> bool:
        v = self.modules_enabled.get(key, default)
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def update_background_image(self, new_image_background):
        new_image_background               = self._validate_background_image(new_image_background)
        self.image_background              = new_image_background
        self.image_height, self.image_width = self.image_background.shape[:2]

    def set_show_grid(self, show: bool):
        self.show_grid = bool(show)

    def on_consigne_key_pressed(self):
        self.waiting_for_consigne_key = False

    def state_popUpCamera_changed(self):
        self.status_popUpCamera = not self.status_popUpCamera

    def stop(self):
        print("[ALGO] STOP demandé")
        self.running = False
        self.cup_tracker.stop()                                            # ← NOUVEAU
        pytime.sleep(0.01)
        self.waiting_for_consigne_key = False

    # ======================================================================
    # Config / assets (inchangé)
    # ======================================================================
    def load_config(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Fichier config introuvable : {path}")
            return {}
        except Exception as e:
            print(f"⚠️ Erreur lecture config ({path}) : {e}")
            return {}

    def prepare_instruction_image(self, img, margin_ratio=0.04):
        target_h = self.image_height
        target_w = self.image_width
        canvas   = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        src_h, src_w = img.shape[:2]
        scale    = min(
            int(target_w * (1 - 2 * margin_ratio)) / src_w,
            int(target_h * (1 - 2 * margin_ratio)) / src_h,
        )
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _show_timeline_assets(self):
        if not self.is_enabled("projection_media", False):
            return
        if not self.timeline_steps:
            return
        assets_by_id = {a.id: a for a in self.assets}
        for step in self.timeline_steps:
            if not self.running:
                break
            if step.pause or step.asset_ref is None:
                t0 = datetime.now()
                while self.running and (datetime.now() - t0).total_seconds() < step.duration_s:
                    self.display_manager.display_image_on_projector_monitor(self.image_background)
                    _ = self.parent.camera_manager.get_frame()
                    cv2.waitKey(1)
                continue
            asset = assets_by_id.get(step.asset_ref)
            if not asset:
                continue
            if asset.asset_type == "image":
                img = cv2.imread(asset.path)
                if img is None:
                    continue
                img = self.prepare_instruction_image(img)
                self.waiting_for_consigne_key = True
                while self.running and self.waiting_for_consigne_key:
                    self.display_manager.display_image_on_projector_monitor(img)
                    _ = self.parent.camera_manager.get_frame()
                    cv2.waitKey(1)

    # ======================================================================
    # Détection ArUco (inchangé)
    # ======================================================================
    def _detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def _update_marker_state(self, marker_id_int, camera_point):
        state = self.marker_states.get(
            marker_id_int,
            {"last_pos": camera_point.copy(), "stable_count": 0, "is_static": False},
        )
        dist = float(np.linalg.norm(camera_point - state["last_pos"]))
        if dist < self.move_threshold:
            state["stable_count"] += 1
            if state["stable_count"] >= self.stable_count_required:
                state["is_static"] = True
        else:
            state["stable_count"]  = 0
            state["is_static"]     = False
            self.current_marker_id = marker_id_int
        state["last_pos"] = camera_point.copy()
        self.marker_states[marker_id_int] = state
        return state

    # ======================================================================
    # Dessin overlays RA (inchangé)
    # ======================================================================
    def draw_from_config(self, img, marker_config, projector_x, projector_y, marker_size, marker_id=None):
        for element in marker_config:
            if element["type"] == "circle":
                self.draw_circle_from_config(img, element, projector_x, projector_y, marker_size, marker_id)
            elif element["type"] == "line":
                self.draw_line_from_config(img, element, projector_x, projector_y, marker_size, marker_id)
            elif element["type"] == "text":
                self.draw_text_from_config(img, element, projector_x, projector_y, marker_size)

    def draw_circle_from_config(self, img, element, px, py, ms, marker_id=None):
        r  = int(element["relative_size"]["radius"] * ms)
        ax = int(px + element["relative_position"]["x"] * ms)
        ay = int(py - element["relative_position"]["y"] * ms)
        color     = self.get_marker_color(marker_id) if (self.mode_multidim and marker_id is not None) else self.parse_color(element.get("color", "#FFFFFF"))
        fill      = element.get("fill", False)
        thickness = -1 if fill else max(1, int(element["relative_size"]["thickness"] * ms))
        cv2.circle(img, (ax, ay), max(1, r), color, thickness)

    def draw_line_from_config(self, img, element, px, py, ms, marker_id=None):
        ax1 = int(px + element["relative_position"]["x1"] * ms)
        ay1 = int(py - element["relative_position"]["y1"] * ms)
        ax2 = int(px + element["relative_position"]["x2"] * ms)
        ay2 = int(py - element["relative_position"]["y2"] * ms)
        color     = self.get_marker_color(marker_id) if (self.mode_multidim and marker_id is not None) else self.parse_color(element.get("color", "#FFFFFF"))
        thickness = max(1, int(element["thickness"] * ms))
        cv2.line(img, (ax1, ay1), (ax2, ay2), color, thickness)

    def draw_text_from_config(self, img, element, px, py, ms):
        ax       = int(px + element["relative_position"]["x"] * ms)
        ay       = int(py - element["relative_position"]["y"] * ms)
        fs       = max(1, int(element["font_size"] * ms))
        rotation = -element.get("rotation", 0)
        color    = self.parse_color(element.get("color", "#FFFFFF"))
        text     = element["text"]
        tw, th   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs / 30, 2)[0]
        cx, cy   = ax - tw // 2, ay + th // 2
        if rotation != 0:
            rm   = cv2.getRotationMatrix2D((ax, ay), rotation, 1.0)
            tmp  = np.zeros_like(img)
            cv2.putText(tmp, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs / 30, color, 2)
            rot  = cv2.warpAffine(tmp, rm, (img.shape[1], img.shape[0]))
            mask = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
            _, mask  = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            img[:]   = cv2.add(cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)), rot)
        else:
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs / 30, color, 2)

    def draw_default_marker(self, img, px, py, ms, marker_id):
        radius  = max(20, int(ms * 0.6))
        overlay = img.copy()
        color   = self.get_marker_color(marker_id)
        cv2.circle(overlay, (px, py), radius, color, 8)
        cv2.putText(overlay, f"ID: {marker_id}", (px - 50, py + radius + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    def parse_color(self, color_str):
        try:
            if color_str.startswith("#") and len(color_str) == 7:
                r, g, b = (int(color_str[i:i+2], 16) for i in (1, 3, 5))
                return (b, g, r)
        except Exception as e:
            print(f"Error processing color: {e}")
        return (255, 255, 255)

    def show_camera_window(self, frame, corners=None):
        preview = frame.copy()
        if corners is not None:
            for mc in corners:
                pts = mc[0].astype(int)
                for i in range(4):
                    cv2.line(preview, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 4)
                cv2.circle(preview, tuple(np.mean(pts, axis=0).astype(int)), 10, (0, 0, 255), -1)
        cv2.imshow("Camera", cv2.resize(preview, (1080, 720), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)

    # ======================================================================
    # Export (inchangé)
    # ======================================================================
    def save_to_buffer(self, graph_coords_ArUco):
        frame_data = {
            "frame":     len(self.data_buffer) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for entry in graph_coords_ArUco:
            marker_id, coords, *_ = entry
            frame_data[f"ID_{marker_id}_x"] = float(coords[0])
            frame_data[f"ID_{marker_id}_y"] = float(coords[1])
            extra = self.extra_dimensions.get(marker_id, [0.0])
            for d, val in enumerate(extra):
                frame_data[f"ID_{marker_id}_dim{d+1}"] = float(val)
        self.data_buffer.append(frame_data)

    def save_to_csv(self):
        pd.DataFrame(self.data_buffer).to_csv(self.output_csv, index=False)
        print(f"Données sauvegardées dans {self.output_csv}")

    # ======================================================================
    # MultiDim (inchangé)
    # ======================================================================
    def select_next_marker(self, direction):
        if not self.marker_states:
            return
        ids = sorted(self.marker_states.keys())
        if self.current_marker_id not in ids:
            self.current_marker_id = ids[0]
        else:
            idx = (ids.index(self.current_marker_id) + direction) % len(ids)
            self.current_marker_id = ids[idx]

    def modify_current_marker_dimension(self, delta):
        if self.current_marker_id is None:
            return
        dims    = self.extra_dimensions.get(self.current_marker_id, [0.0])
        dims[0] = max(-10, min(10, dims[0] + delta))
        self.extra_dimensions[self.current_marker_id] = dims

    def custom_colormap_3(self, norm_value, col1=None, col2=None, col3=None):
        depart = col1 or (80, 0, 0)
        milieu = col2 or (0, 0, 255)
        fin    = col3 or (255, 235, 255)
        if norm_value < 0.5:
            t = norm_value / 0.5
        else:
            t      = (norm_value - 0.5) / 0.5
            depart = milieu
            milieu = fin
        return tuple(int(depart[i] * (1 - t) + milieu[i] * t) for i in range(3))

    def get_marker_color(self, marker_id):
        if not self.mode_multidim:
            return (0, 0, 255)
        value      = self.extra_dimensions.get(marker_id, [0.0])[0]
        norm_value = np.clip((value + 10) / 20, 0, 1)
        return self.custom_colormap_3(norm_value, (0,0,0), (0,0,255), (255,235,255))

    def draw_dim_value(self, img, x, y, marker_id, color):
        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        cv2.putText(img, f"Dim: {value:.1f}", (int(x)-50, int(y)-150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ======================================================================
    # Cup tracking — cam_bottom (inchangé vs v3)
    # ======================================================================
    def update_cups_bottom(self, detected_markers: dict):
        N_LIFT_CONFIRM = getattr(self, "N_LIFT_CONFIRM", 3)
        N_POSE_CONFIRM = getattr(self, "N_POSE_CONFIRM", 6)

        for marker_id, pos in detected_markers.items():
            if marker_id not in self.cups:
                self.cups[marker_id] = {
                    "state":          "POSEE",
                    "last_pos":       pos.copy(),
                    "lost_frames":    0,
                    "lift_frames":    0,
                    "pose_frames":    0,
                    "pos_is_top_mm":  False,
                }

            cup = self.cups[marker_id]
            cup["lost_frames"] = 0

            if cup["state"] in ("SOULEVEE", "PEUT_ETRE_SOULEVEE"):
                cup["pose_frames"] = cup.get("pose_frames", 0) + 1
                if cup["pose_frames"] >= N_POSE_CONFIRM:
                    cup["last_pos"]        = pos.copy()
                    cup["lift_frames"]     = 0
                    cup["pose_frames"]     = 0
                    cup["state"]           = "POSEE"
                    cup["pos_is_top_mm"]   = False
                else:
                    cup["last_pos"]        = pos.copy()
                    cup["pos_is_top_mm"]   = False
            else:
                cup["last_pos"]        = pos.copy()
                cup["lift_frames"]     = 0
                cup["pose_frames"]     = 0
                cup["state"]           = "POSEE"
                cup["pos_is_top_mm"]   = False

        for marker_id, cup in self.cups.items():
            if marker_id in detected_markers:
                continue
            cup["pose_frames"] = 0
            cup["lost_frames"] += 1
            if cup["state"] == "POSEE":
                cup["state"]       = "PEUT_ETRE_SOULEVEE"
                cup["lift_frames"] = 1
            elif cup["state"] == "PEUT_ETRE_SOULEVEE":
                cup["lift_frames"] += 1
                if cup["lift_frames"] >= N_LIFT_CONFIRM:
                    cup["state"] = "SOULEVEE"

        # Règle métier : max 1 tasse soulevée (inchangé)
        soulevees = [mid for mid, c in self.cups.items() if c["state"] == "SOULEVEE"]
        if len(soulevees) > 1:
            soulevees.sort(key=lambda mid: self.cups[mid].get("lift_frames", 0), reverse=True)
            for mid in soulevees[1:]:
                self.cups[mid]["state"]        = "POSEE"
                self.cups[mid]["pos_is_top_mm"] = False

    # ======================================================================
    # Cup tracking — cam_top CSRT                                 ← NOUVEAU
    # ======================================================================
    def _init_csrt_for_cup(self, marker_id: int, frame_top: np.ndarray) -> bool:
        """
        Initialise le tracker CSRT quand une tasse passe à l'état SOULEVEE.

        La bbox est construite en projetant la dernière position ArUco connue
        (en mm) dans l'espace pixel de la cam_top via cup_tracker.table_mm_to_pixel().
        """
        cup = self.cups.get(marker_id)
        if cup is None:
            return False

        x_mm, y_mm = cup["last_pos"]

        # Projeter la position mm → pixel cam_top
        px, py = self.cup_tracker.table_mm_to_pixel(x_mm, y_mm)
        half   = CUP_BBOX_PX // 2
        bbox   = (int(px - half), int(py - half), CUP_BBOX_PX, CUP_BBOX_PX)

        ok = self.cup_tracker.start(frame_top, bbox, cup_id=marker_id)
        return ok

    def _update_csrt(self, frame_top: np.ndarray):
        """
        Met à jour le tracker CSRT et écrit la position dans cups[].

        Appelé à chaque frame quand au moins une tasse est SOULEVEE.
        Si le tracker échoue, la tasse garde sa dernière position connue.
        """
        if not self.cup_tracker.is_active:
            return

        ok, pos_mm = self.cup_tracker.update(frame_top)
        marker_id  = self.cup_tracker.cup_id

        if marker_id not in self.cups:
            self.cup_tracker.stop()
            return

        cup = self.cups[marker_id]

        if ok and pos_mm is not None:
            cup["last_pos"]      = list(pos_mm)   # (x_mm, y_mm) repère table
            cup["pos_is_top_mm"] = True
        else:
            # Tracker perdu → on garde la dernière position, on ne stoppe pas
            # (peut se récupérer si le drift était temporaire)
            # Le stop se fera quand ArUco réapparaît (dans update_cups_bottom)
            pass

    # ======================================================================
    # Boucle principale
    # ======================================================================
    def detect_and_process(self):
        from src.core.config.app_config import CALIBRATION_TAG_IDS

        self.running = True
        print("detect_and_process started — v4 CSRT")

        self._fps_counter_bottom = 0
        self._fps_time_bottom    = pytime.time()

        overlay_on    = self.is_enabled("overlay_ra",       False)
        projection_on = self.is_enabled("projection_media", False)

        ra_config = self.load_config(config_path("ra_config.json")) if overlay_on else {}

        self._show_timeline_assets()
        print("Consignes affichées — démarrage détection.")

        # État CSRT : on track quel marker_id est actuellement suivi
        _csrt_tracking_id: int = -1  # -1 = aucun

        while self.running:
            self._frame_counter += 1
            self._fps_counter_bottom += 1
            now = pytime.time()
            if now - self._fps_time_bottom >= 1.0:
                print(f"[FPS] cam_bottom = {self._fps_counter_bottom} | "
                      f"frame_buffer_age = {self.frame_buffer.last_age_ms} ms")
                self._fps_counter_bottom = 0
                self._fps_time_bottom    = now

            proj_w = 3840
            proj_h = 2160
            current_image_background = np.ones((proj_h, proj_w, 3), dtype=np.uint8) * 255

            frame = self.parent.camera_manager.get_frame()
            if frame is None or frame.size == 0:
                continue

            corners, ids       = self._detect_markers(frame)
            graph_coords_ArUco = []
            detected_markers   = {}
            marker_size        = 40

            if ids is not None and len(ids) > 0:
                valid_indices = [
                    i for i in range(len(ids))
                    if int(ids[i][0]) not in CALIBRATION_TAG_IDS
                ]
                if valid_indices:
                    ids     = ids[valid_indices]
                    corners = [corners[i] for i in valid_indices]
                else:
                    ids     = None
                    corners = []

            if ids is not None:
                for i, corner in enumerate(corners):
                    marker_id_int = int(ids[i][0])
                    camera_point  = self._get_marker_anchor_raw_center(corner)
                    state         = self._update_marker_state(marker_id_int, camera_point)

                    pts = corner[0]
                    cx  = float(np.mean(pts[:, 0]))
                    cy  = float(np.mean(pts[:, 1]))
                    mm  = self.pixel_to_table(cx, cy)
                    if mm is None:
                        continue

                    x_mm, y_mm  = mm
                    proj        = self.table_to_projector(x_mm, y_mm)
                    projector_x = int(proj[0])
                    projector_y = int(proj[1])

                    graph_coords_ArUco.append([marker_id_int, [x_mm, y_mm]])
                    detected_markers[marker_id_int] = np.array([x_mm, y_mm], dtype=np.float32)

                    if state["is_static"] and overlay_on:
                        if f"marker_{marker_id_int}" in ra_config:
                            self.draw_from_config(
                                current_image_background,
                                ra_config[f"marker_{marker_id_int}"],
                                projector_x, projector_y, marker_size, marker_id_int,
                            )
                        else:
                            self.draw_default_marker(
                                current_image_background, projector_x, projector_y,
                                marker_size, marker_id_int,
                            )
                        if self.mode_multidim:
                            self.draw_dim_value(
                                current_image_background, projector_x, projector_y,
                                marker_id_int, self.get_marker_color(marker_id_int),
                            )

            if self.hand_tracking_enabled:
                # ── 1. Mise à jour états depuis cam_bottom ──────────────
                self.update_cups_bottom(detected_markers)

                # ── 2. Récupérer la frame cam_top ───────────────────────
                frame_top = self.frame_buffer.get_latest()

                # ── 3. Gestion du tracker CSRT ──────────────────────────
                if frame_top is not None:

                    # Chercher si une tasse vient de passer SOULEVEE
                    # et que le CSRT n'est pas encore actif pour elle.
                    # On attend CSRT_MIN_LOST_FRAMES frames sans ArUco avant
                    # d'init le CSRT — évite le stop immédiat dû au clignotement
                    # ArUco en bordure de levée.
                    CSRT_MIN_LOST_FRAMES = self.N_POSE_CONFIRM  # même seuil que repose
                    for marker_id, cup in self.cups.items():
                        if (cup["state"] == "SOULEVEE"
                                and cup.get("lost_frames", 0) >= CSRT_MIN_LOST_FRAMES
                                and not self.cup_tracker.is_active
                                and marker_id != _csrt_tracking_id):
                            ok = self._init_csrt_for_cup(marker_id, frame_top)
                            if ok:
                                _csrt_tracking_id = marker_id
                            break

                    # Mettre à jour le tracker si actif
                    if self.cup_tracker.is_active:
                        self._update_csrt(frame_top)

                    # Stopper le tracker si ArUco est revenu de façon stable.
                    # On utilise pose_frames (incrémenté dans update_cups_bottom
                    # quand ArUco est détecté pendant état SOULEVEE) comme garde.
                    if self.cup_tracker.is_active:
                        tracked_id = self.cup_tracker.cup_id
                        if tracked_id in detected_markers:
                            cup = self.cups.get(tracked_id, {})
                            if cup.get("pose_frames", 0) >= self.N_POSE_CONFIRM:
                                print(f"[CSRT] ArUco stable ({self.N_POSE_CONFIRM} frames) "
                                      f"— cup_id={tracked_id}, CSRT stoppé")
                                self.cup_tracker.stop()
                                _csrt_tracking_id = -1

                else:
                    # Pas de frame top disponible — si le tracker était actif,
                    # on garde la dernière position connue (pas de stop brutal)
                    if self.frame_buffer.last_age_ms > 500 and self.cup_tracker.is_active:
                        print("[CSRT] frame_top indisponible > 500 ms — stop tracker")
                        self.cup_tracker.stop()
                        _csrt_tracking_id = -1

            else:
                # Mode hand_tracking désactivé : tasses = positions ArUco uniquement
                self.cups = {
                    marker_id: {
                        "state":         "POSEE",
                        "last_pos":      pos.copy(),
                        "lost_frames":   0,
                        "lift_frames":   0,
                        "pos_is_top_mm": False,
                    }
                    for marker_id, pos in detected_markers.items()
                }

            # ── 4. Rendu projecteur ─────────────────────────────────────
            for marker_id, cup in self.cups.items():
                x_mm, y_mm = cup["last_pos"]
                proj = self.table_to_projector(x_mm, y_mm)
                self._draw_cup_ring_on_projector(
                    current_image_background,
                    int(proj[0]), int(proj[1]),
                    marker_size=marker_size, cup=cup,
                )

            if self.show_grid and self.record_window is not None:
                warped_grid = self._build_warped_grid(proj_w, proj_h, cups=self.cups)
                current_image_background = cv2.addWeighted(
                    warped_grid, 1.0, current_image_background, 1.0, 0
                )

            if self.status_popUpCamera:
                self.show_camera_window(frame, corners=corners if ids is not None else None)
                self._camera_was_active = True
            elif getattr(self, "_camera_was_active", False):
                cv2.destroyWindow("Camera")
                self._camera_was_active = False

            graph_coords_fusion = [
                [marker_id, cup["last_pos"], cup.get("state", "POSEE")]
                for marker_id, cup in self.cups.items()
            ]
            self.display_manager.display_image_on_projector_monitor(current_image_background)
            self.data_signal.emit({"data": graph_coords_fusion})
            self.save_to_buffer(graph_coords_fusion)

        self.save_to_csv()
        print("detect_and_process terminé")
        self.finished_signal.emit()

    # ======================================================================
    # Helpers anchor (inchangé)
    # ======================================================================
    def _select_physical_corner_runtime(self, pts, position):
        if position == "TL":
            return pts[np.argmin(pts[:, 0] + pts[:, 1])].astype(np.float32)
        if position == "TR":
            return pts[np.argmin(-pts[:, 0] + pts[:, 1])].astype(np.float32)
        if position == "BR":
            return pts[np.argmax(pts[:, 0] + pts[:, 1])].astype(np.float32)
        if position == "BL":
            return pts[np.argmax(-pts[:, 0] + pts[:, 1])].astype(np.float32)
        raise ValueError(f"Position inconnue: {position}")

    def _get_marker_anchor_raw(self, corner):
        return self._select_physical_corner_runtime(corner[0].astype(np.float32), "TL")

    def _get_marker_anchor_raw_center(self, corner):
        return np.mean(corner[0].astype(np.float32), axis=0)