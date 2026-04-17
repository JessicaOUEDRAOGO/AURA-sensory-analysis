# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal

from src.core.utils.paths import config_path, data_path
from src.core.projection.display_manager import DisplayManager
from src.core.projection.draw_utils import DrawUtils
from src.core.mapping.coordinate_mapper import CoordinateMapper


class Algorithm_Analysis(QObject):
    data_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_manager: DisplayManager,
        image_background,
        H_projector=None,       # compat legacy
        H_inv_projector=None,   # compat legacy
        H_graph=None,           # compat legacy
        H_inv_graph=None,       # compat legacy
        grid_size=700,
        output_name="data",
        record_window=None,
        output_dir=None,
        modules_enabled=None,
        assets=None,
        timeline_steps=None,
        protocol=None
    ):
        super().__init__()

        self.parent = parent
        self.display_manager = display_manager
        self.grid_size = int(grid_size)

        self.modules_enabled = modules_enabled or {}
        self.assets = assets or []
        self.timeline_steps = timeline_steps or []
        self.protocol = protocol
        self.record_window = record_window

        # ------------------------------------------------------------------
        # Image de fond / média
        # ------------------------------------------------------------------
        self.image_background = self._validate_background_image(image_background)
        self.image_height, self.image_width = self.image_background.shape[:2]

        # ------------------------------------------------------------------
        # Sortie CSV
        # ------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            os.makedirs(data_path(), exist_ok=True)
            output_dir = data_path()
        else:
            os.makedirs(output_dir, exist_ok=True)

        self.output_csv = os.path.join(output_dir, f"{output_name}_{timestamp}.csv")

        # ------------------------------------------------------------------
        # Mapper géométrique
        # ------------------------------------------------------------------
        self.mapper = None

        try:
            self.mapper = CoordinateMapper()
            self.mapper.load()
            print("[RUNTIME] CoordinateMapper chargé avec succès.")
            print("[RUNTIME] H_graph_to_proj chargée depuis calibration_data.json.")
        except Exception as e:
            print(f"[WARNING] Impossible de charger CoordinateMapper : {e}")

        # ------------------------------------------------------------------
        # ArUco
        # ------------------------------------------------------------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # ------------------------------------------------------------------
        # États runtime
        # ------------------------------------------------------------------
        self.data_buffer = []
        self.running = False
        self.status_popUpCamera = False
        self.show_grid = False
        self.waiting_for_consigne_key = False

        # Stabilité des marqueurs
        self.marker_states = {}
        self.move_threshold = 10.0
        self.stable_count_required = 5

        # MultiDim
        self.extra_dimensions = {}
        self.current_marker_id = None
        self.mode_multidim = True
        self.colormap = cv2.COLORMAP_COOL

        # ------------------------------------------------------------------
        # Compatibilité legacy
        # ------------------------------------------------------------------
        self.H_projector = H_projector
        self.H_inv_projector = H_inv_projector
        self.H_graph = H_graph
        self.H_inv_graph = H_inv_graph
        self.projector_offset_x = 30
        self.projector_offset_y = -40

        self.data_buffer = []
        self.running = False

        # =========================
        # Multi-cam tracking V2
        # =========================
        self.cups = {}

        self.N_LIFT = 5                # nb frames avant levée
        self.DIST_HAND_THRESHOLD = 80  # (utilisé plus tard)
        self.DIST_RECOVERY = 50        # (utilisé plus tard)
    # ======================================================================
    # Validation / helpers
    # ======================================================================
    def _apply_projector_offset(self, pt):
        """
        Applique un décalage global manuel dans le repère projecteur.
        """
        pt = np.array(pt, dtype=np.float32).copy()
        pt[0] += float(self.projector_offset_x)
        pt[1] += float(self.projector_offset_y)
        return pt

    def _validate_background_image(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image_background invalide : type={type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image_background invalide : shape={image.shape}. "
                "Attendu une image couleur (H, W, 3)."
            )

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
        new_image_background = self._validate_background_image(new_image_background)
        self.image_background = new_image_background
        self.image_height, self.image_width = self.image_background.shape[:2]

    def set_show_grid(self, show: bool):
        self.show_grid = bool(show)

    def on_consigne_key_pressed(self):
        self.waiting_for_consigne_key = False

    def state_popUpCamera_changed(self):
        self.status_popUpCamera = not self.status_popUpCamera

    def stop(self):
        self.running = False
        self.waiting_for_consigne_key = False

    # ======================================================================
    # Config / assets
    # ======================================================================
    def load_config(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            print(f"⚠️ Fichier config introuvable : {path}")
            return {}
        except Exception as e:
            print(f"⚠️ Erreur lecture config ({path}) : {e}")
            return {}

    def prepare_instruction_image(self, img, margin_ratio=0.04):
        """
        Place l'image dans un canvas noir avec marge de sécurité,
        sans déformation et sans tronquage.
        """
        target_h = self.image_height
        target_w = self.image_width

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        src_h, src_w = img.shape[:2]

        usable_w = int(target_w * (1.0 - 2 * margin_ratio))
        usable_h = int(target_h * (1.0 - 2 * margin_ratio))

        scale = min(usable_w / src_w, usable_h / src_h)

        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2

        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _show_timeline_assets(self):
        """
        Affiche les éventuelles consignes / médias de timeline.
        """
        projection_on = self.is_enabled("projection_media", False)
        adv_logs = self.is_enabled("advanced_logs", False)

        if not projection_on:
            if adv_logs:
                print("[MODULES] projection_media=OFF -> consignes ignorées")
            return

        if not self.timeline_steps:
            if adv_logs:
                print("[MODULES] projection_media=ON mais timeline_steps vide -> aucune consigne")
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

                img = self.prepare_instruction_image(img, margin_ratio=0.04)

                self.waiting_for_consigne_key = True
                while self.running and self.waiting_for_consigne_key:
                    self.display_manager.display_image_on_projector_monitor(img)
                    _ = self.parent.camera_manager.get_frame()
                    cv2.waitKey(1)

    # ======================================================================
    # Détection / stabilité
    # ======================================================================
    def _detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def _update_marker_state(self, marker_id_int, camera_point):
        state = self.marker_states.get(
            marker_id_int,
            {"last_pos": camera_point.copy(), "stable_count": 0, "is_static": False}
        )

        dist = float(np.linalg.norm(camera_point - state["last_pos"]))

        if dist < self.move_threshold:
            state["stable_count"] += 1
            if state["stable_count"] >= self.stable_count_required:
                state["is_static"] = True
        else:
            state["stable_count"] = 0
            state["is_static"] = False
            self.current_marker_id = marker_id_int

        state["last_pos"] = camera_point.copy()
        self.marker_states[marker_id_int] = state
        return state

    # ======================================================================
    # Fond de projection
    # ======================================================================
    def build_graph_useful_background(self):
        """
        Fond logique dans le repère graph.
        """
        return np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)

    def _build_projector_background(self, proj_w, proj_h):
        """
        Construit le fond final dans le repère projecteur.
        Fond nominal uniquement.
        """
        graph_background = self.build_graph_useful_background()

        H_graph_to_proj = self.mapper.get_graph_to_projector_homography()

        warped_background = cv2.warpPerspective(
            graph_background,
            H_graph_to_proj,
            (proj_w, proj_h)
        )

        return warped_background

    # ======================================================================
    # Logique pose_aruco : sélection du point du tag
    # ======================================================================
    def _select_physical_corner_runtime(self, pts: np.ndarray, position: str) -> np.ndarray:
        """
        Sélectionne un coin physique du tag selon la position souhaitée.
        Même logique que dans pose_aruco.
        pts: array (4,2)
        """
        if position == "TL":
            return pts[np.argmin(pts[:, 0] + pts[:, 1])].astype(np.float32)
        elif position == "TR":
            return pts[np.argmin(-pts[:, 0] + pts[:, 1])].astype(np.float32)
        elif position == "BR":
            return pts[np.argmax(pts[:, 0] + pts[:, 1])].astype(np.float32)
        elif position == "BL":
            return pts[np.argmax(-pts[:, 0] + pts[:, 1])].astype(np.float32)
        else:
            raise ValueError(f"Position inconnue: {position}")

    def _get_marker_anchor_raw(self, corner):
        """
        Point de référence unique du tag pour le tracking/runtime.
        Ici on utilise le coin physique TL du tag.
        """
        pts = corner[0].astype(np.float32)
        return self._select_physical_corner_runtime(pts, "TL")
    
    def _get_marker_anchor_raw_center(self, corner):
        pts = corner[0].astype(np.float32)
        return np.mean(pts, axis=0).astype(np.float32)  
    # ======================================================================
    # Calcul des coordonnées de tags
    # ======================================================================
    def _compute_marker_projector_geometry(self, corner):
        """
        Version test minimal :
        on projette uniquement un point d'ancrage rouge, comme dans pose_aruco.
        """
        # anchor_raw = self._get_marker_anchor_raw(corner)
        anchor_raw = self._get_marker_anchor_raw_center(corner)

        graph_anchor = self.mapper.camera_raw_to_graph(anchor_raw)
        projector_anchor = self.mapper.camera_raw_to_projector_nominal(anchor_raw)

        # offset global manuel conservé
        projector_anchor = self._apply_projector_offset(projector_anchor)

        return {
            "graph_center": graph_anchor,
            "projector_center": projector_anchor,
            "marker_size": 40,
        }
    
    def update_cups_bottom(self, detected_markers):
        """
        detected_markers = dict {id: np.array([x, y])} en coordonnées graph
        """

        # =========================
        # 1. Mise à jour des tasses vues
        # =========================
        for marker_id, pos in detected_markers.items():

            if marker_id not in self.cups:
                self.cups[marker_id] = {
                    "state": "POSEE",
                    "last_pos": pos.copy(),
                    "lost_frames": 0,
                    "carrier_hand_id": None
                }

            cup = self.cups[marker_id]

            cup["last_pos"] = pos.copy()
            cup["lost_frames"] = 0
            cup["carrier_hand_id"] = None
            cup["state"] = "POSEE"

        # =========================
        # 2. Gestion des disparitions
        # =========================
        for marker_id, cup in self.cups.items():

            if marker_id not in detected_markers:
                cup["lost_frames"] += 1

                if cup["lost_frames"] < self.N_LIFT:
                    cup["state"] = "PEUT_ETRE_SOULEVEE"
                else:
                    cup["state"] = "SOULEVEE"

    # ======================================================================
    # Dessin overlays
    # ======================================================================
    def draw_from_config(self, img, marker_config, projector_x, projector_y, marker_size, marker_id=None):
        for element in marker_config:
            if element["type"] == "circle":
                self.draw_circle_from_config(img, element, projector_x, projector_y, marker_size, marker_id)
            elif element["type"] == "line":
                self.draw_line_from_config(img, element, projector_x, projector_y, marker_size, marker_id)
            elif element["type"] == "text":
                self.draw_text_from_config(img, element, projector_x, projector_y, marker_size)

    def draw_circle_from_config(self, img, element, projector_x, projector_y, marker_size, marker_id=None):
        abs_radius = int(element["relative_size"]["radius"] * marker_size)
        abs_x = int(projector_x + element["relative_position"]["x"] * marker_size)
        abs_y = int(projector_y - element["relative_position"]["y"] * marker_size)

        if self.mode_multidim and marker_id is not None:
            color = self.get_marker_color(marker_id)
        else:
            color = self.parse_color(element.get("color", "#FFFFFF"))

        fill = element.get("fill", False)
        thickness = -1 if fill else max(1, int(element["relative_size"]["thickness"] * marker_size))
        cv2.circle(img, (abs_x, abs_y), max(1, abs_radius), color, thickness)

    def draw_line_from_config(self, img, element, projector_x, projector_y, marker_size, marker_id=None):
        abs_x1 = int(projector_x + element["relative_position"]["x1"] * marker_size)
        abs_y1 = int(projector_y - element["relative_position"]["y1"] * marker_size)
        abs_x2 = int(projector_x + element["relative_position"]["x2"] * marker_size)
        abs_y2 = int(projector_y - element["relative_position"]["y2"] * marker_size)

        if self.mode_multidim and marker_id is not None:
            color = self.get_marker_color(marker_id)
        else:
            color = self.parse_color(element.get("color", "#FFFFFF"))

        thickness = max(1, int(element["thickness"] * marker_size))
        cv2.line(img, (abs_x1, abs_y1), (abs_x2, abs_y2), color, thickness)

    def draw_text_from_config(self, img, element, projector_x, projector_y, marker_size):
        abs_x = int(projector_x + element["relative_position"]["x"] * marker_size)
        abs_y = int(projector_y - element["relative_position"]["y"] * marker_size)
        font_size = max(1, int(element["font_size"] * marker_size))
        rotation = -element.get("rotation", 0)
        color = self.parse_color(element.get("color", "#FFFFFF"))
        text = element["text"]

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, 2)[0]
        text_width, text_height = text_size
        centered_x = abs_x - text_width // 2
        centered_y = abs_y + text_height // 2

        if rotation != 0:
            rotation_matrix = cv2.getRotationMatrix2D((abs_x, abs_y), rotation, 1.0)
            temp_image = np.zeros_like(img, dtype=np.uint8)

            cv2.putText(
                temp_image,
                text,
                (centered_x, centered_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size / 30,
                color,
                2
            )

            rotated_text = cv2.warpAffine(temp_image, rotation_matrix, (img.shape[1], img.shape[0]))
            mask = cv2.cvtColor(rotated_text, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(img, img, mask=mask_inv)
            img[:] = cv2.add(background, rotated_text)
        else:
            cv2.putText(
                img,
                text,
                (centered_x, centered_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size / 30,
                color,
                2
            )

    def draw_default_marker(self, img, projector_x, projector_y, marker_size, marker_id):
        radius = max(20, int(marker_size * 0.6))
        alpha = 0.5
        overlay = img.copy()
        color = self.get_marker_color(marker_id)

        cv2.circle(overlay, (projector_x, projector_y), radius, color, 8)
        cv2.putText(
            overlay,
            f"ID: {marker_id}",
            (projector_x - 50, projector_y + radius + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def parse_color(self, color_str):
        try:
            if color_str.startswith("#") and len(color_str) == 7:
                color_rgb = tuple(int(color_str.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                return (color_rgb[2], color_rgb[1], color_rgb[0])
            raise ValueError(f"Invalid color format: {color_str}")
        except Exception as e:
            print(f"Error processing color: {e}")
            return (255, 255, 255)

    # ======================================================================
    # UI debug caméra
    # ======================================================================
    def show_camera_window(self, frame, corners=None):
        preview = frame.copy()

        if corners is not None:
            for marker_corners in corners:
                pts = marker_corners[0].astype(int)
                for i in range(4):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % 4])
                    cv2.line(preview, pt1, pt2, (0, 255, 0), thickness=4)

                center = tuple(np.mean(pts, axis=0).astype(int))
                cv2.circle(preview, center, 10, (0, 0, 255), thickness=-1)

        resized_frame = cv2.resize(preview, (1080, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera", resized_frame)
        cv2.waitKey(1)

    # ======================================================================
    # Export
    # ======================================================================
    def save_to_buffer(self, graph_coords_ArUco):
        frame_data = {
            "frame": len(self.data_buffer) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        for marker_id, coords in graph_coords_ArUco:
            frame_data[f"ID_{marker_id}_x"] = float(coords[0])
            frame_data[f"ID_{marker_id}_y"] = float(coords[1])

            extra = self.extra_dimensions.get(marker_id, [0.0])
            for d, val in enumerate(extra):
                frame_data[f"ID_{marker_id}_dim{d + 1}"] = float(val)

        self.data_buffer.append(frame_data)

    def save_to_csv(self):
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.output_csv, index=False)
        print(f"Données sauvegardées dans {self.output_csv}")

    # ======================================================================
    # MultiDim
    # ======================================================================
    def select_next_marker(self, direction):
        if not self.marker_states:
            return

        ids = sorted(self.marker_states.keys())
        if self.current_marker_id not in ids:
            self.current_marker_id = ids[0]
        else:
            idx = ids.index(self.current_marker_id)
            idx = (idx + direction) % len(ids)
            self.current_marker_id = ids[idx]

        print(f"[MultiDim] Tag courant sélectionné : {self.current_marker_id}")

    def modify_current_marker_dimension(self, delta):
        if self.current_marker_id is None:
            return

        dims = self.extra_dimensions.get(self.current_marker_id, [0.0])
        dims[0] += delta
        dims[0] = max(-10, min(10, dims[0]))
        self.extra_dimensions[self.current_marker_id] = dims

        print(f"[MultiDim] Dim du tag {self.current_marker_id} = {dims[0]}")

    def custom_colormap_3(self, norm_value, col1=None, col2=None, col3=None):
        if col1 is None or col2 is None or col3 is None:
            depart = (80, 0, 0)
            milieu = (0, 0, 255)
            fin = (255, 235, 255)
        else:
            depart, milieu, fin = col1, col2, col3

        if norm_value < 0.5:
            t = norm_value / 0.5
            b = int(depart[0] * (1 - t) + milieu[0] * t)
            g = int(depart[1] * (1 - t) + milieu[1] * t)
            r = int(depart[2] * (1 - t) + milieu[2] * t)
        else:
            t = (norm_value - 0.5) / 0.5
            b = int(milieu[0] * (1 - t) + fin[0] * t)
            g = int(milieu[1] * (1 - t) + fin[1] * t)
            r = int(milieu[2] * (1 - t) + fin[2] * t)

        return (b, g, r)

    def get_marker_color(self, marker_id):
        if not self.mode_multidim:
            return (0, 0, 255)

        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        norm_value = np.clip((value + 10) / 20, 0, 1)
        return self.custom_colormap_3(
            norm_value,
            col1=(0, 0, 0),
            col2=(0, 0, 255),
            col3=(255, 235, 255)
        )

    def draw_dim_value(self, img, x, y, marker_id, color):
        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        cv2.putText(
            img,
            f"Dim: {value:.1f}",
            (int(x) - 50, int(y) - 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # ======================================================================
    # Boucle principale
    # ======================================================================
    def detect_and_process(self):
        from src.core.config.app_config import CALIBRATION_TAG_IDS

        self.running = True
        print("detect_and_process started")

        overlay_on = self.is_enabled("overlay_ra", False)
        projection_on = self.is_enabled("projection_media", False)
        adv_logs = self.is_enabled("advanced_logs", False)

        print("=== RUNTIME DEBUG ===")
        print("projection_on:", projection_on, "overlay_on:", overlay_on, "adv_logs:", adv_logs)
        print("timeline_steps:", len(self.timeline_steps))
        print("assets:", len(self.assets))
        if self.timeline_steps:
            s0 = self.timeline_steps[0]
            print(
                "step0:",
                s0.order_index,
                s0.label,
                "asset_ref=",
                s0.asset_ref,
                "pause=",
                s0.pause,
                "duration=",
                s0.duration_s,
            )
        print("=====================")

        if self.mapper is None:
            print("[ERREUR] Mapper ou homographie graph->projecteur indisponible. Arrêt.")
            self.finished_signal.emit()
            return

        ra_config = {}
        if overlay_on:
            ra_config = self.load_config(config_path("ra_config.json"))
        elif adv_logs:
            print("[MODULES] overlay_ra désactivé -> pas de RA")

        self._show_timeline_assets()
        print("Consignes affichées, on continue avec la détection des marqueurs.")

        while self.running:
            proj_w = int(self.mapper.projector_width)
            proj_h = int(self.mapper.projector_height)

            current_image_background = self._build_projector_background(proj_w, proj_h)

            frame = self.parent.camera_manager.get_frame()
            if frame is None:
                print("Erreur : Impossible de lire une frame caméra.")
                break

            if frame.size == 0:
                print("Erreur : Frame vide.")
                break

            corners, ids = self._detect_markers(frame)

            graph_coords_ArUco = []
            detected_markers = {}

            if ids is not None and len(ids) > 0:
                valid_indices = [
                    i for i in range(len(ids))
                    if int(ids[i][0]) not in CALIBRATION_TAG_IDS
                ]

                if len(valid_indices) > 0:
                    ids = ids[valid_indices]
                    corners = [corners[i] for i in valid_indices]
                else:
                    ids = None
                    corners = []

                for i, corner in enumerate(corners):
                    marker_id_int = int(ids[i][0])

                    # camera_point = self._get_marker_anchor_raw(corner)
                    camera_point = self._get_marker_anchor_raw_center(corner)
                    state = self._update_marker_state(marker_id_int, camera_point)

                    geom = self._compute_marker_projector_geometry(corner)

                    graph_center = geom["graph_center"]
                    projector_center = geom["projector_center"]
                    marker_size = geom["marker_size"]

                    graph_coords_ArUco.append([marker_id_int, graph_center])
                    detected_markers[marker_id_int] = graph_center

                    projector_x = int(round(projector_center[0]))
                    projector_y = int(round(projector_center[1]))
                    marker_id = str(marker_id_int)

                        
                    # point rouge au centre
                    cv2.circle(
                        current_image_background,
                        (projector_x, projector_y),
                        10,
                        (0, 0, 255),
                        -1
                    )

                    # anneau autour de la tasse
                    ring_radius = int(marker_size * 2.0)   # ajuste 1.2 / 1.5 / 2.0 si besoin
                    ring_thickness = max(3, int(marker_size * 0.12))

                    cv2.circle(
                        current_image_background,
                        (projector_x, projector_y),
                        ring_radius,
                        (0, 0, 255),
                        ring_thickness
                    )

                    cv2.putText(
                        current_image_background,
                        f"ID {marker_id_int}",
                        (projector_x + 12, projector_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    if state["is_static"] and overlay_on:
                        if f"marker_{marker_id}" in ra_config:
                            self.draw_from_config(
                                current_image_background,
                                ra_config[f"marker_{marker_id}"],
                                projector_x,
                                projector_y,
                                marker_size,
                                marker_id_int
                            )
                        else:
                            self.draw_default_marker(
                                current_image_background,
                                projector_x,
                                projector_y,
                                marker_size,
                                marker_id_int
                            )

                        if self.mode_multidim:
                            color = self.get_marker_color(marker_id_int)
                            self.draw_dim_value(
                                current_image_background,
                                projector_x,
                                projector_y,
                                marker_id_int,
                                color
                            )
                            
            self.update_cups_bottom(detected_markers)
            # Debug
            for marker_id, cup in self.cups.items():
                print(f"[CUP] ID={marker_id} | state={cup['state']} | lost={cup['lost_frames']}")

            if self.show_grid and self.record_window is not None:
                x_min, x_max, y_min, y_max, x_legend, y_legend = self.record_window.get_bounds_from_inputs()

                graph_grid = self.build_graph_useful_background()

                graph_grid = DrawUtils.draw_math_grid_on_image(
                    graph_grid,
                    x_min, x_max,
                    y_min, y_max,
                    x_legend, y_legend,
                    self.grid_size
                )

                H_graph_to_proj = self.mapper.get_graph_to_projector_homography()

                warped_grid = cv2.warpPerspective(
                    graph_grid,
                    H_graph_to_proj,
                    (proj_w, proj_h)
                )

                current_image_background = warped_grid

            if self.status_popUpCamera:
                self.show_camera_window(frame, corners=corners if ids is not None else None)
                self._camera_was_active = True
            else:
                if hasattr(self, "_camera_was_active") and self._camera_was_active:
                    cv2.destroyWindow("Camera")
                    self._camera_was_active = False

            self.display_manager.display_image_on_projector_monitor(current_image_background)
            self.data_signal.emit({"data": graph_coords_ArUco})
            self.save_to_buffer(graph_coords_ArUco)

        self.save_to_csv()
        print("detect_and_process terminé")
        self.finished_signal.emit()