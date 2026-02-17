# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtCore import QThread

from src.core.utils.paths import asset_path, config_path, data_path
from src.core.projection.display_manager import DisplayManager
from src.core.projection.draw_utils import DrawUtils


class Algorithm_Analysis(QObject):
    data_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_manager: DisplayManager,
        H_projector,
        H_inv_projector,
        H_graph,
        H_inv_graph,
        image_background,
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

        self.modules_enabled = modules_enabled or {}
        self.assets = assets or []
        self.timeline_steps = timeline_steps or []
        self.protocol = protocol

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # # Dossier data + nom fichier
        # os.makedirs(data_path(), exist_ok=True)
        # self.output_csv = data_path(f"{output_name}_{timestamp}.csv")

        # Dossier output : V2 si fourni, sinon V1 (data_path)
        if output_dir is None:
            os.makedirs(data_path(), exist_ok=True)
            output_dir = data_path()
        else:
            os.makedirs(output_dir, exist_ok=True)

        self.output_csv = os.path.join(output_dir, f"{output_name}_{timestamp}.csv")

        self.H_projector = H_projector
        self.H_inv_projector = H_inv_projector
        self.H_graph = H_graph
        self.H_inv_graph = H_inv_graph

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.data_buffer = []
        self.running = False

        self.image_background = image_background
        self.image_width = self.image_background.shape[1]
        self.image_height = self.image_background.shape[0]

        self.status_popUpCamera = False

        # --- Stabilité ---
        self.marker_states = {}  # {id: {"last_pos": np.array, "stable_count": int, "is_static": bool}}
        self.fade_speed = 0.08
        self.move_threshold = 10

        self.show_grid = False
        self.record_window = record_window  # peut être None

        # --- MultiDim ---
        self.extra_dimensions = {}
        self.current_marker_id = None
        self.mode_multidim = True
        self.colormap = cv2.COLORMAP_COOL

        # Consignes
        self.waiting_for_consigne_key = False

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
        self.image_background = new_image_background
        self.image_width = self.image_background.shape[1]
        self.image_height = self.image_background.shape[0]

    def set_show_grid(self, show: bool):
        self.show_grid = show

    def on_consigne_key_pressed(self):
        self.waiting_for_consigne_key = False

    def detect_and_process(self):
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
            print("step0:", s0.order_index, s0.label, "asset_ref=", s0.asset_ref, "pause=", s0.pause, "duration=", s0.duration_s)
        print("=====================")
        for k in ["overlay_ra", "projection_media"]:
            v = self.modules_enabled.get(k, None)
            print("DBG", k, v, type(v))



        # Config RA
        # ra_config = self.load_config(config_path("ra_config.json"))
        # Config RA (uniquement si module overlay activé)
        ra_config = {}
        if overlay_on:
            ra_config = self.load_config(config_path("ra_config.json"))
        elif adv_logs:
            print("[MODULES] overlay_ra désactivé -> pas de RA")


        # --- Consignes dynamiques depuis timeline ---
        if projection_on and self.timeline_steps:
            assets_by_id = {a.id: a for a in self.assets}

            for step in self.timeline_steps:
                if not self.running:
                    break

                # Pause
                if step.pause or step.asset_ref is None:
                    t0 = datetime.now()
                    while self.running and (datetime.now() - t0).total_seconds() < step.duration_s:
                        self.display_manager.display_image_on_projector_monitor(self.image_background)
                        _frame = self.parent.camera_manager.get_frame()
                        cv2.waitKey(1)
                    continue

                asset = assets_by_id.get(step.asset_ref)
                if not asset:
                    continue

                # Image
                if asset.asset_type == "image":
                    img = cv2.imread(asset.path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (self.image_width, self.image_height))

                    self.waiting_for_consigne_key = True
                    while True:
                        if not self.running:
                            break
                        if not self.waiting_for_consigne_key:
                            break

                        self.display_manager.display_image_on_projector_monitor(img)
                        _frame = self.parent.camera_manager.get_frame()
                        cv2.waitKey(1)

        elif projection_on and not self.timeline_steps:
            if adv_logs:
                print("[MODULES] projection_media=ON mais timeline_steps vide -> aucune consigne")
        else:
            if adv_logs:
                print("[MODULES] projection_media=OFF -> consignes ignorées")


        print("Consignes affichées, on continue avec la détection des marqueurs.")

        # --- Boucle principale ---
     
        while True:
            if not self.running:
                break
            current_image_background = self.image_background.copy()
            if not self.running:
                break

            frame = self.parent.camera_manager.get_frame()
            if frame is None:
                break

            if frame.size == 0:
                print("Erreur : Frame vide.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            projector_coords_ArUco = []
            graph_coords_ArUco = []

            if ids is not None:
                valid_indices = [i for i in range(len(ids)) if int(ids[i][0]) not in [40, 41, 42, 43]]
                ids = ids[valid_indices]
                corners = [corners[i] for i in valid_indices]

                for i, corner in enumerate(corners):
                    marker_id_int = int(ids[i][0])
                    center_x = int(np.mean(corner[0][:, 0]))
                    center_y = int(np.mean(corner[0][:, 1]))
                    camera_point = np.array([center_x, center_y])

                    # --- Stabilité ---
                    state = self.marker_states.get(
                        marker_id_int,
                        {"last_pos": camera_point, "stable_count": 0, "is_static": False}
                    )

                    dist = np.linalg.norm(camera_point - state["last_pos"])
                    if dist < self.move_threshold:
                        state["stable_count"] += 1
                        if state["stable_count"] >= 5:
                            state["is_static"] = True
                    else:
                        state["stable_count"] = 0
                        state["is_static"] = False
                        self.current_marker_id = marker_id_int

                    state["last_pos"] = camera_point
                    self.marker_states[marker_id_int] = state

                    # Projector coords
                    projector_point = self.transform_to_projector(camera_point)
                    projector_coords_ArUco.append([marker_id_int, projector_point])

                    projector_corners = [self.transform_to_projector(corner[0][j]) for j in range(4)]
                    marker_size = int(np.linalg.norm(projector_corners[0] - projector_corners[1])) * 2

                    # Graph coords
                    graph_point = self.transform_to_graph(camera_point)
                    graph_coords_ArUco.append([marker_id_int, graph_point])

                    projector_x = int(projector_point[0] - ((self.image_width - self.image_height) / 2))
                    projector_y = int(projector_point[1])
                    marker_id = str(ids[i][0])

                    # Afficher uniquement si statique
                    # if state["is_static"]:
                    #     if f"marker_{marker_id}" in ra_config:
                    #         self.draw_from_config(
                    #             current_image_background,
                    #             ra_config[f"marker_{marker_id}"],
                    #             projector_x, projector_y, marker_size, marker_id_int
                    #         )
                    #     else:
                    #         self.draw_default_marker(
                    #             current_image_background,
                    #             projector_x, projector_y, marker_size, ids[i][0]
                    #         )

                    
                    # si overlay_ra=False, aucun overlay ne sera dessiné
                    if state["is_static"] and overlay_on:
                        if f"marker_{marker_id}" in ra_config:
                            self.draw_from_config(
                                current_image_background,
                                ra_config[f"marker_{marker_id}"],
                                projector_x, projector_y, marker_size, marker_id_int
                            )
                        else:
                            self.draw_default_marker(
                                current_image_background,
                                projector_x, projector_y, marker_size, ids[i][0]
                            )


                        if self.mode_multidim:
                            color = self.get_marker_color(marker_id_int)
                            self.draw_dim_value(current_image_background, projector_x, projector_y, marker_id_int, color)

            # Grille
            if self.show_grid and self.record_window is not None:
                x_min, x_max, y_min, y_max, x_legend, y_legend = self.record_window.get_bounds_from_inputs()
                current_image_background[:, :, :] = DrawUtils.draw_math_grid_on_image(
                    current_image_background, x_min, x_max, y_min, y_max, x_legend, y_legend, self.image_width
                )

            # Cam window
            if self.status_popUpCamera:
                self.show_camera_window(frame, corners=corners)
                self._camera_was_active = True
            else:
                if hasattr(self, "_camera_was_active") and self._camera_was_active:
                    cv2.destroyWindow("Camera")
                    self._camera_was_active = False

            # Project
            if self.running:
                self.display_manager.display_image_on_projector_monitor(current_image_background)
                self.data_signal.emit({"data": graph_coords_ArUco})

            self.save_to_buffer(graph_coords_ArUco)

        self.save_to_csv()
        print("detect_and_process terminé")
        self.finished_signal.emit()


    def state_popUpCamera_changed(self):
        self.status_popUpCamera = not self.status_popUpCamera

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
        thickness = -1 if fill else int(element["relative_size"]["thickness"] * marker_size)
        cv2.circle(img, (abs_x, abs_y), abs_radius, color, thickness)

    def draw_line_from_config(self, img, element, projector_x, projector_y, marker_size, marker_id=None):
        abs_x1 = int(projector_x + element["relative_position"]["x1"] * marker_size)
        abs_y1 = int(projector_y - element["relative_position"]["y1"] * marker_size)
        abs_x2 = int(projector_x + element["relative_position"]["x2"] * marker_size)
        abs_y2 = int(projector_y - element["relative_position"]["y2"] * marker_size)

        if self.mode_multidim and marker_id is not None:
            color = self.get_marker_color(marker_id)
        else:
            color = self.parse_color(element.get("color", "#FFFFFF"))

        thickness = int(element["thickness"] * marker_size)
        cv2.line(img, (abs_x1, abs_y1), (abs_x2, abs_y2), color, thickness)

    def draw_text_from_config(self, img, element, projector_x, projector_y, marker_size):
        abs_x = int(projector_x + element["relative_position"]["x"] * marker_size)
        abs_y = int(projector_y - element["relative_position"]["y"] * marker_size)
        font_size = int(element["font_size"] * marker_size)
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
            cv2.putText(temp_image, text, (centered_x, centered_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2)
            rotated_text = cv2.warpAffine(temp_image, rotation_matrix, (img.shape[1], img.shape[0]))
            mask = cv2.cvtColor(rotated_text, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(img, img, mask=mask_inv)
            img[:] = cv2.add(background, rotated_text)
        else:
            cv2.putText(img, text, (centered_x, centered_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2)

    def draw_default_marker(self, img, projector_x, projector_y, marker_size, marker_id):
        r = 100
        alpha = 0.5
        overlay = img.copy()
        color = self.get_marker_color(marker_id)

        cv2.circle(overlay, (projector_x, projector_y), r, color, 20)
        cv2.putText(
            overlay, f"ID: {marker_id}",
            (projector_x - 50, projector_y + 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
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

    def show_camera_window(self, frame, corners=None):
        if corners is not None:
            for marker_corners in corners:
                pts = marker_corners[0].astype(int)
                for i in range(4):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), thickness=6)
                center = tuple(np.mean(pts, axis=0).astype(int))
                cv2.circle(frame, center, 12, (0, 0, 255), thickness=-1)

        resized_frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("Camera", resized_frame)
        cv2.waitKey(1)

    def transform_to_projector(self, camera_point):
        camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
        projected_point = np.dot(self.H_projector, camera_point)
        projected_point /= projected_point[2]
        x, y = projected_point[:2]
        x = (self.image_width - 1) - x
        return np.array([x, y])

    def transform_to_graph(self, camera_point):
        camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
        graph_point = np.dot(self.H_graph, camera_point)
        graph_point /= graph_point[2]
        graph_point[0] = 700 - graph_point[0]
        return graph_point[:2]

    def save_to_buffer(self, graph_coords_ArUco):
        frame_data = {
            "frame": len(self.data_buffer) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for marker_id, coords in graph_coords_ArUco:
            frame_data[f"ID_{marker_id}_x"] = coords[0]
            frame_data[f"ID_{marker_id}_y"] = coords[1]
            extra = self.extra_dimensions.get(marker_id, [0.0])
            for d, val in enumerate(extra):
                frame_data[f"ID_{marker_id}_dim{d + 1}"] = val
        self.data_buffer.append(frame_data)

    def save_to_csv(self):
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.output_csv, index=False)
        print(f"Données sauvegardées dans {self.output_csv}")

    def stop(self):
        self.running = False
        self.waiting_for_consigne_key = False


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
        return self.custom_colormap_3(norm_value, col1=(0, 0, 0), col2=(0, 0, 255), col3=(255, 235, 255))

    def draw_dim_value(self, img, x, y, marker_id, color):
        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        cv2.putText(
            img, f"Dim: {value:.1f}",
            (int(x) - 50, int(y) - 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
