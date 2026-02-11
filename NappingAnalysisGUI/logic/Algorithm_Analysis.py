import cv2
import numpy as np
import pandas as pd
from logic.DisplayManager import DisplayManager
from logic.DrawUtils import DrawUtils
from PyQt6.QtCore import QObject, pyqtSignal
from datetime import datetime
import json


class Algorithm_Analysis(QObject):
    data_signal = pyqtSignal(dict)

    def __init__(self, parent, display_manager: DisplayManager, H_projector, H_inv_projector, H_graph, H_inv_graph, image_background, output_name="data", record_window=None):
        """
        Initialise l'algorithme de l'analyse.
        :param display_manager: Instance de DisplayManager pour afficher les résultats.
        :param H: Matrice d'homographie.
        :param H_inv: Matrice inverse de l'homographie.
        :param output_name: Nom du fichier CSV pour sauvegarder les positions des marqueurs.
        """
        super().__init__()
        self.parent = parent
        self.display_manager = display_manager
        timestamp = datetime.now().strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
        self.output_csv = f"./data/{output_name}_{timestamp}.csv"
        self.H_projector = H_projector
        self.H_inv_projector = H_inv_projector
        self.H_graph = H_graph
        self.H_inv_graph = H_inv_graph
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.data_buffer = []
        self.running = False  # Flag pour arrêter la boucle
        self.image_background = image_background
        self.image_width = self.image_background.shape[1]
        self.image_height = self.image_background.shape[0]
        self.status_popUpCamera = False
        # --- Ajout pour gestion du fade ---
        self.marker_states = {}  # {id: {"last_pos": np.array, "moving": bool, "opacity": float}}
        self.fade_speed = 0.08   # Vitesse du fade in/out (0.0 à 1.0 par frame)
        self.move_threshold = 10 # Seuil de déplacement (pixels) pour considérer un marqueur en mouvement
        self.show_grid = False  # Afficher la grille ou non
        self.record_window = record_window  # <-- Ajoute ceci

        
        self.extra_dimensions = {}  # {marker_id: [val1, val2, ...]}
        self.current_marker_id = None  # Tag courant sélectionné
        self.mode_multidim = True     # Flag pour mode multidimensionnel
        self.colormap = cv2.COLORMAP_COOL  # Ou autre colormap OpenCV
        self.waiting_for_consigne_key = False

    def update_background_image(self, new_image_background):
        """
        Met à jour l'image de fond utilisée pour l'affichage.
        :param new_image_background: Nouvelle image de fond (numpy array).
        """
        self.image_background = new_image_background
        self.image_width = self.image_background.shape[1]
        self.image_height = self.image_background.shape[0]

    def set_show_grid(self, show: bool):
        self.show_grid = show

    def on_consigne_key_pressed(self):
        self.waiting_for_consigne_key = False

    def detect_and_process(self):
        """
        Boucle principale pour détecter les marqueurs ArUco, transformer leurs coordonnées,
        dessiner des cercles autour des marqueurs, et sauvegarder les positions dans un fichier CSV.
        """
        self.running = True
        print("detect_and_process started")

        # Charger la configuration JSON
        ra_config = self.load_config("./config/ra_config.json")

        # --- Affichage des consignes successives ---
        consigne_paths = [
            "./assets/textures/Consigne1.png",
            "./assets/textures/Consigne2.png"
        ]
        for consigne_path in consigne_paths:
            consigne_img = cv2.imread(consigne_path)
            if consigne_img is not None:
                consigne_img = cv2.resize(consigne_img, (self.image_width, self.image_height))
                self.waiting_for_consigne_key = True
                while self.running and self.waiting_for_consigne_key:
                    self.display_manager.display_image_on_projector_monitor(consigne_img)
                    # --- Affichage caméra pendant la consigne ---
                    frame = self.parent.camera_manager.get_frame()
                    cv2.waitKey(30)
            else:
                print(f"⚠️ {consigne_path} introuvable, on continue.")
        print("Consignes affichées, on continue avec la détection des marqueurs.")

        # --- Boucle principale de détection ---
        while self.running:
            current_image_background = self.image_background.copy()
            frame = self.parent.camera_manager.get_frame()
            if frame is None or frame.size == 0:
                print("Erreur : Frame invalide ou vide.")
                break

            # Convertir l'image en niveaux de gris pour la détection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Détection des marqueurs ArUco
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

                    # --- Gestion du mouvement avec stabilité ---
                    state = self.marker_states.get(marker_id_int, {"last_pos": camera_point, "stable_count": 0, "is_static": False})
                    dist = np.linalg.norm(camera_point - state["last_pos"])
                    if dist < self.move_threshold:
                        state["stable_count"] += 1
                        if state["stable_count"] >= 5:  # 5 frames stables nécessaires (modifiable)
                            state["is_static"] = True
                    else:
                        state["stable_count"] = 0
                        state["is_static"] = False
                        self.current_marker_id = marker_id_int
                    state["last_pos"] = camera_point
                    self.marker_states[marker_id_int] = state
                    # --- Fin gestion du mouvement/stabilité ---

                    # Transform the camera point to projector coordinates
                    projector_point = self.transform_to_projector(camera_point)
                    projector_coords_ArUco.append([marker_id_int, projector_point])

                    # Recalculate marker size using projector coordinates
                    projector_corners = [self.transform_to_projector(corner[0][j]) for j in range(4)]
                    marker_size = int(np.linalg.norm(projector_corners[0] - projector_corners[1]))*2

                    # Transform the camera point to graph coordinates
                    graph_point = self.transform_to_graph(camera_point)
                    graph_coords_ArUco.append([marker_id_int, graph_point])

                    projector_x, projector_y = int(projector_point[0]-((self.image_width-self.image_height)/2)), int(projector_point[1])
                    marker_id = str(ids[i][0])  # Pour la clé JSON

                    # Afficher les éléments SEULEMENT si le marqueur est statique depuis plusieurs frames
                    if state["is_static"]:
                        if f"marker_{marker_id}" in ra_config:
                            self.draw_from_config(
                                current_image_background, ra_config[f"marker_{marker_id}"],
                                projector_x, projector_y, marker_size, marker_id_int
                            )
                        else:
                            self.draw_default_marker(
                                current_image_background, projector_x, projector_y, marker_size, ids[i][0]
                            )
                        # Affiche la valeur de la dimension UNE SEULE FOIS par tag
                        if self.mode_multidim:
                            color = self.get_marker_color(marker_id_int)
                            self.draw_dim_value(current_image_background, projector_x, projector_y, marker_id_int, color)

            # --- Affichage de la grille si demandé ---
            if self.show_grid and self.record_window is not None:
                x_min, x_max, y_min, y_max, x_legend, y_legend = self.record_window.get_bounds_from_inputs()
                current_image_background[:,:,:] = DrawUtils.draw_math_grid_on_image(
                    current_image_background, x_min, x_max, y_min, y_max, x_legend, y_legend, self.image_width
                )

            if self.status_popUpCamera:
                self.show_camera_window(frame, corners=corners)
                self._camera_was_active = True
            else:
                if hasattr(self, "_camera_was_active") and self._camera_was_active:
                    cv2.destroyWindow("Camera")
                    self._camera_was_active = False

            if self.running:
                self.display_manager.display_image_on_projector_monitor(current_image_background)
                self.data_signal.emit({"data": graph_coords_ArUco})

            self.save_to_buffer(graph_coords_ArUco)

        self.save_to_csv()
        print("detect_and_process terminé")  # Vérifier si la méthode se termine

    def state_popUpCamera_changed(self):
        self.status_popUpCamera = not self.status_popUpCamera

    def load_config(self, path):
        try:
            with open(path, "r") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            print("⚠️ Fichier de configuration 'ra_config.json' introuvable. Utilisation des paramètres par défaut.")
            return {}

    def update_marker_state(self, marker_id_int, camera_point):
        state = self.marker_states.get(marker_id_int, {"last_pos": camera_point, "stable_count": 0, "is_static": False})
        dist = np.linalg.norm(camera_point - state["last_pos"])
        if dist < self.move_threshold:
            state["stable_count"] += 1
            if state["stable_count"] >= 5:
                state["is_static"] = True
        else:
            state["stable_count"] = 0
            state["is_static"] = False
        state["last_pos"] = camera_point
        self.marker_states[marker_id_int] = state
        return state

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
        # --- Couleur dynamique si mode_multidim ---
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
        # --- Couleur dynamique si mode_multidim ---
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
        # Affiche l'ID en bas
        cv2.putText(
            overlay, f"ID: {marker_id}", 
            (projector_x - 50, projector_y + 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def parse_color(self, color_str):
        try:
            if color_str.startswith("#") and len(color_str) == 7:
                color_rgb = tuple(int(color_str.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                return (color_rgb[2], color_rgb[1], color_rgb[0])
            else:
                raise ValueError(f"Invalid color format: {color_str}")
        except Exception as e:
            print(f"Error processing color: {e}")
            return (255, 255, 255)

    def show_camera_window(self, frame, corners=None):
        # Dessine les contours des marqueurs avec une couleur et une épaisseur personnalisées
        if corners is not None:
            for marker_corners in corners:
                pts = marker_corners[0].astype(int)
                for i in range(4):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), thickness=6)  # Vert, épaisseur 5
                # Optionnel : dessine un cercle au centre
                center = tuple(np.mean(pts, axis=0).astype(int))
                cv2.circle(frame, center, 12, (0, 0, 255), thickness=-1)  # Rouge, plein

        resized_frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("Camera", resized_frame)
        cv2.waitKey(1)

    def transform_to_projector(self, camera_point):
        """
        Transforme un point de la caméra dans le domaine projectif.
        :param camera_point: Coordonnées du point dans le domaine caméra.
        :return: Coordonnées du point dans le domaine projectif.
        """
        camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
        projected_point = np.dot(self.H_projector, camera_point)
        projected_point /= projected_point[2]  # Normalisation
        x, y = projected_point[:2]
        # Inverser l'axe X (horizontal)
        x = (self.image_width - 1) - x
        return np.array([x, y])

    def transform_to_graph(self, camera_point):
        """
        Transforme un point de la caméra dans le domaine graphique.
        :param camera_point: Coordonnées du point dans le domaine caméra.
        :return: Coordonnées du point dans le domaine graphique.
        """
        camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
        graph_point = np.dot(self.H_graph, camera_point)
        graph_point /= graph_point[2]  # Normalisation
        graph_point[0] =  700 - graph_point[0]  # Inverser l'axe X (horizontal)
        return graph_point[:2]

    def save_to_buffer(self, graph_coords_ArUco):
        """
        Sauvegarde les coordonnées des marqueurs dans un buffer.
        :param projector_coords_ArUco: Liste des coordonnées des marqueurs dans le domaine projectif.
        """
        frame_data = { "frame": len(self.data_buffer) + 1, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for marker_id, coords in graph_coords_ArUco:
            frame_data[f"ID_{marker_id}_x"] = coords[0]
            frame_data[f"ID_{marker_id}_y"] = coords[1]
            # Ajout dimension supplémentaire
            extra = self.extra_dimensions.get(marker_id, [0.0])
            for d, val in enumerate(extra):
                frame_data[f"ID_{marker_id}_dim{d+1}"] = val
        self.data_buffer.append(frame_data)

    def save_to_csv(self):
        """
        Sauvegarde les données du buffer dans un fichier CSV.
        """
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.output_csv, index=False)
        print(f"Données sauvegardées dans {self.output_csv}")
               
    def stop(self):
        self.running = False

    def select_next_marker(self, direction):
        """Change le tag courant (direction = +1 ou -1) parmi les tags visibles."""
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
        """Modifie la valeur de la dimension supplémentaire du tag courant, bornée pour la colormap."""
        if self.current_marker_id is None:
            return
        dims = self.extra_dimensions.get(self.current_marker_id, [0.0])
        dims[0] += delta
        # Bornage entre -10 et +10 (adapter si besoin)
        dims[0] = max(-10, min(10, dims[0]))
        self.extra_dimensions[self.current_marker_id] = dims
        print(f"[MultiDim] Dim du tag {self.current_marker_id} = {dims[0]}")

    def custom_colormap_3(self, norm_value, col1=None, col2=None, col3=None):
        """
        norm_value: float entre 0 et 1
        Retourne une couleur RGB (B, G, R) personnalisée.
        Dégradé : rouge sombre (presque noir) -> rouge vif -> rose clair (presque blanc).
        """
        # Définir les couleurs de départ, milieu et fin pour le dégradé
        if col1 == None or col2 == None or col3 == None:
            depart = (80, 0, 0)      # Rouge sombre (B, G, R)
            milieu = (0, 0, 255)     # Rouge vif (B, G, R)
            fin = (255, 235, 255)    # Rose clair (B, G, R)
        else:
            depart = col1
            milieu = col2
            fin = col3

        if norm_value < 0.5:
            # Interpolation de départ à milieu
            t = norm_value / 0.5
            b = int(depart[0] * (1 - t) + milieu[0] * t)
            g = int(depart[1] * (1 - t) + milieu[1] * t)
            r = int(depart[2] * (1 - t) + milieu[2] * t)
        else:
            # Interpolation de milieu à fin
            t = (norm_value - 0.5) / 0.5
            b = int(milieu[0] * (1 - t) + fin[0] * t)
            g = int(milieu[1] * (1 - t) + fin[1] * t)
            r = int(milieu[2] * (1 - t) + fin[2] * t)
        return (b, g, r)
    
    def custom_colormap_2(self, norm_value, col1=None, col2=None):
        """
        norm_value: float entre 0 et 1
        Retourne une couleur RGB (B, G, R) personnalisée.
        Dégradé : rouge sombre (presque noir) -> rouge vif -> rose clair (presque blanc).
        """

        # Définir les couleurs de départ, milieu et fin pour le dégradé
        if col1 == None or col2 == None:
            depart = (0, 0, 80)      # Rouge sombre (B, G, R
            fin = (255, 235, 255)    # Rose clair (B, G, R)
        else:
            depart = col1
            fin = col2

        # Interpolation de départ à fin
        b = int(depart[0] * (1 - norm_value) + fin[0] * norm_value)
        g = int(depart[1] * (1 - norm_value) + fin[1] * norm_value)
        r = int(depart[2] * (1 - norm_value) + fin[2] * norm_value)
        return (b, g, r)

    def get_marker_color(self, marker_id):
        """Retourne la couleur du marqueur selon la valeur de la dimension supplémentaire."""
        if not self.mode_multidim:
            return (0, 0, 255)  # Rouge par défaut
        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        norm_value = np.clip((value + 10) / 20, 0, 1)
        color = self.custom_colormap_3(norm_value, col1=(0,0,0), col2=(0, 0, 255), col3=(255, 235, 255))
        return color
        #color = cv2.applyColorMap(np.uint8([[norm_value * 255]]), self.colormap)[0, 0]
        #return tuple(int(c) for c in color)

    def draw_dim_value(self, img, x, y, marker_id, color):
        """Affiche la valeur de la dimension supplémentaire au-dessus du marqueur."""
        value = self.extra_dimensions.get(marker_id, [0.0])[0]
        cv2.putText(
            img, f"Dim: {value:.1f}",
            (int(x) - 50, int(y) - 150),  # Ajuste la position si besoin
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )