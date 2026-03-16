# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

from src.core.utils.paths import config_path, asset_path
from src.core.utils.utils import QuadrilateralDetector, HomographyTransformer, ProjectorPoint
from src.core.projection.draw_utils import DrawUtils


class Calibration:
    def __init__(self, parent, cam_width, cam_height, grid_size, image_background):
        self.parent = parent
        self.cam_width = cam_width
        self.cam_height = cam_height

        self.homography_transformer = HomographyTransformer()
        self.verif_status = False

        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.update_frame)

        self.frame = None
        self.grid_size = grid_size

        self.H_proj = None
        self.H_inv_proj = None
        self.H_graph = None
        self.H_inv_graph = None

        self.image_background = image_background
        self.image_width = image_background.shape[1]
        self.image_height = image_background.shape[0]

    def run(self):
        self.timer.start(33)  # ~30 FPS
        return False

    def update_frame(self, last_frame=None):
        if last_frame is not None:
            self.parent.display_manager.show_frame(last_frame)
            return last_frame

        frame = self.parent.camera_manager.get_frame()
        if frame is None:
            print("Erreur : Impossible de capturer une image de la caméra.")
            self.timer.stop()
            return None

        # garde l’original pour la calibration
        self.frame = frame

        # preview légère pour Qt
        frame_small = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        self.parent.display_manager.show_frame(frame_small)
        return frame

    def save_calib(self):
        """Sauvegarde les matrices d'homographie dans un fichier JSON."""
        print("Sauvegarde des matrices d'homographie...")

        if any(m is None for m in [self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph]):
            print("Erreur : Les matrices d'homographie ne sont pas définies.")
            return

        calib_data = {
            "H_proj": self.H_proj.tolist(),
            "H_inv_proj": self.H_inv_proj.tolist(),
            "H_graph": self.H_graph.tolist(),
            "H_inv_graph": self.H_inv_graph.tolist()
        }

        calib_file = config_path("calibration_data.json")

        # créer dossier config si besoin
        os.makedirs(os.path.dirname(calib_file), exist_ok=True)

        try:
            with open(calib_file, "w", encoding="utf-8") as json_file:
                json.dump(calib_data, json_file, indent=4)
            print(f"Matrices sauvegardées : {calib_file}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des matrices : {e}")

    def load_calib(self):
        """Charge les matrices d'homographie depuis un fichier JSON."""
        calib_file = config_path("calibration_data.json")

        if not os.path.exists(calib_file):
            print(f"Erreur : Le fichier de calibration n'existe pas : {calib_file}")
            return None

        try:
            with open(calib_file, "r", encoding="utf-8") as json_file:
                calib_data = json.load(json_file)

            self.H_proj = np.array(calib_data["H_proj"])
            self.H_inv_proj = np.array(calib_data["H_inv_proj"])
            self.H_graph = np.array(calib_data["H_graph"])
            self.H_inv_graph = np.array(calib_data["H_inv_graph"])

            print("Les matrices d'homographie ont été chargées avec succès.")
            return self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph

        except Exception as e:
            print(f"Erreur lors du chargement des matrices : {e}")
            return None

    def start_calib(self, label_status):
        import os
        import cv2
        from PyQt6.QtGui import QPixmap
        from src.core.utils.paths import asset_path

        if self.frame is None:
            self.update_frame()
            if self.frame is None:
                raise Exception("Aucune frame caméra disponible pour calibrer.")

        loaded = self.load_calib()
        if loaded is None:
            raise Exception("Impossible de charger calibration_data.json")

        self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph = loaded

        preview = self.frame.copy()

        # petit texte de confirmation sur l'image
        cv2.putText(
            preview,
            "Calibration chargee depuis calibration_data.json",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        self.update_frame(last_frame=preview)

        validate_icon = asset_path("icons", "Validate.png")
        if os.path.exists(validate_icon):
            label_status.setPixmap(QPixmap(validate_icon))

        return self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph
    # def start_calib(self, label_status):
    #     quadrilateral_detector = QuadrilateralDetector()
    #     homography = HomographyTransformer()
    #     proj_point = ProjectorPoint()
    #     drawer = DrawUtils()

    #     if self.frame is None:
    #         # force une capture au moins une fois
    #         self.update_frame()
    #         if self.frame is None:
    #             raise Exception("Aucune frame caméra disponible pour calibrer.")

    #     proj_points = quadrilateral_detector.detect_quadrilateral_from_aruco(self.frame)
    #     if proj_points is None or len(proj_points) != 4:
    #         raise Exception("Impossible de détecter les 4 marqueurs ArUco de calibration.")

    #     image = drawer.draw_points_linked(proj_points, self.frame, "Quadrilatère détecté")
    #     self.proj_points = proj_points
    #     self.update_frame(last_frame=image)

    #     # Domain projecteur (carré basé sur image_height)
    #     cam_points = proj_point.setUp_Projector_Point(proj_points, self.image_height, self.image_height)
    #     S = self.image_height
    #     cam_points = np.array(cam_points, dtype=np.float32)

    #     # Rotation 180° : (x', y') = (S-1-x, S-1-y)
    #     cam_points = np.stack([(S - 1) - cam_points[:, 0], (S - 1) - cam_points[:, 1]], axis=1).astype(np.float32)

    #     # Domain graph (grid)
    #     graph_points = proj_point.setUp_Projector_Point(proj_points, self.grid_size, self.grid_size)
    #     Sg = self.grid_size
    #     graph_points = np.array(graph_points, dtype=np.float32)

    #     # Rotation 180°
    #     graph_points = np.stack([(Sg - 1) - graph_points[:, 0], (Sg - 1) - graph_points[:, 1]], axis=1).astype(np.float32)

    #     self.H_proj, self.H_inv_proj = homography.find_Invers_Homography(proj_points, cam_points)
    #     self.H_graph, self.H_inv_graph = homography.find_Invers_Homography(proj_points, graph_points)

    #     # Icon validate (chemin robuste)
    #     validate_icon = asset_path("icons", "Validate.png")
    #     if os.path.exists(validate_icon):
    #         label_status.setPixmap(QPixmap(validate_icon))
    #     else:
    #         print(f"[WARNING] Icône Validate introuvable : {validate_icon}")

    #     self.save_calib()
    #     return self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph
    def build_v2_reference(self, label_status):
        import os
        import cv2
        from PyQt6.QtGui import QPixmap

        from src.core.calibration.v2_recalibration_core import V2RecalibrationCore
        from src.core.utils.paths import asset_path

        if self.frame is None:
            self.update_frame()
            if self.frame is None:
                raise Exception("Aucune frame caméra disponible pour créer la référence V2.")

        core = V2RecalibrationCore(grid_size=self.grid_size)
        ref_data, ref_path = core.save_reference_bundle(self.frame)

        preview = self.frame.copy()

        for pt in ref_data["reference_camera_points_raw"].values():
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(preview, (x, y), 12, (255, 0, 0), -1)

        self.update_frame(last_frame=preview)

        validate_icon = asset_path("icons", "Validate.png")
        if os.path.exists(validate_icon):
            label_status.setPixmap(QPixmap(validate_icon))

        print(f"Référence V2 sauvegardée : {ref_path}")
        return ref_path

    def get_homography(self):
        if self.homography_transformer:
            return self.homography_transformer.get_matrices()
        return None, None

    def __del__(self):
        pass
