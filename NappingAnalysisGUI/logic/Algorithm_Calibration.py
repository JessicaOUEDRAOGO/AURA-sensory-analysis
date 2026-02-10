import cv2
import numpy as np
from logic.DisplayManager import DisplayManager
from logic.Utils import QuadrilateralDetector, HomographyTransformer, ProjectorPoint
from logic.DrawUtils import DrawUtils
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

import json  # Importer le module JSON
import os    # Pour vérifier l'existence des fichiers

class Calibration:
    def __init__(self, parent, cam_width, cam_height, grid_size, image_background):
        self.parent = parent  # Référence à MainApp
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.homography_transformer = HomographyTransformer()
        self.verif_status = False
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.update_frame)  # Connecte le timer à la mise à jour d'image
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
        self.timer.start(33)  # Rafraîchir environ 30 FPS (1000ms / 30 ≈ 33ms)
        return False

    def update_frame(self, last_frame = None):

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
        print("Sauvegarde des matrices d'homographie...")
        """Sauvegarde les matrices d'homographie dans un fichier JSON."""
        if any(matrix is None for matrix in [self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph]):
            print("Erreur : Les matrices d'homographie ne sont pas définies.")
            return

        calib_data = {
            "H_proj": self.H_proj.tolist(),
            "H_inv_proj": self.H_inv_proj.tolist(),
            "H_graph": self.H_graph.tolist(),
            "H_inv_graph": self.H_inv_graph.tolist()
        }

        try:
            with open("./config/calibration_data.json", "w") as json_file:
                json.dump(calib_data, json_file, indent=4)
            print("Les matrices d'homographie ont été sauvegardées avec succès.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des matrices : {e}")

    def load_calib(self):
        """Charge les matrices d'homographie depuis un fichier JSON."""
        if not os.path.exists("./config/calibration_data.json"):
            print("Erreur : Le fichier de calibration n'existe pas.")
            return

        try:
            with open("./config/calibration_data.json", "r") as json_file:
                calib_data = json.load(json_file)

            self.H_proj = np.array(calib_data["H_proj"])
            self.H_inv_proj = np.array(calib_data["H_inv_proj"])
            self.H_graph = np.array(calib_data["H_graph"])
            self.H_inv_graph = np.array(calib_data["H_inv_graph"])

            print("Les matrices d'homographie ont été chargées avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement des matrices : {e}")
        
        return self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph

    def start_calib(self,label_status):
        quadrilateral_detector = QuadrilateralDetector()
        homography = HomographyTransformer()
        Projec_point = ProjectorPoint()
        drawer = DrawUtils()
        if self.frame is None:
            # force une capture au moins une fois
            self.update_frame()
            if self.frame is None:
                raise Exception("Aucune frame caméra disponible pour calibrer.")

        proj_points = quadrilateral_detector.detect_quadrilateral_from_aruco(self.frame)
        if proj_points is None or len(proj_points) != 4:
            raise Exception("Impossible de détecter les 4 marqueurs ArUco de calibration.")

        image = drawer.draw_points_linked(proj_points, self.frame, "Quadrilatère détecté")
        self.proj_points = proj_points
        self.update_frame(last_frame = image)
        cam_points = Projec_point.setUp_Projector_Point(proj_points, self.image_height, self.image_height)
        graph_points = Projec_point.setUp_Projector_Point(proj_points, self.grid_size, self.grid_size)
        self.H_proj, self.H_inv_proj = homography.find_Invers_Homography(proj_points, cam_points)
        self.H_graph, self.H_inv_graph = homography.find_Invers_Homography(proj_points, graph_points)
        
        label_status.setPixmap(QPixmap("./assets/icons/Validate.png"))

        self.save_calib()

        return self.H_proj, self.H_inv_proj, self.H_graph, self.H_inv_graph

    def get_homography(self):
        if self.homography_transformer:
            return self.homography_transformer.get_matrices()
        return None, None

    def __del__(self):
        #print("⚠️ Calibration supprimée !")
        pass