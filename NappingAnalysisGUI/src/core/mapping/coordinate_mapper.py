# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np

from src.core.utils.paths import config_path


class CoordinateMapper:
    """
    Mapper central des coordonnées.

    Source de vérité retenue :
    - camera_calibration.json : calibration intrinsèque caméra
    - calibration_data.json   : homographies runtime

    Repères :
    - camera raw         : pixels bruts de la caméra
    - camera undist      : pixels caméra après undistortion
    - graph              : repère logique de l'application
    - projector          : repère image final du projecteur

    Conventions chargées depuis calibration_data.json :
    - H_proj      : camera undist -> projector
    - H_inv_proj  : projector -> camera undist
    - H_graph     : camera undist -> graph
    - H_inv_graph : graph -> camera undist

    Homographies reconstruites :
    - H_graph_to_proj = H_proj @ H_inv_graph
    - H_proj_to_graph = H_graph @ H_inv_proj
    """

    def __init__(self):
        # Calibration caméra
        self.K_cam = None
        self.dist_cam = None

        # Homographies de base (issues de calibration_data.json)
        self.H_cam_undist_to_proj = None
        self.H_proj_to_cam_undist = None
        self.H_cam_undist_to_graph = None
        self.H_graph_to_cam_undist = None

        # Homographies dérivées
        self.H_graph_to_proj = None
        self.H_proj_to_graph = None

        # Métadonnées pratiques
        self.grid_size = 700
        self.projector_width = 3840
        self.projector_height = 2160

    # ======================================================================
    # Chargement principal
    # ======================================================================
    def load(self):
        self._load_camera_calibration()
        self._load_calibration_data()
        self._build_composed_homographies()
        self._validate_loaded_data()

    def _load_camera_calibration(self):
        path = config_path("camera_calibration.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.K_cam = np.array(data["camera_matrix"], dtype=np.float64)
        self.dist_cam = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    def _load_calibration_data(self):
        path = config_path("calibration_data.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["H_proj", "H_inv_proj", "H_graph", "H_inv_graph"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Clé absente dans calibration_data.json : {key}")

        self.H_cam_undist_to_proj = np.array(data["H_proj"], dtype=np.float64)
        self.H_proj_to_cam_undist = np.array(data["H_inv_proj"], dtype=np.float64)
        self.H_cam_undist_to_graph = np.array(data["H_graph"], dtype=np.float64)
        self.H_graph_to_cam_undist = np.array(data["H_inv_graph"], dtype=np.float64)

    def _build_composed_homographies(self):
        """
        Reconstruit les homographies composées à partir des homographies de base.
        """
        self.H_graph_to_proj = self.H_cam_undist_to_proj @ self.H_graph_to_cam_undist
        self.H_proj_to_graph = self.H_cam_undist_to_graph @ self.H_proj_to_cam_undist

        self.H_cam_undist_to_proj = self._normalize_homography(self.H_cam_undist_to_proj)
        self.H_proj_to_cam_undist = self._normalize_homography(self.H_proj_to_cam_undist)
        self.H_cam_undist_to_graph = self._normalize_homography(self.H_cam_undist_to_graph)
        self.H_graph_to_cam_undist = self._normalize_homography(self.H_graph_to_cam_undist)
        self.H_graph_to_proj = self._normalize_homography(self.H_graph_to_proj)
        self.H_proj_to_graph = self._normalize_homography(self.H_proj_to_graph)

    def _validate_loaded_data(self):
        required_arrays = {
            "K_cam": self.K_cam,
            "dist_cam": self.dist_cam,
            "H_cam_undist_to_proj": self.H_cam_undist_to_proj,
            "H_proj_to_cam_undist": self.H_proj_to_cam_undist,
            "H_cam_undist_to_graph": self.H_cam_undist_to_graph,
            "H_graph_to_cam_undist": self.H_graph_to_cam_undist,
            "H_graph_to_proj": self.H_graph_to_proj,
            "H_proj_to_graph": self.H_proj_to_graph,
        }

        for name, arr in required_arrays.items():
            if arr is None:
                raise ValueError(f"{name} non chargé.")
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} doit être un np.ndarray.")

        if self.K_cam.shape != (3, 3):
            raise ValueError(f"K_cam doit être une matrice 3x3, reçu {self.K_cam.shape}")

        for name in [
            "H_cam_undist_to_proj",
            "H_proj_to_cam_undist",
            "H_cam_undist_to_graph",
            "H_graph_to_cam_undist",
            "H_graph_to_proj",
            "H_proj_to_graph",
        ]:
            H = getattr(self, name)
            if H.shape != (3, 3):
                raise ValueError(f"{name} doit être une matrice 3x3, reçu {H.shape}")

    # ======================================================================
    # Outils internes
    # ======================================================================
    @staticmethod
    def _normalize_homography(H: np.ndarray) -> np.ndarray:
        H = np.array(H, dtype=np.float64)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    @staticmethod
    def _apply_homography(H: np.ndarray, point_2d) -> np.ndarray:
        pt = np.array([point_2d[0], point_2d[1], 1.0], dtype=np.float64)
        mapped = H @ pt

        if abs(mapped[2]) < 1e-12:
            raise ValueError("Transformation homographique invalide : coordonnée homogène nulle.")

        mapped /= mapped[2]
        return mapped[:2].astype(np.float32)

    # ======================================================================
    # Calibration caméra
    # ======================================================================
    def undistort_camera_point(self, camera_point) -> np.ndarray:
        """
        Convertit un point caméra brut en point caméra undistordu (pixels).
        """
        pt = np.array([[camera_point[:2]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.K_cam, self.dist_cam, P=self.K_cam)
        return und.reshape(2).astype(np.float32)

    def camera_raw_to_camera_undist(self, camera_point) -> np.ndarray:
        return self.undistort_camera_point(camera_point)

    # ======================================================================
    # Conversions point à point
    # ======================================================================
    def undistort_camera_point_scaled(self, camera_point, alpha=1.0):
        pt = np.array([[camera_point[:2]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.K_cam, self.dist_cam, P=self.K_cam).reshape(2).astype(np.float32)
        raw = np.array(camera_point[:2], dtype=np.float32)
        return raw + alpha * (und - raw)

    def camera_raw_to_projector(self, camera_point) -> np.ndarray:
        """
        camera raw -> camera undist -> projector
        """
        cam_undist = self.undistort_camera_point_scaled(camera_point, alpha=2.15)
        return self._apply_homography(self.H_cam_undist_to_proj, cam_undist)

    def camera_raw_to_graph(self, camera_point) -> np.ndarray:
        """
        camera raw -> camera undist -> graph
        """
        cam_undist = self.undistort_camera_point(camera_point)
        return self._apply_homography(self.H_cam_undist_to_graph, cam_undist)

    def graph_to_projector(self, graph_point) -> np.ndarray:
        """
        graph -> projector
        """
        return self._apply_homography(self.H_graph_to_proj, graph_point)

    def projector_to_graph(self, projector_point) -> np.ndarray:
        """
        projector -> graph
        """
        return self._apply_homography(self.H_proj_to_graph, projector_point)

    def graph_to_camera_undist(self, graph_point) -> np.ndarray:
        """
        graph -> camera undist
        """
        return self._apply_homography(self.H_graph_to_cam_undist, graph_point)

    def projector_to_camera_undist(self, projector_point) -> np.ndarray:
        """
        projector -> camera undist
        """
        return self._apply_homography(self.H_proj_to_cam_undist, projector_point)

    # ======================================================================
    # Accès explicite aux homographies
    # ======================================================================
    def get_camera_undist_to_projector_homography(self) -> np.ndarray:
        return self.H_cam_undist_to_proj.copy()

    def get_projector_to_camera_undist_homography(self) -> np.ndarray:
        return self.H_proj_to_cam_undist.copy()

    def get_camera_undist_to_graph_homography(self) -> np.ndarray:
        return self.H_cam_undist_to_graph.copy()

    def get_graph_to_camera_undist_homography(self) -> np.ndarray:
        return self.H_graph_to_cam_undist.copy()

    def get_graph_to_projector_homography(self) -> np.ndarray:
        return self.H_graph_to_proj.copy()

    def get_projector_to_graph_homography(self) -> np.ndarray:
        return self.H_proj_to_graph.copy()
# # -*- coding: utf-8 -*-
# import json
# import cv2
# import numpy as np

# from src.core.utils.paths import config_path


# class CoordinateMapper:
#     """
#     Mapper central des coordonnées.

#     Rôle unique :
#     - charger les calibrations et homographies
#     - convertir des points entre repères géométriques

#     Repères utilisés :
#     - caméra brute            : pixels image issus directement de la caméra
#     - caméra undistortée      : pixels après correction optique
#     - graph                   : repère logique de l'application
#     - projecteur              : repère image final de projection

#     IMPORTANT :
#     - aucune logique d'affichage écran ici
#     - aucune rotation / flip / correction "display" ici
#     """

#     def __init__(self):
#         # Calibration caméra
#         self.K_cam = None
#         self.dist_cam = None

#         # Homographies runtime
#         self.H_cam_undist_to_proj = None
#         self.H_proj_to_cam_undist = None
#         self.H_cam_undist_to_graph = None
#         self.H_graph_to_cam_undist = None
#         self.H_graph_to_proj = None
#         self.H_proj_to_graph = None

#         # Métadonnées
#         self.grid_size = None
#         self.projector_width = None
#         self.projector_height = None

#     # ======================================================================
#     # Chargement principal
#     # ======================================================================
#     def load(self):
#         """
#         Charge la calibration caméra et les homographies runtime.
#         """
#         self._load_camera_calibration()
#         self._load_runtime_mapping()
#         self._validate_loaded_data()

#     def _load_camera_calibration(self):
#         path = config_path("camera_calibration.json")

#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         self.K_cam = np.array(data["camera_matrix"], dtype=np.float64)
#         self.dist_cam = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

#     def _load_runtime_mapping(self):
#         path = config_path("runtime_mapping.json")

#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         metadata = data["metadata"]
#         homographies = data["homographies"]

#         self.grid_size = int(metadata["grid_size"])
#         self.projector_width = int(metadata["projector_width"])
#         self.projector_height = int(metadata["projector_height"])

#         self.H_cam_undist_to_proj = np.array(homographies["H_cam_undist_to_proj"], dtype=np.float64)
#         self.H_proj_to_cam_undist = np.array(homographies["H_proj_to_cam_undist"], dtype=np.float64)
#         self.H_cam_undist_to_graph = np.array(homographies["H_cam_undist_to_graph"], dtype=np.float64)
#         self.H_graph_to_cam_undist = np.array(homographies["H_graph_to_cam_undist"], dtype=np.float64)
#         self.H_graph_to_proj = np.array(homographies["H_graph_to_proj"], dtype=np.float64)
#         self.H_proj_to_graph = np.array(homographies["H_proj_to_graph"], dtype=np.float64)

#         # ------------------------------------------------------------------
#         # Correction de convention du repère graph
#         # ------------------------------------------------------------------
#         # On conserve ici la correction déjà présente dans le code
#         # car elle fait partie de la convention géométrique métier.
#         # Elle n'a rien à voir avec l'affichage écran.
#         F_graph_x = np.array([
#             [-1.0, 0.0, self.grid_size],
#             [0.0, 1.0, 0.0],
#             [0.0, 0.0, 1.0]
#         ], dtype=np.float64)

#         F_graph_x_inv = F_graph_x.copy()

#         self.H_cam_undist_to_graph = F_graph_x @ self.H_cam_undist_to_graph
#         self.H_graph_to_cam_undist = self.H_graph_to_cam_undist @ F_graph_x_inv

#         self.H_graph_to_proj = self.H_graph_to_proj @ F_graph_x_inv
#         self.H_proj_to_graph = F_graph_x @ self.H_proj_to_graph

#     def _validate_loaded_data(self):
#         """
#         Vérifications simples après chargement.
#         """
#         required_arrays = {
#             "K_cam": self.K_cam,
#             "dist_cam": self.dist_cam,
#             "H_cam_undist_to_proj": self.H_cam_undist_to_proj,
#             "H_proj_to_cam_undist": self.H_proj_to_cam_undist,
#             "H_cam_undist_to_graph": self.H_cam_undist_to_graph,
#             "H_graph_to_cam_undist": self.H_graph_to_cam_undist,
#             "H_graph_to_proj": self.H_graph_to_proj,
#             "H_proj_to_graph": self.H_proj_to_graph,
#         }

#         for name, arr in required_arrays.items():
#             if arr is None:
#                 raise ValueError(f"{name} non chargé.")
#             if not isinstance(arr, np.ndarray):
#                 raise TypeError(f"{name} doit être un np.ndarray.")

#         for name in [
#             "H_cam_undist_to_proj",
#             "H_proj_to_cam_undist",
#             "H_cam_undist_to_graph",
#             "H_graph_to_cam_undist",
#             "H_graph_to_proj",
#             "H_proj_to_graph",
#         ]:
#             H = getattr(self, name)
#             if H.shape != (3, 3):
#                 raise ValueError(f"{name} doit être une matrice 3x3, reçu {H.shape}.")

#         if self.K_cam.shape != (3, 3):
#             raise ValueError(f"K_cam doit être une matrice 3x3, reçu {self.K_cam.shape}.")

#         if self.projector_width is None or self.projector_height is None:
#             raise ValueError("Dimensions projecteur non chargées.")

#         if self.grid_size is None:
#             raise ValueError("grid_size non chargé.")

#     # ======================================================================
#     # Outils internes
#     # ======================================================================
#     @staticmethod
#     def _normalize_homography(H: np.ndarray) -> np.ndarray:
#         H = np.array(H, dtype=np.float64)
#         if abs(H[2, 2]) > 1e-12:
#             H = H / H[2, 2]
#         return H

#     @staticmethod
#     def _apply_homography(H: np.ndarray, point_2d) -> np.ndarray:
#         pt = np.array([point_2d[0], point_2d[1], 1.0], dtype=np.float64)
#         mapped = H @ pt

#         if abs(mapped[2]) < 1e-12:
#             raise ValueError("Transformation homographique invalide : coordonnée homogène nulle.")

#         mapped /= mapped[2]
#         return mapped[:2].astype(np.float32)

#     # ======================================================================
#     # Calibration caméra
#     # ======================================================================
#     def undistort_camera_point(self, camera_point) -> np.ndarray:
#         """
#         Convertit un point caméra brut en point caméra undistordu (pixels).
#         """
#         pt = np.array([[camera_point[:2]]], dtype=np.float32)
#         und = cv2.undistortPoints(pt, self.K_cam, self.dist_cam, P=self.K_cam)
#         return und.reshape(2).astype(np.float32)

#     def camera_raw_to_camera_undist(self, camera_point) -> np.ndarray:
#         """
#         Alias explicite.
#         """
#         return self.undistort_camera_point(camera_point)

#     # ======================================================================
#     # Conversions point à point
#     # ======================================================================
#     def camera_raw_to_projector(self, camera_point) -> np.ndarray:
#         """
#         caméra brute -> caméra undistortée -> projecteur
#         """
#         cam_undist = self.undistort_camera_point(camera_point)
#         return self._apply_homography(self.H_cam_undist_to_proj, cam_undist)

#     def camera_raw_to_graph(self, camera_point) -> np.ndarray:
#         """
#         caméra brute -> caméra undistortée -> graph
#         """
#         cam_undist = self.undistort_camera_point(camera_point)
#         return self._apply_homography(self.H_cam_undist_to_graph, cam_undist)

#     def graph_to_projector(self, graph_point) -> np.ndarray:
#         """
#         graph -> projecteur
#         """
#         return self._apply_homography(self.H_graph_to_proj, graph_point)

#     def projector_to_graph(self, projector_point) -> np.ndarray:
#         """
#         projecteur -> graph
#         """
#         return self._apply_homography(self.H_proj_to_graph, projector_point)

#     def projector_to_camera_undist(self, projector_point) -> np.ndarray:
#         """
#         projecteur -> caméra undistortée
#         """
#         return self._apply_homography(self.H_proj_to_cam_undist, projector_point)

#     def graph_to_camera_undist(self, graph_point) -> np.ndarray:
#         """
#         graph -> caméra undistortée
#         """
#         return self._apply_homography(self.H_graph_to_cam_undist, graph_point)

#     # ======================================================================
#     # Homographies utiles
#     # ======================================================================
#     def get_runtime_graph_to_projector_homography(self) -> np.ndarray:
#         """
#         Retourne la homographie graph -> projecteur issue de runtime_mapping.json
#         après application de la convention graph interne.
#         """
#         return self._normalize_homography(self.H_graph_to_proj)

#     def get_runtime_camera_undist_to_projector_homography(self) -> np.ndarray:
#         """
#         Retourne la homographie caméra undistortée -> projecteur.
#         """
#         return self._normalize_homography(self.H_cam_undist_to_proj)

#     def load_graph_to_projector_from_calibration_data(self) -> np.ndarray:
#         """
#         Recharge la homographie graph -> projecteur à partir de calibration_data.json
#         en suivant exactement la logique du script de test qui projetait correctement
#         la grille.

#         Convention du fichier :
#         - H_proj      : camera undistorted -> projector
#         - H_inv_graph : graph -> camera undistorted

#         Donc :
#         graph -> projector = H_proj @ H_inv_graph
#         """
#         path = config_path("calibration_data.json")

#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if "H_proj" not in data:
#             raise KeyError("Clé absente dans calibration_data.json : H_proj")
#         if "H_inv_graph" not in data:
#             raise KeyError("Clé absente dans calibration_data.json : H_inv_graph")

#         H_proj = np.array(data["H_proj"], dtype=np.float64)
#         H_inv_graph = np.array(data["H_inv_graph"], dtype=np.float64)

#         H_graph_to_proj = H_proj @ H_inv_graph
#         return self._normalize_homography(H_graph_to_proj)

#     def compare_runtime_vs_calibration_graph_to_projector(self):
#         """
#         Retourne les deux homographies graph -> projecteur pour diagnostic :
#         - celle issue de runtime_mapping.json
#         - celle reconstruite depuis calibration_data.json
#         """
#         H_runtime = self.get_runtime_graph_to_projector_homography()
#         H_calib = self.load_graph_to_projector_from_calibration_data()
#         return H_runtime, H_calib