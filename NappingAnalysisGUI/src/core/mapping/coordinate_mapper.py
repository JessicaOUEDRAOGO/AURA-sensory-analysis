# -*- coding: utf-8 -*-
import json
from pathlib import Path

import cv2
import numpy as np

from src.core.utils.paths import config_path


class CoordinateMapper:
    """
    Mapper central des coordonnées.

    Sources :
    - camera_calibration.json
    - calibration_data.json
    - projector_correction.json (optionnel)

    Repères :
    - camera raw
    - camera undist
    - graph
    - projector nominal
    - projector corrected

    Important :
    - la calibration principale reste portée par les homographies nominales
    - la correction projecteur finale doit rester une correction RESIDUELLE
    - pour éviter les déformations du fond, on impose ici une correction
      affine (pas de perspective résiduelle)
    """

    def __init__(self):
        # Calibration caméra
        self.K_cam = None
        self.dist_cam = None

        # Homographies nominales de base
        self.H_cam_undist_to_proj = None
        self.H_proj_to_cam_undist = None
        self.H_cam_undist_to_graph = None
        self.H_graph_to_cam_undist = None

        # Homographies nominales dérivées
        self.H_graph_to_proj = None
        self.H_proj_to_graph = None

        # Correction finale projecteur
        # On la stocke sous forme 3x3, mais on impose une forme affine :
        # [a b tx]
        # [c d ty]
        # [0 0  1]
        self.H_proj_correction = np.eye(3, dtype=np.float64)

        # Métadonnées
        self.grid_size = 700
        self.projector_width = 3840
        self.projector_height = 2160

    # ======================================================================
    # Chargement
    # ======================================================================
    def load(self, load_projector_correction: bool = True):
        self._load_camera_calibration()
        self._load_calibration_data()
        self._build_composed_homographies()

        if load_projector_correction:
            self._load_optional_projector_correction()
        else:
            self.H_proj_correction = np.eye(3, dtype=np.float64)
            print("[MAPPER] Chargement sans correction projecteur existante.")

        self._validate_loaded_data()

    def _load_camera_calibration(self):
        path = config_path("camera_calibration.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "camera_matrix" not in data:
            raise KeyError("camera_calibration.json : clé 'camera_matrix' absente")
        if "dist_coeffs" not in data:
            raise KeyError("camera_calibration.json : clé 'dist_coeffs' absente")

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

        self.H_cam_undist_to_proj = self._normalize_homography(
            np.array(data["H_proj"], dtype=np.float64)
        )
        self.H_proj_to_cam_undist = self._normalize_homography(
            np.array(data["H_inv_proj"], dtype=np.float64)
        )
        self.H_cam_undist_to_graph = self._normalize_homography(
            np.array(data["H_graph"], dtype=np.float64)
        )
        self.H_graph_to_cam_undist = self._normalize_homography(
            np.array(data["H_inv_graph"], dtype=np.float64)
        )

        if "grid_size" in data:
            self.grid_size = int(data["grid_size"])
        if "projector_width" in data:
            self.projector_width = int(data["projector_width"])
        if "projector_height" in data:
            self.projector_height = int(data["projector_height"])

    def _build_composed_homographies(self):
        if self.H_cam_undist_to_proj is None or self.H_graph_to_cam_undist is None:
            raise ValueError("Impossible de construire H_graph_to_proj")
        if self.H_cam_undist_to_graph is None or self.H_proj_to_cam_undist is None:
            raise ValueError("Impossible de construire H_proj_to_graph")

        self.H_graph_to_proj = self._normalize_homography(
            self.H_cam_undist_to_proj @ self.H_graph_to_cam_undist
        )
        self.H_proj_to_graph = self._normalize_homography(
            self.H_cam_undist_to_graph @ self.H_proj_to_cam_undist
        )

    def _load_optional_projector_correction(self):
        path = Path(config_path("projector_correction.json"))

        if not path.exists():
            self.H_proj_correction = np.eye(3, dtype=np.float64)
            print("[MAPPER] projector_correction.json absent -> identité")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "H_proj_correction" not in data:
            raise KeyError("projector_correction.json : clé 'H_proj_correction' absente")

        H = np.array(data["H_proj_correction"], dtype=np.float64)
        H = self._normalize_homography(H)

        # On refuse ici les corrections projectives fortes,
        # car elles déforment le fond.
        if not self._is_affine_like(H):
            raise ValueError(
                "projector_correction.json contient une correction projective. "
                "Utilise une correction affine résiduelle à la place."
            )

        self.H_proj_correction = self._force_affine_homography(H)

        print("[MAPPER] H_proj_correction chargée :")
        print(self.H_proj_correction)

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
            "H_proj_correction": self.H_proj_correction,
        }

        for name, arr in required_arrays.items():
            if arr is None:
                raise ValueError(f"{name} non chargé")
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} doit être un np.ndarray")

        if self.K_cam.shape != (3, 3):
            raise ValueError(f"K_cam doit être 3x3, reçu {self.K_cam.shape}")

        for name in [
            "H_cam_undist_to_proj",
            "H_proj_to_cam_undist",
            "H_cam_undist_to_graph",
            "H_graph_to_cam_undist",
            "H_graph_to_proj",
            "H_proj_to_graph",
            "H_proj_correction",
        ]:
            H = getattr(self, name)
            if H.shape != (3, 3):
                raise ValueError(f"{name} doit être 3x3, reçu {H.shape}")

    # ======================================================================
    # Outils internes
    # ======================================================================
    @staticmethod
    def _normalize_homography(H: np.ndarray) -> np.ndarray:
        H = np.array(H, dtype=np.float64)

        if H.shape != (3, 3):
            raise ValueError(f"Homographie invalide : shape={H.shape}, attendu (3,3)")

        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        return H

    @staticmethod
    def _apply_homography(H: np.ndarray, point_2d) -> np.ndarray:
        pt = np.array([point_2d[0], point_2d[1], 1.0], dtype=np.float64)
        mapped = H @ pt

        if abs(mapped[2]) < 1e-12:
            raise ValueError("Transformation homographique invalide : coordonnée homogène nulle")

        mapped /= mapped[2]
        return mapped[:2].astype(np.float32)

    @staticmethod
    def _is_affine_like(H: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Vérifie que la matrice est de forme affine :
        dernière ligne ~ [0, 0, 1]
        """
        return (
            abs(H[2, 0]) <= tol and
            abs(H[2, 1]) <= tol and
            abs(H[2, 2] - 1.0) <= tol
        )

    @staticmethod
    def _force_affine_homography(H: np.ndarray) -> np.ndarray:
        """
        Force explicitement la structure affine.
        """
        H_aff = np.array(H, dtype=np.float64)
        H_aff[2, 0] = 0.0
        H_aff[2, 1] = 0.0
        H_aff[2, 2] = 1.0
        return H_aff

    @staticmethod
    def _affine_2x3_to_homography(A: np.ndarray) -> np.ndarray:
        if A.shape != (2, 3):
            raise ValueError(f"Affine attendue en 2x3, reçu {A.shape}")

        H = np.eye(3, dtype=np.float64)
        H[0:2, :] = A
        return H

    def _get_inverse_projector_correction_homography_internal(self) -> np.ndarray:
        H_inv = np.linalg.inv(self.H_proj_correction)
        return self._normalize_homography(H_inv)

    # ======================================================================
    # Calibration caméra
    # ======================================================================
    def undistort_camera_point(self, camera_point) -> np.ndarray:
        pt = np.array([[camera_point[:2]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.K_cam, self.dist_cam, P=self.K_cam)
        return und.reshape(2).astype(np.float32)

    def camera_raw_to_camera_undist(self, camera_point) -> np.ndarray:
        return self.undistort_camera_point(camera_point)

    # ======================================================================
    # Correction projecteur
    # ======================================================================
    def set_projector_correction_homography(self, H: np.ndarray):
        H = np.array(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"H_proj_correction doit être 3x3, reçu {H.shape}")

        H = self._normalize_homography(H)

        if not self._is_affine_like(H):
            raise ValueError(
                "La correction projecteur doit être affine "
                "(dernière ligne [0, 0, 1])."
            )

        self.H_proj_correction = self._force_affine_homography(H)

    def reset_projector_correction_homography(self):
        self.H_proj_correction = np.eye(3, dtype=np.float64)

    def get_projector_correction_homography(self) -> np.ndarray:
        return self.H_proj_correction.copy()

    def get_inverse_projector_correction_homography(self) -> np.ndarray:
        return self._get_inverse_projector_correction_homography_internal().copy()

    def fit_projector_correction_from_points(self, src_pts, dst_pts, partial=False):
        """
        Calcule une correction AFFINE résiduelle.

        src_pts : points projecteur actuellement obtenus
        dst_pts : points projecteur souhaités

        partial=False -> affine complète (scale anisotrope + shear possibles)
        partial=True  -> similitude (rotation + translation + scale uniforme)

        Retourne une homographie 3x3 affine.
        """
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        if src_pts.shape != dst_pts.shape:
            raise ValueError("src_pts et dst_pts doivent avoir la même forme")

        if src_pts.ndim != 2 or src_pts.shape[1] != 2:
            raise ValueError("src_pts et dst_pts doivent être de forme (N, 2)")

        if src_pts.shape[0] < 3:
            raise ValueError("Il faut au moins 3 points pour calculer une affine")

        if partial:
            A, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=5000,
                confidence=0.99,
                refineIters=10
            )
        else:
            A, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=5000,
                confidence=0.99,
                refineIters=10
            )

        if A is None:
            raise ValueError("Impossible de calculer une correction affine projecteur")

        H_corr = self._affine_2x3_to_homography(A)
        self.set_projector_correction_homography(H_corr)

        return self.get_projector_correction_homography(), inliers

    def save_projector_correction(self, path=None):
        if path is None:
            path = Path(config_path("projector_correction.json"))
        else:
            path = Path(path)

        data = {
            "H_proj_correction": self.get_projector_correction_homography().tolist()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ======================================================================
    # Conversions point à point
    # ======================================================================
    def camera_raw_to_projector(self, camera_point) -> np.ndarray:
        cam_undist = self.undistort_camera_point(camera_point)
        p_nominal = self._apply_homography(self.H_cam_undist_to_proj, cam_undist)
        p_corrected = self._apply_homography(self.H_proj_correction, p_nominal)
        return p_corrected

    def camera_raw_to_projector_nominal(self, camera_point) -> np.ndarray:
        cam_undist = self.undistort_camera_point(camera_point)
        return self._apply_homography(self.H_cam_undist_to_proj, cam_undist)

    def camera_raw_to_graph(self, camera_point) -> np.ndarray:
        cam_undist = self.undistort_camera_point(camera_point)
        return self._apply_homography(self.H_cam_undist_to_graph, cam_undist)

    def graph_to_projector(self, graph_point) -> np.ndarray:
        p_nominal = self._apply_homography(self.H_graph_to_proj, graph_point)
        p_corrected = self._apply_homography(self.H_proj_correction, p_nominal)
        return p_corrected

    def graph_to_projector_nominal(self, graph_point) -> np.ndarray:
        return self._apply_homography(self.H_graph_to_proj, graph_point)

    def projector_to_graph(self, projector_point) -> np.ndarray:
        H_corr_inv = self._get_inverse_projector_correction_homography_internal()
        p_nominal = self._apply_homography(H_corr_inv, projector_point)
        return self._apply_homography(self.H_proj_to_graph, p_nominal)

    def graph_to_camera_undist(self, graph_point) -> np.ndarray:
        return self._apply_homography(self.H_graph_to_cam_undist, graph_point)

    def projector_to_camera_undist(self, projector_point) -> np.ndarray:
        H_corr_inv = self._get_inverse_projector_correction_homography_internal()
        p_nominal = self._apply_homography(H_corr_inv, projector_point)
        return self._apply_homography(self.H_proj_to_cam_undist, p_nominal)

    def projector_nominal_to_camera_undist(self, projector_point) -> np.ndarray:
        return self._apply_homography(self.H_proj_to_cam_undist, projector_point)

    def camera_undist_to_projector(self, camera_undist_point) -> np.ndarray:
        p_nominal = self._apply_homography(self.H_cam_undist_to_proj, camera_undist_point)
        p_corrected = self._apply_homography(self.H_proj_correction, p_nominal)
        return p_corrected

    def camera_undist_to_projector_nominal(self, camera_undist_point) -> np.ndarray:
        return self._apply_homography(self.H_cam_undist_to_proj, camera_undist_point)

    def projector_to_camera_raw_approx(self, projector_point) -> np.ndarray:
        return self.projector_to_camera_undist(projector_point)

    # ======================================================================
    # Accès explicite aux homographies
    # ======================================================================
    def get_camera_undist_to_projector_homography(self) -> np.ndarray:
        return self.H_cam_undist_to_proj.copy()

    def get_camera_undist_to_projector_corrected_homography(self) -> np.ndarray:
        H = self.H_proj_correction @ self.H_cam_undist_to_proj
        return self._normalize_homography(H)

    def get_projector_to_camera_undist_homography(self) -> np.ndarray:
        return self.H_proj_to_cam_undist.copy()

    def get_projector_corrected_to_camera_undist_homography(self) -> np.ndarray:
        H_corr_inv = self._get_inverse_projector_correction_homography_internal()
        H = self.H_proj_to_cam_undist @ H_corr_inv
        return self._normalize_homography(H)

    def get_camera_undist_to_graph_homography(self) -> np.ndarray:
        return self.H_cam_undist_to_graph.copy()

    def get_graph_to_camera_undist_homography(self) -> np.ndarray:
        return self.H_graph_to_cam_undist.copy()

    def get_graph_to_projector_homography(self) -> np.ndarray:
        return self.H_graph_to_proj.copy()

    def get_graph_to_projector_corrected_homography(self) -> np.ndarray:
        H = self.H_proj_correction @ self.H_graph_to_proj
        return self._normalize_homography(H)

    def get_projector_to_graph_homography(self) -> np.ndarray:
        return self.H_proj_to_graph.copy()

    def get_projector_corrected_to_graph_homography(self) -> np.ndarray:
        H_corr_inv = self._get_inverse_projector_correction_homography_internal()
        H = self.H_proj_to_graph @ H_corr_inv
        return self._normalize_homography(H)