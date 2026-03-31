# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np

from src.core.utils.paths import config_path


class CoordinateMapper:

    def __init__(self):
        self.K_cam = None
        self.dist_cam = None

        self.H_cam_undist_to_proj = None
        self.H_proj_to_cam_undist = None
        self.H_cam_undist_to_graph = None
        self.H_graph_to_cam_undist = None

        self.H_graph_to_proj = None
        self.H_proj_to_graph = None

        self.grid_size = 700
        self.projector_width = 3840
        self.projector_height = 2160

    # ======================================================
    # LOAD
    # ======================================================
    def load(self):
        self._load_camera_calibration()
        self._load_calibration_data()
        self._build_composed_homographies()
        self._validate_loaded_data()

        print("[MAPPER] Chargé (SANS correction)")

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

    def _build_composed_homographies(self):
        self.H_graph_to_proj = self._normalize_homography(
            self.H_cam_undist_to_proj @ self.H_graph_to_cam_undist
        )

        self.H_proj_to_graph = self._normalize_homography(
            self.H_cam_undist_to_graph @ self.H_proj_to_cam_undist
        )

    def _validate_loaded_data(self):
        if self.K_cam.shape != (3, 3):
            raise ValueError("K_cam invalide")

    # ======================================================
    # UTILS
    # ======================================================
    @staticmethod
    def _normalize_homography(H):
        H = np.array(H, dtype=np.float64)
        return H / H[2, 2]

    @staticmethod
    def _apply_homography(H, pt):
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        out = H @ p
        out /= out[2]
        return out[:2].astype(np.float32)

    # ======================================================
    # CAMERA
    # ======================================================
    def undistort_camera_point(self, pt):
        p = np.array([[pt]], dtype=np.float32)
        und = cv2.undistortPoints(p, self.K_cam, self.dist_cam, P=self.K_cam)
        return und.reshape(2).astype(np.float32)

    # ======================================================
    # PIPELINES
    # ======================================================
    def camera_raw_to_projector(self, pt):
        und = self.undistort_camera_point(pt)
        return self._apply_homography(self.H_cam_undist_to_proj, und)

    def camera_raw_to_projector_nominal(self, pt):
        return self.camera_raw_to_projector(pt)

    def camera_raw_to_graph(self, pt):
        und = self.undistort_camera_point(pt)
        return self._apply_homography(self.H_cam_undist_to_graph, und)

    def graph_to_projector(self, pt):
        return self._apply_homography(self.H_graph_to_proj, pt)

    def projector_to_graph(self, pt):
        return self._apply_homography(self.H_proj_to_graph, pt)

    # ======================================================
    # GETTERS
    # ======================================================
    def get_graph_to_projector_homography(self):
        return self.H_graph_to_proj.copy()

    def get_camera_undist_to_projector_homography(self):
        return self.H_cam_undist_to_proj.copy()