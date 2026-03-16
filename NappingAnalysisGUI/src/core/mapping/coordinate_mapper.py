# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np

from src.core.utils.paths import config_path


class CoordinateMapper:
    """
    Mapper central des coordonnées pour la V2.

    Conventions :
    - entrée caméra : points image bruts (camera raw pixels)
    - undistortion : faite via K_cam / dist_cam
    - homographies runtime : définies dans runtime_mapping.json
    """

    def __init__(self):
        self.K_cam = None
        self.dist_cam = None

        self.H_cam_undist_to_proj = None
        self.H_proj_to_cam_undist = None
        self.H_cam_undist_to_graph = None
        self.H_graph_to_cam_undist = None
        self.H_graph_to_proj = None
        self.H_proj_to_graph = None

        self.grid_size = None
        self.projector_width = None
        self.projector_height = None

    def load(self):
        self._load_camera_calibration()
        self._load_runtime_mapping()

    def _load_camera_calibration(self):
        path = config_path("camera_calibration.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.K_cam = np.array(data["camera_matrix"], dtype=np.float64)
        self.dist_cam = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    def _load_runtime_mapping(self):
        path = config_path("runtime_mapping.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data["metadata"]
        homographies = data["homographies"]

        self.grid_size = int(metadata["grid_size"])
        self.projector_width = int(metadata["projector_width"])
        self.projector_height = int(metadata["projector_height"])

        self.H_cam_undist_to_proj = np.array(homographies["H_cam_undist_to_proj"], dtype=np.float64)
        self.H_proj_to_cam_undist = np.array(homographies["H_proj_to_cam_undist"], dtype=np.float64)
        self.H_cam_undist_to_graph = np.array(homographies["H_cam_undist_to_graph"], dtype=np.float64)
        self.H_graph_to_cam_undist = np.array(homographies["H_graph_to_cam_undist"], dtype=np.float64)
        self.H_graph_to_proj = np.array(homographies["H_graph_to_proj"], dtype=np.float64)
        self.H_proj_to_graph = np.array(homographies["H_proj_to_graph"], dtype=np.float64)

    @staticmethod
    def _apply_homography(H, point_2d):
        pt = np.array([point_2d[0], point_2d[1], 1.0], dtype=np.float64)
        mapped = H @ pt

        if abs(mapped[2]) < 1e-12:
            raise ValueError("Transformation homographique invalide : coordonnée homogène nulle.")

        mapped /= mapped[2]
        return mapped[:2].astype(np.float32)

    def undistort_camera_point(self, camera_point):
        """
        Convertit un point caméra brut en point caméra undistordu (pixels).
        """
        pt = np.array([[camera_point[:2]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.K_cam, self.dist_cam, P=self.K_cam)
        return und.reshape(2).astype(np.float32)

    def camera_raw_to_projector(self, camera_point):
        """
        camera raw -> camera undist -> projector
        """
        cam_undist = self.undistort_camera_point(camera_point)
        return self._apply_homography(self.H_cam_undist_to_proj, cam_undist)

    def camera_raw_to_graph(self, camera_point):
        """
        camera raw -> camera undist -> graph
        """
        cam_undist = self.undistort_camera_point(camera_point)
        return self._apply_homography(self.H_cam_undist_to_graph, cam_undist)

    def graph_to_projector(self, graph_point):
        """
        graph -> projector
        """
        return self._apply_homography(self.H_graph_to_proj, graph_point)

    def projector_to_graph(self, projector_point):
        """
        projector -> graph
        """
        return self._apply_homography(self.H_proj_to_graph, projector_point)

    def camera_raw_to_camera_undist(self, camera_point):
        return self.undistort_camera_point(camera_point)