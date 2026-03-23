# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CALIB_PATH = CONFIG_DIR / "calibration_data.json"
POSE_PATH = CONFIG_DIR / "screen_pose_manual.json"
RUNTIME_MAPPING_PATH = CONFIG_DIR / "runtime_mapping.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.array(H, dtype=np.float64)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Homographie invalide : H[2,2] trop proche de 0.")
    return H / H[2, 2]


def main():
    calib_data = load_json(CALIB_PATH)
    pose_data = load_json(POSE_PATH)

    H_proj = normalize_homography(np.array(calib_data["H_proj"], dtype=np.float64))
    H_inv_proj = normalize_homography(np.array(calib_data["H_inv_proj"], dtype=np.float64))
    H_graph = normalize_homography(np.array(calib_data["H_graph"], dtype=np.float64))
    H_inv_graph = normalize_homography(np.array(calib_data["H_inv_graph"], dtype=np.float64))

    H_graph_to_proj = normalize_homography(H_proj @ H_inv_graph)
    H_proj_to_graph = normalize_homography(np.linalg.inv(H_graph_to_proj))

    runtime_mapping_data = {
        "metadata": {
            "grid_size": 700,
            "projector_width": 3840,
            "projector_height": 2160,
            "screen_width_mm": pose_data["screen_width_mm"],
            "screen_height_mm": pose_data["screen_height_mm"],
            "source_pose_method": pose_data.get("method", "manual_screen_corners_solvepnp")
        },
        "screen_pose": {
            "rvec_screen_to_camera": pose_data["rvec_screen_to_camera"],
            "tvec_screen_to_camera": pose_data["tvec_screen_to_camera"],
            "R_screen_to_camera": pose_data["R_screen_to_camera"],
            "R_screen_to_projector": pose_data["R_screen_to_projector"],
            "T_screen_to_projector": pose_data["T_screen_to_projector"]
        },
        "homographies": {
            "H_cam_undist_to_proj": H_proj.tolist(),
            "H_proj_to_cam_undist": H_inv_proj.tolist(),
            "H_cam_undist_to_graph": H_graph.tolist(),
            "H_graph_to_cam_undist": H_inv_graph.tolist(),
            "H_graph_to_proj": H_graph_to_proj.tolist(),
            "H_proj_to_graph": H_proj_to_graph.tolist()
        },
        "reference_points": {
            "camera_points_raw_TL_TR_BR_BL": pose_data["camera_points_raw_TL_TR_BR_BL"],
            "camera_points_undist_TL_TR_BR_BL": pose_data["camera_points_undist_TL_TR_BR_BL"],
            "screen_points_mm_TL_TR_BR_BL": pose_data["screen_points_mm_TL_TR_BR_BL"],
            "graph_points_TL_TR_BR_BL": pose_data["graph_points_TL_TR_BR_BL"]
        }
    }

    with open(RUNTIME_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(runtime_mapping_data, f, indent=4)

    print("runtime_mapping.json créé avec succès :")
    print(RUNTIME_MAPPING_PATH)


if __name__ == "__main__":
    main()