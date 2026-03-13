# -*- coding: utf-8 -*-
import cv2
from pathlib import Path

ROOT = Path("projector_calibration_data")
PATTERN_SIZE = (9, 6)

valid_poses = []
invalid_poses = []

for pose_dir in sorted(ROOT.glob("pose_*")):
    img_path = pose_dir / "camera_captures" / "white.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"{pose_dir.name} : white.png introuvable")
        invalid_poses.append(pose_dir.name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, None)

    if found:
        valid_poses.append(pose_dir.name)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, found)
        preview = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow("valid_pose", preview)
        cv2.waitKey(300)
        print(f"{pose_dir.name} : OK")
    else:
        invalid_poses.append(pose_dir.name)
        print(f"{pose_dir.name} : ECHEC")

cv2.destroyAllWindows()

print("\n=== BILAN ===")
print("Poses valides :", valid_poses)
print("Poses invalides :", invalid_poses)
print("Nb valides :", len(valid_poses))