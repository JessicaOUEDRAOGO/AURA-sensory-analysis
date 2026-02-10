import cv2
import os
import sys

# Dictionnaires ArUco prédéfinis et leur nombre de tags
aruco_dicts = {
    "DICT_4X4_50": (cv2.aruco.DICT_4X4_50, 50),
    "DICT_4X4_100": (cv2.aruco.DICT_4X4_100, 100),
    "DICT_4X4_250": (cv2.aruco.DICT_4X4_250, 250),
    "DICT_4X4_1000": (cv2.aruco.DICT_4X4_1000, 1000),
    "DICT_5X5_50": (cv2.aruco.DICT_5X5_50, 50),
    "DICT_5X5_100": (cv2.aruco.DICT_5X5_100, 100),
    "DICT_5X5_250": (cv2.aruco.DICT_5X5_250, 250),
    "DICT_5X5_1000": (cv2.aruco.DICT_5X5_1000, 1000),
    "DICT_6X6_50": (cv2.aruco.DICT_6X6_50, 50),
    "DICT_6X6_100": (cv2.aruco.DICT_6X6_100, 100),
    "DICT_6X6_250": (cv2.aruco.DICT_6X6_250, 250),
    "DICT_6X6_1000": (cv2.aruco.DICT_6X6_1000, 1000),
    "DICT_7X7_50": (cv2.aruco.DICT_7X7_50, 50),
    "DICT_7X7_100": (cv2.aruco.DICT_7X7_100, 100),
    "DICT_7X7_250": (cv2.aruco.DICT_7X7_250, 250),
    "DICT_7X7_1000": (cv2.aruco.DICT_7X7_1000, 1000),
    "DICT_ARUCO_ORIGINAL": (cv2.aruco.DICT_ARUCO_ORIGINAL, 1024),  # estimation
    "DICT_APRILTAG_16h5": (cv2.aruco.DICT_APRILTAG_16h5, 30),      # estimation
    "DICT_APRILTAG_25h9": (cv2.aruco.DICT_APRILTAG_25h9, 35),
    "DICT_APRILTAG_36h10": (cv2.aruco.DICT_APRILTAG_36h10, 2320),
    "DICT_APRILTAG_36h11": (cv2.aruco.DICT_APRILTAG_36h11, 587)}
    
# Taille en pixels des images générées
marker_size = 300

# Dossier de sortie principal
output_dir = "aruco_tags"
os.makedirs(output_dir, exist_ok=True)

# Boucle sur les dictionnaires
for dict_name, (dict_id, num_tags) in aruco_dicts.items():
    print(f"📦 Génération des tags pour {dict_name} ({num_tags} tags)...")
    subdir = os.path.join(output_dir, dict_name)
    os.makedirs(subdir, exist_ok=True)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    for tag_id in range(num_tags):
        try:
            # Utilisation de generateImageMarker pour générer les marqueurs
            img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, marker_size)
            filename = os.path.join(subdir, f"{tag_id}.png")
            cv2.imwrite(filename, img)
        except Exception as e:
            print(f"❌ Erreur lors de la génération du tag {tag_id} pour {dict_name}: {e}")

print("✅ Tous les tags ont été générés avec succès.")
