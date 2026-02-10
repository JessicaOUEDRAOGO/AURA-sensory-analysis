import cv2
import pandas as pd
import numpy as np
import keyboard
import csv
import os
from screeninfo import get_monitors
from pyzbar.pyzbar import decode
import time
import threading

def setUp_Camera(id = 0, width=3840, height=2160):
    # **Ouverture de la webcam**
    cap = cv2.VideoCapture(id)
    #cap = cv2.VideoCapture('GoPro_Close_multisized.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Largeur de la vidéo
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Hauteur de la vidéo
    return cap

def display_image_on_projector_monitor(image_to_display): 
    # Récupérer les informations des écrans
    cv2.imshow("Image", image_to_display)  # Afficher l'image sur l'écran 3
    cv2.waitKey(1)

def setUp_on_main_monitor():
    # **Affichage de la vidéo sur l'écran principal (écran 1)**
    cv2.namedWindow("Capture Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Video", 1920, 1080)  # Ajuster la taille de la fenêtre vidéo

def stop_Camera(cap):
    cap.release()

def ClearAllWindows():
    cv2.destroyAllWindows()

def detect_largest_quadrilateral(image):
    #Convert in grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Seuillage pour isoler le carré blanc
    #_, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    print("--- Seuillage ---")
    #cv2.imshow("thresh", thresh)
    # Détection des contours

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Recherche du plus grand quadrilatère
    quadrilateral = None
    max_area = 0

    #Detect the biggest Area, so that willing quadrilater
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:  # Vérifier si c'est un quadrilatère
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                quadrilateral = approx

    #Find the four corners
    proj_points = quadrilateral.reshape(4, 2)  # Obtenir les coordonnées des coins
    print("Coins du quadrilatère:\n", proj_points)
    return proj_points

def draw_points_linked(proj_points, image, where):
    # Dessiner les coins
    for x, y in proj_points:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

    # Relier les coins par des lignes
    for i in range(4):
        pt1 = tuple(proj_points[i])
        pt2 = tuple(proj_points[(i + 1) % 4])  # Boucle circulaire pour relier tous les points
        cv2.line(image, pt1, pt2, (0, 255, 0), 3)

    # Afficher l'image
    cv2.imshow(f"{where}", image)

def draw_2_linked_points(proj_points_1, proj_points_2, image, additional_data = False):

    pt1 = tuple(proj_points_1.astype(int))
    pt2 = tuple(proj_points_2.astype(int))

    cv2.line(image, pt1, pt2, (255, 0, 255), 5)  # Tracer la ligne en vert (épaisseur 3)

    if additional_data:
        # Calcul des longueurs des segments
        x1, y1 = pt1
        x2, y2 = pt2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        L2 = int(np.sqrt(dx**2 + dy**2))

        # Déterminer la position du triangle rectangle en dessous
        if y1 > y2:  # pt1 est plus bas que pt2
            base_x, base_y = x1, y1
            top_x, top_y = x2, y2
        else:  # pt2 est plus bas que pt1
            base_x, base_y = x2, y2
            top_x, top_y = x1, y1

        # Déplacer les traits sous la ligne L2
        thickness = 2

        # Positions des traits en dessous de la ligne principale
        cv2.line(image, (base_x, base_y), (top_x, base_y), (255, 255, 0), thickness)  # dx en dessous
        cv2.line(image, (top_x, base_y), (top_x, top_y), (0, 255, 255), thickness)  # dy en dessous

        # Ajouter le texte des distances en dessous aussi
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        cv2.putText(image, f"{dx}px", ((base_x + top_x) // 2, base_y + 5 + 15), font, font_scale, (255, 255, 0), font_thickness)  # dx
        cv2.putText(image, f"{dy}px", (top_x + 5, ((top_y + base_y) // 2)+15), font, font_scale, (0, 255, 255), font_thickness)  # dy
        cv2.putText(image, f"{L2}px", ((x1 + x2) // 2, ((y1 + y2) // 2)-15), font, font_scale, (255, 0, 255), font_thickness)  # L2


    # Afficher l'image

def setUp_Projector_Point(proj_points):
    # Récupérer les dimensions de l'image
    width = 3840
    height = 2160

    cam_points = []
    for i in range(4):
        points = []
        if proj_points[i][0] < (proj_points[i][0]+proj_points[1][0]+proj_points[2][0]+proj_points[3][0]) / 4:
            points.append(0)
        else:
            points.append(width)

        if proj_points[i][1] < (proj_points[i][1]+proj_points[1][1]+proj_points[2][1]+proj_points[3][1]) / 4:
            points.append(0)
        else:
            points.append(height)
        cam_points.append(points)
    cam_points = np.array(cam_points)

    print("cam_points (NumPy) :")
    print(cam_points)
    return cam_points

def find_Invers_Homography(proj_points, cam_points):
    # Calcul de l'homographie entre caméra et rétroprojecteur
    H, _ = cv2.findHomography(proj_points, cam_points)
    H_inv = np.linalg.inv(H)  # Matrice inverse pour conversion inverse
    return H, H_inv

def detect_qr_codes(frame):
    qr_codes = decode(frame)  # Détection des QR codes
    if qr_codes:
        qr_data = qr_codes[0].data.decode("utf-8")  # Extraire les données du premier QR code
        x, y, w, h = qr_codes[0].rect  # Extraire les coordonnées sous forme de tuple
        center = (x + w // 2, y + h // 2)  # Calcul du centre
        return np.array(center)  # Retourne le centre du QR Code en numpy array
    return None

def transform_to_projector(camera_point, H):
    camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
    projected_point = np.dot(H, camera_point)
    projected_point /= projected_point[2]  # Normalisation
    return projected_point[:2]

def transform_to_camera(projector_point):
    projector_point = np.array([projector_point[0], projector_point[1], 1], dtype=np.float32)
    camera_point = np.dot(H_inv, projector_point)
    camera_point /= camera_point[2]  # Normalisation
    return camera_point[:2]

def verif_Image_Homo(image, proj_points, H):
    cv2.circle(image, tuple(np.round(transform_to_projector(proj_points[0], H)).astype(int)), 50, (0, 255, 0), 3)
    cv2.circle(image, tuple(np.round(transform_to_projector(proj_points[1], H)).astype(int)), 50, (0, 255, 0), 3)
    cv2.circle(image, tuple(np.round(transform_to_projector(proj_points[2], H)).astype(int)), 50, (0, 255, 0), 3)
    cv2.circle(image, tuple(np.round(transform_to_projector(proj_points[3], H)).astype(int)), 50, (0, 255, 0), 3)
    return image

def calculate_distances(projector_coords_ArUco):
    """Calculer les distances entre chaque paire de points et retourner un DataFrame."""
    data = []
    
    for i in range(len(projector_coords_ArUco)):
        point_1_id, point_1_coord = projector_coords_ArUco[i]
        
        for j in range(i + 1, len(projector_coords_ArUco)):
            point_2_id, point_2_coord = projector_coords_ArUco[j]
            
            # Calcul des distances
            dis_x = point_1_coord[0] - point_2_coord[0]
            dis_y = point_1_coord[1] - point_2_coord[1]
            dis_l2 = np.sqrt(dis_x**2 + dis_y**2)

            # Arrondi
            dis_x, dis_y, dis_l2 = round(dis_x, 1), round(dis_y, 1), round(dis_l2, 1)
            
            # Stocker dans la liste
            data.append([point_1_id, point_2_id, dis_x, dis_y, dis_l2])

    # Création d'un DataFrame
    return pd.DataFrame(data, columns=['Point_1_ID', 'Point_2_ID', 'Dist_X', 'Dist_Y', 'Dist_L2'])

def save_to_csv(frame, projector_coords_ArUco, filename='positions.csv'):
    """
    Sauvegarde les données sous forme de fichier CSV.
    
    :param frame: Numéro de la frame actuelle
    :param projector_coords_ArUco: Liste des coordonnées des marqueurs ArUco sous forme [[ID, [x, y]]]
    :param filename: Nom du fichier CSV de sortie
    """
    
    """Optimisation : Évite les lectures et écritures excessives sur disque"""
    
    # Trier les coordonnées pour un ordre cohérent
    projector_coords_ArUco = sorted(projector_coords_ArUco, key=lambda x: x[0])

    # Convertir les données en une ligne CSV
    ids = [f"ID_{entry[0]}" for entry in projector_coords_ArUco]
    coords = [str(list(entry[1])) for entry in projector_coords_ArUco]

    # Construire une ligne sous forme de dictionnaire
    new_data = {"frame": frame, **dict(zip(ids, coords))}

    # Convertir en DataFrame
    new_df = pd.DataFrame([new_data])

    # Vérifier si le fichier existe et doit contenir un header
    file_exists = os.path.isfile(filename)

    # Sauvegarde en mode append, sans recharger tout le fichier
    new_df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    #print(f"Données enregistrées dans {filename}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

def find_coords_by_index(index, all_coords):
    """Recherche les coordonnées associées à un index dans all_coords."""
    for idx, coords in all_coords:
        if idx == index:
            return coords  # Retourne directement les coordonnées si l'index est trouvé
    return None  # Retourne None si l'index n'est pas trouvé
    
def analysis_loop(frame, detector, image_background, H, proj_points,t1):
    
    global status
    global all_aruco
    global display_image
    global start_loop
    global end_loop

    # Traitement CPU standard
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des marqueurs (toujours sur CPU)
    corners, ids, _ = detector.detectMarkers(gray)
    t1.join()
    
    status = False 

    # Copie de l'image de fond pour éviter de dessiner plusieurs cercles
    #display_image = image_background.copy()    
    
    if ids is not None:
        display_image = image_background.copy()

        # Stocker les coordonnées transformées
        all_aruco = []
        
        cv2.aruco.drawDetectedMarkers(frame, corners)

        # Extraction des informations des ArUco en une seule étape
        aruco_centers = np.array([np.mean(corner[0], axis=0) for corner in corners])  # Calcul vectorisé des centres

        # Transformation en une seule opération si possible
        projector_coords_ArUco = np.array([transform_to_projector(center, H) for center in aruco_centers])
        
        # Boucle optimisée avec `zip()` pour éviter plusieurs accès mémoire
        for aruco_id, center in zip(ids.flatten(), projector_coords_ArUco):
            all_aruco.append([aruco_id, center])

            # Dessin des éléments en OpenCV
            center_int = tuple(center.astype(int))
            cv2.circle(display_image, center_int, 25, (0, 0, 255), 5)
            cv2.putText(display_image, f"ID: {aruco_id}", (center_int[0] + 10, center_int[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        t1.join()
        status = True



def main():
    
    global status
    global c
    global all_aruco
    global display_image
    global start_loop
    global end_loop

    start_loop = 0
    cap = setUp_Camera()
    image_background = cv2.imread("blanc.png")

    # Initialisation de la fenêtre (à faire une seule fois au lancement)
    monitors = get_monitors()
    monitor = monitors[2]  # Écran numéro 3 (indice 2)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image", monitor.x, monitor.y)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    display_image_on_projector_monitor(image_background)
    
    setUp_on_main_monitor()

    while True:
        status = False
        ret, frame = cap.read()
        #print(frame.shape)
        cv2.imshow("Capture Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            Save_Frame = frame.copy()
            break
        elif key == ord('q'):  # Quitter en appuyant sur 'q'
            break

    ClearAllWindows()

    proj_points = detect_largest_quadrilateral(Save_Frame)
    #draw_points_linked(proj_points, Save_Frame, "Quadrilatere detecte")
    cam_points = setUp_Projector_Point(proj_points)
    H, H_inv = find_Invers_Homography(proj_points, cam_points)
    image_verified = verif_Image_Homo(image_background.copy(), proj_points, H)


    display_image_on_projector_monitor(image_verified)

    # Initialisation du dictionnaire et des paramètres de détection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    c= 0
    Status_Index = True
    index_1 = None
    index_2 = None
    #cv2.setUseOptimized(True)
    #cv2.setNumThreads(4)  # Adapter selon le CPU
    minx = min(proj_points[0][0],proj_points[1][0],proj_points[2][0],proj_points[3][0])
    maxx = max(proj_points[0][0],proj_points[1][0],proj_points[2][0],proj_points[3][0])
    miny = min(proj_points[0][1],proj_points[1][1],proj_points[2][1],proj_points[3][1])
    maxy = max(proj_points[0][1],proj_points[1][1],proj_points[2][1],proj_points[3][1])
    t1 = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = threading.Thread(target=analysis_loop, args=(frame, detector, image_background, H, proj_points,t1))
        t1.start()
        
        # Quitter avec 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if Status_Index == True:    
            if key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('6') or key == ord('7') or key == ord('8') or key == ord('9'):
                index_1 = chr(key)
                print(index_1)
                while key != 13:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('6') or key == ord('7') or key == ord('8') or key == ord('9'):
                        index_1 += chr(key)
                index_1 = int(index_1)
                print(index_1)
                Status_Index = False
                
        else:    
            if key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('6') or key == ord('7') or key == ord('8') or key == ord('9'):
                index_2 = chr(key)
                print(index_2)
                while key != 13:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('6') or key == ord('7') or key == ord('8') or key == ord('9'):
                        index_2 += chr(key)
                index_2 = int(index_2)
                print(index_2)
                Status_Index = True
        
        if status == True:
            if index_1 is not None and index_2 is not None:
                index_1_coord = find_coords_by_index(index_1, all_aruco)
                index_2_coord = find_coords_by_index(index_2, all_aruco)
                print(f"coord : i1 = {index_1_coord} , i2 = {index_2_coord}")
                
                if index_1_coord is not None and index_2_coord is not None:
                    draw_2_linked_points(index_1_coord, index_2_coord, display_image, additional_data = True)
            display_image_on_projector_monitor(display_image)
            end_loop = time.time()
            print(f"  - Temps total de la boucle : {end_loop - start_loop:.6f} s - {(end_loop - start_loop)/(end_loop - start_loop)*100:.3f}%")
            start_loop = time.time()

            status = False
            
            save_to_csv(c, all_aruco)

            c+=1
        

        setUp_on_main_monitor()
        draw_points_linked(proj_points, frame, "Capture Video")
        cv2.imshow("Capture Video", frame)
        print("\n")


    stop_Camera(cap)
    ClearAllWindows()


if __name__ == "__main__":
    main()