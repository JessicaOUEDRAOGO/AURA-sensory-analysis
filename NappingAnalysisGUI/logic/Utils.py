import cv2
import numpy as np

class QuadrilateralDetector:
    @staticmethod
    def detect_largest_quadrilateral(image):
        #Convert in grey
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)
        
        blurred = cv2.GaussianBlur(b, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        # Afficher l'image seuillée
        _, thresh = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY)   

        # Seuillage pour isoler le carré blanc
        #_, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        #_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
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
        return np.array([[490,100],[3520,50],[2960,2065],[1050,2074]],dtype=np.float32)
    
    @staticmethod
    def detect_quadrilateral_from_aruco(image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        centers = {40: None, 41: None, 42: None, 43: None}  # To store centers of specific markers

        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id_int = int(ids[i][0])

                if marker_id_int == 40:
                    offset_x, offset_y = -150, 350  # Example offsets for marker 40
                elif marker_id_int == 41:
                    offset_x, offset_y = 135, 260  # Example offsets for marker 41
                elif marker_id_int == 42:
                    offset_x, offset_y = -325, -1225  # Example offsets for marker 42
                elif marker_id_int == 43:
                    offset_x, offset_y = 300, -1250  # Example offsets for marker 43
                else:
                    continue

                center_x = int(np.mean(corner[0][:, 0]) + offset_x)
                center_y = int(np.mean(corner[0][:, 1]) + offset_y)
                centers[marker_id_int] = [center_x, center_y]
                print(f"Point de la caméra pour le marqueur {marker_id_int} : {[center_x, center_y]}")

        # Check if all required markers are detected
        if any(centers[marker] is None for marker in [40, 41, 42, 43]):
            raise ValueError("Erreur : Moins de 4 marqueurs requis (40, 41, 42, 43) détectés.")

        # Return the centers in the order of markers 40, 41, 42, 43
        return np.array([centers[40], centers[41], centers[42], centers[43]], dtype=np.float32)


    """def detect_largest_quadrilateral(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_quad = None
        max_area = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > max_area:
                    largest_quad = approx
                    max_area = area
        return largest_quad"""
    
    
class HomographyTransformer:
    @staticmethod
    def find_Invers_Homography(proj_points, cam_points):
        # Calcul de l'homographie entre caméra et rétroprojecteur
        H, _ = cv2.findHomography(proj_points, cam_points)
        try:
            H_inv = np.linalg.inv(H)  # Matrice inverse pour conversion inverse
        except np.linalg.LinAlgError:
            H_inv = np.zeros_like(H)
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
            print("Erreur : La matrice H n'est pas inversible.")
        return H, H_inv


class ProjectorPoint:
    @staticmethod
    def setUp_Projector_Point(proj_points, width, height):
        # Récupérer les dimensions de l'image
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