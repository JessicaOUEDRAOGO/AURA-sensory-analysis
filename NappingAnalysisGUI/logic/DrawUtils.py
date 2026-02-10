import cv2
import numpy as np

class DrawUtils:
    @staticmethod
    def draw_points_linked(proj_points, image, where):
        """
        Dessine les points projetés et les relie par des lignes.
        :param proj_points: Liste de points (x, y) projetés.
        :param image: Image sur laquelle dessiner.
        :param where: Couleur ou style spécifique.
        """
        # Dessiner les coins
        for x, y in proj_points:
            cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)

        # Relier les coins par des lignes
        for i in range(4):
            pt1 = tuple(proj_points[i])
            pt2 = tuple(proj_points[(i + 1) % 4])  # Boucle circulaire pour relier tous les points
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0),
                     5)
        return image
    
    @staticmethod
    def draw_2_linked_points(proj_points_1, proj_points_2, image, additional_data=False):
        """
        Dessine une ligne entre deux ensembles de points et affiche la distance.
        :param proj_points_1: Premier point (x, y).
        :param proj_points_2: Deuxième point (x, y).
        :param image: Image sur laquelle dessiner.
        :param additional_data: Afficher la distance entre les points.
        """
        cv2.line(image, (int(proj_points_1[0]), int(proj_points_1[1])), 
                 (int(proj_points_2[0]), int(proj_points_2[1])), (0, 0, 255), 2)
        
        if additional_data:
            distance = np.linalg.norm(np.array(proj_points_1) - np.array(proj_points_2))
            midpoint = ((proj_points_1[0] + proj_points_2[0]) // 2, 
                        (proj_points_1[1] + proj_points_2[1]) // 2)
            cv2.putText(image, f"{distance:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    @staticmethod
    def verif_Image_Homo(image, proj_points, H):
        """
        Vérifie la transformation homographique en projetant les points.
        :param image: Image sur laquelle dessiner.
        :param proj_points: Liste de points à projeter.
        :param H: Matrice d'homographie.
        """
        for p in proj_points:
            p_homogeneous = np.array([p[0], p[1], 1])
            projected_p = np.dot(H, p_homogeneous)
            projected_p /= projected_p[2]  # Normalisation
            cv2.circle(image, (int(projected_p[0]), int(projected_p[1])), 5, (0, 255, 255), -1)

    @staticmethod
    def draw_math_grid_on_image(
        image, x_min, x_max, y_min, y_max, x_legend, y_legend, grid_size, color=(200,200,200)
    ):
        """
        Dessine la grille mathématique sur une copie de l'image fournie.
        La grille est centrée sur un carré de taille grid_size x grid_size au centre de l'image.
        Les axes, index et légendes sont dessinés comme dans GraphicsScene/drawMathElements.
        """
        img = image.copy()
        h, w = img.shape[:2]
        # Définir le carré central
        square_size = h  # On prend la hauteur comme référence
        x0 = (w - square_size) // 2
        y0 = 0

        # Fonctions de conversion index <-> pixel
        def index_to_pixel_x(x):
            return int(((x - x_min) / (x_max - x_min)) * square_size) + x0

        def index_to_pixel_y(y):
            return int(((y - y_min) / (y_max - y_min)) * square_size) + y0

        # Espacement optimal (reprendre calculate_grid_spacing)
        def calculate_grid_spacing(delta):
            max_lines = 40
            spacing = delta / max_lines
            power_of_10 = 10 ** int(np.floor(np.log10(spacing)))
            if spacing / power_of_10 < 2:
                spacing = 2 * power_of_10
            elif spacing / power_of_10 < 5:
                spacing = 5 * power_of_10
            else:
                spacing = 10 * power_of_10
            return spacing

        grid_spacing_x = calculate_grid_spacing(x_max - x_min)
        grid_spacing_y = calculate_grid_spacing(y_max - y_min)

        # Lignes verticales
        x = np.ceil(x_min / grid_spacing_x) * grid_spacing_x
        while x <= x_max:
            px = index_to_pixel_x(x)
            cv2.line(img, (px, index_to_pixel_y(y_min)), (px, index_to_pixel_y(y_max)), color, 1, lineType=cv2.LINE_AA)
            x += grid_spacing_x

        # Lignes horizontales
        y = np.ceil(y_min / grid_spacing_y) * grid_spacing_y
        while y <= y_max:
            py = index_to_pixel_y(y)
            cv2.line(img, (index_to_pixel_x(x_min), py), (index_to_pixel_x(x_max), py), color, 1, lineType=cv2.LINE_AA)
            y += grid_spacing_y

        # --- Éléments mathématiques : index, axes, légendes ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, square_size / 700 * 0.6)
        thickness = 2

        # Index sur l'axe X
        x = np.ceil(x_min / grid_spacing_x) * grid_spacing_x
        while x <= x_max:
            px = index_to_pixel_x(x)
            if abs(x) > 1e-6:  # Éviter d'afficher "0" sur l'axe lui-même
                text = f"{x:g}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = px + 2
                ty = index_to_pixel_y(0) + text_size[1] + 8
                cv2.putText(img, text, (tx, ty), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            x += grid_spacing_x

        # Index sur l'axe Y
        y = np.ceil(y_min / grid_spacing_y) * grid_spacing_y
        while y <= y_max:
            py = index_to_pixel_y(y)
            if abs(y) > 1e-6:
                text = f"{y:g}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = index_to_pixel_x(0) + 8
                ty = py - 4
                cv2.putText(img, text, (tx, ty), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            y += grid_spacing_y

        # Axes principaux (en noir)
        cv2.line(img, (index_to_pixel_x(x_min), index_to_pixel_y(0)), (index_to_pixel_x(x_max), index_to_pixel_y(0)), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(img, (index_to_pixel_x(0), index_to_pixel_y(y_min)), (index_to_pixel_x(0), index_to_pixel_y(y_max)), (0,0,0), 2, cv2.LINE_AA)

        # Légendes des axes
        x_legend_text = str(x_legend)
        y_legend_text = str(y_legend)
        x_legend_size, _ = cv2.getTextSize(x_legend_text, font, font_scale, thickness)
        y_legend_size, _ = cv2.getTextSize(y_legend_text, font, font_scale, thickness)

        # Légende X (en bas à droite du carré)
        x_legend_pos = (index_to_pixel_x(x_max) - x_legend_size[0] - 10, index_to_pixel_y(0) - 10)
        cv2.putText(img, x_legend_text, x_legend_pos, font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        # Légende Y (en haut à gauche du carré, à côté de l'axe Y)
        y_legend_pos = (index_to_pixel_x(0) - y_legend_size[0] - 10, index_to_pixel_y(y_max) + y_legend_size[1] + 10)
        cv2.putText(img, y_legend_text, y_legend_pos, font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

        return img
    
