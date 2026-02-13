# -*- coding: utf-8 -*-
import cv2
import numpy as np


class DrawUtils:
    @staticmethod
    def draw_points_linked(proj_points, image, where=None):
        """
        Dessine des points et les relie.
        - Si 4 points: relie en boucle (0-1-2-3-0)
        - Si >=2 points: relie en chaîne (0-1-2-...)
        """
        if image is None or image.size == 0:
            return image
        if proj_points is None or len(proj_points) == 0:
            return image

        pts = [(int(x), int(y)) for (x, y) in proj_points]

        # Dessiner points + index
        for idx, (x, y) in enumerate(pts):
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(
                image,
                str(idx),
                (x + 15, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        # Relier
        if len(pts) >= 2:
            if len(pts) == 4:
                # boucle
                for i in range(4):
                    pt1 = pts[i]
                    pt2 = pts[(i + 1) % 4]
                    cv2.line(image, pt1, pt2, (0, 255, 0), 5)
            else:
                # chaîne
                for i in range(len(pts) - 1):
                    cv2.line(image, pts[i], pts[i + 1], (0, 255, 0), 3)

        return image

    @staticmethod
    def draw_2_linked_points(proj_points_1, proj_points_2, image, additional_data=False):
        """
        Dessine une ligne entre deux points et affiche la distance si demandé.
        """
        if image is None or image.size == 0:
            return image
        if proj_points_1 is None or proj_points_2 is None:
            return image

        p1 = (int(proj_points_1[0]), int(proj_points_1[1]))
        p2 = (int(proj_points_2[0]), int(proj_points_2[1]))

        cv2.line(image, p1, p2, (0, 0, 255), 2)

        if additional_data:
            distance = float(np.linalg.norm(np.array(proj_points_1) - np.array(proj_points_2)))
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                image,
                f"{distance:.2f}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return image

    @staticmethod
    def verif_Image_Homo(image, proj_points, H):
        """
        Vérifie la transformation homographique en projetant les points.
        """
        if image is None or image.size == 0:
            return image
        if proj_points is None or H is None:
            return image

        for p in proj_points:
            p_h = np.array([p[0], p[1], 1.0], dtype=np.float32)
            projected = H @ p_h
            if abs(projected[2]) < 1e-9:
                continue
            projected /= projected[2]
            cv2.circle(image, (int(projected[0]), int(projected[1])), 5, (0, 255, 255), -1)

        return image

    @staticmethod
    def draw_math_grid_on_image(
        image,
        x_min, x_max,
        y_min, y_max,
        x_legend, y_legend,
        grid_size,
        color=(200, 200, 200)
    ):
        """
        Dessine une grille mathématique sur une copie de l'image.
        - Grille centrée sur un carré (square_size) au centre horizontal de l'image.
        - grid_size = taille logique (utilisé seulement pour cohérence, le rendu dépend du square_size).
        """
        if image is None or image.size == 0:
            return image

        img = image.copy()
        h, w = img.shape[:2]

        # Carré central: on prend la hauteur comme référence (comme avant)
        square_size = h
        x0 = (w - square_size) // 2
        y0 = 0

        # Eviter division par zéro
        if abs(x_max - x_min) < 1e-9:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0

        def index_to_pixel_x(x):
            return int(((x - x_min) / (x_max - x_min)) * square_size) + x0

        def index_to_pixel_y(y):
            return int(((y - y_min) / (y_max - y_min)) * square_size) + y0

        def calculate_grid_spacing(delta):
            max_lines = 40
            spacing = delta / max_lines
            power_of_10 = 10 ** int(np.floor(np.log10(spacing))) if spacing > 0 else 1
            ratio = spacing / power_of_10
            if ratio < 2:
                return 2 * power_of_10
            if ratio < 5:
                return 5 * power_of_10
            return 10 * power_of_10

        grid_spacing_x = calculate_grid_spacing(x_max - x_min)
        grid_spacing_y = calculate_grid_spacing(y_max - y_min)

        # Lignes verticales
        x = np.ceil(x_min / grid_spacing_x) * grid_spacing_x
        while x <= x_max + 1e-9:
            px = index_to_pixel_x(x)
            cv2.line(
                img,
                (px, index_to_pixel_y(y_min)),
                (px, index_to_pixel_y(y_max)),
                color,
                1,
                lineType=cv2.LINE_AA
            )
            x += grid_spacing_x

        # Lignes horizontales
        y = np.ceil(y_min / grid_spacing_y) * grid_spacing_y
        while y <= y_max + 1e-9:
            py = index_to_pixel_y(y)
            cv2.line(
                img,
                (index_to_pixel_x(x_min), py),
                (index_to_pixel_x(x_max), py),
                color,
                1,
                lineType=cv2.LINE_AA
            )
            y += grid_spacing_y

        # --- Axes + labels ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, square_size / 700 * 0.6)
        thickness = 2

        # Index axe X
        x = np.ceil(x_min / grid_spacing_x) * grid_spacing_x
        while x <= x_max + 1e-9:
            if abs(x) > 1e-6:
                px = index_to_pixel_x(x)
                text = f"{x:g}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = px + 2
                ty = index_to_pixel_y(0) + text_size[1] + 8
                cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            x += grid_spacing_x

        # Index axe Y
        y = np.ceil(y_min / grid_spacing_y) * grid_spacing_y
        while y <= y_max + 1e-9:
            if abs(y) > 1e-6:
                py = index_to_pixel_y(y)
                text = f"{y:g}"
                tx = index_to_pixel_x(0) + 8
                ty = py - 4
                cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y += grid_spacing_y

        # Axes
        cv2.line(
            img,
            (index_to_pixel_x(x_min), index_to_pixel_y(0)),
            (index_to_pixel_x(x_max), index_to_pixel_y(0)),
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        cv2.line(
            img,
            (index_to_pixel_x(0), index_to_pixel_y(y_min)),
            (index_to_pixel_x(0), index_to_pixel_y(y_max)),
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Légendes axes
        x_legend_text = str(x_legend)
        y_legend_text = str(y_legend)

        x_legend_size, _ = cv2.getTextSize(x_legend_text, font, font_scale, thickness)
        y_legend_size, _ = cv2.getTextSize(y_legend_text, font, font_scale, thickness)

        x_legend_pos = (index_to_pixel_x(x_max) - x_legend_size[0] - 10, index_to_pixel_y(0) - 10)
        cv2.putText(img, x_legend_text, x_legend_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        y_legend_pos = (index_to_pixel_x(0) - y_legend_size[0] - 10, index_to_pixel_y(y_max) + y_legend_size[1] + 10)
        cv2.putText(img, y_legend_text, y_legend_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return img
