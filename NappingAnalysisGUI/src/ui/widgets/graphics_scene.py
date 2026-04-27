from PyQt6.QtWidgets import QGraphicsScene
from PyQt6.QtGui import QPainter, QPen, QBrush, QPixmap
from PyQt6.QtCore import Qt, QRectF
import math
import numpy as np
from PyQt6.QtGui import QColor
class GraphicsScene(QGraphicsScene):
    def __init__(self, grid_size=700, status_mathsElement = False, grid_xmin = -301, grid_xmax = 399, grid_ymin = -336, grid_ymax = 364,
                  x_min=-10, x_max=10, y_min=-10, y_max=10, x_legend = "x", y_legend = "y", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size  # Taille des carreaux de la grille
        self.grid_xmin = grid_xmin  # Taille des carreaux de la grille
        self.grid_xmax = grid_xmax  # Taille des carreaux de la grille
        self.grid_ymin = grid_ymin  # Taille des carreaux de la grille
        self.grid_ymax = grid_ymax  # Taille des carreaux de la grille

        self.x_legend = x_legend
        self.y_legend = y_legend

        self.status_mathsElement = status_mathsElement

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.marker_items = []  # Liste pour stocker les marqueurs affichés

        self.layers = {}  # Store layers by ID
    
    def add_layer(self, layer):
        self.layers[layer.layer_id] = layer
        layer.render(self)

    def update_layer(self, layer_id, **kwargs):
        if layer_id in self.layers:
            self.layers[layer_id].update_properties(**kwargs)
            self.layers[layer_id].render(self)

    def remove_layer(self, layer_id):
        if layer_id in self.layers:
            layer = self.layers.pop(layer_id)
            if layer.graphics_item:
                self.removeItem(layer.graphics_item)

    def update_bounds(self, x_min, x_max, y_min, y_max, x_legend, y_legend):
        """Met à jour les limites de la grille et redessine."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_legend = x_legend
        self.y_legend = y_legend
        self.update()
    
    
    def add_marker(self, x, y, marker_id, state="POSEE"):
        """
        Ajoute un marqueur (point rouge) et son ID à la scène.
        :param x: Coordonnée X dans le domaine graphique.
        :param y: Coordonnée Y dans le domaine graphique.
        :param marker_id: ID du marqueur ArUco.
        :param state: État du marqueur.
        """
        STATE_COLORS = {
            "POSEE":              Qt.GlobalColor.red,
            "PEUT_ETRE_SOULEVEE": Qt.GlobalColor.yellow,   # orange approx Qt
            "SOULEVEE":           Qt.GlobalColor.blue,
        }
        # Pour avoir un vrai orange (non dispo en GlobalColor), on utilise QColor
        
        STATE_QCOLORS = {
            "POSEE":              QColor(255, 0, 0),
            "PEUT_ETRE_SOULEVEE": QColor(255, 165, 0),
            "SOULEVEE":           QColor(0, 100, 255),
        }
        color = STATE_QCOLORS.get(state, QColor(255, 0, 0))

        px = x + self.grid_xmin
        py = y + self.grid_ymin

        point_item = self.addEllipse(
            px, py, 15, 15,
            QPen(color), QBrush(color)
        )
        self.marker_items.append(point_item)

        text_item = self.addText(str(marker_id))
        text_item.setDefaultTextColor(color)
        text_item.setPos(px + 13, py + 13)
        self.marker_items.append(text_item)

    def clear_markers(self):
        """
        Efface tous les marqueurs (points et IDs) de la scène.
        """
        for item in self.marker_items:
            self.removeItem(item)
        self.marker_items.clear()

    def index_to_pixel_x(self, x):
        """Convertit une coordonnée d'index X en pixels."""
        return int( (((x - self.x_min)/(self.x_max-self.x_min)) * self.grid_size) + (self.grid_xmin) )
    
    def index_to_pixel_y(self, y):
        """Convertit une coordonnée d'index Y en pixels."""
        return int( (((y - self.y_min)/(self.y_max-self.y_min)) * self.grid_size) + (self.grid_ymin) )
    
    def pixel_to_index_x(self, px):
        """Convertit une coordonnée en pixels X en index graphique."""
        return px/(self.grid_xmax-self.grid_xmin) * (self.x_max - self.x_min) + self.x_min

    def pixel_to_index_y(self, py):
        """Convertit une coordonnée en pixels Y en index graphique."""
        return py/(self.grid_ymax-self.grid_ymin) * (self.y_max - self.y_min) + self.y_min
    
    def calculate_grid_spacing(self, delta):
        # Choisir un nombre de lignes visible optimal en fonction de la portée (delta)
        max_lines = 40  # Nombre maximal de lignes visibles
        spacing = delta / max_lines  # Espacement de base, divise la portée par le nombre maximal de lignes

        # Ajuster l'espacement pour que ce soit un multiple de 1, 2, 5, etc., pour la lisibilité
        # Choisir la puissance de 10 la plus proche, puis ajuster en conséquence
        power_of_10 = 10 ** int(np.floor(np.log10(spacing)))  # Trouver la puissance de 10 la plus proche
        if spacing / power_of_10 < 2: 
            spacing = 2 * power_of_10
        elif spacing / power_of_10 < 5:
            spacing = 5 * power_of_10
        else:
            spacing = 10 * power_of_10

        return spacing

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Dessine le fond avec les lignes de la grille."""
        
        grid_spacing_x = self.calculate_grid_spacing(self.x_max - self.x_min)
        grid_spacing_y = self.calculate_grid_spacing(self.y_max - self.y_min)
        
        # Dessiner les lignes verticales
        x = math.ceil(self.x_min / grid_spacing_x) * grid_spacing_x
        while x <= self.x_max:
            x = round(x, 2)
            px = self.index_to_pixel_x(x)
            painter.setPen(QPen(Qt.GlobalColor.lightGray, 1))
            painter.drawLine(px, self.index_to_pixel_y(self.y_min), px, self.index_to_pixel_y(self.y_max))
            x += grid_spacing_x
        
        # Dessiner les lignes horizontales
        y = math.ceil(self.y_min / grid_spacing_y) * grid_spacing_y
        while y <= self.y_max:
            y = round(y, 2)
            py = self.index_to_pixel_y(y)
            painter.setPen(QPen(Qt.GlobalColor.lightGray, 1))
            painter.drawLine(self.index_to_pixel_x(self.x_min), py, self.index_to_pixel_x(self.x_max), py)
            y += grid_spacing_y

        if self.status_mathsElement == True:
            self.drawMathElements(painter)

    def drawMathElements(self, painter: QPainter):
        """Dessine les éléments mathématiques : axes principaux, unités et légendes."""
        grid_spacing_x = self.calculate_grid_spacing(self.x_max - self.x_min)
        grid_spacing_y = self.calculate_grid_spacing(self.y_max - self.y_min)
        
        # Dessiner les unités sur les axes
        x = math.ceil(self.x_min / grid_spacing_x) * grid_spacing_x
        while x <= self.x_max:
            x = round(x, 2)
            px = self.index_to_pixel_x(x)
            if x != 0:  # Éviter d'afficher "0" sur l'axe lui-même
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawText(px + 2, self.index_to_pixel_y(0) - 2 + 17, str(x))
            x += grid_spacing_x
        
        y = math.ceil(self.y_min / grid_spacing_y) * grid_spacing_y
        while y <= self.y_max:
            y = round(y, 2)
            py = self.index_to_pixel_y(y)
            if y != 0:  # Éviter d'afficher "0" sur l'axe lui-même
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawText(self.index_to_pixel_x(0) + 5, py - 2, str(y))
            y += grid_spacing_y

        # Dessiner les axes X et Y en noir
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawLine(int(self.index_to_pixel_x(self.x_min)), int(self.index_to_pixel_y(0)), int(self.index_to_pixel_x(self.x_max)), int(self.index_to_pixel_y(0)))  # Axe X
        painter.drawLine(int(self.index_to_pixel_x(0)), int(self.index_to_pixel_y(self.y_min)), int(self.index_to_pixel_x(0)), int(self.index_to_pixel_y(self.y_max)))  # Axe Y

        # Ajouter les légendes des axes
        x_legend_pos = self.index_to_pixel_x(self.x_max) - (int(np.floor(5.3 * len(self.x_legend)))) - 5
        y_legend_pos = self.index_to_pixel_y(0) - 10  # Position légèrement en dessous de l'axe X
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawText(x_legend_pos, y_legend_pos, self.x_legend)

        x_legend_pos = self.index_to_pixel_x(0) - (int(np.floor(5.3 * len(self.y_legend)))) - 5  # Position de la légende Y légèrement à droite de l'axe
        y_legend_pos = self.index_to_pixel_y(self.y_max) - 10  # Position vers le bas de l'axe Y
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawText(x_legend_pos, y_legend_pos, self.y_legend)

    def display_image(self, image_path, x=None, y=None, scale_factor=1.0):
        """
        Affiche une image dans la scène graphique.
        :param image_path: Chemin de l'image à afficher.
        :param x: Coordonnée X du centre de l'image (par défaut, centré dans la scène).
        :param y: Coordonnée Y du centre de l'image (par défaut, centré dans la scène).
        :param scale_factor: Facteur multiplicateur pour la taille de l'image.
        """
        print(x, y, scale_factor)
        # Charger l'image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"⚠️ Impossible de charger l'image : {image_path}")
            return
        # Appliquer le facteur de mise à l'échelle (convertir en entier)
        pixmap = pixmap.scaled(int(pixmap.width() * scale_factor), int(pixmap.height() * scale_factor), Qt.AspectRatioMode.KeepAspectRatio)

        # Calculer la position par défaut si x et y ne sont pas spécifiés
        if x is None:
            x = (self.sceneRect().width() - pixmap.width()) / 2
        if y is None:
            y = (self.sceneRect().height() - pixmap.height()) / 2

        # Ajouter l'image à la scène
        self.clear()  # Effacer les éléments existants si nécessaire
        self.addPixmap(pixmap).setPos(x, y)