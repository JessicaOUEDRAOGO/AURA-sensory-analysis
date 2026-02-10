import numpy as np
import cv2

class GeometryUtils:
    @staticmethod
    def draw_polygon(image, points, color=(0, 255, 0), thickness=2):
        if points is not None:
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        return image
    
    @staticmethod
    def compute_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))