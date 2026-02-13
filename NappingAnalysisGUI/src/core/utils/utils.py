# -*- coding: utf-8 -*-
import cv2
import numpy as np


class QuadrilateralDetector:
    @staticmethod
    def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
        """
        Ordonne 4 points en: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        pts: (4,2)
        """
        pts = np.array(pts, dtype=np.float32)

        s = pts.sum(axis=1)          # x+y
        diff = np.diff(pts, axis=1)  # x-y

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def detect_largest_quadrilateral(image: np.ndarray) -> np.ndarray:
        """
        Détecte le plus grand quadrilatère sur une image (si tu en as besoin).
        Retourne 4 points ordonnés (TL, TR, BR, BL).
        """
        if image is None or image.size == 0:
            raise ValueError("Image invalide.")

        # Traitement (ton approche sur canal bleu conservée)
        b, g, r = cv2.split(image)
        blurred = cv2.GaussianBlur(b, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)

        _, thresh = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quadrilateral = None
        max_area = 0.0

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    quadrilateral = approx

        if quadrilateral is None:
            raise ValueError("Aucun quadrilatère détecté.")

        pts = quadrilateral.reshape(4, 2).astype(np.float32)
        pts = QuadrilateralDetector._order_points_tl_tr_br_bl(pts)

        print("Coins du quadrilatère (ordonnés TL,TR,BR,BL):\n", pts)
        return pts

    @staticmethod
    def detect_quadrilateral_from_aruco(image: np.ndarray) -> np.ndarray:
        """
        Détecte les 4 tags ArUco 40,41,42,43 et retourne leurs centres (avec offsets),
        puis ordonne en TL, TR, BR, BL.
        """
        if image is None or image.size == 0:
            raise ValueError("Image invalide.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        # Centres des markers requis
        centers = {40: None, 41: None, 42: None, 43: None}

        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])

                # Offsets (tes valeurs conservées)
                if marker_id == 40:
                    offset_x, offset_y = -150, 350
                elif marker_id == 41:
                    offset_x, offset_y = 135, 260
                elif marker_id == 42:
                    offset_x, offset_y = -325, -1225
                elif marker_id == 43:
                    offset_x, offset_y = 300, -1250
                else:
                    continue

                cx = float(np.mean(corner[0][:, 0]) + offset_x)
                cy = float(np.mean(corner[0][:, 1]) + offset_y)

                centers[marker_id] = [cx, cy]
                print(f"Point caméra (avec offset) marker {marker_id}: {[cx, cy]}")

        if any(centers[m] is None for m in [40, 41, 42, 43]):
            raise ValueError("Moins de 4 marqueurs requis (40,41,42,43) détectés.")

        pts = np.array([centers[40], centers[41], centers[42], centers[43]], dtype=np.float32)

        # IMPORTANT: ordonner TL,TR,BR,BL pour homographie stable
        pts = QuadrilateralDetector._order_points_tl_tr_br_bl(pts)
        return pts


class HomographyTransformer:
    @staticmethod
    def find_Invers_Homography(proj_points: np.ndarray, cam_points: np.ndarray):
        """
        Calcule H et H_inv. proj_points et cam_points: (4,2) float32 ordonnés de façon cohérente.
        """
        proj_points = np.array(proj_points, dtype=np.float32)
        cam_points = np.array(cam_points, dtype=np.float32)

        if proj_points.shape != (4, 2) or cam_points.shape != (4, 2):
            raise ValueError(f"proj_points/cam_points doivent être (4,2). Reçu: {proj_points.shape} / {cam_points.shape}")

        H, _ = cv2.findHomography(proj_points, cam_points)
        if H is None:
            raise ValueError("Homographie non calculable (H=None). Vérifie tes points.")

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            raise ValueError("La matrice H n'est pas inversible (singulière).")

        return H, H_inv


class ProjectorPoint:
    @staticmethod
    def setUp_Projector_Point(proj_points: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Associe chaque coin détecté à un coin écran:
        - retourne des points dans {0, width-1} x {0, height-1}
        ATTENTION: Ici on suppose proj_points ordonnés TL,TR,BR,BL.
        """
        proj_points = np.array(proj_points, dtype=np.float32)
        if proj_points.shape != (4, 2):
            raise ValueError("proj_points doit être de shape (4,2).")

        # coins de destination (TL,TR,BR,BL)
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        print("Projector points (dst) :\n", dst)
        return dst
