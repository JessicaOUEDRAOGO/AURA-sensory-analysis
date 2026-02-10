import cv2


class CameraManager:
    def __init__(self, camera_index=1, width = 1920, height = 1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
    
    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  # Largeur de la vidéo
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  # Hauteur de la vidéo
        if not self.cap.isOpened():
            raise Exception("Impossible d'ouvrir la caméra")
        print("Camera allumé avec succès")
    
    def get_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None