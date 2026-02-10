from PyQt6.QtGui import QPixmap
from PyQt6 import uic, QtWidgets
import numpy as np
import cv2

from logic.CameraManager import CameraManager
from logic.DisplayManager import DisplayManager
from logic.Algorithm_Calibration import Calibration

class CalibrationMenu(QtWidgets.QWidget):
    def __init__(self, parent, image_background, cam_width = 3840, cam_height = 2160, grid_size = 700):
        super().__init__()

        uic.loadUi("./gui/Calibration_Menu.ui", self)

        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)

        self.parent = parent  # Référence à MainApp
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.grid_size = grid_size
        self.camera_manager = CameraManager(camera_index = self.parent.settings["camera_id"], width = self.cam_width, height = self.cam_height)
        self.display_manager = DisplayManager(label = self.label_frame)
        self.calibration = Calibration(self, self.cam_width, self.cam_height, self.grid_size, image_background)


        # Charger et afficher l'image caméra éteinte à l'initialisation
        self.image_cam_closed = cv2.imread("./assets/images/camera_eteinte.png")        
        if self.image_cam_closed is None:
            # Créer une image noire par défaut si le chargement échoue
            self.image_cam_closed = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pushButton_startCam.clicked.connect(self.start_cam)
        self.pushButton_closeCam.clicked.connect(self.close_camera)
        self.pushButton_StartCali.clicked.connect(self.start_cali)
        self.pushButton_retour.clicked.connect(self.go_to_record)        

    def keyPressEvent(self, event):
        self.key_handler.handle_key(event)
        super().keyPressEvent(event)

    def close_camera(self):
        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)
    
        # Afficher l'image caméra éteinte
        self.display_manager.show_frame(self.image_cam_closed)

        self.calibration.timer.stop()
        self.camera_manager.close_camera()        
 
    def start_cam(self):
        self.pushButton_closeCam.setEnabled(True)
        self.pushButton_StartCali.setEnabled(True)
        self.pushButton_startCam.setEnabled(False)
        self.label_status.setPixmap(QPixmap("./assets/icons/Invalidate.png"))
        self.camera_manager.open_camera()
        success = self.calibration.run()

    def start_cali(self):

        self.pushButton_closeCam.setEnabled(False)
        self.pushButton_StartCali.setEnabled(False)
        self.pushButton_startCam.setEnabled(True)

        self.camera_manager.close_camera()
        H_proj, H_inv_proj, H_graph, H_inv_graph = self.calibration.start_calib(self.label_status)

        print("Matrice H obtenue:", H_proj)
        print("Matrice H_inv obtenue:", H_inv_proj)
        self.calibration.timer.stop()
        self.parent.record_window.calib_data["H"] = H_proj
        self.parent.record_window.calib_data["H_inv"] = H_inv_proj
        self.parent.record_window.calib_data["H_graph"] = H_graph
        self.parent.record_window.calib_data["H_inv_graph"] = H_inv_graph
        self.parent.record_window.pushButton_Start.setEnabled(True)

    def go_to_record(self):
        self.close_camera()
        # Détruire les objets camera_manager et display_manager
        #del self.camera_manager
        #del self.display_manager
    
        self.parent.stacked_widget.setCurrentWidget(self.parent.record_window)  # Revenir au menu principal