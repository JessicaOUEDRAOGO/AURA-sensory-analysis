import numpy as np
import cv2

status_cam = True
status_proj = False

# Initialisation de la capture vidéo pour la caméra
cap = cv2.VideoCapture(0)

# Paramètres de la mire (damier)
pattern_size_proj = (9, 7)  # Nombre de carrés (14x7), donc intersections (13x6)
square_size_proj = 135  # Taille des carrés en pixels# Paramètres de la mire (damier)

pattern_size_cam = (6, 9)  # Nombre de carrés (14x7), donc intersections (13x6)
square_size_cam = 40  # Taille des carrés en mm
 
# Points pour la calibration
objpoints_cam = []  # Points 3D réels
objpoints_proj = []  # Points 3D réels
imgpoints_cam = []  # Points 2D détectés par la caméra
imgpoints_proj = []  # Points 2D détectés par le projecteur

# Modèle de la mire 3D
objp_proj = np.zeros(((pattern_size_proj[0] - 1) * (pattern_size_proj[1] - 1), 3), dtype=np.float32)
objp_proj[:, :2] = np.indices((pattern_size_proj[0] - 1, pattern_size_proj[1] - 1)).T.reshape(-1, 2) * square_size_proj

objp_cam = np.zeros(((pattern_size_cam[0] - 1) * (pattern_size_cam[1] - 1), 3), dtype=np.float32)
objp_cam[:, :2] = np.indices((pattern_size_cam[0] - 1, pattern_size_cam[1] - 1)).T.reshape(-1, 2) * square_size_cam

# Création et affichage de la mire projetée
#checkerboard = np.zeros((pattern_size[1] * square_size, pattern_size[0] * square_size), dtype=np.uint8)
checkerboard = np.ones((pattern_size_proj[1] * square_size_proj, pattern_size_proj[0] * square_size_proj), dtype=np.uint8) * 255
for i in range(pattern_size_proj[0]):
    for j in range(pattern_size_proj[1]):
        if (i + j) % 2 == 0:
            cv2.rectangle(checkerboard, (i * square_size_proj, j * square_size_proj), 
                          ((i + 1) * square_size_proj - 1, (j + 1) * square_size_proj - 1), 0, -1)


cv2.imshow("Checkerboard", checkerboard)
cv2.waitKey(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    if status_cam == True :
        found, corners = cv2.findChessboardCorners(gray, (pattern_size_cam[0] - 1, pattern_size_cam[1] - 1), None)
        if found:
            cv2.drawChessboardCorners(frame, (pattern_size_cam[0] - 1, pattern_size_cam[1] - 1), corners, found)
    
    if status_proj == True :
        found, corners = cv2.findChessboardCorners(gray, (pattern_size_proj[0] - 1, pattern_size_proj[1] - 1), None)
        if found:
            cv2.drawChessboardCorners(frame, (pattern_size_proj[0] - 1, pattern_size_proj[1] - 1), corners, found)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and found:       
        if status_cam == True :
            imgpoints_cam.append(corners)
            objpoints_cam.append(objp_cam)
            print(f"Image enregistrée pour la caméra ! Total : {len(objpoints_cam)}")
            status_cam = False
            status_proj = True

        if status_proj == True :
            imgpoints_proj.append(corners)
            objpoints_proj.append(objp_proj)
            print(f"Image enregistrée pour la projecteur ! Total : {len(objpoints_proj)}")
    
    if key == ord('q'):
        break
    
    cv2.imshow('Capture_Video', frame)

# Calibration de la caméra
if objpoints_cam and imgpoints_cam:
    ret, K_c, dist_c, rvecs, tvecs = cv2.calibrateCamera(objpoints_cam, imgpoints_cam, gray.shape[::-1], None, None)
    print("=== Calibration de la caméra ===")
    print("Matrice de la caméra :\n", K_c)
    print("Coefficients de distorsion :\n", dist_c)

# Calibration du projecteur
if objpoints_proj and imgpoints_proj:
    ret, K_p, dist_p, rvecs_p, tvecs_p = cv2.calibrateCamera(objpoints_proj, imgpoints_proj, gray.shape[::-1], None, None)
    print("=== Calibration du projecteur ===")
    print("Matrice du projecteur :\n", K_p)
    print("Coefficients de distorsion :\n", dist_p)

    # Stéréo calibration pour obtenir la transformation entre caméra et projecteur
    if len(objpoints_cam) > 0 and len(objpoints_proj) > 0:
        ret, K_c, dist_c, K_p, dist_p, R, T, E, F = cv2.stereoCalibrate(
            objpoints_cam, imgpoints_cam, imgpoints_proj, 
            K_c, dist_c, K_p, dist_p, gray.shape[::-1], 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        print("=== Transformation caméra -> projecteur ===")
        print("Matrice de rotation (R) :\n", R)
        print("Vecteur de translation (T) :\n", T)

cap.release()
cv2.destroyAllWindows()