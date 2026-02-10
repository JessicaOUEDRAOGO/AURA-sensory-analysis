import cv2
import numpy as np
from screeninfo import get_monitors
from pyzbar.pyzbar import decode

# **Ouverture de la webcam**
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Largeur de la vidéo
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Hauteur de la vidéo

# Charger l'image
image_background = cv2.imread("blanc.png")

# Récupérer les informations des écrans
monitors = get_monitors()
monitor = monitors[2]  # Écran numéro 3 (indice 2)

# **Affichage de l'image sur l'écran 3**
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image", monitor.x, monitor.y)  # Déplacer l'image sur l'écran 3
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Image", image_background)  # Afficher l'image sur l'écran 3


# **Affichage de la vidéo sur l'écran principal (écran 1)**
cv2.namedWindow("Capture Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Capture Video", 1920, 1080)  # Ajuster la taille de la fenêtre vidéo

while True:
    ret, frame = cap.read()

    cv2.imshow("Capture Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        Save_Frame = frame.copy()
        break
    elif key == ord('q'):  # Quitter en appuyant sur 'q'
        break

cap.release()
cv2.destroyAllWindows()






gray = cv2.cvtColor(Save_Frame, cv2.COLOR_BGR2GRAY)

# Seuillage pour isoler le carré blanc
_, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

print("--- Seuillage ---")
cv2.imshow("thresh", thresh)
cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
# Détection des contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Recherche du plus grand quadrilatère
quadrilateral = None
max_area = 0

for cnt in contours:
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:  # Vérifier si c'est un quadrilatère
        area = cv2.contourArea(approx)
        if area > max_area:
            max_area = area
            quadrilateral = approx


proj_points = quadrilateral.reshape(4, 2)  # Obtenir les coordonnées des coins
print("Coins du quadrilatère:\n", proj_points)

# Dessiner les coins
for x, y in proj_points:
    cv2.circle(Save_Frame, (x, y), 10, (0, 0, 255), -1)

# Relier les coins par des lignes
for i in range(4):
    pt1 = tuple(proj_points[i])
    pt2 = tuple(proj_points[(i + 1) % 4])  # Boucle circulaire pour relier tous les points
    cv2.line(Save_Frame, pt1, pt2, (0, 255, 0), 3)

# Afficher l'image
print("--- Quadrilatère détécté ---")
cv2.imshow("Quadrilatère détecté", Save_Frame)
cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
#cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
#cv2.destroyAllWindows()

# Récupérer les dimensions de l'image
height, width = Save_Frame.shape[:2]

cam_points = []
for i in range(4):
    points = []
    if proj_points[i][0] < (proj_points[i][0]+proj_points[1][0]+proj_points[2][0]+proj_points[3][0]) / 4:
        points.append(0)
    else:
        points.append(1920)

    if proj_points[i][1] < (proj_points[i][1]+proj_points[1][1]+proj_points[2][1]+proj_points[3][1]) / 4:
        points.append(0)
    else:
        points.append(1080)
    cam_points.append(points)
cam_points = np.array(cam_points)

print("cam_points (NumPy) :")
print(cam_points)


# Calcul de l'homographie entre caméra et rétroprojecteur
H, _ = cv2.findHomography(proj_points, cam_points)
H_inv = np.linalg.inv(H)  # Matrice inverse pour conversion inverse

# 2. Détection des QR codes dans l'image de la caméra
def detect_qr_codes(frame):
    qr_codes = decode(frame)  # Détection des QR codes
    print()
    if qr_codes:
        qr_data = qr_codes[0].data.decode("utf-8")  # Extraire les données du premier QR code
        x, y, w, h = qr_codes[0].rect  # Extraire les coordonnées sous forme de tuple
        center = (x + w // 2, y + h // 2)  # Calcul du centre
        return np.array(center)  # Retourne le centre du QR Code en numpy array
    return None

# 3. Transformation des coordonnées d'un point caméra vers le rétroprojecteur
def transform_to_projector(camera_point):
    camera_point = np.array([camera_point[0], camera_point[1], 1], dtype=np.float32)
    projected_point = np.dot(H, camera_point)
    projected_point /= projected_point[2]  # Normalisation
    return projected_point[:2]

# 4. Transformation inverse du projecteur vers la caméra
def transform_to_camera(projector_point):
    projector_point = np.array([projector_point[0], projector_point[1], 1], dtype=np.float32)
    camera_point = np.dot(H_inv, projector_point)
    camera_point /= camera_point[2]  # Normalisation
    return camera_point[:2]

# 5. Test avec une image de la caméra
cv2.circle(image_background, tuple(np.round(transform_to_projector(proj_points[0])).astype(int)), 50, (0, 255, 0), 3)
cv2.circle(image_background, tuple(np.round(transform_to_projector(proj_points[1])).astype(int)), 50, (0, 255, 0), 3)
cv2.circle(image_background, tuple(np.round(transform_to_projector(proj_points[2])).astype(int)), 50, (0, 255, 0), 3)
cv2.circle(image_background, tuple(np.round(transform_to_projector(proj_points[3])).astype(int)), 50, (0, 255, 0), 3)
#cv2.circle(image_background, (0,0), 50, (0, 255, 0), 3)  # Cercle rouge
#cv2.circle(image_background, (1920,0), 50, (0, 255, 0), 3)  # Cercle rouge
#cv2.circle(image_background, (0,1080), 50, (0, 255, 0), 3)  # Cercle rouge
#cv2.circle(image_background, (1920,1080), 50, (0, 255, 0), 3)  # Cercle rouge
# Mettre à jour l'affichage sur l'écran 3 en plein écran
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image", monitor.x, monitor.y)  # Déplacer l'image sur l'écran 3
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Image", image_background)



print("--- Allumage de la la caméra ---")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Largeur de la vidéo
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Hauteur de la vidéo

while True:
    ret, frame = cap.read()

    cv2.circle(frame, (400,400), 100, (255, 0, 0), 7)  # Cercle rouge

    qr_point = detect_qr_codes(frame)  # Détection du QR Code


    if qr_point is not None:

        print(qr_point)
        # Appliquer la transformation directement au point unique
        projector_coords = transform_to_projector(qr_point)
        print(f"QR Code position caméra: {qr_point} -> Position projetée: {projector_coords}")


        # Copie de l'image de fond pour éviter de dessiner plusieurs cercles
        display_image = image_background.copy()
        # Dessiner un cercle rouge autour du QR Code sur l'image de fond
        cv2.circle(display_image, tuple(projector_coords.astype(int)), 50, (0, 0, 255), 3)  # Cercle rouge
        # Afficher les coordonnées projetées sur l'image
        cv2.putText(display_image, f"{int(projector_coords[0])}, {int(projector_coords[1])}",
                    tuple(projector_coords.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        # Convertir les coordonnées projetées en entiers
        proj_x, proj_y = projector_coords.astype(int)
        # Dessiner un cercle rouge sur la frame (au lieu de `image_background`)
        cv2.circle(frame, (proj_x, proj_y), 50, (0, 0, 255), 3)  # Cercle rouge
        # Afficher les coordonnées projetées sur la frame
        cv2.putText(frame, f"{proj_x}, {proj_y}", (proj_x, proj_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        # Mettre à jour l'affichage sur l'écran 3 en plein écran
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Image", monitor.x, monitor.y)  # Déplacer l'image sur l'écran 3
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", display_image)

    # Affichage de la capture vidéo# **Affichage de la vidéo sur l'écran principal (écran 1)**
    cv2.namedWindow("Capture Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Video", 1920, 1080)  # Ajuster la taille de la fenêtre vidéo
    cv2.imshow("Capture Video", frame)

    # Quitter avec 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
