import cv2
import os

save_dir = "img_checker"
os.makedirs(save_dir, exist_ok=True)

index = 0

# DirectShow = souvent plus stable et plus rapide sous Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la caméra")

# Réglages pour éviter la lenteur
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Ne pas démarrer directement en 4K pour l'aperçu
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Camera ouverte")
print("i = capturer | q = quitter")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Erreur lecture caméra")
        break

    # aperçu léger
    preview = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    cv2.putText(
        preview,
        "i = capture | q = quit",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("capture_checker", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("i"):
        filename = os.path.join(save_dir, f"im_{index:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image sauvegardee : {filename}")
        index += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()