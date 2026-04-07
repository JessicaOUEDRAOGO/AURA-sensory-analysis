# scan_cameras.py
import cv2

def list_cameras(max_tested=10):
    print("Scanning cameras...\n")
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera détectée index = {i}")
            available.append(i)
            cap.release()
        else:
            print(f"❌ index {i} indisponible")
    return available

if __name__ == "__main__":
    cams = list_cameras()
    print("\nCaméras disponibles :", cams)


# test_dual_cam_dshow.py
import cv2

CAM1 = 0
CAM2 = 1

cap1 = cv2.VideoCapture(CAM1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(CAM2, cv2.CAP_DSHOW)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

print("Cam1 opened:", cap1.isOpened())
print("Cam2 opened:", cap2.isOpened())

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        cv2.imshow("Camera 1", f1)
    if ret2:
        cv2.imshow("Camera 2", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# test_dual_cam_dshow.py
import cv2

CAM1 = 0
CAM2 = 1

cap1 = cv2.VideoCapture(CAM1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(CAM2, cv2.CAP_DSHOW)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

print("Cam1 opened:", cap1.isOpened())
print("Cam2 opened:", cap2.isOpened())

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        cv2.imshow("Camera 1", f1)
    if ret2:
        cv2.imshow("Camera 2", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# test_dual_cam_default.py
import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

print("Cam1 opened:", cap1.isOpened())
print("Cam2 opened:", cap2.isOpened())

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        cv2.imshow("Camera 1", f1)
    if ret2:
        cv2.imshow("Camera 2", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()


# test_sequential.py
import cv2

print("Test caméra 0")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Opened:", cap.isOpened())

ret, frame = cap.read()
if ret:
    cv2.imshow("Cam0", frame)
    cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

print("\nTest caméra 1")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("Opened:", cap.isOpened())

ret, frame = cap.read()
if ret:
    cv2.imshow("Cam1", frame)
    cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

# test_low_res.py
import cv2

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

print("Cam1:", cap1.isOpened())
print("Cam2:", cap2.isOpened())

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if r1:
        cv2.imshow("cam1_low", f1)
    if r2:
        cv2.imshow("cam2_low", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# test_delay.py
import cv2
import time

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)

cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(1)

print("Cam1:", cap1.isOpened())
print("Cam2:", cap2.isOpened())

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if r1:
        cv2.imshow("cam1", f1)
    if r2:
        cv2.imshow("cam2", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()