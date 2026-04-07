# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time
import cv2
import numpy as np
from screeninfo import get_monitors

# =========================================================
# PARAMETRES
# =========================================================
PROJECTOR_WIDTH = 3840
PROJECTOR_HEIGHT = 2160

CAMERA_ID = 0
GRID_SIZE = 700

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

CAMERA_CALIB_PATH = CONFIG_DIR / "camera_calibration.json"
PROJECTOR_CALIB_PATH = CONFIG_DIR / "projector_calibration_moreno_refined.json"
STEREO_CALIB_PATH = CONFIG_DIR / "stereo_camera_projector_calibration.json"

BOARD_IMAGE_PATH = PROJECT_ROOT / "aruco_a3.jpg"

CORNER_TAG_IDS = {"TL":42,"TR":43,"BR":40,"BL":41}

# =========================================================
# LOAD CALIB
# =========================================================
def load_json(p):
    with open(p,"r") as f:
        return json.load(f)

def load_calib():
    cam = load_json(CAMERA_CALIB_PATH)
    proj = load_json(PROJECTOR_CALIB_PATH)
    st = load_json(STEREO_CALIB_PATH)

    K_cam = np.array(cam["camera_matrix"])
    dist_cam = np.array(cam["dist_coeffs"])

    K_proj = np.array(proj["projector_matrix"])
    dist_proj = np.array(proj["projector_dist_coeffs"])

    R_cp = np.array(st["R"])
    T_cp = np.array(st["T"]).reshape(3,1)

    return K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp

# =========================================================
# CAMERA
# =========================================================
def open_cam():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(3,1920)
    cap.set(4,1080)
    return cap

# =========================================================
# PROJECTOR
# =========================================================
class Proj:
    def __init__(self):
        m = get_monitors()[1]
        self.name="proj"
        cv2.namedWindow(self.name,cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name,m.x,m.y)
        cv2.setWindowProperty(self.name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    def show(self,img):
        cv2.imshow(self.name,img)
        cv2.waitKey(1)

# =========================================================
# ARUCO
# =========================================================
def detector():
    d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p=cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d,p)

def detect(gray,det):
    c,ids,_=det.detectMarkers(gray)
    if ids is None:
        return []
    ids=ids.flatten()
    out=[]
    for i,id_ in enumerate(ids):
        pts=c[i][0]
        out.append({"id":int(id_), "corners":pts, "center":np.mean(pts,axis=0)})
    return out

# =========================================================
# UTILS
# =========================================================
def undist(pt,K,dist):
    return cv2.undistortPoints(pt.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)

def select_corner(c,pos):
    if pos=="TL":return c[np.argmin(c[:,0]+c[:,1])]
    if pos=="TR":return c[np.argmin(-c[:,0]+c[:,1])]
    if pos=="BR":return c[np.argmax(c[:,0]+c[:,1])]
    if pos=="BL":return c[np.argmax(-c[:,0]+c[:,1])]

# =========================================================
# MAIN
# =========================================================
def main():

    K_cam, dist_cam, K_proj, dist_proj, R_cp, T_cp = load_calib()
    det = detector()
    cap = open_cam()
    proj = Proj()

    # =============================
    # IMAGE
    # =============================
    img=cv2.imread(str(BOARD_IMAGE_PATH))
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img=cv2.threshold(g,127,255,cv2.THRESH_BINARY)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    h,w=img.shape[:2]
    scale=0.6
    img=cv2.resize(img,(int(w*scale),int(h*scale)))

    # =============================
    # FOND BLANC + IMAGE
    # =============================
    canvas=np.full((PROJECTOR_HEIGHT,PROJECTOR_WIDTH,3),255,np.uint8)

    ih,iw=img.shape[:2]
    x0=(PROJECTOR_WIDTH-iw)//2
    y0=(PROJECTOR_HEIGHT-ih)//2
    canvas[y0:y0+ih,x0:x0+iw]=img

    proj.show(canvas)
    time.sleep(1)

    print("Detection...")

    while True:
        ret,frame=cap.read()
        if not ret: continue

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detections=detect(gray,det)

        # =============================
        # TAGS PHYSIQUES
        # =============================
        phys={}
        for id_ in CORNER_TAG_IDS.values():
            cand=[d for d in detections if d["id"]==id_]
            if not cand: continue
            phys[id_]=cand[0]

        if len(phys)!=4:
            cv2.imshow("cam",frame)
            if cv2.waitKey(1)==27:break
            continue

        cam_pts=np.array([
            select_corner(phys[42]["corners"],"TL"),
            select_corner(phys[43]["corners"],"TR"),
            select_corner(phys[40]["corners"],"BR"),
            select_corner(phys[41]["corners"],"BL")
        ],dtype=np.float32)

        cam_pts_und=undist(cam_pts,K_cam,dist_cam)

        # =============================
        # REPERE TABLE
        # =============================
        table_pts=np.array([
            [0,0],
            [GRID_SIZE,0],
            [GRID_SIZE,GRID_SIZE],
            [0,GRID_SIZE]
        ],dtype=np.float32)

        H_cam_to_table,_=cv2.findHomography(cam_pts_und,table_pts)

        # =============================
        # POSE 3D
        # =============================
        obj=np.array([
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [0,1,0]
        ],dtype=np.float32)

        ok,rvec,tvec=cv2.solvePnP(obj,cam_pts,K_cam,dist_cam)
        if not ok: continue

        R,_=cv2.Rodrigues(rvec)

        R_tp=R_cp @ R
        T_tp=R_cp @ tvec + T_cp

        H_table_proj = K_proj @ np.column_stack((R_tp[:,0],R_tp[:,1],T_tp.flatten()))

        H_cam_proj = H_table_proj @ np.linalg.inv(
            K_cam @ np.column_stack((R[:,0],R[:,1],tvec.flatten()))
        )

        # =============================
        # IMAGE -> PROJECTEUR
        # =============================
        src=np.array([
            [0,0],
            [iw,0],
            [iw,ih],
            [0,ih]
        ],dtype=np.float32)

        H_img_cam,_=cv2.findHomography(src,cam_pts_und)

        H_img_proj = H_cam_proj @ H_img_cam

        warp=cv2.warpPerspective(img,H_img_proj,(PROJECTOR_WIDTH,PROJECTOR_HEIGHT))

        out=np.full((PROJECTOR_HEIGHT,PROJECTOR_WIDTH,3),255,np.uint8)
        mask=np.any(warp<250,axis=2)
        out[mask]=warp[mask]

        proj.show(out)

        cv2.imshow("cam",frame)

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()