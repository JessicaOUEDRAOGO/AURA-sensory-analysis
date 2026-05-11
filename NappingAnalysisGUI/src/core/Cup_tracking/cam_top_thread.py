# -*- coding: utf-8 -*-
"""
cam_top_thread.py
=================
Thread caméra du haut — KCF tracking PERMANENT.

Architecture IDENTIQUE à test_projection_blanc.py :
  - KCF tourne en permanence sur toutes les tasses visibles
  - Recalage HSV périodique (toutes les 150ms)
  - Conversion pixel → mm via PoseConverter + H_top_to_bottom
  - EMA sur les positions mm

Association ArUco ↔ KCF :
  - À l'initialisation : bbox HSV associée au tag ArUco le plus proche (mm)
  - cup_id interne remplacé par marker_id ArUco
  - IDs temporaires négatifs si pas encore associé

Source de vérité pour la projection :
  self.positions : {marker_id: (x_mm, y_mm)}
  Lu directement par ProjectionLoop — aucun intermédiaire.
"""

import cv2
import json
import numpy as np
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.utils.paths import config_path


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

CAPTURE_W          = 1920
CAPTURE_H          = 1080
PROCESS_W          = 640
PROCESS_H          = 360
PROC_TO_NATIVE     = CAPTURE_W / PROCESS_W   # 3.0
TABLE_SIZE_MM      = 597.0
BOUNDS_MARGIN_MM   = 30.0
CUP_HEIGHT_MM      = 95.0

DETECT_INTERVAL_S  = 0.15
MAX_LOST_FRAMES    = 20
MATCH_MIN_SCORE    = 0.01
MAX_DRIFT_RATIO    = 0.8
STILL_THRESHOLD_PX = 15
STABILITY_FRAMES   = 8

EMA_ALPHA          = 0.35
EMA_MAX_JUMP_MM    = 200.0

V_THRESHOLD        = 110
AREA_MIN           = 800
AREA_MAX           = 40_000
CIRC_MIN           = 0.15
ASPECT_MIN         = 0.25
ASPECT_MAX         = 3.5
MARGIN             = 20


# ══════════════════════════════════════════════════════════════════════════════
#  EMA Filter
# ══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA, max_jump=EMA_MAX_JUMP_MM):
        self._a = alpha; self._mj = max_jump; self._v = None

    def update(self, x, y):
        if self._v is None:
            self._v = (x, y); return self._v
        dx, dy = x-self._v[0], y-self._v[1]
        if (dx*dx+dy*dy)**.5 > self._mj:
            self._v = (x, y); return self._v
        self._v = (self._a*x+(1-self._a)*self._v[0],
                   self._a*y+(1-self._a)*self._v[1])
        return self._v

    def reset(self): self._v = None


# ══════════════════════════════════════════════════════════════════════════════
#  PoseConverter — chaîne exacte de test_projection_blanc
# ══════════════════════════════════════════════════════════════════════════════

class _PoseConverter:
    def __init__(self, pose_path: str):
        d = json.load(open(pose_path, "r", encoding="utf-8"))
        self.rvec = np.array(d["rvec"],          dtype=np.float64)
        self.tvec = np.array(d["tvec"],          dtype=np.float64)
        self.K    = np.array(d["camera_matrix"], dtype=np.float64)
        print(f"[CamTop] fx={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}")
        h_path = pose_path.replace("camtop_table_pose.json",
                                   "H_top_to_bottom.json")
        if not os.path.isfile(h_path):
            raise FileNotFoundError(f"H_top_to_bottom.json manquant → {h_path}")
        self._H     = np.array(json.load(open(h_path))["H_top_to_bottom"],
                               dtype=np.float32)
        self._H_inv = np.linalg.inv(self._H.astype(np.float64)).astype(np.float32)
        print("[CamTop] H_top_to_bottom OK")

    def pixel_to_mm(self, u, v):
        Ki = np.linalg.inv(self.K)
        r  = Ki @ np.array([u, v, 1.0]); r /= np.linalg.norm(r)
        R, _ = cv2.Rodrigues(self.rvec)
        n, o = R[:,2], self.tvec.reshape(3)
        d = np.dot(n, r)
        if abs(d) < 1e-9: return None
        t = np.dot(n, o) / d
        if t < 0: return None
        pt = R.T @ (r*t - o)
        p2 = cv2.perspectiveTransform(
            np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32),
            self._H)
        return float(p2[0,0,0]), float(p2[0,0,1])

    def mm_to_pixel_proc(self, x_mm, y_mm):
        pt = cv2.perspectiveTransform(
            np.array([[[x_mm, y_mm]]], dtype=np.float32), self._H_inv)
        x_top, y_top = float(pt[0,0,0]), float(pt[0,0,1])
        dz = np.zeros((5,1), dtype=np.float64)
        px, _ = cv2.projectPoints(
            np.array([[x_top, y_top, 0.0]]),
            self.rvec, self.tvec, self.K, dz)
        return float(px[0,0,0])/PROC_TO_NATIVE, float(px[0,0,1])/PROC_TO_NATIVE


def _cup_base_correction(cx, cy, rvec, tvec, K):
    dz = np.zeros((5,1), dtype=np.float64)
    Ki = np.linalg.inv(K)
    r  = Ki @ np.array([cx, cy, 1.0]); r /= np.linalg.norm(r)
    R, _ = cv2.Rodrigues(rvec)
    n, o = R[:,2], tvec.reshape(3)
    d = np.dot(n, r)
    if abs(d) < 1e-9: return cx, cy
    t = np.dot(n, o) / d
    if t < 0: return cx, cy
    pt = R.T @ (r*t - o)
    pm, _ = cv2.projectPoints(np.array([[pt[0], pt[1], CUP_HEIGHT_MM/2]]),
                               rvec, tvec, K, dz)
    pb, _ = cv2.projectPoints(np.array([[pt[0], pt[1], 0.0]]),
                               rvec, tvec, K, dz)
    return float(pb[0,0,0]), cy + (cy - float(pm[0,0,1]))


def _proc_to_mm(cx_p, cy_p, conv):
    cx_n, cy_n = cx_p*PROC_TO_NATIVE, cy_p*PROC_TO_NATIVE
    cx_n, cy_n = _cup_base_correction(cx_n, cy_n, conv.rvec, conv.tvec, conv.K)
    mm = conv.pixel_to_mm(cx_n, cy_n)
    if mm is None: return None
    lo, hi = -BOUNDS_MARGIN_MM, TABLE_SIZE_MM+BOUNDS_MARGIN_MM
    return mm if (lo <= mm[0] <= hi and lo <= mm[1] <= hi) else None


# ══════════════════════════════════════════════════════════════════════════════
#  Géométrie
# ══════════════════════════════════════════════════════════════════════════════

def _center(b): return b[0]+b[2]/2., b[1]+b[3]/2.

def _iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    ix=max(0,min(x1+w1,x2+w2)-max(x1,x2))
    iy=max(0,min(y1+h1,y2+h2)-max(y1,y2))
    inter=ix*iy; union=w1*h1+w2*h2-inter
    return inter/union if union>0 else 0.

def _score(bt, bd):
    iou=_iou(bt,bd); x,y,w,h=bt
    diag=max((w**2+h**2)**.5,1.)
    cx1,cy1=_center(bt); cx2,cy2=_center(bd)
    dn=((cx1-cx2)**2+(cy1-cy2)**2)**.5/diag
    return 0. if dn>2. else iou+.4*max(0.,1.-dn)

def _drift_ok(prev, new):
    x,y,w,h=prev; diag=max((w**2+h**2)**.5,1.)
    cx1,cy1=_center(prev); cx2,cy2=_center(new)
    return ((cx1-cx2)**2+(cy1-cy2)**2)**.5 < MAX_DRIFT_RATIO*diag


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedCup — identique à test_projection_blanc
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedCup:
    marker_id:   int      # marker_id ArUco (≥0) ou ID temporaire (<0)
    ema:         EMAFilter
    cv_tracker:  object
    bbox:        Tuple
    pos_mm:      Optional[Tuple] = None
    lost_frames: int  = 0
    active:      bool = True

    def update_kcf(self, frame):
        if not self.active: return False
        ok, raw = self.cv_tracker.update(frame)
        if ok:
            rx,ry,rw,rh=raw
            nb=(int(rx),int(ry),max(4,int(rw)),max(4,int(rh)))
            if _drift_ok(self.bbox, nb):
                self.bbox=nb; return True
        self.active=False; return False

    def reinit_kcf(self, frame, bbox):
        fh,fw=frame.shape[:2]
        x,y,w,h=bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t=cv2.TrackerKCF_create()
        try:
            t.init(frame,(x,y,w,h))
            self.cv_tracker=t; self.bbox=(x,y,w,h); self.active=True
            return True
        except Exception as e:
            print(f"[CamTop] reinit #{self.marker_id}: {e}")
            self.active=False; return False

    def update_mm(self, conv):
        cx,cy=_center(self.bbox)
        mm=_proc_to_mm(cx,cy,conv)
        if mm:
            xs,ys=self.ema.update(*mm)
            self.pos_mm=(xs,ys)


# ══════════════════════════════════════════════════════════════════════════════
#  Détecteur masque HSV — identique à test_projection_blanc
# ══════════════════════════════════════════════════════════════════════════════

class _Detector:
    def __init__(self):
        self.thr=V_THRESHOLD
        self._kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        self._ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    def detect(self, frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,m=cv2.threshold(gray,self.thr,255,cv2.THRESH_BINARY_INV)
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,self._kc,iterations=1)
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN, self._ko,iterations=1)
        m=self._rm(m)
        fh,fw=frame.shape[:2]; out=[]
        for cnt in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            a=cv2.contourArea(cnt)
            if not (AREA_MIN<=a<=AREA_MAX): continue
            hull=cv2.convexHull(cnt); ha=cv2.contourArea(hull); hp=cv2.arcLength(hull,True)
            if hp<1 or ha<1: continue
            if 4*np.pi*ha/hp**2<CIRC_MIN: continue
            x,y,w,h=cv2.boundingRect(cnt)
            if h>0 and not (ASPECT_MIN<=w/float(h)<=ASPECT_MAX): continue
            out.append((max(0,x-MARGIN),max(0,y-MARGIN),
                        min(fw,x+w+MARGIN)-max(0,x-MARGIN),
                        min(fh,y+h+MARGIN)-max(0,y-MARGIN)))
        return out

    @staticmethod
    def _rm(mask):
        h,w=mask.shape; f=mask.copy()
        for x in range(w):
            if f[0,x]:   cv2.floodFill(f,None,(x,0),0)
            if f[h-1,x]: cv2.floodFill(f,None,(x,h-1),0)
        for y in range(h):
            if f[y,0]:   cv2.floodFill(f,None,(0,y),0)
            if f[y,w-1]: cv2.floodFill(f,None,(w-1,y),0)
        return f


# ══════════════════════════════════════════════════════════════════════════════
#  TrackingManager — identique à test_projection_blanc
# ══════════════════════════════════════════════════════════════════════════════

class _Manager:
    def __init__(self, conv):
        self._conv    = conv
        self._det     = _Detector()
        self._cups:   Dict[int, TrackedCup] = {}
        self._pending: Dict[Tuple, int] = {}
        self._det_t   = 0.
        self._tmp_id  = -1

    def update(self, frame) -> List[TrackedCup]:
        """KCF update chaque frame — identique à test_projection_blanc."""
        for cup in list(self._cups.values()):
            ok=cup.update_kcf(frame)
            if ok:
                cup.update_mm(self._conv); cup.lost_frames=0
            else:
                cup.lost_frames+=1
        for mid in [m for m,c in self._cups.items()
                    if c.lost_frames>=MAX_LOST_FRAMES]:
            del self._cups[mid]
            print(f"[CamTop] KCF perdu #{mid} → supprimé")
        return list(self._cups.values())

    def recal(self, frame, aruco_pos: Dict[int, Tuple]) -> None:
        """Recalage HSV périodique — identique à test_projection_blanc."""
        now=time.monotonic()
        if now-self._det_t<DETECT_INTERVAL_S: return
        self._det_t=now
        detected=self._det.detect(frame)
        if not detected:
            self._pending.clear(); return

        cups=list(self._cups.values()); used_d=set(); used_c=set()
        if cups:
            S=np.zeros((len(cups),len(detected)),dtype=np.float32)
            for i,c in enumerate(cups):
                for j,d in enumerate(detected): S[i,j]=_score(c.bbox,d)
            for idx in np.argsort(-S.ravel()):
                i,j=divmod(int(idx),len(detected))
                if i in used_c or j in used_d: continue
                if S[i,j]<MATCH_MIN_SCORE: break
                cup=cups[i]; det=detected[j]
                cx1,cy1=_center(cup.bbox); cx2,cy2=_center(det)
                if ((cx1-cx2)**2+(cy1-cy2)**2)**.5<STILL_THRESHOLD_PX:
                    cup.reinit_kcf(frame,det)
                    cup.update_mm(self._conv)
                    cup.lost_frames=0
                used_c.add(i); used_d.add(j)

        # Nouvelles détections stables → nouveau track avec marker_id ArUco
        cur_keys=set()
        for j,det in enumerate(detected):
            if j in used_d: continue
            cx,cy=_center(det); key=(int(cx/20),int(cy/20))
            cur_keys.add(key)
            cnt=self._pending.get(key,0)+1; self._pending[key]=cnt
            if cnt>=STABILITY_FRAMES:
                mid=self._assign(det,aruco_pos)
                self._create(frame,det,mid)
                self._pending.pop(key,None)
        for k in list(self._pending):
            if k not in cur_keys: del self._pending[k]

    def force_reset(self, frame) -> None:
        """
        Identique à test_projection_blanc.force_reset() :
        Détecte toutes les bboxes HSV et crée un KCF par tasse
        avec des IDs temporaires négatifs.
        L'association ArUco viendra ensuite via assign_aruco_ids().
        """
        self._cups.clear()
        self._pending.clear()
        detected = self._det.detect(frame)
        print(f"[CamTop] force_reset: {len(detected)} bboxes HSV détectées")
        for det in detected:
            self._create(frame, det, self._tmp_id)
            self._tmp_id -= 1

    def assign_aruco_ids(self, aruco_pos: Dict[int, Tuple]) -> None:
        """
        Associe les marker_ids ArUco aux cups KCF existants par proximité mm.
        Appelé une fois que cam_bottom a détecté tous les tags.
        """
        for cup in list(self._cups.values()):
            if cup.marker_id >= 0:
                continue  # déjà associé
            if cup.pos_mm is None:
                continue

            best_mid  = None
            best_dist = float('inf')
            for mid, (ax, ay) in aruco_pos.items():
                # Éviter de réassigner un marker_id déjà utilisé
                if any(c.marker_id == mid for c in self._cups.values()):
                    continue
                d = ((cup.pos_mm[0]-ax)**2 + (cup.pos_mm[1]-ay)**2)**.5
                if d < best_dist:
                    best_dist = d
                    best_mid  = mid

            if best_mid is not None and best_dist < 100.0:
                old_id = cup.marker_id
                # Recréer avec le bon marker_id
                self._cups[best_mid] = TrackedCup(
                    marker_id=best_mid,
                    ema=cup.ema,
                    cv_tracker=cup.cv_tracker,
                    bbox=cup.bbox,
                    pos_mm=cup.pos_mm,
                    lost_frames=cup.lost_frames,
                    active=cup.active,
                )
                del self._cups[old_id]
                print(f"[CamTop] Association #{old_id} → ArUco #{best_mid} "
                    f"dist={best_dist:.0f}mm")

    def _assign(self, bbox, aruco_pos):
        cx,cy=_center(bbox); bi,bd=None,float('inf')
        for mid,(ax,ay) in aruco_pos.items():
            if mid in self._cups: continue
            rx,ry=self._conv.mm_to_pixel_proc(ax,ay)
            d=((cx-rx)**2+(cy-ry)**2)**.5
            if d<bd: bd=d; bi=mid
        if bi and bd<200.: return bi
        tid=self._tmp_id; self._tmp_id-=1; return tid

    def _create(self, frame, bbox, mid):
        fh,fw=frame.shape[:2]
        x,y,w,h=bbox
        x=max(0,min(x,fw-1)); y=max(0,min(y,fh-1))
        w=max(4,min(w,fw-x)); h=max(4,min(h,fh-y))
        t=cv2.TrackerKCF_create()
        try: t.init(frame,(x,y,w,h))
        except Exception as e:
            print(f"[CamTop] init KCF #{mid}: {e}"); return
        cup=TrackedCup(mid,EMAFilter(),t,(x,y,w,h))
        cup.update_mm(self._conv)
        self._cups[mid]=cup
        print(f"[CamTop] KCF créé #{mid} bbox=({x},{y},{w},{h})")

    def pos_all(self):
        return {m:c.pos_mm for m,c in self._cups.items() if c.pos_mm}

    


# ══════════════════════════════════════════════════════════════════════════════
#  CamTopThread
# ══════════════════════════════════════════════════════════════════════════════

class CamTopThread(QThread):
    """
    Thread caméra du haut — KCF tracking permanent.

    Interface :
      self.positions : {marker_id: (x_mm, y_mm)} — lu par ProjectionLoop
      self._pos_lock : threading.Lock pour accès thread-safe
    """

    fps_signal = pyqtSignal(float)
    pos_signal = pyqtSignal(int, float, float)

    def __init__(self, cam_bottom_thread,   # référence à CamBottomThread
                 camera_index, pose_path,
                 show_preview=False, parent=None):
        super().__init__(parent)
        self._cam_bottom  = cam_bottom_thread
        self.camera_index = camera_index
        self.show_preview = show_preview
        self.running      = False

        if not os.path.isfile(pose_path):
            raise FileNotFoundError(f"[CamTop] pose_path manquant → {pose_path}")
        self._conv    = _PoseConverter(pose_path)
        self._manager = _Manager(self._conv)

        # ── Positions exposées à ProjectionLoop ──────────────────────────────
        self.positions: Dict[int, Tuple[float, float]] = {}
        self._pos_lock = threading.Lock()

        # Undistort
        self._map1=self._map2=None
        try:
            c=json.load(open(config_path("camera_calibration_top.json")))
            Kr=np.array(c["camera_matrix"],dtype=np.float64)
            dist=np.array(c["dist_coeffs"],dtype=np.float64)
            nK,_=cv2.getOptimalNewCameraMatrix(Kr,dist,(CAPTURE_W,CAPTURE_H),1,(CAPTURE_W,CAPTURE_H))
            self._map1,self._map2=cv2.initUndistortRectifyMap(
                Kr,dist,None,nK,(CAPTURE_W,CAPTURE_H),cv2.CV_16SC2)
            print("[CamTop] Undistort OK")
        except FileNotFoundError:
            print("[CamTop] Pas de calibration — frames brutes")

        self._fps_t0=0.; self._fps_cnt=0
        self._inited=False

    def run(self):
        self.running=True; self._fps_t0=time.monotonic()
        print(f"[CamTop] démarré — cam={self.camera_index}")

        cap=cv2.VideoCapture(self.camera_index,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAPTURE_H)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if not cap.isOpened():
            print(f"[CamTop] ERREUR cam {self.camera_index}"); return

        while self.running:
            ret,frame=cap.read()
            if not ret or frame is None: continue

            if self._map1 is not None:
                frame=cv2.remap(frame,self._map1,self._map2,cv2.INTER_LINEAR)
            proc=cv2.resize(frame,(PROCESS_W,PROCESS_H),interpolation=cv2.INTER_LINEAR)

            # Positions ArUco depuis cam_bottom (lecture seule)
            aruco_pos=self._cam_bottom.get_aruco_positions()
            # Étape 1 : force_reset sur toutes les bboxes HSV (une seule fois)
            # identique à test_projection_blanc.force_reset()
            if not self._inited:
                    self._manager.force_reset(proc)
                    self._inited=True

                # Étape 2 : associer ArUco dès que des tags sont détectés
                # tenter à chaque frame jusqu'à ce que tous soient associés
            if aruco_pos:
                unassigned=[c for c in self._manager._cups.values()
                            if c.marker_id < 0]
            if unassigned:
                    self._manager.assign_aruco_ids(aruco_pos)

            # KCF update chaque frame — identique à test_projection_blanc
            cups=self._manager.update(proc)

            # Recalage HSV périodique — identique à test_projection_blanc
            self._manager.recal(proc, aruco_pos)

            # Publier positions dans self.positions
            # Filtrer les IDs temporaires négatifs
            with self._pos_lock:
                tracked_ids={cup.marker_id for cup in cups
                             if cup.pos_mm and cup.marker_id >= 0}
                for mid in list(self.positions):
                    if mid not in tracked_ids:
                        del self.positions[mid]
                for cup in cups:
                    if cup.pos_mm and cup.marker_id >= 0:
                        self.positions[cup.marker_id]=cup.pos_mm
                        self.pos_signal.emit(cup.marker_id,
                                             cup.pos_mm[0],cup.pos_mm[1])

            if self.show_preview:
                self._preview(proc,cups,aruco_pos)

            # FPS + debug
            self._fps_cnt+=1
            now=time.monotonic()
            if now-self._fps_t0>=1.0:
                fps=self._fps_cnt/(now-self._fps_t0)
                self.fps_signal.emit(fps)
                self._fps_cnt=0; self._fps_t0=now
                top=self._manager.pos_all()
                print(f"[CamTop] FPS={fps:.1f}  tracks={len(cups)}")
                for mid in sorted(set(top)|set(aruco_pos)):
                    tp=top.get(mid)
                    ap=aruco_pos.get(mid)
                    ts=f"({tp[0]:.0f},{tp[1]:.0f})mm" if tp else "N/A"
                    bs=f"({ap[0]:.0f},{ap[1]:.0f})mm" if ap else "N/A"
                    print(f"  #{mid}  cam_top={ts}  cam_bottom={bs}")

        cap.release()
        if self.show_preview: cv2.destroyWindow("CamTop")
        print("[CamTop] arrêté")

    def stop(self): self.running=False

    def _preview(self, proc, cups, aruco_pos):
        p=cv2.resize(proc,(960,540)); ps=960/PROCESS_W
        for cup in cups:
            bx,by,bw,bh=cup.bbox
            col=(0,180,255) if cup.lost_frames>0 else (0,255,0)
            cv2.rectangle(p,(int(bx*ps),int(by*ps)),
                          (int((bx+bw)*ps),int((by+bh)*ps)),col,2)
            cx,cy=_center(cup.bbox)
            cv2.drawMarker(p,(int(cx*ps),int(cy*ps)),
                           (0,0,255),cv2.MARKER_CROSS,14,2)
            if cup.pos_mm:
                cv2.putText(p,f"#{cup.marker_id} "
                            f"({cup.pos_mm[0]:.0f},{cup.pos_mm[1]:.0f})mm",
                            (int(bx*ps),max(20,int(by*ps)-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
        # Afficher positions ArUco
        for i,(mid,(ax,ay)) in enumerate(aruco_pos.items()):
            cv2.putText(p,f"ArUco #{mid}: ({ax:.0f},{ay:.0f})mm",
                        (10,20+18*i),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,200,0),1)
        cv2.imshow("CamTop",p); cv2.waitKey(1)