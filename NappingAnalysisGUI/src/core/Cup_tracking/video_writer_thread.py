# -*- coding: utf-8 -*-
"""
video_writer_thread.py
======================
Thread d'écriture vidéo asynchrone.

La boucle principale (CupTrackingPipeline) dépose des frames dans la queue
sans attendre — l'encodage H264 se fait dans ce thread séparé, sans impacter
le pipeline KCF/HSV.

Paramètres :
  output_path  : chemin complet du fichier .mp4
  width        : largeur cible (défaut 960)
  height       : hauteur cible (défaut 540)
  fps          : fréquence cible (défaut 30.0)
  queue_maxsize: taille max de la queue (défaut 64 frames — ~2s de buffer)

Usage :
    writer = VideoWriterThread(output_path="...", width=960, height=540, fps=30.0)
    writer.start()

    # dans la boucle principale — non bloquant
    writer.push_frame(frame_bgr)   # frame quelconque, redimensionnée ici

    # à l'arrêt
    writer.stop()   # vide la queue puis ferme le fichier
"""

import queue
import threading
import cv2
import numpy as np


class VideoWriterThread(threading.Thread):
    # Sentinel pour signaler la fin de la queue
    _STOP_SENTINEL = None

    def __init__(
        self,
        output_path: str,
        width: int  = 960,
        height: int = 540,
        fps: float  = 30.0,
        queue_maxsize: int = 64,
    ):
        super().__init__(name="VideoWriterThread", daemon=True)
        self.output_path   = output_path
        self.width         = int(width)
        self.height        = int(height)
        self.fps           = float(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0
        self._dropped_frames = 0

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> bool:
        """
        Dépose une frame dans la queue (non bloquant).
        Retourne False si la queue est pleine (frame droppée).
        La frame est redimensionnée dans le thread d'écriture,
        pas ici — on minimise le travail dans la boucle principale.
        """
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            self._dropped_frames += 1
            return False

    def stop(self) -> None:
        """
        Envoie le sentinel, puis attend la fin du thread (max 10s).
        Garantit que toutes les frames en queue sont écrites avant fermeture.
        """
        self._queue.put(self._STOP_SENTINEL)   # bloquant si queue pleine — voulu
        self.join(timeout=10.0)
        if self.is_alive():
            print("[VideoWriter] TIMEOUT — le thread n'a pas pu se terminer proprement")
        else:
            print(
                f"[VideoWriter] Fermé — {self._frames_written} frames écrites, "
                f"{self._dropped_frames} droppées → {self.output_path}"
            )

    # ------------------------------------------------------------------
    # Thread interne
    # ------------------------------------------------------------------

    def run(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not self._writer.isOpened():
            print(f"[VideoWriter] ERREUR — impossible d'ouvrir : {self.output_path}")
            # Vider la queue proprement quand même
            while True:
                item = self._queue.get()
                if item is self._STOP_SENTINEL:
                    break
            return

        print(f"[VideoWriter] Démarré → {self.output_path}  ({self.width}×{self.height} @ {self.fps:.0f}fps)")

        while True:
            try:
                frame = self._queue.get(timeout=2.0)
            except queue.Empty:
                # Pas de sentinel reçu et queue vide depuis 2s — on continue d'attendre
                continue

            if frame is self._STOP_SENTINEL:
                break

            self._write(frame)

        self._writer.release()

    def _write(self, frame: np.ndarray) -> None:
        try:
            h, w = frame.shape[:2]
            if (w, h) != (self.width, self.height):
                frame = cv2.resize(
                    frame, (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR
                )
            self._writer.write(frame)
            self._frames_written += 1
        except Exception as e:
            print(f"[VideoWriter] Erreur écriture frame : {e}")
