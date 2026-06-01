# -*- coding: utf-8 -*-
"""
plot_trajectory.py  –  v5
=========================
Visualisation multi-tasses / multi-sources — tout dans l'interface.

Nouveautés v5 :
  • Détection des épisodes de pose via la source "bottom"
  • Affichage : surbrillance du segment posé + marqueur central par épisode
  • Checkbox "Poses" dans la barre de contrôles

Usage :
    python plot_trajectory.py chemin/vers/fichier.csv
    python plot_trajectory.py fichier.csv --output img.png
    python plot_trajectory.py fichier.csv --downsample 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtWidgets import (
    QApplication, QMainWindow,
    QLabel, QComboBox, QCheckBox,
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ──────────────────────────────────────────────────────────────────────────────
#  Couleurs & constantes
# ──────────────────────────────────────────────────────────────────────────────

CUP_COLORS = [
    "#E63946", "#4A9FD4", "#2ECC9A", "#F0C040",
    "#F4A261", "#A855F7", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E", "#FFD166", "#8AC926",
    "#9B5DE5", "#1982C4", "#FF595E", "#6A994E",
]

SOURCES   = ["ema", "raw", "filtered", "bottom"]
BG_DARK   = "#0D1B2A"
BG_PANEL  = "#112240"
BG_WIDGET = "#1A3A5C"
ACCENT    = "#2E6DB4"
TEXT_MAIN = "#E8EEF4"
TEXT_DIM  = "#7A9ABF"
SEP_COLOR = "#1E3A5C"

# Couleur et style des marqueurs de pose
POSE_SEGMENT_ALPHA = 0.55   # transparence du segment surligné
POSE_MARKER_COLOR  = "#FFFFFF"
POSE_MARKER_EDGE   = "#000000"
POSE_MARKER_SIZE   = 120
POSE_GAP_TOLERANCE = 5      # frames de gap max dans un même épisode


def cup_color(cup_id: str) -> str:
    try:
        idx = int(cup_id)
    except ValueError:
        idx = abs(hash(cup_id))
    return CUP_COLORS[idx % len(CUP_COLORS)]


# ──────────────────────────────────────────────────────────────────────────────
#  Styles Qt
# ──────────────────────────────────────────────────────────────────────────────

COMBO_STYLE = f"""
    QComboBox {{
        background-color: {BG_WIDGET}; color: {TEXT_MAIN};
        border: 1px solid {ACCENT}; border-radius: 5px;
        padding: 4px 10px; font-size: 12px; min-width: 105px;
    }}
    QComboBox:hover {{ border-color: #4A8FD4; }}
    QComboBox::drop-down {{ border: none; width: 16px; }}
    QComboBox::down-arrow {{
        width: 0; height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {TEXT_DIM};
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_WIDGET}; color: {TEXT_MAIN};
        border: 1px solid {ACCENT};
        selection-background-color: {ACCENT}; selection-color: white;
        outline: none;
    }}
"""

def label_style(size=11, dim=False):
    color = TEXT_DIM if dim else TEXT_MAIN
    return f"color: {color}; font-size: {size}px; background: transparent;"

def sep_style():
    return f"QFrame {{ background-color: {SEP_COLOR}; border: none; }}"

def checkbox_style(color: str) -> str:
    return f"""
        QCheckBox {{
            color: {TEXT_MAIN}; font-size: 12px; font-weight: 600;
            spacing: 5px; background: transparent;
        }}
        QCheckBox::indicator {{
            width: 15px; height: 15px; border-radius: 3px;
        }}
        QCheckBox::indicator:unchecked {{
            background: #1A3A5C; border: 2px solid #5A8ABF;
        }}
        QCheckBox::indicator:unchecked:hover {{
            background: #1F4570; border: 2px solid {color};
        }}
        QCheckBox::indicator:checked {{
            background: {color}; border: 2px solid {color}; image: none;
        }}
    """

def vline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFixedWidth(1)
    line.setStyleSheet(sep_style())
    return line


# ──────────────────────────────────────────────────────────────────────────────
#  Chargement CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df  = pd.read_csv(p, sep=sep)
    print(f"[CSV] {len(df)} lignes — {len(df.columns)} colonnes")
    return df


def extract_cups(df: pd.DataFrame) -> dict:
    """
    Retourne cups[cup_id][source] = DataFrame(frame, x, y).
    Conserve aussi les frames brutes du bottom pour la détection des poses.
    """
    cups: dict[str, dict] = {}
    for source in SOURCES:
        for col_x in [c for c in df.columns
                      if c.endswith(f"_x_{source}") and c.startswith("ID_")]:
            parts  = col_x.split("_")
            cup_id = parts[1]
            col_y  = f"ID_{cup_id}_y_{source}"
            if col_y not in df.columns:
                continue
            sub = df[["frame", col_x, col_y]].copy()
            sub.columns = ["frame", "x", "y"]
            sub = sub.dropna().reset_index(drop=True)
            cups.setdefault(cup_id, {})[source] = sub

    if not cups:
        print("[ERREUR] Aucune colonne ID_N_x_<source> trouvée.")
        sys.exit(1)

    # Stocker les frames valides de bottom (avant dropna) pour détection de pose
    for cup_id in cups:
        col_x_b = f"ID_{cup_id}_x_bottom"
        col_y_b = f"ID_{cup_id}_y_bottom"
        if col_x_b in df.columns and col_y_b in df.columns:
            b = df[["frame", col_x_b, col_y_b]].copy()
            b.columns = ["frame", "x", "y"]
            # garder TOUTES les lignes (NaN inclus) pour avoir l'index des frames
            cups[cup_id]["_bottom_full"] = b
        else:
            cups[cup_id]["_bottom_full"] = pd.DataFrame(columns=["frame", "x", "y"])

    for cid, srcs in sorted(cups.items(),
                             key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        pts = {s: len(v) for s, v in srcs.items() if not s.startswith("_")}
        print(f"  Tasse {cid:>3} | " + " | ".join(f"{s}:{n}" for s, n in pts.items()))
    return cups


# ──────────────────────────────────────────────────────────────────────────────
#  Détection des épisodes de pose
# ──────────────────────────────────────────────────────────────────────────────

def detect_pose_episodes(cups: dict, cup_id: str,
                          gap_tolerance: int = POSE_GAP_TOLERANCE) -> list[dict]:
    """
    Retourne une liste d'épisodes :
        [{ "frames": [f1, f2, …], "cx": float, "cy": float }, …]
    cx/cy = centroïde bottom de l'épisode.
    """
    b = cups[cup_id].get("_bottom_full", pd.DataFrame())
    if b.empty:
        return []

    # Frames où bottom est valide
    valid = b.dropna(subset=["x", "y"]).copy()
    if valid.empty:
        return []

    frames = sorted(valid["frame"].astype(int).tolist())

    # Regrouper en épisodes (gap ≤ tolerance)
    episodes = []
    current  = [frames[0]]
    for f in frames[1:]:
        if f - current[-1] <= gap_tolerance + 1:
            current.append(f)
        else:
            episodes.append(current)
            current = [f]
    episodes.append(current)

    # Calculer le centroïde bottom de chaque épisode
    result = []
    for ep in episodes:
        ep_rows = valid[valid["frame"].isin(ep)]
        cx = float(ep_rows["x"].mean())
        cy = float(ep_rows["y"].mean())
        result.append({"frames": ep, "cx": cx, "cy": cy})

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Stats
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(sub: pd.DataFrame) -> dict:
    dx   = np.diff(sub["x"].values)
    dy   = np.diff(sub["y"].values)
    dist = np.sqrt(dx**2 + dy**2)
    return {
        "n_frames":       len(sub),
        "dist_totale_mm": float(np.sum(dist)),
        "dist_max_mm":    float(np.max(dist)) if len(dist) else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Superposition des poses sur la trajectoire
# ──────────────────────────────────────────────────────────────────────────────

def _draw_poses(ax, cup_id: str, sub: pd.DataFrame,
                episodes: list[dict], cup_col: str):
    """
    Pour chaque épisode :
      • Surbrillance du segment de la trajectoire source pendant la pose
      • Marqueur central (losange blanc cerclé de la couleur de la tasse)
    sub : DataFrame(frame, x, y) de la source affichée (ema/raw/filtered).
    """
    if not episodes or sub.empty:
        return

    # Index rapide frame → (x, y) dans la trajectoire source
    frame_to_xy = {int(r.frame): (r.x, r.y) for r in sub.itertuples()}

    for k, ep in enumerate(episodes):
        ep_frames = set(ep["frames"])

        # ── Segment surligné ──────────────────────────────────────────────────
        # Points de la trajectoire source qui tombent dans la fenêtre de pose
        # (on étend légèrement autour pour inclure les frames intermédiaires)
        f_min, f_max = min(ep["frames"]), max(ep["frames"])
        seg = sub[(sub["frame"] >= f_min) & (sub["frame"] <= f_max)]

        if len(seg) >= 2:
            ax.plot(seg["x"].values, seg["y"].values,
                    color=cup_col, lw=5, alpha=POSE_SEGMENT_ALPHA,
                    solid_capstyle="round", zorder=3)

        # ── Marqueur central (losange) ────────────────────────────────────────
        # Position dans la trajectoire source à la frame médiane de l'épisode
        mid_frame = ep["frames"][len(ep["frames"]) // 2]
        # Chercher la frame source la plus proche
        if mid_frame in frame_to_xy:
            mx, my = frame_to_xy[mid_frame]
        else:
            # Prendre la frame source la plus proche
            src_frames = np.array(list(frame_to_xy.keys()))
            closest    = src_frames[np.argmin(np.abs(src_frames - mid_frame))]
            mx, my     = frame_to_xy[closest]

        ax.scatter(
            mx, my,
            s=POSE_MARKER_SIZE, marker="D",
            facecolor=POSE_MARKER_COLOR,
            edgecolors=cup_col,
            linewidths=2.0,
            zorder=6,
            label=f"#{cup_id} pose ×{len(episodes)}" if k == 0 else "_nolegend_",
        )

        # ── Numéro de l'épisode (petit label) ────────────────────────────────
        ax.annotate(
            str(k + 1),
            xy=(mx, my), xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=cup_col, fontweight="bold", zorder=7,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Dessin d'une tasse
# ──────────────────────────────────────────────────────────────────────────────

def _draw_one(ax, fig, cup_id, sub, source, cups,
              downsample, use_colormap, show_poses, show_stats):
    color = cup_color(cup_id)
    if downsample > 1:
        sub = sub.iloc[::downsample].reset_index(drop=True)
    x, y, n = sub["x"].values, sub["y"].values, len(sub)
    if n == 0:
        return

    label = f"Cup #{cup_id}  [{source}]"

    if use_colormap and n > 1:
        cmap = plt.colormaps["plasma"]
        norm = mcolors.Normalize(vmin=0, vmax=n - 1)
        for i in range(n - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(i)), lw=1.4, alpha=0.85)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, shrink=0.85)
        cb.set_label("Frame", color="#A0B8D0", fontsize=9, labelpad=6)
        cb.ax.yaxis.set_tick_params(color="#A0B8D0", labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#A0B8D0")
    else:
        ax.plot(x, y, color=color, lw=1.4, alpha=0.85, label=label)

    ax.scatter(x[0],  y[0],  s=70, color="#06D6A0", zorder=5,
               marker="o", label=f"#{cup_id} départ")
    ax.scatter(x[-1], y[-1], s=70, color="#E63946",  zorder=5,
               marker="X", label=f"#{cup_id} arrivée")

    # ── Poses ─────────────────────────────────────────────────────────────────
    if show_poses and source != "bottom":
        episodes = detect_pose_episodes(cups, cup_id)
        _draw_poses(ax, cup_id, sub, episodes, color)

    if show_stats:
        st   = compute_stats(sub)
        n_poses = len(detect_pose_episodes(cups, cup_id)) if show_poses else 0
        pose_line = f"Poses détectées : {n_poses}\n" if show_poses else ""
        info = (
            f"Cup #{cup_id}  [{source}]\n"
            f"{st['n_frames']} frames\n"
            f"Dist. totale : {st['dist_totale_mm']:.0f} mm\n"
            f"Saut max     : {st['dist_max_mm']:.1f} mm\n"
            f"{pose_line}"
        ).rstrip()
        ax.text(
            0.02, 0.98, info, transform=ax.transAxes,
            fontsize=8, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0A1E35",
                      edgecolor=color, alpha=0.90),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Rendu complet
# ──────────────────────────────────────────────────────────────────────────────

def render(fig, ax, cups, cup_ids, source, downsample,
           use_colormap, show_poses, file_title):
    for extra in [a for a in fig.axes if a is not ax]:
        extra.remove()
    ax.clear()

    ax.set_facecolor("#0F1E30")
    ax.set_xlabel("X (mm)", color="white", fontsize=11)
    ax.set_ylabel("Y (mm)", color="white", fontsize=11)
    ax.tick_params(colors="#7A9ABF")
    ax.spines[:].set_color("#1E3A5C")
    ax.grid(True, color="#172B40", lw=0.6, linestyle="--")

    if not cup_ids:
        ax.set_title(f"{file_title} — aucune tasse sélectionnée",
                     color="#445566", fontsize=13, pad=12)
        fig.canvas.draw_idle()
        return

    overlay = len(cup_ids) > 1
    t = (f"{file_title} — {len(cup_ids)} tasses — {source}"
         if overlay else
         f"{file_title} — Cup #{cup_ids[0]} — {source}")
    ax.set_title(t, color="#C8D8E8", fontsize=13, pad=12)

    for cup_id in cup_ids:
        sub = cups.get(cup_id, {}).get(source)
        if sub is None or sub.empty:
            print(f"[WARN] Cup #{cup_id} / '{source}' : pas de données")
            continue
        _draw_one(ax, fig, cup_id, sub, source, cups,
                  downsample, use_colormap, show_poses,
                  show_stats=(not overlay))

    if not use_colormap:
        ax.legend(loc="lower right", fontsize=8,
                  facecolor="#0A1E35", edgecolor="#1E3A5C", labelcolor="white")

    fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
#  Barre de contrôles
# ──────────────────────────────────────────────────────────────────────────────

class ControlBar(QWidget):

    def __init__(self, cups, fig, ax, downsample, file_title, parent=None):
        super().__init__(parent)
        self._cups        = cups
        self._fig         = fig
        self._ax          = ax
        self._downsample  = downsample
        self._file_title  = file_title
        self._all_ids     = sorted(cups.keys(),
                                   key=lambda x: int(x) if x.isdigit() else x)
        self._source       = "ema"
        self._active_ids   = [self._all_ids[0]] if self._all_ids else []
        self._use_colormap = False
        self._show_poses   = False

        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: {BG_PANEL};")
        self.setFixedHeight(48)
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        # ── Source ────────────────────────────────────────────────────────────
        lbl_src = QLabel("Source :")
        lbl_src.setStyleSheet(label_style(11))
        layout.addWidget(lbl_src)

        self._combo_src = QComboBox()
        self._combo_src.setStyleSheet(COMBO_STYLE)
        for s in SOURCES:
            self._combo_src.addItem(s, userData=s)
        self._combo_src.setCurrentIndex(0)
        self._combo_src.currentIndexChanged.connect(self._on_source)
        layout.addWidget(self._combo_src)

        layout.addWidget(vline())

        # ── Tasses ────────────────────────────────────────────────────────────
        lbl_cups = QLabel("Tasses :")
        lbl_cups.setStyleSheet(label_style(11))
        layout.addWidget(lbl_cups)

        self._checkboxes: dict[str, QCheckBox] = {}
        for cup_id in self._all_ids:
            color = cup_color(cup_id)
            cb    = QCheckBox(f"#{cup_id}")
            cb.setStyleSheet(checkbox_style(color))
            cb.setChecked(cup_id in self._active_ids)
            cb.stateChanged.connect(self._on_cups)
            self._checkboxes[cup_id] = cb
            layout.addWidget(cb)

        layout.addWidget(vline())

        # ── Dégradé temporel ──────────────────────────────────────────────────
        self._cb_cmap = QCheckBox("Dégradé temporel")
        self._cb_cmap.setStyleSheet(checkbox_style("#A78BFA"))
        self._cb_cmap.setChecked(False)
        self._cb_cmap.stateChanged.connect(self._on_colormap)
        layout.addWidget(self._cb_cmap)

        layout.addWidget(vline())

        # ── Poses ─────────────────────────────────────────────────────────────
        self._cb_poses = QCheckBox("Poses  ◆")
        self._cb_poses.setStyleSheet(checkbox_style("#F0C040"))
        self._cb_poses.setChecked(False)
        self._cb_poses.setToolTip(
            "Affiche les épisodes où la tasse est posée\n"
            "(détectés via la caméra bottom)\n"
            "◆ = centroïde de pose  |  trait épais = durée de l'épisode"
        )
        self._cb_poses.stateChanged.connect(self._on_poses)
        layout.addWidget(self._cb_poses)

        layout.addStretch()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_source(self, _):
        self._source = self._combo_src.currentData()
        # Désactiver "Poses" si on affiche la source bottom (redondant)
        is_bottom = self._source == "bottom"
        self._cb_poses.setEnabled(not is_bottom)
        if is_bottom:
            self._cb_poses.setChecked(False)
        self._refresh()

    def _on_cups(self, _):
        self._active_ids = [cid for cid, cb in self._checkboxes.items()
                            if cb.isChecked()]
        self._refresh()

    def _on_colormap(self, _):
        self._use_colormap = self._cb_cmap.isChecked()
        self._refresh()

    def _on_poses(self, _):
        self._show_poses = self._cb_poses.isChecked()
        self._refresh()

    def _refresh(self):
        render(
            fig=self._fig, ax=self._ax,
            cups=self._cups,
            cup_ids=self._active_ids,
            source=self._source,
            downsample=self._downsample,
            use_colormap=self._use_colormap,
            show_poses=self._show_poses,
            file_title=self._file_title,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Fenêtre principale
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryWindow(QMainWindow):
    def __init__(self, cups, file_title, downsample):
        super().__init__()
        self.setWindowTitle(f"Trajectoires — {file_title}")
        self.setStyleSheet(f"background-color: {BG_DARK};")

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor(BG_DARK)

        all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
        render(fig, ax, cups, [all_ids[0]], "ema", downsample, False, False, file_title)

        canvas  = FigureCanvasQTAgg(fig)
        mpl_bar = NavigationToolbar2QT(canvas, self)
        mpl_bar.setStyleSheet("""
            QToolBar {
                background-color: #2C3E50; border: none;
                border-top: 1px solid #1A2A3A; spacing: 2px; padding: 2px 4px;
            }
            QToolButton {
                background-color: #3A5068; border: 1px solid #4A6280;
                border-radius: 4px; padding: 3px 5px; margin: 1px;
            }
            QToolButton:hover  { background-color: #4A6A8A; border-color: #6A9ABF; }
            QToolButton:checked { background-color: #2E6DB4; border-color: #5A9ADF; }
            QToolButton:pressed { background-color: #1E4A7A; }
        """)

        ctrl_bar = ControlBar(cups, fig, ax, downsample, file_title, self)

        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DARK};")
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(ctrl_bar)
        vbox.addWidget(mpl_bar)
        vbox.addWidget(canvas, 1)
        self.setCentralWidget(central)
        self.resize(1280, 820)


# ──────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectories(cups, file_title="Trajectoires", downsample=1, output_path=None):
    all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if output_path:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG_DARK)
        render(fig, ax, cups, [all_ids[0]], "ema", downsample, False, False, file_title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] → {output_path}")
        plt.close(fig)
        return

    app = QApplication.instance() or QApplication(sys.argv)
    win = TrajectoryWindow(cups, file_title, downsample)
    win.show()
    app.exec()


def main():
    parser = argparse.ArgumentParser(
        description="Trajectoires de tasses — contrôles dans l'interface."
    )
    parser.add_argument("csv")
    parser.add_argument("--downsample", type=int, default=1, metavar="N")
    parser.add_argument("--output",     type=str, default=None, metavar="FICHIER.png")
    args = parser.parse_args()

    df    = load_csv(args.csv)
    cups  = extract_cups(df)
    title = Path(args.csv).stem.replace("_", " ")

    plot_trajectories(
        cups        = cups,
        file_title  = title,
        downsample  = max(1, args.downsample),
        output_path = args.output,
    )


if __name__ == "__main__":
    main()