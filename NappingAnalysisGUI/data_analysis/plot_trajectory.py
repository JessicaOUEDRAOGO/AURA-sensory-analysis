# -*- coding: utf-8 -*-
"""
plot_trajectory.py
==================
Visualisation de la trajectoire d'une ou plusieurs tasses
à partir d'un CSV produit par cup_tracking_pipeline.py.

Format CSV attendu (séparateur ; ou ,) :
    frame;ID_6_x;ID_6_y
    frame;ID_6_x;ID_6_y;ID_8_x;ID_8_y
    ...

Usage :
    python plot_trajectory.py chemin/vers/fichier.csv

Options :
    --downsample N   : affiche 1 point sur N (défaut : 1 = tous)
    --no-colormap    : désactive le dégradé temporel, trace en couleur unie
    --output img.png : sauvegarde l'image au lieu de l'afficher
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

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6.QtWidgets import QLabel, QComboBox, QVBoxLayout
from PyQt6.QtCore import Qt


# ──────────────────────────────────────────────────────────────────────────────
#  Chargement du CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(p, sep=sep)
    print(f"[CSV] {len(df)} frames — colonnes : {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Extraction des tasses
# ──────────────────────────────────────────────────────────────────────────────

def extract_cups(df: pd.DataFrame) -> dict:
    cups = {}
    x_cols = [c for c in df.columns if c.endswith("_x") and c.startswith("ID_")]

    for col_x in x_cols:
        parts = col_x.split("_")
        if len(parts) < 3:
            continue
        cup_id = parts[1]
        col_y  = f"ID_{cup_id}_y"

        if col_y not in df.columns:
            print(f"[WARN] Colonne {col_y} absente, tasse {cup_id} ignorée")
            continue

        sub = df[["frame", col_x, col_y]].copy()
        sub.columns = ["frame", "x", "y"]
        sub = sub.dropna()
        cups[cup_id] = sub
        print(f"[CSV] Tasse ID={cup_id} → {len(sub)} points valides")

    if not cups:
        print("[ERREUR] Aucune colonne ID_N_x / ID_N_y trouvée dans le CSV.")
        sys.exit(1)

    return cups


# ──────────────────────────────────────────────────────────────────────────────
#  Statistiques de mouvement
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
#  Palette
# ──────────────────────────────────────────────────────────────────────────────

CUP_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Dessin d'une tasse dans un axe donné
# ──────────────────────────────────────────────────────────────────────────────

def _draw(fig, ax, cup_id, cup_index, sub, downsample, use_colormap, title):
    ax.clear()

    # Supprimer les anciennes colorbars (axes avec y0 > 0.09 et différent de ax)
    for extra in [a for a in fig.axes if a is not ax]:
        extra.remove()

    ax.set_facecolor("#16213E")

    color_base = CUP_COLORS[cup_index % len(CUP_COLORS)]

    if downsample > 1:
        sub = sub.iloc[::downsample].reset_index(drop=True)

    x = sub["x"].values
    y = sub["y"].values
    n = len(x)

    if n == 0:
        return

    if use_colormap and n > 1:
        cmap = plt.colormaps["plasma"]
        norm = mcolors.Normalize(vmin=0, vmax=n - 1)

        for i in range(n - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(i)),
                    linewidth=1.2, alpha=0.85)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
        cbar.set_label("Frame (début → fin)", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    else:
        ax.plot(x, y, color=color_base, linewidth=1.2, alpha=0.85,
                label=f"Cup #{cup_id}")

    ax.scatter(x[0],  y[0],  s=80, color="#06D6A0", zorder=5,
               marker="o", label=f"Cup #{cup_id} — départ")
    ax.scatter(x[-1], y[-1], s=80, color="#E63946", zorder=5,
               marker="X", label=f"Cup #{cup_id} — arrivée")

    stats = compute_stats(sub)
    info  = (
        f"Cup #{cup_id}\n"
        f"{stats['n_frames']} frames\n"
        f"Dist. totale : {stats['dist_totale_mm']:.0f} mm\n"
        f"Saut max     : {stats['dist_max_mm']:.1f} mm"
    )
    ax.text(
        0.02, 0.98, info,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F3460",
                  edgecolor=color_base, alpha=0.85),
    )

    ax.set_xlabel("X (mm)", color="white", fontsize=11)
    ax.set_ylabel("Y (mm)", color="white", fontsize=11)
    ax.set_title(f"{title} — Cup #{cup_id}", color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A3A5C")
    ax.grid(True, color="#2A2A4A", linewidth=0.6, linestyle="--")

    if not use_colormap:
        ax.legend(loc="lower right", fontsize=8,
                  facecolor="#0F3460", edgecolor="#3A3A5C", labelcolor="white")

    fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
#  Toolbar personnalisée avec menu déroulant de sélection de tasse
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, cups, fig, ax,
                 downsample, use_colormap, title):
        super().__init__(canvas, parent)
        self._cups        = cups
        self._fig         = fig
        self._ax          = ax
        self._downsample  = downsample
        self._use_colormap = use_colormap
        self._title       = title
        self._cup_ids     = list(cups.keys())

        if len(self._cup_ids) > 1:
            self._add_cup_selector()

    def _add_cup_selector(self):
        # Séparateur visuel
        self.addSeparator()

        # Label
        lbl = QLabel("  Tasse : ")
        lbl.setStyleSheet("color: white; font-size: 12px;")
        self.addWidget(lbl)

        # Menu déroulant
        self._combo = QComboBox()
        self._combo.setStyleSheet("""
            QComboBox {
                background-color: #0F3460;
                color: white;
                border: 1px solid #2255AA;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                min-width: 90px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #0F3460;
                color: white;
                selection-background-color: #2255AA;
            }
        """)

        for cup_id in self._cup_ids:
            self._combo.addItem(f"Cup #{cup_id}", userData=cup_id)

        self._combo.currentIndexChanged.connect(self._on_cup_changed)
        self.addWidget(self._combo)

    def _on_cup_changed(self, index):
        cup_id    = self._cup_ids[index]
        cup_index = index
        _draw(
            fig          = self._fig,
            ax           = self._ax,
            cup_id       = cup_id,
            cup_index    = cup_index,
            sub          = self._cups[cup_id],
            downsample   = self._downsample,
            use_colormap = self._use_colormap,
            title        = self._title,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Tracé principal
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectories(
    cups:         dict,
    title:        str  = "Trajectoire tasse(s)",
    downsample:   int  = 1,
    use_colormap: bool = True,
    output_path:  str  = None,
):
    cup_ids = list(cups.keys())
    n_cups  = len(cup_ids)

    # ── Mode sauvegarde directe ───────────────────────────────────────────────
    if output_path:
        for k, (cup_id, sub) in enumerate(cups.items()):
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("#1A1A2E")
            _draw(fig, ax, cup_id, k, sub, downsample, use_colormap, title)
            plt.tight_layout()
            out = output_path if n_cups == 1 else output_path.replace(".", f"_cup{cup_id}.")
            plt.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[OK] Image sauvegardée → {out}")
            plt.close(fig)
        return

    # ── Mode interactif ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1A1A2E")

    # Dessin initial : première tasse
    _draw(fig, ax, cup_ids[0], 0, cups[cup_ids[0]], downsample, use_colormap, title)

    # Remplacer la toolbar standard par notre toolbar personnalisée
    manager = fig.canvas.manager
    if hasattr(manager, "toolbar") and manager.toolbar is not None:
        manager.toolbar.hide()

    win    = manager.window
    canvas = fig.canvas

    toolbar = TrajectoryToolbar(
        canvas       = canvas,
        parent       = win,
        cups         = cups,
        fig          = fig,
        ax           = ax,
        downsample   = downsample,
        use_colormap = use_colormap,
        title        = title,
    )

    # Insérer la toolbar en haut de la fenêtre Qt
    layout = win.layout()
    if hasattr(layout, "insertWidget"):
        layout.insertWidget(0, toolbar)
    else:
        from PyQt6.QtWidgets import QVBoxLayout, QWidget
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        win.setCentralWidget(container)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise la trajectoire d'une tasse depuis un CSV pipeline."
    )
    parser.add_argument("csv", help="Chemin vers le fichier CSV")
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="Affiche 1 point sur N (défaut=1, tous les points)"
    )
    parser.add_argument(
        "--no-colormap", action="store_true",
        help="Désactive le dégradé temporel"
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="FICHIER.png",
        help="Sauvegarde l'image (png/pdf/svg) au lieu de l'afficher"
    )
    args = parser.parse_args()

    df   = load_csv(args.csv)
    cups = extract_cups(df)

    title = Path(args.csv).stem.replace("_", " ")

    plot_trajectories(
        cups         = cups,
        title        = title,
        downsample   = max(1, args.downsample),
        use_colormap = not args.no_colormap,
        output_path  = args.output,
    )


if __name__ == "__main__":
    main()