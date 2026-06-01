# -*- coding: utf-8 -*-
"""
plot_trajectory.py  –  v2
=========================
Visualisation multi-tasses / multi-sources.

Nouvelles fonctionnalités :
  • 4 sources par tasse : ema | raw | filtered | bottom
  • Code couleur fixe par tasse (indépendant du timestamp)
  • Mode superposition : afficher N tasses sur le même axe
  • Le dégradé temporel reste disponible en option (--colormap)

Format CSV attendu (séparateur ; ou ,) :
    frame, timestamp,
    ID_0_x_ema, ID_0_y_ema, ID_0_x_raw, ID_0_y_raw,
    ID_0_x_filtered, ID_0_y_filtered, ID_0_x_bottom, ID_0_y_bottom, …

Usage :
    python plot_trajectory.py chemin/vers/fichier.csv
    python plot_trajectory.py fichier.csv --source filtered --cups 0 2 4
    python plot_trajectory.py fichier.csv --overlay --source ema

Options :
    --source   SOURCE  : ema | raw | filtered | bottom  (défaut : ema)
    --cups     N …     : liste d'IDs à afficher (défaut : toutes)
    --overlay          : superpose toutes les tasses sélectionnées
    --colormap         : active le dégradé temporel (désactivé par défaut)
    --downsample N     : affiche 1 point sur N (défaut : 1)
    --output    img.png: sauvegarde au lieu d'afficher
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
from PyQt6.QtWidgets import (
    QLabel, QComboBox, QCheckBox, QWidget,
    QHBoxLayout, QVBoxLayout, QButtonGroup,
)
from PyQt6.QtCore import Qt


# ──────────────────────────────────────────────────────────────────────────────
#  Palette : une couleur fixe par tasse (jusqu'à 16 tasses)
# ──────────────────────────────────────────────────────────────────────────────

CUP_COLORS = [
    "#E63946",  # 0  rouge
    "#457B9D",  # 1  bleu acier
    "#2A9D8F",  # 2  vert-cyan
    "#E9C46A",  # 3  jaune doré
    "#F4A261",  # 4  orange
    "#8338EC",  # 5  violet
    "#06D6A0",  # 6  menthe
    "#FB5607",  # 7  orange brûlé
    "#3A86FF",  # 8  bleu vif
    "#FF006E",  # 9  rose
    "#FFBE0B",  # 10 jaune
    "#8AC926",  # 11 vert
    "#6A4C93",  # 12 prune
    "#1982C4",  # 13 bleu moyen
    "#FF595E",  # 14 corail
    "#6A994E",  # 15 vert olive
]

SOURCES = ["ema", "raw", "filtered", "bottom"]


def cup_color(cup_id: str) -> str:
    """Retourne la couleur fixe associée à un ID de tasse."""
    try:
        idx = int(cup_id)
    except ValueError:
        idx = abs(hash(cup_id))
    return CUP_COLORS[idx % len(CUP_COLORS)]


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

    df = pd.read_csv(p, sep=sep)
    print(f"[CSV] {len(df)} lignes — colonnes : {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Extraction  cups[id][source] = DataFrame(frame, x, y)
# ──────────────────────────────────────────────────────────────────────────────

def extract_cups(df: pd.DataFrame) -> dict:
    """
    Retourne un dict  { cup_id: { source: DataFrame(frame, x, y) } }
    où source ∈ {"ema", "raw", "filtered", "bottom"}.
    """
    cups: dict[str, dict[str, pd.DataFrame]] = {}

    for source in SOURCES:
        x_cols = [c for c in df.columns
                  if c.endswith(f"_x_{source}") and c.startswith("ID_")]
        for col_x in x_cols:
            # ID_<id>_x_<source>
            parts = col_x.split("_")          # ['ID', '<id>', 'x', '<source>']
            if len(parts) < 4:
                continue
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

    for cid, srcs in cups.items():
        pts = {s: len(v) for s, v in srcs.items()}
        print(f"  Tasse {cid:>3} | " + " | ".join(f"{s}:{n}" for s, n in pts.items()))

    return cups


# ──────────────────────────────────────────────────────────────────────────────
#  Statistiques
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
#  Dessin d'une tasse (une source) dans un axe
# ──────────────────────────────────────────────────────────────────────────────

def _draw_one(
    ax,
    fig,
    cup_id:      str,
    sub:         pd.DataFrame,
    source:      str,
    downsample:  int  = 1,
    use_colormap: bool = False,
    show_stats:  bool = True,
    colorbar_axes=None,   # liste d'axes colorbars existantes à nettoyer
):
    """Trace une trajectoire dans ax. Retourne l'axe colorbar créé (ou None)."""
    color_base = cup_color(cup_id)

    if downsample > 1:
        sub = sub.iloc[::downsample].reset_index(drop=True)

    x = sub["x"].values
    y = sub["y"].values
    n = len(x)
    label = f"Cup #{cup_id} [{source}]"
    new_cbar_ax = None

    if n == 0:
        return new_cbar_ax

    if use_colormap and n > 1:
        cmap = plt.colormaps["plasma"]
        norm = mcolors.Normalize(vmin=0, vmax=n - 1)
        for i in range(n - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(i)),
                    linewidth=1.4, alpha=0.85)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        new_cbar_ax = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
        new_cbar_ax.set_label("Frame (début → fin)", color="white", fontsize=9)
        new_cbar_ax.ax.yaxis.set_tick_params(color="white")
        plt.setp(new_cbar_ax.ax.yaxis.get_ticklabels(), color="white")
    else:
        ax.plot(x, y, color=color_base, linewidth=1.4, alpha=0.85, label=label)

    ax.scatter(x[0],  y[0],  s=70, color="#06D6A0", zorder=5,
               marker="o", label=f"#{cup_id} départ")
    ax.scatter(x[-1], y[-1], s=70, color="#E63946",  zorder=5,
               marker="X", label=f"#{cup_id} arrivée")

    if show_stats:
        stats = compute_stats(sub)
        info  = (
            f"Cup #{cup_id}  [{source}]\n"
            f"{stats['n_frames']} frames\n"
            f"Dist. totale : {stats['dist_totale_mm']:.0f} mm\n"
            f"Saut max     : {stats['dist_max_mm']:.1f} mm"
        )
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F3460",
                      edgecolor=color_base, alpha=0.88),
        )

    return new_cbar_ax


# ──────────────────────────────────────────────────────────────────────────────
#  Rendu complet de l'axe (1 ou N tasses)
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str):
    ax.set_facecolor("#16213E")
    ax.set_xlabel("X (mm)", color="white", fontsize=11)
    ax.set_ylabel("Y (mm)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A3A5C")
    ax.grid(True, color="#2A2A4A", linewidth=0.6, linestyle="--")


def render(
    fig, ax,
    cups:         dict,
    cup_ids:      list,     # tasses à afficher
    source:       str,
    downsample:   int,
    use_colormap: bool,
    file_title:   str,
):
    """Efface et redessine l'axe avec les tasses/source demandées."""
    # Supprimer les anciennes colorbars
    main_ax = ax
    for extra in [a for a in fig.axes if a is not main_ax]:
        extra.remove()

    ax.clear()

    overlay = len(cup_ids) > 1
    title_src = f"source : {source}"

    if overlay:
        title = f"{file_title} — superposition ({len(cup_ids)} tasses) — {title_src}"
    else:
        cid   = cup_ids[0] if cup_ids else "?"
        title = f"{file_title} — Cup #{cid} — {title_src}"

    _style_ax(ax, title)

    for k, cup_id in enumerate(cup_ids):
        src_data = cups.get(cup_id, {})
        sub = src_data.get(source)
        if sub is None or sub.empty:
            print(f"[WARN] Cup #{cup_id} / source '{source}' : aucune donnée")
            continue
        _draw_one(
            ax=ax, fig=fig,
            cup_id=cup_id, sub=sub, source=source,
            downsample=downsample,
            use_colormap=use_colormap,
            show_stats=(not overlay),
        )

    if overlay or not use_colormap:
        ax.legend(
            loc="lower right", fontsize=8,
            facecolor="#0F3460", edgecolor="#3A3A5C", labelcolor="white",
        )

    fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
#  Toolbar personnalisée
# ──────────────────────────────────────────────────────────────────────────────

_COMBO_STYLE = """
    QComboBox {
        background-color: #0F3460; color: white;
        border: 1px solid #2255AA; border-radius: 4px;
        padding: 3px 8px; font-size: 12px; min-width: 90px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #0F3460; color: white;
        selection-background-color: #2255AA;
    }
"""
_CHECK_STYLE = """
    QCheckBox { color: white; font-size: 11px; spacing: 4px; }
    QCheckBox::indicator { width: 14px; height: 14px; }
    QCheckBox::indicator:unchecked { background: #0F3460; border: 1px solid #2255AA; border-radius: 3px; }
    QCheckBox::indicator:checked   { background: #2255AA; border: 1px solid #6699DD; border-radius: 3px; }
"""
_LABEL_STYLE = "color: white; font-size: 12px; padding: 0 4px;"


class TrajectoryToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, cups, fig, ax,
                 initial_cups, initial_source, downsample, use_colormap, file_title):
        super().__init__(canvas, parent)
        self._cups         = cups
        self._fig          = fig
        self._ax           = ax
        self._all_ids      = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
        self._active_ids   = list(initial_cups)
        self._source       = initial_source
        self._downsample   = downsample
        self._use_colormap = use_colormap
        self._file_title   = file_title

        self._build_controls()

    # ── construction des widgets de la toolbar ────────────────────────────────

    def _build_controls(self):
        self.addSeparator()

        # ── Sélecteur source ──────────────────────────────────────────────────
        lbl_src = QLabel("  Source : ")
        lbl_src.setStyleSheet(_LABEL_STYLE)
        self.addWidget(lbl_src)

        self._combo_src = QComboBox()
        self._combo_src.setStyleSheet(_COMBO_STYLE)
        for s in SOURCES:
            self._combo_src.addItem(s, userData=s)
        idx = SOURCES.index(self._source) if self._source in SOURCES else 0
        self._combo_src.setCurrentIndex(idx)
        self._combo_src.currentIndexChanged.connect(self._on_source_changed)
        self.addWidget(self._combo_src)

        self.addSeparator()

        # ── Checkboxes tasses ─────────────────────────────────────────────────
        lbl_cups = QLabel("  Tasses : ")
        lbl_cups.setStyleSheet(_LABEL_STYLE)
        self.addWidget(lbl_cups)

        self._checkboxes: dict[str, QCheckBox] = {}
        for cup_id in self._all_ids:
            color = cup_color(cup_id)
            cb = QCheckBox(f"#{cup_id}")
            cb.setStyleSheet(
                _CHECK_STYLE +
                f"QCheckBox::indicator:checked {{ background: {color}; border: 1px solid {color}; border-radius: 3px; }}"
            )
            cb.setChecked(cup_id in self._active_ids)
            cb.stateChanged.connect(self._on_cups_changed)
            self._checkboxes[cup_id] = cb
            self.addWidget(cb)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_source_changed(self, _):
        self._source = self._combo_src.currentData()
        self._refresh()

    def _on_cups_changed(self, _):
        self._active_ids = [cid for cid, cb in self._checkboxes.items() if cb.isChecked()]
        self._refresh()

    def _refresh(self):
        ids = self._active_ids if self._active_ids else [self._all_ids[0]]
        render(
            fig=self._fig, ax=self._ax,
            cups=self._cups,
            cup_ids=ids,
            source=self._source,
            downsample=self._downsample,
            use_colormap=self._use_colormap,
            file_title=self._file_title,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Point d'entrée graphique
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectories(
    cups:          dict,
    file_title:    str  = "Trajectoires",
    initial_cups:  list = None,
    initial_source: str = "ema",
    downsample:    int  = 1,
    use_colormap:  bool = False,
    output_path:   str  = None,
):
    all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    cup_ids = initial_cups if initial_cups else [all_ids[0]]

    # ── Mode sauvegarde ───────────────────────────────────────────────────────
    if output_path:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1A1A2E")
        render(fig, ax, cups, cup_ids, initial_source, downsample, use_colormap, file_title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Image sauvegardée → {output_path}")
        plt.close(fig)
        return

    # ── Mode interactif ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1A1A2E")

    render(fig, ax, cups, cup_ids, initial_source, downsample, use_colormap, file_title)

    manager = fig.canvas.manager
    if hasattr(manager, "toolbar") and manager.toolbar is not None:
        manager.toolbar.hide()

    win    = manager.window
    canvas = fig.canvas

    toolbar = TrajectoryToolbar(
        canvas         = canvas,
        parent         = win,
        cups           = cups,
        fig            = fig,
        ax             = ax,
        initial_cups   = cup_ids,
        initial_source = initial_source,
        downsample     = downsample,
        use_colormap   = use_colormap,
        file_title     = file_title,
    )

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
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise les trajectoires de tasses (multi-source, superposition)."
    )
    parser.add_argument("csv", help="Chemin vers le fichier CSV")
    parser.add_argument(
        "--source", choices=SOURCES, default="ema",
        help="Source de tracking initiale (défaut : ema)",
    )
    parser.add_argument(
        "--cups", nargs="+", default=None, metavar="ID",
        help="IDs de tasses à afficher (défaut : première tasse)",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Superpose toutes les tasses sélectionnées (--cups)",
    )
    parser.add_argument(
        "--colormap", action="store_true",
        help="Active le dégradé temporel plasma (désactivé par défaut)",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="Affiche 1 point sur N (défaut=1)",
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="FICHIER.png",
        help="Sauvegarde l'image au lieu de l'afficher",
    )
    args = parser.parse_args()

    df    = load_csv(args.csv)
    cups  = extract_cups(df)
    title = Path(args.csv).stem.replace("_", " ")

    all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)

    if args.cups:
        selected = [c for c in args.cups if c in cups]
        if not selected:
            print(f"[WARN] Aucun ID valide parmi {args.cups} — affichage de toutes les tasses")
            selected = all_ids
    elif args.overlay:
        selected = all_ids
    else:
        selected = [all_ids[0]]

    plot_trajectories(
        cups           = cups,
        file_title     = title,
        initial_cups   = selected,
        initial_source = args.source,
        downsample     = max(1, args.downsample),
        use_colormap   = args.colormap,
        output_path    = args.output,
    )


if __name__ == "__main__":
    main()