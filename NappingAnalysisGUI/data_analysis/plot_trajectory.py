# -*- coding: utf-8 -*-
"""
plot_trajectory.py
==================
Interface graphique de visualisation de trajectoire de tasses.
Lance une fenêtre Tkinter avec :
  - des boutons pour choisir la tasse à afficher
  - le graphe matplotlib intégré dans la fenêtre
  - les stats (frames, distance totale, saut max) mises à jour à chaque sélection

Usage :
    python plot_trajectory.py chemin/vers/fichier.csv

Options :
    --downsample N   : affiche 1 point sur N (défaut : 1 = tous)
    --no-colormap    : désactive le dégradé temporel, trace en couleur unie
    --output img.png : sauvegarde la vue courante au lieu d'ouvrir l'interface
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement CSV
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Extraction des tasses
# ══════════════════════════════════════════════════════════════════════════════

def extract_cups(df: pd.DataFrame) -> dict:
    cups   = {}
    x_cols = [c for c in df.columns if c.endswith("_x") and c.startswith("ID_")]
    for col_x in x_cols:
        parts = col_x.split("_")
        if len(parts) < 3:
            continue
        cup_id = parts[1]
        col_y  = f"ID_{cup_id}_y"
        if col_y not in df.columns:
            continue
        sub = df[["frame", col_x, col_y]].copy()
        sub.columns = ["frame", "x", "y"]
        sub = sub.dropna().reset_index(drop=True)
        cups[cup_id] = sub
        print(f"[CSV] Tasse ID={cup_id} → {len(sub)} points valides")
    if not cups:
        print("[ERREUR] Aucune colonne ID_N_x / ID_N_y trouvée.")
        sys.exit(1)
    return cups


# ══════════════════════════════════════════════════════════════════════════════
#  Statistiques
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(sub: pd.DataFrame) -> dict:
    dx   = np.diff(sub["x"].values)
    dy   = np.diff(sub["y"].values)
    dist = np.sqrt(dx**2 + dy**2)
    return {
        "n_frames":       len(sub),
        "dist_totale_mm": float(np.sum(dist)),
        "dist_max_mm":    float(np.max(dist)) if len(dist) else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Dessin dans un axe matplotlib
# ══════════════════════════════════════════════════════════════════════════════

CMAP_NAME  = "plasma"
BG_FIGURE  = "#071626"
BG_AXES    = "#0D1B2A"
COL_GRID   = "#1E3A5F"
COL_TICK   = "#8899AA"
COL_START  = "#00E5A0"
COL_END    = "#FF4560"
CUP_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
              "#F4A261", "#8338EC", "#06D6A0", "#FB5607"]


def draw_on_ax(ax, sub: pd.DataFrame, cup_id: str,
               cup_index: int, use_colormap: bool, downsample: int):
    ax.clear()
    ax.set_facecolor(BG_AXES)

    if downsample > 1:
        sub = sub.iloc[::downsample].reset_index(drop=True)

    x = sub["x"].values
    y = sub["y"].values
    n = len(x)
    if n == 0:
        return

    color_base = CUP_COLORS[cup_index % len(CUP_COLORS)]

    if use_colormap and n > 1:
        cmap = plt.colormaps[CMAP_NAME]
        norm = mcolors.Normalize(vmin=0, vmax=n - 1)
        for i in range(n - 1):
            ax.plot(x[i:i+2], y[i:i+2],
                    color=cmap(norm(i)),
                    linewidth=1.5, alpha=0.9,
                    solid_capstyle="round")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02,
                                   fraction=0.025, aspect=30, shrink=0.7)
        cbar.set_label("Frame  (début → fin)", color=COL_TICK, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=COL_TICK, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COL_TICK)
        cbar.outline.set_edgecolor(COL_GRID)
    else:
        ax.plot(x, y, color=color_base, linewidth=1.5, alpha=0.9)

    ax.scatter(x[0],  y[0],  s=110, color=COL_START, zorder=6,
               marker="o", linewidths=1.5, edgecolors="white")
    ax.scatter(x[-1], y[-1], s=110, color=COL_END,   zorder=6,
               marker="X", linewidths=1.5, edgecolors="white")

    # Stats en bas à gauche
    stats = compute_stats(sub)
    info  = (
        f"{stats['n_frames']} frames\n"
        f"Dist. totale : {stats['dist_totale_mm']:.0f} mm\n"
        f"Saut max     : {stats['dist_max_mm']:.1f} mm"
    )
    ax.text(0.02, 0.03, info,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            color="#AABBCC", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0A1628",
                      edgecolor=COL_GRID, alpha=0.92))

    # Légende départ / arrivée
    legend_items = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_START, markersize=8,
               label="Départ", linewidth=0),
        Line2D([0], [0], marker="X", color="w",
               markerfacecolor=COL_END, markersize=8,
               label="Arrivée", linewidth=0),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              fontsize=8, facecolor="#0A1628",
              edgecolor=COL_GRID, labelcolor="white",
              framealpha=0.9)

    ax.set_title(f"Cup #{cup_id}", color="white",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("X (mm)", color=COL_TICK, fontsize=9)
    ax.set_ylabel("Y (mm)", color=COL_TICK, fontsize=9)
    ax.tick_params(colors=COL_TICK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)
    ax.grid(True, color=COL_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
#  Interface Tkinter
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryApp:
    def __init__(self, root: tk.Tk, cups: dict,
                 title: str, use_colormap: bool, downsample: int):
        self.root         = root
        self.cups         = cups
        self.use_colormap = use_colormap
        self.downsample   = downsample
        self.cup_ids      = list(cups.keys())
        self.active_id    = self.cup_ids[0]
        self._buttons     = {}

        root.title(f"Trajectoire — {title}")
        root.configure(bg="#071626")
        root.resizable(True, True)

        self._build_ui(title)
        self._refresh()

    def _build_ui(self, title: str):
        # ── Bandeau supérieur ──────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#071626", pady=8)
        top.pack(side=tk.TOP, fill=tk.X, padx=12)

        tk.Label(top, text="Tasse :", bg="#071626", fg="#8899AA",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 8))

        for cup_id in self.cup_ids:
            btn = tk.Button(
                top,
                text=f"Cup #{cup_id}",
                bg="#0F3460", fg="white",
                activebackground="#1E5FAA", activeforeground="white",
                relief=tk.FLAT,
                padx=12, pady=4,
                font=("Helvetica", 10),
                cursor="hand2",
                command=lambda cid=cup_id: self._select(cid),
            )
            btn.pack(side=tk.LEFT, padx=3)
            self._buttons[cup_id] = btn

        # Bouton sauvegarde
        tk.Button(
            top,
            text="💾  Sauvegarder PNG",
            bg="#1A3A1A", fg="#80CC80",
            activebackground="#2A5A2A", activeforeground="white",
            relief=tk.FLAT,
            padx=10, pady=4,
            font=("Helvetica", 10),
            cursor="hand2",
            command=self._save_png,
        ).pack(side=tk.RIGHT, padx=3)

        # Nom du fichier
        tk.Label(top, text=title, bg="#071626", fg="#445566",
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=10)

        # ── Canvas matplotlib ─────────────────────────────────────────────
        self.fig = plt.Figure(figsize=(9, 6.5), facecolor=BG_FIGURE)
        self.ax  = self.fig.add_subplot(111)

        canvas_frame = tk.Frame(self.root, bg="#071626")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _select(self, cup_id: str):
        self.active_id = cup_id
        self._refresh()

    def _refresh(self):
        # Mettre en surbrillance le bouton actif
        for cid, btn in self._buttons.items():
            if cid == self.active_id:
                btn.configure(bg="#2255AA", relief=tk.SUNKEN)
            else:
                btn.configure(bg="#0F3460", relief=tk.FLAT)

        # Supprimer l'ancienne colorbar si elle existe
        if self.fig.axes and len(self.fig.axes) > 1:
            for extra_ax in self.fig.axes[1:]:
                extra_ax.remove()

        idx = self.cup_ids.index(self.active_id)
        draw_on_ax(
            ax          = self.ax,
            sub         = self.cups[self.active_id],
            cup_id      = self.active_id,
            cup_index   = idx,
            use_colormap= self.use_colormap,
            downsample  = self.downsample,
        )
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_png(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=f"cup_{self.active_id}_trajectory.png",
            title="Sauvegarder la trajectoire",
        )
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"[OK] Sauvegardé → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Mode sauvegarde directe (--output)
# ══════════════════════════════════════════════════════════════════════════════

def save_direct(cups: dict, title: str, cup_filter: str,
                use_colormap: bool, downsample: int, output: str):
    ids = [cup_filter] if cup_filter and cup_filter in cups else list(cups.keys())
    for cup_id in ids:
        fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG_FIGURE)
        idx = list(cups.keys()).index(cup_id)
        draw_on_ax(ax, cups[cup_id], cup_id, idx, use_colormap, downsample)
        fig.tight_layout()
        out = output if len(ids) == 1 else output.replace(".", f"_cup{cup_id}.")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Sauvegardé → {out}")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interface de visualisation de trajectoire de tasses."
    )
    parser.add_argument("csv",
        help="Chemin vers le fichier CSV")
    parser.add_argument("--cup", type=str, default=None, metavar="ID",
        help="Sélectionne directement la tasse ID=N au démarrage (ex: --cup 6)")
    parser.add_argument("--downsample", type=int, default=1, metavar="N",
        help="Affiche 1 point sur N (défaut=1 = tous)")
    parser.add_argument("--no-colormap", action="store_true",
        help="Désactive le dégradé temporel")
    parser.add_argument("--output", type=str, default=None, metavar="FICHIER",
        help="Sauvegarde directement en PNG/PDF/SVG sans ouvrir l'interface")
    args = parser.parse_args()

    df    = load_csv(args.csv)
    cups  = extract_cups(df)
    title = Path(args.csv).stem

    use_colormap = not args.no_colormap
    downsample   = max(1, args.downsample)

    # Mode sauvegarde directe (pas d'interface)
    if args.output:
        save_direct(cups, title, args.cup, use_colormap, downsample, args.output)
        return

    # Mode interface
    root = tk.Tk()
    app  = TrajectoryApp(root, cups, title, use_colormap, downsample)

    # Sélection initiale via --cup
    if args.cup and args.cup in cups:
        app._select(args.cup)

    root.mainloop()


if __name__ == "__main__":
    main()