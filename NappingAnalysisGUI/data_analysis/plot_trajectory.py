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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import numpy as np
import pandas as pd


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
        "x_range":        (float(sub["x"].min()), float(sub["x"].max())),
        "y_range":        (float(sub["y"].min()), float(sub["y"].max())),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Palette
# ──────────────────────────────────────────────────────────────────────────────

CUP_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
]


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
    bottom_margin = 0.10 if n_cups > 1 else 0.02

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#1A1A2E")

    # L'axe principal occupe toute la zone au-dessus des boutons
    ax = fig.add_axes([0.08, bottom_margin + 0.02, 0.82, 0.88])

    state = {"active_idx": 0}

    # Paramètres des boutons
    btn_w   = min(0.12, 0.9 / max(n_cups, 1))
    btn_gap = 0.01
    total_w = n_cups * btn_w + (n_cups - 1) * btn_gap
    x_start = (1.0 - total_w) / 2

    def make_buttons():
        """Crée les axes boutons et retourne la liste des objets Button."""
        btns = []
        for i, cup_id in enumerate(cup_ids):
            x_pos  = x_start + i * (btn_w + btn_gap)
            btn_ax = fig.add_axes([x_pos, 0.01, btn_w, 0.05])
            color  = "#2255AA" if i == state["active_idx"] else "#0F3460"
            btn    = Button(btn_ax, f"Cup #{cup_id}",
                            color=color, hovercolor="#2255AA")
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
            btns.append(btn)
        return btns

    # Dessin initial
    _draw(fig, ax, cup_ids[0], 0, cups[cup_ids[0]], downsample, use_colormap, title)

    if n_cups > 1:
        btn_list = make_buttons()

        def on_click(event, idx, cid):
            state["active_idx"] = idx

            # Supprimer les anciennes colorbars UNIQUEMENT
            # (les axes boutons sont identifiés par leur position y < 0.10)
            to_remove = [a for a in fig.axes
                         if a is not ax and a.get_position().y0 > 0.09]
            for a in to_remove:
                a.remove()

            _draw(fig, ax, cid, idx, cups[cid], downsample, use_colormap, title)

            # Mettre à jour la couleur des boutons
            for j, b in enumerate(btn_list):
                b.ax.set_facecolor("#2255AA" if j == idx else "#0F3460")
                b.color      = "#2255AA" if j == idx else "#0F3460"
                b.hovercolor = "#2255AA"

            fig.canvas.draw_idle()

        for i, cup_id in enumerate(cup_ids):
            btn_list[i].on_clicked(
                lambda event, idx=i, cid=cup_id: on_click(event, idx, cid)
            )

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Dessin d'une tasse dans un axe donné
# ──────────────────────────────────────────────────────────────────────────────

def _draw(fig, ax, cup_id, cup_index, sub, downsample, use_colormap, title):
    """Efface l'axe et dessine la trajectoire de la tasse."""
    ax.clear()
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
        cmap = plt.colormaps["plasma"]   # API moderne, pas de warning
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