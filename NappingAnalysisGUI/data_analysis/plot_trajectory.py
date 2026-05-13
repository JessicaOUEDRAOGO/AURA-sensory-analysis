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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Chargement du CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Charge le CSV en détectant automatiquement le séparateur (; ou ,).
    Retourne un DataFrame avec les colonnes telles quelles.
    """
    p = Path(path)
    if not p.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)

    # Détection du séparateur
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
    """
    Détecte automatiquement toutes les tasses (colonnes ID_N_x / ID_N_y).
    Retourne {cup_id: DataFrame avec colonnes [frame, x, y]}.
    """
    cups = {}
    x_cols = [c for c in df.columns if c.endswith("_x") and c.startswith("ID_")]

    for col_x in x_cols:
        # Extrait l'ID : "ID_6_x" → "6"
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
    """Calcule quelques statistiques utiles sur la trajectoire."""
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
#  Tracé
# ──────────────────────────────────────────────────────────────────────────────

# Palette de couleurs distinctes pour les tasses
CUP_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
]


def plot_trajectories(
    cups:        dict,
    title:       str  = "Trajectoire tasse(s)",
    downsample:  int  = 1,
    use_colormap: bool = True,
    output_path: str  = None,
):
    """
    Affiche ou sauvegarde le graphique de trajectoire.

    Paramètres
    ----------
    cups         : {cup_id: DataFrame[frame, x, y]}
    title        : titre du graphique
    downsample   : 1 = tous les points, N = 1 point sur N
    use_colormap : True = dégradé temporel (début→fin), False = couleur unie
    output_path  : None = affichage interactif, sinon chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for k, (cup_id, sub) in enumerate(cups.items()):
        color_base = CUP_COLORS[k % len(CUP_COLORS)]

        # Sous-échantillonnage
        if downsample > 1:
            sub = sub.iloc[::downsample].reset_index(drop=True)

        x = sub["x"].values
        y = sub["y"].values
        n = len(x)

        if n == 0:
            continue

        if use_colormap and n > 1:
            # Dégradé temporel : du début (sombre) à la fin (clair)
            # On découpe la trajectoire en segments colorés
            cmap = cm.get_cmap("plasma")
            norm = mcolors.Normalize(vmin=0, vmax=n - 1)

            for i in range(n - 1):
                c = cmap(norm(i))
                ax.plot(x[i:i+2], y[i:i+2], color=c, linewidth=1.2, alpha=0.85)

            # Barre de couleur pour l'axe temporel
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
            cbar.set_label("Frame (début → fin)", color="white", fontsize=10)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        else:
            # Couleur unie
            ax.plot(x, y, color=color_base, linewidth=1.2, alpha=0.85,
                    label=f"Cup #{cup_id}")

        # Point de départ (vert) et d'arrivée (rouge)
        ax.scatter(x[0],  y[0],  s=80, color="#06D6A0", zorder=5,
                   marker="o", label=f"Cup #{cup_id} — départ")
        ax.scatter(x[-1], y[-1], s=80, color="#E63946", zorder=5,
                   marker="X", label=f"Cup #{cup_id} — arrivée")

        # Statistiques dans le graphe
        stats = compute_stats(sub)
        info  = (
            f"Cup #{cup_id}\n"
            f"{stats['n_frames']} frames\n"
            f"Dist. totale : {stats['dist_totale_mm']:.0f} mm\n"
            f"Saut max     : {stats['dist_max_mm']:.1f} mm"
        )
        ax.text(
            0.02, 0.98 - k * 0.16,
            info,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F3460",
                      edgecolor=color_base, alpha=0.85),
        )

    # Mise en forme des axes
    ax.set_xlabel("X (mm)", color="white", fontsize=11)
    ax.set_ylabel("Y (mm)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A3A5C")
    ax.grid(True, color="#2A2A4A", linewidth=0.6, linestyle="--")

    # Inversion Y si besoin (repère image : y croissant vers le bas)
    # Décommentez la ligne suivante si la trajectoire apparaît inversée :
    # ax.invert_yaxis()

    # Légende (seulement si pas de colormap, pour éviter la surcharge)
    if not use_colormap or len(cups) > 1:
        legend = ax.legend(
            loc="lower right",
            fontsize=8,
            facecolor="#0F3460",
            edgecolor="#3A3A5C",
            labelcolor="white",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Image sauvegardée → {output_path}")
    else:
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
