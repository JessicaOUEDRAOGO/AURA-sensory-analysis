# -*- coding: utf-8 -*-
"""
plot_trajectory.py  –  v7
=========================
Visualisation multi-tasses / multi-sources — tout dans l'interface.

Nouveautés v7 :
  • Vue expérimentateur : flip vertical (y → TABLE_SIZE - y) depuis repère bottom
  • Sélecteur de vue : "bottom (caméra)" | "expérimentateur" dans la barre de contrôle
  • TABLE_SIZE configurable (défaut 597 mm, modifiable via --table-size)
  • Les axes X/Y sont renommés selon la vue active

Transformation bottom → expérimentateur :
  x_exp = x_bottom          (gauche-droite inchangé)
  y_exp = TABLE_SIZE - y_bottom   (flip vertical)

Héritage v6 :
  • Couleur unique par tasse (par défaut), dégradé temporel en option
  • Flèches directionnelles activées par défaut, densité adaptative
  • Numérotation temporelle progressive (t1, t2, …) le long de la trajectoire
  • Poses : fusion des épisodes proches (< POSE_MERGE_MM), compteur sur marqueur

Usage :
    python plot_trajectory.py chemin/vers/fichier.csv
    python plot_trajectory.py fichier.csv --output img.png
    python plot_trajectory.py fichier.csv --downsample 3
    python plot_trajectory.py fichier.csv --table-size 597
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
    QWidget, QHBoxLayout, QVBoxLayout, QFrame,
)
from PyQt6.QtCore import Qt


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

# ── Taille de la table (mm) — overridable via --table-size ────────────────────
TABLE_SIZE_MM_DEFAULT = 597.0

# ── Vues disponibles ─────────────────────────────────────────────────────────
VIEW_BOTTOM = "bottom"
VIEW_EXPERIMENTER = "expérimentateur"
VIEWS = [VIEW_BOTTOM, VIEW_EXPERIMENTER]

VIEW_LABELS = {
    VIEW_BOTTOM:       "bottom (caméra)",
    VIEW_EXPERIMENTER: "expérimentateur",
}

# ── Poses ─────────────────────────────────────────────────────────────────────
POSE_MARKER_COLOR  = "#FFFFFF"
POSE_MARKER_SIZE   = 120
POSE_GAP_TOLERANCE = 5       # frames de gap max dans un même épisode
POSE_MERGE_MM      = 30.0    # distance en mm sous laquelle deux poses sont fusionnées

# ── Flèches directionnelles ───────────────────────────────────────────────────
SHOW_DIRECTION_DEFAULT = True

ARROW_SPACING_MM  = 80      # une flèche tous les 80 mm de trajectoire réelle
ARROW_TANGENT_WIN = 2       # demi-fenêtre pour la tangente locale (points)
ARROW_LENGTH_MM   = 8       # longueur visuelle de la flèche en mm
ARROW_ALPHA       = 0.85
ARROW_SIZE        = 10      # mutation_scale matplotlib

# ── Numérotation temporelle ───────────────────────────────────────────────────
SHOW_TIME_LABELS_DEFAULT = True
TIME_LABEL_COUNT   = 8          # nombre de labels "t1…tN" par trajectoire
TIME_LABEL_OFFSET  = (5, 5)     # offset pixels du texte par rapport au point


def cup_color(cup_id: str) -> str:
    try:
        idx = int(cup_id)
    except ValueError:
        idx = abs(hash(cup_id))
    return CUP_COLORS[idx % len(CUP_COLORS)]


# ──────────────────────────────────────────────────────────────────────────────
#  Transformation de vue
# ──────────────────────────────────────────────────────────────────────────────

def apply_view_transform(
    sub: pd.DataFrame,
    view: str,
    table_size: float,
) -> pd.DataFrame:
    """
    Transforme les coordonnées (x, y) d'un DataFrame selon la vue choisie.

    Repère de référence : cam_bottom (toutes les sources y sont déjà ramenées).

    Vue "bottom (caméra)"   → identité, aucune modification
    Vue "expérimentateur"   → rotation 180° autour du centre de la table :
                                x_exp = table_size - x_bottom
                                y_exp = table_size - y_bottom
                              (flip simultané sur X et Y — l'expérimentateur
                               est de l'autre côté de la table, face à la caméra)

    Retourne un nouveau DataFrame (copie) avec les colonnes x, y transformées.
    """
    out = sub.copy()
    if view == VIEW_EXPERIMENTER:
        out["x"] = table_size - out["x"]
        out["y"] = table_size - out["y"]
    # VIEW_BOTTOM : rien à faire
    return out


def axis_labels(view: str) -> tuple[str, str]:
    """Retourne (xlabel, ylabel) selon la vue active."""
    if view == VIEW_EXPERIMENTER:
        return "X — vue expérimentateur (mm)", "Y — vue expérimentateur (mm)"
    return "X (mm)", "Y (mm)"


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
    Charge les trajectoires par tasse et par source.

    Filtrage qualité : pour les sources ema / raw / filtered,
    les frames où ID_{id}_quality > 0 sont écartées :
      1 = hijack   (tracker KCF sur mauvaise cible)
      2 = airborne (tasse en l'air / occlusion)
      3 = bootstrap_pending (tracker respawné non confirmé)
      4 = lost     (aucune donnée fiable)
    La source 'bottom' (ArUco) est immunisée et n'est jamais filtrée.

    Rétrocompatibilité : si la colonne quality est absente mais que
    hijack existe, on l'utilise comme quality==1.
    """
    QUALITY_AFFECTED = {"ema", "raw", "filtered"}

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

            if source in QUALITY_AFFECTED:
                quality_col = f"ID_{cup_id}_quality"
                hijack_col  = f"ID_{cup_id}_hijack"

                if quality_col in df.columns:
                    bad = df[quality_col].fillna(0).astype(int).isin([1, 4])
                    n_bad = int(bad.sum())
                    if n_bad:
                        sub.loc[bad.values, ["x", "y"]] = np.nan
                        counts = df.loc[bad, quality_col].value_counts().sort_index()
                        detail = ", ".join(
                            f"q{int(k)}×{int(v)}"
                            for k, v in counts.items()
                        )
                        print(f"  [quality] Tasse {cup_id:>3} / {source:>8} "
                              f"→ {n_bad} frame(s) masquée(s)  ({detail})")

                elif hijack_col in df.columns:
                    bad = df[hijack_col].fillna(0).astype(int) == 1
                    n_bad = int(bad.sum())
                    if n_bad:
                        sub.loc[bad.values, ["x", "y"]] = np.nan
                        print(f"  [hijack]  Tasse {cup_id:>3} / {source:>8} "
                              f"→ {n_bad} frame(s) masquée(s)  (colonne hijack legacy)")

            sub = sub.dropna().reset_index(drop=True)
            cups.setdefault(cup_id, {})[source] = sub

    if not cups:
        print("[ERREUR] Aucune colonne ID_N_x_<source> trouvée.")
        sys.exit(1)

    for cup_id in cups:
        col_x_b = f"ID_{cup_id}_x_bottom"
        col_y_b = f"ID_{cup_id}_y_bottom"
        if col_x_b in df.columns and col_y_b in df.columns:
            b = df[["frame", col_x_b, col_y_b]].copy()
            b.columns = ["frame", "x", "y"]
            cups[cup_id]["_bottom_full"] = b
        else:
            cups[cup_id]["_bottom_full"] = pd.DataFrame(columns=["frame", "x", "y"])

    # ── Recalage top → bottom par offset médian ──────────────────────────────
    TOP_SOURCES = {"ema", "raw", "filtered"}

    for cup_id, srcs in cups.items():
        ref_ema    = srcs.get("ema")
        ref_bottom = srcs.get("bottom")

        if ref_ema is None or ref_bottom is None or ref_bottom.empty:
            continue

        merged = ref_ema.merge(
            ref_bottom[["frame", "x", "y"]].rename(
                columns={"x": "bx", "y": "by"}),
            on="frame", how="inner",
        )

        if len(merged) < 5:
            print(f"  [offset] Tasse {cup_id:>3} : pas assez de frames communes "
                  f"({len(merged)}) — recalage ignoré")
            continue

        off_x = float(np.median(merged["bx"] - merged["x"]))
        off_y = float(np.median(merged["by"] - merged["y"]))

        if abs(off_x) < 0.5 and abs(off_y) < 0.5:
            continue

        print(f"  [offset] Tasse {cup_id:>3} : "
              f"Dx={off_x:+.1f}mm  Dy={off_y:+.1f}mm  "
              f"(sur {len(merged)} frames communes)")

        for src in TOP_SOURCES:
            if src in srcs and not srcs[src].empty:
                srcs[src] = srcs[src].copy()
                srcs[src]["x"] = srcs[src]["x"] + off_x
                srcs[src]["y"] = srcs[src]["y"] + off_y

        srcs["_top_offset"] = (off_x, off_y)

    for cid, srcs in sorted(cups.items(),
                             key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        pts = {s: len(v) for s, v in srcs.items() if not s.startswith("_")}
        print(f"  Tasse {cid:>3} | " + " | ".join(f"{s}:{n}" for s, n in pts.items()))
    return cups


# ──────────────────────────────────────────────────────────────────────────────
#  Détection des épisodes de pose  +  fusion des poses proches
# ──────────────────────────────────────────────────────────────────────────────

def detect_pose_episodes(cups: dict, cup_id: str,
                          gap_tolerance: int = POSE_GAP_TOLERANCE) -> list[dict]:
    """
    Retourne une liste d'épisodes fusionnés :
        [{ "frames": [...], "cx": float, "cy": float, "count": int }, …]
    """
    b = cups[cup_id].get("_bottom_full", pd.DataFrame())
    if b.empty:
        return []

    valid = b.dropna(subset=["x", "y"]).copy()
    if valid.empty:
        return []

    frames = sorted(valid["frame"].astype(int).tolist())

    raw_episodes = []
    current = [frames[0]]
    for f in frames[1:]:
        if f - current[-1] <= gap_tolerance + 1:
            current.append(f)
        else:
            raw_episodes.append(current)
            current = [f]
    raw_episodes.append(current)

    enriched = []
    for ep in raw_episodes:
        ep_rows = valid[valid["frame"].isin(ep)]
        cx = float(ep_rows["x"].mean())
        cy = float(ep_rows["y"].mean())
        enriched.append({"frames": ep, "cx": cx, "cy": cy, "count": 1})

    merged = [enriched[0]]
    for ep in enriched[1:]:
        prev = merged[-1]
        dist = np.sqrt((ep["cx"] - prev["cx"])**2 + (ep["cy"] - prev["cy"])**2)
        if dist < POSE_MERGE_MM:
            n1 = len(prev["frames"])
            n2 = len(ep["frames"])
            total = n1 + n2
            merged[-1] = {
                "frames": prev["frames"] + ep["frames"],
                "cx": (prev["cx"] * n1 + ep["cx"] * n2) / total,
                "cy": (prev["cy"] * n1 + ep["cy"] * n2) / total,
                "count": prev["count"] + ep["count"],
            }
        else:
            merged.append(ep)

    return merged


# ──────────────────────────────────────────────────────────────────────────────
#  Découpage en segments continus
# ──────────────────────────────────────────────────────────────────────────────

GAP_FRAMES_THRESHOLD = 8

def split_continuous_segments(sub: pd.DataFrame,
                               gap: int = GAP_FRAMES_THRESHOLD) -> list[pd.DataFrame]:
    if sub.empty:
        return []
    frames = sub["frame"].astype(int).values
    diffs  = np.diff(frames)
    cuts   = np.where(diffs > gap)[0] + 1
    segs   = []
    prev   = 0
    for cut in cuts:
        segs.append(sub.iloc[prev:cut].reset_index(drop=True))
        prev = cut
    segs.append(sub.iloc[prev:].reset_index(drop=True))
    return [s for s in segs if len(s) >= 2]


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
#  Flèches directionnelles  (v6 — densité adaptative)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_direction_arrows(ax, sub: pd.DataFrame, color: str):
    n = len(sub)
    if n < ARROW_TANGENT_WIN * 2 + 2:
        return

    xs = sub["x"].values
    ys = sub["y"].values

    dxs     = np.diff(xs)
    dys     = np.diff(ys)
    dists   = np.sqrt(dxs**2 + dys**2)
    cumdist = np.concatenate([[0.0], np.cumsum(dists)])
    total   = cumdist[-1]
    if total < 1.0:
        return

    half    = ARROW_TANGENT_WIN
    targets = np.arange(ARROW_SPACING_MM, total, ARROW_SPACING_MM)

    for target in targets:
        i = int(np.searchsorted(cumdist, target))
        i = max(half, min(i, n - half - 1))

        tx = xs[i + half] - xs[i - half]
        ty = ys[i + half] - ys[i - half]
        norm = np.sqrt(tx**2 + ty**2)
        if norm < 1e-6:
            continue
        tx /= norm
        ty /= norm

        half_len = ARROW_LENGTH_MM / 2.0
        x0 = xs[i] - tx * half_len
        y0 = ys[i] - ty * half_len
        x1 = xs[i] + tx * half_len
        y1 = ys[i] + ty * half_len

        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=2.5,
                alpha=ARROW_ALPHA * 0.55,
                mutation_scale=ARROW_SIZE + 2,
                shrinkA=0, shrinkB=0,
            ),
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.3,
                alpha=ARROW_ALPHA,
                mutation_scale=ARROW_SIZE,
                shrinkA=0, shrinkB=0,
            ),
            zorder=5,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Numérotation temporelle progressive
# ──────────────────────────────────────────────────────────────────────────────

def _draw_time_labels(ax, sub: pd.DataFrame, color: str,
                      n_labels: int = TIME_LABEL_COUNT):
    n = len(sub)
    if n < n_labels * 2:
        return

    indices = np.linspace(0, n - 1, n_labels + 2, dtype=int)[1:-1]

    for rank, idx in enumerate(indices, start=1):
        x = sub["x"].iloc[idx]
        y = sub["y"].iloc[idx]
        label = f"t{rank}"

        ax.annotate(
            label,
            xy=(x, y),
            xytext=TIME_LABEL_OFFSET,
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color=color,
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#0D1B2A",
                edgecolor=color,
                alpha=0.82,
                linewidth=0.8,
            ),
        )
        ax.scatter(x, y, s=18, color=color, zorder=7, alpha=0.9)


# ──────────────────────────────────────────────────────────────────────────────
#  Superposition bottom en surligné (remplace les losanges)
# ──────────────────────────────────────────────────────────────────────────────

# Constantes visuelles du surligné bottom
BOTTOM_OVERLAY_LW    = 5.0    # épaisseur du trait surligné
BOTTOM_OVERLAY_ALPHA = 0.45   # transparence — laisse l'EMA visible par-dessus
BOTTOM_OVERLAY_ZORDER = 2     # sous l'EMA (zorder 3+)


def _draw_bottom_overlay(
    ax,
    cup_id: str,
    cups: dict,
    color: str,
    downsample: int,
    view: str,
    table_size: float,
):
    """
    Superpose la trajectoire 'bottom' (ArUco cam_bottom) en surligné épais
    semi-transparent sous la courbe EMA.

    Uniquement disponible quand source == "ema". La trajectoire bottom est
    tracée avec la même couleur que la tasse, mais plus épaisse et transparente,
    ce qui permet de lire d'un coup d'œil les zones de pose (où bottom est dense)
    vs les zones de mouvement (où bottom est absent ou fragmenté).

    La transformation de vue est appliquée de manière cohérente avec l'EMA.
    """
    bot = cups.get(cup_id, {}).get("bottom")
    if bot is None or bot.empty:
        return

    if downsample > 1:
        bot = bot.iloc[::downsample].reset_index(drop=True)

    # Transformation de vue (même repère que l'EMA affichée)
    bot = apply_view_transform(bot, view, table_size)

    segments = split_continuous_segments(bot)
    for k, seg in enumerate(segments):
        ax.plot(
            seg["x"].values, seg["y"].values,
            color=color,
            lw=BOTTOM_OVERLAY_LW,
            alpha=BOTTOM_OVERLAY_ALPHA,
            solid_capstyle="round",
            zorder=BOTTOM_OVERLAY_ZORDER,
            label=f"#{cup_id} bottom" if k == 0 else "_nolegend_",
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Dessin d'une tasse
# ──────────────────────────────────────────────────────────────────────────────

def _draw_one(
    ax, fig, cup_id, sub, source, cups,
    downsample, use_colormap,
    show_poses,
    show_direction,
    show_time_labels,
    show_stats,
    view,
    table_size,
):
    color = cup_color(cup_id)
    if downsample > 1:
        sub = sub.iloc[::downsample].reset_index(drop=True)

    # ── Appliquer la transformation de vue ───────────────────────────────────
    sub = apply_view_transform(sub, view, table_size)

    n = len(sub)
    if n == 0:
        return

    label    = f"Cup #{cup_id}  [{source}]"
    segments = split_continuous_segments(sub)

    # ── Tracé principal ───────────────────────────────────────────────────────
    if use_colormap and n > 1:
        cmap     = plt.colormaps["plasma"]
        norm     = mcolors.Normalize(vmin=0, vmax=n - 1)
        global_i = 0
        for seg in segments:
            xs, ys = seg["x"].values, seg["y"].values
            for i in range(len(xs) - 1):
                ax.plot(xs[i:i+2], ys[i:i+2],
                        color=cmap(norm(global_i + i)), lw=1.6, alpha=0.90)
            global_i += len(xs)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.018, shrink=0.85)
        cb.set_label("Frame (temps)", color="#A0B8D0", fontsize=9, labelpad=6)
        cb.ax.yaxis.set_tick_params(color="#A0B8D0", labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#A0B8D0")
        mid_color = mcolors.to_hex(cmap(0.5))
    else:
        for k, seg in enumerate(segments):
            ax.plot(seg["x"].values, seg["y"].values,
                    color=color, lw=1.6, alpha=0.88,
                    label=label if k == 0 else "_nolegend_")
        mid_color = color

    # ── Flèches directionnelles ───────────────────────────────────────────────
    if show_direction:
        for seg in segments:
            _draw_direction_arrows(ax, seg, color)

    # ── Numérotation temporelle ───────────────────────────────────────────────
    if show_time_labels:
        _draw_time_labels(ax, sub, color)

    # ── Marqueurs départ / arrivée ────────────────────────────────────────────
    x0, y0 = sub["x"].iloc[0],  sub["y"].iloc[0]
    x1, y1 = sub["x"].iloc[-1], sub["y"].iloc[-1]
    ax.scatter(x0, y0, s=80, color="#06D6A0", zorder=10,
               marker="o", label=f"#{cup_id} départ",
               edgecolors="white", linewidths=1.2)
    ax.scatter(x1, y1, s=80, color="#E63946",  zorder=10,
               marker="X", label=f"#{cup_id} arrivée",
               edgecolors="white", linewidths=0.8)

    # ── Poses bottom en surligné (EMA uniquement) ─────────────────────────────
    if show_poses and source == "ema":
        _draw_bottom_overlay(
            ax, cup_id, cups, color,
            downsample, view, table_size,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────
    if show_stats:
        st = compute_stats(sub)
        bot = cups.get(cup_id, {}).get("bottom")
        n_bot_frames = len(bot) if (show_poses and source == "ema"
                                    and bot is not None) else 0
        pose_line = (f"Bottom : {n_bot_frames} frames\n"
                     if show_poses and source == "ema" else "")
        view_line = f"Vue   : {VIEW_LABELS.get(view, view)}\n"
        info = (
            f"Cup #{cup_id}  [{source}]\n"
            f"{st['n_frames']} frames\n"
            f"Dist. totale : {st['dist_totale_mm']:.0f} mm\n"
            f"Saut max     : {st['dist_max_mm']:.1f} mm\n"
            f"{pose_line}"
            f"{view_line}"
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

def render(
    fig, ax, cups, cup_ids,
    source, downsample,
    use_colormap,
    show_poses,
    show_direction,
    show_time_labels,
    file_title,
    view=VIEW_BOTTOM,
    table_size=TABLE_SIZE_MM_DEFAULT,
):
    for extra in [a for a in fig.axes if a is not ax]:
        extra.remove()
    ax.clear()

    ax.set_facecolor("#0F1E30")
    xlabel, ylabel = axis_labels(view)
    ax.set_xlabel(xlabel, color="white", fontsize=11)
    ax.set_ylabel(ylabel, color="white", fontsize=11)
    ax.tick_params(colors="#7A9ABF")
    ax.spines[:].set_color("#1E3A5C")
    ax.grid(True, color="#172B40", lw=0.6, linestyle="--")

    if not cup_ids:
        ax.set_title(f"{file_title} — aucune tasse sélectionnée",
                     color="#445566", fontsize=13, pad=12)
        fig.canvas.draw_idle()
        return

    overlay = len(cup_ids) > 1
    view_suffix = f" — {VIEW_LABELS.get(view, view)}"
    t = (f"{file_title} — {len(cup_ids)} tasses — {source}{view_suffix}"
         if overlay else
         f"{file_title} — Cup #{cup_ids[0]} — {source}{view_suffix}")
    ax.set_title(t, color="#C8D8E8", fontsize=13, pad=12)

    for cup_id in cup_ids:
        sub = cups.get(cup_id, {}).get(source)
        if sub is None or sub.empty:
            print(f"[WARN] Cup #{cup_id} / '{source}' : pas de données")
            continue
        _draw_one(
            ax, fig,
            cup_id, sub, source, cups,
            downsample,
            use_colormap,
            show_poses,
            show_direction,
            show_time_labels,
            show_stats=(not overlay),
            view=view,
            table_size=table_size,
        )

    if not use_colormap:
        ax.legend(loc="lower right", fontsize=8,
                  facecolor="#0A1E35", edgecolor="#1E3A5C", labelcolor="white")

    fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
#  Barre de contrôles  (v7 — ajout sélecteur de vue)
# ──────────────────────────────────────────────────────────────────────────────

class ControlBar(QWidget):

    def __init__(self, cups, fig, ax, downsample, file_title, table_size, parent=None):
        super().__init__(parent)
        self._cups           = cups
        self._fig            = fig
        self._ax             = ax
        self._downsample     = downsample
        self._file_title     = file_title
        self._table_size     = table_size
        self._all_ids        = sorted(cups.keys(),
                                      key=lambda x: int(x) if x.isdigit() else x)
        self._source         = "ema"
        self._active_ids     = [self._all_ids[0]] if self._all_ids else []
        self._use_colormap   = False
        self._show_poses     = False
        self._show_direction = SHOW_DIRECTION_DEFAULT
        self._show_time_labels = SHOW_TIME_LABELS_DEFAULT
        self._view           = VIEW_BOTTOM

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

        # ── Vue ───────────────────────────────────────────────────────────────
        lbl_vue = QLabel("Vue :")
        lbl_vue.setStyleSheet(label_style(11))
        layout.addWidget(lbl_vue)

        self._combo_vue = QComboBox()
        # Style légèrement plus large pour accueillir "expérimentateur"
        vue_style = COMBO_STYLE.replace("min-width: 105px", "min-width: 160px")
        self._combo_vue.setStyleSheet(vue_style)
        for v in VIEWS:
            self._combo_vue.addItem(VIEW_LABELS[v], userData=v)
        self._combo_vue.setCurrentIndex(0)
        self._combo_vue.setToolTip(
            "bottom (caméra)   : repère ArUco cam_bottom — référence de mesure\n"
            "expérimentateur   : rotation 180° — x → table_size−x, y → table_size−y\n"
            f"                    table_size = {self._table_size:.0f} mm"
        )
        self._combo_vue.currentIndexChanged.connect(self._on_view)
        layout.addWidget(self._combo_vue)

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
        self._cb_cmap = QCheckBox("Dégradé")
        self._cb_cmap.setStyleSheet(checkbox_style("#A78BFA"))
        self._cb_cmap.setChecked(False)
        self._cb_cmap.setToolTip("Dégradé violet→jaune selon le temps\n(violet = début, jaune = fin)")
        self._cb_cmap.stateChanged.connect(self._on_colormap)
        layout.addWidget(self._cb_cmap)

        layout.addWidget(vline())

        # ── Flèches directionnelles ───────────────────────────────────────────
        self._cb_direction = QCheckBox("Flèches →")
        self._cb_direction.setStyleSheet(checkbox_style("#4A9FD4"))
        self._cb_direction.setChecked(self._show_direction)
        self._cb_direction.setToolTip("Flèches directionnelles le long de la trajectoire")
        self._cb_direction.stateChanged.connect(self._on_direction)
        layout.addWidget(self._cb_direction)

        layout.addWidget(vline())

        # ── Repères temporels ─────────────────────────────────────────────────
        self._cb_time = QCheckBox("Repères t")
        self._cb_time.setStyleSheet(checkbox_style("#06D6A0"))
        self._cb_time.setChecked(self._show_time_labels)
        self._cb_time.setToolTip(
            "Affiche des repères t1…t8 régulièrement\n"
            "répartis sur la trajectoire pour lire l'ordre"
        )
        self._cb_time.stateChanged.connect(self._on_time_labels)
        layout.addWidget(self._cb_time)

        layout.addWidget(vline())

        # ── Poses ─────────────────────────────────────────────────────────────
        self._cb_poses = QCheckBox("Poses bottom")
        self._cb_poses.setStyleSheet(checkbox_style("#F0C040"))
        self._cb_poses.setChecked(False)
        self._cb_poses.setToolTip(
            "Superpose la trajectoire bottom (ArUco) en surligné épais\n"
            "sous la courbe EMA — disponible uniquement en source 'ema'.\n"
            "Les zones denses = tasse posée, les zones vides = tasse en mouvement."
        )
        self._cb_poses.stateChanged.connect(self._on_poses)
        layout.addWidget(self._cb_poses)

        layout.addStretch()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_source(self, _):
        self._source = self._combo_src.currentData()
        is_ema = self._source == "ema"
        self._cb_poses.setEnabled(is_ema)
        if not is_ema:
            self._cb_poses.setChecked(False)
        self._refresh()

    def _on_view(self, _):
        self._view = self._combo_vue.currentData()
        self._refresh()

    def _on_cups(self, _):
        self._active_ids = [cid for cid, cb in self._checkboxes.items()
                            if cb.isChecked()]
        self._refresh()

    def _on_colormap(self, _):
        self._use_colormap = self._cb_cmap.isChecked()
        self._refresh()

    def _on_direction(self, _):
        self._show_direction = self._cb_direction.isChecked()
        self._refresh()

    def _on_time_labels(self, _):
        self._show_time_labels = self._cb_time.isChecked()
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
            show_direction=self._show_direction,
            show_time_labels=self._show_time_labels,
            file_title=self._file_title,
            view=self._view,
            table_size=self._table_size,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Fenêtre principale
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryWindow(QMainWindow):
    def __init__(self, cups, file_title, downsample, table_size):
        super().__init__()
        self.setWindowTitle(f"Trajectoires — {file_title}")
        self.setStyleSheet(f"background-color: {BG_DARK};")

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor(BG_DARK)

        all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
        render(
            fig, ax, cups, [all_ids[0]], "ema", downsample,
            use_colormap=False,
            show_poses=False,
            show_direction=SHOW_DIRECTION_DEFAULT,
            show_time_labels=SHOW_TIME_LABELS_DEFAULT,
            file_title=file_title,
            view=VIEW_BOTTOM,
            table_size=table_size,
        )

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

        ctrl_bar = ControlBar(cups, fig, ax, downsample, file_title, table_size, self)

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

def plot_trajectories(cups, file_title="Trajectoires", downsample=1,
                      output_path=None, table_size=TABLE_SIZE_MM_DEFAULT):
    all_ids = sorted(cups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if output_path:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG_DARK)
        render(
            fig, ax, cups, [all_ids[0]], "ema", downsample,
            use_colormap=False,
            show_poses=False,
            show_direction=True,
            show_time_labels=True,
            file_title=file_title,
            view=VIEW_BOTTOM,
            table_size=table_size,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] → {output_path}")
        plt.close(fig)
        return

    app = QApplication.instance() or QApplication(sys.argv)
    win = TrajectoryWindow(cups, file_title, downsample, table_size)
    win.show()
    app.exec()


def main():
    parser = argparse.ArgumentParser(
        description="Trajectoires de tasses — contrôles dans l'interface."
    )
    parser.add_argument("csv",           nargs="?", default=None)
    parser.add_argument("--downsample",  type=int,  default=1,
                        metavar="N")
    parser.add_argument("--output",      type=str,  default=None,
                        metavar="FICHIER.png")
    parser.add_argument("--table-size",  type=float,
                        default=TABLE_SIZE_MM_DEFAULT,
                        metavar="MM",
                        help=f"Taille de la table en mm (défaut : {TABLE_SIZE_MM_DEFAULT})")
    args = parser.parse_args()

    if args.csv is None:
        app = QApplication.instance() or QApplication(sys.argv)
        from PyQt6.QtWidgets import QFileDialog
        csv_path, _ = QFileDialog.getOpenFileName(
            None,
            "Ouvrir un fichier CSV de trajectoires",
            "",
            "Fichiers CSV (*.csv);;Tous les fichiers (*)",
        )
        if not csv_path:
            print("[Annulé] Aucun fichier sélectionné.")
            sys.exit(0)
    else:
        csv_path = args.csv

    df    = load_csv(csv_path)
    cups  = extract_cups(df)
    title = Path(csv_path).stem.replace("_", " ")

    plot_trajectories(
        cups        = cups,
        file_title  = title,
        downsample  = max(1, args.downsample),
        output_path = args.output,
        table_size  = args.table_size,
    )


if __name__ == "__main__":
    main()