# -*- coding: utf-8 -*-
"""
lag_analysis.py
===============
Calcule et rapporte le lag de l'EMA et du filtre de Kalman
à partir du CSV de session napping.

Usage :
    python lag_analysis.py <chemin_vers_session.csv>

Colonnes attendues :
    frame, timestamp,
    ID_{id}_x_raw, ID_{id}_y_raw,
    ID_{id}_x_ema, ID_{id}_y_ema,
    ID_{id}_x_filtered, ID_{id}_y_filtered

Le "lag" mesuré ici est la distance euclidienne entre la position
brute (raw = référence temporelle exacte) et la position filtrée
(ema ou kalman) au même timestamp. C'est la définition opérationnelle
du retard spatial induit par le filtre.
"""

import csv
import sys
import os
import statistics
from datetime import datetime
from collections import defaultdict


# ── Paramètres ────────────────────────────────────────────────────────────────

# Seuils de vitesse (mm/frame) pour classifier les phases
SPEED_STATIC   = 2.0    # en dessous → tasse immobile
SPEED_SLOW     = 5.0    # entre static et slow → mouvement lent
SPEED_FAST     = 10.0   # au-dessus → mouvement rapide

# IDs ArUco à analyser (détectés automatiquement si None)
ARUCO_IDS = None

FPS_NOMINAL = 24.5      # pour convertir frames en secondes


# ── Utilitaires ───────────────────────────────────────────────────────────────

def dist(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

def fmt_mm(v):
    return f"{v:6.2f} mm"

def fmt_pct(n, total):
    return f"{n/total*100:5.1f}%" if total else "  n/a"

def separator(char='─', width=62):
    return char * width


# ── Chargement ────────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def detect_aruco_ids(rows):
    ids = set()
    for col in rows[0].keys():
        if col.startswith('ID_') and col.endswith('_x_raw'):
            ids.add(col.split('_')[1])
    return sorted(ids, key=int)


# ── Calcul du lag ─────────────────────────────────────────────────────────────

def compute_lag(rows, aruco_id):
    """
    Pour chaque frame où raw + ema + filtered sont présents :
      - calcule la vitesse instantanée (mm/frame) sur le raw
      - calcule lag_ema  = dist(raw, ema)
      - calcule lag_kal  = dist(raw, filtered)
    Retourne une liste de dicts.
    """
    cid = aruco_id
    records = []
    prev_raw = None
    fmt = '%Y-%m-%d %H:%M:%S.%f'

    for r in rows:
        xr = r.get(f'ID_{cid}_x_raw', '')
        yr = r.get(f'ID_{cid}_y_raw', '')
        xe = r.get(f'ID_{cid}_x_ema', '')
        ye = r.get(f'ID_{cid}_y_ema', '')
        xf = r.get(f'ID_{cid}_x_filtered', '')
        yf = r.get(f'ID_{cid}_y_filtered', '')

        if not (xr and xe and xf):
            prev_raw = None
            continue

        xr, yr = float(xr), float(yr)
        xe, ye = float(xe), float(ye)
        xf, yf = float(xf), float(yf)

        speed = dist(xr, yr, *prev_raw) if prev_raw is not None else 0.0
        prev_raw = (xr, yr)

        records.append({
            'frame':     int(r['frame']),
            'timestamp': r['timestamp'],
            'raw_x':     xr, 'raw_y': yr,
            'ema_x':     xe, 'ema_y': ye,
            'kal_x':     xf, 'kal_y': yf,
            'speed':     speed,
            'lag_ema':   dist(xr, yr, xe, ye),
            'lag_kal':   dist(xr, yr, xf, yf),
        })

    return records


# ── Rapport ───────────────────────────────────────────────────────────────────

def phase_label(speed):
    if speed < SPEED_STATIC:
        return 'statique'
    elif speed < SPEED_SLOW:
        return 'lent'
    elif speed < SPEED_FAST:
        return 'moyen'
    else:
        return 'rapide'

def report_filter(records, filter_key, filter_name):
    """Imprime le tableau de stats pour un filtre donné."""
    phases = ['statique', 'lent', 'moyen', 'rapide', 'TOTAL']
    buckets = defaultdict(list)
    for rec in records:
        buckets[phase_label(rec['speed'])].append(rec[filter_key])
        buckets['TOTAL'].append(rec[filter_key])

    # En-tête
    print(f"\n  ┌─ {filter_name} {'─'*(44-len(filter_name))}┐")
    print(f"  │ {'Phase':<10} {'N':>5}  {'Moy':>8}  {'Médiane':>8}  {'P95':>8}  {'Max':>8} │")
    print(f"  ├{'─'*54}┤")

    order = ['statique', 'lent', 'moyen', 'rapide', 'TOTAL']
    speed_ranges = {
        'statique': f'< {SPEED_STATIC}mm/f',
        'lent':     f'{SPEED_STATIC}–{SPEED_SLOW}mm/f',
        'moyen':    f'{SPEED_SLOW}–{SPEED_FAST}mm/f',
        'rapide':   f'> {SPEED_FAST}mm/f',
        'TOTAL':    'toutes phases',
    }
    for phase in order:
        data = buckets.get(phase, [])
        if not data:
            continue
        sep = '╞' if phase == 'TOTAL' else ' '
        end = '╡' if phase == 'TOTAL' else ' '
        print(f"  │{sep} {phase:<10} {len(data):>5}  "
              f"{statistics.mean(data):>7.2f}mm  "
              f"{statistics.median(data):>7.2f}mm  "
              f"{percentile(data,95):>7.2f}mm  "
              f"{max(data):>7.2f}mm {end}│")

    print(f"  └{'─'*54}┘")

    return buckets

def report_improvement(rec_ema, rec_kal):
    """Compare EMA vs Kalman par phase."""
    print(f"\n  ┌─ Amélioration Kalman vs EMA {'─'*24}┐")
    print(f"  │ {'Phase':<10} {'EMA moy':>9}  {'Kal moy':>9}  {'Gain':>8}  {'% mieux':>8} │")
    print(f"  ├{'─'*54}┤")

    phases = ['statique', 'lent', 'moyen', 'rapide', 'TOTAL']
    for phase in phases:
        de = rec_ema.get(phase, [])
        dk = rec_kal.get(phase, [])
        if not de or not dk:
            continue
        me, mk = statistics.mean(de), statistics.mean(dk)
        gain = me - mk
        pct  = (1 - mk/me) * 100 if me > 0 else 0
        sep = '╞' if phase == 'TOTAL' else ' '
        end = '╡' if phase == 'TOTAL' else ' '
        print(f"  │{sep} {phase:<10} {me:>8.2f}mm  {mk:>8.2f}mm  "
              f"{gain:>7.2f}mm  {pct:>7.0f}% {end}│")

    print(f"  └{'─'*54}┘")

def report_temporal(records, fps=FPS_NOMINAL):
    """
    Estime le lag temporel en ms à partir du lag spatial et de la vitesse.
    lag_ms ≈ lag_mm / speed_mm_per_frame * frame_duration_ms
    Valide uniquement en phase de mouvement.
    """
    frame_ms = 1000.0 / fps
    ema_ms, kal_ms = [], []
    for rec in records:
        spd = rec['speed']
        if spd < SPEED_SLOW:  # trop lent → division instable
            continue
        ema_ms.append(rec['lag_ema'] / spd * frame_ms)
        kal_ms.append(rec['lag_kal'] / spd * frame_ms)

    if not ema_ms:
        return

    print(f"\n  ┌─ Lag temporel estimé (phases speed > {SPEED_SLOW}mm/f) {'─'*6}┐")
    print(f"  │ {'Filtre':<12} {'Moy':>10}  {'Médiane':>10}  {'P95':>10}    │")
    print(f"  ├{'─'*54}┤")
    for name, data in [('EMA', ema_ms), ('Kalman', kal_ms)]:
        print(f"  │  {name:<12} {statistics.mean(data):>8.1f}ms  "
              f"{statistics.median(data):>8.1f}ms  "
              f"{percentile(data,95):>8.1f}ms    │")
    print(f"  └{'─'*54}┘")
    print(f"  (référence théorique EMA α=0.35 : ~76ms = 1.86 frames)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python lag_analysis.py <session.csv>")
        print("       python lag_analysis.py <session.csv> --ids 6 8")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Fichier introuvable : {csv_path}")
        sys.exit(1)

    # IDs optionnels en argument
    ids_override = None
    if '--ids' in sys.argv:
        idx = sys.argv.index('--ids')
        ids_override = sys.argv[idx+1:]

    print(separator('═'))
    print(f"  LAG ANALYSIS — EMA vs KALMAN")
    print(f"  Fichier : {os.path.basename(csv_path)}")
    print(separator('═'))

    rows = load_csv(csv_path)
    aruco_ids = ids_override or detect_aruco_ids(rows)

    # Infos session
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    ts0 = datetime.strptime(rows[0]['timestamp'], fmt)
    ts1 = datetime.strptime(rows[-1]['timestamp'], fmt)
    duree = (ts1 - ts0).total_seconds()
    intervals = [(datetime.strptime(rows[i+1]['timestamp'], fmt) -
                  datetime.strptime(rows[i]['timestamp'], fmt)).total_seconds()*1000
                 for i in range(min(200, len(rows)-1))]
    fps_real = 1000 / statistics.median(intervals)

    print(f"\n  Durée        : {duree:.0f}s ({duree/60:.1f} min)")
    print(f"  Frames       : {len(rows)}")
    print(f"  FPS mesuré   : {fps_real:.1f} Hz")
    print(f"  Tasses       : {', '.join(f'ID_{i}' for i in aruco_ids)}")
    print(f"\n  Seuils vitesse : statique < {SPEED_STATIC}  lent < {SPEED_SLOW}"
          f"  moyen < {SPEED_FAST}  rapide ≥ {SPEED_FAST}  (mm/frame)")

    for cid in aruco_ids:
        print(f"\n{separator('─')}")
        print(f"  TASSE ID_{cid}")
        print(separator('─'))

        records = compute_lag(rows, cid)
        if not records:
            print("  ✗ Pas de données (colonnes raw/ema/filtered absentes ou vides)")
            continue

        # Couverture
        n_total   = len(rows)
        n_records = len(records)
        print(f"\n  Frames analysées : {n_records} / {n_total} ({n_records/n_total*100:.1f}%)")

        # Phase breakdown
        phase_counts = defaultdict(int)
        for rec in records:
            phase_counts[phase_label(rec['speed'])] += 1
        print(f"  Phases : " + "  ".join(
            f"{p}={phase_counts[p]} ({phase_counts[p]/n_records*100:.0f}%)"
            for p in ['statique','lent','moyen','rapide'] if phase_counts[p]))

        # Tableaux lag
        bk_ema = report_filter(records, 'lag_ema', 'EMA  (α=0.35)')
        bk_kal = report_filter(records, 'lag_kal', 'Kalman (pn=2, mn=8)')

        # Comparaison
        report_improvement(bk_ema, bk_kal)

        # Lag temporel estimé
        report_temporal(records, fps=fps_real)

        # Cas extrêmes
        worst_ema = sorted(records, key=lambda r: r['lag_ema'], reverse=True)[:3]
        worst_kal = sorted(records, key=lambda r: r['lag_kal'], reverse=True)[:3]
        print(f"\n  Top 3 pires frames EMA :")
        for rec in worst_ema:
            print(f"    frame {rec['frame']:>5}  speed={rec['speed']:5.1f}mm/f  "
                  f"lag_ema={rec['lag_ema']:6.2f}mm")
        print(f"  Top 3 pires frames Kalman :")
        for rec in worst_kal:
            print(f"    frame {rec['frame']:>5}  speed={rec['speed']:5.1f}mm/f  "
                  f"lag_kal={rec['lag_kal']:6.2f}mm")

    print(f"\n{separator('═')}")
    print("  INTERPRÉTATION")
    print(separator('─'))
    print("  lag spatial  = distance entre position brute et filtrée au même timestamp")
    print("  lag temporel = estimation du décalage temporel (lag_mm / vitesse * dt)")
    print("  Phase statique : le lag mesure le bruit résiduel du filtre (pas un vrai retard)")
    print("  Phase mobile   : le lag mesure le retard temporel réel induit par le filtre")
    print(separator('═'))


if __name__ == '__main__':
    main()
