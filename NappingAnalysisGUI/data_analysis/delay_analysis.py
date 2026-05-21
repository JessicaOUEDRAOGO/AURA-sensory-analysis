# -*- coding: utf-8 -*-
"""
delay_analysis.py
=================
Analyse le décalage temporel exact entre cam_top et cam_bottom
à partir du CSV enrichi (après application du patch ts_bottom).

Usage :
    python delay_analysis.py <session.csv>
    python delay_analysis.py <session.csv> --ids 6 8

Colonnes requises (ajoutées par ts_bottom_patch) :
    ID_{id}_delay_ms    — décalage en ms : timestamp_cam_top - ts_bottom
    ID_{id}_ts_bottom   — timestamp exact de capture cam_bottom

Colonnes optionnelles (pour analyse approfondie) :
    ID_{id}_x_filtered, ID_{id}_y_filtered  — position Kalman cam_top
    ID_{id}_x_bottom,   ID_{id}_y_bottom    — position ArUco cam_bottom
"""

import csv
import sys
import os
import statistics
from datetime import datetime
from collections import defaultdict


# ── Paramètres ────────────────────────────────────────────────────────────────

# Seuils de qualité de synchronisation (ms)
SYNC_EXCELLENT  =  10.0   # < 10ms  → quasi-simultané
SYNC_GOOD       =  25.0   # < 25ms  → bonne sync
SYNC_ACCEPTABLE =  45.0   # < 45ms  → normal (cam_bottom à 22Hz = 45ms/frame)
# > 45ms → cam_bottom a manqué une frame ou thread en retard

FPS_NOMINAL = 24.5


# ── Utilitaires ───────────────────────────────────────────────────────────────

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

def separator(char='─', width=62):
    return char * width

def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def detect_aruco_ids(rows):
    ids = set()
    for col in rows[0].keys():
        if col.startswith('ID_') and col.endswith('_delay_ms'):
            ids.add(col.split('_')[1])
    return sorted(ids, key=int)

def sync_label(delay_ms):
    if delay_ms < SYNC_EXCELLENT:
        return 'excellent'
    elif delay_ms < SYNC_GOOD:
        return 'bon'
    elif delay_ms < SYNC_ACCEPTABLE:
        return 'acceptable'
    else:
        return 'mauvais'


# ── Chargement des délais ─────────────────────────────────────────────────────

def load_delays(rows, aruco_id):
    """
    Extrait pour chaque frame :
      - delay_ms       : décalage cam_top - cam_bottom
      - spatial_gap    : écart en mm entre x_filtered et x_bottom (si dispo)
      - ts_top         : timestamp cam_top
      - ts_bottom      : timestamp cam_bottom
    """
    cid = aruco_id
    records = []
    fmt = '%Y-%m-%d %H:%M:%S.%f'

    for r in rows:
        delay_str = r.get(f'ID_{cid}_delay_ms', '')
        if not delay_str:
            continue

        delay_ms = float(delay_str)

        # Spatial gap entre cam_top (Kalman) et cam_bottom (ArUco)
        xf = r.get(f'ID_{cid}_x_filtered', '')
        yf = r.get(f'ID_{cid}_y_filtered', '')
        xb = r.get(f'ID_{cid}_x_bottom',   '')
        yb = r.get(f'ID_{cid}_y_bottom',   '')
        spatial_gap = None
        if xf and xb:
            spatial_gap = ((float(xf)-float(xb))**2 +
                           (float(yf)-float(yb))**2) ** 0.5

        records.append({
            'frame':       int(r['frame']),
            'ts_top':      r['timestamp'],
            'ts_bottom':   r.get(f'ID_{cid}_ts_bottom', ''),
            'delay_ms':    delay_ms,
            'spatial_gap': spatial_gap,
            'sync':        sync_label(delay_ms),
        })

    return records


# ── Rapport principal ─────────────────────────────────────────────────────────

def report_delay(records, aruco_id):

    delays = [r['delay_ms'] for r in records]
    n = len(delays)

    if not delays:
        print(f"  ✗ Aucune donnée delay_ms pour ID_{aruco_id}")
        print(f"    → Vérifier que le patch ts_bottom a bien été appliqué")
        return

    print(f"\n  Frames avec delay_ms : {n}")

    # ── Distribution statistique ──────────────────────────────────────────────
    print(f"""
  ┌─ Décalage cam_top − cam_bottom (ms) ──────────────────┐
  │ Métrique          Valeur                               │
  ├────────────────────────────────────────────────────────┤
  │  Minimum        {min(delays):>8.1f} ms                          │
  │  Moyenne        {statistics.mean(delays):>8.1f} ms                          │
  │  Médiane        {statistics.median(delays):>8.1f} ms                          │
  │  P75            {percentile(delays, 75):>8.1f} ms                          │
  │  P90            {percentile(delays, 90):>8.1f} ms                          │
  │  P95            {percentile(delays, 95):>8.1f} ms                          │
  │  Maximum        {max(delays):>8.1f} ms                          │
  │  Écart-type     {statistics.stdev(delays):>8.1f} ms                          │
  └────────────────────────────────────────────────────────┘""")

    # ── Distribution par catégorie ────────────────────────────────────────────
    cats = defaultdict(int)
    for r in records:
        cats[r['sync']] += 1

    n_exc = cats['excellent']
    n_bon = cats['bon']
    n_acc = cats['acceptable']
    n_mau = cats['mauvais']

    print(f"""
  ┌─ Qualité de synchronisation ───────────────────────────┐
  │ Catégorie     Seuil        N       %      Interprétation│
  ├────────────────────────────────────────────────────────┤
  │  Excellent    < 10ms    {n_exc:>5}  {n_exc/n*100:>5.1f}%   quasi-simultané    │
  │  Bon          < 25ms    {n_bon:>5}  {n_bon/n*100:>5.1f}%   très fiable        │
  │  Acceptable   < 45ms    {n_acc:>5}  {n_acc/n*100:>5.1f}%   normal (22Hz bot.) │
  │  Mauvais      > 45ms    {n_mau:>5}  {n_mau/n*100:>5.1f}%   frame sautée       │
  └────────────────────────────────────────────────────────┘""")

    # ── Histogramme ASCII ─────────────────────────────────────────────────────
    print(f"\n  Histogramme du délai (ms) :")
    bins = [(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),
            (30,35),(35,40),(40,45),(45,60),(60,100),(100,999)]
    max_count = 0
    counts = []
    for lo, hi in bins:
        c = sum(1 for d in delays if lo <= d < hi)
        counts.append(c)
        max_count = max(max_count, c)

    bar_width = 35
    for (lo, hi), c in zip(bins, counts):
        bar_len = int(c / max_count * bar_width) if max_count > 0 else 0
        bar = '█' * bar_len
        label = f"{lo:>3}-{hi:<3}ms"
        pct = c / n * 100
        print(f"  {label} │{bar:<{bar_width}} {c:>5} ({pct:>5.1f}%)")

    # ── Impact sur l'écart spatial ────────────────────────────────────────────
    spatial_records = [r for r in records if r['spatial_gap'] is not None]
    if spatial_records:
        print(f"\n  ┌─ Écart spatial top/bottom selon qualité sync ──────────┐")
        print(f"  │ Catégorie    N avec gap   Gap moyen   Gap médiane        │")
        print(f"  ├────────────────────────────────────────────────────────┤")
        for cat in ['excellent', 'bon', 'acceptable', 'mauvais']:
            gaps = [r['spatial_gap'] for r in spatial_records if r['sync'] == cat]
            if gaps:
                print(f"  │  {cat:<12} {len(gaps):>8}   "
                      f"{statistics.mean(gaps):>8.1f}mm  "
                      f"{statistics.median(gaps):>8.1f}mm             │")
        print(f"  └────────────────────────────────────────────────────────┘")
        print(f"""
  → Si le gap spatial diminue quand le délai diminue,
    l'écart top/bottom est principalement dû à la désynchronisation.
  → Si le gap reste constant quelle que soit la sync,
    l'écart est dû à la calibration inter-caméra (biais géométrique).""")

    # ── Cas extrêmes ──────────────────────────────────────────────────────────
    worst = sorted(records, key=lambda r: r['delay_ms'], reverse=True)[:5]
    print(f"\n  Top 5 pires délais :")
    print(f"  {'frame':>6}  {'ts_top':>23}  {'ts_bottom':>23}  {'delay':>8}  {'sync'}")
    print(f"  {'─'*6}  {'─'*23}  {'─'*23}  {'─'*8}  {'─'*10}")
    for r in worst:
        print(f"  {r['frame']:>6}  {r['ts_top']:>23}  "
              f"{r['ts_bottom'] or 'N/A':>23}  "
              f"{r['delay_ms']:>7.1f}ms  {r['sync']}")

    # ── Dérive temporelle sur la session ─────────────────────────────────────
    # Découper en tranches de 10% de la session et voir si le délai dérive
    chunk_size = max(1, n // 10)
    chunks = [records[i:i+chunk_size] for i in range(0, n, chunk_size)]
    if len(chunks) >= 5:
        print(f"\n  Évolution du délai sur la session (découpage en 10ème) :")
        print(f"  {'Tranche':>8}  {'Frames':>10}  {'Délai moy':>12}  {'Délai méd':>12}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")
        for i, chunk in enumerate(chunks):
            d = [r['delay_ms'] for r in chunk]
            f_start = chunk[0]['frame']
            f_end   = chunk[-1]['frame']
            print(f"  {i+1:>4}/10    "
                  f"{f_start:>5}-{f_end:<5}  "
                  f"{statistics.mean(d):>10.1f}ms  "
                  f"{statistics.median(d):>10.1f}ms")

    # ── Recommandation ────────────────────────────────────────────────────────
    med = statistics.median(delays)
    pct_reliable = (n_exc + n_bon) / n * 100

    print(f"\n  ┌─ Recommandation ───────────────────────────────────────┐")
    if med < SYNC_EXCELLENT:
        print(f"  │  ✓ Excellente synchronisation (médiane {med:.1f}ms)          │")
        print(f"  │    x_bottom fiable pour comparaison directe avec top   │")
    elif med < SYNC_GOOD:
        print(f"  │  ✓ Bonne synchronisation (médiane {med:.1f}ms)               │")
        print(f"  │    x_bottom utilisable — filtrer delay_ms < 25ms       │")
    elif med < SYNC_ACCEPTABLE:
        print(f"  │  ⚠ Sync normale (médiane {med:.1f}ms = ~1 frame cam_bottom) │")
        print(f"  │    Utiliser x_bottom uniquement quand tasse immobile   │")
        print(f"  │    Filtrer : delay_ms < 45ms AND state = MATCHED       │")
    else:
        print(f"  │  ✗ Sync dégradée (médiane {med:.1f}ms)                      │")
        print(f"  │    Investiguer : charge CPU ? conflit USB caméras ?     │")
    print(f"  │                                                        │")
    print(f"  │  Frames fiables (delay < 25ms) : {pct_reliable:>5.1f}%              │")
    print(f"  └────────────────────────────────────────────────────────┘")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage : python delay_analysis.py <session.csv>")
        print("        python delay_analysis.py <session.csv> --ids 6 8")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Fichier introuvable : {csv_path}")
        sys.exit(1)

    ids_override = None
    if '--ids' in sys.argv:
        idx = sys.argv.index('--ids')
        ids_override = sys.argv[idx+1:]

    print(separator('═'))
    print(f"  DELAY ANALYSIS — cam_top vs cam_bottom")
    print(f"  Fichier : {os.path.basename(csv_path)}")
    print(separator('═'))

    rows = load_csv(csv_path)

    # Vérifier que le patch est bien appliqué
    first = rows[0]
    has_delay = any(k.endswith('_delay_ms') for k in first.keys())
    if not has_delay:
        print("\n  ✗ Colonne delay_ms introuvable dans ce CSV.")
        print("    Ce fichier a été produit AVANT le patch ts_bottom.")
        print("    Appliquer ts_bottom_patch.py et relancer une session.")
        sys.exit(1)

    aruco_ids = ids_override or detect_aruco_ids(rows)

    # Infos session
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    ts0 = datetime.strptime(rows[0]['timestamp'], fmt)
    ts1 = datetime.strptime(rows[-1]['timestamp'], fmt)
    duree = (ts1 - ts0).total_seconds()
    intervals = [
        (datetime.strptime(rows[i+1]['timestamp'], fmt) -
         datetime.strptime(rows[i]['timestamp'], fmt)).total_seconds() * 1000
        for i in range(min(200, len(rows)-1))
    ]
    fps_real = 1000 / statistics.median(intervals)

    print(f"\n  Durée   : {duree:.0f}s ({duree/60:.1f} min)")
    print(f"  Frames  : {len(rows)}")
    print(f"  FPS     : {fps_real:.1f} Hz")
    print(f"  Tasses  : {', '.join(f'ID_{i}' for i in aruco_ids)}")
    print(f"\n  Seuils sync : excellent < {SYNC_EXCELLENT}ms  "
          f"bon < {SYNC_GOOD}ms  "
          f"acceptable < {SYNC_ACCEPTABLE}ms  "
          f"mauvais ≥ {SYNC_ACCEPTABLE}ms")

    for cid in aruco_ids:
        print(f"\n{separator('─')}")
        print(f"  TASSE ID_{cid}")
        print(separator('─'))

        records = load_delays(rows, cid)
        report_delay(records, cid)

    print(f"\n{separator('═')}")
    print("  INTERPRÉTATION")
    print(separator('─'))
    print("  delay_ms  = timestamp_cam_top − ts_bottom")
    print("            = temps écoulé entre la capture cam_bottom")
    print("              et l'écriture dans le CSV par cam_top")
    print("  Valeur positive toujours : cam_bottom capture avant cam_top")
    print("  Médiane attendue ~20ms   : la moitié d'une frame cam_bottom (45ms)")
    print("  > 45ms                   : cam_bottom a sauté une frame")
    print(separator('═'))


if __name__ == '__main__':
    main()
