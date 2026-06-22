# Table RA Projetée — suivre des tasses identiques sans jamais perdre leur identité

Système de réalité augmentée projective pour l'analyse sensorielle du café (méthode *napping*), développé en stage à Centrale Lyon ENISE / LIRIS avec l'Institut Lyfe. Une table suit en temps réel la position de tasses de café et projette directement sur le plateau l'identité de chacune — pendant que les participants évaluent le café sans rien manipuler d'autre que leur tasse.

> Démo vidéo et contexte du projet : [Portfolio](https://jessicaouedraogo.github.io/Portfolio)

---

## Le problème, version courte

Suivre 9 tasses **strictement identiques** en temps réel, savoir laquelle est laquelle à tout moment — même quand une main la soulève, la repose, ou que deux tasses se croisent — sans jamais permuter leurs identités. Voilà le problème. Ce README raconte comment il a été résolu, par itérations successives, chacune révélant le problème suivant.

---

## Itération 1 — une seule caméra ne suffit pas

Le premier prototype reposait uniquement sur une caméra placée sous la table, détectant des tags ArUco collés au fond de chaque tasse. Identité fiable, position précise — mais un trou béant : **dès qu'une tasse est soulevée, le tag sort du champ et toute la trajectoire est perdue.** Pas de données entre le décollage et la repose — exactement le moment le plus intéressant à observer dans un protocole de napping (un participant qui hésite, repose puis re-soulève une tasse).

**→ Il fallait une deuxième source d'information qui continue de voir la tasse même en l'air.**

## Itération 2 — une caméra du dessus, et un nouveau problème : l'identité

Une caméra plongeante (`cam_top`) a été ajoutée pour suivre visuellement chaque tasse par sa silhouette, indépendamment du tag ArUco. Elle ne perd jamais la tasse de vue tant qu'elle reste sur la table ou juste au-dessus.

Mais cette caméra ne voit que des silhouettes — et les 9 tasses sont identiques. Elle peut suivre une forme, pas dire *laquelle*. Il fallait donc associer chaque silhouette suivie à une identité réelle, sans pouvoir s'appuyer sur l'apparence.

**Solution adoptée :** `cam_top` fait le tracking visuel (détection + suivi de boîte englobante), `cam_bottom` reste la source d'identité via ArUco. Un gestionnaire dédié (`CupIdentityManager`) fait le pont entre les deux : il lie un `tracker_id` (purement visuel, sans signification) à un `aruco_id` (l'identité réelle) dès qu'ils sont détectés à proximité l'un de l'autre.

## Itération 3 — deux caméras, deux threads, et un décalage à minimiser

Faire tourner les deux caméras dans le même thread Python s'est avéré quasiment impossible à maintenir en performance : la lecture bloquante d'une caméra (`cap.read()`) gèle l'autre. Chaque caméra tourne donc dans son propre thread.

Mais deux threads indépendants ne tournent jamais à une fréquence parfaitement identique. Mesuré en session réelle :

| Source | Fréquence réelle |
|---|---|
| `cam_top` (boucle principale) | 25 Hz (40 ms/frame) |
| `cam_bottom` (thread ArUco) | ~22 Hz (asynchrone) |

Conséquence directe : au moment où une ligne du CSV est écrite, la position ArUco disponible (`x_bottom`) n'est pas forcément celle de la même frame physique — l'écart peut aller de 0 à ~45 ms. C'est ce qui explique un écart résiduel de position de ~15-20 mm entre `cam_top` et `cam_bottom`, même quand une tasse est parfaitement immobile et que tout fonctionne normalement (état `MATCHED`). Tout l'effort de cette étape a consisté à réduire au maximum cet écart temporel plutôt qu'à l'éliminer complètement (ce qui demanderait une synchronisation matérielle des deux caméras, hors de portée ici).

## Itération 4 — 9 tasses à ~30 fps : KCF est trop lent

Le protocole final exige de suivre jusqu'à 9 tasses simultanément à une cadence proche de 30 fps. Le tracker visuel utilisé jusque-là, **KCF**, coûte environ 8 ms par tasse suivie. Pour 2 tasses, c'est négligeable (16 ms). Pour 9 tasses, ça représente déjà ~72 ms de calcul par frame — largement incompatible avec un budget de ~33-40 ms par frame.

**→ Migration vers MOSSE**, un tracker par corrélation beaucoup plus léger (sous la milliseconde par instance au lieu de 8 ms). Le gain de performance est net — mais MOSSE a une faiblesse connue : il est nettement moins robuste que KCF aux occlusions et aux changements brusques d'apparence.

*(Trace de cette migration encore visible dans le code aujourd'hui : les méthodes s'appellent toujours `update_kcf()` / `reinit_kcf()`, mais créent en réalité un `cv2.legacy.TrackerMOSSE_create()`. Le nom n'a jamais été mis à jour — l'historique du projet, littéralement, dans le code.)*

## Itération 5 — MOSSE perd pied dès qu'il y a collision

Conséquence prévisible du passage à MOSSE : dès que deux tasses se croisent ou se touchent, le tracker peut décrocher de sa cible — soit il la perd complètement, soit, pire, il **s'accroche à la tasse voisine** en silence, sans jamais signaler d'erreur. Sans garde-fou, ce genre de glissement produit un tracker qui continue de répondre, mais qui ment sur la position réelle de la tasse.

**→ Construction d'un système auto-correctif**, qui ne fait pas confiance à un tracker juste parce qu'il répond. Il croise systématiquement deux sources d'information :

1. **Re-détection périodique** (toutes les ~150 ms, soit 1 frame sur 4) : un masque binaire redétecte les silhouettes de tasses sur l'image et tente de réassocier chaque tracker à la détection la plus proche, le réinitialisant si besoin.
2. **Contrainte de taille de boîte** (`BBOX_GROW_RATIO`) : si la boîte suivie par MOSSE grossit anormalement (signe classique d'un tracker qui a perdu sa cible et suit une zone floue plus large), le tracker est invalidé immédiatement plutôt que de continuer à publier une position fausse.
3. **Vérification croisée avec l'ArUco** (`check_aruco_drift`) : tant qu'un tracker est lié à un tag, sa position est comparée en continu à la position ArUco réelle de la même tasse. Si l'écart dépasse un seuil pendant plusieurs frames consécutives, c'est la preuve que le tracker a glissé sur une autre tasse — il est invalidé, et les frames déjà écrites dans le CSV pendant la dérive sont **marquées rétroactivement comme suspectes** (colonne qualité), plutôt que silencieusement laissées telles quelles.
4. **Récupération des tags orphelins** (bootstrap) : si un tag ArUco reste détecté sans aucun tracker associé pendant un certain temps (tasse repérée mais jamais reliée à un suivi visuel), un nouveau tracker est spawné automatiquement à la position corrigée, avec une zone d'exclusion autour des autres tags connus pour éviter une mauvaise association, puis validé sur plusieurs frames avant confirmation.
5. **Anti-scintillement** : les tags ArUco clignotent naturellement (détectés/perdus en alternance, normal à ~22 Hz). Sans filtrage, ça produirait des cycles `AIRBORNE ↔ MATCHED` plusieurs fois par seconde. Une repose n'est donc confirmée qu'après 3 détections stables consécutives.

Résultat mesurable sur une session réelle (`ArUco #6` et `#8`, ~4500 frames) : chaque tag a connu entre 4 et 6 `tracker_id` différents au cours de la session — preuve que le système recrée des trackers en continu sans jamais perdre l'identité réelle (`aruco_id`), qui elle reste stable du début à la fin.

---

## Architecture résultante

```
cam_top (MOSSE, 25 Hz) ──┐
                          ├──► CupIdentityManager ──► Projecteur (anneaux + identité)
cam_bottom (ArUco, ~22Hz)┘         (PENDING / MATCHED /        └──► CSV + JSON
                                     AIRBORNE / LOST)
```

| État | Signification |
|---|---|
| `PENDING` | Tag vu, pas encore de tracker associé |
| `MATCHED` | Tracker et tag liés et cohérents |
| `AIRBORNE` | Tasse soulevée — tag invisible, tracker continue seul |
| `LOST` | Tracker perdu, tag invisible depuis trop longtemps |

---

## Deux filtres pour deux usages différents

La position brute (`x_raw`) est exacte temporellement mais bruitée (~1,5 mm). Deux traitements distincts répondent à deux besoins différents :

| | EMA (α=0.35) | Kalman (modèle position+vitesse) |
|---|---|---|
| Usage | Projection temps réel | Export CSV pour analyse |
| Lag mesuré (mouvement rapide) | ~24-26 mm | ~2,5 mm |
| Lag max mesuré | ~42 mm | ~10 mm |
| Pourquoi | Lissage visuel, pas de saccade à l'œil | Précision temporelle — quasi zéro retard |

Sur une session de test, le Kalman réduit le décalage de position de **~90 %** par rapport à l'EMA en phase de mouvement rapide. L'EMA reste utile pour l'affichage (l'œil ne voit pas 2 mm de retard, mais voit une saccade), tandis que le Kalman est la colonne à utiliser pour toute analyse de trajectoire.

⚠️ Le timestamp inscrit dans les CSV est pris *après* tout le traitement (EMA, Kalman, écriture), pas au moment de la capture — décalage systématique d'environ 10-15 ms. Sans impact sur les intervalles entre frames (donc sur les vitesses calculées), mais à corriger si une synchronisation externe à <15 ms est nécessaire (capter `datetime.now()` juste après `cap.read()`).

---

## Données exportées

Chaque session produit :

- **`{session}.csv`** — une ligne par frame, 4 sources de position par tasse (`_raw`, `_ema`, `_filtered`, `_bottom`) + une colonne de qualité (0=OK, 1=hijack détecté, 2=airborne, 3=bootstrap en validation, 4=lost).
- **`{session}_trackers_raw.csv`** — une ligne par tracker actif par frame, sans aucune perte, même avant association à un tag.
- **`{session}_associations.json`** — historique complet des liaisons/déliaisons tracker↔tag (bind, unbind, hijack), avec frame et horodatage.
- **`{session}.mp4`** — enregistrement vidéo `cam_top`.

**Pour une analyse post-session :** utiliser `x_filtered`/`y_filtered` (Kalman) comme position de référence, indexer par `aruco_id` (jamais `tracker_id`, qui change à chaque rebind), filtrer sur `state = MATCHED` pour les phases de contact avec la table, et garder `x_raw` comme contrôle qualité.

---

## Limites connues

- L'écart géométrique ~15-20 mm entre `cam_top` et `cam_bottom` est compensé (offset appris) mais pas éliminé à la source — une synchronisation matérielle des deux caméras réduirait le problème à sa racine.
- `cam_bottom` ne corrige pas la distorsion fisheye avant détection ArUco, source d'erreur en périphérie de table.
- La séparation de tasses qui se touchent (blobs fusionnés) reste à traiter proprement dans la version d'analyse sur photo statique (`cup_detect_photo.py`, en cours).

---

## Installation

```bash
conda create -n ra_clean python=3.11
conda activate ra_clean
pip install opencv-contrib-python numpy PyQt6
```

⚠️ `opencv-contrib-python` est obligatoire (pas `opencv-python`) — `cv2.aruco` et `cv2.legacy.TrackerMOSSE_create()` n'existent que dans le build contrib.

Sous Windows, désactiver la **suspension sélective USB** sur les ports caméra (Gestionnaire de périphériques → Concentrateur USB → Gestion de l'alimentation) : sans ça, `cam_top` subit des coupures aléatoires — un problème matériel/OS, pas logiciel.

## Lancement

```bash
python napping_lite.py
```

Lancement interactif (nom du protocole, ID participant, projection on/off). Sorties écrites dans `data/sessions/lite/`.

| Touche | Action |
|---|---|
| `q` | Quitter |
| `r` | Reset complet des trackers |
| `+` / `-` | Ajuster le seuil de détection |
| `c` | Activer/désactiver la correction 3D |
| `i` | Résumé de debug des identités |
| `p` | Activer/désactiver la projection |

---

## Stack technique

Python 3.11 · OpenCV (contrib) · NumPy · PyQt6 · ArUco (`DICT_4X4_50`) · Filtrage EMA + Kalman (modèle vitesse constante)

---

Développé par Jessica Ouedraogo — double diplôme ingénieur électromécanique (ENSAM Meknès) / M2 Technologies 3D Interactives (Arts et Métiers Chalon-sur-Saône).
