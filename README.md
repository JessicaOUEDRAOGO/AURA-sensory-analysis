# Table RA Projetée — Suivi de tasses par réalité augmentée projective

Système de suivi visuel temps réel pour l'analyse sensorielle du café (méthode *napping*), développé dans le cadre d'un stage Centrale Lyon ENISE / LIRIS en collaboration avec l'Institut Lyfe. La table associe une double caméra (vue du dessus + vue du dessous via tags ArUco) à un projecteur, pour suivre la position de tasses et afficher en temps réel leur identité directement sur le plateau.

> Démo vidéo et résumé du projet : [Portfolio — section projets](https://jessicaouedraogo.github.io/Portfolio)

---

## Aperçu du système

Trois éléments physiques coopèrent :

- **Caméra du haut (`cam_top`)** — suit chaque tasse par tracker visuel (KCF / MOSSE) sur l'image en niveaux de gris, détecte les blobs sombres (tasses) par seuillage + morphologie.
- **Caméra du bas (`cam_bottom`)** — détecte des tags ArUco collés sous chaque tasse, donnant une identité fiable et une position de référence (vérité terrain).
- **Projecteur** — affiche un anneau coloré et l'identifiant de chaque tasse directement sur la table, calé sur la position suivie par `cam_top`.

Le problème central : le tracker visuel (rapide, mais sans mémoire d'identité) doit rester synchronisé avec le tag ArUco (lent à apparaître, mais fiable) — y compris quand une tasse est soulevée, quand deux tasses se croisent, ou quand le tracker "saute" sur la tasse voisine.

```
cam_top (tracking KCF/MOSSE) ──┐
                                ├──► CupIdentityManager ──► Projecteur (overlay)
cam_bottom (ArUco)         ─────┘                     └──► Export CSV / JSON
```

---

## Fonctionnalités clés

- **Tracking dual-caméra** avec liaison d'identité tracker ↔ tag ArUco (`CupIdentityManager`), états `PENDING → MATCHED → AIRBORNE / LOST`.
- **Détection de hijacking** : si la position du tracker dérive de plus de `ARUCO_DRIFT_MM` par rapport au tag ArUco pendant `ARUCO_DRIFT_FRAMES` frames consécutives, le tracker est invalidé et les frames contaminées sont marquées rétroactivement dans le CSV (colonne qualité).
- **Anti-scintillement ArUco** : une repose de tasse n'est confirmée qu'après `REPOSE_CONF_FRAMES` détections stables consécutives, pour éviter les faux positifs liés au clignotement naturel de la détection ArUco (~22 fps).
- **Bootstrap des tags orphelins** : si un tag ArUco reste sans tracker associé pendant `ORPHAN_FRAMES_BEFORE_BOOTSTRAP` frames, un nouveau tracker est spawné automatiquement à la position corrigée (avec garde-fou d'ambiguïté spatiale et validation post-spawn sur plusieurs frames avant confirmation ou rollback).
- **Correction de perspective 3D** : la position de la tasse est recalculée à la base du gobelet (et non au centre du blob détecté), pour compenser le décalage de hauteur (`CUP_HEIGHT_MM`) vu depuis une caméra non-zénithale.
- **Recalage cam_top → cam_bottom** : un offset EMA appris en continu corrige le biais géométrique (~13 mm) entre les deux référentiels caméra, appliqué uniquement à l'export — les positions internes du tracker ne sont jamais modifiées.
- **Export multi-sources** : pour chaque tasse, le CSV conserve séparément la position EMA-lissée, la position brute tracker, la position filtrée par Kalman, et la position ArUco — pour permettre une analyse a posteriori de la fiabilité de chaque source.
- **Journal d'associations JSON** : historique complet des liaisons/déliaisons tracker↔tag (bind, unbind, hijack) avec horodatage, pour audit de session.

---

## Architecture technique

### Filtrage des positions

| Filtre | Usage | Latence | Rôle |
|---|---|---|---|
| `EMAFilter` (α=0.35) | Projection temps réel | ~immédiate | Lissage visuel, anti-saut (`EMA_MAX_JUMP_MM`) |
| `KalmanFilter2D` (modèle vitesse constante) | Export CSV | ~2 mm de lag | Précision pour l'analyse post-session |

### Gestion d'identité (`CupIdentityManager`)

Machine à états par tasse (`CupIdentity`), indexée par `aruco_id` :

- `PENDING` — tag vu, pas encore de tracker associé
- `MATCHED` — tracker et tag liés et cohérents
- `AIRBORNE` — tag disparu (tasse soulevée), tracker peut continuer seul un court instant
- `LOST` — tracker disparu, tag non revu depuis trop longtemps

Le matching tracker↔tag se fait par plus proche distance mutuelle (`_match_pending`), sous contrainte de distance maximale `MATCH_DIST_MM`.

### Détection de dérive (anti-hijacking)

Un tracker MOSSE/KCF peut "glisser" sur une tasse voisine pendant un croisement. `check_aruco_drift()` compare en continu la position du tracker à celle du tag ArUco associé ; au-delà du seuil, le tracker est invalidé plutôt que de continuer à publier une position erronée.

### Calibration et repères

Trois transformations géométriques sont nécessaires :

1. **`camtop_table_pose.json`** — pose caméra du haut → plan de la table (rvec/tvec/camera_matrix), via projection plan-image inverse.
2. **`H_top_to_bottom.json`** — homographie recalant le repère `cam_top` sur le repère `cam_bottom` (référence).
3. **`H_table_to_proj.json`** — homographie table → image projecteur, pour positionner les anneaux affichés.

`PoseConverter` encapsule la conversion pixel → mm (et son inverse, utilisée par le bootstrap pour replacer un tracker à une position cible).

---

## Structure du repo

```
.
├── napping_lite.py                  # Pipeline autonome (mono-thread, sans Qt) — version de référence stable
├── NappingAnalysisGUI/               # Application complète (Qt, deux threads : tracking + rendu)
│   ├── src/
│   │   ├── core/
│   │   │   ├── vision/               # camera_manager.py — accès caméra bas niveau
│   │   │   ├── cup_tracking/         # cam_bottom_thread.py, cup_identity_manager.py, video_writer_thread.py
│   │   │   ├── projection/           # display_manager.py, draw_utils.py — sortie projecteur
│   │   │   ├── config/               # app_config.py — constantes globales
│   │   │   └── utils/                # paths.py — résolution des chemins config/data
│   │   └── ...
│   └── config/                       # Fichiers de calibration JSON (pose caméras, homographies)
├── 3D_Solidworks/                    # Conception CAO du support de tasse (impression 3D)
└── Config_AutoPyToExe.json           # Config de packaging en exécutable Windows
```

> ℹ️ Section à confirmer/compléter — dis-moi si certains dossiers ont été renommés ou ajoutés depuis (ex. scripts d'analyse `plot_trajectory.py`, `cup_detect_photo.py`) pour que je les ajoute ici.

---

## Installation

```bash
conda create -n ra_clean python=3.11
conda activate ra_clean
pip install opencv-contrib-python numpy PyQt6
```

⚠️ Utiliser impérativement `opencv-contrib-python` (et non `opencv-python`) — `cv2.aruco` et `cv2.legacy.TrackerMOSSE_create()` ne sont disponibles que dans le build contrib.

Sous Windows, désactiver la **suspension sélective USB** sur les ports caméra (Gestionnaire de périphériques → Propriétés du concentrateur USB → Gestion de l'alimentation) — sans quoi `cam_top` subit des coupures aléatoires, diagnostiquées comme un problème matériel/OS et non logiciel.

## Lancement

```bash
python napping_lite.py
```

Lancement interactif : nom du protocole, ID participant, activation de la projection. Le script crée automatiquement les fichiers de sortie dans `data/sessions/lite/`.

**Touches en session :**

| Touche | Action |
|---|---|
| `q` | Quitter |
| `r` | Reset complet des trackers |
| `+` / `-` | Ajuster le seuil de détection |
| `c` | Activer/désactiver la correction 3D |
| `i` | Afficher le résumé de debug des identités |
| `p` | Activer/désactiver la projection |

Pour l'application complète avec interface Qt, exécuter depuis la racine de `NappingAnalysisGUI/` en mode module : `python -m src.main` (adapter selon le point d'entrée réel).

## Données exportées

Chaque session produit 4 fichiers, horodatés et nommés `{protocole}_{participant}_{timestamp}` :

- **`*.csv`** — une ligne par frame, colonnes `ID_{tag}_x/y_ema`, `_raw`, `_filtered`, `_bottom`, et `_quality` (0=OK, 1=hijack détecté, 2=airborne, 3=bootstrap non confirmé, 4=lost).
- **`*_trackers_raw.csv`** — une ligne par tracker actif par frame, sans aucune perte de donnée même avant association à un tag (utile pour diagnostiquer les pertes de matching).
- **`*_associations.json`** — historique complet bind/unbind/hijack par tag, avec horodatage et numéro de frame.
- **`*.mp4`** — enregistrement vidéo `cam_top` (écriture non bloquante via thread dédié).

## Limites connues

- Écart géométrique résiduel de ~13 mm entre les référentiels `cam_top` et `cam_bottom`, compensé par un offset appris mais non éliminé à la source (calibration à reprendre).
- `cam_bottom` n'applique pas de correction de distorsion fisheye avant détection ArUco — source d'erreur de position en périphérie de table.
- La séparation de tasses qui se touchent (blobs fusionnés) n'est pas encore gérée par un watershed dédié dans la version photo statique (`cup_detect_photo.py`, en cours).

## Stack technique

Python 3.11 · OpenCV (contrib) · NumPy · PyQt6 · ArUco (`DICT_4X4_50`) · Filtrage EMA + Kalman (modèle vitesse constante)

---

Développé par Jessica Ouedraogo dans le cadre d'un double diplôme ingénieur (ENSAM Meknès — électromécanique) / M2 Technologies 3D Interactives (Arts et Métiers Chalon-sur-Saône).
