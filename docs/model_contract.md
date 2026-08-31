# Contrat des modèles CV

Référence : ce document décrit la **structure standardisée** d'un dossier modèle dans `data/models/<model_name>/`. Tout modèle livré dans le plugin doit respecter ce contrat. Un validateur (`scripts/validate_models_metadata.py`) et deux suites pytest (`tests/unit/test_models_metadata.py`, `tests/unit/test_models_connectivity.py`) en garantissent le respect.

## Structure attendue

```
data/models/<model_name>/
├── model_card.yaml              # manifeste humain consolidé + divergences documentées
├── args.yaml                    # SAHI + clustering + postprocess + class_colors
├── classes.txt                  # snake_case ASCII, doublons autorisés (sous-classes)
├── training_params.json         # contexte génération images d'entraînement + UI hints
├── config.json                  # snapshot training (hyperparams, dataset)
├── evaluation_results.json      # recommandé — métriques par split
├── weights/
│   ├── best.onnx                # produit par dev/runner_onnx/export_to_onnx.py
│   └── best.json                # sidecar runtime produit par export_to_onnx.py
└── weights/best.pth             # optionnel (gros, idéalement gitignore)
```

Fichiers tolérés mais hors contrat runtime : `metrics.csv`, `events.out.tfevents.*`, `hparams.yaml`. Si `weights/classes.txt` existe, il doit être **byte-identique** à `classes.txt` racine.

### `entrainement/` — traçabilité (2026-08, jamais lue au runtime, EXCLUE du ZIP)

```
entrainement/
├── evaluation/                  # éval CANONIQUE du modèle déployé (tools/courbes_eval.py, repo training-models)
│   ├── metriques_eval.json      # SOURCE DE VÉRITÉ des seuils du model_card (seuil_f1max global + par classe)
│   ├── appariements.json        # cache d'appariements (re-rendu sans ré-inférence)
│   └── *.png                    # planches P/R/F1/PR
├── comparaison_<vs>/            # superpositions multi-modèles (courbes_eval)
├── metrics.csv (+ historiques + NOTE-metriques.md), hparams.yaml, tfevents, visualizations/
```

`thresholds.confidence_default` et `confidence_per_class` du model_card proviennent de
`entrainement/evaluation/metriques_eval.json` (validateur : |Δ| ≤ 0,05 avec le
`seuil_f1max` mesuré, sinon `thresholds.seuils_provenance` obligatoire pour justifier
l'écart). `dev/package_plugin.py` exclut `entrainement/` du ZIP de distribution.

### Statut honnête des fichiers jamais lus au runtime

`config.json`, `training_params.json` et `evaluation_results.json` ne sont lus QUE par
le validateur (cohérences) — aucun code de `src/` ne les ouvre. `evaluation_results.json`
est l'artefact legacy du notebook (appariement par IoU à seuil fixe 0,3, incomparable au
balayage de `metriques_eval.json`) : documentaire, jamais source de seuils, jamais réécrit.

## Doublons dans `classes.txt` — sous-classes RF-DETR

Les doublons sont **explicitement autorisés**. Le pipeline les fusionne par nom de classe (string equality) au moment de la génération des shapefiles ([conversion_shp.py:1276-1294](../src/pipeline/cv/conversion_shp.py)). Exemple : un modèle RF-DETR entraîné à distinguer 3 sous-types de `charbonniere` aura :

```
charbonniere
charbonniere
charbonniere
circular_depression
four
```

→ 5 lignes côté `classes.txt` (et 5 entrées côté `class_colors`, `model_card.classes`, etc.), mais **un seul shapefile `charbonniere.shp`** en sortie.

**Conséquence importante :** les accents et la casse comptent. `charbonniere` ≠ `charbonnière` côté Python `dict[name]`. Toute la chaîne (classes.txt racine, `weights/classes.txt`, `model_card`, règles de clustering) doit utiliser exactement la même forme. Le contrat impose **snake_case ASCII** comme forme canonique.

## Choix d'inférence divergents du training

Il est légitime d'inférer à une résolution différente de celle d'entraînement (`args.yaml.imgsz ≠ weights/best.json.resolution`), ou d'utiliser une fenêtre SAHI différente de `imgsz` (`sahi.slice_* ≠ imgsz`). Quand c'est le cas, **chaque divergence doit être documentée** dans `model_card.inference_choices` avec une raison explicite. Le validateur tolère la divergence si elle est documentée, sinon il émet un WARN.

## Schémas des fichiers

### `model_card.yaml`

```yaml
id: cratere_circulaire_2
display_name: "Cratères circulaires (Verdun, modèle 2)"
version: "2025-09"
status: production              # production | beta | deprecated | broken
description: |
  Détecte les cratères d'obus circulaires sur RVT LD à 0.5 m.
task: instance_segmentation     # object_detection | instance_segmentation | semantic_segmentation
architecture: RF-DETR-Seg-Large
variant: large
resolution_train: 504           # == weights/best.json.resolution
resolution_inference: 504       # == args.yaml.imgsz
preferred_rvt:
  type: LD
  params: { angular_res: 15, min_radius: 10, max_radius: 20, observer_h: 1.7, ve_factor: 1, save_as_8bit: true }
mnt:
  resolution: 0.5
  filter_expression: "Classification = 2 OR Classification = 6 OR ..."

# Suit l'ordre de classes.txt ligne par ligne. Les doublons sont autorisés.
classes:
  - id: 0
    name: cratere_obus          # snake_case ASCII (= classes.txt ligne 1)
    label_fr: "Cratère d'obus"  # accentué — affichage utilisateur
    color_index: 0              # index dans args.yaml.class_colors
    description: "Cratère d'obus circulaire de Première Guerre mondiale."
    # entity: cratere           # OPTIONNEL — id d'entité du catalogue si != name
                                # (la couverture UI repli sur name == entity.id ;
                                #  une entité hors entities_catalog.json = modèle
                                #  installé mais INVISIBLE — vérifié par le validateur)

thresholds:
  confidence_default: 0.3       # = seuil_f1max de entrainement/evaluation/metriques_eval.json
  min_area_m2: 0
  # confidence_per_class:       # OPTIONNEL — seuils F1-max PAR CLASSE (mesurés) ;
  #   chemin_creux: 0.15        # clés ⊆ classes.txt (validé) ; consommés par le
  #   talus_fosse: 0.30         # fallback Python ET le binaire externe (T1, 2026-08-31)
  # iou: 0.5                    # OPTIONNEL (alias iou_threshold) — jamais exposé UI
  # seuils_provenance: "..."    # traçabilité de la mesure (chemin + date) ; REQUIS
                                # si confidence_default s'écarte >0,05 de la mesure

# Cibles DÉRIVÉES : une sortie de clustering présentée comme entité cochable.
# Chaque output_class DOIT avoir sa règle args.yaml.clustering.output_class_name
# (sinon le plugin l'ignore en silence — validé depuis 2026-08-31).
# derived_targets:
#   - output_class: zone_crateres      # == une clustering.output_class_name
#     entity: regroupement_crateres    # id du catalogue
#     include_source: true             # sortie = zones + détections sources
#     output_label: Regroupements      # nom de la couche cluster (optionnel)
#     source_label: Cratères           # nom de la couche source (optionnel)

# Documente les divergences imgsz / SAHI vs training. Optionnel mais REQUIS si
# divergence. Depuis 2026-08-31 le validateur vérifie que `value` == la valeur
# RÉELLE d'args.yaml (une carte qui documente 350 pour un args.yaml à 140 = ERR).
inference_choices:
  - field: sahi.slice_width
    value: 140
    reason: "SAHI << imgsz (504) : objets ~5 px, densité > num_queries par grande fenêtre."

recommended_use: "RVT LD 0.5 m sur emprises Verdun ou contexte WWI."
known_limitations:
  - "Faux positifs sur trognes de pierre et puits anciens."
```

### `args.yaml`

```yaml
model: RF-DETR-Seg-Large        # parsé par detect_model_type / is_rfdetr_model
task: instance_segmentation     # NORMALISÉ
imgsz: 504

sahi:
  slice_width: 350              # peut différer de imgsz
  slice_height: 350
  overlap_ratio: 0.2

postprocess:
  merge_adjacent: true
  remove_overlaps: true

class_colors: [0]               # longueur == len(classes.txt) (doublons compris)

clustering:                     # optionnel
  - target_classes: ["cratere_obus"]      # ⊆ classes.txt (unique)
    min_confidence: 0.4
    min_confidence_extend: 0.3            # hystérésis ; >= min_confidence
    min_cluster_size: 40
    min_samples: 5
    eps_m: 40
    output_class_name: "zone_crateres"    # ∉ classes.txt (pas de collision)
    output_geometry: "convex_hull"        # convex_hull | concave_hull | bounding_box
    buffer_m: 10
    min_area_m2: 1000
    confidence_weight: 0.0
```

### `classes.txt`

- UTF-8 sans BOM, retour à la ligne LF.
- 1 nom par ligne, regex `^[a-z][a-z0-9_]*$`.
- Pas de ligne vide (y compris finale).
- Doublons autorisés (sous-classes).
- L'ordre est l'ordre de sortie du modèle (post-filtrage des prefixes ignorés).

### `training_params.json`

```json
{
  "description": "Paramètres utilisés pour générer les images d'entraînement",
  "model": {
    "architecture": "RF-DETR-Seg-Large",
    "variant": "large",
    "task": "instance_segmentation",
    "imgsz": 504
  },
  "mnt": {
    "resolution": 0.5,
    "filter_expression": "Classification = 2 OR ..."
  },
  "rvt": {
    "type": "LD",
    "params": { "angular_res": 15, "min_radius": 10, "max_radius": 20, "observer_h": 1.7, "ve_factor": 1, "save_as_8bit": true }
  },
  "detection": { "min_area_m2": 0 }
}
```

### `config.json`

```json
{
  "task": "instance_segmentation",
  "model": {
    "architecture": "RF-DETR-Seg-Large",
    "variant": "large",
    "resolution": 504,
    "num_classes": 1,
    "class_names": ["cratere_obus"]
  },
  "training": { "num_epochs": 30, "batch_size": 4 },
  "dataset": { "workspace": "...", "project": "...", "version": 1 }
}
```

### `weights/best.json`

```json
{
  "model_type": "rfdetr",
  "task": "instance_segmentation",
  "resolution": 504,
  "class_offset": 1,
  "num_classes": 1,
  "class_names": ["cratere_obus"],
  "source": "weights/best.pth"
}
```

`source` doit être **relatif au dossier modèle** (pas de `C:\Users\...`, `/Users/`, `/home/`).

`class_offset` : `1` par défaut pour RF-DETR (l'index 0 est background), `0` pour les modèles sans background.

### `evaluation_results.json`

```json
{
  "run_name": "cratere_circulaire_2",
  "task": "instance_segmentation",
  "test": {
    "num_images": 123,
    "global_metrics": { "mAP_50": 0.78, "mAP_50_95": 0.52, "precision": 0.81, "recall": 0.74, "f1": 0.77 },
    "per_class": { "cratere_obus": { "f1": 0.77, "ap_50_95": 0.52 } }
  }
}
```

## Cohérences inter-fichiers

### Strictes (erreur si violées)

| Cohérence | Vérifié |
|-----------|---------|
| `classes.txt` (lignes) == `config.json.model.class_names` == `weights/best.json.class_names` == `[c.name for c in model_card.classes]` | taille + ordre + valeurs (doublons compris) |
| `len(classes.txt)` == `len(args.yaml.class_colors)` == `weights/best.json.num_classes` == `config.json.model.num_classes` | nombres égaux |
| `model_card.resolution_train` == `weights/best.json.resolution` == `config.json.model.resolution` == `training_params.json.model.imgsz` | entiers égaux |
| `args.yaml.task` == `weights/best.json.task` == `config.json.task` == `training_params.json.model.task` == `model_card.task` | string égal |
| `args.yaml.task` ∈ `{object_detection, instance_segmentation, semantic_segmentation}` | normalisé |
| `args.yaml.clustering[].target_classes` ⊆ `set(classes.txt)` | inclusion |
| `args.yaml.clustering[].output_class_name` ∉ `set(classes.txt)` | exclusion |
| `weights/best.json.source` | pas d'absolu local |
| Si `weights/classes.txt` existe → byte-identique à racine | hash égal |
| Toutes lignes de `classes.txt` matchent `^[a-z][a-z0-9_]*$`, pas de ligne vide | regex + non-vide |

### Relâchées (info, jamais erreur)

| Divergence | Doit être documentée dans `model_card.inference_choices` |
|------------|---------------------------------------------------------|
| `args.yaml.imgsz != weights/best.json.resolution` | oui (sinon WARN) |
| `args.yaml.sahi.slice_* != args.yaml.imgsz` | oui (sinon WARN) |
| `model_card.resolution_inference != model_card.resolution_train` | doit matcher `args.yaml.imgsz` |

## Validation

```bash
# Tous les modèles
python scripts/validate_models_metadata.py

# Un modèle spécifique
python scripts/validate_models_metadata.py data/models/cratere_circulaire_2

# Mode strict : warnings deviennent erreurs
python scripts/validate_models_metadata.py --strict
```

```bash
# Tests pytest associés
python run_tests.py unit -k test_models_metadata
python run_tests.py unit -k test_models_connectivity
```

## Workflow d'ajout d'un nouveau modèle

1. Entraîner avec `data/models/rfdetr_unified_pipeline.ipynb`.
2. Le notebook produit `runs/training/<RUN_ID>/package/` avec les 5 fichiers texte du contrat + `weights/best.pth`.
3. Copier `package/` → `data/models/<RUN_ID>/`.
4. Exporter l'ONNX :
   ```bash
   python dev/runner_onnx/export_to_onnx.py \
       --model data/models/<RUN_ID>/weights/best.pth \
       --output data/models/<RUN_ID>/weights/best.onnx
   ```
5. Vérifier la conformité :
   ```bash
   python scripts/validate_models_metadata.py data/models/<RUN_ID>
   python run_tests.py unit -k test_models
   ```

## Code consommateur (référence)

- [model_profile.py](../src/pipeline/cv/model_profile.py) — lit `args.yaml`, `classes.txt`, `weights/best.json` et expose `ModelProfile.load(model_dir)`.
- [model_config.py](../src/pipeline/cv/model_config.py) — `load_sahi_config_from_model`, `load_clustering_config_from_model`, `load_postprocess_config_from_model`, `is_rfdetr_model`.
- [class_utils.py](../src/pipeline/cv/class_utils.py) — `load_class_names_from_model` (cascade `classes.txt` → `class_names.txt` → `classes.json`).
- [conversion_shp.py:1276-1294](../src/pipeline/cv/conversion_shp.py) — merging des sous-classes par string.
- [clustering.py:285-288](../src/pipeline/cv/clustering.py) — matching `target_classes` par string.
- [computer_vision_onnx.py:287-290](../src/pipeline/cv/computer_vision_onnx.py) — application du `class_offset`.
- [export_to_onnx.py](../dev/runner_onnx/export_to_onnx.py) — produit `weights/best.onnx` + `weights/best.json` (sidecar).
