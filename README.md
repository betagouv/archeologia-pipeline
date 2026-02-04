# Archeolog'IA pipeline (Plugin QGIS)

Plugin QGIS pour exécuter un pipeline de traitement LiDAR et produire des rasters de type MNT / densité / indices RVT, avec une étape optionnelle de détection par *computer vision*.

- Nom du plugin : **Archeolog'IA pipeline**
- Version : **0.1.0**
- QGIS minimum : **3.0**

## Fonctionnalités

- Génération de produits raster :
  - **MNT**
  - **Densité**
  - Indices **RVT** (via *Processing*) : **M-HS**, **SVF**, **SLO**, **LD**, **VAT**
- Export optionnel en **JPG + world file (JGW)** pour certains produits.
- (Optionnel) Détection par computer vision à partir des JPG produits (via runner externe ou dépendances Python).
- Option (configurable) : génération de **pyramides / overviews** GDAL pour les GeoTIFF de sortie.

## Modes de données supportés

Le pipeline peut être lancé dans plusieurs modes (selon l’UI/config) :

- `ign_laz` : téléchargement/consommation de tuiles LAZ depuis une liste IGN.
- `local_laz` : consommation de tuiles LAZ/LAS déjà présentes localement.
- `existing_mnt` : calcul d’indices RVT à partir d’un MNT existant.
- `existing_rvt` : opérations sur RVT existants (selon fonctionnalités disponibles).

## Pré-requis

Le plugin s’exécute dans QGIS et s’appuie sur des outils externes. Un contrôle est effectué au lancement via le **préflight**.

### Dépendances QGIS

- **QGIS 3.x**
- Module **Processing** (fourni avec QGIS)
- Les algorithmes RVT accessibles via Processing (selon installation QGIS)

### Outils externes (CLI)

- **PDAL** (`pdal`) requis pour les modes `ign_laz` et `local_laz`
- **GDAL** utilitaires :
  - `gdalwarp` requis pour `ign_laz`, `local_laz`, `existing_mnt`
  - `gdal_translate` requis pour `existing_mnt` / `existing_rvt` (et utilisé dans d’autres modes)
  - `gdaladdo` optionnel (pyramides / overviews) — si absent, la génération est ignorée

### Computer vision (optionnel)

Deux options :

- **Runner ONNX externe** (recommandé) : `third_party/cv_runner_onnx/windows/cv_runner_onnx.exe` (Windows) / `third_party/cv_runner_onnx/linux/cv_runner_onnx` (Linux)
- Ou dépendances Python dans l'environnement de QGIS (si pas de runner externe) :
  - `onnxruntime`, `sahi`, `PIL` (Pillow)
  - `geopandas` (optionnel)

**Note** : Les modèles doivent être exportés au format ONNX avant utilisation (voir section dédiée).

## Installation

### Installation dans QGIS (utilisateur)

1. Ouvrir **QGIS**.
2. Aller dans :
   - `Profils utilisateurs` → `Ouvrir le dossier du profil actif`
3. Ouvrir le dossier :
   - `python/plugins`
4. Dézipper le plugin : on obtient le dossier :
   - `archeologia-pipeline-lidar-processing`
5. Copier le dossier `archeologia-pipeline-lidar-processing` dans `python/plugins`.
6. Fermer puis relancer QGIS.
7. Activer le plugin :
   - `Extensions` → `Installer/Gérer les extensions…` → rechercher **Archeolog'IA pipeline** → activer.

### Où se trouve le dossier des plugins

Sous Windows (profil par défaut) :

```text
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\
```

### Dépendances à avoir dans QGIS

Le plugin exécute un **préflight** (contrôle des dépendances) au lancement.

- **Processing** : doit être disponible (dans QGIS : `Traitement` → `Boîte à outils`).
- **Algorithmes RVT via Processing** : nécessaires si tu actives des produits RVT (M-HS/SVF/SLO/LD/VAT).

Si un élément est manquant, le préflight affichera une erreur et empêchera le lancement.

### Dépendances externes (CLI)

Certaines étapes reposent sur des exécutables dans le `PATH` :

- `pdal` requis pour `ign_laz` / `local_laz`
- `gdalwarp` requis pour `ign_laz` / `local_laz` / `existing_mnt`
- `gdal_translate` requis pour `existing_mnt` / `existing_rvt`
- `gdaladdo` optionnel (pyramides / overviews). Si absent, la génération de pyramides est ignorée.

## Computer vision : runner ONNX + modèles

### Activer la computer vision

La computer vision est optionnelle. Quand elle est activée, le pipeline peut lancer une étape de détection à partir des images (JPG) exportées.

Le plugin utilise un **runner ONNX unifié** qui supporte les modèles YOLO et RF-DETR exportés au format ONNX.

### Export des modèles vers ONNX

Avant d'utiliser le runner, vous devez exporter vos modèles PyTorch (.pt) vers le format ONNX.

#### 1. Créer un environnement virtuel dédié à l'export

```bash
cd <racine_du_plugin>
python -m venv .venv_export

# Windows
.venv_export\Scripts\activate

# Linux/Mac
source .venv_export/bin/activate
```

#### 2. Installer les dépendances d'export

**Pour YOLO uniquement :**
```bash
pip install ultralytics onnx onnxsim
```

**Pour RF-DETR uniquement :**
```bash
pip install rfdetr torch onnx onnxsim pyyaml
```

**Pour les deux (complet) :**
```bash
pip install ultralytics rfdetr torch onnx onnxsim pyyaml
```

#### 3. Exporter le modèle

```bash
# Exporter un modèle YOLO
python runner_onnx\export_to_onnx.py --model models\mon_modele\weights\best.pt --output models\mon_modele\weights\best.onnx

# Exporter un modèle RF-DETR
python runner_onnx\export_to_onnx.py --model models\mon_modele_rfdetr\weights\best.pt --output models\mon_modele_rfdetr\weights\best.onnx

# Options supplémentaires
python runner_onnx\export_to_onnx.py --model best.pt --output model.onnx --imgsz 640 --simplify
```

Le script détecte automatiquement le type de modèle (YOLO ou RF-DETR).

### Création du runner ONNX (Windows)

Objectif : produire un exécutable :

```text
third_party/cv_runner_onnx/windows/cv_runner_onnx.exe
```

#### Compilation automatique

Le script `runner_onnx/build.py` automatise la création du runner :

```bash
cd runner_onnx

# Compiler le runner (CPU)
python build.py

# Compiler le runner avec support GPU
python build.py --gpu

# Nettoyer les builds précédents
python build.py --clean
```

Le script :
1. Crée un environnement virtuel isolé (`.venv_onnx`)
2. Installe les dépendances nécessaires
3. Compile le runner avec PyInstaller
4. Copie le binaire vers `third_party/cv_runner_onnx/windows/`

**Taille du binaire** : ~1 GB (inclut les dépendances SAHI et ONNX Runtime)

#### Compilation manuelle (alternative)

```bash
cd runner_onnx
python -m venv .venv_onnx
.venv_onnx\Scripts\activate
pip install pyinstaller onnxruntime sahi pillow numpy pyyaml shapely geopandas fiona
python -m PyInstaller --clean cv_runner_onnx.spec
copy dist\cv_runner_onnx.exe ..\third_party\cv_runner_onnx\windows\
```

### Modèles (dossier `models/`)

Structure attendue (1 modèle = 1 dossier) :

```text
models/
  <nom_du_modele>/
    args.yaml
    classes.txt
    weights/
      best.pt      # Modèle PyTorch original
      best.onnx    # Modèle exporté (requis pour le runner)
```

Le fichier `classes.txt` doit contenir **un nom de classe par ligne** :

```text
nomclasse1
nomclasse2
...
```

### Configuration

Dans `config.json`, le modèle doit pointer vers le fichier `.onnx` :

```json
{
  "cv": {
    "enabled": true,
    "selected_model": "models/mon_modele/weights/best.onnx"
  }
}
```

Ou vers le dossier du modèle (le runner cherchera automatiquement `best.onnx`) :

```json
{
  "cv": {
    "enabled": true,
    "selected_model": "mon_modele"
  }
}
```

## Utilisation

1. Ouvrir le plugin : menu **Archeolog'IA pipeline**.
2. Choisir le **mode** de données (IGN / local / MNT existant / RVT existant).
3. Configurer :
   - le répertoire de sortie
   - les paramètres de résolution
   - les produits à générer
   - (optionnel) la génération de pyramides
4. Lancer le pipeline.

## Configuration (`config.json`)

Le plugin persiste sa configuration dans un fichier `config.json` à la racine du plugin.

### Structure (extraits)

- `app.files.output_dir` : dossier de sortie
- `app.files.data_mode` : mode (`ign_laz`, `local_laz`, `existing_mnt`, `existing_rvt`)
- `app.files.input_file` : fichier de liste (IGN ou liste locale selon le mode)
- `processing.mnt_resolution` : résolution du MNT (m)
- `processing.density_resolution` : résolution densité (m)
- `processing.tile_overlap` : recouvrement inter-tuiles (m)
- `processing.filter_expression` : filtre de classes (expression PDAL)
- `processing.products` : activation des produits (MNT/DENSITE/M_HS/SVF/SLO/LD/VAT)
- `processing.output_formats.jpg` : exports JPG par produit
- `processing.pyramids` : génération d’overviews GDAL

Exemple pyramides :

```json
{
  "processing": {
    "pyramids": {
      "enabled": true,
      "levels": [2, 4, 8, 16, 32, 64]
    }
  }
}
```

## Sorties

Les sorties sont écrites dans le dossier `output_dir` configuré.

Selon les produits et options, la structure typique est :

- `results/`
  - `MNT/…/tif/*.tif`
  - `RVT/<PRODUIT>/tif/*.tif`
  - `RVT/<PRODUIT>/jpg/*.jpg` (+ `*.jgw` si activé)

Les GeoTIFF peuvent contenir des **overviews** si l’option pyramides est activée et si `gdaladdo` est disponible.

## Développement

- Point d’entrée plugin : `main.py` (classe `ArcheologiaPipelinePlugin`)
- UI : `src/ui/main_dialog.py`
- Pipeline : `src/pipeline/`
  - prérequis : `src/pipeline/preflight.py`

## Git : Talisman (pre-push)

Le dépôt inclut un hook `pre-push` basé sur **Talisman** pour éviter de pousser des secrets (tokens, clés, etc.).

### Installation de Talisman

Installe `talisman` et assure-toi qu’il est disponible dans le `PATH`.

### Activer les hooks du dépôt

Les hooks Git ne sont pas versionnables directement dans `.git/hooks/`. À la place, ce dépôt fournit un dossier `.githooks/`.

À exécuter **à la racine du dépôt** :

```bash
git config core.hooksPath .githooks
```

Ensuite, un `git push` déclenchera automatiquement Talisman et pourra bloquer le push si un secret est détecté.

## Dépannage

- **Préflight KO** : vérifier que `pdal`, `gdalwarp`, `gdal_translate` sont accessibles dans le `PATH`.
- **Pyramides absentes** : vérifier la présence de `gdaladdo` et que l’option pyramides est activée.
- **RVT indisponible** : vérifier que les algorithmes RVT sont disponibles via QGIS Processing.
- **Computer vision** :
  - soit fournir le runner externe dans `third_party/cv_runner_onnx/...`
  - soit installer les dépendances Python (`onnxruntime`, `sahi`) dans l’environnement QGIS

## Architecture

### Diagramme de flux

```mermaid
flowchart TD
    subgraph QGIS["QGIS Application"]
        A[QGIS démarre] --> B[Charge les plugins]
        B --> C["ArcheologiaPipelinePlugin"]
        C --> D["initGui()"]
    end

    subgraph UI["Interface Utilisateur"]
        E["Clic sur plugin"] --> F["MainDialog"]
        F --> G{"Action ?"}
        G -->|"Run"| H["_on_run_clicked()"]
        G -->|"Save"| I["_save_from_widgets()"]
        G -->|"Cancel"| J["_cancel_event.set()"]
    end

    subgraph Pipeline["Pipeline Execution"]
        H --> K["build_run_context()"]
        K --> L["PipelineController.run()"]
        L --> M["run_preflight()"]
        M --> N{"Mode ?"}
        N -->|"ign_laz / local_laz"| O["IgnOrLocalRunner"]
        N -->|"existing_mnt"| P["ExistingMntRunner"]
        N -->|"existing_rvt"| Q["ExistingRvtRunner"]
    end

    subgraph IgnLocal["Mode ign_laz / local_laz"]
        O --> R1{"ign_laz ?"}
        R1 -->|"Oui"| R2["download_ign_dalles()"]
        R1 -->|"Non"| R3["run_local_laz()"]
        R2 --> R4["prepare_merged_tiles()"]
        R3 --> R4
        R4 --> R5["Boucle par dalle"]
        R5 --> R6["create_terrain_model()"]
        R6 --> R7{"DENSITE ?"}
        R7 -->|"Oui"| R8["create_density_map()"]
        R7 -->|"Non"| R9["create_visualization_products()"]
        R8 --> R9
        R9 --> R10["crop_final_products()"]
        R10 --> R11["copy_final_products_to_results()"]
        R11 --> R12{"CV enabled ?"}
        R12 -->|"Oui"| R13["process_single_jpg()"]
        R12 -->|"Non"| R14["Dalle suivante"]
        R13 --> R14
        R14 --> R5
    end

    subgraph ExistingMnt["Mode existing_mnt"]
        P --> P1["run_existing_mnt()"]
        P1 --> P2{"CV enabled ?"}
        P2 -->|"Oui"| P3["run_existing_rvt()"]
        P2 -->|"Non"| P4["build_vrt_index()"]
        P3 --> P4
    end

    subgraph ExistingRvt["Mode existing_rvt"]
        Q --> Q1["run_existing_rvt()"]
        Q1 --> Q2["build_vrt_index()"]
    end

    subgraph CV["Computer Vision"]
        R13 --> CV1["run_cv_on_folder()"]
        CV1 --> CV2{"Runner externe ?"}
        CV2 -->|"Oui"| CV3["cv_runner_onnx.exe"]
        CV2 -->|"Non"| CV4["Python fallback"]
        CV3 --> CV5["Inférence ONNX"]
        CV4 --> CV5
        CV5 --> CV6["finalize() → shapefiles"]
        CV6 --> CV7["deduplicate_cv_shapefiles_final()"]
    end

    subgraph Finalize["Finalisation"]
        R14 --> F1["build_vrt_index()"]
        P4 --> F1
        Q2 --> F1
        F1 --> F2["load_layers() → QGIS"]
    end
```

### Structure des fichiers

```text
src/
├── app/                          # Orchestration pipeline
│   ├── cancel_token.py           # Encapsule threading.Event
│   ├── pipeline_controller.py    # Orchestre preflight + dispatch
│   ├── progress_reporter.py      # Protocol pour reporting
│   ├── qt_progress_reporter.py   # Implémentation Qt
│   ├── run_context.py            # Dataclass config pipeline
│   ├── runners/
│   │   ├── base.py               # ModeRunner Protocol
│   │   ├── registry.py           # get_runner(mode)
│   │   ├── ign_local_runner.py   # ign_laz + local_laz
│   │   ├── existing_mnt_runner.py
│   │   └── existing_rvt_runner.py
│   └── services/
│       └── cv_service.py         # ComputerVisionService
│
├── pipeline/                     # Logique métier
│   ├── cv/                       # Computer vision
│   ├── ign/                      # Téléchargement + produits
│   ├── modes/                    # Modes existing_mnt, existing_rvt
│   └── preflight.py              # Vérification dépendances
│
└── ui/
    └── main_dialog.py            # Interface Qt
```

## Tests

Le projet utilise **pytest** pour les tests unitaires et d'intégration.

```bash
# Installer pytest
pip install pytest

# Exécuter tous les tests
python -m pytest tests/ -v

# Tests unitaires uniquement
python -m pytest tests/unit/ -v

# Tests d'intégration uniquement
python -m pytest tests/integration/ -v
```

## Licence

Le dépôt contient un fichier `LICENSE.txt` (MIT).
