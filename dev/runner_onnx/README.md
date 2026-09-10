# Runner ONNX

Ce dossier contient le runner unifié pour l'inférence CV avec des modèles ONNX.

## Avantages

- **Un seul binaire très léger** (~100-150 MB CPU, ~300 MB GPU)
- **Pas de dépendances lourdes** (pas de PyTorch, ultralytics, rfdetr, transformers, timm)
- **sahi_lite intégré** : slicing numpy-only sans dépendance torch
- **Runtime ONNX optimisé** pour l'inférence
- **Support CPU et GPU** via onnxruntime

## Structure

```
runner_onnx/
├── README.md                   # Cette documentation
├── build.py                    # Script de compilation (lit dev/requirements/build.txt)
├── export_to_onnx.py           # Script d'export des modèles
├── cv_runner_onnx_cli.py       # Point d'entrée du runner
└── cv_runner_onnx.spec         # Spec PyInstaller
```

## Politique de recompilation (règle 2026-08-31)

L'exe PyInstaller **fige un instantané de `src/`** (`cv_runner_onnx.spec` :
`datas=[(src, 'src')]`) : tout commit touchant `src/pipeline/cv/**` ou
`cv_runner_onnx_cli.py` rend le binaire livré périmé — c'est arrivé 2 mois et
demi sans signal (binaire du 16/06, 4 commits CV morts, audit 2026-08-31).

- **Quand recompiler** : dès que `tests/unit/test_runner_binary_fresh.py` est
  rouge (il compare le sha256 des sources au `build_info.json` posé à côté de
  l'exe par `build.py`).
- **Comment** : `python dev/runner_onnx/build.py` (crée/réutilise `.venv_onnx`,
  installe `dev/requirements/build.txt`, PyInstaller via le spec, copie l'exe +
  `build_info.json` vers `data/third_party/cv_runner_onnx/windows/`).
- **Vérifier** : test de fraîcheur vert + smoke run de l'exe sur une image de
  `dev/image_test/` avec un modèle installé.
- Le gel des versions du venv de build vit dans `dev/requirements/build.lock`
  (régénéré par `pip freeze` après toute évolution de build.txt).

## Export des modèles

Avant d'utiliser le runner, vous devez exporter vos modèles au format ONNX.

**Dépendances** : `pip install -r dev/requirements/export.txt` (depuis la racine du plugin).

```bash
cd dev/runner_onnx

# Auto-détection du type de modèle (YOLO, RF-DETR, SegFormer)
python export_to_onnx.py --model path/to/best.pt --output path/to/model.onnx

# Options supplémentaires
python export_to_onnx.py --model best.pt --output model.onnx --imgsz 640 --simplify
```

> Voir la section **Tâche 2** du README principal pour le détail des options.

## Compilation du runner

**Dépendances** : `pip install -r dev/requirements/build.txt` (depuis la racine du plugin).

```bash
cd dev/runner_onnx
python build.py              # Compile le runner CPU (~100-150 MB)
python build.py --gpu        # Compile le runner GPU (~300 MB)
python build.py --clean      # Nettoie les builds
```

Le script `build.py` lit ses dépendances depuis `dev/requirements/build.txt` (source unique).
Le binaire compilé est copié automatiquement dans `third_party/cv_runner_onnx/<os>/`.

## Dépendances du runner compilé

Le binaire final n'inclut que des bibliothèques légères (pas de PyTorch) :
- `onnxruntime` — Runtime ONNX (~50 MB)
- `sahi_lite` — Slicing numpy-only intégré
- `pillow`, `numpy`, `opencv-python-headless` — Traitement d'images
- `shapely`, `geopandas`, `fiona` — Génération shapefiles

## Configuration

Dans `config.json`, le modèle doit pointer vers un fichier `.onnx` :

```json
{
  "cv": {
    "enabled": true,
    "selected_model": "path/to/model.onnx"
  }
}
```
