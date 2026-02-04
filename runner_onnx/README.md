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
├── build.py                    # Script de compilation
├── export_to_onnx.py           # Script d'export des modèles
├── cv_runner_onnx_cli.py       # Point d'entrée du runner
└── cv_runner_onnx.spec         # Spec PyInstaller
```

## Export des modèles

Avant d'utiliser le runner, vous devez exporter vos modèles au format ONNX :

```bash
# Exporter un modèle YOLO
python export_to_onnx.py --model path/to/yolo/best.pt --output path/to/output.onnx

# Exporter un modèle RF-DETR
python export_to_onnx.py --model path/to/rfdetr/best.pt --output path/to/output.onnx

# Options supplémentaires
python export_to_onnx.py --model best.pt --output model.onnx --imgsz 640 --simplify
```

**Note**: L'export nécessite les dépendances complètes (ultralytics pour YOLO, rfdetr pour RF-DETR).

## Compilation du runner

```bash
cd runner_onnx
python build.py              # Compile le runner CPU (~100-150 MB)
python build.py --gpu        # Compile le runner GPU (~300 MB)
python build.py --clean      # Nettoie les builds
```

## Dépendances du runner

Le runner compilé n'inclut que :
- `onnxruntime` : Runtime ONNX (~50 MB)
- `sahi_lite` : Slicing numpy-only intégré (pas de torch!)
- `pillow`, `numpy`, `opencv-python-headless` : Traitement d'images
- `shapely`, `geopandas`, `fiona` : Génération shapefiles

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
