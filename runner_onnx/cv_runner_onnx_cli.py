"""
CLI du runner ONNX unifié pour l'inférence CV.
Ce runner utilise ONNX Runtime et supporte les modèles YOLO et RF-DETR exportés en ONNX.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Forcer l'encodage UTF-8 sur stdout/stderr pour Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _try_import_version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        v = getattr(mod, "__version__", None)
        return str(v) if v is not None else "unknown"
    except Exception as e:
        return f"MISSING ({e})"


def _try_import_ok(mod_name: str) -> str:
    try:
        __import__(mod_name)
        return "OK"
    except Exception as e:
        return f"MISSING ({e})"


def _print(msg: str) -> None:
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()  # Flush immédiatement pour le streaming en temps réel


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def _plugin_root() -> Path:
    """Retourne la racine du plugin."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def _ensure_src_on_sys_path() -> None:
    root = _resource_root()
    plugin_root = _plugin_root()

    candidates = [
        root / "src",
        plugin_root / "src",
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                return
        except Exception:
            continue

    dev_src = plugin_root / "src"
    if dev_src.exists() and dev_src.is_dir() and str(dev_src) not in sys.path:
        sys.path.insert(0, str(dev_src))


def _resolve_model_path(cv_config: Dict[str, Any]) -> Path:
    """Résout le chemin vers le modèle ONNX."""
    selected_model = str((cv_config or {}).get("selected_model", "")).strip()
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    model_path = Path(selected_model)
    
    # Si c'est un chemin vers un fichier .pt, chercher le .onnx correspondant
    if model_path.suffix.lower() == ".pt":
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            return onnx_path
        # Sinon, erreur explicite
        raise FileNotFoundError(
            f"Modèle ONNX non trouvé: {onnx_path}\n"
            f"Le fichier .pt existe mais pas le .onnx. "
            f"Exportez le modèle avec: python runner_onnx/export_to_onnx.py --model {model_path} --output {onnx_path}"
        )
    
    # Si c'est un chemin absolu vers un fichier ONNX existant
    if model_path.suffix.lower() == ".onnx" and model_path.exists() and model_path.is_file():
        return model_path
    
    # Chercher dans le dossier models
    models_dir = Path((cv_config or {}).get("models_dir", "models"))
    
    # Essayer différents chemins
    candidates = [
        models_dir / selected_model / "weights" / "best.onnx",
        models_dir / selected_model / "best.onnx",
        models_dir / f"{selected_model}.onnx",
        models_dir / selected_model,
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".onnx":
            return candidate
    
    raise FileNotFoundError(
        f"Modèle ONNX non trouvé pour: {selected_model}\n"
        f"Chemins testés: {[str(c) for c in candidates]}"
    )


def _iter_jpgs(jpg_dir: Path, single_jpg: Optional[Path], scan_all: bool) -> list[Path]:
    if single_jpg is not None:
        return [single_jpg]
    if scan_all:
        return sorted(jpg_dir.glob("*.jpg"))
    return sorted(jpg_dir.glob("*.jpg"))[:1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[cv_runner_onnx][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg_path = Path(args.config)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    jpg_dir = Path(payload["jpg_dir"]).resolve()
    target_rvt = str(payload.get("target_rvt", "LD"))
    rvt_base_dir_raw = payload.get("rvt_base_dir")
    rvt_base_dir = Path(rvt_base_dir_raw).resolve() if rvt_base_dir_raw else jpg_dir.parent

    cv_config: Dict[str, Any] = payload.get("cv_config") or {}
    tif_transform_data: Dict[str, Any] = payload.get("tif_transform_data") or {}
    run_shapefile_dedup = bool(payload.get("run_shapefile_dedup", True))

    single_jpg_raw = payload.get("single_jpg")
    single_jpg = Path(single_jpg_raw).resolve() if single_jpg_raw else None

    enabled = bool(cv_config.get("enabled", False))
    if not enabled:
        _print("CV disabled")
        return 0

    # Afficher les versions des dépendances
    _print(f"dep:onnxruntime={_try_import_version('onnxruntime')}")
    _print(f"dep:numpy={_try_import_version('numpy')}")
    _print(f"dep:pillow={_try_import_version('PIL')}")
    _print(f"dep:sahi_lite=internal")

    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    # Vérifier les dépendances critiques
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        _print(f"ERROR: onnxruntime missing ({e})")
        return 2

    try:
        import numpy  # noqa: F401
    except Exception as e:
        _print(f"ERROR: numpy missing ({e})")
        return 2

    annotated_output_dir = rvt_base_dir / "annotated_images"
    shapefile_output_dir = rvt_base_dir / "shapefiles"

    if generate_annotated_images:
        annotated_output_dir.mkdir(parents=True, exist_ok=True)
    if generate_shapefiles:
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_src_on_sys_path()

    from pipeline.cv import computer_vision_onnx as cv_mod
    from pipeline.cv.cv_output import get_detection_output_path

    shp_mod = None
    if generate_shapefiles and run_shapefile_dedup:
        try:
            from pipeline.cv import conversion_shp as shp_mod  # type: ignore
        except Exception as e:
            _print(f"WARN: shapefile deps not available ({e})")

    # Résoudre le chemin du modèle
    try:
        model_path = _resolve_model_path(cv_config)
    except Exception as e:
        _print(f"ERROR: {e}")
        return 1

    _print(f"model_type=onnx")
    _print(f"model_path={model_path}")
    
    # Charger les noms de classes depuis le dossier du modèle
    class_names = None
    class_colors = None
    try:
        from pipeline.cv.class_utils import load_class_names_from_model, load_class_colors_from_model
        class_names = load_class_names_from_model(model_path)
        class_colors = load_class_colors_from_model(model_path)
        if class_names:
            _print(f"class_names={class_names}")
        _print(f"class_colors_loaded={class_colors}")  # Toujours afficher, même si None
    except Exception as e:
        _print(f"WARN: impossible de charger les noms/couleurs de classes: {e}")
    
    # Créer le fichier légende dans annotated_images
    if generate_annotated_images and class_names and annotated_output_dir:
        try:
            from pipeline.cv.cv_output import save_legend_file
            save_legend_file(str(annotated_output_dir), class_names, class_colors)
            _print(f"legend_created={annotated_output_dir}")
        except Exception as e:
            _print(f"WARN: impossible de créer la légende: {e}")

    confidence_threshold = float(cv_config.get("confidence_threshold", 0.3))
    iou_threshold = float(cv_config.get("iou_threshold", 0.5))

    sahi_cfg = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    slice_height = int(sahi_cfg.get("slice_height", 750))
    slice_width = int(sahi_cfg.get("slice_width", 750))
    overlap_ratio = float(sahi_cfg.get("overlap_ratio", 0.2))

    scan_all = bool(cv_config.get("scan_all", False))
    jpg_files = _iter_jpgs(jpg_dir=jpg_dir, single_jpg=single_jpg, scan_all=scan_all)
    jpg_files = [p for p in jpg_files if p.exists()]

    _print(f"jpg_dir={jpg_dir}")
    _print(f"target_rvt={target_rvt}")
    _print(f"rvt_base_dir={rvt_base_dir}")
    
    total_images = len(jpg_files)
    _print(f"images={total_images}")

    success_count = 0
    processed_count = 0
    skipped_already_processed = 0
    total_detections = 0

    for idx, jpg_file in enumerate(jpg_files, 1):
        image_name = jpg_file.stem
        labels_txt = jpg_dir / f"{image_name}.txt"
        labels_json = jpg_dir / f"{image_name}.json"

        detection_output_path = get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if generate_annotated_images else None,
        )
        annotated_img = Path(detection_output_path)

        if annotated_img.exists() or labels_txt.exists() or labels_json.exists():
            skipped_already_processed += 1
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=skipped")
            continue

        processed_count += 1
        _print(f"progress={idx}/{total_images} image={jpg_file.name} status=processing")
        
        try:
            ok, num_dets = cv_mod.run_onnx_inference(
                image_path=str(jpg_file),
                model_path=str(model_path),
                output_path=str(detection_output_path),
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
                generate_annotated_images=generate_annotated_images,
                annotated_output_dir=str(annotated_output_dir) if generate_annotated_images else None,
                jpg_folder_path=str(jpg_dir),
                return_count=True,
                class_names=class_names,
                class_colors=class_colors,
            )
        except Exception as e:
            ok = False
            num_dets = 0
            _print(f"ERROR: run_onnx_inference failed for {jpg_file.name}: {e}")
            import traceback
            traceback.print_exc()

        if ok:
            success_count += 1
            total_detections += num_dets
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=done detections={num_dets}")
        else:
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=done detections=0")

    _print(f"summary: success={success_count} processed={processed_count} skipped={skipped_already_processed} total_detections={total_detections}")

    # Génération des shapefiles
    if generate_shapefiles and run_shapefile_dedup and shp_mod is not None:
        out_shp = shapefile_output_dir / f"detections_{target_rvt}.shp"
        create_fn = getattr(shp_mod, "create_shapefile_from_detections", None)
        if callable(create_fn):
            try:
                create_fn(
                    labels_dir=str(jpg_dir),
                    output_shapefile=str(out_shp),
                    tif_transform_data=tif_transform_data,
                    crs="EPSG:2154",
                    temp_dir=None,
                    class_names=class_names,
                    class_colors=class_colors,
                )
            except Exception as e:
                _print(f"WARN: shapefile creation failed: {e}")

        dedup_fn = getattr(shp_mod, "deduplicate_shapefiles_final", None)
        if callable(dedup_fn):
            shp_paths = [str(p) for p in shapefile_output_dir.glob("*.shp")]
            if shp_paths:
                try:
                    dedup_fn(
                        labels_dir=str(jpg_dir),
                        shapefile_paths=shp_paths,
                        iou_threshold=0.1,
                        crs="EPSG:2154",
                    )
                except Exception as e:
                    _print(f"WARN: shapefile dedup failed: {e}")

        _print(f"shapefiles_dir={shapefile_output_dir}")
        _print(f"detections_shp={out_shp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
