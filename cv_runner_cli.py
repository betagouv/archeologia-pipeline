from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


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
    sys.stdout.flush()


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent

def _ensure_src_on_sys_path() -> None:
    root = _resource_root()

    candidates = [
        root / "src",
        root / "archeologia-pipeline-lidar-processing" / "src",
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                return
        except Exception:
            continue

    # Non-frozen dev mode: repo root contains src/
    dev_src = Path(__file__).resolve().parent / "src"
    if dev_src.exists() and dev_src.is_dir() and str(dev_src) not in sys.path:
        sys.path.insert(0, str(dev_src))


def _resolve_model_paths(cv_config: Dict[str, Any]) -> tuple[Path, Path]:
    selected_model = str((cv_config or {}).get("selected_model", "")).strip()
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    models_dir = Path((cv_config or {}).get("models_dir", "models"))

    model_path = Path(selected_model)
    if model_path.exists() and model_path.is_file():
        weights_path = model_path
        args_path = model_path.parent.parent / "args.yaml"
    else:
        model_dir = models_dir / selected_model
        weights_path = model_dir / "weights" / "best.pt"
        args_path = model_dir / "args.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids du modèle non trouvé: {weights_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"Fichier de configuration du modèle non trouvé: {args_path}")

    return weights_path, args_path


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
        format="[cv_runner][%(levelname)s] %(message)s",
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

    _print(f"dep:ultralytics={_try_import_version('ultralytics')}")
    _print(f"dep:sahi={_try_import_version('sahi')}")
    _print(f"dep:torch={_try_import_version('torch')}")
    _print(f"dep:torchvision={_try_import_version('torchvision')}")
    _print(f"dep:sahi.auto_model={_try_import_ok('sahi.auto_model')}")
    _print(f"dep:sahi.predict={_try_import_ok('sahi.predict')}")

    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    # Option A: le runner doit au minimum pouvoir faire l'inférence.
    # Si SAHI/Ultralytics sont absents, on échoue explicitement (sinon on obtient juste success=0).
    try:
        import ultralytics  # noqa: F401
    except Exception as e:
        _print(f"ERROR: ultralytics missing in cv_runner environment ({e})")
        return 2
    try:
        import sahi  # noqa: F401
    except Exception as e:
        _print(f"ERROR: sahi missing in cv_runner environment ({e})")
        return 2

    # Best-effort import of common SAHI submodules (helps runtime diagnostics, and also
    # makes it easier to force PyInstaller to collect them with hidden-imports).
    try:
        import sahi.auto_model  # noqa: F401
    except Exception:
        pass
    try:
        import sahi.predict  # noqa: F401
    except Exception:
        pass

    annotated_output_dir = rvt_base_dir / "annotated_images"
    shapefile_output_dir = rvt_base_dir / "shapefiles"

    if generate_annotated_images:
        annotated_output_dir.mkdir(parents=True, exist_ok=True)
    if generate_shapefiles:
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_src_on_sys_path()

    from src.pipeline.cv import computer_vision as cv_mod

    shp_mod = None
    if generate_shapefiles and run_shapefile_dedup:
        try:
            from src.pipeline.cv import conversion_shp as shp_mod  # type: ignore
        except Exception as e:
            _print(f"WARN: shapefile deps not available, skipping shapefile generation ({e})")

    weights_path, args_path = _resolve_model_paths(cv_config)

    try:
        get_task = getattr(cv_mod, "_get_ultralytics_task", None)
        if callable(get_task):
            _print(f"model_task={get_task(str(weights_path)) or 'unknown'}")
    except Exception as e:
        _print(f"WARN: unable to detect model task ({e})")

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
    _print(f"model_weights={weights_path}")
    _print(f"model_args={args_path}")
    _print(f"images={len(jpg_files)}")

    success_count = 0
    processed_count = 0
    skipped_already_processed = 0

    for jpg_file in jpg_files:
        image_name = jpg_file.stem
        labels_txt = jpg_dir / f"{image_name}.txt"
        labels_json = jpg_dir / f"{image_name}.json"

        detection_output_path = cv_mod.get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if generate_annotated_images else None,
        )
        annotated_img = Path(detection_output_path)

        if annotated_img.exists() or labels_txt.exists() or labels_json.exists():
            skipped_already_processed += 1
            continue

        processed_count += 1
        try:
            ok = cv_mod.run_inference(
                image_path=str(jpg_file),
                model_path=str(weights_path),
                args_path=str(args_path),
                output_path=str(detection_output_path),
                confidence_threshold=confidence_threshold,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
                generate_annotated_images=generate_annotated_images,
                annotated_output_dir=str(annotated_output_dir) if generate_annotated_images else None,
                iou_threshold=iou_threshold,
                jpg_folder_path=str(jpg_dir),
            )
        except Exception as e:
            ok = False
            _print(f"ERROR: run_inference failed for {jpg_file.name}: {e}")

        if ok:
            success_count += 1
            _print(f"image={jpg_file.name} detections=yes")
        else:
            _print(f"image={jpg_file.name} detections=no")

        if not labels_txt.exists() and not labels_json.exists():
            save_empty = getattr(cv_mod, "_save_empty_outputs", None)
            if callable(save_empty):
                try:
                    save_empty(image_path=str(jpg_file), output_path=str(detection_output_path), jpg_folder_path=str(jpg_dir))
                except Exception as e:
                    _print(f"WARN: failed to write empty outputs for {jpg_file.name}: {e}")

    _print(f"success={success_count}")
    _print(f"processed={processed_count}")
    _print(f"skipped_already_processed={skipped_already_processed}")

    if generate_shapefiles and run_shapefile_dedup and shp_mod is not None:
        out_shp = shapefile_output_dir / f"detections_{target_rvt}.shp"
        create_fn = getattr(shp_mod, "create_shapefile_from_detections", None)
        if callable(create_fn):
            create_fn(
                labels_dir=str(jpg_dir),
                output_shapefile=str(out_shp),
                tif_transform_data=tif_transform_data,
                crs="EPSG:2154",
                temp_dir=None,
                class_names=None,
            )

        dedup_fn = getattr(shp_mod, "deduplicate_shapefiles_final", None)
        if callable(dedup_fn):
            shp_paths = [str(p) for p in shapefile_output_dir.glob("*.shp")]
            if shp_paths:
                dedup_fn(
                    labels_dir=str(jpg_dir),
                    shapefile_paths=shp_paths,
                    iou_threshold=0.1,
                    crs="EPSG:2154",
                )

        _print(f"shapefiles_dir={shapefile_output_dir}")
        _print(f"detections_shp={out_shp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
