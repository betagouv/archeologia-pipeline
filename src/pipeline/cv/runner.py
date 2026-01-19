from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


LogFn = Callable[[str], None]


def _find_external_cv_runner(log: Optional[LogFn] = None) -> Optional[Path]:
    plugin_root = Path(__file__).resolve().parents[3]
    candidates = []
    if os.name == "nt":
        candidates.append(plugin_root / "third_party" / "cv_runner" / "windows" / "cv_runner.exe")
    else:
        candidates.append(plugin_root / "third_party" / "cv_runner" / "linux" / "cv_runner")
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
            elif log:
                log(f"Computer Vision: runner non trouvé à {p}")
        except Exception as e:
            if log:
                log(f"Computer Vision: erreur vérification runner {p}: {e}")
            continue
    return None


def _subprocess_kwargs_no_window() -> Dict[str, Any]:
    if os.name != "nt":
        return {}
    kwargs: Dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW}
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kwargs["startupinfo"] = si
    except Exception:
        pass
    return kwargs


def write_world_file_from_transform(image_path: Path, pixel_width: float, pixel_height: float, x_origin: float, y_origin: float) -> Optional[Path]:
    """
    Crée un fichier world (.jgw pour JPEG, .pgw pour PNG) à partir des données de transformation.
    Utilise le même format que create_world_file_from_tif dans convert_tif_to_jpg.py.
    """
    suffix = image_path.suffix.lower()
    world_ext_map = {".jpg": ".jgw", ".jpeg": ".jgw", ".png": ".pgw"}
    world_ext = world_ext_map.get(suffix)
    if not world_ext:
        return None
    
    world_path = image_path.with_suffix(world_ext)
    with open(world_path, "w") as f:
        f.write(f"{pixel_width:.10f}\n")
        f.write(f"0.0000000000\n")  # row_rotation
        f.write(f"0.0000000000\n")  # col_rotation
        f.write(f"{pixel_height:.10f}\n")
        f.write(f"{x_origin:.10f}\n")
        f.write(f"{y_origin:.10f}\n")
    return world_path


def extract_tif_transform_data(reference_tif_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        import rasterio  # type: ignore

        with rasterio.open(str(reference_tif_path)) as ds:
            transform = ds.transform
            pixel_width = float(transform.a)
            pixel_height = float(transform.e)
            x_origin = float(transform.c)
            y_origin = float(transform.f)
            return pixel_width, pixel_height, x_origin, y_origin
    except Exception:
        pass

    try:
        from osgeo import gdal  # type: ignore

        ds = gdal.Open(str(reference_tif_path))
        if ds is None:
            return None, None, None, None
        geotransform = ds.GetGeoTransform()
        ds = None
        if geotransform:
            x_origin, pixel_width, _row_rot, y_origin, _col_rot, pixel_height = geotransform
            return float(pixel_width), float(pixel_height), float(x_origin), float(y_origin)
    except Exception:
        pass

    return None, None, None, None


def run_cv_on_folder(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    log: LogFn = lambda _: None,
) -> None:
    ext = _find_external_cv_runner(log=log)
    if ext is not None:
        log(f"Computer Vision: utilisation runner externe -> {ext}")
        payload: Dict[str, Any] = {
            "jpg_dir": str(jpg_dir),
            "target_rvt": target_rvt,
            "rvt_base_dir": str(rvt_base_dir) if rvt_base_dir else None,
            "cv_config": cv_config,
            "single_jpg": str(single_jpg) if single_jpg else None,
            "run_shapefile_dedup": bool(run_shapefile_dedup),
            "tif_transform_data": tif_transform_data or {},
        }

        if bool((cv_config or {}).get("export_runner_config", False)):
            try:
                base = rvt_base_dir or jpg_dir.parent
                out_dir = Path(base) / "cv_runner_configs"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                suffix = single_jpg.stem if single_jpg else "folder"
                out_path = out_dir / f"cv_runner_{target_rvt}_{suffix}_{ts}.json"
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                log(f"Computer Vision: config runner exportée -> {out_path}")
            except Exception as e:
                log(f"Computer Vision: impossible d'exporter la config runner: {e}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload, f)
            cfg_path = Path(f.name)

        try:
            cmd = [str(ext), "--config", str(cfg_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, **_subprocess_kwargs_no_window())
            if r.stdout:
                for line in r.stdout.splitlines():
                    if line.strip():
                        log(f"[cv_runner] {line}")
            if r.stderr:
                for line in r.stderr.splitlines():
                    if line.strip():
                        log(f"[cv_runner][stderr] {line}")
            if r.returncode != 0:
                raise RuntimeError(f"cv_runner failed (code={r.returncode})")
            
            # Créer les fichiers world pour les images annotées générées par le cv_runner
            generate_annotated = bool((cv_config or {}).get("generate_annotated_images", False))
            if generate_annotated and tif_transform_data:
                base = rvt_base_dir or jpg_dir.parent
                annotated_dir = Path(base) / "annotated_images"
                if annotated_dir.exists():
                    for annotated_img in annotated_dir.glob("*.jpg"):
                        # Le nom de l'image annotée est: {original_stem}_detections.jpg
                        # On doit retrouver le stem original pour chercher dans tif_transform_data
                        stem = annotated_img.stem
                        if stem.endswith("_detections"):
                            original_stem = stem[:-11]  # Enlever "_detections"
                        else:
                            original_stem = stem
                        transform = tif_transform_data.get(original_stem)
                        if transform and len(transform) == 4:
                            pixel_width, pixel_height, x_origin, y_origin = transform
                            world_path = write_world_file_from_transform(
                                annotated_img, pixel_width, pixel_height, x_origin, y_origin
                            )
                            if world_path:
                                log(f"Fichier world créé: {world_path.name}")
            return
        except Exception as e:
            log(f"Computer Vision: échec runner externe, fallback Python interne: {e}")
        finally:
            try:
                cfg_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    expected = (
        "third_party/cv_runner/windows/cv_runner.exe" if os.name == "nt" else "third_party/cv_runner/linux/cv_runner"
    )
    log(f"Computer Vision: runner externe absent (attendu: {expected})")

    from . import computer_vision as cv_mod
    log(f"Computer Vision: fallback interne -> src.pipeline.cv.computer_vision")

    enabled = bool((cv_config or {}).get("enabled", False))
    if not enabled:
        return

    selected_model = cv_config.get("selected_model", "")
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    confidence_threshold = float(cv_config.get("confidence_threshold", 0.3))
    iou_threshold = float(cv_config.get("iou_threshold", 0.5))
    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    sahi_config = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    slice_height = int(sahi_config.get("slice_height", 640))
    slice_width = int(sahi_config.get("slice_width", 640))
    overlap_ratio = float(sahi_config.get("overlap_ratio", 0.2))

    models_dir = Path(cv_config.get("models_dir", "models"))

    model_path = Path(selected_model)
    if model_path.exists() and model_path.is_file():
        weights_path = model_path
        args_path = model_path.parent.parent / "args.yaml"
    else:
        model_dir = models_dir / str(selected_model)
        weights_path = model_dir / "weights" / "best.pt"
        args_path = model_dir / "args.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids du modèle non trouvé: {weights_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"Fichier de configuration du modèle non trouvé: {args_path}")

    annotated_output_dir: Optional[Path] = None
    shapefile_output_dir: Optional[Path] = None

    rvt_base = rvt_base_dir or jpg_dir.parent

    if generate_annotated_images:
        annotated_output_dir = rvt_base / "annotated_images"
        annotated_output_dir.mkdir(parents=True, exist_ok=True)

    if generate_shapefiles:
        shapefile_output_dir = rvt_base / "shapefiles"
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    if single_jpg is not None:
        jpg_files = [single_jpg]
        scan_all = False
    else:
        scan_all = bool(cv_config.get("scan_all", False))
        if scan_all:
            jpg_files = sorted(jpg_dir.glob("*.jpg"))
        else:
            jpg_files = sorted(jpg_dir.glob("*.jpg"))[:1]

    jpg_files = [p for p in jpg_files if p and Path(p).exists()]
    if not jpg_files:
        return

    success_count = 0
    skipped_already_processed = 0

    for jpg_file in jpg_files:
        image_name = jpg_file.stem
        labels_txt = jpg_dir / f"{image_name}.txt"
        labels_json = jpg_dir / f"{image_name}.json"

        detection_output_path = cv_mod.get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if annotated_output_dir else None,
        )
        annotated_img = Path(detection_output_path)

        if annotated_img.exists() or labels_txt.exists() or labels_json.exists():
            skipped_already_processed += 1
            continue

        log(f"Inférence CV sur: {jpg_file.name}")
        ok = cv_mod.run_inference(
            image_path=str(jpg_file),
            model_path=str(weights_path),
            args_path=str(args_path),
            output_path=detection_output_path,
            confidence_threshold=confidence_threshold,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
            generate_annotated_images=generate_annotated_images,
            annotated_output_dir=str(annotated_output_dir) if annotated_output_dir else None,
            iou_threshold=iou_threshold,
            jpg_folder_path=str(jpg_dir),
        )
        if ok:
            success_count += 1
            # Créer le fichier world pour géoréférencer l'image annotée
            if generate_annotated_images and annotated_output_dir is not None:
                annotated_path = Path(detection_output_path)
                if annotated_path.exists() and tif_transform_data:
                    # Chercher les données de transformation pour cette image
                    jpg_stem = jpg_file.stem
                    transform = tif_transform_data.get(jpg_stem)
                    if transform and len(transform) == 4:
                        pixel_width, pixel_height, x_origin, y_origin = transform
                        world_path = write_world_file_from_transform(annotated_path, pixel_width, pixel_height, x_origin, y_origin)
                        if world_path:
                            log(f"Fichier world créé: {world_path.name}")

    if scan_all and success_count == 0 and skipped_already_processed == len(jpg_files):
        return

    if run_shapefile_dedup and generate_shapefiles and shapefile_output_dir is not None:
        deduplicate_cv_shapefiles_final(
            labels_dir=jpg_dir,
            shp_dir=shapefile_output_dir,
            target_rvt=target_rvt,
            cv_config=cv_config,
            tif_transform_data=tif_transform_data,
            crs="EPSG:2154",
            log=log,
        )


def deduplicate_cv_shapefiles_final(
    *,
    labels_dir: Path,
    shp_dir: Path,
    target_rvt: str,
    cv_config: Optional[Dict[str, Any]] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    temp_dir: Optional[Path] = None,
    crs: str = "EPSG:2154",
    log: LogFn = lambda _: None,
) -> None:
    from . import conversion_shp as shp_mod
    log(f"Computer Vision: conversion shapefile -> src.pipeline.cv.conversion_shp")

    class_names = None
    try:
        if isinstance(cv_config, dict):
            selected_model = str(cv_config.get("selected_model", "")).strip()
            models_dir = Path(cv_config.get("models_dir", "models"))
            weights_path = Path(selected_model)
            if not weights_path.exists() or not weights_path.is_file():
                weights_path = models_dir / selected_model / "weights" / "best.pt"

            if weights_path.exists():
                model_dir = weights_path.parent.parent
                for candidate in (
                    model_dir / "classes.txt",
                    model_dir / "classes.txt.txt",
                    model_dir / "class_names.txt",
                    model_dir / "class_names.txt.txt",
                    model_dir / "classes.json",
                    model_dir / "class_names.json",
                ):
                    try:
                        if not candidate.exists() or not candidate.is_file():
                            continue
                        if candidate.suffix.lower() == ".json":
                            parsed = json.loads(candidate.read_text(encoding="utf-8"))
                            if isinstance(parsed, (list, tuple, dict)) and parsed:
                                class_names = parsed
                                break
                        else:
                            lines = [ln.strip() for ln in candidate.read_text(encoding="utf-8-sig").splitlines()]
                            lines = [ln for ln in lines if ln]
                            if lines:
                                class_names = lines
                                break
                    except Exception:
                        continue
    except Exception as e:
        log(f"Computer Vision: impossible de récupérer les noms de classes depuis le modèle: {e}")

    shp_dir.mkdir(parents=True, exist_ok=True)

    shapefile_paths = [str(p) for p in shp_dir.glob("*.shp")]

    # Générer le shapefile global si la fonction existe côté legacy
    create_fn = getattr(shp_mod, "create_shapefile_from_detections", None)
    if callable(create_fn):
        out_shp = shp_dir / f"detections_{target_rvt}.shp"
        selected_classes = (cv_config or {}).get("selected_classes") or None
        try:
            create_fn(
                labels_dir=str(labels_dir),
                output_shapefile=str(out_shp),
                tif_transform_data=tif_transform_data,
                crs=str(crs),
                temp_dir=str(temp_dir) if temp_dir is not None else None,
                class_names=class_names,
                selected_classes=selected_classes,
            )
            qgs_root = shp_dir.parent if shp_dir.name.lower() in {"shapefiles", "shp"} else shp_dir
            qgs_path = qgs_root / "detections_validation.qgs"
            if qgs_path.exists():
                log(f"Computer Vision: projet QGIS généré -> {qgs_path}")
        except Exception as e:
            log(f"Computer Vision: génération shapefile/projet QGIS ignorée (erreur): {e}")

    shapefile_paths = [str(p) for p in shp_dir.glob("*.shp")]
    dedup_fn = getattr(shp_mod, "deduplicate_shapefiles_final", None)
    if callable(dedup_fn) and shapefile_paths:
        try:
            size_filter_cfg = (cv_config or {}).get("size_filter", {}) if isinstance((cv_config or {}).get("size_filter"), dict) else {}
            size_filter_enabled = bool(size_filter_cfg.get("enabled", False))
            size_filter_max_meters = float(size_filter_cfg.get("max_meters", 50.0))
            
            dedup_fn(
                labels_dir=str(labels_dir),
                shapefile_paths=shapefile_paths,
                iou_threshold=0.1,
                crs=str(crs),
                size_filter_enabled=size_filter_enabled,
                size_filter_max_meters=size_filter_max_meters,
            )
        except Exception as e:
            log(f"Computer Vision: déduplication shapefiles ignorée (erreur): {e}")
