from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..geo_utils import extract_tif_transform_data, write_world_file as write_world_file_from_transform
from ..types import LogFn, CancelCheckFn
from .external_runner import find_external_cv_runner, run_external_cv_runner

# Re-export pour compatibilité (utilisé par preflight.py)
_find_external_cv_runner = find_external_cv_runner


def _get_model_slug(cv_config: Dict[str, Any]) -> str:
    """Retourne un slug court et sûr pour le nom du modèle (pour nommer les sous-dossiers)."""
    selected = cv_config.get("selected_model", "")
    if not selected:
        return "unknown"
    p = Path(selected)
    # Remonter au dossier modèle (weights/best.onnx -> model_name)
    model_dir = p.parent
    if model_dir.name == "weights":
        model_dir = model_dir.parent
    slug = model_dir.name or p.stem
    # Nettoyer pour un nom de dossier sûr
    slug = re.sub(r'[^\w\-.]', '_', slug)
    return slug or "model"


def _prepare_model_workdir(
    jpg_dir: Path,
    rvt_base_dir: Optional[Path],
    model_slug: str,
    log: LogFn,
) -> tuple:
    """Crée un dossier de travail par modèle avec des liens/copies vers les JPG.

    Retourne (model_jpg_dir, model_rvt_base).
    """
    model_rvt_base = (rvt_base_dir or jpg_dir.parent) / model_slug
    model_rvt_base.mkdir(parents=True, exist_ok=True)

    model_jpg_dir = model_rvt_base / "jpg"
    model_jpg_dir.mkdir(parents=True, exist_ok=True)

    # Créer des liens (ou copies) vers les JPG et world files originaux
    src_files = sorted(jpg_dir.glob("*.jpg"))
    total = len(src_files)
    if total > 0:
        log(f"Préparation dossier modèle [{model_slug}]: {total} images à lier…")

    linked = 0
    for idx, jpg in enumerate(src_files):
        # Lier le JPG et son éventuel world file (.jgw)
        for src in [jpg] + list(jpg.parent.glob(f"{jpg.stem}.jgw")):
            dest = model_jpg_dir / src.name
            if dest.exists():
                continue
            try:
                os.link(str(src), str(dest))
            except OSError:
                try:
                    dest.symlink_to(src)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(str(src), str(dest))
            linked += 1

        if total > 100 and (idx + 1) % 500 == 0:
            log(f"  … {idx + 1}/{total} images liées")

    if linked > 0:
        log(f"Préparation dossier modèle [{model_slug}]: {linked} fichiers liés/copiés")

    return model_jpg_dir, model_rvt_base


def run_cv_on_folder(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
) -> None:
    # ── Isolation par modèle ──────────────────────────────────────────
    # Chaque modèle écrit ses labels (.txt/.json), images annotées et
    # shapefiles dans un sous-dossier dédié pour éviter les collisions
    # quand plusieurs modèles ciblent le même RVT.
    model_slug = _get_model_slug(cv_config)
    effective_jpg_dir, effective_rvt_base = _prepare_model_workdir(
        jpg_dir, rvt_base_dir, model_slug, log,
    )

    # Générer le fichier classes.txt dans le dossier JPG du modèle
    try:
        from .class_utils import load_class_names_from_model, resolve_model_weights_path
        weights_path = resolve_model_weights_path(cv_config)
        if weights_path and weights_path.exists():
            class_names = load_class_names_from_model(weights_path)
            if class_names:
                classes_file = effective_jpg_dir / "classes.txt"
                if not classes_file.exists():
                    if isinstance(class_names, dict):
                        sorted_names = [class_names[k] for k in sorted(class_names.keys())]
                        classes_file.write_text("\n".join(sorted_names), encoding="utf-8")
                    elif isinstance(class_names, (list, tuple)):
                        classes_file.write_text("\n".join(str(n) for n in class_names), encoding="utf-8")
                    log(f"Fichier classes.txt créé: {classes_file}")
    except Exception as e:
        log(f"Avertissement: impossible de créer classes.txt: {e}")

    # Log SAHI config pour ce modèle
    sahi_cfg = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    log(f"Computer Vision [{model_slug}]: SAHI slice={sahi_cfg.get('slice_height', 640)}×{sahi_cfg.get('slice_width', 640)}, overlap={sahi_cfg.get('overlap_ratio', 0.2)}")

    # 1) Essayer le runner ONNX externe (compilé)
    ext = find_external_cv_runner(log=log)
    if ext is not None:
        log(f"Computer Vision: utilisation runner externe -> {ext}")
        try:
            run_external_cv_runner(
                ext=ext,
                jpg_dir=effective_jpg_dir,
                target_rvt=target_rvt,
                rvt_base_dir=effective_rvt_base,
                cv_config=cv_config,
                single_jpg=single_jpg,
                run_shapefile_dedup=run_shapefile_dedup,
                tif_transform_data=tif_transform_data,
                global_color_map=global_color_map,
                log=log,
                cancel_check=cancel_check,
            )
            return
        except Exception as e:
            log(f"Computer Vision: échec runner externe, fallback Python ONNX: {e}")

    # 2) Fallback : inférence ONNX en Python (onnxruntime)
    expected = (
        "third_party/cv_runner_onnx/windows/cv_runner_onnx.exe" if os.name == "nt" else "third_party/cv_runner_onnx/linux/cv_runner_onnx"
    )
    log(f"Computer Vision: runner externe absent (attendu: {expected})")
    log("Computer Vision: fallback interne ONNX -> src.pipeline.cv.computer_vision_onnx")

    enabled = bool((cv_config or {}).get("enabled", False))
    if not enabled:
        return

    _run_fallback_inference(
        jpg_dir=effective_jpg_dir,
        cv_config=cv_config,
        target_rvt=target_rvt,
        rvt_base_dir=effective_rvt_base,
        tif_transform_data=tif_transform_data,
        single_jpg=single_jpg,
        run_shapefile_dedup=run_shapefile_dedup,
        global_color_map=global_color_map,
        log=log,
    )


# ------------------------------------------------------------------ #
#  Fallback : inférence ONNX en Python (onnxruntime)                   #
# ------------------------------------------------------------------ #

def _run_fallback_inference(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
) -> None:
    """Inférence ONNX image par image via computer_vision_onnx (fallback Python)."""
    from . import computer_vision_onnx as cv_mod
    from .class_utils import (
        resolve_model_weights_path,
        load_class_names_from_model,
        load_class_colors_from_model,
    )
    from .cv_output import get_detection_output_path

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

    weights_path = resolve_model_weights_path(cv_config)
    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids du modèle non trouvé: {weights_path}")

    class_names = load_class_names_from_model(weights_path)
    class_colors = load_class_colors_from_model(weights_path)
    log(f"Computer Vision: {len(class_names or [])} classes, couleurs={'oui' if class_colors else 'non'}")
    log(f"SAHI: slice={slice_height}×{slice_width}, overlap={overlap_ratio}")

    # Charger les métadonnées du modèle pour afficher les paramètres de segmentation
    _model_meta = {}
    _meta_path = weights_path.with_suffix('.json')
    if _meta_path.exists():
        try:
            _model_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    _is_segmentation = _model_meta.get("model_type") in ("segformer", "smp") or _model_meta.get("task") == "semantic_segmentation"
    _use_sahi_meta = _model_meta.get("use_sahi", True)
    if _is_segmentation:
        _bg_bias = float(_model_meta.get("bg_bias", 0.0))
        _model_conf = _model_meta.get("confidence_threshold")
        _eff_conf = float(_model_conf) if _model_conf is not None else confidence_threshold
        log(f"Computer Vision: Paramètres segmentation -> confidence_threshold={_eff_conf} bg_bias={_bg_bias} use_sahi={_use_sahi_meta}")

    rvt_base = rvt_base_dir or jpg_dir.parent
    annotated_output_dir: Optional[Path] = None
    shapefile_output_dir: Optional[Path] = None

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

        detection_output_path = get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if annotated_output_dir else None,
        )
        annotated_img = Path(detection_output_path)

        if annotated_img.exists() or labels_txt.exists() or labels_json.exists():
            skipped_already_processed += 1
            continue

        log(f"Inférence CV sur: {jpg_file.name} (SAHI: {slice_width}x{slice_height}, overlap={overlap_ratio})")

        ok = cv_mod.run_onnx_inference(
            image_path=str(jpg_file),
            model_path=str(weights_path),
            output_path=detection_output_path,
            confidence_threshold=confidence_threshold,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
            generate_annotated_images=generate_annotated_images,
            annotated_output_dir=str(annotated_output_dir) if annotated_output_dir else None,
            iou_threshold=iou_threshold,
            jpg_folder_path=str(jpg_dir),
            class_names=class_names,
            class_colors=class_colors,
        )
        if ok:
            success_count += 1
            if generate_annotated_images and annotated_output_dir is not None:
                annotated_path = Path(detection_output_path)
                if annotated_path.exists() and tif_transform_data:
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
            global_color_map=global_color_map,
            log=log,
        )


# ------------------------------------------------------------------ #
#  Génération shapefiles + déduplication                               #
# ------------------------------------------------------------------ #

def deduplicate_cv_shapefiles_final(
    *,
    labels_dir: Path,
    shp_dir: Path,
    target_rvt: str,
    cv_config: Optional[Dict[str, Any]] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    global_color_map: Optional[Dict[str, int]] = None,
    temp_dir: Optional[Path] = None,
    crs: str = "EPSG:2154",
    log: LogFn = lambda _: None,
) -> None:
    from .conversion_shp import create_shapefile_from_detections, deduplicate_shapefiles_final
    log("Computer Vision: conversion shapefile -> src.pipeline.cv.conversion_shp")

    class_names = None
    class_colors = None
    try:
        from .class_utils import resolve_model_weights_path, load_class_names_from_model, load_class_colors_from_model
        if isinstance(cv_config, dict):
            weights_path = resolve_model_weights_path(cv_config)
            if weights_path and weights_path.exists():
                class_names = load_class_names_from_model(weights_path)
                class_colors = load_class_colors_from_model(weights_path)
    except Exception as e:
        log(f"Computer Vision: impossible de récupérer les noms de classes depuis le modèle: {e}")

    shp_dir.mkdir(parents=True, exist_ok=True)

    # Générer les shapefiles par classe
    out_shp = shp_dir / f"detections_{target_rvt}.shp"
    _raw_classes = (cv_config or {}).get("selected_classes")
    selected_classes = _raw_classes if isinstance(_raw_classes, list) else None
    try:
        create_shapefile_from_detections(
            labels_dir=str(labels_dir),
            output_shapefile=str(out_shp),
            tif_transform_data=tif_transform_data,
            crs=str(crs),
            temp_dir=str(temp_dir) if temp_dir is not None else None,
            class_names=class_names,
            selected_classes=selected_classes,
            class_colors=class_colors,
            global_color_map=global_color_map if global_color_map else None,
        )
        qgs_root = shp_dir.parent if shp_dir.name.lower() in {"shapefiles", "shp"} else shp_dir
        qgs_path = qgs_root / "detections_validation.qgs"
        if qgs_path.exists():
            log(f"Computer Vision: projet QGIS généré -> {qgs_path}")
    except Exception as e:
        log(f"Computer Vision: génération shapefile/projet QGIS ignorée (erreur): {e}")

    # Déduplication par IoU + filtrage par aire minimale
    shapefile_paths = [str(p) for p in shp_dir.glob("*.shp")]
    if shapefile_paths:
        try:
            min_area_m2 = float((cv_config or {}).get("min_area_m2", 0.0))
            area_filter_enabled = min_area_m2 > 0

            deduplicate_shapefiles_final(
                labels_dir=str(labels_dir),
                shapefile_paths=shapefile_paths,
                iou_threshold=0.1,
                crs=str(crs),
                area_filter_enabled=area_filter_enabled,
                area_filter_min_m2=min_area_m2,
            )
        except Exception as e:
            log(f"Computer Vision: déduplication shapefiles ignorée (erreur): {e}")
