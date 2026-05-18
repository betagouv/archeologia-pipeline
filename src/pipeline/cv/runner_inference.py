"""Inférence ONNX en Python (fallback) — utilisée quand le runner externe est absent.

Le runner externe compilé (``cv_runner_onnx``) est préféré en production
pour des raisons de performance et d'isolation. Quand il n'est pas
disponible, ce module exécute l'inférence directement via
:mod:`computer_vision_onnx` (onnxruntime), image par image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..geo_utils import write_world_file as write_world_file_from_transform
from ..types import LogFn, CancelCheckFn
from .external_runner import ImageProgressFn
from .runner_shapefiles import deduplicate_cv_shapefiles_final


def run_fallback_inference(
    *,
    jpg_dir: Path,
    raw_dir: Optional[Path] = None,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    effective_detection_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
    image_progress: Optional[ImageProgressFn] = None,
) -> None:
    """Inférence ONNX image par image via computer_vision_onnx (fallback Python).

    jpg_dir  : dossier source contenant les PNG d'entrée (indices/<RVT>/png/)
    raw_dir  : dossier de sortie pour les JSON/TXT (detections/<model>/raw_detections/)
               Si None, utilise jpg_dir (comportement rétrocompat).
    """
    # raw_dir = dossier de sortie JSON/TXT ; par défaut = jpg_dir (rétrocompat)
    _raw_dir = raw_dir if raw_dir is not None else jpg_dir
    from . import computer_vision_onnx as cv_mod
    from .cv_output import get_detection_output_path
    from .model_config import resolve_model_weights_path
    from .model_profile import ModelProfile

    selected_model = cv_config.get("selected_model", "")
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    confidence_threshold = float(cv_config.get("confidence_threshold", 0.3))
    iou_threshold = float(cv_config.get("iou_threshold", 0.5))
    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    weights_path = resolve_model_weights_path(cv_config)
    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids du modèle non trouvé: {weights_path}")

    profile = ModelProfile.load(weights_path)

    slice_height = profile.sahi.slice_height
    slice_width = profile.sahi.slice_width
    overlap_ratio = profile.sahi.overlap_ratio

    class_names = list(profile.class_names) if profile.class_names else None
    class_colors = list(profile.class_colors) if profile.class_colors else None
    log(f"Computer Vision: {len(class_names or [])} classes, couleurs={'oui' if class_colors else 'non'}")
    log(f"SAHI: slice={slice_height}×{slice_width}, overlap={overlap_ratio}")

    # Charger la session ONNX une seule fois pour toutes les images
    onnx_session = cv_mod._load_onnx_model(str(weights_path))
    log(f"Computer Vision: session ONNX chargée -> {weights_path.name}")

    _is_segmentation = profile.model_type in ("segformer", "smp") or profile.task in (
        "semantic_segmentation",
        "instance_segmentation",
    )
    _use_sahi_meta = profile.metadata.get("use_sahi", True)
    if _is_segmentation:
        _bg_bias = float(profile.metadata.get("bg_bias", 0.0))
        _eff_conf = profile.effective_confidence_threshold(confidence_threshold)
        log(f"Computer Vision: Paramètres segmentation -> confidence_threshold={_eff_conf} bg_bias={_bg_bias} use_sahi={_use_sahi_meta}")

    rvt_base = rvt_base_dir or jpg_dir.parent
    det_base = effective_detection_dir if effective_detection_dir is not None else rvt_base
    annotated_output_dir: Optional[Path] = None
    shapefile_output_dir: Optional[Path] = None

    if generate_annotated_images:
        annotated_output_dir = det_base / "annotated_images"
        annotated_output_dir.mkdir(parents=True, exist_ok=True)

    if generate_shapefiles:
        shapefile_output_dir = det_base / "shapefiles"
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    if single_jpg is not None:
        jpg_files = [single_jpg]
        scan_all = False
    else:
        scan_all = bool(cv_config.get("scan_all", False))
        if scan_all:
            jpg_files = sorted(jpg_dir.glob("*.png"))
        else:
            jpg_files = sorted(jpg_dir.glob("*.png"))[:1]

    jpg_files = [p for p in jpg_files if p and Path(p).exists()]
    if not jpg_files:
        return

    force_reprocess = bool(cv_config.get("force_reprocess", False))
    success_count = 0
    skipped_already_processed = 0

    total_images = len(jpg_files)
    for image_index, jpg_file in enumerate(jpg_files, start=1):
        if cancel_check and cancel_check():
            log("Computer Vision: Annulation demandée, arrêt de l'inférence...")
            raise RuntimeError("Computer Vision: Annulé par l'utilisateur")
        image_name = jpg_file.stem
        labels_txt = _raw_dir / f"{image_name}.txt"
        labels_json = _raw_dir / f"{image_name}.json"

        if image_progress is not None:
            try:
                image_progress(image_index, total_images, jpg_file.name)
            except Exception:
                pass

        detection_output_path = get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if annotated_output_dir else None,
        )
        annotated_img = Path(detection_output_path)

        if not force_reprocess and (annotated_img.exists() or labels_txt.exists() or labels_json.exists()):
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
            jpg_folder_path=str(_raw_dir),
            class_names=class_names,
            class_colors=class_colors,
            onnx_session=onnx_session,
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
            labels_dir=_raw_dir,
            png_dir=jpg_dir,
            shp_dir=shapefile_output_dir,
            target_rvt=target_rvt,
            cv_config=cv_config,
            tif_transform_data=tif_transform_data,
            crs="EPSG:2154",
            global_color_map=global_color_map,
            log=log,
        )
