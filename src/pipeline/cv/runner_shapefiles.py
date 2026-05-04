"""Génération des shapefiles à partir des détections brutes.

Encapsule l'étape « après inférence » qui transforme les ``.txt``/``.json``
écrits dans ``raw_detections/`` en GeoPackage géoréférencé. Délègue
l'essentiel à :mod:`conversion_shp` mais centralise la résolution des
métadonnées du modèle (classes, couleurs, clustering, post-traitement)
qui pilotent cette conversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..types import LogFn


def deduplicate_cv_shapefiles_final(
    *,
    labels_dir: Path,
    png_dir: Optional[Path] = None,
    shp_dir: Path,
    target_rvt: str,
    cv_config: Optional[Dict[str, Any]] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    global_color_map: Optional[Dict[str, int]] = None,
    temp_dir: Optional[Path] = None,
    crs: str = "EPSG:2154",
    log: LogFn = lambda _: None,
) -> None:
    from .conversion_shp import create_shapefile_from_detections, _filter_gpkg_by_min_area
    log("Computer Vision: conversion GeoPackage -> src.pipeline.cv.conversion_shp")

    class_names = None
    class_colors = None
    model_task = None
    clustering_configs = None
    postprocess_config = None
    try:
        from .class_utils import resolve_model_weights_path, load_class_names_from_model, load_class_colors_from_model
        from .model_config import load_clustering_config_from_model, load_postprocess_config_from_model
        if isinstance(cv_config, dict):
            weights_path = resolve_model_weights_path(cv_config)
            if weights_path and weights_path.exists():
                class_names = load_class_names_from_model(weights_path)
                class_colors = load_class_colors_from_model(weights_path)
                # Lire le type de tâche depuis les métadonnées du modèle
                meta_path = weights_path.with_suffix('.json')
                if meta_path.exists():
                    try:
                        import json as _json
                        _meta = _json.loads(meta_path.read_text(encoding='utf-8'))
                        model_task = _meta.get('task')
                        log(f"Computer Vision: tâche du modèle = {model_task}")
                    except Exception:
                        pass
                # Charger la configuration de clustering
                clustering_configs = load_clustering_config_from_model(weights_path)
                if clustering_configs:
                    log(f"Computer Vision: {len(clustering_configs)} config(s) de clustering chargée(s)")
                # Charger la configuration de post-traitement géométrique (merge/overlap)
                postprocess_config = load_postprocess_config_from_model(weights_path)
    except Exception as e:
        log(f"Computer Vision: impossible de récupérer les noms de classes depuis le modèle: {e}")

    # Filtrer les configs de clustering selon selected_classes :
    # une config n'est activée que si son output_class_name est sélectionné
    _raw_classes = (cv_config or {}).get("selected_classes")
    selected_classes = _raw_classes if isinstance(_raw_classes, list) else None
    if clustering_configs and selected_classes is not None:
        clustering_configs = [
            cc for cc in clustering_configs
            if str(cc.get("output_class_name") or "").strip() in selected_classes
        ]
        if not clustering_configs:
            log("Computer Vision: clustering désactivé (output_class_name non sélectionné)")
        else:
            log(f"Computer Vision: {len(clustering_configs)} config(s) de clustering actives après filtrage")

    shp_dir.mkdir(parents=True, exist_ok=True)

    # Générer les shapefiles par classe (le post-processing global
    # — fusion des polygones adjacents + suppression des superpositions —
    # est intégré directement dans create_shapefile_from_detections)
    out_shp = shp_dir / f"detections_{target_rvt}.gpkg"
    try:
        create_shapefile_from_detections(
            labels_dir=str(labels_dir),
            png_dir=str(png_dir) if png_dir is not None else None,
            output_shapefile=str(out_shp),
            tif_transform_data=tif_transform_data,
            crs=str(crs),
            temp_dir=str(temp_dir) if temp_dir is not None else None,
            class_names=class_names,
            selected_classes=selected_classes,
            class_colors=class_colors,
            global_color_map=global_color_map if global_color_map else None,
            model_task=model_task,
            clustering_configs=clustering_configs,
            postprocess_config=postprocess_config,
            min_confidence=float((cv_config or {}).get("confidence_threshold", 0.0) or 0.0),
        )
        qgs_root = shp_dir.parent if shp_dir.name.lower() in {"shapefiles", "shp"} else shp_dir
        qgs_path = qgs_root / "detections_validation.qgs"
        if qgs_path.exists():
            log(f"Computer Vision: projet QGIS généré -> {qgs_path}")
    except Exception as e:
        log(f"Computer Vision: génération shapefile/projet QGIS ignorée (erreur): {e}")

    # Filtrage par aire minimale (optionnel)
    min_area_m2 = float((cv_config or {}).get("min_area_m2", 0.0))
    if min_area_m2 > 0:
        gpkg_paths = [str(p) for p in shp_dir.glob("*.gpkg")]
        if gpkg_paths:
            try:
                _filter_gpkg_by_min_area(
                    gpkg_paths=gpkg_paths,
                    min_area_m2=min_area_m2,
                    crs=str(crs),
                )
            except Exception as e:
                log(f"Computer Vision: filtrage par aire ignoré (erreur): {e}")
