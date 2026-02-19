from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..progress_reporter import ProgressReporter
    from ..structured_logger import StructuredLogger

LogFn = Callable[[str], None]


def _collect_vrt_paths_and_build(results_dir: Path, log: LogFn) -> List[str]:
    """Parcourt results/ pour créer les index.vrt (tif/, jpg/, annotated_images/) et retourne les chemins VRT."""
    from ...pipeline.ign.products.results import build_vrt_index

    vrt_paths: List[str] = []
    if not results_dir.exists():
        return vrt_paths

    # VRT pour chaque dossier de produit TIF
    for tif_dir in results_dir.rglob("tif"):
        if tif_dir.is_dir() and list(tif_dir.glob("*.tif")):
            build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=log)
            vrt_path = tif_dir / "index.vrt"
            if vrt_path.exists():
                vrt_paths.append(str(vrt_path))

    # VRT pour chaque dossier JPG (images géoréférencées)
    for jpg_dir in results_dir.rglob("jpg"):
        if jpg_dir.is_dir() and list(jpg_dir.glob("*.jpg")):
            build_vrt_index(jpg_dir, pattern="*.jpg", output_name="index.vrt", log=log)

    # VRT pour annotated_images si présent
    annotated_dir = results_dir / "annotated_images"
    if annotated_dir.exists() and list(annotated_dir.glob("*.jpg")):
        build_vrt_index(annotated_dir, pattern="*.jpg", output_name="index.vrt", log=log)

    return vrt_paths


def _collect_shapefiles(results_dir: Path, target_rvt: str, rvt_params: Dict[str, Any]) -> List[str]:
    """Collecte les shapefiles de détection CV depuis results/RVT/<target_rvt>/**/shapefiles/."""
    shapefile_paths: List[str] = []
    if not results_dir.exists():
        return shapefile_paths

    try:
        from ...pipeline.ign.products.rvt_naming import get_rvt_param_suffix
        param_suffix = get_rvt_param_suffix(target_rvt, rvt_params)
        target_rvt_dir = f"{target_rvt}{param_suffix}" if param_suffix else target_rvt
        rvt_root = results_dir / "RVT" / target_rvt_dir
        if rvt_root.exists():
            # Chercher dans tous les sous-dossiers shapefiles/ (y compris per-model)
            for shp_dir in rvt_root.rglob("shapefiles"):
                if shp_dir.is_dir():
                    for shp_file in shp_dir.glob("*.shp"):
                        shapefile_paths.append(str(shp_file))
    except Exception:
        # Fallback: chercher dans tous les sous-dossiers correspondant au target_rvt
        for rvt_subdir in results_dir.glob("RVT/*"):
            if rvt_subdir.is_dir() and rvt_subdir.name.startswith(target_rvt):
                for shp_dir in rvt_subdir.rglob("shapefiles"):
                    if shp_dir.is_dir():
                        for shp_file in shp_dir.glob("*.shp"):
                            shapefile_paths.append(str(shp_file))

    return shapefile_paths


def _load_class_colors(cv_cfg: Dict[str, Any]) -> Optional[list]:
    """Charge les couleurs de classes depuis le modèle CV sélectionné."""
    try:
        from ...pipeline.cv.class_utils import load_class_colors_from_model, resolve_model_weights_path
        weights_path = resolve_model_weights_path(cv_cfg)
        if weights_path and weights_path.exists():
            return load_class_colors_from_model(weights_path)
    except Exception:
        pass
    return None


def _build_global_class_color_map(cv_runs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Construit un mapping global {class_name: palette_index} unique pour toutes les classes de tous les modèles.

    Chaque classe unique reçoit un index de palette distinct, en respectant
    les couleurs définies dans args.yaml quand elles existent (sans collision).
    """
    from ...pipeline.cv.class_utils import (
        resolve_model_weights_path,
        load_class_names_from_model,
        load_class_colors_from_model,
        BASE_COLOR_PALETTE,
    )

    palette_size = len(BASE_COLOR_PALETTE)
    class_color_map: Dict[str, int] = {}
    used_indices: set = set()

    # Premier passage: respecter les couleurs explicites de args.yaml
    for run_cfg in cv_runs:
        try:
            weights = resolve_model_weights_path(run_cfg)
            if not weights or not weights.exists():
                continue
            names = load_class_names_from_model(weights)
            colors = load_class_colors_from_model(weights)
            if not names:
                continue
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys())]
            for i, name in enumerate(names):
                if name in class_color_map:
                    continue
                if colors and i < len(colors):
                    idx = colors[i] % palette_size
                    if idx not in used_indices:
                        class_color_map[name] = idx
                        used_indices.add(idx)
        except Exception:
            continue

    # Deuxième passage: attribuer des couleurs aux classes restantes
    next_free = 0
    for run_cfg in cv_runs:
        try:
            weights = resolve_model_weights_path(run_cfg)
            if not weights or not weights.exists():
                continue
            names = load_class_names_from_model(weights)
            if not names:
                continue
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys())]
            for name in names:
                if name in class_color_map:
                    continue
                # Trouver le prochain index libre
                while next_free in used_indices:
                    next_free += 1
                class_color_map[name] = next_free % palette_size
                used_indices.add(next_free % palette_size)
                next_free += 1
        except Exception:
            continue

    return class_color_map


def _collect_all_classes(cv_runs: List[Dict[str, Any]]) -> List[str]:
    """Agrège les noms de classes de tous les modèles CV (sans doublons, ordre stable)."""
    from ...pipeline.cv.class_utils import resolve_model_weights_path, load_class_names_from_model
    all_classes: List[str] = []
    seen: set = set()
    for run_cfg in cv_runs:
        try:
            weights = resolve_model_weights_path(run_cfg)
            if weights and weights.exists():
                names = load_class_names_from_model(weights)
                if names:
                    for n in names:
                        if n not in seen:
                            seen.add(n)
                            all_classes.append(n)
        except Exception:
            continue
    return all_classes


def _generate_consolidated_qgs_project(
    shapefile_paths: List[str],
    cv_runs: List[Dict[str, Any]],
    class_colors: Optional[list],
    results_dir: Path,
    log: LogFn,
    global_color_map: Optional[Dict[str, int]] = None,
) -> None:
    """Génère un projet QGIS consolidé avec les shapefiles de tous les runs."""
    if not shapefile_paths:
        return
    try:
        all_classes = _collect_all_classes(cv_runs)
        output_shapefile = shapefile_paths[0] if shapefile_paths else ""

        from ...pipeline.cv.qgs_project import generate_qgs_project
        qgs_path = generate_qgs_project(
            created_shapefiles=shapefile_paths,
            output_shapefile=output_shapefile,
            all_classes=all_classes,
            crs="EPSG:2154",
            class_colors=class_colors,
            global_color_map=global_color_map,
        )
        if qgs_path:
            log(f"Projet QGIS consolidé (multi-modèles) généré: {qgs_path}")
    except Exception as e:
        log(f"Note: Génération du projet QGIS consolidé échouée: {e}")


def finalize_pipeline(
    *,
    output_dir: Path,
    cv_cfg: Dict[str, Any],
    rvt_params: Dict[str, Any],
    reporter: "ProgressReporter",
    slog: Optional["StructuredLogger"] = None,
    start_time: float,
    tiles_processed: int = 0,
    active_products: Optional[List[str]] = None,
    extra_label: str = "",
) -> None:
    """
    Finalisation commune à tous les runners :
    1. Création des index VRT (tif/, jpg/, annotated_images/)
    2. Collecte des shapefiles CV
    3. Chargement des couleurs de classes
    4. Logs de fin de pipeline
    5. Chargement des couches dans QGIS
    """
    import time

    results_dir = output_dir / "results"
    log: LogFn = lambda m: reporter.info(m)

    # 1. Création des index VRT
    reporter.stage("Création des index VRT")
    reporter.info("Création des fichiers VRT d'indexation...")
    vrt_paths = _collect_vrt_paths_and_build(results_dir, log)

    # 2. Collecte des shapefiles CV (tous les runs)
    from ...pipeline.cv.class_utils import resolve_cv_runs
    cv_runs = resolve_cv_runs(cv_cfg or {})
    shapefile_paths: List[str] = []
    seen_rvts: set = set()
    for run_cfg in cv_runs:
        run_rvt = str(run_cfg.get("target_rvt", "LD"))
        if run_rvt not in seen_rvts:
            seen_rvts.add(run_rvt)
            shapefile_paths.extend(_collect_shapefiles(results_dir, run_rvt, rvt_params or {}))
    # Fallback: ancien format mono-modèle
    if not shapefile_paths and not cv_runs:
        target_rvt = str((cv_cfg or {}).get("target_rvt", "LD"))
        shapefile_paths = _collect_shapefiles(results_dir, target_rvt, rvt_params or {})

    # 3. Construire un mapping global classe -> couleur unique
    global_color_map: Dict[str, int] = {}
    class_colors: Optional[list] = None
    if cv_runs:
        global_color_map = _build_global_class_color_map(cv_runs)
    if not global_color_map:
        # Fallback mono-modèle
        class_colors = _load_class_colors(cv_cfg or {})

    # 3b. Projet QGIS consolidé (multi-modèles)
    if shapefile_paths and cv_runs:
        _generate_consolidated_qgs_project(
            shapefile_paths=shapefile_paths,
            cv_runs=cv_runs,
            class_colors=class_colors,
            results_dir=results_dir,
            log=log,
            global_color_map=global_color_map,
        )

    # 4. Logs de fin de pipeline
    elapsed = time.time() - start_time
    products_list = active_products or []

    if slog:
        slog.end_pipeline(
            success=True,
            tiles_processed=tiles_processed,
            tiles_total=tiles_processed,
            products=products_list,
        )
    else:
        reporter.info("")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info(f"  ⏱️ Durée totale : {elapsed:.1f}s")
        if extra_label:
            reporter.info(f"  📄 {extra_label} : {tiles_processed}")
        elif tiles_processed > 0:
            reporter.info(f"  📄 Dalles traitées : {tiles_processed}")
        reporter.info(f"  📦 Produits : {', '.join(products_list) if products_list else 'aucun'}")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("")

    # 5. Chargement des couches dans QGIS
    if vrt_paths or shapefile_paths:
        reporter.stage("Chargement des couches")
        reporter.info(f"Chargement de {len(vrt_paths)} VRT et {len(shapefile_paths)} shapefile(s) dans QGIS...")
        # Passer le mapping global si disponible (encodé comme dict dans la liste)
        colors_param = class_colors or []
        if global_color_map:
            colors_param = [global_color_map]  # dict wrappé dans une liste
        try:
            reporter.load_layers(vrt_paths, shapefile_paths, colors_param)
        except Exception as e:
            reporter.info(f"Note: Chargement des couches non disponible ({e})")

    reporter.stage("Terminé")
    reporter.progress(100)
