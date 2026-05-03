from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..progress_reporter import ProgressReporter
    from ..structured_logger import StructuredLogger

LogFn = Callable[[str], None]


def _collect_vrt_paths_and_build(idx_dir: Path, det_dir: Path, log: LogFn) -> List[str]:
    """Parcourt indices/ et detections/ pour créer les index.vrt et retourne les chemins VRT."""
    from ...pipeline.ign.products.results import build_vrt_index

    vrt_paths: List[str] = []

    # VRT pour chaque dossier de produit TIF dans indices/
    if idx_dir.exists():
        for tif_dir in idx_dir.rglob("tif"):
            if not tif_dir.is_dir():
                continue
            vrt_path = tif_dir / "index.vrt"
            if vrt_path.exists():
                log(f"VRT déjà existant, ignoré: {vrt_path.name}")
                vrt_paths.append(str(vrt_path))
                continue
            if list(tif_dir.glob("*.tif")):
                build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=log)
                if vrt_path.exists():
                    vrt_paths.append(str(vrt_path))

        # VRT pour chaque dossier PNG (images géoréférencées) dans indices/
        for png_dir in idx_dir.rglob("png"):
            if not png_dir.is_dir():
                continue
            vrt_path = png_dir / "index.vrt"
            if vrt_path.exists():
                log(f"VRT déjà existant, ignoré: {vrt_path.name}")
                continue
            if list(png_dir.glob("*.png")):
                build_vrt_index(png_dir, pattern="*.png", output_name="index.vrt", log=log)

    # VRT pour annotated_images dans detections/
    if det_dir.exists():
        for annotated_dir in det_dir.rglob("annotated_images"):
            if not annotated_dir.is_dir():
                continue
            vrt_path = annotated_dir / "index.vrt"
            if not vrt_path.exists() and list(annotated_dir.glob("*.png")):
                build_vrt_index(annotated_dir, pattern="*.png", output_name="index.vrt", log=log)

    return vrt_paths


def _list_gpkg_layers(gpkg_path: Path) -> List[str]:
    """Liste les couches d'un GeoPackage avec plusieurs méthodes de fallback."""
    # Méthode 1 : fiona
    try:
        import fiona
        return list(fiona.listlayers(str(gpkg_path)))
    except Exception:
        pass
    # Méthode 2 : osgeo.ogr (toujours disponible dans OSGeo4W)
    try:
        from osgeo import ogr
        ds = ogr.Open(str(gpkg_path))
        if ds is not None:
            layers = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            ds = None
            return layers
    except Exception:
        pass
    # Méthode 3 : geopandas (lecture du fichier)
    try:
        import geopandas as gpd
        return gpd.list_layers(str(gpkg_path))["name"].tolist()
    except Exception:
        pass
    return []


def _collect_shapefiles(det_dir: Path) -> List[str]:
    """Collecte les couches GeoPackage de détection CV depuis detections/**/shapefiles/."""
    shapefile_paths: List[str] = []
    if not det_dir.exists():
        return shapefile_paths

    for shp_dir in det_dir.rglob("shapefiles"):
        if not shp_dir.is_dir():
            continue
        for gpkg_file in shp_dir.glob("*.gpkg"):
            layers = _list_gpkg_layers(gpkg_file)
            if layers:
                for layer in layers:
                    shapefile_paths.append(f"{gpkg_file}|layername={layer}")
            else:
                # Dernier recours : on inscrit le GPKG seul (nom de couche inconnu)
                shapefile_paths.append(str(gpkg_file))

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


def _resolve_model_dir_from_run(run_cfg: Dict[str, Any]) -> Optional[Path]:
    """Résout le dossier racine du modèle depuis un run_cfg.

    Supporte les deux formats :
    - runs bruts (clé 'model') : chemin absolu vers le fichier weights
    - runs résolus par resolve_cv_runs (clé 'selected_model') : même chose
    Le fichier weights peut ne pas exister (gitignored) ; on remonte quand même.
    """
    cfg = run_cfg or {}
    # Priorité : 'model' (runs bruts), puis 'selected_model' (runs résolus)
    model_val = str(cfg.get("model") or cfg.get("selected_model") or "").strip()
    if not model_val:
        return None
    p = Path(model_val)
    # Fichier weights existant
    if p.is_file():
        return p.parent.parent if p.parent.name == "weights" else p.parent
    # Déjà un dossier
    if p.is_dir():
        return p
    # Le fichier n'existe pas (gitignored) — remonter quand même
    parent = p.parent
    if parent.name == "weights":
        return parent.parent
    # Si c'est juste un nom de modèle sans chemin complet, chercher dans models_dir
    if not p.is_absolute():
        models_dir_val = str(cfg.get("models_dir") or "").strip()
        if models_dir_val:
            candidate = Path(models_dir_val) / model_val
            if candidate.is_dir():
                return candidate
    return parent if parent != p else None


def _build_global_class_color_map(cv_runs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Construit un mapping global {class_name: palette_index} unique pour toutes les classes de tous les modèles.

    Chaque classe unique reçoit un index de palette distinct, en respectant
    les couleurs définies dans args.yaml quand elles existent (sans collision).
    """
    from ...pipeline.cv.class_utils import (
        load_class_names_from_model,
        load_class_colors_from_model,
        BASE_COLOR_PALETTE,
    )

    palette_size = len(BASE_COLOR_PALETTE)
    class_color_map: Dict[str, int] = {}
    used_indices: set = set()

    def _get_model_dir(run_cfg):
        """Retourne le dossier modèle même si les weights sont absents (gitignored)."""
        model_dir = _resolve_model_dir_from_run(run_cfg)
        if model_dir and model_dir.is_dir():
            return model_dir
        return None

    # Premier passage: respecter les couleurs explicites de args.yaml
    for run_cfg in cv_runs:
        try:
            model_dir = _get_model_dir(run_cfg)
            if not model_dir:
                continue
            names = load_class_names_from_model(model_dir)
            colors = load_class_colors_from_model(model_dir)
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
            model_dir = _get_model_dir(run_cfg)
            if not model_dir:
                continue
            names = load_class_names_from_model(model_dir)
            if not names:
                continue
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys())]
            for name in names:
                if name in class_color_map:
                    continue
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
    log: LogFn,
    global_color_map: Optional[Dict[str, int]] = None,
    cluster_class_names: Optional[set] = None,
    output_dir: Optional[Path] = None,
    min_confidence: float = 0.0,
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
            cluster_class_names=cluster_class_names,
            output_dir=output_dir,
            min_confidence=min_confidence,
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
    ui_config: Optional[Dict[str, Any]] = None,
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

    from ...pipeline.output_paths import indices_dir, detections_dir

    idx_dir = indices_dir(output_dir)
    det_dir = detections_dir(output_dir)
    log: LogFn = lambda m: reporter.info(m)

    # 1. Création des index VRT
    reporter.stage("Création des index VRT")
    reporter.info("Création des fichiers VRT d’indexation...")
    vrt_paths = _collect_vrt_paths_and_build(idx_dir, det_dir, log)

    # 2. Collecte des shapefiles CV (tous les runs)
    from ...pipeline.cv.class_utils import resolve_cv_runs
    cv_runs = resolve_cv_runs(cv_cfg or {})
    shapefile_paths: List[str] = _collect_shapefiles(det_dir)

    # 3. Construire un mapping global classe -> couleur unique
    global_color_map: Dict[str, int] = {}
    class_colors: Optional[list] = None
    if cv_runs:
        global_color_map = _build_global_class_color_map(cv_runs)
    if not global_color_map:
        # Fallback mono-modèle
        class_colors = _load_class_colors(cv_cfg or {})

    # 3b. Collecter les noms de classes cluster depuis les configs des modèles
    cluster_class_names: set = set()
    try:
        from ...pipeline.cv.model_config import load_clustering_config_from_model
        from ...pipeline.cv.class_utils import resolve_model_weights_path
        for run_cfg in cv_runs:
            wp = resolve_model_weights_path(run_cfg)
            if wp and wp.exists():
                cc = load_clustering_config_from_model(wp)
                if cc:
                    for c in cc:
                        cluster_class_names.add(c.get("output_class_name", ""))
    except Exception:
        pass

    # 3c. Projet QGIS consolidé (multi-modèles)
    if shapefile_paths and cv_runs:
        _min_conf_sym = float((cv_cfg or {}).get("confidence_threshold", 0.0) or 0.0)
        _generate_consolidated_qgs_project(
            shapefile_paths=shapefile_paths,
            cv_runs=cv_runs,
            class_colors=class_colors,
            log=log,
            global_color_map=global_color_map,
            cluster_class_names=cluster_class_names if cluster_class_names else None,
            output_dir=output_dir,
            min_confidence=_min_conf_sym,
        )

    # 4. Génération du fichier metadata.json
    try:
        import json as _json
        import datetime as _dt

        meta = {
            "pipeline_version": "2.0",
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "tiles_processed": tiles_processed,
            "active_products": active_products or [],
            "rvt_params": rvt_params or {},
            "cv_runs": [
                {
                    "model": r.get("selected_model", ""),
                    "target_rvt": r.get("target_rvt", ""),
                }
                for r in cv_runs
            ],
            "structure": {
                "indices": str(idx_dir),
                "detections": str(det_dir),
            },
            "ui_config": ui_config or {},
        }
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        reporter.info(f"Métadonnées enregistrées: {meta_path.name}")
    except Exception as _meta_e:
        reporter.info(f"Note: métadonnées non écrites ({_meta_e})")

    # 5. Logs de fin de pipeline
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
