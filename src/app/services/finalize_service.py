from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..progress_reporter import ProgressReporter
    from ..structured_logger import StructuredLogger

LogFn = Callable[[str], None]


def build_entity_grouping(
    runs: Optional[List[Dict[str, Any]]],
) -> "tuple[Dict[str, str], set]":
    """Depuis les runs, renvoie ``(entity_labels: slug→libellé, derived_slugs)``.

    ``derived_slugs`` = slugs des entités **dérivées** (zone + constituants) : seules
    elles forment un groupe de couches — dans le ``.qgs`` (``ui/qgs_writer``) **et** au
    chargement live (``layer_loader.load_result_layers``). Helper partagé par les deux
    chemins pour garantir le **même** regroupement.
    """
    entity_labels: Dict[str, str] = {}
    derived_slugs: set = set()
    for r in runs or []:
        if not isinstance(r, dict):
            continue
        for ent in (r.get("entities") or []):
            if not isinstance(ent, dict):
                continue
            slug = str(ent.get("slug") or "").strip()
            if not slug:
                continue
            entity_labels[slug] = str(ent.get("label") or slug)
            if ent.get("is_derived"):
                derived_slugs.add(slug)
    return entity_labels, derived_slugs


def build_min_confidence_by_slug(
    cv_runs: Optional[List[Dict[str, Any]]],
    *,
    model_slug_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, float]:
    """Mappe chaque slug de couche → seuil de confiance du run qui la produit.

    La symbologie catégorisée du ``.qgs`` consolidé doit utiliser, par couche,
    le **même** ``min_confidence`` que celui ayant servi à binner le champ
    ``conf_bin`` à la conversion (``cv_config["confidence_threshold"]`` du run,
    cf. ``runner_shapefiles.deduplicate_cv_shapefiles_final``). Sinon les
    libellés de catégories (ex. ``[0.2:0.4[``) ne matchent pas les ``conf_bin``
    réels (ex. ``[0.3:0.4[``) et la tranche basse devient invisible.

    Les seuils étant **par entité** (réglages avancés), un unique seuil global
    ne convient pas : on indexe par slug d'entité (route entité-centrée
    ``detections/<slug>/<slug>.gpkg``), avec repli sur le slug de modèle pour
    les runs legacy sans ``entities``.

    Si un slug est alimenté par plusieurs runs, on conserve le **minimum** :
    la 1ʳᵉ tranche de légende couvre alors toutes les ``conf_bin`` présentes.

    ``model_slug_fn`` : injecté pour les tests ; par défaut, import différé de
    ``pipeline.cv.runner_cache.get_model_slug`` (évite de tirer QGIS/shapely
    sur le chemin entité, pur).
    """
    result: Dict[str, float] = {}

    def _put(slug: str, conf: float) -> None:
        slug = (slug or "").strip()
        if not slug:
            return
        prev = result.get(slug)
        result[slug] = conf if prev is None else min(prev, conf)

    for run in cv_runs or []:
        if not isinstance(run, dict):
            continue
        conf = float(run.get("confidence_threshold", 0.0) or 0.0)
        entities = run.get("entities") or []
        if entities:
            for ent in entities:
                if isinstance(ent, dict):
                    _put(str(ent.get("slug") or ""), conf)
        else:
            fn = model_slug_fn
            if fn is None:
                from ...pipeline.cv.runner_cache import get_model_slug as fn  # import différé
            _put(fn(run), conf)

    return result


def _collect_vrt_paths_and_build(idx_dir: Path, det_dir: Path, log: LogFn) -> List[str]:
    """Parcourt indices/ pour créer les index.vrt et retourne les chemins VRT."""
    try:
        from ...pipeline.ign.products.results import build_vrt_index
    except ImportError:
        from pipeline.ign.products.results import build_vrt_index

    vrt_paths: List[str] = []

    # VRT pour chaque dossier de produit TIF dans indices/
    if idx_dir.exists():
        for tif_dir in idx_dir.rglob("tif"):
            if not tif_dir.is_dir():
                continue
            vrt_path = tif_dir / "index.vrt"
            if list(tif_dir.glob("*.tif")):
                build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=log)
                if vrt_path.exists():
                    vrt_paths.append(str(vrt_path))

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
    """Collecte les couches GeoPackage de détection CV (organisation entité-centrée).

    Parcourt ``detections/<entity_slug>/<entity_slug>.gpkg`` (et tout ``.gpkg``
    livrable) en **excluant** l'échafaudage technique ``detections/_technique/``
    (dumps d'inférence, GeoPackage modèle de repli vide). Reste tolérant aux
    anciens layouts (``shapefiles/``) tant qu'ils ne sont pas sous ``_technique/``.
    """
    shapefile_paths: List[str] = []
    if not det_dir.exists():
        return shapefile_paths

    try:
        from ...pipeline.output_paths import DIR_TECHNIQUE
    except ImportError:
        from pipeline.output_paths import DIR_TECHNIQUE

    for gpkg_file in sorted(det_dir.rglob("*.gpkg")):
        # Exclure l'échafaudage technique (detections/_technique/…)
        if DIR_TECHNIQUE in gpkg_file.relative_to(det_dir).parts:
            continue
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
            models_dir_path = Path(models_dir_val)
            if not models_dir_path.is_absolute():
                # models_dir relatif (« data/models ») → résoudre contre la
                # racine du plugin, sinon il pointe à côté selon le CWD
                # (sous QGIS 4 le CWD est le dossier d'install de QGIS, d'où
                # « Aucun fichier de classes trouvé dans . » + couleurs vides).
                models_dir_path = Path(__file__).resolve().parents[3] / models_dir_path
            candidate = models_dir_path / model_val
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
    1. Création des index VRT (tif/)
    2. Collecte des shapefiles CV
    3. Chargement des couleurs de classes
    4. Logs de fin de pipeline
    5. Chargement des couches dans QGIS
    """
    import time

    from ...pipeline.output_paths import indices_dir, detections_dir
    from ..progress_reporter import report_busy, report_stage_id
    from ..progress_stages import Stage
    from ..user_narrator import create_user_narrator

    narrator = create_user_narrator(reporter)

    idx_dir = indices_dir(output_dir)
    det_dir = detections_dir(output_dir)
    log: LogFn = lambda m: reporter.info(m)

    # 1. Création des index VRT
    # La finalisation occupe la bande 95→100 du plan (toutes phases CV/produits
    # terminées) : on entre à 95 pour garantir la continuité avec la phase
    # précédente, et la barre passe à 100 en sortie (load_layers fait).
    report_stage_id(reporter, Stage.FINALIZE)
    report_busy(reporter, False)  # garde-fou : sortir d'un éventuel mode indéterminé
    reporter.progress(95)
    reporter.stage("Création des index VRT")
    reporter.info("Création des fichiers VRT d’indexation...")
    narrator.finalize_start()
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

    # 3b. Le projet QGIS consolidé (detections_validation.qgs) n'est PLUS écrit ici.
    # L'API QGIS n'est pas thread-safe et finalize_pipeline tourne sur le thread worker ;
    # l'écriture XML à la main produisait des projets non relisables (CRS absent, couche
    # invalide). Le .qgs est désormais généré via l'API QGIS (QgsProject.write) sur le
    # thread principal, par ui/qgs_writer.write_validation_project, déclenché depuis
    # run_view._on_load_layers (même chemin que le chargement live, qui fonctionne).

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
            # Correspondance entité (vocabulaire utilisateur) → dossier/fichier
            # livrable. Trace le lien entre ce que l'utilisateur a coché et où
            # les résultats atterrissent (detections/<slug>/<slug>.gpkg).
            "detections_entities": [
                {
                    "entity": ent.get("id", ""),
                    "label": ent.get("label", ""),
                    "slug": ent.get("slug", ""),
                    "folder": f"detections/{ent.get('slug', '')}",
                    "gpkg": f"detections/{ent.get('slug', '')}/{ent.get('slug', '')}.gpkg",
                    "is_derived": bool(ent.get("is_derived", False)),
                    "model": r.get("selected_model", ""),
                }
                for r in cv_runs
                for ent in (r.get("entities") or [])
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
    # Annonce narrative à l'utilisateur (visible dans la fenêtre QGIS).
    narrator.pipeline_complete(
        tiles_processed=tiles_processed,
        products=products_list,
        start_time=start_time,
    )
    if not slog:
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
