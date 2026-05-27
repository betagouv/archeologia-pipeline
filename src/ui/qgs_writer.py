"""Écriture du projet QGIS de validation (``detections_validation.qgs``) via l'API QGIS.

QGIS-side, **thread principal uniquement** (l'API QGIS n'est pas thread-safe). Construit
un ``QgsProject`` **dédié** (jamais le singleton de la session de l'utilisateur) puis le
sérialise avec ``QgsProject.write()`` : QGIS écrit lui-même un projet qu'il sait relire
(CRS complet en WKT, ids ASCII, datasources cohérentes), ce qui élimine la classe de bugs
de l'ancienne écriture XML à la main (« sans projection », couche invalide, désalignement).

Source unique de vérité : réutilise la fabrique de couche et les helpers de
``layer_loader`` (CRS, symbologie cluster/confiance, couleurs) — exactement ce que fait le
chargement live, qui fonctionne.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .layer_loader import (
    _ensure_layer_crs,
    _parse_gpkg_source,
    _resolve_color_idx,
    build_detection_vector_layer,
)

_ALIASES = {
    "corr_pred": "correction_prediction",
    "confidence": "confiance",
    "conf_bin": "tranche_confiance",
    "conf_color": "couleur_confiance",
    "model_name": "modele_detection",
}
# Champs en lecture seule dans le formulaire (les autres — validation, corr_pred — éditables).
_READONLY = ("model_pred", "model_name", "confidence", "conf_bin", "conf_color")


def _rvt_type_of(vrt) -> str:
    """Type de produit RVT depuis le chemin du VRT (ex. ``MNT``, ``LD_...``, ``HS_...``)."""
    vp = Path(str(vrt))
    return vp.parent.parent.name if vp.parent.name == "tif" else vp.parent.name


def _add_rasters(proj, root, vrt_paths: List[str], logger: logging.Logger) -> None:
    """Ajoute les VRT comme couches raster (à plat, sous les vecteurs).

    Le MNT est placé **en dernier** → tout en bas de l'arbre (sous les indices),
    car ``root.addLayer`` ajoute en fin de liste = bas de l'arbre. Le MNT (modèle
    d'élévation brut) sert de fond ; les indices RVT (hillshade, SVF, LD…) s'affichent
    par-dessus.
    """
    from qgis.core import QgsRasterLayer

    items = [v for v in (vrt_paths or []) if v]
    items.sort(key=lambda v: _rvt_type_of(v).upper() == "MNT")  # tri stable : MNT en dernier
    for vrt in items:
        rvt_type = _rvt_type_of(vrt)
        name = f"Dalles RVT {rvt_type} (index)" if rvt_type else "Dalles RVT (index)"
        layer = QgsRasterLayer(str(vrt), name, "gdal")
        if not layer.isValid():
            logger.warning(f"VRT invalide, ignoré: {vrt}")
            continue
        _ensure_layer_crs(layer, logger)
        proj.addMapLayer(layer, False)
        root.addLayer(layer)


def _apply_validation_form(layer, all_classes: List[str], logger: logging.Logger) -> None:
    """Reproduit le formulaire de validation (alias FR, ValueMap, onglet, read-only).

    Best-effort : tout est encapsulé — un échec (API form variable selon la version QGIS)
    n'invalide pas la couche, qui reste affichable.
    """
    try:
        from qgis.core import (
            QgsAttributeEditorContainer,
            QgsAttributeEditorField,
            QgsEditFormConfig,
            QgsEditorWidgetSetup,
            QgsExpression,
            QgsOptionalExpression,
        )

        fields = layer.fields()

        def _idx(name: str) -> int:
            return fields.indexFromName(name)

        layer.setDisplayExpression('"model_pred"')

        for fld, alias in _ALIASES.items():
            i = _idx(fld)
            if i >= 0:
                layer.setFieldAlias(i, alias)

        vm_validation = QgsEditorWidgetSetup(
            "ValueMap",
            {"map": [{"oui": "oui"}, {"non": "non"}, {"peut-être": "peut-être"}]},
        )
        vm_classes = QgsEditorWidgetSetup("ValueMap", {"map": [{c: c} for c in (all_classes or [])]})
        for fld, setup in (("validation", vm_validation), ("model_pred", vm_classes), ("corr_pred", vm_classes)):
            i = _idx(fld)
            if i >= 0:
                layer.setEditorWidgetSetup(i, setup)

        cfg = layer.editFormConfig()
        cfg.setLayout(QgsEditFormConfig.TabLayout)
        root_c = cfg.invisibleRootContainer()
        try:
            root_c.clear()
        except Exception:
            pass
        container = QgsAttributeEditorContainer("Validation", root_c)

        def _add(name: str) -> None:
            i = _idx(name)
            if i >= 0:
                container.addChildElement(QgsAttributeEditorField(name, i, container))

        for name in ("model_name", "model_pred", "validation"):
            _add(name)
        ci = _idx("corr_pred")
        if ci >= 0:
            # corr_pred dans un groupe CONDITIONNEL : visible seulement si
            # validation ∈ {non, peut-être}. La visibilité conditionnelle est portée
            # par le CONTENEUR (groupe) — l'API field-level ``setVisibilityExpression``
            # n'existe pas selon la version QGIS. Garde défensive : si même le conteneur
            # ne le supporte pas, corr_pred reste simplement toujours visible.
            corr_box = QgsAttributeEditorContainer("Correction", container)
            try:
                corr_box.setVisibilityExpression(
                    QgsOptionalExpression(QgsExpression('"validation" IN (\'non\', \'peut-être\')'))
                )
            except Exception:  # noqa: BLE001
                pass
            corr_box.addChildElement(QgsAttributeEditorField("corr_pred", ci, corr_box))
            container.addChildElement(corr_box)
        for name in ("confidence", "conf_bin", "conf_color"):
            _add(name)
        root_c.addChildElement(container)

        for fld in _READONLY:
            i = _idx(fld)
            if i >= 0:
                cfg.setReadOnly(i, True)
        for fld in ("validation", "corr_pred"):
            i = _idx(fld)
            if i >= 0:
                cfg.setReadOnly(i, False)

        layer.setEditFormConfig(cfg)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Formulaire de validation non appliqué sur « {layer.name()} »: {e}")


def write_validation_project(
    qgs_path,
    vrt_paths: list,
    shapefile_paths: list,
    *,
    global_color_map: Optional[dict] = None,
    all_classes: Optional[List[str]] = None,
    confidence_threshold: float = 0.0,
    entity_labels: Optional[dict] = None,
    derived_slugs: Optional[set] = None,
    min_conf_by_slug: Optional[dict] = None,
    logger: logging.Logger,
    crs_authid: str = "EPSG:2154",
) -> Optional[Path]:
    """Écrit ``qgs_path`` (le projet de validation consolidé) via ``QgsProject.write()``.

    Construit un ``QgsProject`` dédié (pas le singleton), y ajoute les vecteurs de détection
    (groupés par entité dérivée) puis les rasters VRT, fixe le CRS projet (EPSG:2154 → WKT
    complet écrit, fin du « sans projection ») et l'étendue initiale, puis sérialise.
    Retourne le chemin si écrit, sinon ``None``. À appeler **sur le thread principal**.
    """
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsProject,
            QgsRectangle,
            QgsReferencedRectangle,
        )

        qgs_path = Path(qgs_path)
        entity_labels = entity_labels or {}
        derived_slugs = derived_slugs or set()
        min_conf_by_slug = min_conf_by_slug or {}
        global_color_map = global_color_map or {}

        proj = QgsProject()
        proj.setFileName(str(qgs_path))  # ancre les chemins relatifs
        crs = QgsCoordinateReferenceSystem(crs_authid)
        if crs.isValid():
            proj.setCrs(crs)
        proj.writeEntry("Paths", "/Absolute", False)  # datasources relatives au .qgs
        root = proj.layerTreeRoot()

        extent = QgsRectangle()
        extent.setMinimal()

        # 1) Vecteurs de détection (au-dessus des rasters) — groupés par entité dérivée.
        for shp in shapefile_paths or []:
            if not shp:
                continue
            shp_str = str(shp)
            gpkg, layer_name, class_name = _parse_gpkg_source(shp_str)
            slug = Path(gpkg).parent.name
            color_idx = _resolve_color_idx(global_color_map, class_name, shp_str, layer_name, logger)
            # Seuil par entité (= seuil du run qui a binné conf_bin) ; repli sur le
            # seuil global pour un slug absent de la map (run legacy / sécurité).
            layer_conf = min_conf_by_slug.get(slug, confidence_threshold)
            layer = build_detection_vector_layer(
                shp_str, layer_name, color_idx=color_idx,
                confidence_threshold=layer_conf, logger=logger,
            )
            if layer is None:
                logger.warning(f"Couche détection invalide, ignorée: {shp_str}")
                continue
            _apply_validation_form(layer, all_classes or [], logger)
            if slug in derived_slugs:
                label = entity_labels.get(slug, slug)
                proj.addMapLayer(layer, False)
                grp = root.findGroup(label) or root.insertGroup(0, label)
                grp.addLayer(layer)
            else:
                proj.addMapLayer(layer, False)
                root.addLayer(layer)
            ext = layer.extent()
            if ext is not None and not ext.isEmpty():
                extent.combineExtentWith(ext)

        # 2) Rasters VRT (en bas de l'arbre).
        _add_rasters(proj, root, vrt_paths, logger)

        # 3) Étendue initiale du projet (zoom sur les données à la réouverture).
        try:
            if crs.isValid() and not extent.isNull() and not extent.isEmpty():
                extent.scale(1.05)
                proj.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, crs))
        except Exception:
            pass

        if proj.write():
            logger.info(f"Projet QGIS de validation écrit: {qgs_path}")
            return qgs_path
        logger.warning(f"Échec de l'écriture du projet QGIS: {qgs_path}")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Écriture du projet QGIS de validation échouée: {e}")
        return None
