"""Chargement des couches résultat dans le projet QGIS (raster VRT + vecteurs).

Porté depuis ``main_dialog._load_layers_to_project`` pour être partagé par la
V2 (``run_view``). Fonctions QGIS-side : ne pas importer hors QGIS.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from ..app.progress_reporter import USER_INFO


def _apply_cluster_style(layer, logger: logging.Logger) -> None:
    """Hachures croisées noires + contour : style des zones de clustering."""
    try:
        from qgis.core import (
            QgsFillSymbol,
            QgsLinePatternFillSymbolLayer,
            QgsSimpleLineSymbolLayer,
            QgsSingleSymbolRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        symbol = QgsFillSymbol()
        symbol.deleteSymbolLayer(0)
        for angle in (45, 135):
            hatch = QgsLinePatternFillSymbolLayer()
            hatch.setLineAngle(angle)
            hatch.setDistance(3.0)
            hatch.setLineWidth(0.4)
            hatch.setColor(QColor(0, 0, 0))
            symbol.appendSymbolLayer(hatch)
        outline = QgsSimpleLineSymbolLayer()
        outline.setColor(QColor(0, 0, 0))
        outline.setWidth(0.6)
        symbol.appendSymbolLayer(outline)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible d'appliquer le style cluster: {e}")


def _apply_confidence_style(
    layer, color_idx: int, get_color_for_confidence_fn, confidence_threshold: float,
    logger: logging.Logger,
) -> None:
    """Symbologie catégorisée par tranche de confiance (contour coloré)."""
    try:
        from qgis.core import (
            QgsCategorizedSymbolRenderer,
            QgsFillSymbol,
            QgsRendererCategory,
        )

        from ..pipeline.cv.class_utils import compute_confidence_bins

        bins = compute_confidence_bins(max(0.0, float(confidence_threshold or 0.0)))
        categories = []
        for b in bins:
            r, g, bl = get_color_for_confidence_fn(color_idx, b["repr"])
            symbol = QgsFillSymbol.createSimple({
                "color": "0,0,0,0",
                "outline_color": f"{r},{g},{bl},255",
                "outline_width": "0.6",
                "outline_style": "solid",
            })
            categories.append(QgsRendererCategory(b["label"], symbol, b["label"]))
        if not categories:
            return
        layer.setRenderer(QgsCategorizedSymbolRenderer("conf_bin", categories))
        layer.triggerRepaint()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible d'appliquer le style: {e}")


def load_result_layers(
    vrt_paths: list,
    shapefile_paths: list,
    class_colors: Optional[list],
    logger: logging.Logger,
    confidence_threshold: float = 0.0,
) -> None:
    """Charge les VRT (raster) et vecteurs (détections) dans le projet QGIS,
    applique la symbologie, puis zoome sur l'étendue combinée (déféré)."""
    try:
        from qgis.core import (
            QgsProject,
            QgsRasterLayer,
            QgsRectangle,
            QgsVectorLayer,
        )

        from ..pipeline.cv.class_utils import BASE_COLOR_PALETTE, get_color_for_confidence

        project = QgsProject.instance()
        loaded_count = 0
        combined_extent = QgsRectangle()
        loaded_layers: List = []
        class_colors = class_colors or []

        global_color_map: dict = {}
        if class_colors and len(class_colors) == 1 and isinstance(class_colors[0], dict):
            global_color_map = class_colors[0]
            class_colors = []

        # ── Rasters (VRT d'indices) ──
        for vrt_path in vrt_paths:
            if not vrt_path:
                continue
            vrt_path_str = str(vrt_path)
            _vp = Path(vrt_path_str)
            rvt_type = _vp.parent.parent.name if _vp.parent.name == "tif" else _vp.parent.name
            layer_name = f"index_{rvt_type}" if rvt_type else "index"

            if _reuse_existing(project, layer_name, vrt_path_str, loaded_layers, combined_extent):
                logger.info(f"Couche raster déjà présente: {layer_name}")
                continue

            layer = QgsRasterLayer(vrt_path_str, layer_name, "gdal")
            if layer.isValid():
                _ensure_layer_crs(layer, logger)
                project.addMapLayer(layer)
                loaded_layers.append(layer)
                combined_extent = _combine(combined_extent, layer.extent())
                loaded_count += 1
                logger.info(f"Couche raster chargée: {layer_name}")
            else:
                logger.warning(f"Impossible de charger le VRT: {vrt_path_str}")

        # ── Vecteurs (détections : shapefile ou GPKG) ──
        for shp_path in shapefile_paths:
            if not shp_path:
                continue
            shp_path_str = str(shp_path)
            if "|layername=" in shp_path_str:
                _gpkg, layer_name = shp_path_str.split("|layername=", 1)
                class_name = layer_name
            else:
                layer_name = Path(shp_path_str).stem
                parts = layer_name.split("_")
                class_name = (
                    "_".join(parts[2:])
                    if len(parts) >= 3 and parts[0] == "detections"
                    else layer_name
                )
            ogr_source = shp_path_str

            color_idx = _resolve_color_idx(global_color_map, class_name, ogr_source, layer_name, logger)

            if _reuse_existing(project, layer_name, ogr_source, loaded_layers, combined_extent):
                logger.info(f"Couche vecteur déjà présente: {layer_name}")
                continue

            layer = QgsVectorLayer(ogr_source, layer_name, "ogr")
            if layer.isValid():
                _ensure_layer_crs(layer, logger)
                if layer.fields().indexFromName("nb_detect") >= 0:
                    _apply_cluster_style(layer, logger)
                else:
                    _apply_confidence_style(
                        layer, color_idx, get_color_for_confidence, confidence_threshold, logger
                    )
                project.addMapLayer(layer)
                loaded_layers.append(layer)
                combined_extent = _combine(combined_extent, layer.extent())
                loaded_count += 1
                base_color = BASE_COLOR_PALETTE[color_idx % len(BASE_COLOR_PALETTE)]
                logger.info(
                    f"Couche vecteur chargée: {layer_name} "
                    f"(classe={class_name}, couleur={color_idx} RGB{base_color})"
                )
            else:
                logger.warning(f"Impossible de charger la couche: {ogr_source}")

        if loaded_count > 0:
            from ..app.user_narrator import _human_count
            logger.log(
                USER_INFO,
                f"📂 {_human_count(loaded_count, 'couche ajoutée', 'couches ajoutées')} au projet QGIS",
            )

        if loaded_layers:
            _deferred_zoom(loaded_layers, project, logger)

    except Exception as e:  # noqa: BLE001
        logger.error(f"Erreur lors du chargement des couches: {e}")


# ── helpers ─────────────────────────────────────────────────────────


def _combine(extent, new_extent):
    if extent.isNull():
        return new_extent
    extent.combineExtentWith(new_extent)
    return extent


def _ensure_layer_crs(layer, logger, fallback_authid: str = "EPSG:2154") -> None:
    """Affecte le CRS canonique du pipeline si la couche n'en a pas d'exploitable.

    Un CRS local « unnamed » (ex. MNT que PDAL a émis sans georéférencement
    reconnu) est invalide ou dépourvu de code d'autorité (``authid`` vide) côté
    QGIS. Comme tout le pipeline est en Lambert-93, on lui **affecte** EPSG:2154
    (sans reprojeter — les coordonnées sont déjà correctes), ce qui évite l'erreur
    « Pas de transformation disponible entre unnamed et EPSG:2154 » au zoom final.
    Garde-fou défensif : la correction de fond se fait à la source (``mnt.py``).
    """
    try:
        from qgis.core import QgsCoordinateReferenceSystem

        crs = layer.crs()
        if crs.isValid() and crs.authid():
            return  # CRS exploitable (code d'autorité présent) → ne pas toucher
        target = QgsCoordinateReferenceSystem(fallback_authid)
        if target.isValid():
            layer.setCrs(target)
            logger.warning(f"CRS absent/local sur « {layer.name()} » → affecté {fallback_authid}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible de fixer le CRS de « {layer.name()} »: {e}")


def _reuse_existing(project, layer_name, source, loaded_layers, combined_extent) -> bool:
    """Si une couche de même nom + source existe déjà, la réutilise."""
    for existing in project.mapLayersByName(layer_name):
        if existing.source() == source:
            loaded_layers.append(existing)
            _combine(combined_extent, existing.extent())
            return True
    return False


def _resolve_color_idx(global_color_map, class_name, ogr_source, layer_name, logger) -> int:
    if global_color_map and class_name in global_color_map:
        return global_color_map[class_name]
    if global_color_map:
        for cname, cidx in global_color_map.items():
            if cname.lower() in class_name.lower():
                return cidx
        return 0
    # Pas de map globale : lire l'attribut conf_color de la première entité.
    try:
        from qgis.core import QgsVectorLayer
        temp = QgsVectorLayer(ogr_source, "temp", "ogr")
        if temp.isValid() and temp.featureCount() > 0:
            for feat in temp.getFeatures():
                val = feat.attribute("conf_color")
                if val:
                    m = re.match(r"color(\d+)_", str(val))
                    if m:
                        return int(m.group(1))
                break
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible d'extraire conf_color de {layer_name}: {e}")
    return 0


def _deferred_zoom(loaded_layers, project, logger) -> None:
    """Zoom différé (QTimer 0) : le canvas n'a pas encore synchronisé son CRS
    dans le même tour de boucle que addMapLayer."""
    try:
        from qgis.PyQt.QtCore import QTimer
        from qgis.utils import iface

        if not (iface and iface.mapCanvas()):
            logger.info("iface indisponible — zoom non effectué")
            return
        layers_for_zoom = list(loaded_layers)

        def _do_zoom():
            try:
                from qgis.core import QgsCoordinateTransform, QgsRectangle
                canvas = iface.mapCanvas()
                canvas_crs = canvas.mapSettings().destinationCrs()
                combined = QgsRectangle()
                combined.setMinimal()
                for lay in layers_for_zoom:
                    ext = lay.extent()
                    if ext.isNull() or ext.isEmpty():
                        continue
                    l_crs = lay.crs()
                    if l_crs.isValid() and canvas_crs.isValid() and l_crs != canvas_crs:
                        try:
                            xf = QgsCoordinateTransform(l_crs, canvas_crs, project)
                            ext = xf.transformBoundingBox(ext)
                        except Exception:
                            continue
                    if not (ext.isNull() or ext.isEmpty()):
                        combined.combineExtentWith(ext)
                if not (combined.isNull() or combined.isEmpty()):
                    combined.scale(1.05)
                    canvas.setExtent(combined)
                    canvas.refresh()
                    logger.log(USER_INFO, "Zoom sur l'étendue des résultats")
                    return
                iface.setActiveLayer(layers_for_zoom[0])
                iface.zoomToActiveLayer()
                logger.log(USER_INFO, "Zoom sur l'étendue des résultats")
            except Exception as ze:  # noqa: BLE001
                logger.warning(f"Impossible de zoomer: {ze}")

        QTimer.singleShot(0, _do_zoom)
    except Exception as zoom_err:  # noqa: BLE001
        logger.warning(f"Impossible de programmer le zoom: {zoom_err}")
