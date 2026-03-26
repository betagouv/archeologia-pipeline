"""
Génération du projet QGIS (.qgs) pour la validation des détections CV.

Extrait de conversion_shp.py pour améliorer la lisibilité.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring, parse as et_parse
import xml.dom.minidom as minidom

import geopandas as gpd

from .class_utils import get_color_for_confidence

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers internes                                                    #
# ------------------------------------------------------------------ #

def _find_tif_dir(index_root: Path) -> Path:
    """Cherche le dossier ``tif/`` contenant les dalles RVT."""
    tif_dir = index_root / "tif"
    if tif_dir.exists():
        return tif_dir
    # Chercher un dossier "tif" frère ou dans les parents
    for parent in [index_root] + list(index_root.parents)[:3]:
        candidate = parent / "tif"
        if candidate.exists() and candidate.is_dir():
            return candidate
    # Chercher index.vrt via rglob
    for search_root in [index_root] + list(index_root.parents)[:3]:
        try:
            vrt_candidates = list(search_root.glob("**/tif/index.vrt"))
            if vrt_candidates:
                logger.info(f"[QGIS Project] VRT trouvé via recherche: {vrt_candidates[0]}")
                return vrt_candidates[0].parent
        except Exception:
            continue
    return tif_dir  # fallback


def _compute_combined_extent(shapefile_paths: List[str]) -> Optional[List[float]]:
    """Calcule l'étendue combinée [minx, miny, maxx, maxy] de tous les shapefiles."""
    combined: Optional[List[float]] = None
    for shp in shapefile_paths:
        try:
            gdf = gpd.read_file(shp)
            if not gdf.empty:
                bounds = gdf.total_bounds
                if combined is None:
                    combined = list(bounds)
                else:
                    combined[0] = min(combined[0], bounds[0])
                    combined[1] = min(combined[1], bounds[1])
                    combined[2] = max(combined[2], bounds[2])
                    combined[3] = max(combined[3], bounds[3])
        except Exception:
            pass
    return combined


def _load_qml_style(style_path: Path):
    """Charge le style QML et retourne (renderer, selection, customprops) ou (None, None, None)."""
    try:
        if style_path.exists():
            tree = et_parse(str(style_path))
            root = tree.getroot()
            logger.info(f"Style QML des détections chargé depuis {style_path}")
            return root.find('renderer-v2'), root.find('selection'), root.find('customproperties')
        else:
            logger.info("Style QML des détections introuvable")
    except Exception as e:
        logger.warning(f"Échec du chargement du style QML des détections: {e}")
    return None, None, None


def _apply_cluster_symbology(maplayer_el: Element) -> None:
    """Applique un style quadrillage noir (cross hatch) pour les couches de cluster."""
    try:
        renderer = SubElement(
            maplayer_el,
            'renderer-v2',
            attrib={
                'type': 'singleSymbol',
                'forceraster': '0',
                'symbollevels': '0',
                'enableorderby': '0',
            },
        )
        symbols = SubElement(renderer, 'symbols')
        sym = SubElement(symbols, 'symbol', attrib={
            'type': 'fill', 'name': '0', 'alpha': '1',
            'clip_to_extent': '1', 'force_rhr': '0',
        })
        # Quadrillage noir (LinePatternFill)
        fill_layer = SubElement(sym, 'layer', attrib={
            'class': 'LinePatternFill', 'enabled': '1', 'locked': '0', 'pass': '0',
        })
        opt = SubElement(fill_layer, 'Option', attrib={'type': 'Map'})
        SubElement(opt, 'Option', attrib={'name': 'angle', 'value': '45', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'color', 'value': '0,0,0,255', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'distance', 'value': '3', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'distance_unit', 'value': 'MM', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'line_width', 'value': '0.4', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})
        SubElement(opt, 'Option', attrib={'name': 'offset', 'value': '0', 'type': 'QString'})
        # Sous-symbole ligne noire
        sub_sym = SubElement(fill_layer, 'symbol', attrib={
            'type': 'line', 'name': '@0@0', 'alpha': '1',
            'clip_to_extent': '1', 'force_rhr': '0',
        })
        sub_layer = SubElement(sub_sym, 'layer', attrib={
            'class': 'SimpleLine', 'enabled': '1', 'locked': '0', 'pass': '0',
        })
        sub_opt = SubElement(sub_layer, 'Option', attrib={'type': 'Map'})
        SubElement(sub_opt, 'Option', attrib={'name': 'line_color', 'value': '0,0,0,255', 'type': 'QString'})
        SubElement(sub_opt, 'Option', attrib={'name': 'line_style', 'value': 'solid', 'type': 'QString'})
        SubElement(sub_opt, 'Option', attrib={'name': 'line_width', 'value': '0.4', 'type': 'QString'})
        SubElement(sub_opt, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})
        # Deuxième direction du quadrillage (perpendiculaire)
        fill_layer2 = SubElement(sym, 'layer', attrib={
            'class': 'LinePatternFill', 'enabled': '1', 'locked': '0', 'pass': '0',
        })
        opt2 = SubElement(fill_layer2, 'Option', attrib={'type': 'Map'})
        SubElement(opt2, 'Option', attrib={'name': 'angle', 'value': '135', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'color', 'value': '0,0,0,255', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'distance', 'value': '3', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'distance_unit', 'value': 'MM', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'line_width', 'value': '0.4', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})
        SubElement(opt2, 'Option', attrib={'name': 'offset', 'value': '0', 'type': 'QString'})
        sub_sym2 = SubElement(fill_layer2, 'symbol', attrib={
            'type': 'line', 'name': '@0@1', 'alpha': '1',
            'clip_to_extent': '1', 'force_rhr': '0',
        })
        sub_layer2 = SubElement(sub_sym2, 'layer', attrib={
            'class': 'SimpleLine', 'enabled': '1', 'locked': '0', 'pass': '0',
        })
        sub_opt2 = SubElement(sub_layer2, 'Option', attrib={'type': 'Map'})
        SubElement(sub_opt2, 'Option', attrib={'name': 'line_color', 'value': '0,0,0,255', 'type': 'QString'})
        SubElement(sub_opt2, 'Option', attrib={'name': 'line_style', 'value': 'solid', 'type': 'QString'})
        SubElement(sub_opt2, 'Option', attrib={'name': 'line_width', 'value': '0.4', 'type': 'QString'})
        SubElement(sub_opt2, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})
        # Contour noir
        outline_layer = SubElement(sym, 'layer', attrib={
            'class': 'SimpleLine', 'enabled': '1', 'locked': '0', 'pass': '0',
        })
        outline_opt = SubElement(outline_layer, 'Option', attrib={'type': 'Map'})
        SubElement(outline_opt, 'Option', attrib={'name': 'line_color', 'value': '0,0,0,255', 'type': 'QString'})
        SubElement(outline_opt, 'Option', attrib={'name': 'line_style', 'value': 'solid', 'type': 'QString'})
        SubElement(outline_opt, 'Option', attrib={'name': 'line_width', 'value': '0.6', 'type': 'QString'})
        SubElement(outline_opt, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})
    except Exception as e:
        logger.warning(f"Impossible d'appliquer la symbologie cluster: {e}")


def _apply_confidence_symbology(maplayer_el: Element, color_index: int = 0) -> None:
    """Applique un renderer catégorisé QGIS sur le champ conf_bin."""
    try:
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        palette_rgb = []
        for conf in confidence_levels:
            r, g, b = get_color_for_confidence(color_index, conf)
            palette_rgb.append(f"{r},{g},{b}")

        renderer = SubElement(
            maplayer_el,
            'renderer-v2',
            attrib={
                'type': 'categorizedSymbol',
                'attr': 'conf_bin',
                'forceraster': '0',
                'referencescale': '-1',
                'symbollevels': '0',
                'enableorderby': '0',
            },
        )

        categories = SubElement(renderer, 'categories')
        bins = ['[0:0.2[', '[0.2:0.4[', '[0.4:0.6[', '[0.6:0.8[', '[0.8:1]']
        for i, b in enumerate(bins):
            SubElement(categories, 'category', attrib={'value': b, 'label': b, 'symbol': str(i), 'render': 'true'})

        symbols = SubElement(renderer, 'symbols')
        for i, rgb in enumerate(palette_rgb):
            sym = SubElement(symbols, 'symbol', attrib={'type': 'fill', 'name': str(i), 'alpha': '1', 'clip_to_extent': '1', 'force_rhr': '0'})
            layer = SubElement(sym, 'layer', attrib={'class': 'SimpleLine', 'enabled': '1', 'locked': '0', 'pass': '0'})
            opt = SubElement(layer, 'Option', attrib={'type': 'Map'})
            SubElement(opt, 'Option', attrib={'name': 'line_color', 'value': f"{rgb},255", 'type': 'QString'})
            SubElement(opt, 'Option', attrib={'name': 'line_style', 'value': 'solid', 'type': 'QString'})
            SubElement(opt, 'Option', attrib={'name': 'line_width', 'value': '0.6', 'type': 'QString'})
            SubElement(opt, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})

    except Exception as sym_e:
        logger.warning(f"Impossible d'appliquer la symbologie confiance: {sym_e}")


def _add_vector_layer(
    *,
    ml: Element,
    shp_ds: str,
    layer_id: str,
    crs: str,
    all_classes: List[str],
    shp_idx: int,
    class_colors: Optional[Any],
    style_selection: Optional[Element],
    style_customprops: Optional[Element],
    global_color_map: Optional[Dict[str, int]] = None,
    cluster_class_names: Optional[set] = None,
) -> None:
    """Configure un maplayer vecteur (shapefile de détections) dans le projet QGIS."""
    ml.set('dataSourcePaletteIndex', str(shp_idx % 6))
    SubElement(ml, 'id').text = layer_id
    SubElement(ml, 'layername').text = layer_id
    SubElement(ml, 'datasource').text = shp_ds
    SubElement(ml, 'provider').text = 'ogr'
    try:
        srs = SubElement(ml, 'srs')
        sref = SubElement(srs, 'spatialrefsys')
        SubElement(sref, 'authid').text = crs
    except Exception:
        pass
    SubElement(ml, 'displayfield').text = 'model_pred'
    SubElement(ml, 'previewExpression').text = '"model_pred"'

    # fieldConfiguration
    fcfg = SubElement(ml, 'fieldConfiguration')

    # validation
    f_val = SubElement(fcfg, 'field', attrib={'name': 'validation'})
    ew_val = SubElement(f_val, 'editWidget', attrib={'type': 'ValueMap'})
    cfg_val = SubElement(ew_val, 'config')
    opt_val_root = SubElement(cfg_val, 'Option', attrib={'type': 'Map'})
    opt_val_map = SubElement(opt_val_root, 'Option', attrib={'name': 'map', 'type': 'List'})
    for label in ["oui", "non", "peut-être"]:
        item = SubElement(opt_val_map, 'Option', attrib={'type': 'Map'})
        SubElement(item, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
    SubElement(opt_val_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'true'})

    # model_pred
    f_pred = SubElement(fcfg, 'field', attrib={'name': 'model_pred'})
    ew_pred = SubElement(f_pred, 'editWidget', attrib={'type': 'ValueMap'})
    cfg_pred = SubElement(ew_pred, 'config')
    opt_pred_root = SubElement(cfg_pred, 'Option', attrib={'type': 'Map'})
    opt_pred_map = SubElement(opt_pred_root, 'Option', attrib={'name': 'map', 'type': 'Map'})
    for label in all_classes:
        SubElement(opt_pred_map, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
    SubElement(opt_pred_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'false'})

    # model_name
    SubElement(fcfg, 'field', attrib={'name': 'model_name'})

    # corr_pred
    f_corr = SubElement(fcfg, 'field', attrib={'name': 'corr_pred'})
    ew_corr = SubElement(f_corr, 'editWidget', attrib={'type': 'ValueMap'})
    cfg_corr = SubElement(ew_corr, 'config')
    opt_corr_root = SubElement(cfg_corr, 'Option', attrib={'type': 'Map'})
    opt_corr_map = SubElement(opt_corr_root, 'Option', attrib={'name': 'map', 'type': 'Map'})
    for label in all_classes:
        SubElement(opt_corr_map, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
    SubElement(opt_corr_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'true'})

    # aliases
    aliases = SubElement(ml, 'aliases')
    SubElement(aliases, 'alias', attrib={'field': 'corr_pred', 'index': '-1', 'name': 'correction_prediction'})
    SubElement(aliases, 'alias', attrib={'field': 'confidence', 'index': '-1', 'name': 'confiance'})
    SubElement(aliases, 'alias', attrib={'field': 'conf_bin', 'index': '-1', 'name': 'tranche_confiance'})
    SubElement(aliases, 'alias', attrib={'field': 'conf_color', 'index': '-1', 'name': 'couleur_confiance'})
    SubElement(aliases, 'alias', attrib={'field': 'model_name', 'index': '-1', 'name': 'modele_detection'})

    # attributeEditorForm
    aef = SubElement(ml, 'attributeEditorForm')
    aec = SubElement(aef, 'attributeEditorContainer', attrib={'name': '', 'groupBox': '0', 'visibilityExpressionEnabled': '0'})
    SubElement(aec, 'attributeEditorField', attrib={'name': 'model_name', 'index': '-1', 'showLabel': '1'})
    SubElement(aec, 'attributeEditorField', attrib={'name': 'model_pred', 'index': '-1', 'showLabel': '1'})
    SubElement(aec, 'attributeEditorField', attrib={'name': 'validation', 'index': '-1', 'showLabel': '1'})
    aef_corr = SubElement(aec, 'attributeEditorField', attrib={'name': 'corr_pred', 'index': '-1', 'showLabel': '1', 'visibilityExpressionEnabled': '1'})
    SubElement(aef_corr, 'visibilityExpression').text = '"validation" IN (\'non\', \'peut-être\')'
    SubElement(aec, 'attributeEditorField', attrib={'name': 'confidence', 'index': '-1', 'showLabel': '1'})
    SubElement(aec, 'attributeEditorField', attrib={'name': 'conf_bin', 'index': '-1', 'showLabel': '1'})
    SubElement(aec, 'attributeEditorField', attrib={'name': 'conf_color', 'index': '-1', 'showLabel': '1'})

    SubElement(ml, 'editorlayout').text = 'tablayout'

    editable = SubElement(ml, 'editable')
    SubElement(editable, 'field', attrib={'name': 'validation', 'editable': '1'})
    SubElement(editable, 'field', attrib={'name': 'model_pred', 'editable': '0'})
    SubElement(editable, 'field', attrib={'name': 'model_name', 'editable': '0'})
    SubElement(editable, 'field', attrib={'name': 'corr_pred', 'editable': '1'})
    SubElement(editable, 'field', attrib={'name': 'confidence', 'editable': '0'})
    SubElement(editable, 'field', attrib={'name': 'conf_bin', 'editable': '0'})
    SubElement(editable, 'field', attrib={'name': 'conf_color', 'editable': '0'})

    # Symbologie par confiance
    shp_color_index = shp_idx
    # Extraire le nom de classe depuis le layer_id (format: detections_{RVT}_{class_name})
    _lid_parts = layer_id.split("_")
    _class_from_lid = "_".join(_lid_parts[2:]) if len(_lid_parts) >= 3 and _lid_parts[0] == "detections" else ""
    if global_color_map and _class_from_lid and _class_from_lid in global_color_map:
        shp_color_index = global_color_map[_class_from_lid]
    elif global_color_map:
        # Fallback: chercher par correspondance partielle
        for cls_name, cidx in global_color_map.items():
            if cls_name.lower() in layer_id.lower():
                shp_color_index = cidx
                break
    elif class_colors:
        for cls_idx, cls_name in enumerate(all_classes):
            if cls_name.lower() in layer_id.lower():
                if cls_idx < len(class_colors):
                    shp_color_index = class_colors[cls_idx]
                else:
                    shp_color_index = cls_idx
                break
    # Détecter si c'est une couche cluster
    _is_cluster = False
    if cluster_class_names:
        for cname in cluster_class_names:
            if cname.lower() in layer_id.lower():
                _is_cluster = True
                break

    if _is_cluster:
        _apply_cluster_symbology(ml)
    else:
        _apply_confidence_symbology(ml, shp_color_index)

    if style_selection is not None:
        ml.append(copy.deepcopy(style_selection))
    if style_customprops is not None:
        ml.append(copy.deepcopy(style_customprops))


def _add_raster_vrt_layer(
    *,
    layer_tree: Element,
    maplayers: Element,
    vrt_path: Path,
    index_root: Path,
    crs: str,
) -> None:
    """Ajoute le VRT comme couche raster unique dans le projet."""
    try:
        vrt_ds_rel = os.path.relpath(str(vrt_path.resolve()), start=str(index_root.resolve()))
        vrt_ds_rel = vrt_ds_rel.replace('\\', '/')
    except Exception:
        vrt_ds_rel = str(vrt_path.resolve()).replace('\\', '/')

    vrt_ds = "./" + vrt_ds_rel if not vrt_ds_rel.startswith('.') else vrt_ds_rel
    # ID et nom uniques basés sur le dossier parent du tif/ (ex: LD, SVF, etc.)
    rvt_type = vrt_path.parent.parent.name if vrt_path.parent.name == "tif" else vrt_path.parent.name
    raster_id = f"index_rvt_{rvt_type}"
    raster_name = f"Dalles RVT {rvt_type} (index)"

    SubElement(layer_tree, 'layer-tree-layer', attrib={
        'id': raster_id, 'name': raster_name,
        'checked': 'Qt::Checked', 'expanded': '1',
        'source': vrt_ds, 'providerKey': 'gdal',
        'legend_exp': '', 'patch_size': '-1,-1',
        'legend_split_behavior': '0',
    })

    ml_raster = SubElement(maplayers, 'maplayer', attrib={
        'type': 'raster', 'autoRefreshMode': 'Disabled',
        'autoRefreshTime': '0', 'refreshOnNotifyEnabled': '0',
        'refreshOnNotifyMessage': '', 'hasScaleBasedVisibilityFlag': '0',
        'styleCategories': 'AllStyleCategories',
        'minScale': '1e+08', 'maxScale': '0', 'legendPlaceholderImage': '',
    })
    SubElement(ml_raster, 'id').text = raster_id
    SubElement(ml_raster, 'datasource').text = vrt_ds
    SubElement(ml_raster, 'layername').text = raster_name
    SubElement(ml_raster, 'provider').text = 'gdal'

    flags = SubElement(ml_raster, 'flags')
    SubElement(flags, 'Identifiable').text = '1'
    SubElement(flags, 'Removable').text = '1'
    SubElement(flags, 'Searchable').text = '1'
    SubElement(flags, 'Private').text = '0'

    try:
        srs_r = SubElement(ml_raster, 'srs')
        sref_r = SubElement(srs_r, 'spatialrefsys', attrib={'nativeFormat': 'Wkt'})
        SubElement(sref_r, 'authid').text = crs
        SubElement(sref_r, 'srid').text = crs.split(':')[1] if ':' in crs else '2154'
    except Exception:
        pass

    pipe = SubElement(ml_raster, 'pipe')
    provider_elem = SubElement(pipe, 'provider')
    SubElement(provider_elem, 'resampling', attrib={
        'zoomedInResamplingMethod': 'nearestNeighbour',
        'zoomedOutResamplingMethod': 'nearestNeighbour',
        'maxOversampling': '2', 'enabled': 'false',
    })
    renderer = SubElement(pipe, 'rasterrenderer', attrib={
        'type': 'singlebandgray', 'gradient': 'BlackToWhite',
        'grayBand': '1', 'opacity': '1', 'alphaBand': '-1', 'nodataColor': '',
    })
    SubElement(renderer, 'rasterTransparency')
    minmax = SubElement(renderer, 'minMaxOrigin')
    SubElement(minmax, 'limits').text = 'MinMax'
    SubElement(minmax, 'extent').text = 'WholeRaster'
    SubElement(minmax, 'statAccuracy').text = 'Estimated'
    SubElement(minmax, 'cumulativeCutLower').text = '0.02'
    SubElement(minmax, 'cumulativeCutUpper').text = '0.98'
    SubElement(minmax, 'stdDevFactor').text = '2'
    ce = SubElement(renderer, 'contrastEnhancement')
    SubElement(ce, 'minValue').text = '0'
    SubElement(ce, 'maxValue').text = '255'
    SubElement(ce, 'algorithm').text = 'StretchToMinimumMaximum'

    SubElement(pipe, 'brightnesscontrast', attrib={'contrast': '0', 'brightness': '0', 'gamma': '1'})
    SubElement(pipe, 'huesaturation', attrib={
        'saturation': '0', 'grayscaleMode': '0', 'invertColors': '0',
        'colorizeOn': '0', 'colorizeRed': '255', 'colorizeGreen': '128',
        'colorizeBlue': '128', 'colorizeStrength': '100',
    })
    SubElement(pipe, 'rasterresampler', attrib={'maxOversampling': '2'})
    SubElement(pipe, 'resamplingStage').text = 'resamplingFilter'
    SubElement(ml_raster, 'blendMode').text = '0'

    logger.info(f"[QGIS Project] VRT datasource={vrt_ds}")


def _add_raster_tif_layers(
    *,
    layer_tree: Element,
    maplayers: Element,
    tif_files: List[Path],
    index_root: Path,
    crs: str,
) -> None:
    """Fallback: ajoute les TIF individuellement si pas de VRT."""
    for tif_path in tif_files:
        tif_path = Path(tif_path)
        try:
            tif_ds = os.path.relpath(str(tif_path.resolve()), start=str(index_root.resolve()))
            tif_ds = tif_ds.replace('\\', '/')
        except Exception:
            tif_ds = str(tif_path.resolve())
        raster_id = tif_path.stem
        SubElement(layer_tree, 'layer-tree-layer', attrib={'id': raster_id, 'name': raster_id})
        ml_raster = SubElement(maplayers, 'maplayer', attrib={'type': 'raster'})
        SubElement(ml_raster, 'id').text = raster_id
        SubElement(ml_raster, 'layername').text = raster_id
        SubElement(ml_raster, 'datasource').text = tif_ds
        SubElement(ml_raster, 'provider').text = 'gdal'
        try:
            srs_r = SubElement(ml_raster, 'srs')
            sref_r = SubElement(srs_r, 'spatialrefsys')
            SubElement(sref_r, 'authid').text = crs
        except Exception:
            pass


def _add_map_extent(project: Element, combined_extent: List[float], crs: str) -> None:
    """Ajoute l'étendue initiale du projet avec une marge de 5%."""
    try:
        width = combined_extent[2] - combined_extent[0]
        height = combined_extent[3] - combined_extent[1]
        margin_x = width * 0.05
        margin_y = height * 0.05

        mapcanvas = SubElement(project, 'mapcanvas', attrib={'name': 'theMapCanvas', 'annotationsVisible': '1'})
        extent = SubElement(mapcanvas, 'extent')
        SubElement(extent, 'xmin').text = str(combined_extent[0] - margin_x)
        SubElement(extent, 'ymin').text = str(combined_extent[1] - margin_y)
        SubElement(extent, 'xmax').text = str(combined_extent[2] + margin_x)
        SubElement(extent, 'ymax').text = str(combined_extent[3] + margin_y)

        dest_crs = SubElement(mapcanvas, 'destinationsrs')
        sref_canvas = SubElement(dest_crs, 'spatialrefsys', attrib={'nativeFormat': 'Wkt'})
        SubElement(sref_canvas, 'authid').text = crs
    except Exception as e:
        logger.warning(f"Impossible de définir l'étendue du projet: {e}")


# ------------------------------------------------------------------ #
#  Point d'entrée public                                               #
# ------------------------------------------------------------------ #

def generate_qgs_project(
    *,
    created_shapefiles: List[str],
    output_shapefile: str,
    all_classes: List[str],
    crs: str = "EPSG:2154",
    class_colors: Optional[Any] = None,
    global_color_map: Optional[Dict[str, int]] = None,
    cluster_class_names: Optional[set] = None,
) -> Optional[Path]:
    """
    Génère un projet QGIS ``detections_validation.qgs`` contenant les shapefiles
    de détection et les rasters RVT.

    Returns:
        Le chemin du fichier .qgs généré, ou None en cas d'échec.
    """
    try:
        # Déterminer les racines RVT depuis tous les shapefiles
        index_roots: List[Path] = []
        for shp in created_shapefiles:
            shp_path = Path(shp)
            shp_parent = shp_path.parent
            root = shp_parent.parent if shp_parent.name.lower() in {"shapefiles", "shp"} else shp_parent
            if root not in index_roots:
                index_roots.append(root)
        if not index_roots:
            index_roots = [Path(output_shapefile).parent]

        # Racine commune pour le projet (ancêtre commun de tous les index_roots)
        if len(index_roots) == 1:
            project_root = index_roots[0]
        else:
            # Trouver l'ancêtre commun (ex: results/RVT/)
            try:
                common = Path(os.path.commonpath([str(r.resolve()) for r in index_roots]))
                project_root = common
            except Exception:
                project_root = index_roots[0]

        # Collecter tous les dossiers tif/ depuis chaque index_root
        all_tif_dirs: List[Path] = []
        for root in index_roots:
            tif_dir = _find_tif_dir(root)
            if tif_dir.exists() and tif_dir.is_dir() and tif_dir not in all_tif_dirs:
                all_tif_dirs.append(tif_dir)

        project = Element('qgis', attrib={'version': '3.34.0', 'projectname': 'detections_validation'})

        combined_extent = _compute_combined_extent(created_shapefiles)

        try:
            props = SubElement(project, 'properties')
            paths = SubElement(props, 'Paths')
            SubElement(paths, 'Absolute').text = '0'
        except Exception:
            pass

        try:
            project_crs = SubElement(project, 'projectCrs')
            SubElement(project_crs, 'authid').text = crs
        except Exception:
            pass

        layer_tree = SubElement(project, 'layer-tree-group', attrib={'name': ''})
        maplayers = SubElement(project, 'projectlayers')

        # Charger le style QML
        style_path = Path(__file__).parents[2] / 'resources' / 'styles' / 'style_detections.qml'
        _, style_selection, style_customprops = _load_qml_style(style_path)

        # 1) Shapefiles de détection (au-dessus des rasters)
        for shp_idx, shp in enumerate(created_shapefiles):
            shp_path = Path(shp)
            try:
                shp_ds = os.path.relpath(str(shp_path.resolve()), start=str(project_root.resolve()))
                shp_ds = shp_ds.replace('\\', '/')
            except Exception:
                shp_ds = str(shp_path.resolve())
            layer_id = shp_path.stem
            SubElement(layer_tree, 'layer-tree-layer', attrib={'id': layer_id, 'name': layer_id})

            ml = SubElement(maplayers, 'maplayer', attrib={'type': 'vector'})
            _add_vector_layer(
                ml=ml,
                shp_ds=shp_ds,
                layer_id=layer_id,
                crs=crs,
                all_classes=all_classes,
                shp_idx=shp_idx,
                class_colors=class_colors,
                style_selection=style_selection,
                style_customprops=style_customprops,
                global_color_map=global_color_map,
                cluster_class_names=cluster_class_names,
            )

        # 2) Rasters (VRT ou TIF individuels) — pour chaque dossier tif/ trouvé
        for tif_dir in all_tif_dirs:
            tif_files = sorted(tif_dir.glob("*.tif"))
            logger.info(f"[QGIS Project] tif_dir={tif_dir}, tif_files={len(tif_files)}")
            vrt_path = tif_dir / "index.vrt"

            if vrt_path.exists():
                _add_raster_vrt_layer(
                    layer_tree=layer_tree,
                    maplayers=maplayers,
                    vrt_path=vrt_path,
                    index_root=project_root,
                    crs=crs,
                )
            elif tif_files:
                _add_raster_tif_layers(
                    layer_tree=layer_tree,
                    maplayers=maplayers,
                    tif_files=tif_files,
                    index_root=project_root,
                    crs=crs,
                )

        # 3) Étendue initiale
        if combined_extent is not None:
            _add_map_extent(project, combined_extent, crs)

        # Écriture du fichier .qgs
        xml_bytes = tostring(project, encoding='utf-8')
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent='  ', encoding='utf-8')
        qgs_path = project_root / 'detections_validation.qgs'
        with open(qgs_path, 'wb') as f:
            f.write(pretty)

        logger.info("════════════════════════════════════════════════════════════")
        logger.info("📋 PROJET QGIS GÉNÉRÉ")
        logger.info("════════════════════════════════════════════════════════════")
        logger.info(f"   Fichier: {qgs_path}")
        logger.info("   Ce projet contient les couches de détection et les rasters RVT.")
        logger.info("   Ouvrez-le dans QGIS pour valider les détections.")
        return qgs_path

    except Exception as e:
        logger.warning(f"Échec génération du projet QGIS: {e}")
        return None
