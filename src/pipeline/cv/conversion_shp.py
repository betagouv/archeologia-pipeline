import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import re
import json
import tempfile
import subprocess
import copy

# DIAGNOSTIC DÉTAILLÉ POUR PYINSTALLER - IMPORTS CRITIQUES
logger = logging.getLogger(__name__)

# Test des imports critiques avec diagnostic complet
try:
    logger.info("🔍 DIAGNOSTIC CONVERSION_SHP: Test des imports critiques...")

    # Test pandas d'abord
    logger.info("   📦 Import pandas...")
    import pandas as pd

    logger.info(f"   ✅ pandas importé: {pd.__version__}")

    # Test des modules pandas critiques
    critical_modules = [
        "pandas._libs.window.aggregations",
        "pandas._libs.window",
        "pandas._libs.lib",
        "pandas._libs.algos",
        "pandas._libs.groupby",
        "pandas._libs.ops",
        "pandas.core.groupby.ops",
    ]

    for mod in critical_modules:
        try:
            __import__(mod)
            logger.info(f"   ✅ {mod} disponible")
        except Exception as mod_e:
            logger.error(f"   ❌ {mod}: {mod_e}")
            logger.error(f"      Type: {type(mod_e).__name__}")

    # Test shapely
    logger.info("   📦 Import shapely...")
    from shapely.geometry import Point, Polygon

    logger.info("   ✅ shapely importé")

    # Test geopandas (le plus critique)
    logger.info("   📦 Import geopandas...")
    import geopandas as gpd

    logger.info(f"   ✅ geopandas importé: {gpd.__version__}")

    logger.info("✅ DIAGNOSTIC CONVERSION_SHP: Tous les imports réussis")

except Exception as import_e:
    logger.error(f"❌ ERREUR IMPORT CONVERSION_SHP: {import_e}")
    logger.error(f"   Type d'erreur: {type(import_e).__name__}")

    # Traceback détaillé
    import traceback

    tb_lines = traceback.format_exc().split("\n")
    for i, line in enumerate(tb_lines):
        if line.strip():
            logger.error(f"   TB[{i}]: {line}")

    # Re-raise l'erreur pour que le module ne se charge pas
    raise


def _confidence_bucket(confidence_value: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if confidence_value is None:
        return None, None
    try:
        c = float(confidence_value)
    except Exception:
        return None, None

    # Normaliser si la confiance semble être sur [0,10]
    if c > 1.0 and c <= 10.0:
        c = c / 10.0

    # Intervalles de 0.2
    if c < 0.2:
        return "[0:0.2[", "yellow"
    if c < 0.4:
        return "[0.2:0.4[", "yellow_orange"
    if c < 0.6:
        return "[0.4:0.6[", "orange"
    if c < 0.8:
        return "[0.6:0.8[", "orange_red"
    return "[0.8:1]", "red"


def read_world_file(world_file_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Lit un fichier world (.jgw) pour extraire les informations de géoréférencement

    Args:
        world_file_path (str): Chemin vers le fichier .jgw

    Returns:
        Tuple: (pixel_width, pixel_height, x_origin, y_origin) ou (None, None, None, None) si erreur
    """
    logger = logging.getLogger(__name__)

    try:
        with open(world_file_path, "r") as f:
            lines = f.readlines()

        if len(lines) >= 6:
            pixel_width = float(lines[0].strip())
            row_rotation = float(lines[1].strip())
            col_rotation = float(lines[2].strip())
            pixel_height = float(lines[3].strip())
            x_origin = float(lines[4].strip())
            y_origin = float(lines[5].strip())

            return pixel_width, pixel_height, x_origin, y_origin

    except (FileNotFoundError, ValueError, IndexError) as e:
        logger.warning(f"Impossible de lire le fichier world {world_file_path}: {e}")

    return None, None, None, None


def extract_coordinates_from_filename(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extrait les coordonnées X et Y depuis le nom de fichier d'une image RVT (méthode de fallback)

    Args:
        filename (str): Nom du fichier (ex: "LHD_FXX_0929_6613_LDO_A_LAMB93.jpg")

    Returns:
        Tuple[Optional[float], Optional[float]]: (x_anchor, y_anchor) ou (None, None) si erreur
    """
    logger = logging.getLogger(__name__)

    try:
        pattern = r"LHD_FXX_(\d{4})_(\d{4})_"
        match = re.search(pattern, filename)

        if match:
            x_km = int(match.group(1))
            y_km = int(match.group(2))

            x_anchor = x_km * 1000
            y_anchor = y_km * 1000

            return x_anchor, y_anchor

    except (ValueError, AttributeError) as e:
        logger.warning(f"Impossible d'extraire les coordonnées de {filename}: {e}")

    return None, None


def extract_tile_coordinates(filename: str) -> Optional[Tuple[str, str]]:
    """Extrait les coordonnées de tuile (XXXX_YYYY) depuis un nom de fichier."""
    try:
        parts = filename.split("_")
        if len(parts) >= 4:
            x_coord = parts[2]
            y_coord = parts[3]
            return x_coord, y_coord
    except (ValueError, IndexError) as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Impossible d'extraire les coordonnées de tuile de {filename}: {e}")

    return None, None


def calculate_neighbor_tile_keys(x_coord: str, y_coord: str) -> List[str]:
    """Calcule les clés des 8 tuiles voisines + la tuile centrale."""
    try:
        x_int = int(x_coord)
        y_int = int(y_coord)

        neighbor_keys = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_x = x_int + dx
                neighbor_y = y_int + dy
                neighbor_key = f"{neighbor_x:04d}_{neighbor_y:04d}"
                neighbor_keys.append(neighbor_key)

        return neighbor_keys

    except (ValueError, TypeError) as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Erreur lors du calcul des voisins pour {x_coord}_{y_coord}: {e}")
        return [f"{x_coord}_{y_coord}"]


def load_iou_threshold_from_config() -> float:
    """Charge le seuil IoU depuis config.json"""
    try:
        from pathlib import Path
        import json

        config_path = Path(__file__).parents[3] / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("computer_vision", {}).get("iou_threshold", 0.3)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Impossible de charger le seuil IoU depuis config.json: {e}")

    return 0.3


def load_model_name_from_config() -> str:
    try:
        from pathlib import Path
        import json

        config_path = Path(__file__).parents[3] / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("computer_vision", {}).get("selected_model", "")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Impossible de charger le nom du modèle depuis config.json: {e}")

    return ""


def _polygon_iou(poly_a, poly_b) -> Optional[float]:
    try:
        if poly_a is None or poly_b is None:
            return None
        if not getattr(poly_a, "bounds", None) or not getattr(poly_b, "bounds", None):
            return None
        if not poly_a.intersects(poly_b):
            return 0.0
        inter = poly_a.intersection(poly_b)
        inter_area = inter.area
        if inter_area <= 0:
            return 0.0
        union_area = poly_a.area + poly_b.area - inter_area
        if union_area <= 0:
            return None
        return inter_area / union_area
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Echec du calcul IoU: {e}")
        return None


def _remove_shapefile_set(shp_path: "Path") -> None:
    try:
        base = shp_path.with_suffix("")
        exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".fix", ".qmd", ".qix"]
        for ext in exts:
            p = base.with_suffix(ext)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def _tile_extent_polygon_from_jpg(jpg_path: "Path") -> Optional["Polygon"]:
    try:
        if not jpg_path.exists():
            return None
        jgw_path = jpg_path.with_suffix(".jgw")
        if not jgw_path.exists():
            return None
        pw, ph, xo, yo = read_world_file(str(jgw_path))
        if not all(val is not None for val in [pw, ph, xo, yo]):
            return None
        from PIL import Image

        with Image.open(jpg_path) as img:
            img_width, img_height = img.size

        x_min = xo
        x_max = xo + (pw * img_width)
        y_max = yo
        y_min = yo + (ph * img_height)

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)])
    except Exception:
        return None


def _deduplicate_shapefiles_by_tile_extents(
    labels_dir: str,
    shapefile_paths: List[str],
    iou_threshold: float = 0.1,
    crs: str = "EPSG:2154",
) -> None:
    logger = logging.getLogger(__name__)
    try:
        labels_path = Path(labels_dir)
        parent_dir = labels_path.parent
        jpg_dir = parent_dir / "jpg"

        tile_extents = []
        seen = set()
        for label_file in labels_path.glob("*.txt"):
            base_name = label_file.stem
            if base_name in seen:
                continue
            seen.add(base_name)
            jpg_name = base_name.replace("_detections", "") + ".jpg"
            jpg_path = (jpg_dir / jpg_name) if jpg_dir.exists() else (labels_path / jpg_name)
            poly = _tile_extent_polygon_from_jpg(jpg_path)
            if poly is not None:
                tile_extents.append((base_name, poly))

        if not tile_extents:
            logger.info("Aucune emprise de dalle trouvée (JGW manquant) - déduplication finale ignorée")
            return

        gdfs = []
        for shp in shapefile_paths:
            shp_path = Path(shp)
            if not shp_path.exists():
                continue
            try:
                gdf = gpd.read_file(str(shp_path))
                if len(gdf) == 0:
                    continue
                gdf = gdf.copy()
                gdf["__src_shp"] = str(shp_path)
                gdf["__src_idx"] = list(range(len(gdf)))
                if "confidence" not in gdf.columns:
                    gdf["confidence"] = -1.0
                try:
                    gdf["confidence"] = gdf["confidence"].apply(
                        lambda v: float(v) if v is not None and str(v).strip() != "" else float("nan")
                    )
                    gdf["confidence"] = gdf["confidence"].fillna(-1.0)
                except Exception:
                    gdf["confidence"] = -1.0
                if gdf.crs is None:
                    try:
                        gdf = gdf.set_crs(crs, allow_override=True)
                    except Exception:
                        pass
                gdfs.append(gdf)
            except Exception as e:
                logger.warning(f"Impossible de lire shapefile pour déduplication finale: {shp_path} ({e})")

        if not gdfs:
            return

        all_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry")
        if all_gdf.crs is None:
            try:
                all_gdf = all_gdf.set_crs(crs, allow_override=True)
            except Exception:
                pass

        to_drop_global = set()

        try:
            from shapely.ops import unary_union

            tile_union = unary_union([p for _, p in tile_extents if p is not None])
            if tile_union is not None:
                try:
                    in_any_tile = all_gdf.geometry.notnull() & all_gdf.geometry.intersects(tile_union)
                except Exception:
                    in_any_tile = all_gdf.geometry.notnull()
                outside_idx = all_gdf.index[~in_any_tile]
                if len(outside_idx) > 0:
                    for idx in outside_idx:
                        to_drop_global.add(idx)
                    logger.info(
                        f"Déduplication finale: {len(outside_idx)} suppression(s) hors emprise de toutes les dalles"
                    )
        except Exception as e:
            logger.warning(f"Déduplication finale: suppression hors emprise ignorée (erreur): {e}")

        for tile_name, tile_poly in tile_extents:
            try:
                subset = all_gdf[all_gdf.geometry.notnull() & all_gdf.geometry.intersects(tile_poly)].copy()
            except Exception:
                continue

            if len(subset) < 2:
                continue

            kept_indices = list(subset.index)
            local_dropped = set()

            for i_pos in range(len(kept_indices)):
                i = kept_indices[i_pos]
                if i in local_dropped or i in to_drop_global:
                    continue
                geom_i = all_gdf.at[i, "geometry"]
                if geom_i is None:
                    continue
                conf_i = all_gdf.at[i, "confidence"]

                for j_pos in range(i_pos + 1, len(kept_indices)):
                    j = kept_indices[j_pos]
                    if j in local_dropped or j in to_drop_global:
                        continue
                    geom_j = all_gdf.at[j, "geometry"]
                    if geom_j is None:
                        continue
                    iou = _polygon_iou(geom_i, geom_j)
                    if iou is None or iou <= iou_threshold:
                        continue

                    conf_j = all_gdf.at[j, "confidence"]
                    if conf_i >= conf_j:
                        local_dropped.add(j)
                        to_drop_global.add(j)
                    else:
                        local_dropped.add(i)
                        to_drop_global.add(i)
                        break
            if local_dropped:
                logger.info(
                    f"Déduplication finale: {len(local_dropped)} suppression(s) dans l'emprise {tile_name} (IoU>{iou_threshold})"
                )

        if not to_drop_global:
            return

        remaining = all_gdf.drop(index=list(to_drop_global)).copy()

        for shp in shapefile_paths:
            shp_path = Path(shp)
            if not shp_path.exists():
                continue
            out_gdf = remaining[remaining["__src_shp"] == str(shp_path)].copy()
            if "__src_shp" in out_gdf.columns:
                out_gdf = out_gdf.drop(columns=["__src_shp", "__src_idx"], errors="ignore")
            try:
                _remove_shapefile_set(shp_path)
                if out_gdf.crs is None:
                    try:
                        out_gdf = out_gdf.set_crs(crs, allow_override=True)
                    except Exception:
                        pass
                out_gdf.to_file(str(shp_path))
            except Exception as e:
                logger.warning(f"Échec réécriture shapefile après déduplication finale: {shp_path} ({e})")
    except Exception as e:
        logger.warning(f"Déduplication finale ignorée (erreur): {e}")


def deduplicate_shapefiles_final(
    labels_dir: str,
    shapefile_paths: List[str],
    iou_threshold: float = 0.1,
    crs: str = "EPSG:2154",
    size_filter_enabled: bool = False,
    size_filter_max_meters: float = 50.0,
) -> None:
    _deduplicate_shapefiles_by_tile_extents(
        labels_dir=labels_dir,
        shapefile_paths=shapefile_paths,
        iou_threshold=iou_threshold,
        crs=crs,
    )
    
    if size_filter_enabled:
        _filter_shapefiles_by_size(
            shapefile_paths=shapefile_paths,
            max_size_meters=size_filter_max_meters,
            crs=crs,
        )


def _filter_shapefiles_by_size(
    shapefile_paths: List[str],
    max_size_meters: float = 50.0,
    crs: str = "EPSG:2154",
) -> None:
    """
    Filtre les détections trop grandes (faux positifs) en fonction de leur taille en mètres.
    Les géométries dont la plus grande dimension (largeur ou hauteur de la bbox) dépasse max_size_meters
    sont déplacées dans un shapefile séparé suffixé "_filtered_oversized" pour visualisation.
    
    Args:
        shapefile_paths: Liste des chemins vers les shapefiles à filtrer
        max_size_meters: Taille maximale autorisée en mètres (dimension max de la bbox)
        crs: Système de coordonnées
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Filtrage par taille: déplacement des détections > {max_size_meters} m vers shapefiles séparés")
    
    total_removed = 0
    all_oversized = []
    
    for shp in shapefile_paths:
        shp_path = Path(shp)
        if not shp_path.exists():
            continue
        
        try:
            gdf = gpd.read_file(str(shp_path))
            if len(gdf) == 0:
                continue
            
            initial_count = len(gdf)
            
            def get_max_dimension(geom):
                """Retourne la plus grande dimension (largeur ou hauteur) de la bbox en mètres."""
                if geom is None or not hasattr(geom, 'bounds'):
                    return 0.0
                try:
                    minx, miny, maxx, maxy = geom.bounds
                    width = abs(maxx - minx)
                    height = abs(maxy - miny)
                    return max(width, height)
                except Exception:
                    return 0.0
            
            gdf['__max_dim'] = gdf.geometry.apply(get_max_dimension)
            gdf_filtered = gdf[gdf['__max_dim'] <= max_size_meters].copy()
            gdf_oversized = gdf[gdf['__max_dim'] > max_size_meters].copy()
            
            gdf_filtered = gdf_filtered.drop(columns=['__max_dim'], errors='ignore')
            gdf_oversized = gdf_oversized.drop(columns=['__max_dim'], errors='ignore')
            
            removed_count = len(gdf_oversized)
            total_removed += removed_count
            
            if removed_count > 0:
                logger.info(f"Filtrage taille: {removed_count} détection(s) déplacée(s) depuis {shp_path.name} (>{max_size_meters}m)")
                
                gdf_oversized['__src_class'] = shp_path.stem
                all_oversized.append(gdf_oversized)
                
                _remove_shapefile_set(shp_path)
                if gdf_filtered.crs is None:
                    try:
                        gdf_filtered = gdf_filtered.set_crs(crs, allow_override=True)
                    except Exception:
                        pass
                gdf_filtered.to_file(str(shp_path))
            
        except Exception as e:
            logger.warning(f"Échec filtrage par taille pour {shp_path}: {e}")
    
    if total_removed > 0 and all_oversized:
        try:
            combined_oversized = gpd.GeoDataFrame(pd.concat(all_oversized, ignore_index=True), geometry="geometry")
            if combined_oversized.crs is None:
                try:
                    combined_oversized = combined_oversized.set_crs(crs, allow_override=True)
                except Exception:
                    pass
            
            first_shp = Path(shapefile_paths[0])
            oversized_path = first_shp.parent / "detections_filtered_oversized.shp"
            _remove_shapefile_set(oversized_path)
            combined_oversized.to_file(str(oversized_path))
            
            logger.info(f"Filtrage par taille terminé: {total_removed} détection(s) déplacée(s) vers {oversized_path.name}")
        except Exception as e:
            logger.warning(f"Échec création du shapefile des détections filtrées: {e}")
    else:
        logger.info("Filtrage par taille: aucune détection filtrée")


def _normalize_class_label(label: str) -> str:
    try:
        s = str(label)
    except Exception:
        return ""
    s_clean = s.strip()
    return s_clean

def create_shapefile_from_detections(
    labels_dir: str,
    output_shapefile: str,
    tif_transform_data: dict = None,
    crs: str = "EPSG:2154",
    temp_dir: str = None,
    class_names: dict = None,
    selected_classes: list = None,
) -> bool:
    """
    Crée des shapefiles géoréférencés à partir des fichiers de détection YOLO
    en utilisant les données de géoréférencement des fichiers TIF sources.
    Crée un shapefile séparé pour chaque classe détectée.
    
    Args:
        labels_dir (str): Répertoire contenant les fichiers .txt de détection
        output_shapefile (str): Chemin de base pour les shapefiles de sortie
        tif_transform_data (dict): Dictionnaire contenant les données de transformation des TIF
                                 Format: {filename: (pixel_width, pixel_height, x_origin, y_origin)}
        crs (str): Système de coordonnées (défaut: "EPSG:2154")
        temp_dir (str): Répertoire Temp contenant les TIF sources pour géoréférencement
        class_names (dict): Dictionnaire des noms de classes {class_id: "nom_classe"}
        selected_classes (list): Liste des noms de classes à inclure (None = toutes)
    
    Returns:
        bool: True si succès, False sinon
    """
    logger = logging.getLogger(__name__)
    
    try:
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            logger.error(f"Répertoire de labels non trouvé: {labels_dir}")
            return False
        
        # Dictionnaire pour regrouper les détections par classe
        data_by_class_and_tile = {}
        processed_files = 0

        jgw_logged_for_jpg = set()
        
        iou_threshold = load_iou_threshold_from_config()
        logger.info(f"Seuil IoU chargé depuis config.json: {iou_threshold}")

        # Nom du modèle utilisé pour les détections (stocké comme attribut non éditable)
        model_name = load_model_name_from_config()
        if model_name:
            logger.info(f"Nom du modèle chargé depuis config.json: {model_name}")
        else:
            logger.info("Aucun nom de modèle trouvé dans config.json (computer_vision.selected_model)")

        # Liste des classes disponibles (pour ValueMap QGIS)
        # Si class_names n'est pas fourni via le dossier du modèle, on bascule sur des libellés numériques.
        if isinstance(class_names, dict):
            all_classes = [_normalize_class_label(class_names[k]) for k in sorted(class_names.keys())]
        elif isinstance(class_names, list):
            all_classes = [_normalize_class_label(x) for x in list(class_names)]
        else:
            all_classes = []
        
        # Parcourir tous les fichiers .txt dans le répertoire
        for label_file in labels_path.glob("*.txt"):
            base_name = label_file.stem
            
            # Récupérer les données de transformation du TIF correspondant
            pixel_width = pixel_height = x_origin = y_origin = None
            transform_source = "unknown"

            # Essayer d'abord d'utiliser tif_transform_data (référentiel exact du JPG utilisé pour YOLO)
            tif_key = None
            if tif_transform_data:
                # Essayer avec le nom exact
                if base_name in tif_transform_data:
                    tif_key = base_name
                else:
                    # Si le nom se termine par "_detections", essayer sans ce suffixe
                    if base_name.endswith("_detections"):
                        potential_key = base_name[:-11]  # Enlever "_detections"
                        if potential_key in tif_transform_data:
                            tif_key = potential_key
                    # Sinon, essayer de trouver une clé qui correspond au début du nom
                    if not tif_key:
                        for key in tif_transform_data.keys():
                            if base_name.startswith(key):
                                tif_key = key
                                break

            if tif_key:
                # Utiliser les données de transformation du TIF source (même raster que le JPG YOLO)
                pixel_width, pixel_height, x_origin, y_origin = tif_transform_data[tif_key]
                transform_source = "tif_source"
                logger.debug(
                    f"Utilisation des données TIF pour {base_name} (clé: {tif_key}): px_w={pixel_width}, px_h={pixel_height}, x_orig={x_origin}, y_orig={y_origin}"
                )

            # Si aucune donnée n'a été trouvée dans tif_transform_data, chercher un TIF correspondant dans Temp
            if not all(v is not None for v in [pixel_width, pixel_height, x_origin, y_origin]) and temp_dir:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    # Chercher un TIF avec un nom similaire
                    potential_tif_files = []

                    # IMPORTANT: Priorité absolue au TIF NON ROGNÉ (avec marges PTS_C_LAMB93_IGN69)
                    # Les bounding boxes YOLO sont relatives aux images JPG créées depuis les TIF avec marges
                    # Il faut donc utiliser les coordonnées globales des TIF avec marges pour la transformation
                    base_without_detections = base_name.replace('_detections', '')
                    search_patterns = [
                        f"*_PTS_C_LAMB93_IGN69_LD.tif",  # Priorité 1: TIF avec marges (coordonnées globales correctes)
                        f"*_PTS_C_LAMB93_IGN69_MNT.tif",  # Priorité 2: MNT avec marges
                        f"{base_without_detections}.tif",  # Priorité 3: nom exact
                        f"*_LD_A_LAMB93.tif",  # Priorité 4: TIF ROGNÉ (fallback)
                        f"{base_name}.tif"  # Priorité 5: avec _detections
                    ]

                    for search_pattern in search_patterns:
                        matching_files = list(temp_path.glob(search_pattern))
                        if matching_files:
                            # Filtrer pour trouver le fichier qui correspond à la zone géographique
                            if any(
                                pattern in search_pattern
                                for pattern in [
                                    "_PTS_C_LAMB93_IGN69_LD.tif",
                                    "_PTS_C_LAMB93_IGN69_MNT.tif",
                                    "_LD_A_LAMB93.tif",
                                    "_MNT_A_0M50_LAMB93_IGN69.tif",
                                ]
                            ):
                                # Chercher le fichier qui correspond à la zone géographique (XXXX_YYYY)
                                for tif_file in matching_files:
                                    try:
                                        # Extraire les coordonnées du nom de fichier de détection
                                        det_coords = base_without_detections.split('_')[2:4]  # [XXXX, YYYY]
                                        # Extraire les coordonnées du nom de fichier TIF
                                        tif_coords = tif_file.stem.split('_')[2:4]  # [XXXX, YYYY]
                                        if det_coords == tif_coords:
                                            potential_tif_files = [tif_file]
                                            if "_PTS_C_LAMB93_IGN69" in tif_file.name:
                                                logger.info(
                                                    f"TIF avec marges trouvé pour {base_name}: {tif_file.name} (coordonnées: {det_coords})"
                                                )
                                            else:
                                                logger.info(
                                                    f"TIF rogné trouvé pour {base_name}: {tif_file.name} (coordonnées: {det_coords})"
                                                )
                                            break
                                    except (IndexError, ValueError):
                                        # Si l'extraction des coordonnées échoue, prendre le premier fichier
                                        potential_tif_files = [tif_file]
                                        break
                            else:
                                potential_tif_files.extend(matching_files)

                            if potential_tif_files:
                                break  # Prendre le premier match trouvé selon la priorité

                    if potential_tif_files:
                        # Prendre le premier fichier trouvé
                        tif_file = potential_tif_files[0]
                        logger.debug(f"TIF trouvé dans Temp pour {base_name}: {tif_file.name}")

                        # Extraire les données de géoréférencement directement du TIF
                        try:
                            import rasterio

                            with rasterio.open(str(tif_file)) as src:
                                transform = src.transform
                                pixel_width = abs(transform.a)
                                pixel_height = transform.e  # Garder le signe négatif pour Y
                                x_origin = transform.c
                                y_origin = transform.f
                                transform_source = "temp_tif_rasterio"
                                logger.info(
                                    f"✅ Utilisation du TIF Temp {tif_file.name}: dimensions={src.width}x{src.height}, px_w={pixel_width}, px_h={pixel_height}, x_orig={x_origin}, y_orig={y_origin}"
                                )
                        except Exception as e:
                            logger.warning(f"❌ Erreur rasterio pour {tif_file}: {e}")
                            # Fallback: utiliser gdalinfo CLI si rasterio échoue (cas PyInstaller)
                            try:
                                import subprocess
                                import json
                                import shutil

                                gdalinfo = shutil.which('gdalinfo') or shutil.which(r'C:\\OSGeo4W\\bin\\gdalinfo.exe')
                                if gdalinfo:
                                    result = subprocess.run(
                                        [gdalinfo, '-json', str(tif_file)],
                                        capture_output=True,
                                        text=True,
                                        check=True,
                                    )
                                    info = json.loads(result.stdout)
                                    gt = info.get('geoTransform', [])
                                    if len(gt) >= 6:
                                        x_origin, pixel_width, _, y_origin, _, pixel_height = gt[:6]
                                        pixel_width = abs(pixel_width)  # Assurer valeur positive
                                        transform_source = "temp_tif_gdalinfo"
                                        logger.info(
                                            f"✅ Fallback gdalinfo pour {tif_file.name}: px_w={pixel_width}, px_h={pixel_height}, x_orig={x_origin}, y_orig={y_origin}"
                                        )
                                    else:
                                        logger.warning(f"❌ gdalinfo: geoTransform invalide pour {tif_file}")
                                else:
                                    logger.warning(f"❌ gdalinfo non trouvé pour fallback")
                            except Exception as gdal_e:
                                logger.warning(f"❌ Fallback gdalinfo échoué pour {tif_file}: {gdal_e}")
            
            # Vérifier si on a trouvé des données de géoréférencement
            if not all(v is not None for v in [pixel_width, pixel_height, x_origin, y_origin]):
                # Méthode de dernier recours : extraction approximative depuis le nom de fichier
                logger.warning(
                    f"Données de géoréférencement non disponibles pour {base_name}, fallback approximatif basé sur le nom"
                )
                x_anchor, y_anchor = extract_coordinates_from_filename(base_name)
                if x_anchor is None or y_anchor is None:
                    logger.warning(f"Coordonnées non extraites pour: {base_name}")
                    continue
                # Hypothèses par défaut si rien d'autre n'est disponible
                # On considère une dalle de 1000 m (2000 px à 0.5 m/px)
                pixel_width = 0.5
                pixel_height = -0.5  # Négatif car l'axe Y est inversé
                x_origin = x_anchor
                y_origin = y_anchor + 1000.0
                transform_source = "filename_fallback"
            
            # Chercher d'abord le fichier JSON correspondant pour les données de confiance
            json_file = label_file.with_suffix('.json')
            confidence_data = {}
            
            if json_file.exists():
                try:
                    import json as json_module
                    with open(json_file, 'r') as f:
                        json_data = json_module.load(f)
                        # Indexer les détections par leur position dans la liste
                        detections_list = json_data.get('detections', [])
                        for idx, detection in enumerate(detections_list):
                            confidence_data[idx] = detection.get('confidence')
                    logger.info(
                        f"Données de confiance chargées depuis {json_file.name}: {len(confidence_data)} détections avec confiance"
                    )
                    logger.debug(f"Données de confiance: {confidence_data}")
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture du fichier JSON {json_file}: {e}")
            else:
                logger.info(f"Fichier JSON non trouvé: {json_file}")
            
            # Lire les détections dans le fichier
            try:
                with open(label_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        # Format YOLO standard: class_id x_center y_center width height
                        # Format YOLO-seg: class_id x1 y1 x2 y2 ... xn yn
                        is_segmentation = False
                        seg_points_rel = None
                        confidence = None

                        if len(parts) == 5:
                            class_id, x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts)
                            detection_idx = line_num - 1  # Index 0-based
                            confidence = confidence_data.get(detection_idx)
                            if confidence is not None:
                                logger.debug(f"Confiance trouvée pour détection {detection_idx}: {confidence}")
                            else:
                                logger.debug(f"Aucune confiance trouvée pour détection {detection_idx}")
                        elif len(parts) == 6:
                            # Support legacy: format YOLO avec confiance (non standard)
                            class_id, x_center_rel, y_center_rel, width_rel, height_rel, confidence = map(float, parts)
                        elif (len(parts) - 1) >= 6 and ((len(parts) - 1) % 2 == 0):
                            # YOLO-seg: class_id + au moins 3 points (6 valeurs)
                            class_id = float(parts[0])
                            coords = list(map(float, parts[1:]))
                            seg_points_rel = list(zip(coords[0::2], coords[1::2]))
                            is_segmentation = True
                            detection_idx = line_num - 1  # Index 0-based
                            confidence = confidence_data.get(detection_idx)
                        else:
                            continue

                        # Récupérer les vraies dimensions de l'image JPG correspondante
                        jpg_name = base_name.replace("_detections", "") + ".jpg"
                        jpg_path = None
                        if labels_dir:
                            parent_dir = Path(labels_dir).parent
                            jpg_dir = parent_dir / "jpg"
                            if jpg_dir.exists():
                                jpg_path = jpg_dir / jpg_name
                        if jpg_path is None or not jpg_path.exists():
                            jpg_path = label_file.parent / jpg_name

                        img_width = img_height = 2000
                        if jpg_path.exists():
                            try:
                                from PIL import Image
                                with Image.open(jpg_path) as img:
                                    img_width, img_height = img.size
                                    logger.debug(f"Dimensions réelles de {jpg_name}: {img_width}x{img_height}")
                            except Exception as e:
                                logger.warning(f"Impossible de lire les dimensions de {jpg_name}: {e}")
                        else:
                            logger.warning(
                                f"Image JPG non trouvée: {jpg_path}, utilisation des dimensions par défaut"
                            )
                            logger.debug(
                                f"Recherche JPG - base_name: {base_name}, jpg_name: {jpg_name}, jpg_path: {jpg_path}"
                            )

                        # IMPORTANT: Utiliser en priorité le fichier world (.jgw) du JPG
                        # Cela garantit que python run_pipeline et l'exécutable utilisent
                        # exactement la même source de géoréférencement pour les bboxes.
                        if jpg_path and jpg_path.exists():
                            jgw_path = jpg_path.with_suffix('.jgw')
                            if jgw_path.exists():
                                pw, ph, xo, yo = read_world_file(str(jgw_path))
                                if all(val is not None for val in [pw, ph, xo, yo]):
                                    pixel_width, pixel_height, x_origin, y_origin = pw, ph, xo, yo
                                    transform_source = "jpg_world"
                                    try:
                                        jpg_key = str(jpg_path)
                                    except Exception:
                                        jpg_key = None
                                    if jpg_key is None or jpg_key not in jgw_logged_for_jpg:
                                        logger.info(
                                            f"Utilisation du .jgw pour {jpg_path.name}: px_w={pixel_width}, px_h={pixel_height}, x_orig={x_origin}, y_orig={y_origin}"
                                        )
                                        if jpg_key is not None:
                                            jgw_logged_for_jpg.add(jpg_key)

                        if is_segmentation and seg_points_rel is not None:
                            # Conversion des sommets relatifs → pixels → géo
                            poly_coords_geo = []
                            for x_rel, y_rel in seg_points_rel:
                                x_px = x_rel * img_width
                                y_px = y_rel * img_height
                                x_geo = x_origin + (x_px * pixel_width)
                                y_geo = y_origin + (y_px * pixel_height)
                                poly_coords_geo.append((x_geo, y_geo))

                            if len(poly_coords_geo) < 3:
                                continue

                            # Fermer le polygone si nécessaire
                            if poly_coords_geo[0] != poly_coords_geo[-1]:
                                poly_coords_geo.append(poly_coords_geo[0])

                            bbox_polygon = Polygon(poly_coords_geo)
                        else:
                            # Coordonnées relatives → pixels (centre de la bbox)
                            x_px = x_center_rel * img_width
                            y_px = y_center_rel * img_height

                            # Dimensions de la bbox en pixels
                            width_px = width_rel * img_width
                            height_px = height_rel * img_height

                            # Conversion pixels → coordonnées géographiques
                            # Utilisation de la transformation affine des données TIF
                            x_center_geo = x_origin + (x_px * pixel_width)
                            y_center_geo = y_origin + (y_px * pixel_height)  # pixel_height est négatif
                            
                            # Créer la géométrie de la bounding box
                            half_width_geo = (width_px * abs(pixel_width)) / 2
                            half_height_geo = (height_px * abs(pixel_height)) / 2
                            
                            # Coordonnées des coins de la bbox
                            x_min = x_center_geo - half_width_geo
                            x_max = x_center_geo + half_width_geo
                            y_min = y_center_geo - half_height_geo
                            y_max = y_center_geo + half_height_geo
                            
                            # Créer le polygone de la bounding box
                            bbox_polygon = Polygon([
                                (x_min, y_min),  # Coin inférieur gauche
                                (x_max, y_min),  # Coin inférieur droit
                                (x_max, y_max),  # Coin supérieur droit
                                (x_min, y_max),  # Coin supérieur gauche
                                (x_min, y_min)   # Fermer le polygone
                            ])
                        
                        # Initialiser la structure pour cette classe si nécessaire
                        class_id_int = int(class_id)
                        if class_id_int not in data_by_class_and_tile:
                            data_by_class_and_tile[class_id_int] = {}

                        # Extraire les coordonnées de tuile depuis le nom de fichier
                        x_coord, y_coord = extract_tile_coordinates(base_name)
                        if x_coord is None or y_coord is None:
                            logger.warning(
                                f"Impossible d'extraire les coordonnées de tuile de {base_name}, déduplication globale utilisée"
                            )
                            tile_key = "global"
                        else:
                            tile_key = f"{x_coord}_{y_coord}"

                        # Initialiser la liste pour cette tuile si nécessaire
                        if tile_key not in data_by_class_and_tile[class_id_int]:
                            data_by_class_and_tile[class_id_int][tile_key] = []
                        
                        # Récupérer le nom de classe selon le type de class_names.
                        # PRIORITÉ: class_names fourni par le dossier du modèle. Sinon: libellé numérique.
                        if isinstance(class_names, dict):
                            class_name = class_names.get(class_id_int, f"classe_{class_id_int + 1}")
                        elif isinstance(class_names, list) and class_id_int < len(class_names):
                            class_name = class_names[class_id_int]
                        else:
                            class_name = f"classe_{class_id_int + 1}"

                        class_name = _normalize_class_label(class_name)
                        
                        # Préparer les attributs de sortie (ordre: validation, correction, modèle, nom de modèle)
                        detection_attrs = {
                            "validation": "",
                            "corr_pred": None,
                            "model_pred": class_name,
                            "model_name": model_name,
                            "geometry": bbox_polygon,
                        }

                        if confidence is not None:
                            detection_attrs["confidence"] = confidence
                        conf_bin, conf_color = _confidence_bucket(confidence)
                        detection_attrs["conf_bin"] = conf_bin
                        detection_attrs["conf_color"] = conf_color

                        data_by_class_and_tile[class_id_int][tile_key].append(detection_attrs)
                        
            except Exception as e:
                logger.error(f"Erreur lors de la lecture de {label_file}: {e}")
                continue
            
            processed_files += 1
        
        if not data_by_class_and_tile:
            logger.warning("Aucune détection trouvée pour créer les shapefiles")
            return False
        
        # Créer le répertoire de sortie si nécessaire
        output_path = Path(output_shapefile)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer un shapefile pour chaque classe
        total_detections = 0
        created_shapefiles = []
        
        # Aplatir la structure par tuiles pour créer les shapefiles
        data_by_class = {}
        for class_id, tiles_dict in data_by_class_and_tile.items():
            data_by_class[class_id] = []
            for tile_key, detections in tiles_dict.items():
                data_by_class[class_id].extend(detections)
        
        for class_id, detections in data_by_class.items():
            # Récupérer le nom de classe selon le type de class_names.
            # PRIORITÉ: class_names fourni par le dossier du modèle. Sinon: libellé numérique.
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, f"classe_{class_id + 1}")
            elif isinstance(class_names, list) and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"classe_{class_id + 1}"

            class_name = _normalize_class_label(class_name)
            
            # Filtrer par classes sélectionnées si spécifié
            if selected_classes is not None and len(selected_classes) > 0:
                if class_name not in selected_classes:
                    logger.info(f"Classe '{class_name}' ignorée (non sélectionnée)")
                    continue
            
            # Nom du shapefile pour cette classe
            class_shapefile = output_dir / f"{output_path.stem}_{class_name}.shp"
            
            # Créer le GeoDataFrame pour cette classe
            gdf = gpd.GeoDataFrame(detections, geometry="geometry", crs=crs)
            # S'assurer qu'un CRS est défini (par défaut EPSG:2154)
            if gdf.crs is None:
                try:
                    gdf = gdf.set_crs(crs, allow_override=True)
                    logger.info(f"   ➕ CRS par défaut appliqué: {crs}")
                except Exception as crs_e:
                    logger.warning(f"   Impossible d'appliquer le CRS par défaut ({crs}): {crs_e}")

            # Nettoyage/normalisation pour éviter des blocages QGIS lors du chargement
            # - Supprimer géométries NULL/vides
            # - Réparer géométries invalides (buffer(0))
            # - Normaliser les types de colonnes (Shapefile est sensible aux types mixtes)
            try:
                # 1) Géométries NULL/vides
                gdf = gdf[gdf.geometry.notnull()].copy()
                try:
                    gdf = gdf[~gdf.geometry.is_empty].copy()
                except Exception:
                    pass

                # 2) Filtrer les géométries avec coordonnées non finies
                try:
                    import math

                    def _geom_has_finite_coords(geom) -> bool:
                        if geom is None:
                            return False
                        try:
                            b = geom.bounds
                            return all(math.isfinite(v) for v in b)
                        except Exception:
                            return False

                    gdf = gdf[gdf.geometry.apply(_geom_has_finite_coords)].copy()
                except Exception:
                    pass

                # 3) Réparation simple des géométries invalides
                try:
                    def _fix_geom(geom):
                        try:
                            if geom is None:
                                return None
                            if hasattr(geom, "is_valid") and not geom.is_valid:
                                return geom.buffer(0)
                            return geom
                        except Exception:
                            return geom

                    gdf["geometry"] = gdf["geometry"].apply(_fix_geom)
                except Exception:
                    pass

                # 4) Normalisation des colonnes attributaires (évite types mixtes)
                text_cols = ["validation", "corr_pred", "model_pred", "model_name", "conf_bin", "conf_color"]
                for col in text_cols:
                    if col in gdf.columns:
                        gdf[col] = gdf[col].fillna("").astype(str)

                if "confidence" in gdf.columns:
                    gdf["confidence"] = gdf["confidence"].fillna(-1.0)
                    try:
                        gdf["confidence"] = gdf["confidence"].astype(float)
                    except Exception:
                        gdf["confidence"] = gdf["confidence"].astype(str)

                if len(gdf) == 0:
                    logger.warning(
                        f"Aucune géométrie valide après nettoyage pour la classe '{class_name}' - shapefile non écrit"
                    )
                    continue
            except Exception as clean_e:
                logger.warning(f"Nettoyage/normalisation shapefile ignoré (erreur): {clean_e}")
            
            # Sauvegarder en shapefile avec diagnostic détaillé
            try:
                # DIAGNOSTIC DÉTAILLÉ POUR PYINSTALLER - CONVERSION_SHP
                logger.info(f"🔍 DIAGNOSTIC CONVERSION_SHP: Tentative de sauvegarde shapefile...")
                logger.info(f"   Classe: {class_name}")
                logger.info(f"   GeoDataFrame shape: {gdf.shape}")
                logger.info(f"   Colonnes: {list(gdf.columns)}")
                logger.info(f"   CRS: {gdf.crs}")
                logger.info(f"   Chemin de sortie: {class_shapefile}")
                
                # Test des modules critiques avant sauvegarde
                try:
                    import pandas._libs.window.aggregations
                    logger.info("   ✅ pandas._libs.window.aggregations disponible")
                except Exception as mod_e:
                    logger.error(f"   ❌ pandas._libs.window.aggregations: {mod_e}")
                
                # Test d'autres modules pandas critiques
                critical_modules = [
                    'pandas._libs.lib',
                    'pandas._libs.algos',
                    'pandas._libs.groupby',
                    'pandas._libs.ops',
                    'pandas.core.groupby.ops'
                ]
                
                for mod in critical_modules:
                    try:
                        __import__(mod)
                        logger.info(f"   ✅ {mod} disponible")
                    except Exception as mod_e:
                        logger.error(f"   ❌ {mod}: {mod_e}")
                
                _remove_shapefile_set(class_shapefile)
                gdf.to_file(str(class_shapefile))
                logger.info(f"✅ Shapefile créé avec succès: {class_shapefile}")
                
            except Exception as save_e:
                logger.error(f"❌ ERREUR SAUVEGARDE SHAPEFILE CONVERSION_SHP: {save_e}")
                logger.error(f"   Type d'erreur: {type(save_e).__name__}")
                
                # Traceback détaillé pour PyInstaller
                import traceback
                tb_lines = traceback.format_exc().split('\n')
                for i, line in enumerate(tb_lines):
                    if line.strip():
                        logger.error(f"   TB[{i}]: {line}")
                
                # Essayer des alternatives de sauvegarde
                try:
                    logger.info("🔄 Tentative de sauvegarde alternative...")
                    # Essayer sans CRS
                    gdf_no_crs = gdf.copy()
                    gdf_no_crs.crs = None
                    gdf_no_crs.to_file(str(class_shapefile))
                    logger.info("✅ Sauvegarde alternative réussie (sans CRS)")
                except Exception as alt_e:
                    logger.error(f"❌ Sauvegarde alternative échouée: {alt_e}")
                    # Fallback final: écrire un GeoJSON simple et utiliser ogr2ogr (CLI) pour produire le shapefile
                    try:
                        logger.info("🛠️ Fallback CLI: export GeoJSON + ogr2ogr → Shapefile")
                        # Construire un GeoJSON FeatureCollection à partir de 'detections'
                        features = []
                        for det in detections:
                            geom = det.get("geometry")
                            try:
                                geom_mapping = geom.__geo_interface__
                            except Exception:
                                logger.warning("   Géométrie invalide, détection ignorée dans le GeoJSON")
                                continue
                            props = {k: v for k, v in det.items() if k != "geometry"}
                            features.append({
                                "type": "Feature",
                                "geometry": geom_mapping,
                                "properties": props,
                            })
                        fc = {"type": "FeatureCollection", "features": features}
                        # Écrire dans un fichier temporaire
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False, encoding='utf-8') as tmp_geojson:
                            json.dump(fc, tmp_geojson)
                            tmp_geojson_path = tmp_geojson.name
                        logger.info(f"   GeoJSON temporaire écrit: {tmp_geojson_path}")
                        # Déterminer ogr2ogr
                        osgeo4w_root = os.environ.get("OSGEO4W_ROOT", r"C:\\OSGeo4W")
                        candidates = [
                            os.path.join(osgeo4w_root, 'bin', 'ogr2ogr.exe'),
                            r"C:\\OSGeo4W64\\bin\\ogr2ogr.exe",
                        ]
                        ogr2ogr = None
                        for c in candidates:
                            if os.path.exists(c):
                                ogr2ogr = c
                                break
                        if not ogr2ogr:
                            logger.error("ogr2ogr introuvable. Veuillez installer OSGeo4W et définir OSGEO4W_ROOT.")
                            raise alt_e
                        # Exécuter ogr2ogr pour créer le shapefile, forcer le CRS si disponible
                        crs_arg = []
                        try:
                            crs_str = None
                            if gdf.crs is not None:
                                crs_str = getattr(gdf.crs, 'to_string', lambda: str(gdf.crs))()
                            if not crs_str:
                                crs_str = crs
                            if crs_str:
                                crs_arg = ["-a_srs", crs_str]
                        except Exception:
                            crs_arg = ["-a_srs", crs]
                        cmd = [ogr2ogr, "-overwrite", *crs_arg, str(class_shapefile), tmp_geojson_path]
                        logger.info("   Exécution: " + " ".join(cmd))
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        if res.returncode != 0:
                            logger.error(f"   ogr2ogr a échoué (code {res.returncode})")
                            if res.stdout:
                                logger.error(f"   STDOUT:\n{res.stdout}")
                            if res.stderr:
                                logger.error(f"   STDERR:\n{res.stderr}")
                            raise alt_e
                        logger.info("✅ Shapefile créé via ogr2ogr (CLI)")
                    except Exception as cli_fallback_e:
                        logger.error(f"❌ Fallback ogr2ogr échoué: {cli_fallback_e}")
                        raise save_e
            
            created_shapefiles.append(str(class_shapefile))
            total_detections += len(detections)
            
            logger.info(f"✅ Shapefile créé pour la classe '{class_name}': {class_shapefile}")
            logger.info(f"📊 {len(detections)} détections de classe '{class_name}'")
        
        logger.info(f"🎯 {len(created_shapefiles)} shapefiles créés au total")
        logger.info(f"📊 {total_detections} détections dans {processed_files} fichiers")

        # Générer un projet QGIS avec Value Map pour les champs (validation, model_pred, correction_prediction)
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring, parse as et_parse
            import xml.dom.minidom as minidom
            import os

            # Déterminer la racine de l'indice RVT (parent du dossier shapefiles)
            index_root = None
            if created_shapefiles:
                first_shp = Path(created_shapefiles[0])
                shp_parent = first_shp.parent
                if shp_parent.name.lower() in {"shapefiles", "shp"}:
                    index_root = shp_parent.parent
                else:
                    index_root = shp_parent
            else:
                # Fallback: utiliser le parent du chemin de sortie de base
                index_root = Path(output_shapefile).parent

            # Dossier contenant les TIF de l'indice (ex: .../RVT/LD/tif)
            tif_dir = index_root / "tif"
            tif_files = []
            if tif_dir.exists() and tif_dir.is_dir():
                tif_files = sorted(tif_dir.glob("*.tif"))

            project = Element('qgis', attrib={'version': '3.34.0', 'projectname': 'detections_validation'})

            try:
                props = SubElement(project, 'properties')
                paths = SubElement(props, 'Paths')
                SubElement(paths, 'Absolute').text = '0'
            except Exception:
                pass

            # Définir le CRS du projet (utilise le paramètre crs, ex. "EPSG:2154")
            try:
                project_crs = SubElement(project, 'projectCrs')
                SubElement(project_crs, 'authid').text = crs
            except Exception:
                pass

            layer_tree = SubElement(project, 'layer-tree-group', attrib={'name': ''})
            maplayers = SubElement(project, 'projectlayers')

            # Charger le style QML des détections si disponible
            style_renderer = None
            style_selection = None
            style_customprops = None
            try:
                style_path = Path(__file__).parents[2] / 'resources' / 'styles' / 'style_detections.qml'
                if style_path.exists():
                    style_tree = et_parse(str(style_path))
                    style_root = style_tree.getroot()
                    style_renderer = style_root.find('renderer-v2')
                    style_selection = style_root.find('selection')
                    style_customprops = style_root.find('customproperties')
                    logger.info(f"Style QML des détections chargé depuis {style_path}")
                else:
                    logger.info("Style QML des détections introuvable (models/style_detections.qml)")
            except Exception as style_e:
                logger.warning(f"Échec du chargement du style QML des détections: {style_e}")

            def _apply_confidence_symbology(maplayer_el) -> None:
                """Applique un renderer catégorisé QGIS sur le champ conf_bin.

                Le style_detections.qml actuel est de type singleSymbol, donc on force ici
                un rendu low/mid/high.
                """
                try:
                    palettes = [
                        ['255,255,0', '255,204,0', '255,153,0', '255,102,0', '255,0,0'],
                        ['204,229,255', '153,204,255', '102,178,255', '51,153,255', '0,102,204'],
                        ['235,224,255', '204,179,255', '178,128,255', '153,77,255', '102,0,204'],
                        ['204,255,204', '153,255,153', '102,204,102', '51,153,51', '0,102,0'],
                        ['240,240,240', '200,200,200', '160,160,160', '120,120,120', '80,80,80'],
                        ['255,235,204', '255,204,153', '230,170,115', '204,136,85', '153,85,34'],
                    ]

                    palette_rgb = palettes[0]
                    try:
                        palette_index = int(maplayer_el.get('dataSourcePaletteIndex'))
                        palette_rgb = palettes[palette_index % len(palettes)]
                    except Exception:
                        palette_rgb = palettes[0]

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
                    SubElement(categories, 'category', attrib={'value': '[0:0.2[', 'label': '[0:0.2[', 'symbol': '0', 'render': 'true'})
                    SubElement(categories, 'category', attrib={'value': '[0.2:0.4[', 'label': '[0.2:0.4[', 'symbol': '1', 'render': 'true'})
                    SubElement(categories, 'category', attrib={'value': '[0.4:0.6[', 'label': '[0.4:0.6[', 'symbol': '2', 'render': 'true'})
                    SubElement(categories, 'category', attrib={'value': '[0.6:0.8[', 'label': '[0.6:0.8[', 'symbol': '3', 'render': 'true'})
                    SubElement(categories, 'category', attrib={'value': '[0.8:1]', 'label': '[0.8:1]', 'symbol': '4', 'render': 'true'})

                    symbols = SubElement(renderer, 'symbols')

                    def _add_symbol(symbol_name: str, rgb: str) -> None:
                        sym = SubElement(symbols, 'symbol', attrib={'type': 'fill', 'name': symbol_name, 'alpha': '1', 'clip_to_extent': '1', 'force_rhr': '0'})
                        layer = SubElement(sym, 'layer', attrib={'class': 'SimpleLine', 'enabled': '1', 'locked': '0', 'pass': '0'})
                        opt = SubElement(layer, 'Option', attrib={'type': 'Map'})
                        SubElement(opt, 'Option', attrib={'name': 'line_color', 'value': f"{rgb},255", 'type': 'QString'})
                        SubElement(opt, 'Option', attrib={'name': 'line_style', 'value': 'solid', 'type': 'QString'})
                        SubElement(opt, 'Option', attrib={'name': 'line_width', 'value': '0.6', 'type': 'QString'})
                        SubElement(opt, 'Option', attrib={'name': 'line_width_unit', 'value': 'MM', 'type': 'QString'})

                    _add_symbol('0', palette_rgb[0])
                    _add_symbol('1', palette_rgb[1])
                    _add_symbol('2', palette_rgb[2])
                    _add_symbol('3', palette_rgb[3])
                    _add_symbol('4', palette_rgb[4])

                except Exception as sym_e:
                    logger.warning(f"Impossible d'appliquer la symbologie confiance: {sym_e}")

            # 1) Ajouter d'abord les shapefiles de détection comme couches vecteur
            #    afin qu'ils soient affichés au-dessus des rasters dans QGIS
            for shp_idx, shp in enumerate(created_shapefiles):
                shp_path = Path(shp)
                try:
                    shp_ds = os.path.relpath(str(shp_path.resolve()), start=str(index_root.resolve()))
                    shp_ds = shp_ds.replace('\\', '/')
                except Exception:
                    shp_ds = str(shp_path.resolve())
                layer_id = shp_path.stem
                SubElement(layer_tree, 'layer-tree-layer', attrib={'id': layer_id, 'name': layer_id})

                ml = SubElement(maplayers, 'maplayer', attrib={'type': 'vector'})
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
                fcfg = SubElement(ml, 'fieldConfiguration')
                f_val = SubElement(fcfg, 'field', attrib={'name': 'validation'})
                ew_val = SubElement(f_val, 'editWidget', attrib={'type': 'ValueMap'})
                cfg_val = SubElement(ew_val, 'config')
                opt_val_root = SubElement(cfg_val, 'Option', attrib={'type': 'Map'})
                opt_val_map = SubElement(opt_val_root, 'Option', attrib={'name': 'map', 'type': 'List'})
                for label in ["oui", "non", "peut-être"]:
                    item = SubElement(opt_val_map, 'Option', attrib={'type': 'Map'})
                    SubElement(item, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
                SubElement(opt_val_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'true'})
                f_pred = SubElement(fcfg, 'field', attrib={'name': 'model_pred'})
                ew_pred = SubElement(f_pred, 'editWidget', attrib={'type': 'ValueMap'})
                cfg_pred = SubElement(ew_pred, 'config')
                opt_pred_root = SubElement(cfg_pred, 'Option', attrib={'type': 'Map'})
                opt_pred_map = SubElement(opt_pred_root, 'Option', attrib={'name': 'map', 'type': 'Map'})
                for label in all_classes:
                    SubElement(opt_pred_map, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
                SubElement(opt_pred_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'false'})
                SubElement(fcfg, 'field', attrib={'name': 'model_name'})
                f_corr = SubElement(fcfg, 'field', attrib={'name': 'corr_pred'})
                ew_corr = SubElement(f_corr, 'editWidget', attrib={'type': 'ValueMap'})
                cfg_corr = SubElement(ew_corr, 'config')
                opt_corr_root = SubElement(cfg_corr, 'Option', attrib={'type': 'Map'})
                opt_corr_map = SubElement(opt_corr_root, 'Option', attrib={'name': 'map', 'type': 'Map'})
                for label in all_classes:
                    SubElement(opt_corr_map, 'Option', attrib={'name': label, 'type': 'QString', 'value': label})
                SubElement(opt_corr_root, 'Option', attrib={'name': 'AllowNull', 'type': 'bool', 'value': 'true'})

                aliases = SubElement(ml, 'aliases')
                SubElement(aliases, 'alias', attrib={'field': 'corr_pred', 'index': '-1', 'name': 'correction_prediction'})
                SubElement(aliases, 'alias', attrib={'field': 'confidence', 'index': '-1', 'name': 'confiance'})
                SubElement(aliases, 'alias', attrib={'field': 'conf_bin', 'index': '-1', 'name': 'tranche_confiance'})
                SubElement(aliases, 'alias', attrib={'field': 'conf_color', 'index': '-1', 'name': 'couleur_confiance'})
                SubElement(aliases, 'alias', attrib={'field': 'model_name', 'index': '-1', 'name': 'modele_detection'})

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

                _apply_confidence_symbology(ml)

                if style_selection is not None:
                    ml.append(copy.deepcopy(style_selection))
                if style_customprops is not None:
                    ml.append(copy.deepcopy(style_customprops))

            # 2) Ajouter le VRT comme couche raster unique (en dessous des shapefiles)
            #    Utilise index.vrt s'il existe, sinon fallback sur les TIF individuels
            vrt_path = tif_dir / "index.vrt" if tif_dir.exists() else None
            if vrt_path and vrt_path.exists():
                # Utiliser le VRT (une seule couche pour toutes les dalles)
                try:
                    vrt_ds = os.path.relpath(str(vrt_path.resolve()), start=str(index_root.resolve()))
                    vrt_ds = vrt_ds.replace('\\', '/')
                except Exception:
                    vrt_ds = str(vrt_path.resolve())
                raster_id = "index_rvt"
                SubElement(layer_tree, 'layer-tree-layer', attrib={'id': raster_id, 'name': 'Dalles RVT (index)'})

                ml_raster = SubElement(maplayers, 'maplayer', attrib={'type': 'raster'})
                SubElement(ml_raster, 'id').text = raster_id
                SubElement(ml_raster, 'layername').text = 'Dalles RVT (index)'
                SubElement(ml_raster, 'datasource').text = vrt_ds
                SubElement(ml_raster, 'provider').text = 'gdal'
                try:
                    srs_r = SubElement(ml_raster, 'srs')
                    sref_r = SubElement(srs_r, 'spatialrefsys')
                    SubElement(sref_r, 'authid').text = crs
                except Exception:
                    pass
            else:
                # Fallback: ajouter les TIF individuellement si pas de VRT
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

            xml_bytes = tostring(project, encoding='utf-8')
            pretty = minidom.parseString(xml_bytes).toprettyxml(indent='  ', encoding='utf-8')
            qgs_path = index_root / 'detections_validation.qgs'
            with open(qgs_path, 'wb') as f:
                f.write(pretty)
            logger.info("════════════════════════════════════════════════════════════")
            logger.info("📋 PROJET QGIS GÉNÉRÉ")
            logger.info("════════════════════════════════════════════════════════════")
            logger.info(f"   Fichier: {qgs_path}")
            logger.info("   Ce projet contient les couches de détection et les rasters RVT.")
            logger.info("   Ouvrez-le dans QGIS pour valider les détections.")
        except Exception as e:
            logger.warning(f"Échec génération du projet QGIS: {e}")

        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du shapefile: {e}")
        return False


if __name__ == "__main__":
    import sys

    print("conversion_shp.py est un module. Utilisez le pipeline pour générer les shapefiles.")
    sys.exit(0)
