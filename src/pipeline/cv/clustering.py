"""
Module de clustering spatial des détections CV.

Identifie des clusters (zones) à partir des détections individuelles
en utilisant un algorithme DBSCAN implémenté avec scipy.spatial.cKDTree.
Chaque cluster est exporté sous forme de polygone (enveloppe convexe ou bounding box).

Dépendances : numpy, scipy (disponibles dans QGIS/OSGeo4W).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  DBSCAN via scipy.spatial.cKDTree (pas de dépendance sklearn)       #
# ------------------------------------------------------------------ #

def _dbscan_scipy(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN implémenté avec scipy.spatial.cKDTree.
    
    Args:
        points: Array (N, 2) de coordonnées
        eps: Distance maximale entre deux points pour être voisins
        min_samples: Nombre minimum de points pour former un cluster
        
    Returns:
        Array (N,) d'étiquettes de cluster (-1 = bruit)
    """
    from scipy.spatial import cKDTree

    n = len(points)
    labels = np.full(n, -1, dtype=int)
    tree = cKDTree(points)
    
    # Pré-calculer tous les voisinages
    neighborhoods = tree.query_ball_tree(tree, r=eps)
    
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        
        neighbors = neighborhoods[i]
        if len(neighbors) < min_samples:
            continue
        
        # Nouveau cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = neighborhoods[q]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        
        cluster_id += 1
    
    return labels


# ------------------------------------------------------------------ #
#  Génération des géométries de cluster                                #
# ------------------------------------------------------------------ #

def _build_cluster_geometry(
    points: np.ndarray,
    geometries: List[Polygon],
    output_geometry: str,
    buffer_m: float,
) -> Optional[Polygon]:
    """
    Construit la géométrie d'un cluster à partir de ses points/polygones membres.
    
    Args:
        points: Centroïdes des détections du cluster (N, 2)
        geometries: Polygones des détections du cluster
        output_geometry: "convex_hull" ou "bounding_box"
        buffer_m: Marge en mètres autour de la géométrie
        
    Returns:
        Polygone du cluster ou None si impossible
    """
    if len(points) < 3:
        # Moins de 3 points → on utilise l'union des géométries + buffer
        merged = unary_union(geometries)
        if merged.is_empty:
            return None
        cluster_geom = merged.convex_hull
    else:
        if output_geometry == "bounding_box":
            mp = MultiPoint(points.tolist())
            cluster_geom = mp.minimum_rotated_rectangle
        else:
            # convex_hull par défaut
            mp = MultiPoint(points.tolist())
            cluster_geom = mp.convex_hull
    
    if buffer_m > 0:
        cluster_geom = cluster_geom.buffer(buffer_m, join_style=2)
    
    if cluster_geom.is_empty or not cluster_geom.is_valid:
        return None
    
    return cluster_geom


# ------------------------------------------------------------------ #
#  Fonction principale de clustering                                   #
# ------------------------------------------------------------------ #

def run_clustering(
    data_by_class_name: Dict[str, List[Dict]],
    clustering_configs: List[Dict],
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Exécute le clustering spatial sur les détections post-processées.
    
    Args:
        data_by_class_name: Détections regroupées par nom de classe.
            Chaque détection est un dict avec au minimum:
                - "geometry": Polygon shapely
                - "confidence": float (optionnel)
                - "model_pred": str
                - "model_name": str
        clustering_configs: Liste de configurations de clustering
            (voir load_clustering_config_from_model dans model_config.py)
    
    Returns:
        Tuple (cluster_detections_by_class, data_by_class_name_updated):
            - cluster_detections_by_class: Nouvelles détections de type cluster
              regroupées par output_class_name
            - data_by_class_name_updated: Détections originales avec cluster_id ajouté
    """
    t0 = time.perf_counter()
    
    cluster_detections_by_class: Dict[str, List[Dict]] = {}
    # Copie pour ajouter cluster_id sans modifier l'original
    updated_data = {k: list(v) for k, v in data_by_class_name.items()}
    
    for cfg_idx, cfg in enumerate(clustering_configs):
        target_classes = cfg["target_classes"]
        min_confidence = cfg["min_confidence"]
        min_cluster_size = cfg["min_cluster_size"]
        min_samples = cfg["min_samples"]
        eps_m = cfg["eps_m"]
        output_class_name = cfg["output_class_name"]
        output_geometry = cfg["output_geometry"]
        buffer_m = cfg["buffer_m"]
        min_area_m2 = cfg["min_area_m2"]
        
        logger.info(
            f"Clustering [{cfg_idx+1}/{len(clustering_configs)}]: "
            f"classes={target_classes}, eps={eps_m}m, "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
            f"min_confidence={min_confidence}"
        )
        
        # Collecter les détections cibles
        candidates = []  # (index_class, index_det, centroid, geometry, confidence)
        for class_name in target_classes:
            if class_name not in updated_data:
                logger.warning(f"Clustering: classe '{class_name}' non trouvée dans les détections")
                continue
            for det_idx, det in enumerate(updated_data[class_name]):
                geom = det.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                conf = det.get("confidence", 0.0)
                if conf is not None and conf < min_confidence:
                    continue
                centroid = geom.centroid
                candidates.append((class_name, det_idx, np.array([centroid.x, centroid.y]), geom, conf))
        
        if len(candidates) < min_cluster_size:
            logger.info(
                f"Clustering: seulement {len(candidates)} détections candidates "
                f"(minimum requis: {min_cluster_size}), clustering ignoré"
            )
            continue
        
        # Extraire les centroïdes en array numpy
        centroids = np.array([c[2] for c in candidates])
        
        # Exécuter DBSCAN
        try:
            labels = _dbscan_scipy(centroids, eps=eps_m, min_samples=min_samples)
        except Exception as e:
            logger.error(f"Clustering DBSCAN échoué: {e}")
            continue
        
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)
        n_noise = int(np.sum(labels == -1))
        
        logger.info(
            f"Clustering: {n_clusters} cluster(s) trouvé(s), "
            f"{n_noise} détections isolées (bruit) sur {len(candidates)} candidates"
        )
        
        # Assigner cluster_id aux détections originales
        for cand_idx, (class_name, det_idx, _, _, _) in enumerate(candidates):
            label = int(labels[cand_idx])
            if label >= 0:
                updated_data[class_name][det_idx]["cluster_id"] = f"{output_class_name}_{label}"
        
        # Construire les géométries de cluster
        cluster_dets = []
        for label_id in sorted(unique_labels):
            mask = labels == label_id
            cluster_points = centroids[mask]
            cluster_geoms = [candidates[i][3] for i in range(len(candidates)) if mask[i]]
            cluster_confs = [candidates[i][4] for i in range(len(candidates)) if mask[i]]
            
            if len(cluster_points) < min_cluster_size:
                continue
            
            cluster_geom = _build_cluster_geometry(
                cluster_points, cluster_geoms, output_geometry, buffer_m
            )
            if cluster_geom is None:
                continue
            
            area_m2 = cluster_geom.area
            if min_area_m2 > 0 and area_m2 < min_area_m2:
                logger.debug(
                    f"Cluster {label_id} filtré: aire={area_m2:.0f}m² < min_area={min_area_m2:.0f}m²"
                )
                continue
            
            nb_detections = int(mask.sum())
            valid_confs = [c for c in cluster_confs if c is not None and c > 0]
            mean_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0
            density = nb_detections / area_m2 if area_m2 > 0 else 0.0
            
            # Récupérer model_name depuis une détection du cluster
            sample_det = candidates[np.where(mask)[0][0]]
            sample_class = sample_det[0]
            sample_idx = sample_det[1]
            model_name = updated_data[sample_class][sample_idx].get("model_name", "")
            
            cluster_det = {
                "validation": "",
                "corr_pred": None,
                "model_pred": output_class_name,
                "model_name": model_name,
                "geometry": cluster_geom,
                "confidence": mean_confidence,
                "nb_detect": nb_detections,
                "area_m2": round(area_m2, 1),
                "density": round(density, 6),
                "cluster_id": f"{output_class_name}_{label_id}",
            }
            cluster_dets.append(cluster_det)
        
        if cluster_dets:
            cluster_detections_by_class[output_class_name] = cluster_dets
            logger.info(
                f"Clustering: {len(cluster_dets)} cluster(s) '{output_class_name}' générés "
                f"(après filtrage min_area={min_area_m2:.0f}m²)"
            )
    
    elapsed = time.perf_counter() - t0
    total_clusters = sum(len(v) for v in cluster_detections_by_class.values())
    logger.info(f"Clustering terminé: {total_clusters} cluster(s) total en {elapsed:.2f}s")
    
    return cluster_detections_by_class, updated_data
