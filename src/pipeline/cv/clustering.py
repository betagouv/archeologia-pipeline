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
import shapely
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# concave_hull n'existe qu'à partir de Shapely 2.0
_CONCAVE_HULL_AVAILABLE = hasattr(shapely, "concave_hull")
_CONCAVE_HULL_WARNED = False


# ------------------------------------------------------------------ #
#  DBSCAN via scipy.spatial.cKDTree (pas de dépendance sklearn)       #
# ------------------------------------------------------------------ #

def _dbscan_scipy(
    points: np.ndarray,
    eps: float,
    min_samples: int,
    is_core_eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    DBSCAN implémenté avec scipy.spatial.cKDTree.

    Optimisations :
    - Voisinages calculés à la demande (lazy) au lieu de pré-calculer
      tous les voisinages d'un coup, ce qui économise mémoire et CPU
      pour les points qui ne sont jamais visités par expansion.
    - Seed set sans doublons grâce à un set de suivi, évitant la
      croissance quadratique de la liste dans les zones denses.

    Mode hystérésis (paramètre `is_core_eligible`) :
    - Si fourni, seuls les points marqués True peuvent devenir des "core
      points" (initier ou étendre un cluster). Les autres points peuvent
      uniquement être absorbés comme points "border" s'ils tombent dans
      le voisinage `eps` d'un core point. Ils ne propagent jamais le
      cluster. Inspiré du seuillage à hystérésis (Canny).
    - Si None, comportement DBSCAN classique (tous les points sont
      éligibles).

    Args:
        points: Array (N, 2) de coordonnées
        eps: Distance maximale entre deux points pour être voisins
        min_samples: Nombre minimum de points pour former un cluster
        is_core_eligible: Array booléen (N,) optionnel. True = peut être
            core. Si None, tous éligibles.

    Returns:
        Array (N,) d'étiquettes de cluster (-1 = bruit)
    """
    from collections import deque
    from scipy.spatial import cKDTree

    n = len(points)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    tree = cKDTree(points)

    if is_core_eligible is None:
        is_core_eligible = np.ones(n, dtype=bool)
    elif len(is_core_eligible) != n:
        raise ValueError(
            f"is_core_eligible doit avoir la même taille que points "
            f"(got {len(is_core_eligible)} vs {n})"
        )

    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Un point non-core-éligible ne peut pas initier un cluster.
        # On le laisse à -1 ; il pourra être assigné plus tard via
        # expansion s'il tombe dans le voisinage d'un core.
        if not is_core_eligible[i]:
            continue

        neighbors = tree.query_ball_point(points[i], r=eps)
        if len(neighbors) < min_samples:
            continue

        # Nouveau cluster
        labels[i] = cluster_id
        queue = deque()
        in_queue = set()
        for nb in neighbors:
            if nb != i:
                queue.append(nb)
                in_queue.add(nb)

        while queue:
            q = queue.popleft()
            if not visited[q]:
                visited[q] = True
                # Seuls les points core-éligibles peuvent étendre le cluster.
                if is_core_eligible[q]:
                    q_neighbors = tree.query_ball_point(points[q], r=eps)
                    if len(q_neighbors) >= min_samples:
                        for nb in q_neighbors:
                            if nb not in in_queue and not visited[nb]:
                                queue.append(nb)
                                in_queue.add(nb)
            if labels[q] == -1:
                labels[q] = cluster_id

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
    concave_ratio: float = 0.3,
) -> Optional[Polygon]:
    """
    Construit la géométrie d'un cluster à partir de ses points/polygones membres.

    Args:
        points: Centroïdes des détections du cluster (N, 2)
        geometries: Polygones des détections du cluster
        output_geometry: "convex_hull", "concave_hull" ou "bounding_box"
        buffer_m: Marge en mètres autour de la géométrie
        concave_ratio: Paramètre du concave hull (Shapely 2.0+).
            Plage [0, 1]. 0 = très concave (dense, beaucoup de sommets,
            colle aux points), 1 = équivalent au convex hull.
            Valeurs typiques 0.2-0.5 pour des zones archéologiques.
            Ignoré si output_geometry != "concave_hull".

    Returns:
        Polygone du cluster ou None si impossible
    """
    global _CONCAVE_HULL_WARNED

    if len(points) < 3:
        # Moins de 3 points → on utilise l'union des géométries + buffer
        merged = unary_union(geometries)
        if merged.is_empty:
            return None
        cluster_geom = merged.convex_hull
    else:
        mp = MultiPoint(points.tolist())
        if output_geometry == "bounding_box":
            cluster_geom = mp.minimum_rotated_rectangle
        elif output_geometry == "concave_hull":
            if _CONCAVE_HULL_AVAILABLE:
                try:
                    cluster_geom = shapely.concave_hull(
                        mp,
                        ratio=concave_ratio,
                        allow_holes=False,
                    )
                    # concave_hull peut retourner LineString/Point sur points
                    # quasi-colinéaires : fallback sur convex_hull dans ce cas.
                    if (
                        cluster_geom is None
                        or cluster_geom.is_empty
                        or cluster_geom.geom_type != "Polygon"
                    ):
                        logger.debug(
                            "concave_hull a retourné une géométrie non polygonale "
                            f"({getattr(cluster_geom, 'geom_type', None)}), "
                            "fallback convex_hull"
                        )
                        cluster_geom = mp.convex_hull
                except Exception as e:
                    logger.warning(
                        f"Erreur shapely.concave_hull ({e}), fallback convex_hull"
                    )
                    cluster_geom = mp.convex_hull
            else:
                if not _CONCAVE_HULL_WARNED:
                    logger.warning(
                        "shapely.concave_hull indisponible (Shapely < 2.0), "
                        "fallback sur convex_hull"
                    )
                    _CONCAVE_HULL_WARNED = True
                cluster_geom = mp.convex_hull
        else:
            # convex_hull par défaut
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
        # Seuil d'extension (hystérésis) : détections entre min_confidence_extend
        # et min_confidence sont absorbables comme points "border" dans un cluster
        # existant mais ne peuvent ni initier ni étendre un cluster.
        # Défaut = min_confidence → comportement DBSCAN classique (rétro-compatible).
        min_confidence_extend = cfg.get("min_confidence_extend", min_confidence)
        if min_confidence_extend > min_confidence:
            logger.warning(
                f"Clustering: min_confidence_extend ({min_confidence_extend}) > "
                f"min_confidence ({min_confidence}) — incohérent. "
                f"Utilisation de min_confidence_extend = min_confidence."
            )
            min_confidence_extend = min_confidence
        min_cluster_size = cfg["min_cluster_size"]
        min_samples = cfg["min_samples"]
        eps_m = cfg["eps_m"]
        output_class_name = cfg["output_class_name"]
        output_geometry = cfg["output_geometry"]
        buffer_m = cfg["buffer_m"]
        min_area_m2 = cfg["min_area_m2"]
        concave_ratio = cfg.get("concave_ratio", 0.3)
        hysteresis_active = min_confidence_extend < min_confidence
        logger.info(
            f"Clustering [{cfg_idx+1}/{len(clustering_configs)}]: "
            f"classes={target_classes}, eps={eps_m}m, "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
            f"min_confidence={min_confidence}"
            + (f", min_confidence_extend={min_confidence_extend} (hystérésis)" if hysteresis_active else "")
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
                conf_value = conf if conf is not None else 0.0
                # Filtre dur sur le seuil bas (extension). Les détections sous
                # min_confidence_extend sont totalement écartées.
                if conf_value < min_confidence_extend:
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

        # Mask de core-éligibilité : seules les détections avec
        # conf >= min_confidence peuvent initier/étendre un cluster.
        is_core_eligible = np.array(
            [
                ((c[4] if c[4] is not None else 0.0) >= min_confidence)
                for c in candidates
            ],
            dtype=bool,
        )
        n_core = int(is_core_eligible.sum())
        n_extend = int((~is_core_eligible).sum())
        if hysteresis_active:
            logger.info(
                f"Clustering: {n_core} détection(s) core-éligibles "
                f"(conf>={min_confidence}), {n_extend} détection(s) extension "
                f"(conf in [{min_confidence_extend}, {min_confidence}[)"
            )

        # Exécuter DBSCAN avec hystérésis
        try:
            labels = _dbscan_scipy(
                centroids,
                eps=eps_m,
                min_samples=min_samples,
                is_core_eligible=is_core_eligible,
            )
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
                cluster_points,
                cluster_geoms,
                output_geometry,
                buffer_m,
                concave_ratio=concave_ratio,
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
