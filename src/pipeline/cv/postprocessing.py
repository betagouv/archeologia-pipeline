"""
Post-traitement centralisé des détections de segmentation.

Ce module est appelé depuis runner.py après l'inférence (runner externe ou fallback)
pour appliquer les corrections géométriques sur les polygones produits :
  1. Validation et correction des self-intersections (boucles)
  2. Suppression des superpositions entre polygones

Il opère sur les fichiers JSON/TXT produits par le runner et les réécrit corrigés.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _merge_touching_same_class(
    indexed: list,
    img_width: int,
    img_height: int,
    min_area: float,
) -> list:
    """
    Fusionne les polygones de même classe qui se touchent ou sont séparés
    par un gap ≤ TOUCH_BUFFER_PX (artefact de pixellisation).

    Args:
        indexed: liste de tuples (ShapelyPolygon, det_dict)
        img_width, img_height: dimensions image en pixels
        min_area: aire minimale en px²

    Returns:
        Nouvelle liste de tuples (ShapelyPolygon, det_dict).
    """
    try:
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError:
        return indexed

    TOUCH_BUFFER_PX = 1.5  # micro-buffer pour combler les gaps de pixellisation

    # Grouper par class_id
    by_class: dict = {}
    for sp, det in indexed:
        cid = det.get("class_id", 0)
        by_class.setdefault(cid, []).append((sp, det))

    result = []
    n_before = len(indexed)

    for class_id, items in by_class.items():
        if len(items) < 2:
            result.extend(items)
            continue

        polys = [sp for sp, _ in items]
        confs = [d.get("confidence", 0.5) for _, d in items]
        areas = [sp.area for sp in polys]

        # Buffer → union → débuffer pour fusionner les polygones qui se touchent
        try:
            buffered = [p.buffer(TOUCH_BUFFER_PX, join_style=2) for p in polys]
            merged = unary_union(buffered).buffer(-TOUCH_BUFFER_PX, join_style=2)
        except Exception as e:
            logger.debug(f"Merge touching: erreur union classe {class_id}: {e}")
            result.extend(items)
            continue

        if merged.is_empty:
            result.extend(items)
            continue

        # Extraire les polygones résultants
        if merged.geom_type == "Polygon":
            merged_polys = [merged]
        elif merged.geom_type == "MultiPolygon":
            merged_polys = list(merged.geoms)
        elif merged.geom_type == "GeometryCollection":
            merged_polys = [g for g in merged.geoms if g.geom_type == "Polygon"]
        else:
            result.extend(items)
            continue

        # Pour chaque polygone fusionné, calculer la confiance pondérée par l'aire
        # des polygones sources qui le composent
        for mp in merged_polys:
            if not mp.is_valid:
                mp = make_valid(mp)
                if mp.geom_type != "Polygon":
                    candidates = [g for g in getattr(mp, "geoms", [])
                                  if g.geom_type == "Polygon"]
                    mp = max(candidates, key=lambda g: g.area) if candidates else mp
            if mp.is_empty or mp.area < min_area:
                continue

            # Confiance = moyenne pondérée par l'aire des sources qui intersectent
            total_w = 0.0
            weighted_conf = 0.0
            contributing_holes = []
            for sp, det in items:
                try:
                    if mp.intersects(sp.buffer(TOUCH_BUFFER_PX)):
                        w = sp.area
                        total_w += w
                        weighted_conf += det.get("confidence", 0.5) * w
                        # Collecter les trous des sources contribuantes
                        for hole in det.get("polygon_holes", []):
                            contributing_holes.append(hole)
                except Exception:
                    pass
            conf = weighted_conf / total_w if total_w > 0 else 0.5

            new_det = {
                "class_id": class_id,
                "confidence": conf,
                "polygon": [],  # sera reconverti plus tard si besoin
                "bbox": list(mp.bounds),
                "area": float(mp.area),
            }
            if contributing_holes:
                new_det["polygon_holes"] = contributing_holes

            result.append((mp, new_det))

    if len(result) != n_before:
        logger.info(f"Fusion polygones adjacents même classe: {n_before} -> {len(result)}")

    return result


def postprocess_detections(
    detections: List[Dict],
    img_width: int,
    img_height: int,
    min_area: float = 10.0,
) -> List[Dict]:
    """
    Post-traitement centralisé des détections de segmentation :
      1. Valide et corrige les polygones (élimine les self-intersections / boucles)
      1.5. Fusionne les polygones de même classe qui se touchent ou quasi-touchent
      2. Élimine les superpositions entre polygones (soustrait les zones de chevauchement)

    Les détections sont traitées par ordre de confiance décroissante :
    le polygone le plus confiant conserve sa géométrie intacte, les suivants
    sont découpés pour ne garder que la partie non recouverte.

    Args:
        detections: Liste des détections avec "polygon" normalisé [x1,y1,x2,y2,...]
        img_width, img_height: Dimensions de l'image en pixels
        min_area: Aire minimum en pixels² pour conserver un polygone

    Returns:
        Liste de détections nettoyées (polygones valides, sans superposition).
    """
    if not detections:
        return detections

    # Vérifier si des polygones existent (mode détection bbox-only → rien à faire)
    has_polygons = any("polygon" in d for d in detections)
    if not has_polygons:
        return detections

    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.validation import make_valid
    except ImportError:
        logger.warning("shapely non disponible, post-traitement polygones ignoré")
        return detections

    def _norm_to_shapely(polygon_norm):
        """Convertit une liste plate normalisée en ShapelyPolygon pixel."""
        if len(polygon_norm) < 6:
            return None
        coords = [(polygon_norm[i] * img_width, polygon_norm[i + 1] * img_height)
                   for i in range(0, len(polygon_norm), 2)]
        try:
            poly = ShapelyPolygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.geom_type == 'Polygon':
                return poly if (not poly.is_empty and poly.area >= min_area) else None
            elif poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
                candidates = [g for g in poly.geoms
                              if g.geom_type == 'Polygon' and g.area >= min_area]
                return max(candidates, key=lambda g: g.area) if candidates else None
            return None
        except Exception:
            return None

    def _shapely_to_norm(poly):
        """Convertit un ShapelyPolygon en liste plate normalisée."""
        try:
            coords = list(poly.exterior.coords)
        except Exception:
            return None
        result = []
        for x, y in coords[:-1]:
            result.extend([max(0.0, min(x, img_width)) / img_width,
                           max(0.0, min(y, img_height)) / img_height])
        return result if len(result) >= 6 else None

    # --- Étape 1 : convertir et valider tous les polygones ---
    indexed = []  # (ShapelyPolygon, det)
    bbox_only = []  # détections sans polygone (bbox-only)
    n_invalid = 0

    for det in detections:
        if "polygon" not in det:
            bbox_only.append(det)
            continue
        sp = _norm_to_shapely(det["polygon"])
        if sp is None:
            n_invalid += 1
            continue
        # Valider aussi les trous
        validated_holes = []
        for hole in det.get("polygon_holes", []):
            h = _norm_to_shapely(hole)
            if h is not None:
                validated_holes.append(hole)
        det_copy = dict(det)
        if validated_holes:
            det_copy["polygon_holes"] = validated_holes
        elif "polygon_holes" in det_copy:
            del det_copy["polygon_holes"]
        indexed.append((sp, det_copy))

    if n_invalid > 0:
        logger.info(f"Post-traitement: {n_invalid} polygone(s) invalide(s) supprimé(s)")

    if not indexed:
        return bbox_only

    # --- Étape 1.5 : fusionner les polygones de même classe qui se touchent ---
    indexed = _merge_touching_same_class(indexed, img_width, img_height, min_area)

    # --- Étape 2 : éliminer les superpositions ---
    # Trier par confiance décroissante : le plus confiant a priorité
    indexed.sort(key=lambda t: t[1].get("confidence", 0), reverse=True)

    occupied = None  # union des géométries déjà attribuées
    result_dets = []

    for sp, det in indexed:
        if occupied is not None:
            try:
                remainder = sp.difference(occupied)
            except Exception:
                remainder = sp
            if remainder.is_empty:
                continue
            # remainder peut être MultiPolygon → garder le plus grand morceau
            if remainder.geom_type == 'Polygon':
                sp = remainder
            elif remainder.geom_type in ('MultiPolygon', 'GeometryCollection'):
                candidates = [g for g in remainder.geoms
                              if g.geom_type == 'Polygon' and g.area >= min_area]
                if not candidates:
                    continue
                sp = max(candidates, key=lambda g: g.area)
            else:
                continue

        if sp.is_empty or sp.area < min_area:
            continue

        # Mettre à jour la zone occupée
        try:
            occupied = sp if occupied is None else occupied.union(sp)
        except Exception:
            occupied = sp

        # Reconvertir en coordonnées normalisées
        polygon_norm = _shapely_to_norm(sp)
        if polygon_norm is None:
            continue

        det["polygon"] = polygon_norm
        minx, miny, maxx, maxy = sp.bounds
        det["bbox"] = [minx, miny, maxx, maxy]
        det["area"] = float(sp.area)
        result_dets.append(det)

    logger.info(
        f"Post-traitement: {len(detections)} détections -> "
        f"{len(result_dets)} polygones valides sans superposition"
    )
    return bbox_only + result_dets


def postprocess_detection_files(
    json_path: Path,
    log=None,
    annotated_images_dir: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
    class_colors: Optional[List[int]] = None,
) -> bool:
    """
    Applique le post-traitement sur un fichier JSON de détections existant.
    Réécrit le JSON, le TXT YOLO, et régénère l'image annotée si elle existe.

    Args:
        json_path: Chemin vers le fichier JSON des détections
        log: Fonction de logging optionnelle
        annotated_images_dir: Dossier contenant les images annotées (pour régénération)
        class_names: Noms des classes (pour régénération image annotée)
        class_colors: Indices de couleurs par classe (pour régénération image annotée)

    Returns:
        True si le fichier a été modifié, False sinon.
    """
    _log = log or (lambda msg: None)

    if not json_path.exists():
        return False

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"Post-traitement: erreur lecture {json_path.name}: {e}")
        return False

    detections = payload.get("detections", [])
    dims = payload.get("image_dimensions", {})
    img_width = dims.get("width", 0)
    img_height = dims.get("height", 0)

    if not detections or not img_width or not img_height:
        return False

    # Ne traiter que les tâches de segmentation
    has_polygons = any("polygon" in d for d in detections)
    if not has_polygons:
        return False

    n_before = len(detections)
    cleaned = postprocess_detections(detections, img_width, img_height)
    n_after = len(cleaned)

    if n_before == n_after and all(
        d.get("polygon") == c.get("polygon")
        for d, c in zip(detections, cleaned)
    ):
        return False  # rien n'a changé

    # Réécrire le JSON
    payload["detections"] = cleaned
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Réécrire le TXT YOLO
    txt_path = json_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        for det in cleaned:
            class_id = det.get("class_id", 0)
            if "polygon" in det:
                polygon = det["polygon"]
                f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon) + "\n")
            elif "bbox_absolute" in det:
                bbox = det["bbox_absolute"]
                x1, y1 = bbox["minx"], bbox["miny"]
                x2, y2 = bbox["maxx"], bbox["maxy"]
                x_center = ((x1 + x2) / 2.0) / float(img_width)
                y_center = ((y1 + y2) / 2.0) / float(img_height)
                w_rel = (x2 - x1) / float(img_width)
                h_rel = (y2 - y1) / float(img_height)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}\n")

    # Régénérer l'image annotée si elle existe
    if annotated_images_dir is not None:
        _regenerate_annotated_image(
            payload, cleaned, annotated_images_dir, class_names, class_colors, _log
        )

    _log(f"Post-traitement: {json_path.stem} -> {n_before} -> {n_after} détections")
    return True


def _regenerate_annotated_image(
    payload: dict,
    detections: List[Dict],
    annotated_images_dir: Path,
    class_names: Optional[List[str]],
    class_colors: Optional[List[int]],
    log,
) -> None:
    """Régénère l'image annotée après post-traitement des polygones."""
    image_path = payload.get("image_path", "")
    if not image_path or not Path(image_path).exists():
        return

    # Chercher l'image annotée correspondante (uniquement les images, pas les .jgw/.pgw)
    stem = Path(image_path).stem
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    candidates = [
        p for p in annotated_images_dir.glob(f"{stem}_detections.*")
        if p.suffix.lower() in _IMG_EXTS
    ]
    if not candidates:
        return

    try:
        from PIL import Image
        from .cv_output import save_annotated_image

        with Image.open(image_path) as img:
            pil_image = img.convert("RGB")

        for annotated_path in candidates:
            save_annotated_image(
                pil_image, detections, str(annotated_path),
                class_names=class_names, class_colors=class_colors,
            )
            log(f"Post-traitement: image annotée régénérée -> {annotated_path.name}")
    except Exception as e:
        log(f"Post-traitement: erreur régénération image annotée: {e}")


# ------------------------------------------------------------------ #
#  Post-processing global en coordonnées géographiques                 #
# ------------------------------------------------------------------ #

def postprocess_geo_detections(
    data_by_class_name: Dict[str, List[Dict]],
    merge_buffer_m: float = 0.5,
    min_area_m2: float = 0.0,
) -> Dict[str, List[Dict]]:
    """
    Post-traitement global des détections en coordonnées géographiques,
    opérant sur l'ensemble des dalles à la fois.

    Étapes :
      1. Validation et réparation des géométries invalides
      2. Fusion des polygones de même classe qui se touchent ou sont
         séparés par un gap ≤ *merge_buffer_m* (en mètres, adapté au CRS)
      3. Suppression des superpositions inter-classes (le polygone le
         plus confiant conserve sa géométrie, les autres sont découpés)

    Args:
        data_by_class_name: ``{class_name: [det_dict, ...]}``
            Chaque ``det_dict`` **doit** contenir une clé ``"geometry"``
            (``shapely.geometry.Polygon``) et ``"confidence"`` (float).
        merge_buffer_m: Distance max (mètres) pour fusionner deux
            polygones de même classe. 0.5 m par défaut (~1 pixel à 0.5 m/px).
        min_area_m2: Aire minimale en m² pour conserver un polygone
            après fusion. 0 = pas de filtre.

    Returns:
        Nouveau ``{class_name: [det_dict, ...]}`` post-traité.
    """
    import time as _time

    try:
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
        from shapely import STRtree
    except ImportError:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
            from shapely.ops import unary_union
            from shapely.validation import make_valid
            STRtree = None
        except ImportError:
            logger.warning("shapely non disponible, post-traitement géo ignoré")
            return data_by_class_name

    t_start = _time.perf_counter()

    # ── Étape 1+2 : validation + fusion intra-classe ────────────────
    merged_by_class: Dict[str, List[Dict]] = {}
    total_before = sum(len(v) for v in data_by_class_name.values())

    t1 = _time.perf_counter()

    for class_name, detections in data_by_class_name.items():
        # Valider les géométries
        valid_dets = []
        for det in detections:
            geom = det.get("geometry")
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                try:
                    geom = make_valid(geom)
                    det = dict(det, geometry=geom)
                except Exception:
                    continue
            if geom.geom_type != "Polygon":
                # Extraire le plus grand Polygon d'un MultiPolygon/GeometryCollection
                candidates = [g for g in getattr(geom, "geoms", [])
                              if g.geom_type == "Polygon" and not g.is_empty]
                if not candidates:
                    continue
                geom = max(candidates, key=lambda g: g.area)
                det = dict(det, geometry=geom)
            if min_area_m2 > 0 and geom.area < min_area_m2:
                continue
            valid_dets.append(det)

        if not valid_dets:
            continue

        if len(valid_dets) < 2:
            merged_by_class[class_name] = valid_dets
            continue

        # Fusion des polygones qui se touchent (buffer → union → débuffer)
        polys = [d["geometry"] for d in valid_dets]

        try:
            buffered = [p.buffer(merge_buffer_m, join_style=2) for p in polys]
            merged = unary_union(buffered).buffer(-merge_buffer_m, join_style=2)
        except Exception as e:
            logger.debug(f"Post-traitement géo: erreur union classe '{class_name}': {e}")
            merged_by_class[class_name] = valid_dets
            continue

        if merged.is_empty:
            merged_by_class[class_name] = valid_dets
            continue

        # Extraire les Polygons résultants
        if merged.geom_type == "Polygon":
            result_polys = [merged]
        elif merged.geom_type == "MultiPolygon":
            result_polys = list(merged.geoms)
        elif merged.geom_type == "GeometryCollection":
            result_polys = [g for g in merged.geoms if g.geom_type == "Polygon"]
        else:
            merged_by_class[class_name] = valid_dets
            continue

        # Reconstruire les détections fusionnées
        # Utiliser un STRtree sur les sources bufferisées pour trouver rapidement
        # quels polygones sources contribuent à chaque polygone fusionné.
        src_buffered = [d["geometry"].buffer(merge_buffer_m) for d in valid_dets]
        if STRtree is not None:
            tree = STRtree(src_buffered)
        else:
            tree = None

        class_result = []
        for mp in result_polys:
            if not mp.is_valid:
                mp = make_valid(mp)
                if mp.geom_type != "Polygon":
                    candidates = [g for g in getattr(mp, "geoms", [])
                                  if g.geom_type == "Polygon"]
                    if not candidates:
                        continue
                    mp = max(candidates, key=lambda g: g.area)
            if mp.is_empty:
                continue
            if min_area_m2 > 0 and mp.area < min_area_m2:
                continue

            # Confiance = moyenne pondérée par l'aire des sources contribuantes
            total_w = 0.0
            weighted_conf = 0.0
            template_det = None

            if tree is not None:
                # Requête spatiale rapide : indices des sources dont le buffer intersecte mp
                candidate_idxs = tree.query(mp, predicate="intersects")
                for idx in candidate_idxs:
                    det = valid_dets[idx]
                    w = det["geometry"].area
                    total_w += w
                    weighted_conf += det.get("confidence", 0.5) * w
                    if template_det is None or det.get("confidence", 0) > template_det.get("confidence", 0):
                        template_det = det
            else:
                # Fallback sans STRtree (shapely < 2.0)
                for det in valid_dets:
                    src_geom = det["geometry"]
                    try:
                        if mp.intersects(src_geom.buffer(merge_buffer_m)):
                            w = src_geom.area
                            total_w += w
                            weighted_conf += det.get("confidence", 0.5) * w
                            if template_det is None or det.get("confidence", 0) > template_det.get("confidence", 0):
                                template_det = det
                    except Exception:
                        pass

            conf = weighted_conf / total_w if total_w > 0 else 0.5
            if template_det is None:
                template_det = valid_dets[0]

            new_det = dict(template_det, geometry=mp, confidence=conf)
            class_result.append(new_det)

        merged_by_class[class_name] = class_result if class_result else valid_dets

    total_after_merge = sum(len(v) for v in merged_by_class.values())
    t2 = _time.perf_counter()
    logger.info(
        f"Post-traitement géo: fusion intra-classe {total_before} -> {total_after_merge} polygones ({t2 - t1:.1f}s)"
    )

    # ── Étape 3 : suppression des superpositions inter-classes ──────
    # Collecter toutes les détections, trier par confiance décroissante
    all_dets = []
    for class_name, dets in merged_by_class.items():
        for det in dets:
            all_dets.append((class_name, det))

    if len(all_dets) < 2:
        t_end = _time.perf_counter()
        logger.info(f"Post-traitement géo terminé en {t_end - t_start:.1f}s")
        return merged_by_class

    all_dets.sort(key=lambda t: t[1].get("confidence", 0), reverse=True)

    # Utiliser un index spatial STRtree pour ne tester difference() que
    # contre les géométries déjà acceptées qui intersectent réellement.
    accepted_geoms: list = []     # géométries acceptées (même index que accepted_dets)
    accepted_classes: list = []   # classe pour chaque acceptée
    accepted_dets: list = []      # détections acceptées
    n_removed = 0

    # Reconstruire le STRtree toutes les N nouvelles géométries acceptées
    _REBUILD_EVERY = 200
    _occ_tree = None
    _tree_built_at = 0  # nombre d'accepted_geoms lors du dernier build

    for i, (class_name, det) in enumerate(all_dets):
        geom = det["geometry"]

        if accepted_geoms:
            if STRtree is not None and len(accepted_geoms) >= 2:
                # Reconstruire le tree si assez de nouvelles géométries ont été ajoutées
                if _occ_tree is None or (len(accepted_geoms) - _tree_built_at) >= _REBUILD_EVERY:
                    _occ_tree = STRtree(accepted_geoms)
                    _tree_built_at = len(accepted_geoms)

                # Trouver les géométries acceptées qui intersectent la bbox de geom
                candidate_idxs = _occ_tree.query(geom, predicate="intersects")

                # Ajouter aussi les géométries récentes non encore dans le tree
                recent_start = _tree_built_at
                extra_overlapping = []
                for ri in range(recent_start, len(accepted_geoms)):
                    try:
                        if geom.intersects(accepted_geoms[ri]):
                            extra_overlapping.append(accepted_geoms[ri])
                    except Exception:
                        pass

                overlapping = [accepted_geoms[idx] for idx in candidate_idxs] + extra_overlapping

                if overlapping:
                    try:
                        local_occupied = unary_union(overlapping)
                        remainder = geom.difference(local_occupied)
                    except Exception:
                        remainder = geom
                else:
                    remainder = geom
            else:
                # Fallback sans STRtree : union globale (lent mais correct)
                try:
                    occupied_union = unary_union(accepted_geoms)
                    remainder = geom.difference(occupied_union)
                except Exception:
                    remainder = geom

            if remainder.is_empty:
                n_removed += 1
                continue

            if remainder.geom_type == "Polygon":
                geom = remainder
            elif remainder.geom_type in ("MultiPolygon", "GeometryCollection"):
                candidates = [g for g in remainder.geoms
                              if g.geom_type == "Polygon" and not g.is_empty]
                if not candidates:
                    n_removed += 1
                    continue
                geom = max(candidates, key=lambda g: g.area)
            else:
                n_removed += 1
                continue

        if geom.is_empty:
            n_removed += 1
            continue
        if min_area_m2 > 0 and geom.area < min_area_m2:
            n_removed += 1
            continue

        accepted_geoms.append(geom)
        accepted_classes.append(class_name)
        accepted_dets.append(dict(det, geometry=geom))

    # Reconstruire result_by_class
    result_by_class: Dict[str, List[Dict]] = {}
    for cls, det in zip(accepted_classes, accepted_dets):
        result_by_class.setdefault(cls, []).append(det)

    total_final = sum(len(v) for v in result_by_class.values())
    t3 = _time.perf_counter()
    logger.info(
        f"Post-traitement géo: suppression superpositions {total_after_merge} -> {total_final} polygones ({t3 - t2:.1f}s)"
    )
    logger.info(f"Post-traitement géo terminé en {t3 - t_start:.1f}s")

    return result_by_class
