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
from typing import Any, Dict, List, Optional

from .types import Detection

logger = logging.getLogger(__name__)

# IoS (aire intersection / aire du plus petit) au-delà duquel deux polygones de
# même classe sont considérés en confinement quasi-total : ils fusionnent
# TOUJOURS, même si le garde-fou de similarité de taille (``min_area_ratio``)
# l'interdirait — sinon un petit polygone entièrement dans un grand
# ré-apparaîtrait (artefact d'imbrication). Cf. _resolve_same_class_overlaps.
_CONTAINMENT_IOS = 0.9


def buffer_union_debuffer(
    polys: List[Any],
    buffer_px: float,
    join_style: int = 2,
) -> Optional[List[Any]]:
    """Pattern de fusion de polygones via buffer/union/debuffer.

    Encapsule l'incantation Shapely partagée par tous les sites de
    fusion intra-classe : on dilate chaque polygone d'un petit buffer,
    on prend l'union (les polygones qui se touchaient ou se chevauchaient
    forment une seule géométrie), on rétrécit du même buffer pour
    récupérer la taille d'origine.

    Cette fonction expose **uniquement** le noyau géométrique : les
    politiques métier (filtrage par compactness, limite d'aire,
    confiance pondérée…) restent dans les wrappers des appelants.

    Args:
        polys: liste de ``shapely.geometry.Polygon`` (déjà validés).
        buffer_px: rayon du buffer en pixels (ou en unités du SCR si
            on travaille en géo). ``join_style=2`` = mitre (défaut
            historique des deux call-sites).

    Returns:
        Liste de Polygons résultant de l'union, ou ``None`` si
        Shapely n'est pas disponible ou que l'union a échoué (les
        appelants doivent fallback sur leurs polygones d'origine).
    """
    try:
        from shapely.ops import unary_union
    except ImportError:
        return None
    if not polys:
        return []
    try:
        buffered = [p.buffer(buffer_px, join_style=join_style) for p in polys]
        merged = unary_union(buffered).buffer(-buffer_px, join_style=join_style)
    except Exception as e:
        logger.debug(f"buffer_union_debuffer: erreur union: {e}")
        return None
    if merged.is_empty:
        return []
    if merged.geom_type == "Polygon":
        return [merged]
    if merged.geom_type == "MultiPolygon":
        return list(merged.geoms)
    if merged.geom_type == "GeometryCollection":
        return [g for g in merged.geoms if g.geom_type == "Polygon"]
    return None


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

        # Noyau partagé : buffer → union → debuffer (V2.3).
        merged_polys = buffer_union_debuffer(polys, TOUCH_BUFFER_PX)
        if merged_polys is None:
            result.extend(items)
            continue
        if not merged_polys:
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

        typed_detections = [Detection.from_disk_dict(d) for d in detections]
        for annotated_path in candidates:
            save_annotated_image(
                pil_image, typed_detections, str(annotated_path),
                class_names=class_names, class_colors=class_colors,
            )
            log(f"Post-traitement: image annotée régénérée -> {annotated_path.name}")
    except Exception as e:
        log(f"Post-traitement: erreur régénération image annotée: {e}")


# ------------------------------------------------------------------ #
#  Post-processing global en coordonnées géographiques                 #
# ------------------------------------------------------------------ #

def _validate_class_dets(detections, make_valid_fn, min_area_m2: float) -> list:
    """
    Filtre + répare les géométries d'une classe avant fusion.

    Returns:
        Liste des détections valides (Polygon, non vides, aire ≥ ``min_area_m2``).
    """
    valid = []
    for det in detections:
        geom = det.get("geometry")
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            try:
                geom = make_valid_fn(geom)
                det = dict(det, geometry=geom)
            except Exception:
                continue
        if geom.geom_type != "Polygon":
            candidates = [g for g in getattr(geom, "geoms", [])
                          if g.geom_type == "Polygon" and not g.is_empty]
            if not candidates:
                continue
            geom = max(candidates, key=lambda g: g.area)
            det = dict(det, geometry=geom)
        if min_area_m2 > 0 and geom.area < min_area_m2:
            continue
        valid.append(det)
    return valid


def _connected_components_via_strtree(
    polys: list,
    merge_buffer_m: float,
    STRtree,
) -> list:
    """
    Identifie les composantes connexes de polygones qui se touchent ou sont
    distants de ≤ ``merge_buffer_m`` (vraie distance, prédicat ``dwithin``).

    Utilise STRtree + union-find pour grouper en O(N log N) au lieu de
    ``unary_union`` global O(N²) en pratique. Pour les modèles à détections
    majoritairement disjointes (ex. cratères), la plupart des composantes sont
    de taille 1 → traitement quasi-gratuit.

    Args:
        polys: Liste de Polygon shapely.
        merge_buffer_m: Distance max pour considérer deux polygones connectés.
        STRtree: Classe ``shapely.STRtree`` (passée en argument pour éviter
            l'import si shapely<2.0).

    Returns:
        Liste de listes d'indices ``[[i1, i2, ...], [j1, ...], ...]``.
        Chaque sous-liste est une composante connexe.
    """
    n = len(polys)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Connexion par VRAIE distance (F6) : deux polygones sont voisins si leur
    # écart est ≤ ``merge_buffer_m``. On interroge le STRtree avec le prédicat
    # ``dwithin`` (shapely ≥ 2.0) sur les polygones d'origine — et NON en testant
    # l'intersection de polygones bufferisés de ``merge_buffer_m`` chacun, ce qui
    # connectait dès qu'un gap ≤ ``2×merge_buffer_m`` (1,0 m au lieu des 0,5 m
    # annoncés dans la doc).
    tree = STRtree(polys)

    # Requête bulk : ``shapely.STRtree.query`` accepte une liste de géométries
    # en entrée et renvoie deux arrays d'indices (input_idx, tree_idx).
    try:
        input_idx, tree_idx = tree.query(
            polys, predicate="dwithin", distance=merge_buffer_m
        )
    except TypeError:
        # API plus ancienne (shapely 2.0.0) : query() ne prend qu'une seule géom
        input_idx_list, tree_idx_list = [], []
        for i, p in enumerate(polys):
            for j in tree.query(p, predicate="dwithin", distance=merge_buffer_m):
                input_idx_list.append(i)
                tree_idx_list.append(int(j))
        import numpy as _np
        input_idx = _np.array(input_idx_list)
        tree_idx = _np.array(tree_idx_list)

    # Union-find avec compression de chemin
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Unir les paires (i, j) avec j > i pour éviter les doublons et auto-paires
    for i, j in zip(input_idx, tree_idx):
        if int(j) > int(i):
            union(int(i), int(j))

    # Regrouper par racine
    components: dict = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    return list(components.values())


def _merge_intra_class_components(
    valid_dets: list,
    merge_buffer_m: float,
    unary_union_fn,
    make_valid_fn,
    STRtree,
    min_area_m2: float,
) -> list:
    """
    Fusion intra-classe par composantes connexes (Proposition #2 + #3).

    - Les polygones isolés (composantes de taille 1) sont conservés tels
      quels, sans buffer/union/debuffer (Proposition #3 : skip pur).
    - Les polygones d'une même composante sont fusionnés via
      ``unary_union`` mais uniquement sur leur sous-ensemble local
      (Proposition #2 : pas d'union globale).

    Pour 311k cratères majoritairement isolés, on passe de ~76 minutes
    (``unary_union`` global) à environ 30 secondes (STRtree + union-find).
    """
    import time as _time
    n = len(valid_dets)
    if n == 0:
        return []
    if n == 1:
        return list(valid_dets)

    polys = [d["geometry"] for d in valid_dets]

    t0 = _time.perf_counter()
    components = _connected_components_via_strtree(polys, merge_buffer_m, STRtree)
    t1 = _time.perf_counter()

    n_isolated = sum(1 for c in components if len(c) == 1)
    n_groups = len(components) - n_isolated
    largest_group = max((len(c) for c in components), default=0)
    logger.info(
        f"  Composantes connexes: {len(components)} total — {n_isolated} isolées, "
        f"{n_groups} groupes (max {largest_group} polygones), {t1 - t0:.1f}s"
    )

    result = []
    for comp_indices in components:
        if len(comp_indices) == 1:
            # Fast-path : polygone isolé, aucune fusion à faire
            result.append(valid_dets[comp_indices[0]])
            continue

        # Fusion locale par composante
        comp_polys = [polys[i] for i in comp_indices]
        try:
            buffered = [p.buffer(merge_buffer_m, join_style=2) for p in comp_polys]
            merged = unary_union_fn(buffered).buffer(-merge_buffer_m, join_style=2)
        except Exception as e:
            logger.debug(f"  Fusion composante {comp_indices}: erreur union: {e}")
            result.extend(valid_dets[i] for i in comp_indices)
            continue

        if merged.is_empty:
            result.extend(valid_dets[i] for i in comp_indices)
            continue

        # Extraire les Polygons résultants
        if merged.geom_type == "Polygon":
            result_polys = [merged]
        elif merged.geom_type == "MultiPolygon":
            result_polys = list(merged.geoms)
        elif merged.geom_type == "GeometryCollection":
            result_polys = [g for g in merged.geoms if g.geom_type == "Polygon"]
        else:
            result.extend(valid_dets[i] for i in comp_indices)
            continue

        # Pour chaque polygone fusionné, attribuer une confiance pondérée
        # par l'aire des sources de la composante qui le constituent
        for mp in result_polys:
            if not mp.is_valid:
                mp = make_valid_fn(mp)
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

            total_w = 0.0
            weighted_conf = 0.0
            template_det = None
            for src_idx in comp_indices:
                src_geom = polys[src_idx]
                try:
                    if mp.intersects(src_geom):
                        det = valid_dets[src_idx]
                        w = src_geom.area
                        total_w += w
                        weighted_conf += det.get("confidence", 0.5) * w
                        if (template_det is None
                                or det.get("confidence", 0) > template_det.get("confidence", 0)):
                            template_det = det
                except Exception:
                    pass

            conf = weighted_conf / total_w if total_w > 0 else 0.5
            if template_det is None:
                template_det = valid_dets[comp_indices[0]]
            result.append(dict(template_det, geometry=mp, confidence=conf))

    return result


def _resolve_same_class_overlaps(
    dets: List[Dict],
    ios_threshold: float,
    unary_union_fn,
    STRtree,
    min_area_ratio: float = 0.0,
) -> List[Dict]:
    """Résout les superpositions entre détections d'UNE même classe par union.

    Deux polygones dont l'IoS (aire intersection / aire du plus petit) est
    ≥ ``ios_threshold`` sont **fusionnés par union** : un petit polygone
    entièrement contenu dans un grand (IoS≈1) est absorbé par l'union (= le
    grand, donc **sans anneau**), et deux fragments fortement chevauchants sont
    soudés (donc **sans arête de découpe**). Les polygones disjoints ou
    faiblement chevauchants (IoS < seuil) sont conservés intacts — deux cratères
    sécants légitimes ne sont pas perdus.

    ``min_area_ratio`` (garde-fou de similarité de taille, 0 = désactivé) : quand
    > 0, deux polygones ne fusionnent que si leur ratio d'aire
    ``min_aire / max_aire`` est ≥ ``min_area_ratio`` — SAUF confinement
    quasi-total (IoS ≥ :data:`_CONTAINMENT_IOS`) qui fusionne toujours. Cela
    permet d'abaisser ``ios_threshold`` pour rattraper les doublons modérés (deux
    détections de taille proche d'un même cratère) sans fusionner un petit
    cratère distinct posé sur le bord d'un grand.

    Implémentation : graphe de chevauchement (arête = IoS ≥ seuil, garde-fou
    respecté) via STRtree (fallback O(N²) si ``STRtree is None``), composantes
    connexes par union-find, puis ``unary_union`` par composante. Le gabarit
    (attributs) de chaque composante est la détection la plus confiante, avec la
    géométrie unionnée.
    """
    items = [
        d for d in dets
        if d.get("geometry") is not None and not d["geometry"].is_empty
    ]
    n = len(items)
    if n < 2:
        return list(items)

    geoms = [d["geometry"] for d in items]

    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    tree = None
    if STRtree is not None:
        try:
            tree = STRtree(geoms)
        except Exception:
            tree = None

    for i in range(n):
        gi = geoms[i]
        if tree is not None:
            try:
                cand = [int(k) for k in tree.query(gi, predicate="intersects")]
            except Exception:
                cand = list(range(n))
        else:
            cand = list(range(n))
        for j in cand:
            if j <= i:
                continue
            gj = geoms[j]
            try:
                if not gi.intersects(gj):
                    continue
                inter = gi.intersection(gj).area
            except Exception:
                continue
            if inter <= 0:
                continue
            min_area = min(gi.area, gj.area)
            if min_area <= 0:
                continue
            ios = inter / min_area
            if ios < ios_threshold:
                continue
            # Garde-fou de similarité de taille : sur la bande de chevauchement
            # modéré on n'unit que des tailles proches (vrais doublons). Le
            # confinement quasi-total (IoS ≥ _CONTAINMENT_IOS) y échappe pour
            # préserver la correction de l'imbrication petit-dans-grand.
            if min_area_ratio > 0 and ios < _CONTAINMENT_IOS:
                if min_area / max(gi.area, gj.area) < min_area_ratio:
                    continue
            _union(i, j)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(_find(i), []).append(i)

    out: List[Dict] = []
    for members in comps.values():
        if len(members) == 1:
            out.append(items[members[0]])
            continue
        best = max(members, key=lambda k: items[k].get("confidence", 0.0))
        try:
            merged = unary_union_fn([geoms[k] for k in members])
        except Exception:
            merged = geoms[best]
        if merged.geom_type in ("MultiPolygon", "GeometryCollection"):
            polys = [
                g for g in merged.geoms
                if g.geom_type == "Polygon" and not g.is_empty
            ]
            merged = max(polys, key=lambda g: g.area) if polys else geoms[best]
        out.append(dict(items[best], geometry=merged))
    return out


def _remove_cross_class_overlaps(
    data_by_class_name: Dict[str, List[Dict]],
    unary_union_fn,
    min_area_m2: float = 0.0,
) -> Dict[str, List[Dict]]:
    """Suppression des superpositions INTER-classes par ordre de confiance.

    Chaque détection est rognée (``difference``) de l'union des détections déjà
    acceptées d'une **autre** classe qui l'intersectent. Les détections de même
    classe ne se rognent jamais (résolues en amont par
    :func:`_resolve_same_class_overlaps`). Pour un modèle mono-classe, c'est un
    no-op (aucune autre classe).
    """
    all_dets = [(cls, d) for cls, ds in data_by_class_name.items() for d in ds]
    all_dets.sort(key=lambda t: t[1].get("confidence", 0), reverse=True)

    accepted: list = []  # (class_name, geom, det)
    for class_name, det in all_dets:
        geom = det.get("geometry")
        if geom is None or geom.is_empty:
            continue
        diff_geoms = [
            g for (c, g, _) in accepted
            if c != class_name and geom.intersects(g)
        ]
        if diff_geoms:
            try:
                geom = geom.difference(unary_union_fn(diff_geoms))
            except Exception:
                pass
            if geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                pass
            elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
                polys = [
                    g for g in geom.geoms
                    if g.geom_type == "Polygon" and not g.is_empty
                ]
                if not polys:
                    continue
                geom = max(polys, key=lambda g: g.area)
            else:
                continue
        if geom.is_empty:
            continue
        if min_area_m2 > 0 and geom.area < min_area_m2:
            continue
        accepted.append((class_name, geom, dict(det, geometry=geom)))

    out: Dict[str, List[Dict]] = {}
    for class_name, _, det in accepted:
        out.setdefault(class_name, []).append(det)
    return out


def postprocess_geo_detections(
    data_by_class_name: Dict[str, List[Dict]],
    merge_buffer_m: float = 0.5,
    min_area_m2: float = 0.0,
    do_merge: bool = True,
    do_remove_overlaps: bool = True,
    overlap_strategy: str = "difference",
    overlap_ios_threshold: float = 0.5,
    overlap_min_area_ratio: float = 0.0,
) -> Dict[str, List[Dict]]:
    """
    Post-traitement global des détections en coordonnées géographiques,
    opérant sur l'ensemble des dalles à la fois.

    Étapes (toutes optionnelles via flags) :
      1. Validation et réparation des géométries invalides — toujours faite
      2. **Fusion intra-classe** des polygones qui se touchent ou sont séparés
         par un gap ≤ ``merge_buffer_m`` — activée par ``do_merge=True``.
         Implémentée via STRtree + union-find + composantes connexes :
         O(N log N) au lieu de l'ancien ``unary_union`` global.
         Les polygones isolés (sans voisin) sont passés directement sans
         buffer/union/debuffer (gros gain sur les modèles type cratère).
      3. **Suppression des superpositions** — activée par
         ``do_remove_overlaps=True``. Deux stratégies (``overlap_strategy``) :

         - ``"difference"`` (défaut, historique) : **class-agnostic**, par ordre
           de confiance décroissante, chaque polygone est rogné de l'union des
           polygones déjà acceptés qui l'intersectent (``geom.difference``).
           ⚠ Pour un modèle mono-classe ce découpage fabrique des artefacts :
           anneau troué (petit imbriqué dans un grand plus confiant) ou arête
           droite partagée (deux polygones accolés).
         - ``"relation"`` : pour les détections de **même classe**, on raisonne
           en confinement — si l'IoS (aire intersection / aire du plus petit) de
           deux polygones est ≥ ``overlap_ios_threshold`` on les **fusionne par
           union** (l'union absorbe le petit dans le grand sans anneau, et soude
           deux fragments fortement chevauchants sans arête). Les superpositions
           **inter-classes** restent gérées par ``difference``.

    Args:
        data_by_class_name: ``{class_name: [det_dict, ...]}``. Chaque
            ``det_dict`` doit contenir ``"geometry"`` (Polygon shapely) et
            ``"confidence"`` (float).
        merge_buffer_m: Distance max (mètres) pour fusionner deux polygones de
            même classe. 0.5 m par défaut (~1 pixel à 0.5 m/px).
        min_area_m2: Aire minimale en m² pour conserver un polygone après
            fusion. 0 = pas de filtre.
        do_merge: Si ``True`` (défaut), exécute l'étape 2.
        do_remove_overlaps: Si ``True`` (défaut), exécute l'étape 3.
        overlap_strategy: ``"difference"`` (défaut) ou ``"relation"`` — voir ci-dessus.
        overlap_ios_threshold: seuil IoS (]0, 1]) au-delà duquel deux polygones
            de même classe sont fusionnés en stratégie ``"relation"``. 0.5 par défaut.
        overlap_min_area_ratio: garde-fou de similarité de taille ([0, 1], 0 =
            désactivé) appliqué en stratégie ``"relation"`` — sur la bande de
            chevauchement modéré, ne fusionne que des polygones de taille proche
            (ratio min_aire/max_aire ≥ ce seuil), sauf confinement quasi-total.

    Returns:
        Nouveau ``{class_name: [det_dict, ...]}`` post-traité. Si les deux
        flags sont à ``False``, seule la validation des géométries est faite.
    """
    import time as _time

    try:
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
        from shapely import STRtree
    except ImportError:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon  # noqa: F401 — sonde de dispo
            from shapely.ops import unary_union
            from shapely.validation import make_valid
            STRtree = None  # type: ignore[assignment]
        except ImportError:
            logger.warning("shapely non disponible, post-traitement géo ignoré")
            return data_by_class_name

    t_start = _time.perf_counter()
    total_before = sum(len(v) for v in data_by_class_name.values())

    # ── Étape 1 : validation des géométries (toujours) ───────────────
    validated_by_class: Dict[str, List[Dict]] = {}
    for class_name, detections in data_by_class_name.items():
        valid_dets = _validate_class_dets(detections, make_valid, min_area_m2)
        if valid_dets:
            validated_by_class[class_name] = valid_dets

    # ── Étape 2 : fusion intra-classe ────────────────────────────────
    if do_merge and STRtree is not None:
        merged_by_class: Dict[str, List[Dict]] = {}
        t1 = _time.perf_counter()
        for class_name, valid_dets in validated_by_class.items():
            if len(valid_dets) < 2:
                merged_by_class[class_name] = valid_dets
                continue
            logger.info(
                f"Fusion intra-classe '{class_name}': {len(valid_dets)} polygones"
            )
            merged_by_class[class_name] = _merge_intra_class_components(
                valid_dets,
                merge_buffer_m=merge_buffer_m,
                unary_union_fn=unary_union,
                make_valid_fn=make_valid,
                STRtree=STRtree,
                min_area_m2=min_area_m2,
            )
        total_after_merge = sum(len(v) for v in merged_by_class.values())
        t2 = _time.perf_counter()
        logger.info(
            f"Post-traitement géo: fusion intra-classe {total_before} -> "
            f"{total_after_merge} polygones ({t2 - t1:.1f}s)"
        )
    else:
        if do_merge and STRtree is None:
            logger.warning(
                "Fusion intra-classe demandée mais shapely.STRtree indisponible "
                "(shapely<2.0), étape ignorée"
            )
        merged_by_class = validated_by_class
        total_after_merge = sum(len(v) for v in merged_by_class.values())

    # ── Étape 3 : suppression des superpositions ──────
    if not do_remove_overlaps:
        t_end = _time.perf_counter()
        logger.info(
            f"Suppression superpositions désactivée (do_remove_overlaps=False), "
            f"{total_after_merge} polygones conservés"
        )
        logger.info(f"Post-traitement géo terminé en {t_end - t_start:.1f}s")
        return merged_by_class

    # ── Stratégie "relation" : fusion intra-classe par confinement (IoS),
    #    puis découpage inter-classes seulement. Corrige les artefacts
    #    "anneau troué" et "arête droite partagée" du découpage class-agnostic.
    if overlap_strategy == "relation":
        resolved_by_class: Dict[str, List[Dict]] = {}
        for class_name, dets in merged_by_class.items():
            resolved_by_class[class_name] = _resolve_same_class_overlaps(
                dets, overlap_ios_threshold, unary_union, STRtree,
                min_area_ratio=overlap_min_area_ratio,
            )
        result_by_class = _remove_cross_class_overlaps(
            resolved_by_class, unary_union, min_area_m2
        )
        total_final = sum(len(v) for v in result_by_class.values())
        t_end = _time.perf_counter()
        logger.info(
            f"Post-traitement géo (relation, IoS>={overlap_ios_threshold}): "
            f"{total_after_merge} -> {total_final} polygones "
            f"({t_end - t_start:.1f}s)"
        )
        return result_by_class

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

    t2 = _time.perf_counter()
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
