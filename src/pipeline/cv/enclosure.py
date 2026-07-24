"""Brique de synthèse « enclosure » : détection d'enclos par fermeture
vectorielle + scoring.

Principe (spec ``.claude/plans/2026-07-23-brique-enclos.md``) : les détections
sources (polygones de segmentation ``parcellaire``/``talus_fosse``) sont
unies puis **dilatées** de T/2 (joints mitre) — tout trou ≤ ``gap_tolerance_m``
est ponté. Les **anneaux intérieurs** de la forme dilatée, ré-étendus de +T/2
(l'érosion classique du closing re-sectionnerait les ponts sur bandes fines),
sont les surfaces encloses candidates. Trois filtres durs (aire, élongation,
``closure_ratio``), le reste est publié en attributs : l'archéologue tranche
dans QGIS.

Contrainte structurelle : un enclos dont la largeur intérieure est < T est
rempli par la fermeture (taille minimale détectable ≈ T).

Module pur (shapely + stdlib, ni scipy ni Qt) → testable hors QGIS, sûr sur
le thread worker. Même contrat que ``clustering.run_clustering``.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..cancellation import check_cancelled
from ..types import CancelCheckFn

logger = logging.getLogger(__name__)

# Tolérance de couverture (m) : un point du contour candidat est « couvert »
# s'il est à moins de cette distance d'une détection source. Sert au
# closure_ratio, à l'isolement et au rattachement des fragments sources.
COVER_EPS_M = 3.0

# Tolérance Douglas-Peucker (part du périmètre) avant classification de forme.
_SIMPLIFY_PERIMETER_RATIO = 0.03

# join_style=2 : mitre — préserve les angles droits des enclos quadrangulaires
# à l'aller-retour dilatation/érosion (le style rond les arrondirait).
_MITRE = 2


def _mrr_sides(geom) -> Tuple[float, float]:
    """Côtés (long, court) du rectangle englobant minimal orienté."""
    mrr = geom.minimum_rotated_rectangle
    coords = list(getattr(mrr, "exterior", mrr).coords) if hasattr(mrr, "exterior") else []
    if len(coords) < 5:
        return 0.0, 0.0
    a = math.dist(coords[0], coords[1])
    b = math.dist(coords[1], coords[2])
    return (max(a, b), min(a, b))


def _angle_concentration(ring_coords) -> float:
    """Concentration directionnelle des côtés, modulo 90° (∈ [0, 1]).

    Somme vectorielle pondérée par longueur des azimuts multipliés par 4
    (période 90°) : un quadrilatère aux côtés ⊥ → ~1, un cercle → ~0.
    """
    vx = vy = total = 0.0
    for (x1, y1), (x2, y2) in zip(ring_coords[:-1], ring_coords[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        theta = math.atan2(dy, dx)
        vx += length * math.cos(4.0 * theta)
        vy += length * math.sin(4.0 * theta)
        total += length
    return math.hypot(vx, vy) / total if total > 0 else 0.0


def _classify_shape(cand: Polygon, rectangularite: float) -> Tuple[str, float]:
    """Famille de forme (quadrangulaire / curviligne / irregulier) + compacité.

    # ponytail: heuristique concentration angulaire + compacité ; histogramme
    # d'angles complet (trapèzes obliques) si les cas réels la mettent en défaut.
    """
    perimeter = cand.exterior.length
    simplified = cand.simplify(_SIMPLIFY_PERIMETER_RATIO * perimeter)
    if simplified.is_empty or simplified.exterior is None:
        simplified = cand
    simp_perimeter = simplified.exterior.length
    compacite = (
        4.0 * math.pi * simplified.area / (simp_perimeter ** 2)
        if simp_perimeter > 0 else 0.0
    )
    concentration = _angle_concentration(list(simplified.exterior.coords))
    if concentration >= 0.75 and rectangularite >= 0.6:
        return "quadrangulaire", compacite
    if compacite >= 0.7:
        return "curviligne", compacite
    return "irregulier", compacite


def run_enclosure(
    data_by_class_name: Dict[str, List[Dict]],
    enclosure_configs: List[Dict],
    *,
    cancel_check: Optional[CancelCheckFn] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Exécute les règles « enclosure » sur les détections post-processées.

    Même contrat que :func:`clustering.run_clustering` :

    Returns:
        ``(enclos_by_class, data_by_class_name_annotées)`` — les détections
        sources contributrices reçoivent un ``enclos_id`` (traçabilité,
        symétrique de ``cluster_id``).
    """
    t0 = time.perf_counter()
    out_by_class: Dict[str, List[Dict]] = {}
    updated = {k: list(v) for k, v in data_by_class_name.items()}

    for cfg_idx, cfg in enumerate(enclosure_configs):
        check_cancelled(cancel_check)
        target_classes = cfg["target_classes"]
        gap_m = float(cfg["gap_tolerance_m"])
        min_area = float(cfg["min_area_m2"])
        max_area = float(cfg["max_area_m2"])
        min_closure = float(cfg["min_closure"])
        max_elongation = float(cfg["max_elongation"])
        min_ancrage = float(cfg.get("min_ancrage", 0.5))
        min_confidence = float(cfg.get("min_confidence", 0.0))
        # Mode calibration (campagnes ground-truth) : les candidats REJETÉS
        # sont aussi publiés, avec ``statut`` = premier filtre qui rejette —
        # analysables dans QGIS sans rejouer le pipeline en labo.
        mode_calibration = bool(cfg.get("mode_calibration", False))
        output_class = cfg["output_class_name"]
        logger.info(
            f"Enclosure [{cfg_idx + 1}/{len(enclosure_configs)}]: "
            f"classes={target_classes}, T={gap_m}m, aire=[{min_area:.0f};{max_area:.0f}]m², "
            f"closure>={min_closure}, elongation<={max_elongation}, ancrage>={min_ancrage}"
        )

        # Collecte des sources (conf >= min_confidence ; None = conservée).
        sources = []  # (class_name, det_idx, geometry, confidence)
        for class_name in target_classes:
            for det_idx, det in enumerate(updated.get(class_name, [])):
                geom = det.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                conf = det.get("confidence")
                if conf is not None and conf < min_confidence:
                    continue
                sources.append((class_name, det_idx, geom, conf))
        if not sources:
            logger.info("Enclosure: aucune détection source, règle ignorée")
            continue

        # Fermeture par dilatation seule : la dilatation de T/2 ponte tout trou
        # bout-à-bout ≤ T (les extrémités s'étendent de T/2 chacune). On ne
        # fait PAS l'érosion classique du closing : sur des bandes fines, elle
        # re-sectionne tout pont plus large que ~la largeur du fossé (le pont
        # est une lentille trop mince pour contenir le disque d'érosion). À la
        # place, chaque TROU de la forme dilatée est ré-étendu de +T/2 — ce qui
        # restitue exactement la cour côté fossés couverts, avec un léger
        # débord dans les entrées (acceptable, ce sont les zones non détectées).
        union = unary_union([s[2] for s in sources])
        half = gap_m / 2.0
        dilated = union.buffer(half, join_style=_MITRE)
        polys = list(getattr(dilated, "geoms", [dilated]))
        cover = union.buffer(COVER_EPS_M)

        # Candidats = anneaux intérieurs ré-étendus, filtres durs. Les rejets
        # sont comptés PAR filtre : « N surfaces, 0 candidat » sans explication
        # rendait le diagnostic impossible (cf. calibration Bretagne).
        candidates = []  # (polygon, surface, elongation, closure, ancrage, contrib_idx)
        rejected_dets: List[Dict] = []  # mode calibration uniquement
        n_rings = 0
        rejects = {"aire": 0, "elongation": 0, "closure": 0, "ancrage": 0}
        for poly in polys:
            check_cancelled(cancel_check)
            if poly.is_empty or not isinstance(poly, Polygon):
                continue
            for ring in poly.interiors:
                n_rings += 1
                cand = Polygon(ring).buffer(half)
                if not cand.is_valid:
                    cand = cand.buffer(0)
                if cand.is_empty or not isinstance(cand, Polygon):
                    continue
                # Les quatre métriques de filtre sont TOUTES calculées (le mode
                # calibration a besoin des scores même pour les rejetés) ; le
                # statut = premier filtre qui rejette, dans l'ordre.
                area = cand.area
                long_side, short_side = _mrr_sides(cand)
                elongation = (long_side / short_side) if short_side > 0 else float("inf")
                ring_line = cand.exterior
                covered = ring_line.intersection(cover)
                closure = (covered.length / ring_line.length) if ring_line.length > 0 else 0.0
                # Ancrage : part de l'aire des fragments contributeurs qui reste
                # au voisinage de l'anneau. Un vrai enclos EST sa détection
                # (ancrage ≈ 1) ; une cour incidente entre des lanières de
                # parcellaire qui continuent au loin a un ancrage faible — LE
                # discriminant des faux positifs inter-lanières (test Bretagne :
                # vrais 0,68/0,96, faux ≤ 0,59, médiane 0,11).
                contrib_idx = [
                    j for j, s in enumerate(sources)
                    if s[2].distance(ring_line) <= COVER_EPS_M
                ]
                a_tot = sum(sources[j][2].area for j in contrib_idx)
                if a_tot > 0:
                    ring_zone = ring_line.buffer(2 * COVER_EPS_M)
                    a_in = sum(
                        sources[j][2].intersection(ring_zone).area for j in contrib_idx
                    )
                    ancrage = a_in / a_tot
                else:
                    ancrage = 0.0

                if area < min_area or area > max_area:
                    statut = "rejete_aire"
                elif elongation > max_elongation:
                    statut = "rejete_elongation"
                elif closure < min_closure:
                    statut = "rejete_closure"
                elif ancrage < min_ancrage:
                    statut = "rejete_ancrage"
                else:
                    statut = "publie"

                if statut != "publie":
                    rejects[statut.removeprefix("rejete_")] += 1
                    if mode_calibration:
                        confs_r = [
                            sources[j][3] for j in contrib_idx
                            if sources[j][3] is not None and sources[j][3] > 0
                        ]
                        cf = (sum(confs_r) / len(confs_r)) if confs_r else 0.0
                        rejected_dets.append({
                            "validation": "", "corr_pred": None,
                            "model_pred": output_class, "model_name": "",
                            "geometry": cand,
                            "confidence": (cf * closure * ancrage) ** (1.0 / 3.0) if cf > 0 else 0.0,
                            "conf_fragments": round(cf, 3),
                            "surface_m2": round(area, 1),
                            "closure_ratio": round(closure, 3),
                            "ancrage": round(ancrage, 3),
                            "elongation": round(elongation, 2),
                            "nb_sources": len(contrib_idx),
                            "enclos_id": "", "statut": statut,
                        })
                    continue
                candidates.append((cand, area, elongation, closure, ancrage, contrib_idx))
        logger.info(
            f"Enclosure: {n_rings} surface(s) enclose(s), "
            f"{len(candidates)} candidat(s) après filtres durs "
            f"(rejets: aire {rejects['aire']}, élongation {rejects['elongation']}, "
            f"closure {rejects['closure']}, ancrage {rejects['ancrage']})"
        )

        # Isolement : part du périmètre à ≤ COVER_EPS_M du contour des AUTRES
        # candidats (une maille de parcellaire partage ses bords, pas un enclos).
        # ponytail: double boucle O(n²) avec pré-filtre intersects ; STRtree si
        # un chantier réel dépasse le millier de candidats.
        neighborhoods = [c[0].exterior.buffer(COVER_EPS_M) for c in candidates]
        dets: List[Dict] = []
        for i, (cand, area, elongation, closure, ancrage, contrib_idx) in enumerate(candidates):
            check_cancelled(cancel_check)
            ring_line = cand.exterior
            others = [
                neighborhoods[j] for j in range(len(candidates))
                if j != i and neighborhoods[j].intersects(ring_line)
            ]
            if others:
                shared = ring_line.intersection(unary_union(others))
                isolement = shared.length / ring_line.length if ring_line.length > 0 else 0.0
            else:
                isolement = 0.0

            mrr = cand.minimum_rotated_rectangle
            rectangularite = (area / mrr.area) if mrr.area > 0 else 0.0
            forme, compacite = _classify_shape(cand, rectangularite)

            # Fragments contributeurs (déjà résolus au filtrage d'ancrage) :
            # confiance moyenne + traçabilité enclos_id sur les sources.
            enclos_id = f"{output_class}_{i}"
            confs: List[float] = []
            model_name = ""
            for j in contrib_idx:
                class_name, det_idx, _geom, conf = sources[j]
                updated[class_name][det_idx]["enclos_id"] = enclos_id
                if not model_name:
                    model_name = updated[class_name][det_idx].get("model_name", "")
                if conf is not None and conf > 0:
                    confs.append(conf)
            nb_sources = len(contrib_idx)
            conf_fragments = (sum(confs) / len(confs)) if confs else 0.0
            # Confiance composite : moyenne géométrique des trois axes de
            # qualité (modèle, fermeture, appartenance) — aucun ne peut être
            # masqué par les autres. Binnable en conf_bin comme une détection.
            mean_confidence = (
                (conf_fragments * closure * ancrage) ** (1.0 / 3.0)
                if conf_fragments > 0 else 0.0
            )

            dets.append({
                "validation": "",
                "corr_pred": None,
                "model_pred": output_class,
                "model_name": model_name,
                "geometry": cand,
                "confidence": mean_confidence,
                "conf_fragments": round(conf_fragments, 3),
                "surface_m2": round(area, 1),
                "closure_ratio": round(closure, 3),
                "ancrage": round(ancrage, 3),
                "isolement": round(isolement, 3),
                "rectangularite": round(rectangularite, 3),
                "compacite": round(compacite, 3),
                "elongation": round(elongation, 2),
                "forme": forme,
                "nb_sources": nb_sources,
                "enclos_id": enclos_id,
                "statut": "publie",
            })

        if mode_calibration and rejected_dets:
            logger.info(
                f"Enclosure: mode calibration — {len(rejected_dets)} candidat(s) "
                f"rejeté(s) publié(s) avec leur statut"
            )
            dets.extend(rejected_dets)
        if dets:
            out_by_class[output_class] = dets
            logger.info(f"Enclosure: {len(dets)} enclos '{output_class}' publiés")

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in out_by_class.values())
    logger.info(f"Enclosure terminé: {total} enclos en {elapsed:.2f}s")
    return out_by_class, updated
