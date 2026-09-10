"""Brique de synthèse « enclosure » : détection d'enclos.

Trois générateurs de candidats (``generator`` de la règle) :

- ``auto`` (défaut, V3) : union de trois familles, dédoublonnée par IoU —
  les **anneaux pontés** (dilation), les **cours d'enveloppe** (hull) et les
  **blobs compacts** (la détection elle-même quand le modèle sort l'enclos en
  masse pleine — cas fréquent constaté en Bretagne, invisible aux deux autres
  générateurs). Priorité au dédoublonnage : anneau > cour > blob.
- ``hull`` (V2, corrigé) : pour chaque composante **brute** de l'union des
  détections, la « cour » candidate est l'enveloppe convexe moins la
  détection. L'ouverture d'un enclos en C est une *mesure* (``closure_ratio``)
  au lieu d'un obstacle à ponter. La liaison T/2 initiale a été retirée :
  sur le terrain elle soudait des lanières sans rapport et produisait des
  enveloppes géantes de plusieurs hectares (verdict campagne Bretagne).
- ``dilation`` (V1, conservé) : fermeture par dilatation T/2 + ré-extension
  des anneaux intérieurs — exige des circuits quasi complets.

Filtres durs (ordre du ``statut``) : aire → élongation → closure → ancrage →
isolement → rectangularité. Le reste est publié en attributs : l'archéologue
tranche dans QGIS. ``mode_calibration`` publie aussi les rejetés, avec
l'intégralité des scores.

Module pur (shapely + stdlib, ni scipy ni Qt) — même contrat que
``clustering.run_clustering``.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..cancellation import check_cancelled
from ..types import CancelCheckFn

logger = logging.getLogger(__name__)

# Tolérance de couverture (m) : un point du contour candidat est « couvert »
# s'il est à moins de cette distance d'une détection source. Sert au
# closure_ratio, à l'ancrage, à l'isolement et au rattachement des sources.
COVER_EPS_M = 3.0

# Tolérance Douglas-Peucker (part du périmètre) avant classification de forme.
_SIMPLIFY_PERIMETER_RATIO = 0.03

# join_style=2 : mitre — préserve les angles droits des enclos quadrangulaires.
_MITRE = 2

# Garde anti-réseau des générateurs hull/blob : une composante plus étendue
# qu'un grand enclos est une trame parcellaire, pas un enclos (Bretagne :
# enveloppe GT max ~720 000 m² mais étendue < 400 m hors réseaux).
SPAN_MIN_M = 10.0
SPAN_MAX_M = 400.0

# Plancher anti-slivers : les poches < 20 m² sont du bruit géométrique des
# enveloppes (Bretagne : 35 000+ slivers scorés pour rien) — éliminées AVANT
# scoring, jamais publiées, même en mode calibration.
SLIVER_MIN_M2 = 20.0

# Un blob n'est candidat que s'il est massif (aire / enveloppe convexe) :
# une bande fine ou une lanière n'est pas une « masse pleine ».
_BLOB_MIN_SOLIDITY = 0.6

# Deux candidats de générateurs différents qui se recouvrent à > 0,7 IoU
# sont le même enclos → on garde le plus prioritaire (anneau > cour > blob).
_DEDUP_IOU = 0.7

_REJECT_ORDER = ("aire", "elongation", "closure", "ancrage", "isolement", "rectangularite")


def _mrr_sides(geom) -> Tuple[float, float, Optional[object]]:
    """(côté long, côté court, rectangle orienté minimal)."""
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
    except Exception:
        return 0.0, 0.0, None
    if len(coords) < 5:
        return 0.0, 0.0, None
    a = math.dist(coords[0], coords[1])
    b = math.dist(coords[1], coords[2])
    return max(a, b), min(a, b), mrr


def _angle_concentration(ring_coords) -> float:
    """Concentration directionnelle des côtés, modulo 90° (∈ [0, 1])."""
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
    if simplified.is_empty or simplified.geom_type != "Polygon":
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


def _generate_dilation(union, half: float) -> List[Polygon]:
    """V1 : dilatation T/2 → anneaux intérieurs ré-étendus de +T/2.

    L'érosion classique du closing re-sectionnerait les ponts sur bandes
    fines — d'où la ré-extension des trous (cf. historique de la brique).
    """
    dilated = union.buffer(half, join_style=_MITRE)
    cands: List[Polygon] = []
    for poly in getattr(dilated, "geoms", [dilated]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        for ring in poly.interiors:
            cands.append(Polygon(ring).buffer(half))
    return cands


def _spanned_parts(union):
    """Composantes BRUTES de l'union, filtrées par la garde d'étendue.

    Aucune liaison préalable : la liaison T/2 de la V2 soudait des lanières
    sans rapport (enveloppes géantes — leçon Bretagne).
    """
    for part in getattr(union, "geoms", [union]):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        minx, miny, maxx, maxy = part.bounds
        if SPAN_MIN_M <= max(maxx - minx, maxy - miny) <= SPAN_MAX_M:
            yield part


def _generate_hull(union) -> List[Polygon]:
    """V2 corrigé : cour = enveloppe convexe − détection, par composante brute."""
    cands: List[Polygon] = []
    for part in _spanned_parts(union):
        pockets = part.convex_hull.difference(part)
        for pk in getattr(pockets, "geoms", [pockets]):
            if not pk.is_empty and pk.geom_type == "Polygon":
                cands.append(pk)
    return cands


def _generate_blobs(union) -> List[Polygon]:
    """V3 : la détection elle-même quand c'est une masse pleine compacte.

    Cas Bretagne : le modèle détecte souvent l'enclos en masse (aucune cour
    dans la détection) — anneaux et enveloppes sont structurellement aveugles.
    """
    cands: List[Polygon] = []
    for part in _spanned_parts(union):
        hull_area = part.convex_hull.area
        if hull_area > 0 and part.area / hull_area >= _BLOB_MIN_SOLIDITY:
            cands.append(Polygon(part.exterior))  # trous résiduels comblés
    return cands


def _dedup_candidates(cands: List[Polygon]) -> List[Polygon]:
    """Garde le premier de chaque groupe IoU > seuil (liste déjà en ordre
    de priorité anneau > cour > blob)."""
    if len(cands) < 2:
        return cands
    tree = STRtree(cands)
    kept_flags = [False] * len(cands)
    kept: List[Polygon] = []
    for i, cand in enumerate(cands):
        dup = False
        for j in tree.query(cand):
            j = int(j)
            if j == i or not kept_flags[j]:
                continue
            other = cands[j]
            inter = cand.intersection(other).area
            denom = cand.area + other.area - inter
            if denom > 0 and inter / denom > _DEDUP_IOU:
                dup = True
                break
        if not dup:
            kept_flags[i] = True
            kept.append(cand)
    return kept


def run_enclosure(
    data_by_class_name: Dict[str, List[Dict]],
    enclosure_configs: List[Dict],
    *,
    cancel_check: Optional[CancelCheckFn] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Exécute les règles « enclosure ». Même contrat que ``run_clustering`` :
    renvoie ``(enclos_by_class, données_annotées)`` — les détections sources
    contributrices reçoivent un ``enclos_id`` (traçabilité)."""
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
        max_isolement = float(cfg.get("max_isolement", 0.3))
        min_rect = float(cfg.get("min_rectangularite", 0.0))
        min_confidence = float(cfg.get("min_confidence", 0.0))
        mode_calibration = bool(cfg.get("mode_calibration", False))
        generator = str(cfg.get("generator", "auto")).strip().lower()
        output_class = cfg["output_class_name"]
        logger.info(
            f"Enclosure [{cfg_idx + 1}/{len(enclosure_configs)}]: "
            f"generator={generator}, classes={target_classes}, T={gap_m}m, "
            f"aire=[{min_area:.0f};{max_area:.0f}]m², closure>={min_closure}, "
            f"elongation<={max_elongation}, ancrage>={min_ancrage}, "
            f"isolement<={max_isolement}, rectangularite>={min_rect}"
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

        union = unary_union([s[2] for s in sources])
        stree = STRtree([s[2] for s in sources])
        half = gap_m / 2.0

        if generator == "dilation":
            raw = _generate_dilation(union, half)
        elif generator == "hull":
            raw = _generate_hull(union)
        else:  # auto — concaténation en ordre de priorité pour le dédoublonnage
            raw = (_generate_dilation(union, half) + _generate_hull(union)
                   + _generate_blobs(union))
        raw_cands = [c for c in raw if c.area >= SLIVER_MIN_M2]
        if generator not in ("dilation", "hull"):
            raw_cands = _dedup_candidates(raw_cands)
        n_rings = len(raw_cands)
        check_cancelled(cancel_check)

        # ── Métriques de TOUS les candidats (le mode calibration et les
        # filtres isolement/rectangularité en ont besoin avant le statut). ──
        metas: List[Dict] = []
        for cand in raw_cands:
            check_cancelled(cancel_check)
            if not cand.is_valid:
                cand = cand.buffer(0)
            if cand.is_empty or cand.geom_type != "Polygon":
                continue
            area = cand.area
            long_side, short_side, mrr = _mrr_sides(cand)
            elongation = (long_side / short_side) if short_side > 0 else float("inf")
            rectangularite = (area / mrr.area) if (mrr is not None and mrr.area > 0) else 0.0
            ring_line = cand.exterior
            # Contributeurs d'abord (STRtree) : seules les sources à
            # ≤ COVER_EPS_M du contour peuvent le couvrir — le closure_ratio se
            # calcule donc contre leur union LOCALE, jamais contre l'union
            # globale (l'intersection contre ~16 000 polygones par candidat
            # coûtait ~3 h sur une campagne de 201 dalles, pour un résultat
            # strictement identique).
            contrib_idx = [
                int(j) for j in stree.query(ring_line.buffer(COVER_EPS_M))
                if sources[int(j)][2].distance(ring_line) <= COVER_EPS_M
            ]
            if contrib_idx and ring_line.length > 0:
                local_cover = unary_union(
                    [sources[j][2] for j in contrib_idx]
                ).buffer(COVER_EPS_M)
                closure = ring_line.intersection(local_cover).length / ring_line.length
            else:
                closure = 0.0
            # Ancrage : part de l'aire des fragments contributeurs qui reste
            # dans le candidat (+ tolérance). Zone UNIFIÉE entre générateurs :
            # un blob isolé (la détection EST l'enclos) ≈ 1, une cour incidente
            # entre lanières qui filent au loin reste faible.
            a_tot = sum(sources[j][2].area for j in contrib_idx)
            if a_tot > 0:
                cand_zone = cand.buffer(COVER_EPS_M)
                a_in = sum(sources[j][2].intersection(cand_zone).area for j in contrib_idx)
                ancrage = a_in / a_tot
            else:
                ancrage = 0.0
            forme, compacite = _classify_shape(cand, rectangularite)
            metas.append({
                "cand": cand, "area": area, "elongation": elongation,
                "rectangularite": rectangularite, "closure": closure,
                "ancrage": ancrage, "forme": forme, "compacite": compacite,
                "contrib_idx": contrib_idx,
            })

        # Isolement : part du périmètre à ≤ COVER_EPS_M du contour des AUTRES
        # candidats (une maille de trame partage ses bords, pas un enclos).
        neighborhoods = [m["cand"].exterior.buffer(COVER_EPS_M) for m in metas]
        ntree = STRtree(neighborhoods)
        for i, m in enumerate(metas):
            ring_line = m["cand"].exterior
            others = [
                neighborhoods[int(j)] for j in ntree.query(ring_line)
                if int(j) != i and neighborhoods[int(j)].intersects(ring_line)
            ]
            if others:
                shared = ring_line.intersection(unary_union(others))
                m["isolement"] = shared.length / ring_line.length if ring_line.length > 0 else 0.0
            else:
                m["isolement"] = 0.0

        # ── Statut = premier filtre qui rejette, dans l'ordre. ──
        rejects = {k: 0 for k in _REJECT_ORDER}
        dets: List[Dict] = []
        n_publies = 0
        for m in metas:
            if m["area"] < min_area or m["area"] > max_area:
                statut = "rejete_aire"
            elif m["elongation"] > max_elongation:
                statut = "rejete_elongation"
            elif m["closure"] < min_closure:
                statut = "rejete_closure"
            elif m["ancrage"] < min_ancrage:
                statut = "rejete_ancrage"
            elif m["isolement"] > max_isolement:
                statut = "rejete_isolement"
            elif m["rectangularite"] < min_rect:
                statut = "rejete_rectangularite"
            else:
                statut = "publie"

            confs = [
                sources[j][3] for j in m["contrib_idx"]
                if sources[j][3] is not None and sources[j][3] > 0
            ]
            conf_fragments = (sum(confs) / len(confs)) if confs else 0.0
            # Confiance composite : moyenne géométrique des trois axes de
            # qualité (modèle, fermeture, appartenance).
            confidence = (
                (conf_fragments * m["closure"] * m["ancrage"]) ** (1.0 / 3.0)
                if conf_fragments > 0 else 0.0
            )

            if statut != "publie":
                rejects[statut.removeprefix("rejete_")] += 1
                if not mode_calibration:
                    continue
                enclos_id = ""
                model_name = ""
            else:
                enclos_id = f"{output_class}_{n_publies}"
                n_publies += 1
                model_name = ""
                for j in m["contrib_idx"]:
                    class_name, det_idx = sources[j][0], sources[j][1]
                    updated[class_name][det_idx]["enclos_id"] = enclos_id
                    if not model_name:
                        model_name = updated[class_name][det_idx].get("model_name", "")

            dets.append({
                "validation": "",
                "corr_pred": None,
                "model_pred": output_class,
                "model_name": model_name,
                "geometry": m["cand"],
                "confidence": confidence,
                "conf_fragments": round(conf_fragments, 3),
                "surface_m2": round(m["area"], 1),
                "closure_ratio": round(m["closure"], 3),
                "ancrage": round(m["ancrage"], 3),
                "isolement": round(m["isolement"], 3),
                "rectangularite": round(m["rectangularite"], 3),
                "compacite": round(m["compacite"], 3),
                "elongation": round(m["elongation"], 2),
                "forme": m["forme"],
                "nb_sources": len(m["contrib_idx"]),
                "enclos_id": enclos_id,
                "statut": statut,
            })

        n_rejetes = len(dets) - n_publies
        logger.info(
            f"Enclosure: {n_rings} surface(s) candidate(s), "
            f"{n_publies} publiée(s) après filtres durs "
            f"(rejets: aire {rejects['aire']}, élongation {rejects['elongation']}, "
            f"closure {rejects['closure']}, ancrage {rejects['ancrage']}, "
            f"isolement {rejects['isolement']}, rectangularité {rejects['rectangularite']})"
        )
        if mode_calibration and n_rejetes:
            logger.info(
                f"Enclosure: mode calibration — {n_rejetes} candidat(s) "
                f"rejeté(s) publié(s) avec leur statut"
            )
        if dets:
            out_by_class[output_class] = dets
            logger.info(f"Enclosure: {len(dets)} enclos '{output_class}' publiés")

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in out_by_class.values())
    logger.info(f"Enclosure terminé: {total} enclos en {elapsed:.2f}s")
    return out_by_class, updated
