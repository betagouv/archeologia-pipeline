"""Brique de synthèse « alignment » : axes linéaires par bandes directionnelles.

Détecte les enfilades de détections co-orientées — le signal « spaghettis
alignés » d'une voie ancienne : plusieurs brins parallèles (fossés bordiers,
agger, tronçons décalés) dans une bande étroite, continue sur des centaines de
mètres. Spec : ``.claude/plans/2026-07-23-brique-axe-lineaire.md``.

Pipeline : axe de chaque fragment (``minimum_rotated_rectangle``) → familles
d'orientation (glouton pondéré par longueur, azimut modulo 180°) → rotation du
repère par famille → bandes par chaînage latéral → chaînage longitudinal coupé
aux trous > ``max_gap_m`` → filtres durs (longueur, couverture, nb) → scores
en attributs (l'archéologue tranche).

Module pur (shapely + stdlib, ni scipy ni Qt) — même contrat que
``clustering.run_clustering`` / ``enclosure.run_enclosure``.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon

from ..cancellation import check_cancelled
from ..types import CancelCheckFn

logger = logging.getLogger(__name__)

# Constantes internes (non exposées — dérivées de la sémantique de bande).
BRIN_LINK_M = 5.0        # liaison latérale : fragments d'un même brin
BAND_LINK_M = 20.0       # liaison latérale : brins d'une même bande
PERP_TOL_DEG = 20.0      # tolérance des connecteurs perpendiculaires
NEIGHBOR_RADIUS_M = 500.0  # rayon du « grain local » (discordance)
PARALLEL_MIN_M = 50.0    # bande voisine comptée dans parallelisme : min…
PARALLEL_MAX_M = 300.0   # …et max de distance latérale
CORRIDOR_MARGIN_M = 5.0  # marge du polygone corridor


def _circ_dist(a: float, b: float) -> float:
    """Distance angulaire circulaire modulo 180° (axes non orientés)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _axis_of(geom) -> Optional[Tuple[float, float]]:
    """(azimut mod 180°, longueur) du grand axe du rectangle orienté minimal."""
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
    except Exception:
        return None
    if len(coords) < 5:
        return None
    e1 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    e2 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    vx, vy, length = (e1[0], e1[1], l1) if l1 >= l2 else (e2[0], e2[1], l2)
    if length <= 0:
        return None
    return math.degrees(math.atan2(vy, vx)) % 180.0, length


def _exterior_coords(geom) -> Sequence[Tuple[float, float]]:
    if geom.geom_type == "Polygon":
        return list(geom.exterior.coords)
    return list(geom.convex_hull.exterior.coords)


def _union_length(intervals: List[Tuple[float, float]]) -> float:
    """Longueur de l'union d'intervalles 1D (les brins superposés ne comptent
    pas double dans la couverture)."""
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_b is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _chain_1d(values: List[float], link: float) -> List[List[int]]:
    """Groupes d'indices par chaînage 1D (écart au précédent ≤ link)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: List[List[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][-1]] <= link:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def run_alignment(
    data_by_class_name: Dict[str, List[Dict]],
    alignment_configs: List[Dict],
    *,
    cancel_check: Optional[CancelCheckFn] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Exécute les règles « alignment ». Même contrat que ``run_clustering`` :
    renvoie ``(axes_par_classe, données_annotées)`` — les fragments membres
    reçoivent un ``axe_id`` (traçabilité, symétrique de ``cluster_id``)."""
    t0 = time.perf_counter()
    out_by_class: Dict[str, List[Dict]] = {}
    updated = {k: list(v) for k, v in data_by_class_name.items()}

    for cfg_idx, cfg in enumerate(alignment_configs):
        check_cancelled(cancel_check)
        target_classes = cfg["target_classes"]
        band_width = float(cfg["band_width_m"])
        angle_tol = float(cfg["angle_tolerance_deg"])
        min_length = float(cfg["min_length_m"])
        max_gap = float(cfg["max_gap_m"])
        min_coverage = float(cfg["min_coverage"])
        min_sources = int(cfg["min_sources"])
        min_conf = float(cfg.get("min_confidence", 0.0))
        output_class = cfg["output_class_name"]
        logger.info(
            f"Alignment [{cfg_idx + 1}/{len(alignment_configs)}]: "
            f"classes={target_classes}, bande={band_width}m, ±{angle_tol}°, "
            f"L>={min_length}m, gap<={max_gap}m"
        )

        # Collecte : (class_name, det_idx, geom, conf, azimut, longueur d'axe)
        sources = []
        for class_name in target_classes:
            for det_idx, det in enumerate(updated.get(class_name, [])):
                geom = det.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                conf = det.get("confidence")
                if conf is not None and conf < min_conf:
                    continue
                axis = _axis_of(geom)
                if axis is None:
                    continue
                sources.append((class_name, det_idx, geom, conf, axis[0], axis[1]))
        if len(sources) < min_sources:
            logger.info("Alignment: sources insuffisantes, règle ignorée")
            continue

        # Familles d'orientation — glouton déterministe pondéré par longueur.
        # ponytail: un fragment à ±α de deux graines va à la première (la plus
        # longue) ; affinage par réassignation si les cas réels l'exigent.
        order = sorted(range(len(sources)), key=lambda i: -sources[i][5])
        assigned = [False] * len(sources)
        families: List[Tuple[float, List[int]]] = []
        for seed in order:
            if assigned[seed]:
                continue
            theta_seed = sources[seed][4]
            members = [
                i for i in range(len(sources))
                if not assigned[i] and _circ_dist(sources[i][4], theta_seed) <= angle_tol
            ]
            for i in members:
                assigned[i] = True
            # Direction = moyenne circulaire pondérée (angles doublés, mod 180°).
            sx = sum(sources[i][5] * math.cos(math.radians(2 * sources[i][4])) for i in members)
            sy = sum(sources[i][5] * math.sin(math.radians(2 * sources[i][4])) for i in members)
            theta = (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0 if (sx or sy) else theta_seed
            families.append((theta, members))

        # Chaînes candidates (toutes, AVANT filtres durs — parallelisme doit
        # voir les corridors coaxiaux même s'ils échouent individuellement).
        # Chaque chaîne : dict avec fam (index famille), membres, bornes x/y…
        chains: List[Dict] = []
        for fam_idx, (theta, members) in enumerate(families):
            check_cancelled(cancel_check)
            u = (math.cos(math.radians(theta)), math.sin(math.radians(theta)))
            nrm = (-u[1], u[0])
            proj = []  # (idx source, x1, x2, y_centre, y_min, y_max)
            for i in members:
                coords = _exterior_coords(sources[i][2])
                xs = [c[0] * u[0] + c[1] * u[1] for c in coords]
                ys = [c[0] * nrm[0] + c[1] * nrm[1] for c in coords]
                proj.append((i, min(xs), max(xs), (min(ys) + max(ys)) / 2.0, min(ys), max(ys)))
            # Bandes : chaînage latéral (liaison BAND_LINK_M, étalement ≤ bande).
            proj.sort(key=lambda p: p[3])
            bands: List[List[tuple]] = []
            for p in proj:
                if bands and (p[3] - bands[-1][-1][3] <= BAND_LINK_M
                              and p[3] - bands[-1][0][3] <= band_width):
                    bands[-1].append(p)
                else:
                    bands.append([p])
            # Chaînage longitudinal dans chaque bande.
            for band in bands:
                band.sort(key=lambda p: p[1])
                cur: List[tuple] = []
                covered_until = None
                for p in band:
                    if cur and p[1] > covered_until + max_gap:
                        chains.append({"fam": fam_idx, "theta": theta, "u": u,
                                       "nrm": nrm, "items": cur})
                        cur, covered_until = [], None
                    cur.append(p)
                    covered_until = p[2] if covered_until is None else max(covered_until, p[2])
                if cur:
                    chains.append({"fam": fam_idx, "theta": theta, "u": u,
                                   "nrm": nrm, "items": cur})

        # Mesures par chaîne.
        for c in chains:
            items = c["items"]
            c["x0"] = min(p[1] for p in items)
            c["x1"] = max(p[2] for p in items)
            c["span"] = c["x1"] - c["x0"]
            c["coverage"] = (_union_length([(p[1], p[2]) for p in items]) / c["span"]
                             if c["span"] > 0 else 0.0)
            c["y_center"] = sum(p[3] for p in items) / len(items)
            c["y_lo"] = min(p[4] for p in items)
            c["y_hi"] = max(p[5] for p in items)

        published = [
            c for c in chains
            if c["span"] >= min_length and c["coverage"] >= min_coverage
            and len(c["items"]) >= min_sources
        ]
        logger.info(
            f"Alignment: {len(chains)} chaîne(s) candidate(s), "
            f"{len(published)} publiée(s) après filtres durs"
        )

        dets: List[Dict] = []
        for i, c in enumerate(published):
            check_cancelled(cancel_check)
            items = c["items"]
            member_idx = {p[0] for p in items}
            u, nrm, theta = c["u"], c["nrm"], c["theta"]

            # Corridor : rectangle orienté (re-transformé, base orthonormée).
            x0, x1 = c["x0"] - CORRIDOR_MARGIN_M, c["x1"] + CORRIDOR_MARGIN_M
            y0, y1 = c["y_lo"] - CORRIDOR_MARGIN_M, c["y_hi"] + CORRIDOR_MARGIN_M
            corners = [
                (x0 * u[0] + y0 * nrm[0], x0 * u[1] + y0 * nrm[1]),
                (x1 * u[0] + y0 * nrm[0], x1 * u[1] + y0 * nrm[1]),
                (x1 * u[0] + y1 * nrm[0], x1 * u[1] + y1 * nrm[1]),
                (x0 * u[0] + y1 * nrm[0], x0 * u[1] + y1 * nrm[1]),
            ]
            corridor = Polygon(corners)

            # Brins : re-chaînage fin des positions latérales des membres.
            y_vals = [p[3] for p in items]
            brins = _chain_1d(y_vals, BRIN_LINK_M)
            brin_centers = sorted(
                sum(y_vals[j] for j in g) / len(g) for g in brins
            )
            gaps = [b - a for a, b in zip(brin_centers[:-1], brin_centers[1:])]
            gaps.sort()
            espacement = gaps[len(gaps) // 2] if gaps else 0.0

            # Parallélisme : autres chaînes candidates de la même famille à
            # distance latérale [50;300] m avec recouvrement longitudinal.
            parallelisme = sum(
                1 for o in chains
                if o is not c and o["fam"] == c["fam"]
                and PARALLEL_MIN_M <= abs(o["y_center"] - c["y_center"]) <= PARALLEL_MAX_M
                and min(o["x1"], c["x1"]) > max(o["x0"], c["x0"])
            )

            # Connecteurs ⊥ et discordance du grain local (sources non membres).
            perp_count = 0
            neighbor_diffs: List[float] = []
            for j, (_cls, _di, geom, _conf, az, _l) in enumerate(sources):
                if j in member_idx:
                    continue
                if (_circ_dist(az, theta + 90.0) <= PERP_TOL_DEG
                        and geom.intersects(corridor)):
                    perp_count += 1
                if geom.distance(corridor) <= NEIGHBOR_RADIUS_M:
                    neighbor_diffs.append(_circ_dist(az, theta))
            neighbor_diffs.sort()
            # Aucun grain local = isolé → discordance maximale (documenté).
            discordance = (neighbor_diffs[len(neighbor_diffs) // 2]
                           if neighbor_diffs else 90.0)
            connecteurs_perp = perp_count / (c["span"] / 1000.0) if c["span"] > 0 else 0.0

            axe_id = f"{output_class}_{i}"
            confs: List[float] = []
            model_name = ""
            for p in items:
                class_name, det_idx = sources[p[0]][0], sources[p[0]][1]
                updated[class_name][det_idx]["axe_id"] = axe_id
                if not model_name:
                    model_name = updated[class_name][det_idx].get("model_name", "")
                conf = sources[p[0]][3]
                if conf is not None and conf > 0:
                    confs.append(conf)

            dets.append({
                "validation": "",
                "corr_pred": None,
                "model_pred": output_class,
                "model_name": model_name,
                "geometry": corridor,
                "confidence": (sum(confs) / len(confs)) if confs else 0.0,
                "longueur_m": round(c["span"], 1),
                "couverture": round(c["coverage"], 3),
                "largeur_m": round(c["y_hi"] - c["y_lo"], 1),
                "azimut_deg": round(theta, 1),
                "nb_brins": len(brins),
                "espacement_brins_m": round(espacement, 1),
                "parallelisme": parallelisme,
                "connecteurs_perp": round(connecteurs_perp, 2),
                "discordance_deg": round(discordance, 1),
                "nb_sources": len(items),
                "axe_id": axe_id,
            })

        if dets:
            out_by_class[output_class] = dets
            logger.info(f"Alignment: {len(dets)} axe(s) '{output_class}' publiés")

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in out_by_class.values())
    logger.info(f"Alignment terminé: {total} axe(s) en {elapsed:.2f}s")
    return out_by_class, updated
