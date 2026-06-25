"""F6 / F10 (audit « rassemblement de polygones », 2026-06-12).

Le noyau géométrique de la fusion intra-classe n'avait **aucun test** (F10).
La distance de fusion effective valait ``2×merge_buffer_m`` : les DEUX polygones
étaient bufferisés de ``merge_buffer_m`` puis on testait leur intersection, donc
deux polygones se connectaient dès que leur écart ≤ ``2×merge_buffer_m`` (1,0 m
pour 0,5 m) — alors que la docstring/README annoncent ``merge_buffer_m`` (0,5 m)
(F6). Ces tests épinglent la **vraie distance** : connexion si gap ≤
``merge_buffer_m``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__ tire shapely

from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from pipeline.cv.postprocessing import (  # noqa: E402
    _connected_components_via_strtree,
    _resolve_same_class_overlaps,
    postprocess_geo_detections,
)


def _square(x0: float, y0: float, side: float = 10.0) -> Polygon:
    return Polygon(
        [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    )


def _circle(cx: float, cy: float, r: float) -> Polygon:
    """Disque (les cratères sont circulaires) — aire ≈ π·r²."""
    return Point(cx, cy).buffer(r)


def _det(geom, conf: float, cls: str = "cratere") -> dict:
    return {"geometry": geom, "confidence": conf, "model_pred": cls}


def _two_squares_gap(gap: float):
    """Deux carrés 10×10 m alignés, séparés horizontalement de ``gap`` mètres."""
    return _square(0.0, 0.0), _square(10.0 + gap, 0.0)


def _dets(*polys):
    return [{"geometry": p, "confidence": 0.9} for p in polys]


class TestMergeDistanceIsTrueDistance:
    """Fusion intra-classe = vraie distance ``merge_buffer_m``, pas ``2×``."""

    def test_gap_below_buffer_merges(self):
        a, b = _two_squares_gap(0.4)  # 0,4 m ≤ 0,5 m → fusionnés
        out = postprocess_geo_detections(
            {"parcellaire": _dets(a, b)},
            merge_buffer_m=0.5,
            do_merge=True,
            do_remove_overlaps=False,
        )
        assert len(out["parcellaire"]) == 1

    def test_gap_above_buffer_does_not_merge(self):
        a, b = _two_squares_gap(0.6)  # 0,6 m > 0,5 m → restent distincts
        out = postprocess_geo_detections(
            {"parcellaire": _dets(a, b)},
            merge_buffer_m=0.5,
            do_merge=True,
            do_remove_overlaps=False,
        )
        assert len(out["parcellaire"]) == 2

    def test_gap_exactly_buffer_merges(self):
        # ``dwithin`` est inclusif (≤) : un écart EXACTEMENT égal à
        # merge_buffer_m doit fusionner (borne inférieure du non-merge).
        a, b = _two_squares_gap(0.5)
        out = postprocess_geo_detections(
            {"parcellaire": _dets(a, b)},
            merge_buffer_m=0.5,
            do_merge=True,
            do_remove_overlaps=False,
        )
        assert len(out["parcellaire"]) == 1

    def test_touching_polygons_merge(self):
        a, b = _two_squares_gap(0.0)  # se touchent → fusionnés
        out = postprocess_geo_detections(
            {"parcellaire": _dets(a, b)},
            merge_buffer_m=0.5,
            do_merge=True,
            do_remove_overlaps=False,
        )
        assert len(out["parcellaire"]) == 1


class TestConnectedComponentsTrueDistance:
    """Locus exact du bug : ``_connected_components_via_strtree``."""

    def test_gap_within_buffer_is_one_component(self):
        a, b = _two_squares_gap(0.4)
        comps = _connected_components_via_strtree([a, b], 0.5, STRtree)
        assert len(comps) == 1

    def test_gap_beyond_buffer_is_two_components(self):
        a, b = _two_squares_gap(0.6)
        comps = _connected_components_via_strtree([a, b], 0.5, STRtree)
        assert len(comps) == 2


# ======================================================================
# Audit chevauchement (2026-06-18) : résolution par RELATION (IoS).
#
# `remove_overlaps` historique DÉCOUPE (geom.difference) le polygone le moins
# confiant -> fabrique soit un anneau troué (petit imbriqué dans un grand),
# soit une arête droite partagée (deux polygones accolés). Le correctif retenu
# raisonne en confinement : on FUSIONNE par union les polygones de même classe
# dont l'IoS = inter/min_aire >= seuil. L'union absorbe le petit dans le grand
# (pas d'anneau) et soude deux fragments fortement chevauchants (pas d'arête) ;
# les polygones disjoints / faiblement chevauchants restent intacts.
# ======================================================================
class TestResolveSameClassOverlapsByIoS:
    """Fonction pure `_resolve_same_class_overlaps` (une seule classe)."""

    def test_small_fully_inside_large_keeps_large_no_donut(self):
        # Petit cratère PLUS confiant entièrement dans un grand : on garde le
        # GRAND (union), sans trou — peu importe la confiance (géométrie d'abord).
        big = _det(_circle(0, 0, 10), conf=0.5)
        small = _det(_circle(0, 0, 2), conf=0.9)
        out = _resolve_same_class_overlaps([big, small], 0.5, unary_union, STRtree)
        assert len(out) == 1
        geom = out[0]["geometry"]
        assert geom.area > 300  # ≈ π·100, le grand disque
        assert len(list(geom.interiors)) == 0  # pas d'anneau

    def test_containment_large_more_confident_also_keeps_large(self):
        big = _det(_circle(0, 0, 10), conf=0.9)
        small = _det(_circle(0, 0, 2), conf=0.5)
        out = _resolve_same_class_overlaps([big, small], 0.5, unary_union, STRtree)
        assert len(out) == 1
        assert out[0]["geometry"].area > 300

    def test_strong_partial_overlap_merges_by_union(self):
        # Deux disques r=5, centres distants de 3 : IoS ≈ 0,62 ≥ 0,5 -> union.
        a = _det(_circle(0, 0, 5), conf=0.8)
        b = _det(_circle(3, 0, 5), conf=0.6)
        out = _resolve_same_class_overlaps([a, b], 0.5, unary_union, STRtree)
        assert len(out) == 1
        # Union (≈108) : > un seul disque (≈78,5) et < la somme (≈157).
        assert 90 < out[0]["geometry"].area < 150

    def test_weak_overlap_keeps_both(self):
        # Disques r=5, centres distants de 9,5 : IoS ≈ 0,01 < 0,5 -> intacts.
        a = _det(_circle(0, 0, 5), conf=0.8)
        b = _det(_circle(9.5, 0, 5), conf=0.6)
        out = _resolve_same_class_overlaps([a, b], 0.5, unary_union, STRtree)
        assert len(out) == 2

    def test_disjoint_keeps_both(self):
        a = _det(_circle(0, 0, 5), conf=0.8)
        b = _det(_circle(50, 0, 5), conf=0.6)
        out = _resolve_same_class_overlaps([a, b], 0.5, unary_union, STRtree)
        assert len(out) == 2


class TestPostprocessRelationStrategy:
    """Intégration via `postprocess_geo_detections(overlap_strategy='relation')`."""

    def _run(self, by_class):
        return postprocess_geo_detections(
            by_class,
            do_merge=False,
            do_remove_overlaps=True,
            overlap_strategy="relation",
            overlap_ios_threshold=0.5,
        )

    def test_nested_same_class_resolved_to_one_polygon(self):
        out = self._run(
            {"cratere": [_det(_circle(0, 0, 10), 0.5), _det(_circle(0, 0, 2), 0.9)]}
        )
        assert len(out["cratere"]) == 1
        assert len(list(out["cratere"][0]["geometry"].interiors)) == 0

    def test_strong_overlap_same_class_merged(self):
        out = self._run(
            {"cratere": [_det(_circle(0, 0, 5), 0.8), _det(_circle(3, 0, 5), 0.6)]}
        )
        assert len(out["cratere"]) == 1

    def test_weak_overlap_same_class_both_kept(self):
        out = self._run(
            {"cratere": [_det(_circle(0, 0, 5), 0.8), _det(_circle(9.5, 0, 5), 0.6)]}
        )
        assert len(out["cratere"]) == 2

    def test_cross_class_overlap_still_clipped(self):
        # Deux classes DIFFÉRENTES qui se chevauchent : le moins confiant est
        # rogné (difference) — comportement multi-classes conservé.
        c2 = _circle(3, 0, 5)
        out = self._run(
            {
                "cratere": [_det(_circle(0, 0, 5), 0.9, "cratere")],
                "tumulus": [_det(c2, 0.5, "tumulus")],
            }
        )
        assert len(out["cratere"]) == 1
        assert len(out["tumulus"]) == 1
        assert out["tumulus"][0]["geometry"].area < c2.area  # rogné


class TestResolveSameClassSizeGate:
    """Garde-fou de similarité de taille (``min_area_ratio``) : sur la bande de
    chevauchement modéré, ne fusionner que des polygones de taille PROCHE
    (vrais doublons), pas un petit cratère distinct posé sur un grand. Le
    confinement quasi-total (IoS≈1) fusionne toujours malgré le garde-fou."""

    def test_similar_size_moderate_overlap_merges_when_gate_on(self):
        # Deux disques égaux r=5 (centres distants de 5,5) : IoS ≈ 0,34, ratio 1.
        a = _det(_circle(0, 0, 5), 0.8)
        b = _det(_circle(5.5, 0, 5), 0.6)
        out = _resolve_same_class_overlaps(
            [a, b], 0.3, unary_union, STRtree, min_area_ratio=0.7
        )
        assert len(out) == 1

    def test_dissimilar_size_moderate_overlap_kept_when_gate_on(self):
        # Petit (r=3) sur le bord d'un grand (r=10) : IoS ≈ 0,36 mais ratio
        # d'aire ≈ 0,09 < 0,7 -> NON fusionnés (cratère distinct préservé).
        big = _det(_circle(0, 0, 10), 0.8)
        small = _det(_circle(10.5, 0, 3), 0.6)
        out = _resolve_same_class_overlaps(
            [big, small], 0.3, unary_union, STRtree, min_area_ratio=0.7
        )
        assert len(out) == 2

    def test_dissimilar_moderate_overlap_merges_when_gate_off(self):
        # Contrôle : sans garde-fou (ratio=0), la même paire fusionne (IoS≥0,3).
        big = _det(_circle(0, 0, 10), 0.8)
        small = _det(_circle(10.5, 0, 3), 0.6)
        out = _resolve_same_class_overlaps(
            [big, small], 0.3, unary_union, STRtree, min_area_ratio=0.0
        )
        assert len(out) == 1

    def test_containment_still_merges_despite_gate(self):
        # L'imbrication quasi-totale (IoS≈1) fusionne malgré le garde-fou de taille.
        big = _det(_circle(0, 0, 10), 0.5)
        small = _det(_circle(0, 0, 2), 0.9)
        out = _resolve_same_class_overlaps(
            [big, small], 0.3, unary_union, STRtree, min_area_ratio=0.7
        )
        assert len(out) == 1
        assert out[0]["geometry"].area > 300
        assert len(list(out[0]["geometry"].interiors)) == 0


class TestPostprocessRelationSizeGate:
    """Intégration du garde-fou via postprocess_geo_detections."""

    def _run(self, by_class):
        return postprocess_geo_detections(
            by_class,
            do_merge=False,
            do_remove_overlaps=True,
            overlap_strategy="relation",
            overlap_ios_threshold=0.3,
            overlap_min_area_ratio=0.7,
        )

    def test_similar_moderate_merged(self):
        out = self._run(
            {"cratere": [_det(_circle(0, 0, 5), 0.8), _det(_circle(5.5, 0, 5), 0.6)]}
        )
        assert len(out["cratere"]) == 1

    def test_dissimilar_moderate_kept(self):
        out = self._run(
            {"cratere": [_det(_circle(0, 0, 10), 0.8), _det(_circle(10.5, 0, 3), 0.6)]}
        )
        assert len(out["cratere"]) == 2


class TestPostprocessDifferenceStrategyUnchanged:
    """Garde-fou : la stratégie 'difference' (défaut) reste le découpage legacy."""

    def test_default_strategy_still_clips_same_class(self):
        c2 = _circle(3, 0, 5)
        out = postprocess_geo_detections(
            {"cratere": [_det(_circle(0, 0, 5), 0.9), _det(c2, 0.5)]},
            do_merge=False,
            do_remove_overlaps=True,
        )
        # Legacy : les deux survivent, le moins confiant est rogné (pas fusionné).
        assert len(out["cratere"]) == 2
        areas = sorted(d["geometry"].area for d in out["cratere"])
        assert areas[0] < c2.area  # le moins confiant a été découpé
