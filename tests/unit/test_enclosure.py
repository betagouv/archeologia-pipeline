"""Moteur ``run_enclosure`` (brique enclosure) — fermeture vectorielle + scoring.

Fixtures géométriques pures (Lambert-93 fictif, mètres) : les côtés d'un
enclos sont des tronçons de fossé simulés par des lignes bufferisées (~2 m de
large), avec des trous contrôlés. On vérifie la fermeture (T ponte les trous
≤ T, pas au-delà), les filtres durs (aire, élongation, closure), les scores
(closure_ratio, isolement, forme) et la traçabilité (enclos_id).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("shapely")

from shapely.geometry import LineString, Point

from pipeline.cv.enclosure import run_enclosure


# ----------------------------------------------------------------------
# Fixtures géométriques
# ----------------------------------------------------------------------
def _side(x1, y1, x2, y2):
    """Tronçon de fossé : ligne bufferisée (~2 m de large)."""
    return LineString([(x1, y1), (x2, y2)]).buffer(1.0)


def _det(geom, conf=0.5):
    return {"geometry": geom, "confidence": conf,
            "model_pred": "parcellaire", "model_name": "formes"}


def _cfg(**over):
    cfg = {"type": "enclosure", "target_classes": ["parcellaire"],
           "output_class_name": "enclos", "gap_tolerance_m": 8.0,
           "min_area_m2": 50.0, "max_area_m2": 60000.0,
           "min_closure": 0.6, "max_elongation": 3.0, "min_ancrage": 0.5,
           "min_confidence": 0.0}
    cfg.update(over)
    return cfg


def _rect_fragments(cx, cy, w, h, gaps=()):
    """Côtés d'un rectangle avec trous.

    ``gaps`` : liste de ``(side_index 0-3, start_m, gap_len_m)`` — côtés dans
    l'ordre bas, droite, haut, gauche (sens trigonométrique inverse depuis le
    coin bas-gauche).
    """
    hw, hh = w / 2.0, h / 2.0
    corners = [(cx - hw, cy - hh), (cx + hw, cy - hh),
               (cx + hw, cy + hh), (cx - hw, cy + hh)]
    sides = [(corners[0], corners[1]), (corners[1], corners[2]),
             (corners[2], corners[3]), (corners[3], corners[0])]
    frags = []
    for idx, ((x1, y1), (x2, y2)) in enumerate(sides):
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        side_gaps = sorted((s, s + g) for i, s, g in gaps if i == idx)
        pos, intervals = 0.0, []
        for gs, ge in side_gaps:
            if gs > pos:
                intervals.append((pos, gs))
            pos = max(pos, ge)
        if pos < length:
            intervals.append((pos, length))
        for a, b in intervals:
            if b - a < 0.5:
                continue
            frags.append(_side(x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b))
    return frags


def _run(frags, cfg=None):
    data = {"parcellaire": [_det(g) for g in frags]}
    return run_enclosure(data, [cfg or _cfg()])


# ----------------------------------------------------------------------
# Fermeture
# ----------------------------------------------------------------------
class TestClosing:
    def test_fragmented_square_detected(self):
        frags = _rect_fragments(0, 0, 40, 40,
                                gaps=[(0, 18, 4), (1, 10, 6), (2, 20, 3)])
        out, updated = _run(frags)
        assert "enclos" in out and len(out["enclos"]) == 1
        d = out["enclos"][0]
        assert 900 < d["surface_m2"] < 1700
        assert d["closure_ratio"] > 0.8
        assert d["forme"] == "quadrangulaire"
        assert d["nb_sources"] == len(frags)
        assert d["enclos_id"] == "enclos_0"
        assert all(det.get("enclos_id") == "enclos_0"
                   for det in updated["parcellaire"])

    def test_gap_wider_than_tolerance_stays_open(self):
        frags = _rect_fragments(0, 0, 40, 40, gaps=[(0, 14, 12)])
        out, _ = _run(frags)  # trou 12 m > T=8 → circuit ouvert
        assert out == {}

    def test_sparse_fragments_no_candidate(self):
        out, _ = _run([_side(0, 0, 20, 0), _side(200, 200, 220, 200)])
        assert out == {}

    def test_no_sources_no_candidate(self):
        out, _ = run_enclosure({"autre": [_det(_side(0, 0, 20, 0))]}, [_cfg()])
        assert out == {}


# ----------------------------------------------------------------------
# Filtres durs
# ----------------------------------------------------------------------
class TestHardFilters:
    def test_corridor_rejected_by_elongation(self):
        frags = _rect_fragments(0, 0, 100, 20)  # couloir fermé 100×20 (élong. ~5)
        out, _ = _run(frags)
        assert out == {}

    def test_min_closure_accepts_then_rejects(self):
        # Carré 100 m, 2 trous de 20 m par côté (T=24 les ponte tous) :
        # closure ≈ 1 − 8×(20−2·eps)/400 ≈ 0,72.
        gaps = [(i, s, 20.0) for i in range(4) for s in (10.0, 55.0)]
        frags = _rect_fragments(0, 0, 100, 100, gaps=gaps)
        out_ok, _ = _run(frags, _cfg(gap_tolerance_m=24.0, min_closure=0.6))
        assert len(out_ok.get("enclos", [])) == 1
        assert 0.6 < out_ok["enclos"][0]["closure_ratio"] < 0.85
        out_ko, _ = _run(frags, _cfg(gap_tolerance_m=24.0, min_closure=0.85))
        assert out_ko == {}

    def test_min_and_max_area(self):
        frags = _rect_fragments(0, 0, 40, 40)  # intérieur ≈ 1 400 m²
        out_min, _ = _run(frags, _cfg(min_area_m2=2000.0))
        assert out_min == {}
        out_max, _ = _run(frags, _cfg(max_area_m2=1000.0))
        assert out_max == {}

    def test_inter_strip_courtyard_rejected_by_ancrage(self):
        # Le faux positif structurel constaté en Bretagne : une « cour »
        # incidente scellée ENTRE deux lanières de parcellaire très longues
        # (+ deux bouchons courts). Les sources débordent loin de l'anneau
        # → ancrage ≪ 0,5 → rejeté. Un vrai enclos (l'anneau EST la
        # détection) garde un ancrage ≈ 1.
        frags = [
            _side(-500, 0, 500, 0),      # lanière sud (1 km)
            _side(-500, 30, 500, 30),    # lanière nord
            _side(0, 0, 0, 30),          # bouchon ouest
            _side(40, 0, 40, 30),        # bouchon est → cour ~40×30
        ]
        out, _ = _run(frags)
        assert out == {}
        # sans le filtre d'ancrage, la cour serait publiée (contrôle)
        out_off, _ = _run(frags, _cfg(min_ancrage=0.0))
        assert len(out_off.get("enclos", [])) == 1
        assert out_off["enclos"][0]["ancrage"] < 0.3

    def test_true_ring_has_high_ancrage(self):
        frags = _rect_fragments(0, 0, 40, 40, gaps=[(0, 18, 4)])
        out, _ = _run(frags)
        assert len(out.get("enclos", [])) == 1
        assert out["enclos"][0]["ancrage"] > 0.8


# ----------------------------------------------------------------------
# Scores et cas archéologiques
# ----------------------------------------------------------------------
class TestScores:
    def test_nested_enclosures_both_published(self):
        frags = _rect_fragments(0, 0, 80, 80) + _rect_fragments(0, 0, 30, 30)
        out, _ = _run(frags)
        assert len(out.get("enclos", [])) == 2
        surfaces = sorted(d["surface_m2"] for d in out["enclos"])
        assert surfaces[0] < 1000 < 4000 < surfaces[1]

    def test_isolation_grid_vs_isolated(self):
        grid = _rect_fragments(0, 0, 80, 80) + [
            _side(-40, 0, 40, 0), _side(0, -40, 0, 40)]  # croix centrale → 4 mailles
        lone = _rect_fragments(300, 0, 40, 40)
        # min_ancrage=0 : on teste ici le score d'ISOLEMENT — les mailles de
        # grille ont un ancrage limite (~0,5) par construction de la fixture.
        out, _ = _run(grid + lone, _cfg(min_ancrage=0.0))
        dets = out.get("enclos", [])
        assert len(dets) == 5
        lone_dets = [d for d in dets if d["geometry"].centroid.x > 200]
        grid_dets = [d for d in dets if d["geometry"].centroid.x <= 200]
        assert len(lone_dets) == 1
        assert lone_dets[0]["isolement"] < 0.05
        assert all(d["isolement"] > 0.3 for d in grid_dets)

    def test_circular_enclosure_forme_curviligne(self):
        band = Point(0, 0).buffer(20.0).exterior.buffer(1.0)
        out, _ = _run([band])
        assert len(out.get("enclos", [])) == 1
        d = out["enclos"][0]
        assert d["forme"] == "curviligne"
        assert d["rectangularite"] < 0.9
        assert 900 < d["surface_m2"] < 1300  # r≈19 → ~1134 m²

    def test_confidence_composite_geometric_mean(self):
        # confiance = ∛(conf_fragments × closure × ancrage) — les trois axes
        # de qualité, aucun ne pouvant être masqué par les autres. La moyenne
        # brute des fragments reste dans conf_fragments.
        frags = _rect_fragments(0, 0, 40, 40)
        data = {"parcellaire": [_det(g, conf=0.2 if i % 2 else 0.6)
                                for i, g in enumerate(frags)]}
        out, _ = run_enclosure(data, [_cfg()])
        d = out["enclos"][0]
        assert 0.2 < d["conf_fragments"] < 0.6
        attendu = (d["conf_fragments"] * d["closure_ratio"] * d["ancrage"]) ** (1 / 3)
        assert d["confidence"] == pytest.approx(attendu, abs=0.02)

    def test_far_fragment_not_tagged(self):
        frags = _rect_fragments(0, 0, 40, 40)
        far = _det(_side(300, 300, 320, 300))
        data = {"parcellaire": [_det(g) for g in frags] + [far]}
        out, updated = run_enclosure(data, [_cfg()])
        d = out["enclos"][0]
        assert d["nb_sources"] == len(frags)
        assert "enclos_id" not in updated["parcellaire"][-1]

    def test_min_confidence_filters_sources(self):
        frags = _rect_fragments(0, 0, 40, 40, gaps=[(0, 18, 4)])
        # Un côté entier sous le seuil de confiance → circuit ouvert (le côté
        # gauche est le fragment d'index -1 dans _rect_fragments).
        data = {"parcellaire": [_det(g, conf=0.5) for g in frags[:-1]]
                + [_det(frags[-1], conf=0.05)]}
        out, _ = run_enclosure(data, [_cfg(min_confidence=0.3)])
        assert out == {}


class TestDiagnostics:
    def test_rejection_counts_logged_per_filter(self, caplog):
        import logging
        # cour inter-lanières : rejetée par l'ancrage → le log doit dire par
        # QUEL filtre les candidats meurent (diagnostic Bretagne : « 69
        # surfaces, 0 candidat » sans explication).
        frags = [
            _side(-500, 0, 500, 0), _side(-500, 30, 500, 30),
            _side(0, 0, 0, 30), _side(40, 0, 40, 30),
        ]
        with caplog.at_level(logging.INFO, logger="pipeline.cv.enclosure"):
            out, _ = _run(frags)
        assert out == {}
        assert any("rejets" in m and "ancrage 1" in m for m in caplog.messages), caplog.messages


class TestModeCalibration:
    def test_rejected_candidates_published_with_statut(self):
        # cour inter-lanières (rejetée par ancrage) + vrai anneau : en mode
        # calibration les DEUX sortent, avec la colonne statut — pour analyser
        # les détections manquées d'une campagne ground-truth sans rejouer
        # le pipeline en labo.
        frags = [
            _side(-500, 200, 500, 200), _side(-500, 230, 500, 230),
            _side(0, 200, 0, 230), _side(40, 200, 40, 230),
        ] + _rect_fragments(0, 0, 40, 40, gaps=[(0, 18, 4)])
        out, _ = _run(frags, _cfg(mode_calibration=True))
        dets = out.get("enclos", [])
        statuts = sorted(d["statut"] for d in dets)
        assert statuts == ["publie", "rejete_ancrage"]
        publie = next(d for d in dets if d["statut"] == "publie")
        assert publie["ancrage"] > 0.8 and publie["enclos_id"]
        rejete = next(d for d in dets if d["statut"] == "rejete_ancrage")
        assert rejete["ancrage"] < 0.3
        assert rejete["closure_ratio"] > 0.0  # scores de filtres renseignés

    def test_default_mode_only_published_with_statut_publie(self):
        frags = _rect_fragments(0, 0, 40, 40, gaps=[(0, 18, 4)])
        out, _ = _run(frags)
        assert [d["statut"] for d in out["enclos"]] == ["publie"]
