"""Moteur ``run_alignment`` (brique axe linéaire) — bandes directionnelles.

Fixtures géométriques pures (mètres) : les fragments de parcellaire sont des
lignes bufferisées (~2 m de large) posées sur un ou plusieurs brins parallèles.
On vérifie la constitution des bandes (plusieurs brins = UN axe), le chaînage
longitudinal (coupure aux trous > max_gap), les filtres durs (longueur,
couverture, nb de fragments), les scores (nb_brins, parallelisme,
connecteurs_perp, discordance) et la traçabilité (axe_id).
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")

from shapely.affinity import rotate
from shapely.geometry import LineString

from pipeline.cv.alignment import run_alignment


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _frag(x1, x2, y, angle_deg=0.0, conf=0.5):
    geom = LineString([(x1, y), (x2, y)]).buffer(1.0)
    if angle_deg:
        geom = rotate(geom, angle_deg, origin=(0, 0))
    return {"geometry": geom, "confidence": conf,
            "model_pred": "parcellaire", "model_name": "formes"}


def _vfrag(x, y1, y2, conf=0.5):
    return {"geometry": LineString([(x, y1), (x, y2)]).buffer(1.0),
            "confidence": conf, "model_pred": "parcellaire", "model_name": "formes"}


def _cfg(**over):
    cfg = {"type": "alignment", "target_classes": ["parcellaire"],
           "output_class_name": "axe_lineaire", "band_width_m": 40.0,
           "angle_tolerance_deg": 20.0, "min_length_m": 500.0,
           "max_gap_m": 200.0, "min_coverage": 0.25, "min_sources": 5,
           "min_confidence": 0.0}
    cfg.update(over)
    return cfg


# Enfilade mono-brin de ~800 m (5 fragments, trous < 200 m)
STRAND = [(0, 120), (180, 300), (360, 480), (540, 660), (700, 800)]


def _strand(y=0.0, angle_deg=0.0, conf=0.5, intervals=STRAND):
    return [_frag(a, b, y, angle_deg=angle_deg, conf=conf) for a, b in intervals]


def _run(frags, cfg=None):
    data = {"parcellaire": list(frags)}
    return run_alignment(data, [cfg or _cfg()])


def _azimut_folded(d):
    return min(d["azimut_deg"], 180.0 - d["azimut_deg"])


# ----------------------------------------------------------------------
# Bandes et chaînage
# ----------------------------------------------------------------------
class TestBands:
    def test_single_strand_series(self):
        out, updated = _run(_strand())
        assert "axe_lineaire" in out and len(out["axe_lineaire"]) == 1
        d = out["axe_lineaire"][0]
        assert 700 < d["longueur_m"] < 900
        assert 0.6 < d["couverture"] < 0.85
        assert d["nb_brins"] == 1
        assert _azimut_folded(d) < 3.0
        assert d["nb_sources"] == 5
        assert d["parallelisme"] == 0
        assert d["discordance_deg"] == 90.0  # aucun grain local : isolé
        assert all(det.get("axe_id") == d["axe_id"]
                   for det in updated["parcellaire"])

    def test_three_strands_make_one_axis(self):
        frags = []
        for y in (0.0, 12.0, 24.0):
            frags += _strand(y=y, intervals=[(0, 180), (240, 420), (480, 600)])
        out, _ = _run(frags, _cfg(min_sources=3))
        assert len(out.get("axe_lineaire", [])) == 1  # UNE bande, pas 3 axes
        d = out["axe_lineaire"][0]
        assert d["nb_brins"] == 3
        assert 9.0 < d["espacement_brins_m"] < 15.0
        assert d["largeur_m"] >= 24.0
        assert d["nb_sources"] == 9

    def test_gap_beyond_max_splits_chains(self):
        near = [(0, 150), (200, 350), (400, 600)]
        far = [(900, 1050), (1100, 1250), (1300, 1500)]
        out, _ = _run(_strand(intervals=near + far), _cfg(min_sources=3))
        assert len(out.get("axe_lineaire", [])) == 2

    def test_rotated_fixture_same_result(self):
        out, _ = _run(_strand(angle_deg=37.0))
        assert len(out.get("axe_lineaire", [])) == 1
        assert 32.0 < out["axe_lineaire"][0]["azimut_deg"] < 42.0

    def test_bend_makes_two_families(self):
        leg1 = _strand(intervals=[(0, 150), (200, 350), (400, 600)])
        leg2 = _strand(angle_deg=45.0,
                       intervals=[(700, 850), (900, 1050), (1100, 1300)])
        out, _ = _run(leg1 + leg2, _cfg(min_sources=3))
        dets = out.get("axe_lineaire", [])
        assert len(dets) == 2
        azs = sorted(_azimut_folded(d) for d in dets)
        assert azs[0] < 5.0 and 40.0 < azs[1] < 50.0


# ----------------------------------------------------------------------
# Filtres durs
# ----------------------------------------------------------------------
class TestHardFilters:
    def test_short_alignment_rejected(self):
        out, _ = _run(
            _strand(intervals=[(0, 100), (150, 250), (280, 300)]),
            _cfg(min_sources=3))
        assert out == {}

    def test_low_coverage_rejected(self):
        intervals = [(x, x + 10) for x in (0, 190, 380, 570, 760, 950)]
        out, _ = _run(_strand(intervals=intervals), _cfg(min_sources=3))
        assert out == {}

    def test_min_sources_rejects(self):
        out, _ = _run(_strand(intervals=[(0, 300), (400, 700)]),
                      _cfg(min_sources=3))
        assert out == {}

    def test_min_confidence_filters_sources(self):
        frags = _strand(conf=0.05)[:3] + _strand(conf=0.5)[3:]
        out, _ = _run(frags, _cfg(min_sources=3, min_confidence=0.3))
        assert out == {}  # il ne reste que 2 fragments confiants


# ----------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------
class TestScores:
    def test_coaxial_bands_parallelisme(self):
        frags = []
        for y in (0.0, 80.0, 160.0):
            frags += _strand(y=y, intervals=[(0, 150), (200, 350), (400, 600)])
        out, _ = _run(frags, _cfg(min_sources=3))
        dets = out.get("axe_lineaire", [])
        assert len(dets) == 3
        assert all(d["parallelisme"] == 2 for d in dets)

    def test_perpendicular_connectors(self):
        frags = _strand() + [_vfrag(x, -30, 30) for x in (150, 400, 650)]
        out, _ = _run(frags)
        dets = out.get("axe_lineaire", [])
        assert len(dets) == 1  # pas d'axe parasite depuis les ⊥ (n < min)
        assert dets[0]["connecteurs_perp"] > 0

    def test_discordance_low_with_concordant_grain(self):
        # grain local co-orienté (2 fragments à y=60, hors bande, sous min_sources)
        frags = _strand() + [_frag(100, 250, 60.0), _frag(400, 550, 60.0)]
        out, _ = _run(frags)
        dets = out.get("axe_lineaire", [])
        assert len(dets) == 1
        assert dets[0]["discordance_deg"] < 5.0

    def test_confidence_is_mean_of_members(self):
        frags = [_frag(a, b, 0.0, conf=(0.2 if i % 2 else 0.6))
                 for i, (a, b) in enumerate(STRAND)]
        out, _ = _run(frags)
        assert 0.2 < out["axe_lineaire"][0]["confidence"] < 0.6

    def test_far_fragment_not_tagged(self):
        far = _frag(5000, 5150, 5000.0)
        out, updated = run_alignment(
            {"parcellaire": _strand() + [far]}, [_cfg()])
        assert len(out["axe_lineaire"]) == 1
        assert "axe_id" not in updated["parcellaire"][-1]
