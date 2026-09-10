"""Dispatch ``run_synthesis`` : route les règles de synthèse typées.

``type: dbscan`` → run_clustering (moteur actuel), ``type: enclosure`` →
run_enclosure, type inconnu → warning + règle ignorée, jamais fatal. Les
sorties des deux moteurs sont fusionnées dans un seul dict par output_class.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")
pytest.importorskip("scipy")  # run_clustering (cKDTree)

from shapely.geometry import LineString, Point

from pipeline.cv.clustering import run_synthesis


def _band(x1, y1, x2, y2):
    return LineString([(x1, y1), (x2, y2)]).buffer(1.0)


def _square_bands(cx, cy, side):
    h = side / 2.0
    return [
        _band(cx - h, cy - h, cx + h, cy - h),
        _band(cx + h, cy - h, cx + h, cy + h),
        _band(cx + h, cy + h, cx - h, cy + h),
        _band(cx - h, cy + h, cx - h, cy - h),
    ]


DBSCAN_CFG = {
    "type": "dbscan", "target_classes": ["cratere"], "min_confidence": 0.0,
    "min_cluster_size": 3, "min_samples": 2, "eps_m": 30.0,
    "output_class_name": "zone_crateres", "output_geometry": "convex_hull",
    "buffer_m": 5.0, "min_area_m2": 0.0,
}
ENCLOS_CFG = {
    "type": "enclosure", "target_classes": ["parcellaire"],
    "output_class_name": "enclos", "gap_tolerance_m": 8.0,
    "min_area_m2": 50.0, "max_area_m2": 60000.0,
    "min_closure": 0.6, "max_elongation": 3.0, "min_confidence": 0.0,
}
ALIGN_CFG = {
    "type": "alignment", "target_classes": ["parcellaire"],
    "output_class_name": "axe_lineaire", "band_width_m": 40.0,
    "angle_tolerance_deg": 20.0, "min_length_m": 500.0, "max_gap_m": 200.0,
    "min_coverage": 0.25, "min_sources": 5, "min_confidence": 0.0,
}


def _strand_frags():
    return [{"geometry": _band(a, 600.0, b, 600.0), "confidence": 0.5,
             "model_pred": "parcellaire", "model_name": "m"}
            for a, b in ((0, 120), (180, 300), (360, 480), (540, 660), (700, 800))]


def _data():
    craters = [{"geometry": Point(i * 5.0, (i % 2) * 5.0).buffer(2.0),
                "confidence": 0.6, "model_pred": "cratere", "model_name": "m"}
               for i in range(5)]
    frags = [{"geometry": g, "confidence": 0.5, "model_pred": "parcellaire",
              "model_name": "m"} for g in _square_bands(200.0, 0.0, 40.0)]
    return {"cratere": craters, "parcellaire": frags}


class TestRunSynthesis:
    def test_mixed_types_produce_both_outputs(self):
        out, updated = run_synthesis(_data(), [DBSCAN_CFG, ENCLOS_CFG])
        assert set(out) == {"zone_crateres", "enclos"}
        assert any("cluster_id" in d for d in updated["cratere"])
        assert any("enclos_id" in d for d in updated["parcellaire"])

    def test_unknown_type_skipped_without_error(self):
        data = _data()
        out, updated = run_synthesis(
            data, [{"type": "banane", "target_classes": ["cratere"],
                    "output_class_name": "x"}])
        assert out == {}
        assert {k: len(v) for k, v in updated.items()} == \
               {k: len(v) for k, v in data.items()}

    def test_empty_configs_noop(self):
        data = _data()
        out, updated = run_synthesis(data, [])
        assert out == {}
        assert set(updated) == set(data)

    def test_three_types_produce_three_outputs(self):
        data = _data()
        data["parcellaire"] = data["parcellaire"] + _strand_frags()
        out, updated = run_synthesis(data, [DBSCAN_CFG, ENCLOS_CFG, ALIGN_CFG])
        assert set(out) == {"zone_crateres", "enclos", "axe_lineaire"}
        assert any("axe_id" in d for d in updated["parcellaire"])
        assert any("enclos_id" in d for d in updated["parcellaire"])

    def test_type_absent_defaults_to_dbscan(self):
        cfg = {k: v for k, v in DBSCAN_CFG.items() if k != "type"}
        out, _ = run_synthesis(_data(), [cfg])
        assert "zone_crateres" in out
