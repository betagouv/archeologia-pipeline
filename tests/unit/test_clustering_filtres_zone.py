"""Filtres de ZONE du clustering DBSCAN (audit 2026-09-14) : ``min_conf_p90`` et
``max_elong_med`` écartent une zone d'après ses cratères membres, et les deux
mesures sont écrites comme attributs de chaque zone gardée."""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")
pytest.importorskip("scipy")

from shapely.geometry import Point, box

from pipeline.cv.clustering import _median_elongation, run_clustering


def _cfg(**extra):
    base = {"type": "dbscan", "target_classes": ["cratere"], "min_confidence": 0.0,
            "min_cluster_size": 3, "min_samples": 2, "eps_m": 30.0,
            "output_class_name": "zone_crateres", "output_geometry": "convex_hull",
            "buffer_m": 5.0, "min_area_m2": 0.0}
    base.update(extra)
    return base


def _groupe(x0, confs, elong=1.0):
    """Cinq cratères alignés tous les 5 m, confiances données, boîtes d'allongement ``elong``."""
    return [{"geometry": box(x0 + i * 5.0, 0.0, x0 + i * 5.0 + 2.0 * elong, 2.0),
             "confidence": c, "model_pred": "cratere", "model_name": "m"}
            for i, c in enumerate(confs)]


class TestFiltreConfP90:
    def test_sans_filtre_la_zone_porte_les_deux_attributs(self):
        data = {"cratere": _groupe(0.0, [0.3, 0.4, 0.5, 0.6, 0.9])}
        zones, _ = run_clustering(data, [_cfg()])
        z = zones["zone_crateres"][0]
        assert z["nb_detect"] == 5
        assert z["conf_p90"] == pytest.approx(0.78, abs=0.01)   # percentile 90 de [0.3..0.9]
        assert z["elong_med"] == pytest.approx(1.0, abs=0.01)

    def test_zone_peu_sure_ecartee(self):
        data = {"cratere": _groupe(0.0, [0.3, 0.35, 0.4, 0.45, 0.5])}
        zones, _ = run_clustering(data, [_cfg(min_conf_p90=0.60)])
        assert "zone_crateres" not in zones

    def test_zone_sure_gardee(self):
        data = {"cratere": _groupe(0.0, [0.3, 0.4, 0.5, 0.7, 0.9])}
        zones, _ = run_clustering(data, [_cfg(min_conf_p90=0.60)])
        assert len(zones["zone_crateres"]) == 1

    def test_deux_groupes_un_seul_passe(self):
        data = {"cratere": _groupe(0.0, [0.3, 0.3, 0.3, 0.3, 0.3]) + _groupe(500.0, [0.8, 0.8, 0.8, 0.8, 0.8])}
        zones, _ = run_clustering(data, [_cfg(min_conf_p90=0.60)])
        assert [z["nb_detect"] for z in zones["zone_crateres"]] == [5]
        assert zones["zone_crateres"][0]["geometry"].centroid.x > 400


class TestFiltreAllongement:
    def test_boites_allongees_ecartees(self):
        data = {"cratere": _groupe(0.0, [0.8] * 5, elong=1.8)}
        zones, _ = run_clustering(data, [_cfg(max_elong_med=1.32)])
        assert "zone_crateres" not in zones

    def test_boites_rondes_gardees(self):
        data = {"cratere": _groupe(0.0, [0.8] * 5, elong=1.1)}
        zones, _ = run_clustering(data, [_cfg(max_elong_med=1.32)])
        assert zones["zone_crateres"][0]["elong_med"] == pytest.approx(1.1, abs=0.01)

    def test_median_elongation_ignore_les_geometries_sans_bounds(self):
        assert _median_elongation([]) == 1.0
        assert _median_elongation([Point(0, 0).buffer(1.0)]) == pytest.approx(1.0, abs=0.01)
