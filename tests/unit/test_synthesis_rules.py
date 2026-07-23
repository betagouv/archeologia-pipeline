"""Règles de synthèse typées (type: dbscan | enclosure) — parsing et bornes.

La section ``clustering:`` d'args.yaml devient des règles typées : ``type``
absent ⇒ dbscan (rétro-compat totale), ``type: enclosure`` ⇒ EnclosureRule,
type inconnu ⇒ règle ignorée avec warning. Bornes et surcharges UI par type.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")

from pipeline.cv.clustering_bounds import (
    BOUNDS_BY_TYPE,
    ENCLOSURE_BOUNDS,
    sanitize_clustering_overrides,
)
from pipeline.cv.model_config import load_clustering_config_from_model
from pipeline.cv.model_profile import ClusteringRule, EnclosureRule, _parse_clustering


DBSCAN_CFG = {"target_classes": ["cratere"], "output_class_name": "zone_crateres",
              "min_confidence": 0.4}
ENCLOS_CFG = {"type": "enclosure", "target_classes": ["parcellaire", "talus_fosse"],
              "output_class_name": "enclos"}


class TestParseTyped:
    def test_default_type_is_dbscan(self):
        (rule,) = _parse_clustering({"clustering": [DBSCAN_CFG]})
        assert isinstance(rule, ClusteringRule)
        assert rule.type == "dbscan"
        assert rule.to_dict()["type"] == "dbscan"

    def test_enclosure_rule_defaults(self):
        (rule,) = _parse_clustering({"clustering": [ENCLOS_CFG]})
        assert isinstance(rule, EnclosureRule)
        assert rule.type == "enclosure"
        assert rule.target_classes == ("parcellaire", "talus_fosse")
        assert rule.output_class_name == "enclos"
        assert rule.gap_tolerance_m == 10.0
        assert rule.min_area_m2 == 50.0
        assert rule.max_area_m2 == 60000.0
        assert rule.min_closure == 0.6
        assert rule.max_elongation == 3.0
        assert rule.min_confidence == 0.0
        assert rule.to_dict()["type"] == "enclosure"

    def test_enclosure_bounds_clamped(self):
        cfg = dict(ENCLOS_CFG, gap_tolerance_m=500, min_closure=2.0)
        (rule,) = _parse_clustering({"clustering": [cfg]})
        assert rule.gap_tolerance_m == 50.0
        assert rule.min_closure == 1.0

    def test_max_area_below_min_coerced(self):
        cfg = dict(ENCLOS_CFG, min_area_m2=5000, max_area_m2=100)
        (rule,) = _parse_clustering({"clustering": [cfg]})
        assert rule.max_area_m2 >= rule.min_area_m2

    def test_unknown_type_ignored(self, caplog):
        cfg = dict(ENCLOS_CFG, type="banane")
        rules = _parse_clustering({"clustering": [cfg, DBSCAN_CFG]})
        assert len(rules) == 1 and isinstance(rules[0], ClusteringRule)

    def test_mixed_rules_both_parsed(self):
        rules = _parse_clustering({"clustering": [DBSCAN_CFG, ENCLOS_CFG]})
        assert [r.type for r in rules] == ["dbscan", "enclosure"]

    def test_enclosure_default_output_name(self):
        cfg = {"type": "enclosure", "target_classes": ["parcellaire"]}
        (rule,) = _parse_clustering({"clustering": [cfg]})
        assert rule.output_class_name == "enclos_parcellaire"


class TestOverridesByType:
    def test_enclosure_overrides_kept(self):
        out = sanitize_clustering_overrides(
            {"gap_tolerance_m": 12, "min_closure": 0.7}, rule_type="enclosure")
        assert out == {"gap_tolerance_m": 12.0, "min_closure": 0.7}

    def test_dbscan_key_rejected_for_enclosure(self):
        out = sanitize_clustering_overrides({"eps_m": 40}, rule_type="enclosure")
        assert out == {}

    def test_enclosure_key_rejected_for_dbscan(self):
        out = sanitize_clustering_overrides({"gap_tolerance_m": 12}, rule_type="dbscan")
        assert out == {}

    def test_output_geometry_only_for_dbscan(self):
        assert sanitize_clustering_overrides(
            {"output_geometry": "concave_hull"}, rule_type="dbscan"
        ) == {"output_geometry": "concave_hull"}
        assert sanitize_clustering_overrides(
            {"output_geometry": "concave_hull"}, rule_type="enclosure") == {}

    def test_bounds_registry_exposes_both_types(self):
        assert set(BOUNDS_BY_TYPE) == {"dbscan", "enclosure"}
        assert "gap_tolerance_m" in ENCLOSURE_BOUNDS


class TestLegacyLoader:
    def test_legacy_loader_tags_type(self, tmp_path):
        d = tmp_path / "model" / "weights"
        d.mkdir(parents=True)
        (d / "best.onnx").write_bytes(b"")
        (tmp_path / "model" / "args.yaml").write_text(
            "clustering:\n"
            "  - target_classes: [cratere]\n"
            "    output_class_name: zone_crateres\n"
            "    min_confidence: 0.4\n"
            "  - type: enclosure\n"
            "    target_classes: [parcellaire]\n"
            "    output_class_name: enclos\n",
            encoding="utf-8",
        )
        configs = load_clustering_config_from_model(d / "best.onnx")
        assert [c["type"] for c in configs] == ["dbscan", "enclosure"]
        enclos = configs[1]
        # dict minimal : pas de min_confidence ⇒ _peek_clustering_min_confidence
        # (run_context) ignore naturellement les règles non-dbscan.
        assert "min_confidence" not in enclos
        assert "eps_m" not in enclos
