from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.cv.model_config import (
    _resolve_model_dir,
    resolve_model_weights_path,
    resolve_cv_runs,
    is_rfdetr_model,
    load_clustering_config_from_model,
)


# ── _resolve_model_dir ──────────────────────────────────────────

class TestResolveModelDir:
    def test_returns_parent_parent_for_weights_file(self, tmp_path: Path):
        weights = tmp_path / "my_model" / "weights" / "best.onnx"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fake")
        assert _resolve_model_dir(weights) == tmp_path / "my_model"

    def test_returns_parent_for_non_weights_file(self, tmp_path: Path):
        f = tmp_path / "my_model" / "model.pt"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"fake")
        assert _resolve_model_dir(f) == tmp_path / "my_model"

    def test_returns_dir_itself_for_directory(self, tmp_path: Path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        assert _resolve_model_dir(model_dir) == model_dir

    def test_returns_path_for_nonexistent(self, tmp_path: Path):
        p = tmp_path / "nonexistent"
        assert _resolve_model_dir(p) == p


# ── resolve_model_weights_path ──────────────────────────────────

class TestResolveModelWeightsPath:
    def test_returns_none_for_empty_model(self):
        assert resolve_model_weights_path({}) is None
        assert resolve_model_weights_path({"selected_model": ""}) is None

    def test_returns_direct_path_if_exists(self, tmp_path: Path):
        f = tmp_path / "model.onnx"
        f.write_bytes(b"fake")
        result = resolve_model_weights_path({"selected_model": str(f)})
        assert result == f

    def test_finds_onnx_in_models_dir(self, tmp_path: Path):
        onnx = tmp_path / "models" / "m1" / "weights" / "best.onnx"
        onnx.parent.mkdir(parents=True)
        onnx.write_bytes(b"fake")
        result = resolve_model_weights_path({
            "selected_model": "m1",
            "models_dir": str(tmp_path / "models"),
        })
        assert result == onnx

    def test_prefers_onnx_over_pt(self, tmp_path: Path):
        base = tmp_path / "models" / "m1" / "weights"
        base.mkdir(parents=True)
        (base / "best.onnx").write_bytes(b"fake")
        (base / "best.pt").write_bytes(b"fake")
        result = resolve_model_weights_path({
            "selected_model": "m1",
            "models_dir": str(tmp_path / "models"),
        })
        assert result.name == "best.onnx"

    def test_fallback_to_pt(self, tmp_path: Path):
        pt = tmp_path / "models" / "m1" / "weights" / "best.pt"
        pt.parent.mkdir(parents=True)
        pt.write_bytes(b"fake")
        result = resolve_model_weights_path({
            "selected_model": "m1",
            "models_dir": str(tmp_path / "models"),
        })
        assert result == pt

    def test_returns_default_path_when_nothing_exists(self):
        result = resolve_model_weights_path({
            "selected_model": "missing_model",
            "models_dir": "models",
        })
        assert result is not None
        assert "missing_model" in str(result)

    def test_handles_none_config(self):
        assert resolve_model_weights_path(None) is None


# ── resolve_cv_runs ─────────────────────────────────────────────

class TestResolveCvRuns:
    def test_returns_empty_for_non_dict(self):
        assert resolve_cv_runs(None) == []
        assert resolve_cv_runs("invalid") == []

    def test_returns_empty_for_no_model(self):
        assert resolve_cv_runs({"selected_model": ""}) == []

    def test_old_format_single_model(self):
        cfg = {"selected_model": "m1", "target_rvt": "SVF", "models_dir": "models"}
        runs = resolve_cv_runs(cfg)
        assert len(runs) == 1
        assert runs[0]["selected_model"] == "m1"
        assert runs[0]["target_rvt"] == "SVF"

    def test_new_format_multiple_runs(self):
        cfg = {
            "runs": [
                {"model": "m1", "target_rvt": "LD"},
                {"model": "m2", "target_rvt": "SVF"},
            ],
            "models_dir": "models",
            "confidence_threshold": 0.3,
        }
        runs = resolve_cv_runs(cfg)
        assert len(runs) == 2
        assert runs[0]["selected_model"] == "m1"
        assert runs[1]["selected_model"] == "m2"

    def test_propagates_selected_classes(self):
        cfg = {
            "runs": [{"model": "m1", "target_rvt": "LD", "selected_classes": ["cls1"]}],
            "models_dir": "models",
        }
        runs = resolve_cv_runs(cfg)
        assert runs[0]["selected_classes"] == ["cls1"]

    def test_propagates_min_area(self):
        cfg = {
            "runs": [{"model": "m1", "target_rvt": "LD", "min_area_m2": 50}],
            "models_dir": "models",
        }
        runs = resolve_cv_runs(cfg)
        assert runs[0]["min_area_m2"] == 50.0

    def test_skips_runs_without_model(self):
        cfg = {
            "runs": [{"model": "", "target_rvt": "LD"}, {"model": "m1", "target_rvt": "SVF"}],
            "models_dir": "models",
        }
        runs = resolve_cv_runs(cfg)
        assert len(runs) == 1

    def test_default_target_rvt_is_LD(self):
        cfg = {"runs": [{"model": "m1"}], "models_dir": "models"}
        runs = resolve_cv_runs(cfg)
        assert runs[0]["target_rvt"] == "LD"


# ── is_rfdetr_model ─────────────────────────────────────────────

class TestIsRfdetrModel:
    def test_returns_false_for_nonexistent(self, tmp_path: Path):
        assert is_rfdetr_model(tmp_path / "nonexistent") is False

    def test_returns_false_for_no_args_yaml(self, tmp_path: Path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        assert is_rfdetr_model(model_dir) is False

    def test_detects_rfdetr(self, tmp_path: Path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "args.yaml").write_text("model: RF-DETR-base\n", encoding="utf-8")
        assert is_rfdetr_model(model_dir) is True

    def test_detects_rfdetr_lowercase(self, tmp_path: Path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "args.yaml").write_text("model: rfdetr_large\n", encoding="utf-8")
        assert is_rfdetr_model(model_dir) is True

    def test_returns_false_for_yolo(self, tmp_path: Path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "args.yaml").write_text("model: yolov8n-seg\n", encoding="utf-8")
        assert is_rfdetr_model(model_dir) is False

    def test_resolves_through_weights_path(self, tmp_path: Path):
        weights = tmp_path / "my_model" / "weights" / "best.onnx"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fake")
        (tmp_path / "my_model" / "args.yaml").write_text("model: RF-DETR\n", encoding="utf-8")
        assert is_rfdetr_model(weights) is True


# ── load_clustering_config_from_model ───────────────────────────

class TestLoadClusteringConfig:
    def test_returns_none_for_nonexistent(self, tmp_path: Path):
        assert load_clustering_config_from_model(tmp_path / "nope") is None

    def test_returns_none_for_no_args_yaml(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        assert load_clustering_config_from_model(d) is None

    def test_returns_none_for_no_clustering_section(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "args.yaml").write_text("model: yolov8\n", encoding="utf-8")
        assert load_clustering_config_from_model(d) is None

    def test_parses_single_clustering_config(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  - target_classes: [cratere_obus]\n"
            "    min_confidence: 0.5\n"
            "    min_cluster_size: 10\n"
            "    min_samples: 5\n"
            "    eps_m: 30\n"
            "    output_class_name: zone_crateres\n"
            "    output_geometry: convex_hull\n"
            "    buffer_m: 10\n"
            "    min_area_m2: 500\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result is not None
        assert len(result) == 1
        assert result[0]["target_classes"] == ["cratere_obus"]
        assert result[0]["output_class_name"] == "zone_crateres"
        assert result[0]["eps_m"] == 30.0
        assert result[0]["min_area_m2"] == 500.0

    def test_accepts_single_dict_format(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  target_classes: [cls1]\n"
            "  eps_m: 20\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result is not None
        assert len(result) == 1

    def test_accepts_target_class_string(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  - target_class: single_class\n"
            "    eps_m: 15\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result is not None
        assert result[0]["target_classes"] == ["single_class"]

    def test_generates_default_output_class_name(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  - target_classes: [a, b]\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result[0]["output_class_name"] == "cluster_a_b"

    def test_confidence_weight_default(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  - target_classes: [cls]\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result[0]["confidence_weight"] == 0.0

    def test_confidence_weight_custom(self, tmp_path: Path):
        d = tmp_path / "model"
        d.mkdir()
        yaml_content = (
            "clustering:\n"
            "  - target_classes: [cls]\n"
            "    confidence_weight: 0.5\n"
        )
        (d / "args.yaml").write_text(yaml_content, encoding="utf-8")
        result = load_clustering_config_from_model(d)
        assert result[0]["confidence_weight"] == 0.5
