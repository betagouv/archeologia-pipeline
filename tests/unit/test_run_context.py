from __future__ import annotations

from pathlib import Path

import pytest

from app.run_context import RunContext, build_run_context


class TestBuildRunContext:
    def test_extracts_mode_from_config(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.mode == "ign_laz"

    def test_extracts_output_dir_as_path(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.output_dir == Path("/tmp/output")

    def test_extracts_files_cfg(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.files_cfg["input_file"] == "/tmp/dalles.txt"
        assert ctx.files_cfg["local_laz_dir"] == "/tmp/laz"

    def test_extracts_processing_cfg(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.processing_cfg["mnt_resolution"] == 0.5
        assert ctx.processing_cfg["tile_overlap"] == 5

    def test_extracts_products_cfg(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.products_cfg["MNT"] is True
        assert ctx.products_cfg["DENSITE"] is False
        assert ctx.products_cfg["LD"] is True

    def test_extracts_cv_cfg(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.cv_cfg["enabled"] is True
        assert ctx.cv_cfg["target_rvt"] == "LD"
        assert ctx.cv_cfg["confidence_threshold"] == 0.3

    def test_extracts_rvt_params(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.rvt_params["mdh"]["num_directions"] == 16
        assert ctx.rvt_params["svf"]["radius"] == 10

    def test_handles_empty_config(self, minimal_config: dict):
        ctx = build_run_context(minimal_config)
        assert ctx.mode == ""
        assert ctx.output_dir is None
        assert ctx.files_cfg == {}
        assert ctx.processing_cfg == {}
        assert ctx.products_cfg == {}
        assert ctx.cv_cfg == {}
        assert ctx.rvt_params == {}

    def test_handles_none_config(self):
        ctx = build_run_context(None)
        assert ctx.mode == ""
        assert ctx.output_dir is None

    def test_handles_missing_app_key(self):
        config = {"processing": {"mnt_resolution": 1.0}}
        ctx = build_run_context(config)
        assert ctx.mode == ""
        assert ctx.output_dir is None
        assert ctx.processing_cfg["mnt_resolution"] == 1.0

    def test_handles_empty_output_dir(self):
        config = {"app": {"files": {"data_mode": "local_laz", "output_dir": ""}}}
        ctx = build_run_context(config)
        assert ctx.mode == "local_laz"
        assert ctx.output_dir is None

    def test_handles_whitespace_output_dir(self):
        config = {"app": {"files": {"data_mode": "local_laz", "output_dir": "   "}}}
        ctx = build_run_context(config)
        assert ctx.output_dir is None


class TestRunContextDataclass:
    def test_is_frozen(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        with pytest.raises(AttributeError):
            ctx.mode = "other_mode"

    def test_all_fields_present(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert hasattr(ctx, "mode")
        assert hasattr(ctx, "output_dir")
        assert hasattr(ctx, "files_cfg")
        assert hasattr(ctx, "processing_cfg")
        assert hasattr(ctx, "products_cfg")
        assert hasattr(ctx, "cv_cfg")
        assert hasattr(ctx, "rvt_params")
