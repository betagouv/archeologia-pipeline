from __future__ import annotations

from pathlib import Path

import pytest

from app.run_context import build_run_context, RunContext


class TestBuildRunContextIntegration:
    def test_builds_context_from_sample_config(self, sample_config_dict: dict):
        ctx = build_run_context(sample_config_dict)
        
        assert isinstance(ctx, RunContext)
        assert ctx.mode == "ign_laz"
        assert ctx.processing_cfg["mnt_resolution"] == 0.5

    def test_builds_context_with_output_dir(self, config_with_output_dir: dict, temp_output_dir: Path):
        ctx = build_run_context(config_with_output_dir)
        
        assert ctx.output_dir == temp_output_dir
        assert ctx.output_dir.exists()

    def test_products_config_is_extracted(self, sample_config_dict: dict):
        ctx = build_run_context(sample_config_dict)
        
        assert ctx.products_cfg["MNT"] is True
        assert ctx.products_cfg["LD"] is True
        assert ctx.products_cfg["VAT"] is False

    def test_cv_config_is_extracted(self, sample_config_dict: dict):
        ctx = build_run_context(sample_config_dict)
        
        assert ctx.cv_cfg["enabled"] is False
        assert ctx.cv_cfg["target_rvt"] == "LD"
        assert ctx.cv_cfg["confidence_threshold"] == 0.3

    def test_rvt_params_are_extracted(self, sample_config_dict: dict):
        ctx = build_run_context(sample_config_dict)
        
        assert ctx.rvt_params["mdh"]["num_directions"] == 16
        assert ctx.rvt_params["svf"]["radius"] == 10
