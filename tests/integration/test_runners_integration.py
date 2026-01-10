from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.cancel_token import CancelToken
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context
from app.runners.registry import get_runner
from app.runners.ign_local_runner import IgnOrLocalRunner
from app.runners.existing_mnt_runner import ExistingMntRunner
from app.runners.existing_rvt_runner import ExistingRvtRunner


class TestRunnersIntegration:
    def test_ign_local_runner_initializes(self):
        runner = IgnOrLocalRunner()
        assert runner is not None
        assert hasattr(runner, "run")

    def test_existing_mnt_runner_initializes(self):
        runner = ExistingMntRunner()
        assert runner is not None
        assert hasattr(runner, "run")

    def test_existing_rvt_runner_initializes(self):
        runner = ExistingRvtRunner()
        assert runner is not None
        assert hasattr(runner, "run")

    def test_registry_returns_correct_runner_types(self):
        assert isinstance(get_runner("ign_laz"), IgnOrLocalRunner)
        assert isinstance(get_runner("local_laz"), IgnOrLocalRunner)
        assert isinstance(get_runner("existing_mnt"), ExistingMntRunner)
        assert isinstance(get_runner("existing_rvt"), ExistingRvtRunner)

    def test_registry_raises_for_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_runner("unknown_mode")

    def test_run_context_extracts_mode_correctly(self, config_with_output_dir: dict):
        config_with_output_dir["app"]["files"]["data_mode"] = "existing_mnt"
        ctx = build_run_context(config_with_output_dir)
        
        assert ctx.mode == "existing_mnt"

    def test_run_context_extracts_files_config(self, config_with_output_dir: dict):
        config_with_output_dir["app"]["files"]["existing_mnt_dir"] = "/tmp/mnt"
        ctx = build_run_context(config_with_output_dir)
        
        assert ctx.files_cfg["existing_mnt_dir"] == "/tmp/mnt"
