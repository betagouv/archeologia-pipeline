from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.cancel_token import CancelToken
from app.pipeline_controller import PipelineController
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context


class TestPipelineControllerIntegration:
    def test_controller_initializes(self):
        controller = PipelineController()
        assert controller is not None

    def test_controller_has_run_method(self):
        controller = PipelineController()
        assert hasattr(controller, "run")
        assert callable(controller.run)

    def test_run_context_builds_correctly(self, config_with_output_dir: dict, temp_output_dir: Path):
        ctx = build_run_context(config_with_output_dir)
        
        assert ctx.output_dir == temp_output_dir
        assert ctx.mode == "ign_laz"

    def test_cancel_token_is_set(self):
        cancel_event = threading.Event()
        cancel = CancelToken(cancel_event)
        
        assert cancel.is_cancelled() is False
        cancel_event.set()
        assert cancel.is_cancelled() is True
