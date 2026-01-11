from __future__ import annotations

from pathlib import Path

import pytest


class TestPreflightChecks:
    def test_preflight_import_succeeds(self):
        from pipeline.preflight import run_preflight
        assert callable(run_preflight)

    def test_preflight_has_expected_signature(self):
        from pipeline.preflight import run_preflight
        import inspect
        
        sig = inspect.signature(run_preflight)
        param_names = list(sig.parameters.keys())
        
        assert "log" in param_names
        assert "mode" in param_names
        assert "cv_config" in param_names
        assert "products" in param_names
