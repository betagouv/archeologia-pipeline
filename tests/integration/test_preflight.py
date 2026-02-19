from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pipeline.preflight import run_preflight, CheckResult


class TestPreflightChecks:
    def test_preflight_import_succeeds(self):
        assert callable(run_preflight)

    def test_preflight_has_expected_signature(self):
        sig = inspect.signature(run_preflight)
        param_names = list(sig.parameters.keys())

        assert "log" in param_names
        assert "mode" in param_names
        assert "cv_config" in param_names
        assert "products" in param_names
        assert "files_config" in param_names
        assert "output_dir" in param_names

    def test_preflight_returns_bool(self):
        logs = []
        result = run_preflight(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={},
            log=logs.append,
            files_config={"existing_rvt_dir": "/nonexistent"},
            output_dir=None,
        )
        assert isinstance(result, bool)

    def test_check_result_dataclass(self):
        r = CheckResult(name="test", ok=True, details="ok", critical=False)
        assert r.name == "test"
        assert r.ok is True
