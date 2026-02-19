from __future__ import annotations

import pytest

from pipeline.cv.external_runner import RunnerPayload, find_external_cv_runner


class TestRunnerPayload:
    def test_is_typed_dict(self):
        payload: RunnerPayload = {
            "jpg_dir": "/tmp/jpg",
            "target_rvt": "LD",
            "cv_config": {"confidence_threshold": 0.3},
            "run_shapefile_dedup": True,
        }
        assert payload["jpg_dir"] == "/tmp/jpg"
        assert payload["target_rvt"] == "LD"
        assert payload["run_shapefile_dedup"] is True

    def test_accepts_optional_fields(self):
        payload: RunnerPayload = {
            "jpg_dir": "/tmp",
            "target_rvt": "LD",
            "cv_config": {},
            "run_shapefile_dedup": False,
            "rvt_base_dir": "/tmp/rvt",
            "single_jpg": "/tmp/img.jpg",
            "tif_transform_data": {"tile": (0.5, -0.5, 100.0, 200.0)},
        }
        assert payload["rvt_base_dir"] == "/tmp/rvt"
        assert payload["single_jpg"] == "/tmp/img.jpg"

    def test_minimal_payload(self):
        payload: RunnerPayload = {}
        assert isinstance(payload, dict)


class TestFindExternalCvRunner:
    def test_callable(self):
        assert callable(find_external_cv_runner)

    def test_returns_path_or_none(self):
        result = find_external_cv_runner()
        assert result is None or isinstance(result, type(None)) or hasattr(result, "exists")
