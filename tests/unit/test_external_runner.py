from __future__ import annotations

import pytest

from pipeline.cv.external_runner import (
    RunnerPayload,
    _parse_runner_stdout,
    find_external_cv_runner,
)


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


class TestParseRunnerStdoutTileProgress:
    """Progression par tuile SAHI (``SAHI: X/Y tuiles traitées``).

    Le binaire émet ces lignes toutes les ~10 tuiles ; sans remontée UI,
    une grande dalle (144 tuiles ≈ 3 min CPU) fige l'affichage sur
    « Image i/N » pendant toute son inférence.
    """

    def test_tile_line_invokes_callback(self):
        calls, logs = [], []
        _parse_runner_stdout(
            "[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 10/144 tuiles traitées",
            logs.append,
            tile_progress=lambda c, t: calls.append((c, t)),
        )
        assert calls == [(10, 144)]
        # La trace fichier [cv_runner] est conservée à l'identique.
        assert any("10/144" in m for m in logs)

    def test_announce_line_without_slash_does_not_invoke(self):
        # « SAHI: 144 tuiles » (annonce du total) n'est pas une progression.
        calls = []
        _parse_runner_stdout(
            "[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 144 tuiles",
            lambda _m: None,
            tile_progress=lambda c, t: calls.append((c, t)),
        )
        assert calls == []

    def test_segformer_variant_invokes_callback(self):
        calls = []
        _parse_runner_stdout(
            "[cv_runner_onnx][INFO] SegFormer SAHI: 3/36 tuiles traitées",
            lambda _m: None,
            tile_progress=lambda c, t: calls.append((c, t)),
        )
        assert calls == [(3, 36)]

    def test_tile_line_without_callback_does_not_raise(self):
        _parse_runner_stdout(
            "[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 10/144 tuiles traitées",
            lambda _m: None,
        )

    def test_callback_exception_is_swallowed(self):
        def _boom(_c, _t):
            raise RuntimeError("ui went away")

        _parse_runner_stdout(
            "[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 10/144 tuiles traitées",
            lambda _m: None,
            tile_progress=_boom,
        )


class TestParseRunnerStdoutSummary:
    """La ligne ``summary:`` porte le total de détections du run — la
    fonction le renvoie pour que le narrateur puisse annoncer
    « Détection terminée : N zones »."""

    def test_summary_line_returns_total_detections(self):
        result = _parse_runner_stdout(
            "summary: success=3 total_detections=47", lambda _m: None
        )
        assert result == 47

    def test_summary_without_detections_returns_none(self):
        assert _parse_runner_stdout("summary: success=3", lambda _m: None) is None

    def test_other_lines_return_none(self):
        assert (
            _parse_runner_stdout(
                "progress=1/3 image=a.png status=processing", lambda _m: None
            )
            is None
        )
        assert _parse_runner_stdout("[cv_runner_onnx][INFO] divers", lambda _m: None) is None
