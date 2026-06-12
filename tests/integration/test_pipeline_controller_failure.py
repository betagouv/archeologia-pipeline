"""ROB-01 (audit v1, couvert par ROB-14 v2) : une exception inattendue qui
s'échappe d'un runner remontait au worker sans AUCUN message exploitable côté
contrôleur. Le contrôleur doit émettre une erreur actionnable (reporter.error)
avant de laisser remonter le traceback.
"""
from __future__ import annotations

import threading

import pytest

from app.cancel_token import CancelToken
from app.pipeline_controller import PipelineController
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context


class RecordingReporter(NullProgressReporter):
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)


class _ExplodingRunner:
    def run(self, **kwargs):
        raise RuntimeError("boom interne du runner")


def test_exception_runner_emet_une_erreur_actionnable(
    config_with_output_dir, tmp_path, monkeypatch
):
    rvt_dir = tmp_path / "rvt_in"
    rvt_dir.mkdir()
    cfg = config_with_output_dir
    cfg["app"]["files"]["data_mode"] = "existing_rvt"
    cfg["app"]["files"]["existing_rvt_dir"] = str(rvt_dir)
    ctx = build_run_context(cfg)

    monkeypatch.setattr("pipeline.preflight.run_preflight", lambda **kw: True)
    monkeypatch.setattr(
        "app.runners.registry.get_runner", lambda mode: _ExplodingRunner()
    )

    reporter = RecordingReporter()
    with pytest.raises(RuntimeError):
        PipelineController().run(ctx, reporter, CancelToken(threading.Event()))

    assert any("boom interne du runner" in e for e in reporter.errors)
