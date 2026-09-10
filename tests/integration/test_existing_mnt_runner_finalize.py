"""TEST-11 / ROB-14 (audit v2) : verrouille le câblage réel du correctif
bc6eb33 sur ExistingMntRunner — finalize_pipeline TOUJOURS appelée (finally),
avec l'issue réelle du run (success / cancelled / failed).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("shapely")

from app.cancel_token import CancelToken
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context
from app.runners.existing_mnt_runner import ExistingMntRunner
from pipeline.cancellation import PipelineCancelled


def _ctx(config_with_output_dir: dict, tmp_path):
    mnt_dir = tmp_path / "mnt_in"
    mnt_dir.mkdir()
    cfg = config_with_output_dir
    cfg["app"]["files"]["data_mode"] = "existing_mnt"
    cfg["app"]["files"]["existing_mnt_dir"] = str(mnt_dir)
    cfg["computer_vision"] = {"enabled": False}
    return build_run_context(cfg)


@pytest.fixture
def finalized(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "app.runners.existing_mnt_runner.finalize_pipeline",
        lambda **kw: calls.append(kw),
    )
    return calls


def _run(ctx, monkeypatch, finalized, behaviour):
    monkeypatch.setattr("pipeline.modes.existing_mnt.run_existing_mnt", behaviour)
    ExistingMntRunner().run(
        ctx=ctx,
        reporter=NullProgressReporter(),
        cancel=CancelToken(threading.Event()),
    )


class TestExistingMntRunnerFinalize:
    def test_succes_finalise_avec_outcome_success(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        _run(
            _ctx(config_with_output_dir, tmp_path),
            monkeypatch,
            finalized,
            lambda **kw: SimpleNamespace(total=3, candidates=3),
        )
        assert len(finalized) == 1
        assert finalized[0]["outcome"] == "success"
        assert finalized[0]["tiles_processed"] == 3

    def test_exception_fatale_finalise_avec_outcome_failed(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)

        def boom(**kw):
            raise RuntimeError("dossier MNT vide")

        monkeypatch.setattr("pipeline.modes.existing_mnt.run_existing_mnt", boom)
        with pytest.raises(RuntimeError):
            ExistingMntRunner().run(
                ctx=ctx,
                reporter=NullProgressReporter(),
                cancel=CancelToken(threading.Event()),
            )
        assert len(finalized) == 1
        assert finalized[0]["outcome"] == "failed"

    def test_annulation_finalise_avec_outcome_cancelled(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        def cancelling(**kw):
            raise PipelineCancelled()

        _run(
            _ctx(config_with_output_dir, tmp_path),
            monkeypatch,
            finalized,
            cancelling,
        )
        assert len(finalized) == 1
        assert finalized[0]["outcome"] == "cancelled"
