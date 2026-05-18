from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from app.cancel_token import CancelToken
from app.run_context import CvConfig, FilesConfig, ProcessingConfig, RunContext
from app.runners.existing_rvt_runner import ExistingRvtRunner


class _Reporter:
    def __init__(self):
        self.messages = []
        self.stages = []
        self.progress_values = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)

    def error(self, msg: str) -> None:
        self.messages.append(msg)

    def user_info(self, msg: str) -> None:
        self.messages.append(msg)

    def user_warning(self, msg: str) -> None:
        self.messages.append(msg)

    def user_success(self, msg: str) -> None:
        self.messages.append(msg)

    def stage(self, msg: str) -> None:
        self.stages.append(msg)

    def progress(self, pct: int) -> None:
        self.progress_values.append(pct)

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        return


def _ctx(tmp_path: Path, cv_raw: dict) -> RunContext:
    rvt_dir = tmp_path / "rvt"
    out_dir = tmp_path / "out"
    rvt_dir.mkdir()
    out_dir.mkdir()
    files = FilesConfig(
        data_mode="existing_rvt",
        output_dir=out_dir,
        existing_rvt_dir=rvt_dir,
    )
    return RunContext(
        mode="existing_rvt",
        output_dir=out_dir,
        files=files,
        processing=ProcessingConfig(),
        cv=CvConfig(enabled=bool(cv_raw.get("enabled", False)), runs=[], raw=cv_raw),
        rvt_params={},
        ui_config={},
    )


class TestExistingRvtRunner:
    def test_processes_existing_rvt_once_when_cv_disabled_and_no_runs(self, tmp_path: Path, monkeypatch):
        calls = []

        def fake_run_existing_rvt(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(total_images=3)

        finalize_calls = []
        monkeypatch.setattr("pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr(
            "app.runners.existing_rvt_runner.finalize_pipeline",
            lambda **kwargs: finalize_calls.append(kwargs),
        )

        reporter = _Reporter()
        ExistingRvtRunner().run(
            ctx=_ctx(tmp_path, {"enabled": False, "target_rvt": "LD"}),
            reporter=reporter,
            cancel=CancelToken(threading.Event()),
        )

        assert len(calls) == 1
        assert calls[0]["cv_config"]["enabled"] is False
        assert calls[0]["indices_folder_name"] == "RVT"
        assert finalize_calls[0]["tiles_processed"] == 3
        assert finalize_calls[0]["active_products"] == ["LD"]

    def test_enabled_cv_without_model_logs_and_copies_rasters_without_inference(self, tmp_path: Path, monkeypatch):
        calls = []

        def fake_run_existing_rvt(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(total_images=1)

        monkeypatch.setattr("pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr("app.runners.existing_rvt_runner.finalize_pipeline", lambda **_kwargs: None)

        reporter = _Reporter()
        ExistingRvtRunner().run(
            ctx=_ctx(tmp_path, {"enabled": True, "target_rvt": "LD", "runs": []}),
            reporter=reporter,
            cancel=CancelToken(threading.Event()),
        )

        assert len(calls) == 1
        assert calls[0]["cv_config"]["enabled"] is False
        assert any("aucun modèle" in msg for msg in reporter.messages)
