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

    def test_halo_resolver_wired_when_intermediaires_exists(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run_existing_rvt(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_images=1, total_detections=None)

        monkeypatch.setattr("pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr("app.runners.existing_rvt_runner.finalize_pipeline", lambda **_kwargs: None)

        ctx = _ctx(tmp_path, {"enabled": True, "target_rvt": "LD"})
        (ctx.output_dir / "intermediaires").mkdir()

        resolved = tmp_path / "source_non_rognee.tif"
        seen = {}

        def fake_resolve(cropped, src, rvt, params):
            seen.update(cropped=cropped, src=src, rvt=rvt, params=params)
            return resolved

        monkeypatch.setattr(
            "app.runners.existing_rvt_runner.resolve_uncropped_tif", fake_resolve
        )

        ExistingRvtRunner().run(
            ctx=ctx, reporter=_Reporter(), cancel=CancelToken(threading.Event())
        )

        resolver = captured.get("inference_tif_resolver")
        assert callable(resolver)
        assert resolver(Path("LHD_FXX_0390_6818_LD.tif")) == resolved
        assert seen["src"] == ctx.output_dir / "intermediaires"
        assert seen["rvt"] == "LD"

    def test_no_halo_resolver_without_intermediaires(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run_existing_rvt(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_images=1, total_detections=None)

        monkeypatch.setattr("pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr("app.runners.existing_rvt_runner.finalize_pipeline", lambda **_kwargs: None)

        ExistingRvtRunner().run(
            ctx=_ctx(tmp_path, {"enabled": True, "target_rvt": "LD"}),
            reporter=_Reporter(),
            cancel=CancelToken(threading.Event()),
        )

        assert captured.get("inference_tif_resolver") is None

    def test_progress_advances_in_cv_band_and_wires_on_busy(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run_existing_rvt(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_images=10)

        # Un run CV unique, sans dépendre de la résolution réelle des modèles.
        monkeypatch.setattr(
            "pipeline.cv.class_utils.resolve_cv_runs",
            lambda _cfg: [{"selected_model": "m", "target_rvt": "LD", "enabled": True}],
        )
        monkeypatch.setattr("pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr("app.runners.existing_rvt_runner.finalize_pipeline", lambda **_kwargs: None)

        reporter = _Reporter()
        ExistingRvtRunner().run(
            ctx=_ctx(tmp_path, {"enabled": True, "target_rvt": "LD"}),
            reporter=reporter,
            cancel=CancelToken(threading.Event()),
        )

        # on_busy doit être câblé (régime large-raster).
        assert callable(captured.get("on_busy"))

        # En simulant la sous-progression image, la barre avance dans [10, 95].
        img_cb = captured.get("image_progress")
        assert callable(img_cb)
        before = len(reporter.progress_values)
        for i in range(0, 11):
            img_cb(i, 10, f"img{i}")
        emitted = reporter.progress_values[before:]
        assert emitted, "image_progress doit émettre de la progression"
        assert all(10 <= v <= 95 for v in emitted), emitted
        assert emitted == sorted(emitted), "progression monotone"
        assert emitted[-1] == 95
