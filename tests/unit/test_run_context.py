from __future__ import annotations

from pathlib import Path

import pytest

from app.run_context import (
    CvConfig,
    FilesConfig,
    ProcessingConfig,
    ProductsConfig,
    RunContext,
    build_run_context,
    validate_run_context,
)


class TestBuildRunContext:
    def test_extracts_mode_from_config(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.mode == "ign_laz"

    def test_extracts_output_dir_as_path(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.output_dir == Path("/tmp/output")

    def test_files_config_typed(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert isinstance(ctx.files, FilesConfig)
        assert ctx.files.input_file == Path("/tmp/dalles.txt")
        assert ctx.files.local_laz_dir == Path("/tmp/laz")
        assert ctx.files.existing_mnt_dir == Path("/tmp/mnt")
        assert ctx.files.existing_rvt_dir == Path("/tmp/rvt")

    def test_processing_config_typed(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert isinstance(ctx.processing, ProcessingConfig)
        assert ctx.processing.mnt_resolution == 0.5
        assert ctx.processing.tile_overlap == 5.0
        assert ctx.processing.density_resolution == 1.0
        assert ctx.processing.max_workers == 4

    def test_products_config_typed(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert isinstance(ctx.processing.products, ProductsConfig)
        assert ctx.processing.products.MNT is True
        assert ctx.processing.products.DENSITE is False
        assert ctx.processing.products.LD is True

    def test_cv_config_typed(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert isinstance(ctx.cv, CvConfig)
        assert ctx.cv.enabled is True
        # raw conserve toutes les clés héritées (target_rvt, confidence_threshold…)
        assert ctx.cv.raw["target_rvt"] == "LD"
        assert ctx.cv.raw["confidence_threshold"] == 0.3

    def test_extracts_rvt_params(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        assert ctx.rvt_params["mdh"]["num_directions"] == 16
        assert ctx.rvt_params["svf"]["radius"] == 10

    def test_handles_empty_config(self, minimal_config: dict):
        ctx = build_run_context(minimal_config)
        assert ctx.mode == ""
        assert ctx.output_dir is None
        assert ctx.files.input_file is None
        assert ctx.processing.mnt_resolution == 0.5  # défaut
        assert ctx.processing.products.MNT is True   # défaut
        assert ctx.cv.enabled is False
        assert ctx.cv.runs == []
        assert ctx.rvt_params == {}

    def test_handles_none_config(self):
        ctx = build_run_context(None)
        assert ctx.mode == ""
        assert ctx.output_dir is None

    def test_handles_missing_app_key(self):
        config = {"processing": {"mnt_resolution": 1.0}}
        ctx = build_run_context(config)
        assert ctx.mode == ""
        assert ctx.output_dir is None
        assert ctx.processing.mnt_resolution == 1.0

    def test_handles_empty_output_dir(self):
        config = {"app": {"files": {"data_mode": "local_laz", "output_dir": ""}}}
        ctx = build_run_context(config)
        assert ctx.mode == "local_laz"
        assert ctx.output_dir is None

    def test_handles_whitespace_output_dir(self):
        config = {"app": {"files": {"data_mode": "local_laz", "output_dir": "   "}}}
        ctx = build_run_context(config)
        assert ctx.output_dir is None


class TestProductsConfigBehavior:
    def test_active_returns_only_enabled(self):
        p = ProductsConfig(MNT=True, SVF=True, M_HS=False)
        assert "MNT" in p.active()
        assert "SVF" in p.active()
        assert "M_HS" not in p.active()

    def test_active_preserves_canonical_order(self):
        p = ProductsConfig(MNT=True, DENSITE=True, M_HS=True, SVF=True, SLO=True, LD=True, SLRM=True, VAT=True)
        assert p.active() == ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]

    def test_needs_mnt_true_when_mnt_only(self):
        assert ProductsConfig(MNT=True).needs_mnt() is True

    def test_needs_mnt_true_when_visualization_index(self):
        assert ProductsConfig(MNT=False, SVF=True).needs_mnt() is True
        assert ProductsConfig(MNT=False, M_HS=True).needs_mnt() is True

    def test_needs_mnt_false_when_density_only(self):
        # Densité ne dépend pas du MNT.
        assert ProductsConfig(MNT=False, DENSITE=True).needs_mnt() is False

    def test_as_dict_round_trip(self):
        p = ProductsConfig(MNT=True, SVF=True)
        d = p.as_dict()
        assert d["MNT"] is True
        assert d["SVF"] is True
        assert d["M_HS"] is False
        assert set(d.keys()) == {"MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"}


class TestFilesConfigBehavior:
    def test_input_path_for_mode_ign_laz(self):
        f = FilesConfig(data_mode="ign_laz", input_file=Path("/x"))
        assert f.input_path_for_mode() == Path("/x")

    def test_input_path_for_mode_local_laz(self):
        f = FilesConfig(data_mode="local_laz", local_laz_dir=Path("/y"))
        assert f.input_path_for_mode() == Path("/y")

    def test_input_path_for_mode_existing_mnt(self):
        f = FilesConfig(data_mode="existing_mnt", existing_mnt_dir=Path("/z"))
        assert f.input_path_for_mode() == Path("/z")

    def test_input_path_for_unknown_mode(self):
        assert FilesConfig(data_mode="bogus").input_path_for_mode() is None


class TestCvConfigBehavior:
    def test_runs_filtered_to_dicts(self):
        cv = build_run_context(
            {"computer_vision": {"enabled": True, "runs": [{"x": 1}, "garbage", {"y": 2}]}}
        ).cv
        assert cv.runs == [{"x": 1}, {"y": 2}]

    def test_runs_default_empty_when_not_a_list(self):
        cv = build_run_context(
            {"computer_vision": {"enabled": True, "runs": "not_a_list"}}
        ).cv
        assert cv.runs == []

    def test_raw_preserves_input(self):
        cfg = {"computer_vision": {"enabled": True, "selected_model": "X", "extra": 42}}
        cv = build_run_context(cfg).cv
        assert cv.raw["selected_model"] == "X"
        assert cv.raw["extra"] == 42


class TestRunContextDataclass:
    def test_is_frozen(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        with pytest.raises(AttributeError):
            ctx.mode = "other_mode"

    def test_all_fields_present(self, sample_config: dict):
        ctx = build_run_context(sample_config)
        for field in ("mode", "output_dir", "files", "processing", "cv", "rvt_params", "ui_config"):
            assert hasattr(ctx, field), f"missing field {field}"


class TestValidateRunContext:
    """Vérifications métier centralisées (V3.3)."""

    def _ctx(self, **overrides) -> RunContext:
        files = FilesConfig(**{
            "data_mode": "ign_laz",
            "output_dir": Path("/tmp/out"),
            "input_file": None,
            "local_laz_dir": None,
            "existing_mnt_dir": None,
            "existing_rvt_dir": None,
            **overrides.pop("files", {}),
        })
        return RunContext(
            mode=files.data_mode,
            output_dir=files.output_dir,
            files=files,
            processing=ProcessingConfig(),
            cv=CvConfig(),
            rvt_params={},
            ui_config={},
        )

    def test_no_mode_returns_error(self):
        ctx = self._ctx(files={"data_mode": ""})
        errors = validate_run_context(ctx)
        assert any("mode" in e.lower() for e in errors)

    def test_missing_output_dir(self):
        ctx = self._ctx(files={"data_mode": "ign_laz", "output_dir": None})
        errors = validate_run_context(ctx)
        assert any("dossier de sortie" in e for e in errors)

    def test_ign_laz_missing_input_file(self):
        ctx = self._ctx(files={"data_mode": "ign_laz", "input_file": None})
        errors = validate_run_context(ctx)
        assert any("zone/liste" in e for e in errors)

    def test_ign_laz_input_file_not_found(self, tmp_path: Path):
        ghost = tmp_path / "doesnotexist.txt"
        ctx = self._ctx(files={"data_mode": "ign_laz", "input_file": ghost})
        errors = validate_run_context(ctx)
        assert any("introuvable" in e for e in errors)

    def test_ign_laz_input_file_ok(self, tmp_path: Path):
        f = tmp_path / "ok.txt"
        f.write_text("https://example.com/dalle.laz\n")
        ctx = self._ctx(files={"data_mode": "ign_laz", "input_file": f})
        errors = validate_run_context(ctx)
        assert errors == []

    def test_local_laz_missing_dir(self):
        ctx = self._ctx(files={"data_mode": "local_laz", "local_laz_dir": None})
        errors = validate_run_context(ctx)
        assert any("local" in e.lower() for e in errors)

    def test_local_laz_dir_not_found(self, tmp_path: Path):
        ghost = tmp_path / "ghost"
        ctx = self._ctx(files={"data_mode": "local_laz", "local_laz_dir": ghost})
        errors = validate_run_context(ctx)
        assert any("introuvable" in e for e in errors)

    def test_local_laz_dir_ok(self, tmp_path: Path):
        d = tmp_path / "laz"
        d.mkdir()
        ctx = self._ctx(files={"data_mode": "local_laz", "local_laz_dir": d})
        errors = validate_run_context(ctx)
        assert errors == []

    def test_existing_mnt_missing_dir(self):
        ctx = self._ctx(files={"data_mode": "existing_mnt", "existing_mnt_dir": None})
        errors = validate_run_context(ctx)
        assert any("MNT" in e for e in errors)

    def test_existing_rvt_missing_dir(self):
        ctx = self._ctx(files={"data_mode": "existing_rvt", "existing_rvt_dir": None})
        errors = validate_run_context(ctx)
        assert any("RVT" in e for e in errors)

    def test_unknown_mode(self):
        ctx = self._ctx(files={"data_mode": "drone_lidar"})
        errors = validate_run_context(ctx)
        assert any("inconnu" in e for e in errors)

    def test_collects_multiple_errors(self):
        """Pas de short-circuit : on rend toutes les erreurs d'un coup."""
        ctx = self._ctx(files={"data_mode": "ign_laz", "output_dir": None, "input_file": None})
        errors = validate_run_context(ctx)
        assert len(errors) >= 2  # output + input_file

    def test_valid_config_returns_empty(self, tmp_path: Path):
        f = tmp_path / "dalles.txt"
        f.write_text("\n")
        ctx = self._ctx(files={"data_mode": "ign_laz", "input_file": f})
        assert validate_run_context(ctx) == []
