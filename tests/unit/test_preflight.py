from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pipeline.preflight import CheckResult, _check_input_path, collect_preflight_results


class TestCheckResult:
    def test_frozen_dataclass(self):
        r = CheckResult(name="test", ok=True, details="ok", critical=False)
        with pytest.raises(AttributeError):
            r.ok = False

    def test_fields(self):
        r = CheckResult(name="pdal", ok=True, details="found", critical=True)
        assert r.name == "pdal"
        assert r.ok is True
        assert r.details == "found"
        assert r.critical is True


class TestCheckInputPath:
    def test_missing_key_appends_not_configured(self):
        results = []
        _check_input_path({}, "missing_key", "label", results=results)
        assert len(results) == 1
        assert results[0].ok is False
        assert "non configuré" in results[0].details

    def test_empty_string_appends_not_configured(self):
        results = []
        _check_input_path({"k": ""}, "k", "label", results=results)
        assert len(results) == 1
        assert results[0].ok is False

    def test_whitespace_only_appends_not_configured(self):
        results = []
        _check_input_path({"k": "   "}, "k", "label", results=results)
        assert len(results) == 1
        assert results[0].ok is False

    def test_nonexistent_dir_appends_not_found(self):
        results = []
        _check_input_path(
            {"k": "/nonexistent/path/xyz"},
            "k", "label",
            expect_dir=True,
            results=results,
        )
        assert len(results) == 1
        assert results[0].ok is False
        assert "introuvable" in results[0].details

    def test_existing_dir_appends_ok(self):
        with tempfile.TemporaryDirectory() as td:
            results = []
            _check_input_path(
                {"k": td}, "k", "label",
                expect_dir=True,
                results=results,
            )
            assert len(results) == 1
            assert results[0].ok is True

    def test_dir_with_matching_extensions(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "file1.tif").write_bytes(b"x")
            (Path(td) / "file2.tif").write_bytes(b"x")
            results = []
            _check_input_path(
                {"k": td}, "k", "label",
                expect_dir=True,
                extensions=["tif"],
                results=results,
            )
            assert len(results) == 1
            assert results[0].ok is True
            assert "2 fichiers" in results[0].details

    def test_dir_with_no_matching_extensions(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "file.txt").write_bytes(b"x")
            results = []
            _check_input_path(
                {"k": td}, "k", "label",
                expect_dir=True,
                extensions=["laz", "las"],
                results=results,
            )
            assert len(results) == 1
            assert results[0].ok is False
            assert "aucun fichier" in results[0].details

    def test_expect_file_existing(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            results = []
            _check_input_path(
                {"k": path}, "k", "label",
                expect_dir=False,
                results=results,
            )
            assert len(results) == 1
            assert results[0].ok is True
        finally:
            Path(path).unlink(missing_ok=True)

    def test_expect_file_nonexistent(self):
        results = []
        _check_input_path(
            {"k": "/nonexistent/file.txt"}, "k", "label",
            expect_dir=False,
            results=results,
        )
        assert len(results) == 1
        assert results[0].ok is False


class TestCollectPreflightResults:
    """La fonction pure consommée par le panneau « État du système » (étape 4)."""

    def test_returns_list_of_check_results(self):
        results = collect_preflight_results(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={},
            files_config={"existing_rvt_dir": "/nonexistent"},
            output_dir=None,
        )
        assert isinstance(results, list)
        assert results  # au moins une vérification
        assert all(isinstance(r, CheckResult) for r in results)

    def test_missing_input_dir_is_critical_failure(self):
        results = collect_preflight_results(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={},
            files_config={"existing_rvt_dir": "/nonexistent/path/xyz"},
            output_dir=None,
        )
        assert any(r.critical and not r.ok for r in results)

    def test_output_dir_reports_free_space(self):
        with tempfile.TemporaryDirectory() as td:
            results = collect_preflight_results(
                mode="existing_rvt",
                cv_config={"enabled": False},
                products={},
                files_config={"existing_rvt_dir": td},
                output_dir=Path(td),
            )
        out = [r for r in results if r.name == "Dossier de sortie"]
        assert len(out) == 1
        assert out[0].ok is True
        assert "Go libres" in out[0].details
