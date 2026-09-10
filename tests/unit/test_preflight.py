from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pipeline.preflight import (
    CheckResult,
    _check_input_path,
    _check_raster_crs,
    collect_preflight_results,
)


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

    def test_existing_mnt_accepts_asc_only_dir(self):
        """Un dossier MNT ne contenant que des .asc doit passer le preflight
        (le runtime existing_mnt convertit les .asc en TIF via gdal_translate)."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "mnt.asc").write_bytes(b"x")
            results = collect_preflight_results(
                mode="existing_mnt",
                cv_config={"enabled": False},
                products={},
                files_config={"existing_mnt_dir": td},
                output_dir=Path(td),
            )
        mnt = [r for r in results if r.name == "Dossier MNT existants"]
        assert len(mnt) == 1
        assert mnt[0].ok is True

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


class TestRvtAlgosCheck:
    """« Algorithmes RVT (Processing) » : vérification RÉELLE du registre.

    Bug SRA HDF 2026-08-31 : avec un rvt-qgis absent/trop vieux (non chargé
    par QGIS 4), l'ancien check supposait la disponibilité (« expected
    available in QGIS », critical=False) et le run mourait au premier
    ``processing.run("rvt:...")`` avec un « Algorithm not found » opaque.
    """

    NAME = "Algorithmes RVT (Processing)"

    @staticmethod
    def _install_fake_qgis(monkeypatch, *, algos, has_provider=True):
        import sys
        import types

        class _FakeRegistry:
            def algorithmById(self, algo_id):
                return object() if algo_id in algos else None

            def providerById(self, provider_id):
                return object() if (provider_id == "rvt" and has_provider) else None

        class _FakeQgsApplication:
            @staticmethod
            def processingRegistry():
                return _FakeRegistry()

        core = types.ModuleType("qgis.core")
        core.QgsApplication = _FakeQgsApplication
        qgis = types.ModuleType("qgis")
        qgis.core = core
        monkeypatch.setitem(sys.modules, "qgis", qgis)
        monkeypatch.setitem(sys.modules, "qgis.core", core)

    def _collect(self, products):
        return collect_preflight_results(
            mode="ign_laz",
            cv_config={"enabled": False},
            products=products,
            files_config={},
            output_dir=None,
        )

    def _check(self, results):
        found = [r for r in results if r.name == self.NAME]
        assert len(found) == 1
        return found[0]

    def test_no_rvt_product_no_check(self):
        results = self._collect({"MNT": True, "DENSITE": True})
        assert not any(r.name == self.NAME for r in results)

    def test_outside_qgis_is_non_blocking(self):
        # Standalone (pas de module qgis) : non vérifiable → on ne bloque pas.
        r = self._check(self._collect({"LD": True}))
        assert r.ok is True
        assert r.critical is False
        assert "non vérifiable" in r.details

    def test_all_algos_present_is_green_and_critical(self, monkeypatch):
        self._install_fake_qgis(monkeypatch, algos={"rvt:rvt_ld", "rvt:rvt_svf"})
        r = self._check(self._collect({"LD": True, "SVF": True}))
        assert r.ok is True
        assert r.critical is True

    def test_provider_absent_is_critical_with_actionable_message(self, monkeypatch):
        self._install_fake_qgis(monkeypatch, algos=set(), has_provider=False)
        r = self._check(self._collect({"LD": True}))
        assert r.ok is False
        assert r.critical is True
        assert "Relief Visualization Toolbox" in r.details
        assert "Extensions" in r.details

    def test_provider_too_old_names_missing_algos(self, monkeypatch):
        # rvt-qgis chargé mais sans rvt_mstp (version trop ancienne)
        self._install_fake_qgis(monkeypatch, algos={"rvt:rvt_ld"}, has_provider=True)
        r = self._check(self._collect({"LD": True, "MSTP": True}))
        assert r.ok is False
        assert r.critical is True
        assert "rvt_mstp" in r.details
        assert "jour" in r.details  # « mettez à jour »

    def test_existing_rvt_mode_skips_check(self, monkeypatch):
        self._install_fake_qgis(monkeypatch, algos=set(), has_provider=False)
        results = collect_preflight_results(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={"LD": True},
            files_config={},
            output_dir=None,
        )
        assert not any(r.name == self.NAME for r in results)


class TestCheckRasterCrs:
    """« CRS des rasters » : vérifié → vert bloquant ; invérifiable → ⚠ non bloquant."""

    def _fake_plan(self, **over):
        from pipeline.ingest_plan import IngestPlan

        base = dict(
            tiles=[object(), object()], crs="EPSG:2154", mosaicable=True,
            skipped=[], warnings=[], crs_verified=True,
        )
        base.update(over)
        return IngestPlan(**base)

    def test_verified_crs_is_green_and_critical(self, monkeypatch, tmp_path):
        (tmp_path / "mnt.tif").write_bytes(b"x")
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "plan_raster_inputs", lambda *a, **k: self._fake_plan())
        results = []
        _check_raster_crs(str(tmp_path), ("tif",), results)
        assert len(results) == 1
        assert results[0].ok is True
        assert results[0].critical is True
        assert "EPSG:2154" in results[0].details

    def test_unverifiable_crs_warns_without_blocking(self, monkeypatch, tmp_path):
        (tmp_path / "mnt.tif").write_bytes(b"x")
        import pipeline.ingest_plan as ip
        plan = self._fake_plan(
            crs='PROJCS["x"]', mosaicable=False, crs_verified=False,
            warnings=["CRS « PROJCS[...] » non vérifiable (aucun backend) — vérifiez EPSG:2154."],
        )
        monkeypatch.setattr(ip, "plan_raster_inputs", lambda *a, **k: plan)
        results = []
        _check_raster_crs(str(tmp_path), ("tif",), results)
        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].critical is False  # ⚠ n'empêche pas le lancement
        assert "vérifiable" in results[0].details

    def test_degenerate_tiles_shown_as_noncritical(self, monkeypatch, tmp_path):
        (tmp_path / "mnt.tif").write_bytes(b"x")
        import pipeline.ingest_plan as ip
        from pipeline.ingest_plan import DEGENERATE_SKIP_REASON
        plan = self._fake_plan(skipped=[
            (Path("p1.tif"), DEGENERATE_SKIP_REASON),
            (Path("p2.tif"), DEGENERATE_SKIP_REASON),
            (Path("broken.tif"), "illisible/corrompu"),  # ne doit PAS être compté
        ])
        monkeypatch.setattr(ip, "plan_raster_inputs", lambda *a, **k: plan)
        results = []
        _check_raster_crs(str(tmp_path), ("tif",), results)
        deg = [r for r in results if r.name == "Dalles dégénérées"]
        assert len(deg) == 1
        assert deg[0].ok is False
        assert deg[0].critical is False  # ⚠ n'empêche pas le lancement
        assert "2" in deg[0].details

    def test_no_degenerate_result_when_none(self, monkeypatch, tmp_path):
        (tmp_path / "mnt.tif").write_bytes(b"x")
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "plan_raster_inputs", lambda *a, **k: self._fake_plan())
        results = []
        _check_raster_crs(str(tmp_path), ("tif",), results)
        assert not any(r.name == "Dalles dégénérées" for r in results)
