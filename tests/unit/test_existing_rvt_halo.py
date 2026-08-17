"""Option B (halo inter-dalles) : run_existing_rvt et la source d'inférence.

Quand un ``inference_tif_resolver`` est fourni (mode ign_laz/local_laz),
le PNG d'inférence, la garde GEO-03 et le géotransform doivent tous les
trois venir du TIF résolu (non rogné, avec marge) — jamais un panachage
avec le TIF rogné, sous peine de détections décalées de la marge (GEO-03).
Le nom du PNG reste celui du TIF rogné (stems stables : cache, couches,
images annotées).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

import pipeline.modes.existing_rvt as er
import pipeline.cv.runner as cv_runner


_CROPPED = "LHD_FXX_0872_6904_LD_A15_Rmin10_Rmax20_H1p7_V1_A_LAMB93.tif"
_UNCROPPED = "LHD_FXX_0872_6904_PTS_C_LAMB93_IGN69_LD_A15_Rmin10_Rmax20_H1p7_V1.tif"

# Origines distinctes : le TIF non rogné démarre 200 m plus à l'ouest/nord.
_TRANSFORM_UNCROPPED = (0.5, -0.5, 871800.0, 6904200.0)
_TRANSFORM_CROPPED = (0.5, -0.5, 872000.0, 6904000.0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    tif_dir = tmp_path / "indices" / "LD_TEST" / "tif"
    tif_dir.mkdir(parents=True)
    cropped = tif_dir / _CROPPED
    cropped.write_bytes(b"tif")
    uncropped = tmp_path / "intermediaires" / _UNCROPPED
    uncropped.parent.mkdir()
    uncropped.write_bytes(b"tif")

    calls = {"convert": [], "consistent": [], "cv": []}

    def fake_convert(src, dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"png")
        calls["convert"].append((Path(src), Path(dst)))

    def fake_transform(path):
        if Path(path).name == _UNCROPPED:
            return _TRANSFORM_UNCROPPED
        return _TRANSFORM_CROPPED

    def fake_cv(**kwargs):
        calls["cv"].append(kwargs)

    monkeypatch.setattr(er, "_convert_tif_to_png_with_world", fake_convert)
    monkeypatch.setattr(er, "extract_tif_transform_data", fake_transform)
    monkeypatch.setattr(cv_runner, "run_cv_on_folder", fake_cv)

    return {
        "output_dir": tmp_path,
        "tif_dir": tif_dir,
        "cropped": cropped,
        "uncropped": uncropped,
        "calls": calls,
    }


def _run(env_data, **kwargs):
    return er.run_existing_rvt(
        existing_rvt_dir=env_data["tif_dir"],
        output_dir=env_data["output_dir"],
        cv_config={"enabled": True},
        output_structure={},
        indices_folder_name="LD_TEST",
        **kwargs,
    )


class TestInferenceTifResolver:
    def test_png_converti_depuis_le_tif_resolu(self, env):
        _run(env, inference_tif_resolver=lambda cropped: env["uncropped"])

        assert env["calls"]["convert"], "aucune conversion PNG"
        src, dst = env["calls"]["convert"][0]
        assert src == env["uncropped"]
        # Le nom du PNG reste celui du TIF rogné (stems stables).
        assert dst.stem == env["cropped"].stem

    def test_transform_extrait_du_tif_resolu(self, env):
        _run(env, inference_tif_resolver=lambda cropped: env["uncropped"])

        assert env["calls"]["cv"], "run_cv_on_folder non appelé"
        transforms = env["calls"]["cv"][0]["tif_transform_data"]
        assert transforms[env["cropped"].stem] == _TRANSFORM_UNCROPPED

    def test_geo03_compare_le_png_au_tif_resolu(self, env, monkeypatch):
        # Un PNG rogné préexistant (2000 px) doit être détecté incohérent
        # avec le TIF résolu (2800 px) et régénéré depuis ce dernier.
        jpg_dir = env["output_dir"] / "indices" / "LD_TEST" / "png"
        jpg_dir.mkdir(parents=True)
        stale = jpg_dir / (env["cropped"].stem + ".png")
        stale.write_bytes(b"old png")

        def fake_consistent(png, tif, **kw):
            env["calls"]["consistent"].append((Path(png), Path(tif)))
            return False

        monkeypatch.setattr(er, "_png_consistent_with_tif", fake_consistent)

        _run(env, inference_tif_resolver=lambda cropped: env["uncropped"])

        assert env["calls"]["consistent"][0][1] == env["uncropped"]
        assert env["calls"]["convert"][0][0] == env["uncropped"]

    def test_resolveur_none_retombe_sur_le_tif_rogne(self, env):
        _run(env, inference_tif_resolver=lambda cropped: None)

        src, _dst = env["calls"]["convert"][0]
        assert src == env["cropped"]
        transforms = env["calls"]["cv"][0]["tif_transform_data"]
        assert transforms[env["cropped"].stem] == _TRANSFORM_CROPPED

    def test_sans_resolveur_comportement_historique(self, env):
        _run(env)

        src, _dst = env["calls"]["convert"][0]
        assert src == env["cropped"]

    def test_region_valide_transmise_avec_halo(self, env, monkeypatch):
        # Le clip aval (bruit du halo extérieur) reçoit l'union des emprises
        # des TIF ROGNÉS = la donnée réellement commandée par le run.
        cell = (872000.0, 6903000.0, 873000.0, 6904000.0)
        monkeypatch.setattr(er, "get_raster_bounds", lambda p: cell)

        _run(env, inference_tif_resolver=lambda cropped: env["uncropped"])

        assert env["calls"]["cv"][0]["valid_region_bounds"] == [cell]

    def test_region_valide_none_si_emprise_illisible(self, env, monkeypatch):
        # Région incomplète = clip faux (il couperait les dalles illisibles) :
        # on désactive le clip plutôt que de clipper partiellement.
        monkeypatch.setattr(er, "get_raster_bounds", lambda p: None)

        _run(env, inference_tif_resolver=lambda cropped: env["uncropped"])

        assert env["calls"]["cv"][0]["valid_region_bounds"] is None

    def test_region_valide_none_sans_halo(self, env):
        _run(env)

        assert env["calls"]["cv"][0]["valid_region_bounds"] is None


class _Stub:
    """Encaisse tout appel de méthode (reporter, narrator) sans effet."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class TestRunCvPostLoopHalo:
    """run_cv_post_loop branche le résolveur halo quand halo_source_dir est fourni."""

    @pytest.fixture
    def loop_env(self, tmp_path, monkeypatch):
        import app.services.cv_post_service as cps
        import pipeline.output_paths as op

        tif_dir = tmp_path / "indices" / "LD_X" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / _CROPPED).write_bytes(b"t")
        monkeypatch.setattr(op, "resolve_rvt_tif_dir", lambda *a, **k: tif_dir)

        captured = {}

        def fake_run_existing_rvt(**kwargs):
            captured.update(kwargs)
            return er.ExistingRvtResult(total_images=0, total_detections=0)

        monkeypatch.setattr(er, "run_existing_rvt", fake_run_existing_rvt)
        monkeypatch.setattr(cps, "create_user_narrator", lambda r: _Stub())
        monkeypatch.setattr(cps, "log_section", lambda *a, **k: None)
        monkeypatch.setattr(cps, "report_stage_id", lambda *a, **k: None)
        monkeypatch.setattr(cps, "report_busy", lambda *a, **k: None)

        from types import SimpleNamespace
        ctx = SimpleNamespace(
            cv=SimpleNamespace(raw={"selected_model": "fake_model", "target_rvt": "LD"}),
            output_dir=tmp_path,
        )
        cancel = SimpleNamespace(is_cancelled=lambda: False)
        return {"cps": cps, "ctx": ctx, "cancel": cancel, "captured": captured}

    def test_halo_source_dir_construit_le_resolveur(self, loop_env, tmp_path, monkeypatch):
        cps = loop_env["cps"]
        sentinel = tmp_path / "resolved.tif"
        rec = {}

        def fake_resolve(cropped, temp_dir, target_rvt, rvt_params, **kw):
            rec["args"] = (cropped, temp_dir, target_rvt, rvt_params)
            return sentinel

        monkeypatch.setattr(cps, "resolve_uncropped_tif", fake_resolve)

        cps.run_cv_post_loop(
            ctx=loop_env["ctx"], output_structure={}, rvt_params={"svf": {"radius": 12}},
            reporter=_Stub(), cancel=loop_env["cancel"], slog=None,
            halo_source_dir=tmp_path / "intermediaires",
        )

        resolver = loop_env["captured"].get("inference_tif_resolver")
        assert resolver is not None
        assert resolver(Path("a.tif")) == sentinel
        assert rec["args"] == (
            Path("a.tif"), tmp_path / "intermediaires", "LD", {"svf": {"radius": 12}},
        )

    def test_sans_halo_source_dir_pas_de_resolveur(self, loop_env):
        cps = loop_env["cps"]

        cps.run_cv_post_loop(
            ctx=loop_env["ctx"], output_structure={}, rvt_params={},
            reporter=_Stub(), cancel=loop_env["cancel"], slog=None,
        )

        assert loop_env["captured"], "run_existing_rvt non appelé"
        assert loop_env["captured"].get("inference_tif_resolver") is None
