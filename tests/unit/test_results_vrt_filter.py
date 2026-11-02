"""Robustesse VRT : ``build_vrt_index`` ne doit pas mosaïquer les dalles dégénérées
(placeholder 1×1 / origine 0) — sinon ``gdalbuildvrt`` étire l'emprise jusqu'à (0,0).

On teste les deux helpers PURS (sélection des entrées + contrôle d'emprise) ; l'appel
``gdalbuildvrt`` lui-même reste hors test unitaire (déjà stubé ailleurs).
"""
from __future__ import annotations

from pathlib import Path

from pipeline.ign.products.results import _select_vrt_inputs, _vrt_bounds_look_suspect
from pipeline.tilespec import TileSpec


def _spec(path, **over):
    base = dict(
        source_path=Path(path),
        bounds=(654000.0, 6808000.0, 654500.0, 6808500.0),
        pixel_size_x=1.0, pixel_size_y=-1.0,
        width_px=500, height_px=500, crs="EPSG:2154",
    )
    base.update(over)
    return TileSpec.from_values(**base)


class TestSelectVrtInputs:
    def test_excludes_degenerate_keeps_normal_and_unreadable(self):
        files = [Path("good.tif"), Path("placeholder.tif"), Path("broken.tif"), Path("good2.tif")]

        def read_spec(f):
            if f.name == "placeholder.tif":
                return _spec(f, width_px=1, height_px=1, bounds=(-0.5, -0.5, 0.5, 0.5))
            if f.name == "broken.tif":
                return None  # illisible → conservé (conservateur)
            return _spec(f)

        kept, dropped = _select_vrt_inputs(files, read_spec=read_spec)
        assert [f.name for f in kept] == ["good.tif", "broken.tif", "good2.tif"]
        assert [f.name for f in dropped] == ["placeholder.tif"]

    def test_all_normal_drops_nothing(self):
        files = [Path("a.tif"), Path("b.tif")]
        kept, dropped = _select_vrt_inputs(files, read_spec=lambda f: _spec(f))
        assert kept == files
        assert dropped == []


class TestVrtBoundsLookSuspect:
    def test_normal_mosaic_not_suspect(self):
        assert _vrt_bounds_look_suspect((654000.0, 6790000.0, 681000.0, 6814500.0)) is False

    def test_origin_touching_mosaic_is_suspect(self):
        # Le cas réel : emprise polluée s'étendant jusqu'à (≈0, ≈0).
        assert _vrt_bounds_look_suspect((-0.5, -0.44, 681000.0, 6824248.0)) is True

    def test_implausible_span_is_suspect(self):
        # Une dalle posée à des milliers de km du lot (géoréférencement faux) : 2 500 km d'emprise.
        assert _vrt_bounds_look_suspect((654000.0, 6790000.0, 3_154_000.0, 6814500.0)) is True

    def test_lot_national_hors_france_pas_suspect(self):
        # Run flamand du 2026-11-01 (EPSG:31370) : 202 km de fenêtres dispersées, alerte fausse au seuil de 200 km.
        assert _vrt_bounds_look_suspect((21_000.0, 153_000.0, 223_054.0, 226_880.0)) is False
        # La France entière en Lambert-93, la Suède en SWEREF99 TM (≈ 1 600 km nord-sud) : légitimes aussi.
        assert _vrt_bounds_look_suspect((100_000.0, 6_040_000.0, 1_250_000.0, 7_120_000.0)) is False
        assert _vrt_bounds_look_suspect((260_000.0, 6_130_000.0, 920_000.0, 7_680_000.0)) is False

    def test_none_is_not_suspect(self):
        assert _vrt_bounds_look_suspect(None) is False


def test_vrt_index_a_la_resolution_la_plus_fine(monkeypatch, tmp_path):
    """Audit Fable 2026-11-02 (A4, #29) : gdalbuildvrt sans -resolution prend la MOYENNE des pas ; une relance à
    une autre résolution dans le même dossier mêle des dalles _0M50_ et _1M00_ → -resolution highest."""
    import subprocess

    import pipeline.ign.products.results as results

    for nom in ("a.tif", "b.tif"):
        (tmp_path / nom).write_bytes(b"x")
    vus = []

    class _R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(results.shutil, "which", lambda _n: "gdalbuildvrt")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: vus.append(cmd) or _R())
    monkeypatch.setattr(results, "get_raster_bounds", lambda _p: None)
    assert results.build_vrt_index(tmp_path, exclude_degenerate=False) is True
    (cmd,) = vus
    i = cmd.index("-resolution")
    assert cmd[i + 1] == "highest"
