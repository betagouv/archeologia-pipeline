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
        assert _vrt_bounds_look_suspect((654000.0, 6790000.0, 1_000_000.0, 6814500.0)) is True

    def test_none_is_not_suspect(self):
        assert _vrt_bounds_look_suspect(None) is False
