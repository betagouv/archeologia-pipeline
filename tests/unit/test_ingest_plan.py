from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.ingest_plan import (
    IngestValidationError,
    plan_from_specs,
    plan_raster_inputs,
)
from pipeline.tilespec import TileSpec


def _spec(name, *, crs="EPSG:2154", declared_crs=None, px=0.5, x=700000.0, y=6600000.0):
    return TileSpec.from_values(
        source_path=Path(name),
        bounds=(x, y - 500.0, x + 500.0, y),
        pixel_size_x=px,
        pixel_size_y=-px,
        width_px=1000,
        height_px=1000,
        crs=crs,
        declared_crs=declared_crs,
    )


class TestPlanFromSpecs:
    def test_uniform_projected_crs_is_mosaicable(self):
        plan = plan_from_specs([_spec("a.tif", x=700000.0), _spec("b.tif", x=700500.0)])
        assert plan.crs == "EPSG:2154"
        assert plan.mosaicable is True
        assert plan.warnings == []

    def test_single_tile_not_mosaicable(self):
        plan = plan_from_specs([_spec("only.tif")])
        assert plan.mosaicable is False

    def test_geographic_crs_rejected(self):
        with pytest.raises(IngestValidationError, match="géographique"):
            plan_from_specs([_spec("wgs.tif", crs="EPSG:4326")])

    def test_mixed_crs_rejected(self):
        with pytest.raises(IngestValidationError, match="[Mm]élange"):
            plan_from_specs([_spec("a.tif", crs="EPSG:2154"), _spec("b.tif", crs="EPSG:32631")])

    def test_absent_crs_rejected(self):
        with pytest.raises(IngestValidationError, match="introuvable"):
            plan_from_specs([_spec("nocrs.asc", crs=None)])

    def test_declared_crs_satisfies_absent(self):
        plan = plan_from_specs([_spec("nocrs.asc", crs=None, declared_crs="EPSG:2154")])
        assert plan.crs == "EPSG:2154"

    def test_mixed_resolution_warns_and_blocks_mosaic(self):
        plan = plan_from_specs([_spec("a.tif", px=0.5), _spec("b.tif", px=1.0)])
        assert plan.mosaicable is False
        assert any("hétérogènes" in w for w in plan.warnings)

    def test_model_resolution_mismatch_warns(self):
        plan = plan_from_specs([_spec("a.tif", px=1.0), _spec("b.tif", px=1.0)], model_resolution=0.5)
        assert any("entraînement du modèle" in w for w in plan.warnings)

    def test_no_tiles_raises(self):
        with pytest.raises(IngestValidationError, match="Aucune"):
            plan_from_specs([])

    def test_expected_crs_identique_ok(self):
        # GEO-02 : un CRS conforme au CRS attendu (EPSG:2154) passe.
        plan = plan_from_specs([_spec("a.tif", crs="EPSG:2154")], expected_crs="EPSG:2154")
        assert plan.crs == "EPSG:2154"

    def test_expected_crs_different_refuse(self):
        # GEO-02 : un raster projeté mais en Lambert-II (27572) est REFUSÉ
        # (sinon ses détections seraient étiquetées 2154 → mal placées).
        with pytest.raises(IngestValidationError, match="2154"):
            plan_from_specs([_spec("a.tif", crs="EPSG:27572")], expected_crs="EPSG:2154")

    def test_sans_expected_crs_comportement_inchange(self):
        # Sans expected_crs : tout CRS projeté unique reste accepté (rétrocompat).
        plan = plan_from_specs([_spec("a.tif", crs="EPSG:27572")])
        assert plan.crs == "EPSG:27572"

    def test_skipped_passed_through(self):
        plan = plan_from_specs(
            [_spec("a.tif"), _spec("b.tif")],
            skipped=[(Path("broken.tif"), "illisible/corrompu")],
        )
        assert plan.skipped == [(Path("broken.tif"), "illisible/corrompu")]
        assert "1 ignorée" in plan.summary


@pytest.fixture
def _rasterio():
    return pytest.importorskip("rasterio")


def _write_tif(rasterio, path, *, crs, nodata=None, fill=0.0, width=40, height=40,
               origin=(700000.0, 6600000.0), px=0.5):
    import numpy as np
    from rasterio.transform import from_origin

    transform = from_origin(origin[0], origin[1], px, px)
    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   dtype="float32", transform=transform)
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as ds:
        ds.write(np.full((1, height, width), fill, dtype="float32"))


class TestPlanRasterInputs:
    def test_reads_and_plans_contiguous_tiles(self, _rasterio, tmp_path):
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(_rasterio, a, crs="EPSG:2154", origin=(700000.0, 6600000.0))
        _write_tif(_rasterio, b, crs="EPSG:2154", origin=(700020.0, 6600000.0))
        plan = plan_raster_inputs([a, b])
        assert plan.crs == "EPSG:2154"
        assert plan.mosaicable is True
        assert len(plan.tiles) == 2

    def test_skips_unreadable(self, _rasterio, tmp_path):
        good = tmp_path / "good.tif"
        _write_tif(_rasterio, good, crs="EPSG:2154")
        bad = tmp_path / "bad.tif"
        bad.write_bytes(b"not a raster")
        plan = plan_raster_inputs([good, bad])
        assert len(plan.tiles) == 1
        assert any("illisible" in reason for _, reason in plan.skipped)

    def test_skips_all_nodata_tile(self, _rasterio, tmp_path):
        good = tmp_path / "good.tif"
        _write_tif(_rasterio, good, crs="EPSG:2154", nodata=-99999.0, fill=1.0)
        empty = tmp_path / "empty.tif"
        _write_tif(_rasterio, empty, crs="EPSG:2154", nodata=-99999.0, fill=-99999.0)
        plan = plan_raster_inputs([good, empty])
        assert len(plan.tiles) == 1
        assert any("vide" in reason for _, reason in plan.skipped)

    def test_asc_without_crs_uses_declared(self, _rasterio, tmp_path):
        # AAIGrid (.asc) has no CRS → declared_crs must apply.
        import numpy as np
        from rasterio.transform import from_origin

        p = tmp_path / "t.asc"
        with _rasterio.open(
            str(p), "w", driver="AAIGrid", width=40, height=40, count=1,
            dtype="float32", transform=from_origin(700000.0, 6600000.0, 0.5, 0.5),
        ) as ds:
            ds.write(np.ones((1, 40, 40), dtype="float32"))
        plan = plan_raster_inputs([p], declared_crs="EPSG:2154")
        assert plan.crs == "EPSG:2154"
        assert plan.tiles[0].crs is None  # none in metadata
