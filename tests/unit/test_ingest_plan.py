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


def _degenerate_spec(name="placeholder.tif", *, crs="EPSG:2154"):
    """Dalle « placeholder » 1×1 px posée à l'origine du CRS (cas réel FONT_*)."""
    return TileSpec.from_values(
        source_path=Path(name),
        bounds=(-0.5, -0.5, 0.5, 0.5),
        pixel_size_x=1.0, pixel_size_y=-1.0,
        width_px=1, height_px=1, crs=crs,
    )


class TestDegenerateTileFiltering:
    """Les dalles dégénérées (1×1 / origine 0) sont retirées AVANT les contrôles CRS
    et signalées une fois (sinon elles gonflent l'emprise du VRT jusqu'à (0,0))."""

    def test_degenerate_moved_to_skipped_and_warned(self):
        plan = plan_from_specs([
            _spec("a.tif"), _spec("b.tif", x=700500.0),
            _degenerate_spec("p1.tif"), _degenerate_spec("p2.tif"),
        ])
        assert len(plan.tiles) == 2
        names = {p.name for p, _ in plan.skipped}
        assert names == {"p1.tif", "p2.tif"}
        assert all("dégénér" in reason for _, reason in plan.skipped)
        assert sum("dégénér" in w for w in plan.warnings) == 1

    def test_all_degenerate_raises_aucune(self):
        with pytest.raises(IngestValidationError, match="Aucune"):
            plan_from_specs([_degenerate_spec("p1.tif"), _degenerate_spec("p2.tif")])

    def test_degenerate_does_not_trip_crs_error(self):
        # Un placeholder sans CRS ne doit PAS déclencher l'erreur « CRS introuvable » :
        # il est filtré avant le contrôle CRS.
        plan = plan_from_specs([
            _spec("good.tif", crs="EPSG:2154"),
            _degenerate_spec("nocrs_placeholder.tif", crs=None),
        ])
        assert len(plan.tiles) == 1
        assert plan.crs == "EPSG:2154"


class TestCrsVerificationPolicy:
    """Garde-fou EPSG + politique « avertir sans bloquer » quand le CRS n'est pas
    classable (ex. backend de lecture indisponible dans QGIS)."""

    def test_crs_verified_true_by_default(self):
        plan = plan_from_specs([_spec("a.tif", crs="EPSG:2154")], expected_crs="EPSG:2154")
        assert plan.crs_verified is True

    def test_garde_fou_accepts_when_unclassifiable_but_epsg_matches(self, monkeypatch):
        # Aucun backend (crs_is_projected → None) mais code EPSG == attendu → accepté.
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: None)
        plan = plan_from_specs([_spec("a.tif", crs="EPSG:2154")], expected_crs="EPSG:2154")
        assert plan.crs == "EPSG:2154"
        assert plan.crs_verified is True
        assert not any("vérifiable" in w for w in plan.warnings)

    def test_unverifiable_crs_warns_not_blocks(self, monkeypatch):
        # CRS en WKT (pas de code EPSG) + aucun backend → on AVERTIT, on ne bloque pas.
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: None)
        wkt = 'PROJCS["unknown",GEOGCS["x"]]'
        plan = plan_from_specs([_spec("a.tif", crs=wkt)], expected_crs="EPSG:2154")
        assert plan.crs == wkt
        assert plan.crs_verified is False
        assert any("vérifiable" in w for w in plan.warnings)

    def test_unverifiable_without_expected_crs_warns(self, monkeypatch):
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: None)
        wkt = 'PROJCS["unknown",GEOGCS["x"]]'
        plan = plan_from_specs([_spec("a.tif", crs=wkt)])
        assert plan.crs_verified is False
        assert any("vérifiable" in w for w in plan.warnings)

    def test_geographic_still_hard_error(self, monkeypatch):
        # crs_is_projected → False doit toujours LEVER (erreur dure, pas un warn).
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: False)
        with pytest.raises(IngestValidationError, match="géographique"):
            plan_from_specs([_spec("a.tif", crs="EPSG:4326")])

    def test_custom_wkt_geometrically_2154_accepted_without_warning(self, monkeypatch):
        # Garde-fou : un Lambert custom (WKT sans code EPSG) reconnu géométriquement
        # == EPSG:2154 est accepté SANS avertissement (cas exact de l'utilisateur).
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: True)
        monkeypatch.setattr(ip, "same_crs_geometry", lambda a, b: True)
        wkt = 'PROJCS["Lambert Conformal Conic",GEOGCS["grs80"]]'
        plan = plan_from_specs([_spec("a.tif", crs=wkt)], expected_crs="EPSG:2154")
        assert plan.crs == wkt
        assert plan.crs_verified is True
        assert not any("mal plac" in w for w in plan.warnings)

    def test_different_projected_wkt_warns_not_blocks(self, monkeypatch):
        # Garde-fou : un AUTRE CRS projeté en WKT (pas de code EPSG comparable, mais
        # géométriquement ≠ 2154) → on AVERTIT sans bloquer (l'utilisateur a choisi
        # « avertir, ne pas forcer »).
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: True)
        monkeypatch.setattr(ip, "same_crs_geometry", lambda a, b: False)
        wkt = 'PROJCS["WGS 84 / UTM zone 31N",GEOGCS["x"]]'
        plan = plan_from_specs([_spec("a.tif", crs=wkt)], expected_crs="EPSG:2154")
        assert plan.crs == wkt
        assert plan.crs_verified is False
        assert any("mal plac" in w for w in plan.warnings)

    def test_indeterminate_geometry_no_mismatch_warning(self, monkeypatch):
        # Géométrie indéterminable (aucun backend) → pas d'avertissement de
        # non-correspondance (on n'accuse pas sans preuve).
        import pipeline.ingest_plan as ip
        monkeypatch.setattr(ip, "crs_is_projected", lambda c: True)
        monkeypatch.setattr(ip, "same_crs_geometry", lambda a, b: None)
        wkt = 'PROJCS["custom",GEOGCS["x"]]'
        plan = plan_from_specs([_spec("a.tif", crs=wkt)], expected_crs="EPSG:2154")
        assert plan.crs_verified is True
        assert not any("mal plac" in w for w in plan.warnings)


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
