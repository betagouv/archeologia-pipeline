from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.tilespec import (
    DEGENERATE_MIN_PX,
    DEGENERATE_ORIGIN_RADIUS_M,
    TileSpec,
    _crs_from_rasterio,
    assign_crs_if_missing,
    crs_is_projected,
    disambiguate,
    is_degenerate_tile,
    make_uid,
    partition_degenerate,
    same_crs_geometry,
)


class TestFromValues:
    def _spec(self, **over):
        base = dict(
            source_path=Path("05_MNT_6240_68640.asc"),
            bounds=(624000.0, 6864000.0, 624500.0, 6864500.0),
            pixel_size_x=0.5,
            pixel_size_y=-0.5,
            width_px=1000,
            height_px=1000,
            crs="EPSG:2154",
        )
        base.update(over)
        return TileSpec.from_values(**base)

    def test_bounds_and_dimensions_in_meters(self):
        s = self._spec()
        assert s.width_m == 500.0
        assert s.height_m == 500.0

    def test_geotransform_derived_from_bounds_when_absent(self):
        s = self._spec()
        # GDAL order: (x_origin, px_w, row_rot, y_origin, col_rot, px_h)
        assert s.geotransform == (624000.0, 0.5, 0.0, 6864500.0, 0.0, -0.5)

    def test_effective_crs_prefers_real_over_declared(self):
        s = self._spec(crs="EPSG:32631", declared_crs="EPSG:2154")
        assert s.effective_crs == "EPSG:32631"

    def test_effective_crs_falls_back_to_declared_when_absent(self):
        s = self._spec(crs=None, declared_crs="EPSG:2154")
        assert s.effective_crs == "EPSG:2154"

    def test_effective_crs_none_when_both_absent(self):
        s = self._spec(crs=None, declared_crs=None)
        assert s.effective_crs is None

    def test_nodata_carried(self):
        s = self._spec(nodata=-99999.0)
        assert s.nodata == -99999.0

    def test_uid_derived_from_sanitized_stem(self):
        s = self._spec(source_path=Path("site archéo_MNT.tif"))
        # "_MNT" stripped, non-alphanumerics collapsed to "_"
        assert s.uid == "site_arch_o"

    def test_explicit_uid_respected(self):
        s = self._spec(uid="custom_id")
        assert s.uid == "custom_id"


class TestCosmeticXY:
    def test_aligned_kilometer_corner(self):
        s = TileSpec.from_values(
            source_path=Path("t.tif"),
            bounds=(700000.0, 6599000.0, 701000.0, 6600000.0),
            pixel_size_x=0.5, pixel_size_y=-0.5, width_px=2000, height_px=2000,
        )
        assert s.cosmetic_xy() == (700, 6600)

    def test_four_subkm_tiles_in_one_km_cell_share_cosmetic_xy(self):
        # The exact collision from the reported bug: cosmetic_xy snaps on the NW
        # corner (xmin, ymax). ymax = yll + 500, so the 4 tiles colliding into
        # cosmetic cell (624, 6864) have xll in {624000, 624500} and
        # yll in {6863500, 6864000} (ymax in {6864000, 6864500}, both → 6864).
        corners = [
            (624000.0, 6863500.0), (624000.0, 6864000.0),
            (624500.0, 6863500.0), (624500.0, 6864000.0),
        ]
        specs = [
            TileSpec.from_values(
                source_path=Path(f"05_MNT_{int(x / 100)}_{int(y / 100)}.asc"),
                bounds=(x, y, x + 500.0, y + 500.0),
                pixel_size_x=0.5, pixel_size_y=-0.5, width_px=1000, height_px=1000,
            )
            for x, y in corners
        ]
        # All 4 share the same cosmetic label...
        assert {s.cosmetic_xy() for s in specs} == {(624, 6864)}
        # ...but their uids are distinct, so output names will NOT collide.
        assert len({s.uid for s in specs}) == 4


class TestUidUniqueness:
    def test_make_uid_strips_mnt_and_sanitizes(self):
        assert make_uid(Path("LHD_FXX_0624_6864_MNT.tif")) == "LHD_FXX_0624_6864"

    def test_disambiguate_resolves_repeated_uids(self):
        seen: set[str] = set()
        out = [disambiguate("tile", seen) for _ in range(4)]
        assert out == ["tile", "tile_2", "tile_3", "tile_4"]
        assert len(set(out)) == 4


class _StubCRS:
    """Imite l'interface minimale d'un objet CRS rasterio."""

    def __init__(self, *, geographic, projected, authority=None, wkt="WKT"):
        self.is_geographic = geographic
        self.is_projected = projected
        self._authority = authority
        self._wkt = wkt

    def to_authority(self):
        return self._authority

    def to_wkt(self):
        return self._wkt


class TestCrsFromRasterio:
    def test_projected_with_authority_returns_authid(self):
        crs = _StubCRS(geographic=False, projected=True, authority=("EPSG", "2154"))
        assert _crs_from_rasterio(crs) == "EPSG:2154"

    def test_projected_without_authority_returns_wkt(self):
        crs = _StubCRS(geographic=False, projected=True, authority=None, wkt="PROJCS[...]")
        assert _crs_from_rasterio(crs) == "PROJCS[...]"

    def test_local_engineering_crs_treated_as_absent(self):
        # LOCAL_CS: neither geographic nor projected → must be treated as None
        crs = _StubCRS(geographic=False, projected=False, authority=None)
        assert _crs_from_rasterio(crs) is None

    def test_none_is_none(self):
        assert _crs_from_rasterio(None) is None


class TestCrsIsProjected:
    def test_projected_epsg(self):
        assert crs_is_projected("EPSG:2154") is True

    def test_geographic_epsg(self):
        assert crs_is_projected("EPSG:4326") is False

    def test_absent_is_none(self):
        assert crs_is_projected(None) is None
        assert crs_is_projected("") is None


class TestCrsIsProjectedGdalFallback:
    """Sans rasterio (cas QGIS), ``crs_is_projected`` doit basculer sur ``osgeo.osr``.

    On bloque rasterio (``sys.modules`` → ``None`` → ImportError sur l'import différé)
    et on injecte un faux ``osgeo.osr`` pour rendre le test hermétique (osgeo n'est pas
    forcément installé dans l'environnement de test standalone).
    """

    @staticmethod
    def _install(monkeypatch, *, with_osgeo=True):
        import sys
        import types

        # rasterio absent → ImportError sur ``from rasterio.crs import CRS``
        monkeypatch.setitem(sys.modules, "rasterio", None)
        monkeypatch.setitem(sys.modules, "rasterio.crs", None)

        if not with_osgeo:
            monkeypatch.setitem(sys.modules, "osgeo", None)
            monkeypatch.setitem(sys.modules, "osgeo.osr", None)
            return

        class _FakeSRS:
            def __init__(self):
                self._in = None

            def SetFromUserInput(self, s):  # 0 = OGRERR_NONE (succès)
                self._in = s
                return 0 if s in ("EPSG:2154", "EPSG:4326", "LOCAL") else 1

            def IsLocal(self):
                return 1 if self._in == "LOCAL" else 0

            def IsGeographic(self):
                return 1 if self._in == "EPSG:4326" else 0

            def IsProjected(self):
                return 1 if self._in == "EPSG:2154" else 0

        osr = types.ModuleType("osgeo.osr")
        osr.SpatialReference = _FakeSRS
        osgeo = types.ModuleType("osgeo")
        osgeo.osr = osr
        monkeypatch.setitem(sys.modules, "osgeo", osgeo)
        monkeypatch.setitem(sys.modules, "osgeo.osr", osr)

    def test_projected_via_gdal_when_rasterio_absent(self, monkeypatch):
        self._install(monkeypatch)
        assert crs_is_projected("EPSG:2154") is True

    def test_geographic_via_gdal(self, monkeypatch):
        self._install(monkeypatch)
        assert crs_is_projected("EPSG:4326") is False

    def test_local_via_gdal_is_none(self, monkeypatch):
        self._install(monkeypatch)
        assert crs_is_projected("LOCAL") is None

    def test_none_when_no_backend_available(self, monkeypatch):
        self._install(monkeypatch, with_osgeo=False)
        assert crs_is_projected("EPSG:2154") is None


@pytest.fixture
def _rasterio():
    return pytest.importorskip("rasterio")


def _write_tif(rasterio, path, *, crs, width=20, height=10,
               origin=(700000.0, 6600000.0), px=0.5):
    """Écrit un GeoTIFF minimal ; ``crs=None`` → raster sans CRS."""
    import numpy as np
    from rasterio.transform import from_origin

    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="float32",
        transform=from_origin(origin[0], origin[1], px, px),
    )
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(str(path), "w", **profile) as ds:
        ds.write(np.zeros((1, height, width), dtype="float32"))


class TestAssignCrsIfMissing:
    """Tampon CRS : affecte EPSG:2154 (sans reprojeter) si le raster n'a pas de
    CRS exploitable — corrige les MNT que PDAL émet en CRS « unnamed »."""

    def test_stamps_epsg2154_when_crs_absent(self, _rasterio, tmp_path):
        p = tmp_path / "nocrs.tif"
        _write_tif(_rasterio, p, crs=None)
        assert TileSpec.from_raster(p).crs is None  # pré-condition : pas de CRS
        assert assign_crs_if_missing(p) == "EPSG:2154"
        assert TileSpec.from_raster(p).crs == "EPSG:2154"

    def test_noop_when_real_crs_present(self, _rasterio, tmp_path):
        p = tmp_path / "withcrs.tif"
        _write_tif(_rasterio, p, crs="EPSG:2154")
        assert assign_crs_if_missing(p) is None  # rien à faire
        assert TileSpec.from_raster(p).crs == "EPSG:2154"

    def test_does_not_clobber_other_projected_crs(self, _rasterio, tmp_path):
        p = tmp_path / "utm.tif"
        _write_tif(_rasterio, p, crs="EPSG:32631")
        assert assign_crs_if_missing(p) is None
        assert TileSpec.from_raster(p).crs == "EPSG:32631"

    def test_custom_fallback_authid(self, _rasterio, tmp_path):
        p = tmp_path / "nocrs2.tif"
        _write_tif(_rasterio, p, crs=None)
        assert assign_crs_if_missing(p, fallback_authid="EPSG:32631") == "EPSG:32631"
        assert TileSpec.from_raster(p).crs == "EPSG:32631"

    def test_missing_file_returns_none(self, tmp_path):
        assert assign_crs_if_missing(tmp_path / "nope.tif") is None


class TestFromRaster:
    def _write_tif(self, rasterio, path, *, crs, nodata=None, width=20, height=10,
                   origin=(700000.0, 6600000.0), px=0.5):
        import numpy as np
        from rasterio.transform import from_origin

        transform = from_origin(origin[0], origin[1], px, px)
        profile = dict(
            driver="GTiff", width=width, height=height, count=1, dtype="float32",
            transform=transform,
        )
        if crs is not None:
            profile["crs"] = crs
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(str(path), "w", **profile) as ds:
            ds.write(np.zeros((1, height, width), dtype="float32"))

    def test_reads_projected_crs_bounds_and_nodata(self, _rasterio, tmp_path):
        p = tmp_path / "tile.tif"
        self._write_tif(_rasterio, p, crs="EPSG:2154", nodata=-99999.0,
                        width=20, height=10, origin=(700000.0, 6600000.0), px=0.5)
        spec = TileSpec.from_raster(p)
        assert spec is not None
        assert spec.crs == "EPSG:2154"
        assert spec.effective_crs == "EPSG:2154"
        assert spec.nodata == -99999.0
        assert spec.width_px == 20 and spec.height_px == 10
        # 20 px * 0.5 m = 10 m wide ; 10 px * 0.5 m = 5 m tall
        assert spec.width_m == 10.0 and spec.height_m == 5.0
        assert spec.bounds == (700000.0, 6599995.0, 700010.0, 6600000.0)

    def test_non_2154_crs_carried_without_reprojection(self, _rasterio, tmp_path):
        p = tmp_path / "utm.tif"
        self._write_tif(_rasterio, p, crs="EPSG:32631", origin=(500000.0, 5400000.0))
        spec = TileSpec.from_raster(p)
        assert spec is not None
        assert spec.crs == "EPSG:32631"

    def test_crs_less_raster_falls_back_to_declared(self, _rasterio, tmp_path):
        p = tmp_path / "nocrs.tif"
        self._write_tif(_rasterio, p, crs=None)
        spec = TileSpec.from_raster(p, declared_crs="EPSG:2154")
        assert spec is not None
        assert spec.crs is None
        assert spec.effective_crs == "EPSG:2154"

    def test_metadata_wins_when_filename_disagrees(self, _rasterio, tmp_path):
        # Filename claims one km cell, geotransform says another → placement must
        # follow the geotransform, never the name.
        p = tmp_path / "LHD_FXX_0123_4567_MNT.tif"
        self._write_tif(_rasterio, p, crs="EPSG:2154", origin=(840000.0, 6520000.0))
        spec = TileSpec.from_raster(p)
        assert spec is not None
        assert spec.cosmetic_xy() == (840, 6520)  # from bounds, not "0123_4567"

    def test_unreadable_file_returns_none(self, _rasterio, tmp_path):
        p = tmp_path / "broken.tif"
        p.write_bytes(b"not a raster")
        assert TileSpec.from_raster(p) is None


def _deg_spec(**over):
    """TileSpec helper for degeneracy tests (a normal 1 km tile by default)."""
    base = dict(
        source_path=Path("FONT_654500_6814500.tif"),
        bounds=(654000.0, 6808000.0, 654500.0, 6808500.0),
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        width_px=500,
        height_px=500,
        crs="EPSG:2154",
    )
    base.update(over)
    return TileSpec.from_values(**base)


class TestIsDegenerateTile:
    """Détecte les dalles « placeholder » (1×1 px / posées à l'origine du CRS) qui,
    incluses dans un gdalbuildvrt, gonflent l'emprise de la mosaïque jusqu'à (0,0)."""

    def test_normal_tile_is_not_degenerate(self):
        assert is_degenerate_tile(_deg_spec()) is False

    def test_one_by_one_pixel_tile_is_degenerate(self):
        # Signal A : taille triviale (le cas exact des placeholders FONT_*).
        s = _deg_spec(width_px=1, height_px=1,
                      bounds=(654000.0, 6808499.0, 654001.0, 6808500.0))
        assert is_degenerate_tile(s) is True

    def test_zero_pixel_tile_is_degenerate(self):
        assert is_degenerate_tile(_deg_spec(width_px=0)) is True

    def test_origin_at_crs_zero_is_degenerate(self):
        # Signal B : coin NO ≈ (0,0) même avec des dimensions « normales »
        # (le placeholder non carré qui n'est pas attrapé par la taille seule).
        s = _deg_spec(width_px=500, height_px=500,
                      bounds=(-0.5, -500.5, 499.5, 0.5))
        assert is_degenerate_tile(s) is True

    def test_slightly_off_grid_valid_tile_is_not_degenerate(self):
        s = _deg_spec(bounds=(654001.3, 6789500.7, 654501.3, 6790000.7))
        assert is_degenerate_tile(s) is False

    def test_thresholds_are_parameterizable(self):
        s = _deg_spec(width_px=3, height_px=3,
                      bounds=(654000.0, 6808497.0, 654003.0, 6808500.0))
        assert is_degenerate_tile(s) is False
        assert is_degenerate_tile(s, min_px=4) is True

    def test_default_thresholds_exposed(self):
        assert DEGENERATE_MIN_PX == 2
        assert DEGENERATE_ORIGIN_RADIUS_M == 10.0


class TestPartitionDegenerate:
    def test_splits_and_preserves_order(self):
        good_a = _deg_spec(source_path=Path("a.tif"))
        bad = _deg_spec(source_path=Path("placeholder.tif"), width_px=1, height_px=1,
                        bounds=(-0.5, -0.5, 0.5, 0.5))
        good_b = _deg_spec(source_path=Path("b.tif"), bounds=(655000.0, 6808000.0, 655500.0, 6808500.0))
        kept, degenerate = partition_degenerate([good_a, bad, good_b])
        assert [s.source_path.name for s in kept] == ["a.tif", "b.tif"]
        assert [s.source_path.name for s in degenerate] == ["placeholder.tif"]

    def test_all_good_returns_empty_degenerate(self):
        kept, degenerate = partition_degenerate([_deg_spec()])
        assert len(kept) == 1
        assert degenerate == []


class TestSameCrsGeometry:
    """Compare le PLACEMENT (où atterrit un point lon/lat), pas l'identité stricte du
    datum — pour reconnaître un Lambert-93 custom (WKT sans code EPSG) comme EPSG:2154."""

    def test_none_for_empty_inputs(self):
        assert same_crs_geometry(None, "EPSG:2154") is None
        assert same_crs_geometry("", "EPSG:2154") is None
        assert same_crs_geometry("EPSG:2154", None) is None

    def test_identical_epsg_is_same(self):
        pytest.importorskip("osgeo")
        assert same_crs_geometry("EPSG:2154", "EPSG:2154") is True

    def test_different_projection_is_not_same(self):
        pytest.importorskip("osgeo")
        assert same_crs_geometry("EPSG:2154", "EPSG:32631") is False

    def test_custom_lambert_wkt_equals_epsg2154(self):
        # Le SRC exact du fichier de l'utilisateur : Lambert custom, datum « unnamed »
        # GRS80, paramètres = EPSG:2154, SANS code d'autorité → doit être reconnu == 2154.
        pytest.importorskip("osgeo")
        wkt = (
            'PROJCS["Lambert Conformal Conic",'
            'GEOGCS["grs80",DATUM["unnamed",SPHEROID["unnamed",6378137,298.257222101004]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic_2SP"],'
            'PARAMETER["latitude_of_origin",46.5],PARAMETER["central_meridian",3],'
            'PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",44],'
            'PARAMETER["false_easting",700000],PARAMETER["false_northing",6600000],'
            'UNIT["metre",1]]'
        )
        assert same_crs_geometry(wkt, "EPSG:2154") is True
