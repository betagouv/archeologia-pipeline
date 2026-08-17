"""GEO-03 (audit v2) : invariant « le PNG d'inférence vient du même raster que
le transform ». Si un PNG préexistant (ex. export d'indices non rogné, 2200 px)
ne correspond pas aux dimensions du TIF rogné (2000 px), il doit être régénéré
— sinon les détections sont décalées de la marge de dalle (~200 m).
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.modes.existing_rvt import _png_consistent_with_tif, _tif_size


class TestPngConsistentWithTif:
    def test_dimensions_egales_consistant(self):
        ok = _png_consistent_with_tif(
            "x.png", "x.tif",
            png_size_fn=lambda _: (2000, 2000),
            tif_size_fn=lambda _: (2000, 2000),
        )
        assert ok is True

    def test_png_non_rogne_vs_tif_rogne_incoherent(self):
        # PNG 2200 px (avec marges) vs TIF rogné 2000 px → à régénérer.
        ok = _png_consistent_with_tif(
            "x.png", "x.tif",
            png_size_fn=lambda _: (2200, 2200),
            tif_size_fn=lambda _: (2000, 2000),
        )
        assert ok is False

    def test_lecture_impossible_conservateur(self):
        # Incertitude (lecture None) → ne pas régénérer (on garde l'existant).
        ok = _png_consistent_with_tif(
            "x.png", "x.tif",
            png_size_fn=lambda _: None,
            tif_size_fn=lambda _: (2000, 2000),
        )
        assert ok is True


class TestTifSizeGdalFallback:
    def test_repli_gdal_sans_rasterio(self, monkeypatch):
        # Sans rasterio (QGIS n'en garantit pas la présence), la garde GEO-03
        # doit encore lire les dimensions du TIF via osgeo.gdal — sinon elle
        # devient silencieusement inopérante.
        import sys
        from types import SimpleNamespace

        monkeypatch.setitem(sys.modules, "rasterio", None)  # import -> ImportError

        fake_ds = SimpleNamespace(RasterXSize=2800, RasterYSize=2800)
        fake_gdal = SimpleNamespace(Open=lambda path: fake_ds)
        fake_osgeo = SimpleNamespace(gdal=fake_gdal)
        monkeypatch.setitem(sys.modules, "osgeo", fake_osgeo)
        monkeypatch.setitem(sys.modules, "osgeo.gdal", fake_gdal)

        assert _tif_size("x.tif") == (2800, 2800)
