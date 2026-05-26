from __future__ import annotations

import pytest

# Ces helpers vivent dans cv/conversion_shp (importable hors QGIS : shapely seulement).
pytest.importorskip("shapely")
PIL_Image = pytest.importorskip("PIL.Image")

from pipeline.cv.conversion_shp import _tile_extent_polygon_from_jpg


def _make_png_with_world(tmp_path, name, ext, *, px=0.5, x0=700000.0, y0=6600000.0,
                         w=20, h=10):
    img = tmp_path / name
    PIL_Image.new("L", (w, h)).save(img)
    # World file : pixel_width, rot, rot, pixel_height(négatif), x_origin, y_origin
    img.with_suffix(ext).write_text(f"{px}\n0\n0\n{-px}\n{x0}\n{y0}\n")
    return img


class TestTileExtentPolygon:
    def test_reads_pgw_for_png(self, tmp_path):
        """Le bug : un PNG a un .pgw (pas un .jgw) → l'emprise doit être lue."""
        png = _make_png_with_world(tmp_path, "tile.png", ".pgw", w=20, h=10)
        poly = _tile_extent_polygon_from_jpg(png)
        assert poly is not None
        minx, miny, maxx, maxy = poly.bounds
        assert (minx, maxy) == (700000.0, 6600000.0)
        assert maxx == 700010.0   # 20 px * 0.5 m
        assert miny == 6599995.0  # 6600000 - 10 px * 0.5 m

    def test_still_reads_jgw_for_jpg(self, tmp_path):
        """Rétro-compat : un JPEG avec .jgw fonctionne toujours."""
        jpg = _make_png_with_world(tmp_path, "tile.jpg", ".jgw")
        assert _tile_extent_polygon_from_jpg(jpg) is not None

    def test_returns_none_without_world_file(self, tmp_path):
        png = tmp_path / "lonely.png"
        PIL_Image.new("L", (10, 10)).save(png)
        assert _tile_extent_polygon_from_jpg(png) is None

    def test_two_overlapping_png_tiles_intersect(self, tmp_path):
        """Deux dalles PNG voisines qui se chevauchent → emprises qui s'intersectent."""
        a = _make_png_with_world(tmp_path, "a.png", ".pgw", x0=700000.0, w=20, h=20)
        b = _make_png_with_world(tmp_path, "b.png", ".pgw", x0=700005.0, w=20, h=20)
        pa = _tile_extent_polygon_from_jpg(a)
        pb = _tile_extent_polygon_from_jpg(b)
        assert pa.intersects(pb)
        assert pa.intersection(pb).area > 0
