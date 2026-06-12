"""GEO-04 (audit v2) : résolution du fichier world d'une image d'inférence.

Le world file dépend du format : ``.pgw`` pour PNG, ``.jgw`` pour JPEG,
``.wld`` en repli. Le bug : la branche « source de géoréférencement
prioritaire » de ``create_shapefile_from_detections`` ne tentait QUE ``.jgw``
à côté de PNG → code mort pour tout le pipeline PNG actuel (le même bug,
déjà corrigé dans ``_tile_extent_polygon_from_jpg``, persistait ici).

Le helper vit dans ``pipeline.geo_utils`` (léger, importable sans
pandas/geopandas) et est consommé par ``conversion_shp``.
"""
from __future__ import annotations

from pipeline.geo_utils import find_world_file


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_text("0.5\n0\n0\n-0.5\n700000\n6600000\n")
    return p


def test_png_resout_pgw(tmp_path):
    img = _touch(tmp_path, "tile.png")
    pgw = _touch(tmp_path, "tile.pgw")
    assert find_world_file(img) == pgw


def test_png_replie_sur_wld(tmp_path):
    img = _touch(tmp_path, "tile.png")
    wld = _touch(tmp_path, "tile.wld")
    assert find_world_file(img) == wld


def test_jpg_resout_jgw(tmp_path):
    img = _touch(tmp_path, "tile.jpg")
    jgw = _touch(tmp_path, "tile.jgw")
    assert find_world_file(img) == jgw


def test_aucun_world_file_renvoie_none(tmp_path):
    img = _touch(tmp_path, "tile.png")
    assert find_world_file(img) is None
