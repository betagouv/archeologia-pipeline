"""Fonctions pures de ``pipeline.cv.conversion_shp`` (lot 6 du chantier d'outillage, 2026-10-20) : les
petites fonctions que ``create_shapefile_from_detections`` appelle à chaque dalle et à chaque détection —
fichier world, coordonnées des noms IGN, voisinage, IoU, couches d'un GeoPackage, filtrage par aire."""
from __future__ import annotations

import pytest

pytest.importorskip("geopandas")
PIL_Image = pytest.importorskip("PIL.Image")

import geopandas as gpd  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from pipeline.cv import conversion_shp as cs  # noqa: E402


def test_fichier_world_lu_ou_refuse(tmp_path):
    w = tmp_path / "a.pgw"
    w.write_text("0.5\n0\n0\n-0.5\n800000.25\n6300000.75\n", encoding="utf-8")
    assert cs.read_world_file(str(w)) == (0.5, -0.5, 800000.25, 6300000.75)
    (tmp_path / "court.pgw").write_text("0.5\n0\n0\n", encoding="utf-8")
    assert cs.read_world_file(str(tmp_path / "court.pgw")) == (None, None, None, None)
    (tmp_path / "faux.pgw").write_text("x\n0\n0\n-0.5\n1\n2\n", encoding="utf-8")
    assert cs.read_world_file(str(tmp_path / "faux.pgw")) == (None, None, None, None)
    assert cs.read_world_file(str(tmp_path / "absent.pgw")) == (None, None, None, None)


def test_coordonnees_des_noms_ign():
    assert cs.extract_tile_coordinates("LHD_FXX_0929_6613_PTS_C") == ("0929", "6613")
    assert cs.extract_tile_coordinates("trop_court") == (None, None)


def test_voisinage_3x3_et_entree_invalide():
    cles = cs.calculate_neighbor_tile_keys("0929", "6613")
    assert len(cles) == 9 and "0929_6613" in cles and cles[0] == "0928_6612" and cles[-1] == "0930_6614"
    assert cs.calculate_neighbor_tile_keys("abc", "6613") == ["abc_6613"]


def test_iou_de_polygones():
    a, b = box(0, 0, 10, 10), box(5, 0, 15, 10)
    assert cs._polygon_iou(a, a) == pytest.approx(1.0)
    assert cs._polygon_iou(a, b) == pytest.approx(50 / 150)
    assert cs._polygon_iou(a, box(20, 20, 30, 30)) == 0.0
    assert cs._polygon_iou(a, None) is None
    assert cs._polygon_iou(Polygon(), a) == 0.0                       # polygone vide : n'intersecte rien


def test_etiquette_de_classe_normalisee():
    assert cs._normalize_class_label("  four ") == "four"
    assert cs._normalize_class_label(3) == "3"


def test_emprise_d_une_image_par_son_fichier_world(tmp_path):
    img = tmp_path / "A.png"
    PIL_Image.new("L", (200, 100)).save(img)
    (tmp_path / "A.pgw").write_text("0.5\n0\n0\n-0.5\n800000\n6300000\n", encoding="utf-8")
    emprise = cs._tile_extent_polygon_from_jpg(img)
    assert emprise.bounds == (800000.0, 6299950.0, 800100.0, 6300000.0)
    assert cs._tile_extent_polygon_from_jpg(tmp_path / "absente.png") is None
    (tmp_path / "B.png").write_bytes(img.read_bytes())
    assert cs._tile_extent_polygon_from_jpg(tmp_path / "B.png") is None  # sans fichier world


def _gpkg(tmp_path, couches: dict):
    p = tmp_path / "d.gpkg"
    for nom, (geoms, attrs) in couches.items():
        gpd.GeoDataFrame(attrs, geometry=geoms, crs="EPSG:2154").to_file(p, layer=nom, driver="GPKG")
    return p


def test_couches_d_un_geopackage(tmp_path):
    p = _gpkg(tmp_path, {"four": ([box(0, 0, 1, 1)], {"confidence": [0.9]}), "tumulus": ([box(0, 0, 1, 1)], {"confidence": [0.5]})})
    assert sorted(cs._list_gpkg_layers(str(p))) == ["four", "tumulus"]
    assert cs._list_gpkg_layers(str(tmp_path / "absent.gpkg")) == []


def test_filtrage_par_aire_dans_un_geopackage(tmp_path):
    """Les petites entités partent, une entité taguée par une brique de synthèse (cluster_id) reste, la couche
    exemptée n'est pas lue, un GeoPackage entièrement vidé est supprimé."""
    p = _gpkg(tmp_path, {
        "four": ([box(0, 0, 2, 2), box(0, 0, 20, 20), box(10, 10, 12, 12)],
                 {"confidence": [0.9, 0.8, 0.7], "cluster_id": [None, None, "c1"]}),
        "zone_de_fours": ([box(0, 0, 3, 3)], {"confidence": [1.0]}),
    })
    cs._filter_gpkg_by_min_area([str(p)], min_area_m2=100.0, exempt_layers=frozenset({"zone_de_fours"}))
    fours = gpd.read_file(p, layer="four")
    assert sorted(fours.geometry.area) == [4.0, 400.0]               # 4 m² gardée par son cluster_id
    assert len(gpd.read_file(p, layer="zone_de_fours")) == 1          # exemptée (9 m² < 100)

    (tmp_path / "v").mkdir()
    vide = _gpkg(tmp_path / "v", {"four": ([box(0, 0, 2, 2)], {"confidence": [0.9]})})
    cs._filter_gpkg_by_min_area([str(vide)], min_area_m2=100.0)
    assert not vide.exists()


def test_suppression_d_un_jeu_shapefile(tmp_path):
    gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:2154").to_file(tmp_path / "s.shp")
    assert (tmp_path / "s.dbf").exists()
    cs._remove_shapefile_set(tmp_path / "s.shp")
    assert not list(tmp_path.glob("s.*"))
    cs._remove_shapefile_set(tmp_path / "inexistant.shp")            # silencieux


def test_tranche_de_confiance_deleguee_a_class_utils():
    from pipeline.cv.class_utils import assign_confidence_bin
    assert cs._confidence_bucket(0.85, 2, 0.3) == assign_confidence_bin(0.85, 2, 0.3)
    assert cs._confidence_bucket(None) == (None, None)
