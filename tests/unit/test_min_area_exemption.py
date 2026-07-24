"""``_filter_gpkg_by_min_area`` : exemption des couches synthétiques.

Le seuil d'aire global du modèle (``thresholds.min_area_m2``, ex. 500 m² sur
le modèle formes linéaires) est appliqué APRÈS écriture à toutes les couches
des GeoPackages. Les sorties des briques de synthèse (enclos, zones de
clusters) portent leur propre ``min_area_m2`` par règle : sans exemption, un
enclos funéraire de 80 m² serait silencieusement effacé.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import box

from pipeline.cv.conversion_shp import _filter_gpkg_by_min_area


def _write_layer(path, layer, area_m2):
    side = area_m2 ** 0.5
    gdf = gpd.GeoDataFrame(
        {"model_pred": [layer]}, geometry=[box(0.0, 0.0, side, side)],
        crs="EPSG:2154",
    )
    gdf.to_file(str(path), layer=layer, driver="GPKG")


def test_exempt_layer_survives_global_min_area(tmp_path):
    p = tmp_path / "enclos.gpkg"
    _write_layer(p, "Enclos", 80.0)  # < 500 m² mais couche synthétique
    _filter_gpkg_by_min_area(
        [str(p)], min_area_m2=500.0, exempt_layers=frozenset({"Enclos"})
    )
    assert p.exists()
    assert len(gpd.read_file(str(p), layer="Enclos")) == 1


def test_non_exempt_layer_filtered_and_empty_gpkg_removed(tmp_path):
    p = tmp_path / "parcellaire.gpkg"
    _write_layer(p, "parcellaire", 80.0)
    _filter_gpkg_by_min_area([str(p)], min_area_m2=500.0)
    assert not p.exists()  # couche vidée → GeoPackage supprimé (existant)


def test_tagged_source_features_survive_min_area(tmp_path):
    # Les fragments sources TAGUÉS par une brique (enclos_id/axe_id/cluster_id)
    # survivent au filtre d'aire global : la traçabilité membre→synthèse prime.
    # (Bug Bretagne : 434/493 fragments sources < 500 m² effacés → enclos
    # détectés sans aucun contour source visible.)
    p = tmp_path / "enclos.gpkg"
    side = 80 ** 0.5
    gdf = gpd.GeoDataFrame(
        {"model_pred": ["parcellaire", "parcellaire"],
         "enclos_id": ["enclos_0", ""]},
        geometry=[box(0.0, 0.0, side, side),
                  box(100.0, 100.0, 100.0 + side, 100.0 + side)],
        crs="EPSG:2154",
    )
    gdf.to_file(str(p), layer="Linéaments sources", driver="GPKG")
    _filter_gpkg_by_min_area([str(p)], min_area_m2=500.0)
    out = gpd.read_file(str(p), layer="Linéaments sources")
    assert len(out) == 1
    assert out.iloc[0]["enclos_id"] == "enclos_0"
