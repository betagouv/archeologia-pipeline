"""``pipeline.ign.tile_resolver`` (lot 6 du chantier d'outillage, 2026-10-20) : intersection d'un polygone
avec un mini quadrillage IGN de 5 dalles écrit par geopandas dans ``tmp_path`` (schéma ``nom_pkk`` /
``url_telech``), sortie au format ``nom,url`` du téléchargeur.

OGR est requis : à lancer avec le Python de QGIS (``.venv_dev`` n'a pas de roue GDAL ; le test saute)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("osgeo")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import box  # noqa: E402

from pipeline.ign import tile_resolver  # noqa: E402
from pipeline.ign.downloader import parse_ign_input_file  # noqa: E402

# Cinq dalles de 1 km en Lambert-93 (nom = coin nord-ouest en km) ; la dernière n'a pas d'URL.
_DALLES = ["0800_6300", "0801_6300", "0802_6300", "0800_6301", "0801_6301"]
_SANS_URL = {"0801_6301"}


def _nom(c: str) -> str:
    return f"LHD_FXX_{c}_PTS_C_LAMB93_IGN69"


def _url(c: str) -> str:
    return f"https://data.geopf.fr/telechargement/download/LiDAR-HD/{_nom(c)}.copc.laz"


@pytest.fixture
def quadrillage(tmp_path) -> Path:
    lignes, geoms = [], []
    for c in _DALLES:
        x, y = (int(v) * 1000 for v in c.split("_"))
        geoms.append(box(x, y - 1000, x + 1000, y))
        lignes.append({"nom_pkk": _nom(c), "url_telech": "" if c in _SANS_URL else _url(c)})
    d = tmp_path / "data" / "quadrillage_france"
    d.mkdir(parents=True)
    shp = d / "TA_diff_pkk_lidarhd_classe.shp"
    gpd.GeoDataFrame(lignes, geometry=geoms, crs="EPSG:2154").to_file(shp)
    return shp


def _zone(tmp_path: Path, geom, crs: str = "EPSG:2154", nom: str = "zone.gpkg", couche: str = "zone") -> Path:
    p = tmp_path / nom
    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:2154").to_crs(crs).to_file(p, layer=couche, driver="GPKG")
    return p


def _resoudre(zone: Path, quadrillage, sortie: Path, **kw):
    journal = []
    n = tile_resolver.resolve_tiles_from_polygon(zone, sortie, quadrillage_path=quadrillage, log=journal.append, **kw)
    return n, journal


# Bande de 1 km × 100 m qui touche 0800_6300 et 0801_6300 seulement.
_DEUX_DALLES = box(800500, 6299500, 801500, 6299600)


def test_deux_dalles_touchees_au_format_du_telechargeur(tmp_path, quadrillage):
    sortie = tmp_path / "sortie" / "dalles_urls.txt"
    n, journal = _resoudre(_zone(tmp_path, _DEUX_DALLES), quadrillage, sortie)
    assert n == 2
    attendu = [(f"{_nom(c)}.copc.laz", _url(c)) for c in ("0800_6300", "0801_6300")]   # nom = celui de l'URL
    lignes = sorted(ligne.split(",") for ligne in sortie.read_text(encoding="utf-8").splitlines())
    assert lignes == [list(t) for t in attendu]
    assert sorted(parse_ign_input_file(sortie)) == attendu                        # ce que lit download_ign_dalles
    assert any("2 dalle(s) identifiée(s)" in m for m in journal)


def test_dalle_sans_url_ignoree_et_signalee(tmp_path, quadrillage):
    zone = _zone(tmp_path, box(800900, 6300500, 801100, 6300600))   # 0800_6301 (URL) et 0801_6301 (sans URL)
    n, journal = _resoudre(zone, quadrillage, tmp_path / "s.txt")
    assert n == 1
    assert any("1 dalle(s) ignorée(s) (URL manquante)" in m for m in journal)


def test_polygone_en_wgs84_reprojete_en_lambert93(tmp_path, quadrillage):
    n, journal = _resoudre(_zone(tmp_path, _DEUX_DALLES, crs="EPSG:4326"), quadrillage, tmp_path / "s.txt")
    assert n == 2
    assert any("Reprojection" in m and "EPSG:2154" in m for m in journal)


def test_plusieurs_couches_unionnees(tmp_path, quadrillage):
    p = tmp_path / "groupe.gpkg"
    gpd.GeoDataFrame(geometry=[box(800100, 6299100, 800200, 6299200)], crs="EPSG:2154").to_file(p, layer="a", driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(802100, 6299100, 802200, 6299200)], crs="EPSG:2154").to_file(p, layer="b", driver="GPKG")
    sortie = tmp_path / "s.txt"
    n, journal = _resoudre(p, quadrillage, sortie)
    assert n == 2
    assert {ligne.split(",")[0] for ligne in sortie.read_text(encoding="utf-8").splitlines()} == {
        f"{_nom('0800_6300')}.copc.laz", f"{_nom('0802_6300')}.copc.laz"}
    assert any("2 couches" in m for m in journal)


def test_hors_grille_zero_dalle_et_pas_de_fichier(tmp_path, quadrillage):
    sortie = tmp_path / "s.txt"
    n, journal = _resoudre(_zone(tmp_path, box(900000, 6400000, 900100, 6400100)), quadrillage, sortie)
    assert n == 0 and not sortie.exists()
    assert any("Aucune dalle" in m for m in journal)


def test_annulation(tmp_path, quadrillage):
    n, journal = _resoudre(_zone(tmp_path, _DEUX_DALLES), quadrillage, tmp_path / "s.txt", cancel=lambda: True)
    assert n == 0 and any("Annulation" in m for m in journal)


def test_fichiers_absents(tmp_path, quadrillage):
    with pytest.raises(FileNotFoundError, match="zone d'étude"):
        tile_resolver.resolve_tiles_from_polygon(tmp_path / "absente.gpkg", tmp_path / "s.txt", quadrillage_path=quadrillage)
    with pytest.raises(FileNotFoundError, match="Quadrillage"):
        tile_resolver.resolve_tiles_from_polygon(_zone(tmp_path, _DEUX_DALLES), tmp_path / "s.txt",
                                                 quadrillage_path=tmp_path / "nulle_part.shp")


def test_quadrillage_par_defaut_sous_la_racine_du_plugin(tmp_path, quadrillage, monkeypatch):
    """Sans ``quadrillage_path`` : ``data/quadrillage_france/`` de la racine du plugin (assurer_quadrillage)."""
    monkeypatch.setattr(tile_resolver, "_get_plugin_root", lambda: tmp_path)
    n, _ = _resoudre(_zone(tmp_path, _DEUX_DALLES), None, tmp_path / "s.txt")
    assert n == 2
