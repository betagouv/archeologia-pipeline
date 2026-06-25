"""Tests du résolveur de chemin du quadrillage IGN (pur, hors QGIS).

Le quadrillage lourd (shapefile ~176 Mo, sans index spatial) est remplacé par
un GeoPackage slim à R-tree. Le résolveur permet une bascule transparente :
il préfère le ``.gpkg`` s'il est présent, sinon retombe sur le ``.shp`` legacy.
C'est la source de vérité unique partagée par ``tile_resolver`` et l'outil UI
de sélection des dalles.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.ign.quadrillage_paths import resolve_quadrillage_path

_RELDIR = Path("data") / "quadrillage_france"
_GPKG = _RELDIR / "TA_diff_pkk_lidarhd_classe.gpkg"
_SHP = _RELDIR / "TA_diff_pkk_lidarhd_classe.shp"


def _touch(root: Path, rel: Path) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


class TestResolveQuadrillagePath:
    def test_prefers_gpkg_when_present(self, tmp_path):
        """Le GeoPackage slim est préféré dès qu'il existe."""
        _touch(tmp_path, _GPKG)
        _touch(tmp_path, _SHP)  # les deux présents → on prend quand même le .gpkg
        assert resolve_quadrillage_path(tmp_path) == tmp_path / _GPKG

    def test_falls_back_to_shp_when_no_gpkg(self, tmp_path):
        """Sans GeoPackage, on retombe sur le shapefile legacy."""
        _touch(tmp_path, _SHP)
        assert resolve_quadrillage_path(tmp_path) == tmp_path / _SHP

    def test_returns_shp_path_when_neither_exists(self, tmp_path):
        """Aucun des deux : on renvoie le chemin .shp (la vérif d'existence /
        l'erreur reste à la charge de l'appelant, p.ex. resolve_tiles_from_polygon)."""
        assert resolve_quadrillage_path(tmp_path) == tmp_path / _SHP
