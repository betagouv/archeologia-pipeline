"""Tests du résolveur de chemin du quadrillage IGN (pur, hors QGIS).

Le quadrillage lourd (shapefile ~176 Mo, sans index spatial) est remplacé par
un GeoPackage slim à R-tree. Le résolveur permet une bascule transparente :
il préfère le ``.gpkg`` s'il est présent, sinon retombe sur le ``.shp`` legacy.
C'est la source de vérité unique partagée par ``tile_resolver`` et l'outil UI
de sélection des dalles.
"""
from __future__ import annotations

from pathlib import Path

import datetime

from pipeline.ign.quadrillage_paths import phrase_quadrillage, quadrillage_info, resolve_quadrillage_path

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


class TestQuadrillageInfo:
    """A8 : date et effectif de la grille lus dans l'en-tête DBF, pour le bandeau de l'étape 1."""

    def test_lit_date_et_effectif_dans_l_en_tete_dbf(self, tmp_path):
        shp = _touch(tmp_path, _SHP)
        en_tete = bytes([3, 126, 10, 7]) + (524687).to_bytes(4, "little") + bytes(24)
        shp.with_suffix(".dbf").write_bytes(en_tete)
        info = quadrillage_info(shp)
        assert info == (datetime.date(2026, 10, 7), 524687)
        assert phrase_quadrillage(info) == "Grille IGN du 7 octobre 2026, 524 687 dalles."
        assert phrase_quadrillage((datetime.date(2027, 1, 1), 530000)) == "Grille IGN du 1er janvier 2027, 530 000 dalles."

    def test_gpkg_sans_en_tete_et_fichier_absent(self, tmp_path):
        gpkg = _touch(tmp_path, _GPKG)
        info = quadrillage_info(gpkg)
        assert info is not None and info[1] is None and info[0] == datetime.date.today()
        assert phrase_quadrillage(info).startswith("Grille IGN du ") and "dalles" not in phrase_quadrillage(info)
        assert quadrillage_info(tmp_path / "absent.shp") is None
        assert phrase_quadrillage(None) == ""
        # en-tête tronqué (fichier corrompu) → date du fichier, pas d'exception
        shp = _touch(tmp_path, _SHP)
        shp.with_suffix(".dbf").write_bytes(b"\x03\x7e")
        assert quadrillage_info(shp) == (datetime.date.today(), None)

    def test_le_bandeau_de_l_etape_1_affiche_la_phrase(self):
        src = (Path(__file__).resolve().parents[2] / "src/ui/steps/step_1_source.py").read_text(encoding="utf-8")
        assert "phrase_quadrillage(quadrillage_info(" in src
