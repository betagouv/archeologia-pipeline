"""Tests du formatage du ``dalles_urls.txt`` issu de la sélection sur carte (pur).

L'utilisateur sélectionne des dalles IGN sur le canevas QGIS ; la page lit
``(nom_pkk, url_telech)`` des entités retenues et délègue le formatage à
``format_dalles_urls``. Le fichier produit alimente directement le téléchargeur
(``parse_ign_input_file`` : lignes ``nom,url``, lignes ``#`` ignorées) — donc le
même format que ``resolve_tiles_from_polygon``.
"""
from __future__ import annotations

from app.services.tile_selection import estimate_download_size, format_dalles_urls, urls_de_la_liste

# Le nom de fichier enregistré DOIT être le basename de l'URL (avec extension),
# pas le nom_pkk (sans extension) — sinon PDAL ne peut pas lire le fichier.
_NAME_A = "LHD_FXX_0946_6744.copc.laz"
_NAME_B = "LHD_FXX_0947_6744.copc.laz"
_URL_A = f"https://data.geopf.fr/telechargement/download/{_NAME_A}"
_URL_B = f"https://data.geopf.fr/telechargement/download/{_NAME_B}"


def _data_lines(text: str) -> list[str]:
    """Lignes utiles : ni vides, ni commentaires ``#``."""
    return [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


class TestFormatDallesUrls:
    def test_filename_comes_from_url_with_extension(self):
        # Le nom_pkk (sans extension) est ignoré ; on prend le basename de l'URL.
        out = format_dalles_urls([("LHD_A", _URL_A), ("LHD_B", _URL_B)])
        assert _data_lines(out) == [f"{_NAME_A},{_URL_A}", f"{_NAME_B},{_URL_B}"]

    def test_filename_carries_url_extension(self):
        """Régression : le nom de fichier DOIT porter l'extension de l'URL
        (sinon PDAL : « Cannot determine reader » → dalle supprimée)."""
        out = format_dalles_urls([("LHD_FXX_0946_6744", _URL_A)])  # nom_pkk nu, sans ext
        fname = _data_lines(out)[0].split(",", 1)[0]
        assert fname == _NAME_A
        assert fname.endswith(".copc.laz")

    def test_dedups_by_url_preserving_order(self):
        out = format_dalles_urls([("LHD_A", _URL_A), ("LHD_B", _URL_B), ("LHD_A_bis", _URL_A)])
        assert _data_lines(out) == [f"{_NAME_A},{_URL_A}", f"{_NAME_B},{_URL_B}"]

    def test_ignores_entries_without_http_url(self):
        out = format_dalles_urls([
            ("LHD_A", _URL_A),
            ("LHD_NO_URL", ""),
            ("LHD_NONE", None),
            ("LHD_FTP", "ftp://example/file.laz"),
        ])
        assert _data_lines(out) == [f"{_NAME_A},{_URL_A}"]

    def test_empty_input_has_no_data_lines(self):
        out = format_dalles_urls([])
        assert _data_lines(out) == []

    def test_empty_nom_still_uses_url_filename(self):
        """Même avec un nom vide, le nom de fichier vient de l'URL (avec extension)."""
        out = format_dalles_urls([("", _URL_A), (None, _URL_B)])
        assert _data_lines(out) == [f"{_NAME_A},{_URL_A}", f"{_NAME_B},{_URL_B}"]

    def test_strips_whitespace(self):
        out = format_dalles_urls([("  LHD_A  ", f"  {_URL_A}  ")])
        assert _data_lines(out) == [f"{_NAME_A},{_URL_A}"]

    def test_has_comment_header_with_count(self):
        out = format_dalles_urls([("LHD_A", _URL_A), ("LHD_B", _URL_B)])
        first = out.splitlines()[0]
        assert first.startswith("#") and "2" in first


class TestEstimateDownloadSize:
    def test_zero_or_negative_is_empty(self):
        assert estimate_download_size(0) == ""
        assert estimate_download_size(-3) == ""

    def test_small_selection_in_megabytes(self):
        # 1 dalle ≈ 50–400 Mo (bornes par dalle).
        out = estimate_download_size(1)
        assert "Mo" in out and "Go" not in out
        assert "50" in out and "400" in out
        assert out.startswith("≈")

    def test_two_tiles_still_megabytes(self):
        out = estimate_download_size(2)
        assert "Mo" in out and "100" in out and "800" in out

    def test_large_selection_switches_to_gigabytes(self):
        # 10 dalles : max = 4000 Mo → exprimé en Go, décimale française.
        out = estimate_download_size(10)
        assert "Go" in out and "Mo" not in out
        assert "," in out  # séparateur décimal français

    def test_threshold_three_tiles_is_gigabytes(self):
        # 3 dalles : max = 1200 Mo ≥ 1000 → bascule en Go.
        assert "Go" in estimate_download_size(3)


def test_urls_de_la_liste_suit_la_regle_du_telechargeur():
    """Lignes « nom,url » ou « url », vides et « # » ignorées (parse_ign_input_file) : ce que le run
    téléchargera, ni plus ni moins — pour compter sous le champ et resélectionner sur la carte."""
    texte = format_dalles_urls([("A", "https://x/a.copc.laz"), ("B", "https://x/b.copc.laz"),
                                ("A2", "https://x/a.copc.laz"), ("C", ""), ("D", "ftp://x/d")])
    assert urls_de_la_liste(texte) == ["https://x/a.copc.laz", "https://x/b.copc.laz"]
    assert urls_de_la_liste("# en-tête\n\nhttps://x/seule.laz\nnom,url\n  n,https://x/n.laz  \n") == [
        "https://x/seule.laz", "https://x/n.laz"]
    assert urls_de_la_liste("") == [] and urls_de_la_liste(None) == []


def test_la_page_relit_la_liste_et_suit_la_couche():
    """Garde-fou sans QGIS (lu en source). Constats du 2026-11-10 : « 0 dalle sur la carte alors que 25
    sélectionnées » (rouvrir « Choisir sur la carte » repartait d'une sélection vide, la liste du champ jamais
    relue) ; libellé « 5 dalles choisies » figé après une seconde validation (setText d'un chemin inchangé
    n'émet pas textChanged) ; compteur du bouton aveugle à une sélection faite autrement que par l'outil."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[2] / "src/ui/steps/step_1_source.py").read_text(encoding="utf-8")
    assert "urls_de_la_liste(" in src
    assert "selectionChanged.connect(" in src and "selectionChanged.disconnect(" in src
    assert "self._source_edit.setText(str(out_path))\n        self._refresh_source_validation()" in src
