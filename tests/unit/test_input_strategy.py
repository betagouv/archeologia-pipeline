"""Persistance de la liste de dalles au démarrage d'un run ``ign_laz``.

Quand l'entrée est une liste ``.txt`` déjà résolue (sélection des dalles sur la
carte, ou fichier pré-établi), elle vit dans un dossier temporaire écrasé à chaque
sélection. Si le run est interrompu, l'utilisateur perd la liste et ne peut pas
reprendre. ``persist_resolved_dalles_list`` la recopie dans le dossier de sortie
(``<output_dir>/dalles_urls.txt``) dès le début, AVANT tout téléchargement — au
même emplacement que la branche polygone (``resolve_tiles_from_polygon``).
"""
from __future__ import annotations

from pathlib import Path

from app.runners.input_strategy import persist_resolved_dalles_list

_CONTENT = (
    "# 3 dalle(s) sélectionnée(s) sur la carte\n"
    "LHD_FXX_0655_6811.copc.laz,https://data.geopf.fr/x/LHD_FXX_0655_6811.copc.laz\n"
    "LHD_FXX_0655_6812.copc.laz,https://data.geopf.fr/x/LHD_FXX_0655_6812.copc.laz\n"
    "LHD_FXX_0655_6813.copc.laz,https://data.geopf.fr/x/LHD_FXX_0655_6813.copc.laz\n"
)


def test_copies_txt_to_output_dir_dalles_urls(tmp_path):
    src = tmp_path / "temp_zones" / "dalles_selection.txt"
    src.parent.mkdir()
    src.write_text(_CONTENT, encoding="utf-8")
    out = tmp_path / "sortie"
    out.mkdir()

    dest = persist_resolved_dalles_list(src, out)

    assert dest == out / "dalles_urls.txt"
    assert dest.read_text(encoding="utf-8") == _CONTENT


def test_creates_output_dir_if_missing(tmp_path):
    src = tmp_path / "dalles_selection.txt"
    src.write_text(_CONTENT, encoding="utf-8")
    out = tmp_path / "sortie_absente"  # n'existe pas encore

    dest = persist_resolved_dalles_list(src, out)

    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == _CONTENT


def test_noop_when_source_is_already_the_destination(tmp_path):
    # Re-run pointant directement sur <output_dir>/dalles_urls.txt : pas d'auto-copie
    # (shutil lèverait SameFileError), le contenu reste intact.
    out = tmp_path / "sortie"
    out.mkdir()
    dest_existing = out / "dalles_urls.txt"
    dest_existing.write_text(_CONTENT, encoding="utf-8")

    dest = persist_resolved_dalles_list(dest_existing, out)

    assert dest == dest_existing
    assert dest.read_text(encoding="utf-8") == _CONTENT


def test_overwrites_stale_dalles_urls_with_new_selection(tmp_path):
    out = tmp_path / "sortie"
    out.mkdir()
    (out / "dalles_urls.txt").write_text("# ancienne sélection\nVIEUX,url\n", encoding="utf-8")
    src = tmp_path / "dalles_selection.txt"
    src.write_text(_CONTENT, encoding="utf-8")

    dest = persist_resolved_dalles_list(src, out)

    assert dest.read_text(encoding="utf-8") == _CONTENT
