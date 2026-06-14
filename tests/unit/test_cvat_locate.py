"""Localisation pure du plugin rvt-qgis pour le calcul CVAT in-process.

On ne teste pas le calcul raster lui-même (il dépend du paquet ``rvt`` et d'un
vrai MNT, validé en test manuel QGIS), mais la logique pure de découverte du
dossier rvt-qgis parmi les plugins frères.
"""
from __future__ import annotations

from pipeline.ign.products.cvat import _find_rvt_dir, _settings_paths


def _make_rvt_plugin(plugins_dir, name="rvt-qgis"):
    """Crée une arborescence rvt-qgis minimale valide sous ``plugins_dir``."""
    rvt = plugins_dir / name
    (rvt / "rvt").mkdir(parents=True)
    (rvt / "rvt" / "blend.py").write_text("# stub", encoding="utf-8")
    (rvt / "settings").mkdir()
    (rvt / "settings" / "blender_VAT.json").write_text("{}", encoding="utf-8")
    (rvt / "settings" / "default_terrains_settings.json").write_text("{}", encoding="utf-8")
    return rvt


class TestFindRvtDir:
    def test_finds_sibling_rvt_plugin(self, tmp_path):
        plugins = tmp_path / "plugins"
        rvt = _make_rvt_plugin(plugins)
        # Le point de départ imite cvat.py : profond dans un plugin frère.
        start = plugins / "archeologia-pipeline" / "src" / "pipeline" / "ign" / "products" / "cvat.py"
        start.parent.mkdir(parents=True)
        start.write_text("# stub", encoding="utf-8")
        assert _find_rvt_dir(start) == rvt

    def test_matches_case_insensitively(self, tmp_path):
        plugins = tmp_path / "plugins"
        rvt = _make_rvt_plugin(plugins, name="RVT")
        start = plugins / "archeologia-pipeline" / "x.py"
        start.parent.mkdir(parents=True)
        assert _find_rvt_dir(start) == rvt

    def test_returns_none_when_absent(self, tmp_path):
        plugins = tmp_path / "plugins"
        start = plugins / "archeologia-pipeline" / "x.py"
        start.parent.mkdir(parents=True)
        assert _find_rvt_dir(start) is None

    def test_ignores_rvt_dir_without_settings(self, tmp_path):
        # Un dossier "rvt*" présent mais incomplet ne doit pas matcher.
        plugins = tmp_path / "plugins"
        incomplete = plugins / "rvt-broken" / "rvt"
        incomplete.mkdir(parents=True)
        (incomplete / "blend.py").write_text("# stub", encoding="utf-8")
        start = plugins / "archeologia-pipeline" / "x.py"
        start.parent.mkdir(parents=True)
        assert _find_rvt_dir(start) is None


class TestSettingsPaths:
    def test_ok_when_files_present(self, tmp_path):
        rvt = _make_rvt_plugin(tmp_path)
        paths = _settings_paths(rvt)
        assert paths is not None
        assert paths.blender_vat_json == rvt / "settings" / "blender_VAT.json"
        assert paths.terrains_json == rvt / "settings" / "default_terrains_settings.json"

    def test_none_when_missing(self, tmp_path):
        assert _settings_paths(tmp_path / "nope") is None
