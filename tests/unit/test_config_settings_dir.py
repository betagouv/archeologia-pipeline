"""CFG-02 (audit v2) : last_ui_config.json vivait DANS le dossier du plugin —
remplacé à chaque mise à jour par ZIP → l'archéologue perdait TOUS ses
réglages (sortie, mode, entités, seuils). Les réglages vivent désormais dans
le dossier de profil fourni par l'hôte (QgsApplication.qgisSettingsDirPath()/
archeologia côté QGIS), avec migration automatique de l'ancien emplacement.

UIX-07/CFG-06 au passage : écriture ATOMIQUE (tmp + os.replace) — un crash
pendant l'autosave n'efface plus toute la configuration.
"""
from __future__ import annotations

import json

from config.config_manager import ConfigManager


def _cfg_with_output(cm: ConfigManager, value: str) -> dict:
    cfg = cm.default_config()
    cfg["app"]["files"]["output_dir"] = value
    return cfg


def test_settings_dir_heberge_last_ui_config(tmp_path):
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    settings = tmp_path / "profil" / "archeologia"

    cm = ConfigManager(plugin_root, settings_dir=settings)
    cm.save_last_ui_config(_cfg_with_output(cm, "X:/out"))

    assert (settings / "last_ui_config.json").exists()
    assert not (plugin_root / "last_ui_config.json").exists()
    assert cm.load_last_ui_config()["app"]["files"]["output_dir"] == "X:/out"


def test_migration_depuis_le_dossier_plugin(tmp_path):
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    legacy = plugin_root / "last_ui_config.json"
    legacy.write_text(
        json.dumps({"app": {"files": {"output_dir": "D:/ancien"}}}),
        encoding="utf-8",
    )
    settings = tmp_path / "profil" / "archeologia"

    cm = ConfigManager(plugin_root, settings_dir=settings)

    # Les réglages d'avant la migration sont conservés…
    assert cm.load_last_ui_config()["app"]["files"]["output_dir"] == "D:/ancien"
    # …déplacés (pas dupliqués) vers le nouveau dossier.
    assert (settings / "last_ui_config.json").exists()
    assert not legacy.exists()


def test_sans_settings_dir_comportement_inchange(tmp_path):
    cm = ConfigManager(tmp_path)
    assert cm.last_ui_path == tmp_path / "last_ui_config.json"


def test_ecriture_atomique_ne_laisse_pas_de_tmp(tmp_path):
    cm = ConfigManager(tmp_path)
    cm.save_last_ui_config(_cfg_with_output(cm, "X:/out"))
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert cm.load_last_ui_config()["app"]["files"]["output_dir"] == "X:/out"
