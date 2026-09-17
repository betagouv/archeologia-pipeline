"""Configurations nommées (bibliothèque de configs) — module pur.

Avant le 2026-09-17, « Enregistrer la config » ouvrait un navigateur de
fichiers : l'utilisateur choisissait un dossier au hasard, et « Charger une
config » lui redemandait de le retrouver. Les configs enregistrées vivent
désormais dans UN dossier connu (``<profil QGIS>/archeologia/configs/``, pas le
dossier du plugin — remplacé à chaque mise à jour, cf. CFG-02), et le
chargement se fait par un menu qui liste ce dossier.

Ce module porte la bibliothèque, et rien d'autre : pas de Qt, pas de QGIS. Le
câblage vit dans ``ui/wizard_dialog.py`` et ``ui/dialogs/config_dialogs.py``,
qui ne sont pas collectés par pytest.
"""
from __future__ import annotations

import json

import pytest

from app.services.config_store import ConfigStore, InvalidConfigName


def _store(tmp_path) -> ConfigStore:
    return ConfigStore(tmp_path / "configs")


# --- enregistrer / lister / charger ---------------------------------------

def test_dossier_absent_liste_vide(tmp_path):
    assert _store(tmp_path).list_names() == []


def test_enregistrer_puis_recharger_rend_la_meme_config(tmp_path):
    store = _store(tmp_path)
    store.save("Prospection Gard", {"app": {"files": {"output_dir": "X:/out"}}})

    assert store.list_names() == ["Prospection Gard"]
    assert store.load("Prospection Gard") == {"app": {"files": {"output_dir": "X:/out"}}}


def test_enregistrer_cree_le_dossier(tmp_path):
    store = _store(tmp_path)
    store.save("a", {})
    assert (tmp_path / "configs" / "a.json").is_file()


def test_liste_triee_sans_tenir_compte_de_la_casse(tmp_path):
    store = _store(tmp_path)
    for name in ("zone b", "Alpha", "beta"):
        store.save(name, {})
    assert store.list_names() == ["Alpha", "beta", "zone b"]


def test_liste_ignore_les_fichiers_non_json(tmp_path):
    store = _store(tmp_path)
    store.save("garde", {})
    (tmp_path / "configs" / "notes.txt").write_text("bruit", encoding="utf-8")
    assert store.list_names() == ["garde"]


def test_enregistrer_deux_fois_ecrase_sans_doublon(tmp_path):
    store = _store(tmp_path)
    store.save("a", {"v": 1})
    store.save("a", {"v": 2})
    assert store.list_names() == ["a"]
    assert store.load("a") == {"v": 2}


def test_existe_dit_si_le_nom_est_deja_pris(tmp_path):
    store = _store(tmp_path)
    assert not store.exists("a")
    store.save("a", {})
    assert store.exists("a")


def test_charger_un_json_illisible_leve(tmp_path):
    store = _store(tmp_path)
    store.save("casse", {})
    (tmp_path / "configs" / "casse.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load("casse")


def test_charger_un_nom_absent_leve(tmp_path):
    with pytest.raises(FileNotFoundError):
        _store(tmp_path).load("fantome")


def test_ecriture_atomique_ne_laisse_pas_de_tmp(tmp_path):
    store = _store(tmp_path)
    store.save("a", {"v": 1})
    assert [p.name for p in (tmp_path / "configs").iterdir()] == ["a.json"]


def test_accents_preserves_dans_le_fichier(tmp_path):
    store = _store(tmp_path)
    store.save("forêt", {"z": "dépression"})
    texte = (tmp_path / "configs" / "forêt.json").read_text(encoding="utf-8")
    assert "dépression" in texte
    assert json.loads(texte) == {"z": "dépression"}


# --- noms invalides --------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", "a/b", r"a\b", "a:b", "a*b",
                                  "a?b", 'a"b', "a<b", "a>b", "a|b",
                                  ".", "..", "con", "a\tb", "a\nb"])
def test_nom_invalide_refuse_a_l_enregistrement(tmp_path, name):
    with pytest.raises(InvalidConfigName):
        _store(tmp_path).save(name, {})


def test_nom_invalide_n_ecrit_rien(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(InvalidConfigName):
        store.save("a/b", {})
    assert not (tmp_path / "configs").exists()


def test_espaces_de_bord_rognes(tmp_path):
    store = _store(tmp_path)
    store.save("  Zone A  ", {})
    assert store.list_names() == ["Zone A"]


# --- supprimer / renommer --------------------------------------------------

def test_supprimer_retire_de_la_liste(tmp_path):
    store = _store(tmp_path)
    store.save("a", {})
    store.save("b", {})
    store.delete("a")
    assert store.list_names() == ["b"]


def test_supprimer_un_nom_absent_leve(tmp_path):
    with pytest.raises(FileNotFoundError):
        _store(tmp_path).delete("fantome")


def test_renommer_deplace_le_contenu(tmp_path):
    store = _store(tmp_path)
    store.save("avant", {"v": 1})
    store.rename("avant", "après")
    assert store.list_names() == ["après"]
    assert store.load("après") == {"v": 1}


def test_renommer_vers_un_nom_pris_leve_et_ne_detruit_rien(tmp_path):
    store = _store(tmp_path)
    store.save("a", {"v": 1})
    store.save("b", {"v": 2})
    with pytest.raises(FileExistsError):
        store.rename("a", "b")
    assert store.load("a") == {"v": 1}
    assert store.load("b") == {"v": 2}


def test_renommer_vers_le_meme_nom_ne_leve_pas(tmp_path):
    store = _store(tmp_path)
    store.save("a", {"v": 1})
    store.rename("a", "a")
    assert store.load("a") == {"v": 1}


def test_renommer_change_seulement_la_casse(tmp_path):
    store = _store(tmp_path)
    store.save("zone a", {"v": 1})
    store.rename("zone a", "Zone A")
    assert store.list_names() == ["Zone A"]


def test_renommer_vers_un_nom_invalide_leve(tmp_path):
    store = _store(tmp_path)
    store.save("a", {})
    with pytest.raises(InvalidConfigName):
        store.rename("a", "a/b")
    assert store.list_names() == ["a"]


def test_renommer_un_nom_absent_leve(tmp_path):
    with pytest.raises(FileNotFoundError):
        _store(tmp_path).rename("fantome", "b")
