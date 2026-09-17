"""Bibliothèque de configurations nommées — module pur.

« Enregistrer la config » ouvrait un navigateur de fichiers : l'archéologue
choisissait un dossier au hasard, et « Charger une config » lui redemandait de
le retrouver. Depuis le 2026-09-17 (demande utilisateur), les configurations
enregistrées vivent dans UN dossier connu et le chargement se fait par un menu
qui liste ce dossier.

Ce dossier est ``<profil QGIS>/archeologia/configs/`` — le profil, **pas** le
dossier du plugin, remplacé à chaque mise à jour par ZIP (CFG-02 : c'est
exactement ce qui avait fait perdre ``last_ui_config.json``).

Pas de Qt, pas de QGIS ici : seulement le dossier et les noms. Le câblage vit
dans ``ui/wizard_dialog.py`` et ``ui/dialogs/config_dialogs.py``, qui ne sont
pas collectés par pytest.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

#: Un nom de configuration est un nom de FICHIER : ce qui est interdit à un
#: fichier l'est à une config. Refuser à la saisie vaut mieux que translittérer
#: en silence — « Zone A/B » deviendrait « Zone A_B » et l'utilisateur ne
#: retrouverait pas son nom dans le menu.
_FORBIDDEN_CHARS = set('/:*?"<>|') | {chr(92)}  # antislash inclus

#: Noms de périphériques DOS : Windows refuse d'ouvrir « con.json » quel que
#: soit le dossier. Le plugin tourne majoritairement sous Windows.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class InvalidConfigName(ValueError):
    """Nom de configuration inutilisable comme nom de fichier."""


def normalize_name(name: str) -> str:
    """Rogne les espaces de bord et refuse ce qui ne peut pas être un fichier."""
    clean = str(name).strip()
    if not clean:
        raise InvalidConfigName("Le nom ne peut pas être vide.")
    if any(c < " " for c in clean):
        raise InvalidConfigName("Le nom ne peut pas contenir de caractère de contrôle.")
    if set(clean) & _FORBIDDEN_CHARS:
        raise InvalidConfigName(
            "Le nom ne peut pas contenir : " + " ".join(sorted(_FORBIDDEN_CHARS))
        )
    if clean.strip(".") == "" or clean.lower() in _RESERVED:
        raise InvalidConfigName(f"« {clean} » est un nom réservé par le système.")
    return clean


class ConfigStore:
    """Les configurations enregistrées d'un dossier, adressées par leur nom."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path_for(self, name: str) -> Path:
        return self.directory / f"{normalize_name(name)}.json"

    def list_names(self) -> List[str]:
        try:
            entries = list(self.directory.iterdir())
        except OSError:
            return []  # dossier absent : aucune config, pas une erreur
        names = [p.stem for p in entries if p.is_file() and p.suffix.lower() == ".json"]
        return sorted(names, key=str.casefold)

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def load(self, name: str) -> Dict[str, Any]:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {name}")
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Fichier illisible ({path.name}) : {e}") from e

    def save(self, name: str, config: Dict[str, Any]) -> Path:
        """Écriture atomique (tmp + os.replace), comme l'autosave : une coupure
        en cours d'écriture ne laisse pas une config à moitié écrite."""
        path = self.path_for(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def delete(self, name: str) -> None:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {name}")
        path.unlink()

    def rename(self, old: str, new: str) -> Path:
        src = self.path_for(old)
        dst = self.path_for(new)
        if not src.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {old}")
        if dst != src and dst.is_file():
            raise FileExistsError(f"« {normalize_name(new)} » existe déjà.")
        # samefile : sous Windows « zone a » et « Zone A » sont le MÊME fichier —
        # un simple test d'égalité de chemins laisserait passer le rename (voulu,
        # c'est un changement de casse) mais `dst.is_file()` crierait à tort.
        if dst != src and src.resolve() == dst.resolve():
            os.replace(src, dst)  # même fichier, casse différente
            return dst
        src.rename(dst)
        return dst
