"""Lecture de ``metadata.txt`` à la racine du plugin.

``metadata.txt`` reste la source unique de vérité pour la version
(c'est le fichier qu'inspecte QGIS) ; ce helper le parse au runtime
pour exposer la version à l'UI sans dupliquer la valeur.
"""
from __future__ import annotations

import configparser
from functools import lru_cache
from pathlib import Path


_FALLBACK = "?"


def _metadata_path() -> Path:
    return Path(__file__).resolve().parents[2] / "metadata.txt"


@lru_cache(maxsize=1)
def get_plugin_version(path: Path | None = None) -> str:
    """Retourne la version déclarée dans ``[general] version`` de ``metadata.txt``.

    En cas d'erreur (fichier absent, section manquante, parsing
    invalide), retourne ``"?"`` : l'affichage d'une version est
    cosmétique, il ne doit jamais faire crasher l'UI.
    """
    target = path or _metadata_path()
    try:
        parser = configparser.ConfigParser()
        if not parser.read(target, encoding="utf-8"):
            return _FALLBACK
        return parser.get("general", "version", fallback=_FALLBACK) or _FALLBACK
    except (configparser.Error, OSError):
        return _FALLBACK
