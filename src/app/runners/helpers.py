"""Helpers internes au package ``app.runners``.

Ce module ne contient que des utilitaires consommés exclusivement par
les runners. Toute fonction transverse aux packages (``app`` ↔ ``pipeline``)
doit rester dans :mod:`pipeline.types` ; ici on expose une copie locale
de :func:`safe_float` pour éviter un import absolu ``from pipeline.X``
au top-level (qui ne fonctionne pas dans le contexte d'exécution QGIS,
où le plugin est chargé sous un nom de package différent).
"""
from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float) -> float:
    """Convertit ``value`` en float, retourne ``default`` en cas d'échec.

    Copie locale de :func:`pipeline.types.safe_float` pour éviter un
    import inter-package au top-level. Si la signature évolue, mettre
    les deux à jour.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
