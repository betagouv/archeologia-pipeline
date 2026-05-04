"""
Type aliases et helpers transverses partagés par tous les modules du pipeline.
"""
from __future__ import annotations

from typing import Any, Callable


LogFn = Callable[[str], None]
CancelCheckFn = Callable[[], bool]
CancelFn = Callable[[], bool]
ProgressFn = Callable[[int], None]


def safe_float(value: Any, default: float) -> float:
    """Convertit ``value`` en float, retourne ``default`` en cas d'échec.

    Utilisé pour parser des valeurs qui peuvent venir de la config UI
    (strings) sans propager les ``ValueError`` aux call sites.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
