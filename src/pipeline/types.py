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


def _format_param_value(value: Any) -> str:
    """Formatte une valeur de paramètre pour :func:`format_params_line`.

    Conventions :
    - ``None`` ou string vide → ``∅`` (signale explicitement l'absence).
    - ``bool`` → ``True`` / ``False``.
    - ``str`` avec espace → entre guillemets pour lisibilité (longues
      expressions PDAL par exemple).
    - ``float`` → format compact ``%g`` (pas de zéros traînants).
    """
    if value is None:
        return "∅"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        if not value:
            return "∅"
        return f'"{value}"' if " " in value else value
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_params_line(step_name: str, params: dict) -> str:
    """Construit une ligne ``[PARAMS <step>] key=value | …`` traçable.

    Utilisé pour logger les paramètres effectivement appliqués à
    chaque étape majeure du pipeline (MNT, RVT, CV…). Le format
    compact tient sur une ligne et reste lisible dans le fichier
    ``pipeline_log_*.txt``. Voir :func:`_format_param_value` pour les
    conventions de formatage.
    """
    parts = [f"{k}={_format_param_value(v)}" for k, v in params.items()]
    body = " | ".join(parts) if parts else "(aucun paramètre)"
    return f"[PARAMS {step_name}] {body}"
