"""Utilitaires partagés entre les runners (logging, config, chemins RVT)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..progress_reporter import ProgressReporter
    from ..structured_logger import StructuredLogger


# ------------------------------------------------------------------ #
#  Logging de section                                                  #
# ------------------------------------------------------------------ #

_SECTION_ICONS = {
    "download": "📥",
    "process": "🔧",
    "mnt": "🔧",
    "cv": "🤖",
    "info": "ℹ️",
}


def log_section(
    title: str,
    tag: str,
    *,
    slog: Optional["StructuredLogger"],
    reporter: "ProgressReporter",
) -> None:
    """Affiche un bandeau de section via le StructuredLogger ou en fallback texte."""
    if slog:
        slog.section(title, tag)
    else:
        icon = _SECTION_ICONS.get(tag, "▶")
        reporter.info("")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info(f"{icon} {title}")
        reporter.info("════════════════════════════════════════════════════════════")


# ------------------------------------------------------------------ #
#  Conversion float sûre                                               #
# ------------------------------------------------------------------ #

def safe_float(value: Any, default: float) -> float:
    """Convertit *value* en float, retourne *default* en cas d'échec."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ #
#  Résolution du chemin RVT cible                                      #
# ------------------------------------------------------------------ #

def resolve_rvt_tif_dir(
    output_dir: Path,
    target_rvt: str,
    output_structure: Dict[str, Any],
    rvt_params: Dict[str, Any],
) -> Path:
    """Construit ``indices/<PRODUCT>/tif`` à partir de la config."""
    from ...pipeline.output_paths import resolve_rvt_tif_dir as _resolve

    return _resolve(output_dir, target_rvt, output_structure, rvt_params)
