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
    """Construit ``results/RVT/<type>[<suffix>]/tif`` à partir de la config."""
    from ...pipeline.ign.products.rvt_naming import get_rvt_param_suffix

    rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
    base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
    type_dir_base = str(rvt_cfg.get(target_rvt, target_rvt))
    param_suffix = get_rvt_param_suffix(target_rvt, rvt_params)
    type_dir_name = f"{type_dir_base}{param_suffix}" if param_suffix else type_dir_base
    return (output_dir / "results") / base_dir_name / type_dir_name / "tif"
