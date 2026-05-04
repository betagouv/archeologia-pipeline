from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .progress_reporter import ProgressReporter

LogFunc = Callable[[str], None]


_SECTION_ICONS_FALLBACK = {
    "download": "📥",
    "process": "🔧",
    "mnt": "🔧",
    "cv": "🤖",
    "info": "ℹ️",
}


class StructuredLogger:
    """Logger structuré avec sections visuelles pour améliorer la lisibilité."""

    SEPARATOR = "═" * 60
    THIN_SEP = "─" * 60

    ICONS = {
        "start": "🚀",
        "download": "📥",
        "process": "🔧",
        "product": "📦",
        "cv": "🔍",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "time": "⏱️",
        "folder": "📁",
        "file": "📄",
    }

    def __init__(self, log_func: LogFunc):
        self._log = log_func
        self._start_time: Optional[float] = None
        self._section_start: Optional[float] = None

    def start_pipeline(self, mode: str, output_dir: str) -> None:
        self._start_time = time.time()
        self._log("")
        self._log(self.SEPARATOR)
        self._log(f"{self.ICONS['start']} DÉMARRAGE DU PIPELINE")
        self._log(self.SEPARATOR)
        self._log(f"  Mode        : {mode}")
        self._log(f"  Sortie      : {output_dir}")
        self._log(f"  Démarré à   : {time.strftime('%H:%M:%S')}")
        self._log("")

    def end_pipeline(self, success: bool = True, tiles_processed: int = 0, tiles_total: int = 0, products: Optional[list] = None) -> None:
        duration = self._format_duration(self._start_time)
        self._log("")
        self._log(self.SEPARATOR)
        if success:
            self._log(f"{self.ICONS['success']} PIPELINE TERMINÉ AVEC SUCCÈS")
        else:
            self._log(f"{self.ICONS['error']} PIPELINE TERMINÉ AVEC ERREURS")
        self._log(self.SEPARATOR)
        self._log(f"  {self.ICONS['time']} Durée totale : {duration}")
        if tiles_total > 0:
            self._log(f"  {self.ICONS['file']} Dalles traitées : {tiles_processed}/{tiles_total}")
        if products:
            self._log(f"  {self.ICONS['product']} Produits : {', '.join(products)}")
        self._log(self.SEPARATOR)
        self._log("")

    def section(self, title: str, icon: str = "info") -> None:
        self._section_start = time.time()
        icon_char = self.ICONS.get(icon, self.ICONS["info"])
        self._log("")
        self._log(self.SEPARATOR)
        self._log(f"{icon_char} {title.upper()}")
        self._log(self.SEPARATOR)

    def subsection(self, title: str) -> None:
        self._log("")
        self._log(self.THIN_SEP)
        self._log(f"  {title}")
        self._log(self.THIN_SEP)

    def item(self, message: str, icon: str = "info", indent: int = 1) -> None:
        icon_char = self.ICONS.get(icon, "")
        prefix = "  " * indent
        if icon_char:
            self._log(f"{prefix}{icon_char} {message}")
        else:
            self._log(f"{prefix}{message}")

    def success(self, message: str, indent: int = 1) -> None:
        self.item(message, "success", indent)

    def error(self, message: str, indent: int = 1) -> None:
        self.item(message, "error", indent)

    def warning(self, message: str, indent: int = 1) -> None:
        self.item(message, "warning", indent)

    def info(self, message: str, indent: int = 1) -> None:
        prefix = "  " * indent
        self._log(f"{prefix}{message}")

    def params(self, step_name: str, params: dict) -> None:
        """Logge les paramètres effectifs d'une étape du pipeline.

        Émis au niveau ``INFO`` standard — visible dans le fichier de
        log mais filtré côté UI Qt (qui ne montre que ``USER_INFO+``).
        Permet à l'utilisateur (ou au support) de retracer post-hoc
        quelles valeurs ont été réellement appliquées à PDAL/RVT/CV
        pour chaque dalle, sans surcharger la fenêtre QGIS.

        Format délégué à :func:`pipeline.types.format_params_line` —
        partagé avec les call-sites pipeline qui n'ont qu'un ``log``
        callable et pas accès à cette instance.
        """
        # Dual-context : en QGIS le plugin est chargé comme package
        # (``archeologia.src.app``) → import relatif. En dev (tests
        # via conftest qui ajoute src/ au sys.path), ``app`` et
        # ``pipeline`` sont top-level → import absolu.
        try:
            from ..pipeline.types import format_params_line
        except ImportError:
            from pipeline.types import format_params_line  # type: ignore[no-redef]
        self._log(format_params_line(step_name, params))

    def progress(self, current: int, total: int, item_name: str = "") -> None:
        pct = (current / total * 100) if total > 0 else 0
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        if item_name:
            self._log(f"  [{bar}] {current}/{total} ({pct:.0f}%) - {item_name}")
        else:
            self._log(f"  [{bar}] {current}/{total} ({pct:.0f}%)")

    def tile_start(self, index: int, total: int, tile_name: str) -> None:
        self._section_start = time.time()
        self._log("")
        self._log(self.THIN_SEP)
        self._log(f"  {self.ICONS['process']} DALLE {index}/{total}: {tile_name}")
        self._log(self.THIN_SEP)

    def tile_end(self, tile_name: str, products_generated: Optional[list] = None) -> None:
        duration = self._format_duration(self._section_start)
        if products_generated:
            self._log(f"    {self.ICONS['success']} Terminé en {duration} → {', '.join(products_generated)}")
        else:
            self._log(f"    {self.ICONS['success']} Terminé en {duration}")

    def products_list(self, products: list, title: str = "Produits générés") -> None:
        self._log(f"  {title}:")
        for p in products:
            self._log(f"    • {p}")

    def preflight_result(self, name: str, status: str, detail: str = "") -> None:
        if status.upper() == "OK":
            icon = self.ICONS["success"]
        elif status.upper() == "WARN":
            icon = self.ICONS["warning"]
        else:
            icon = self.ICONS["error"]
        
        if detail:
            self._log(f"  {icon} {name}: {detail}")
        else:
            self._log(f"  {icon} {name}")

    def _format_duration(self, start_time: Optional[float]) -> str:
        if start_time is None:
            return "N/A"
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            return f"{mins}min {secs}s"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}min"


def create_structured_logger(log_func: LogFunc) -> StructuredLogger:
    return StructuredLogger(log_func)


def log_section(
    title: str,
    tag: str,
    *,
    slog: Optional[StructuredLogger],
    reporter: "ProgressReporter",
) -> None:
    """Affiche un bandeau de section.

    Si ``slog`` est fourni, délègue à :meth:`StructuredLogger.section` ;
    sinon, retombe sur un affichage texte via ``reporter.info``. Garder
    cette fonction *en dehors* de la classe permet de la consommer même
    quand le contexte d'exécution n'a pas instancié de logger structuré.
    """
    if slog:
        slog.section(title, tag)
        return
    icon = _SECTION_ICONS_FALLBACK.get(tag, "▶")
    reporter.info("")
    reporter.info("════════════════════════════════════════════════════════════")
    reporter.info(f"{icon} {title}")
    reporter.info("════════════════════════════════════════════════════════════")
