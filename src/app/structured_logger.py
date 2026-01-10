from __future__ import annotations

import time
from typing import Callable, Optional

LogFunc = Callable[[str], None]


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
