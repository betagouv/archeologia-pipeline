from __future__ import annotations

import logging

from typing import Any


class QtProgressReporter:
    def __init__(self, logger: logging.Logger, emitter: Any):
        self._logger = logger
        self._emitter = emitter

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def stage(self, msg: str) -> None:
        try:
            self._emitter.stage.emit(str(msg))
        except Exception:
            pass

    def progress(self, pct: int) -> None:
        try:
            self._emitter.progress.emit(int(pct))
        except Exception:
            pass

    def load_layers(self, vrt_paths: list, shapefile_paths: list) -> None:
        """Émet le signal pour charger les couches dans le projet QGIS."""
        try:
            self._emitter.load_layers.emit(vrt_paths, shapefile_paths)
        except Exception:
            pass
