from __future__ import annotations

from typing import Protocol


class ProgressReporter(Protocol):
    def info(self, msg: str) -> None: ...

    def error(self, msg: str) -> None: ...

    def stage(self, msg: str) -> None: ...

    def progress(self, pct: int) -> None: ...

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None: ...


class NullProgressReporter:
    def info(self, msg: str) -> None:
        return

    def error(self, msg: str) -> None:
        return

    def stage(self, msg: str) -> None:
        return

    def progress(self, pct: int) -> None:
        return

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        return
