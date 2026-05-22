from __future__ import annotations

import logging

from typing import Any

from .progress_reporter import USER_INFO


class QtProgressReporter:
    """Reporter qui émet vers un :class:`logging.Logger` Python.

    Le logger est configuré côté UI avec deux handlers :

    - ``QtLogHandler`` filtré à ``USER_INFO`` (l'utilisateur ne voit que
      les messages narratifs et les warnings/errors) ;
    - ``FileHandler`` non filtré (tout va dans le fichier de log).

    Cette classe ignore donc complètement la question du filtrage : elle
    se contente d'émettre au bon niveau, c'est la chaîne de handlers qui
    décide qui voit quoi.
    """

    def __init__(self, logger: logging.Logger, emitter: Any):
        self._logger = logger
        self._emitter = emitter

    # ------------------------------------------------------------------
    # Canal technique (fichier-only après filtrage UI)
    # ------------------------------------------------------------------
    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ------------------------------------------------------------------
    # Canal narratif (UI + fichier)
    # ------------------------------------------------------------------
    def user_info(self, msg: str) -> None:
        self._logger.log(USER_INFO, msg)

    def user_warning(self, msg: str) -> None:
        # WARNING (30) > USER_INFO (25), donc visible UI.
        self._logger.warning(msg)

    def user_success(self, msg: str) -> None:
        # Pas de niveau "SUCCESS" en stdlib ; on se cale sur USER_INFO
        # mais avec un préfixe visuel pour différencier.
        self._logger.log(USER_INFO, msg)

    def user_info_transient(self, msg: str, group: str) -> None:
        """Émet à USER_INFO en marquant le record comme "transient".

        L'attribut ``transient_group`` est lu par :class:`QtLogHandler`
        qui le route vers un signal Qt distinct ; côté zone log Qt, la
        ligne précédente du même ``group`` est remplacée plutôt que
        d'ajouter une nouvelle ligne. Le ``FileHandler`` du pipeline
        ignore cet attribut et écrit la ligne normalement dans le
        ``.txt`` — toutes les sous-étapes restent tracées.
        """
        self._logger.log(USER_INFO, msg, extra={"transient_group": str(group)})

    # ------------------------------------------------------------------
    # Signaux UI directs (barre de progression, étape courante)
    # ------------------------------------------------------------------
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

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        """Émet le signal pour charger les couches dans le projet QGIS."""
        try:
            self._emitter.load_layers.emit(vrt_paths, shapefile_paths, class_colors or [])
        except Exception:
            pass

    def metric(self, current: int, total: int, label: str) -> None:
        """Compteur structuré de progression d'une phase (i/n) pour l'UI."""
        try:
            self._emitter.metric.emit(int(current), int(total), str(label))
        except Exception:
            pass

    def stage_id(self, stage: str) -> None:
        """ID d'étape sémantique → avancée de la timeline mode-aware."""
        try:
            self._emitter.stage_id.emit(str(stage))
        except Exception:
            pass

    def busy(self, active: bool) -> None:
        """Bascule barre indéterminée (True) / déterminée (False)."""
        try:
            self._emitter.busy.emit(bool(active))
        except Exception:
            pass
