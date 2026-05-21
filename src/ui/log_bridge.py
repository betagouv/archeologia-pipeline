"""Pont logging Python → signaux Qt pour la zone de journal du wizard.

Le pipeline émet tout via le logger ``archeologia_pipeline``. :class:`QtLogHandler`
filtre à ``USER_INFO`` (l'utilisateur ne voit que les messages narratifs +
warnings/errors ; le détail technique va dans le fichier de log via
``pipeline_controller.file_logging``) et route chaque enregistrement vers un
signal Qt thread-safe porté par :class:`QtLogEmitter`.
"""
from __future__ import annotations

import logging

from qgis.PyQt.QtCore import QObject, pyqtSignal

from ..app.progress_reporter import USER_INFO


class QtLogEmitter(QObject):
    """Signaux Qt consommés par le ``RunView`` (émis depuis le thread worker)."""

    message = pyqtSignal(str)
    # Ligne « transiente » : (group, message). La zone log réécrit la dernière
    # ligne du même ``group`` au lieu d'empiler (sous-progressions Dalle i/N…).
    message_transient = pyqtSignal(str, str)
    progress = pyqtSignal(int)            # 0-100
    stage = pyqtSignal(str)               # libellé d'étape (texte libre)
    run_enabled = pyqtSignal(bool)        # True = run terminé → réactiver l'UI
    load_layers = pyqtSignal(list, list, list)  # (vrt_paths, shp_paths, class_colors)


class QtLogHandler(logging.Handler):
    """Handler filtré à ``USER_INFO`` qui propage les logs vers l'UI."""

    def __init__(self, emitter: QtLogEmitter):
        super().__init__(level=USER_INFO)
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        transient_group = getattr(record, "transient_group", None)
        if transient_group:
            self._emitter.message_transient.emit(str(transient_group), msg)
        else:
            self._emitter.message.emit(msg)
