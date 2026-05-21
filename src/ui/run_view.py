"""RunView — vue d'exécution du pipeline (timeline + barre + journal).

Consomme les signaux d'un :class:`log_bridge.QtLogEmitter` alimenté par le
``QtProgressReporter`` côté worker. La timeline à 5 étapes (Téléchargement, MNT,
Indices, Détection, Finalisation) est dérivée du texte libre des ``stage(msg)``
via une table mot-clé → étape, **monotone** (jamais en arrière). Aucune
modification du reporter : le mapping vit entièrement ici.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QTextCursor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..app.progress_reporter import USER_INFO
from .layer_loader import load_result_layers
from .log_bridge import QtLogEmitter, QtLogHandler

# Étapes de la timeline (ordre = progression).
_STAGES = ["Téléchargement", "MNT", "Indices", "Détection", "Finalisation"]

# Règles texte-libre → indice d'étape (premier match gagne, puis garde monotone).
# Ordonnées par spécificité pour lever les ambiguïtés (« dalles à télécharger »
# contient « dalle » mais doit rester en Téléchargement, etc.).
_STAGE_RULES = [
    ("télécharg", 0),
    ("identification des dalles", 0),
    ("download", 0),
    ("computer vision", 3),
    ("détection", 3),
    ("inférence", 3),
    ("création des index", 4),
    ("index vrt", 4),
    ("chargement des couches", 4),
    ("finalisation", 4),
    ("terminé", 4),
    ("rvt", 2),
    ("traitement des dalles", 2),
    ("indice", 2),
    ("visualisation", 2),
    ("indexation", 1),
    ("traitement dalle", 1),
    ("fusion", 1),
    ("merge", 1),
    ("voisins", 1),
    ("mnt", 1),
    ("nuage", 1),
]


def _stage_bucket(msg: str) -> Optional[int]:
    low = (msg or "").lower()
    for needle, idx in _STAGE_RULES:
        if needle in low:
            return idx
    return None


class _TimelineStep(QFrame):
    """Pastille numérotée + libellé, état pending / active / done."""

    def __init__(self, index: int, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("RunStep")
        self._index = index
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(3)
        lay.setAlignment(Qt.AlignHCenter)
        self._circle = QLabel(str(index + 1))
        self._circle.setObjectName("RunStepCircle")
        self._circle.setFixedSize(24, 24)
        self._circle.setAlignment(Qt.AlignCenter)
        self._caption = QLabel(label)
        self._caption.setObjectName("RunStepCaption")
        self._caption.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._circle, 0, Qt.AlignHCenter)
        lay.addWidget(self._caption, 0, Qt.AlignHCenter)
        self.set_state("pending")

    def set_state(self, state: str) -> None:
        self._circle.setText("✓" if state == "done" else str(self._index + 1))
        for w in (self, self._circle, self._caption):
            w.setProperty("state", state)
            w.style().unpolish(w)
            w.style().polish(w)


class _Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._steps: List[_TimelineStep] = []
        for i, label in enumerate(_STAGES):
            step = _TimelineStep(i, label)
            self._steps.append(step)
            lay.addWidget(step)
            if i < len(_STAGES) - 1:
                line = QFrame()
                line.setObjectName("RunStepLine")
                line.setFixedHeight(2)
                lay.addWidget(line, 1)

    def set_active(self, bucket: int) -> None:
        for i, step in enumerate(self._steps):
            if i < bucket:
                step.set_state("done")
            elif i == bucket:
                step.set_state("active")
            else:
                step.set_state("pending")

    def mark_all_done(self) -> None:
        for step in self._steps:
            step.set_state("done")

    def reset(self) -> None:
        for step in self._steps:
            step.set_state("pending")


class RunView(QWidget):
    """Vue d'exécution : timeline + barre de progression + journal + annulation."""

    run_started = pyqtSignal()
    run_finished = pyqtSignal()

    def __init__(self, config_ref: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._config = config_ref or {}
        self._cancel_event = threading.Event()
        self._bucket = -1
        self._running = False
        self._last_transient_group: Optional[str] = None
        self._last_transient_block: Optional[int] = None

        # ── Logger + pont Qt ──
        self._logger = logging.getLogger("archeologia_pipeline")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._emitter = QtLogEmitter()
        self._emitter.message.connect(self._append_log)
        self._emitter.message_transient.connect(self._update_transient_log)
        self._emitter.progress.connect(self._set_progress)
        self._emitter.stage.connect(self._set_stage)
        self._emitter.run_enabled.connect(self._on_run_enabled)
        self._emitter.load_layers.connect(self._on_load_layers)
        self._log_handler = QtLogHandler(self._emitter)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        # Retire les handlers d'une instance précédente (réouverture du dialogue) :
        # sinon les logs partiraient vers un émetteur mort et le journal resterait
        # muet. On garantit qu'un seul QtLogHandler — celui-ci — est attaché.
        for h in list(self._logger.handlers):
            if isinstance(h, QtLogHandler):
                self._logger.removeHandler(h)
        self._logger.addHandler(self._log_handler)

        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self._timeline = _Timeline()
        root.addWidget(self._timeline)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(10)
        self._stage_label = QLabel("En attente")
        self._stage_label.setObjectName("RunStageLabel")
        self._progress = QProgressBar()
        self._progress.setObjectName("RunProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        bar_row.addWidget(self._stage_label)
        bar_row.addWidget(self._progress, 1)
        root.addLayout(bar_row)

        jhead = QHBoxLayout()
        jhead.setSpacing(8)
        jtitle = QLabel("Journal d'exécution")
        jtitle.setObjectName("RunJournalTitle")
        self._autoscroll_check = QCheckBox("Auto-défilement")
        self._autoscroll_check.setChecked(True)
        copy_btn = QPushButton("Copier")
        copy_btn.setObjectName("GhostButton")
        copy_btn.clicked.connect(self._copy_journal)
        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("GhostButton")
        clear_btn.clicked.connect(self._clear_journal)
        jhead.addWidget(jtitle)
        jhead.addStretch(1)
        jhead.addWidget(self._autoscroll_check)
        jhead.addWidget(copy_btn)
        jhead.addWidget(clear_btn)
        root.addLayout(jhead)

        self._journal = QPlainTextEdit()
        self._journal.setObjectName("RunJournal")
        self._journal.setReadOnly(True)
        self._journal.setMaximumBlockCount(5000)
        root.addWidget(self._journal, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self._cancel_btn = QPushButton("Annuler")
        self._cancel_btn.setObjectName("RunCancelBtn")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.request_cancel)
        actions.addWidget(self._cancel_btn)
        root.addLayout(actions)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def start_run(self, config: dict) -> None:
        """Valide la config puis lance le pipeline dans un thread worker."""
        if self._running:
            return
        self._config = config or {}

        try:
            from ..app.run_context import build_run_context, validate_run_context
            ctx = build_run_context(self._config)
            errors, warnings = validate_run_context(ctx)
        except Exception as e:  # noqa: BLE001
            self._reset_view()
            self._append_log(f"❌ Configuration invalide : {e}")
            return

        if errors:
            self._reset_view()
            self._append_log("❌ Impossible de lancer — corrigez les points suivants :")
            for err in errors:
                self._append_log(f"   • {err}")
            return

        self._reset_view()
        for w in warnings:
            self._logger.warning(w)
        self._logger.log(USER_INFO, "Lancement du pipeline…")

        self._running = True
        self._cancel_event.clear()
        self._cancel_btn.setEnabled(True)
        self.run_started.emit()
        threading.Thread(target=self._worker, args=(ctx,), daemon=True).start()

    def request_cancel(self) -> None:
        if self._running and not self._cancel_event.is_set():
            self._cancel_event.set()
            self._cancel_btn.setEnabled(False)
            self._logger.log(USER_INFO, "Annulation demandée…")

    # ------------------------------------------------------------------
    def _worker(self, ctx) -> None:
        try:
            from ..app.cancel_token import CancelToken
            from ..app.pipeline_controller import PipelineController, file_logging
            from ..app.qt_progress_reporter import QtProgressReporter

            reporter = QtProgressReporter(self._logger, self._emitter)
            with file_logging(ctx.output_dir, reporter):
                PipelineController().run(
                    ctx=ctx, reporter=reporter, cancel=CancelToken(self._cancel_event)
                )
        except Exception:
            self._logger.exception("Erreur pendant l'exécution du pipeline")
        finally:
            self._emitter.run_enabled.emit(True)

    # ------------------------------------------------------------------
    # Slots (signaux émis depuis le worker → thread UI)
    # ------------------------------------------------------------------
    def _reset_view(self) -> None:
        self._bucket = -1
        self._timeline.reset()
        self._progress.setValue(0)
        self._stage_label.setText("Préparation…")
        self._journal.clear()
        self._last_transient_group = None
        self._last_transient_block = None

    def _maybe_scroll(self) -> None:
        if self._autoscroll_check.isChecked():
            sb = self._journal.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _copy_journal(self) -> None:
        QApplication.clipboard().setText(self._journal.toPlainText())

    def _clear_journal(self) -> None:
        self._journal.clear()
        self._last_transient_group = None
        self._last_transient_block = None

    def _append_log(self, msg: str) -> None:
        self._last_transient_group = None
        self._last_transient_block = None
        self._journal.appendPlainText(msg)
        self._maybe_scroll()

    def _update_transient_log(self, group: str, msg: str) -> None:
        doc = self._journal.document()
        if self._last_transient_group == group and self._last_transient_block is not None:
            block = doc.findBlockByNumber(self._last_transient_block)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.StartOfBlock)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.insertText(msg)
                self._maybe_scroll()
                return
        self._journal.appendPlainText(msg)
        self._last_transient_group = group
        self._last_transient_block = doc.blockCount() - 1
        self._maybe_scroll()

    def _set_progress(self, value: int) -> None:
        self._progress.setValue(int(value))

    def _set_stage(self, text: str) -> None:
        self._stage_label.setText(text or "")
        bucket = _stage_bucket(text)
        if bucket is not None and bucket > self._bucket:
            self._bucket = bucket
            self._timeline.set_active(bucket)

    def _on_run_enabled(self, enabled: bool) -> None:
        if not enabled:
            return
        self._running = False
        self._cancel_btn.setEnabled(False)
        self._cancel_event.clear()
        # Pipeline arrivé au bout sans annulation → timeline complète.
        if self._bucket >= len(_STAGES) - 1:
            self._timeline.mark_all_done()
        self.run_finished.emit()

    def _on_load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list) -> None:
        cv = (self._config or {}).get("computer_vision") or {}
        try:
            conf = float(cv.get("confidence_threshold", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        load_result_layers(vrt_paths, shapefile_paths, class_colors, self._logger, conf)
