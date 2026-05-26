"""RunView — vue d'exécution du pipeline (en-tête + timeline + journal).

Consomme les signaux d'un :class:`log_bridge.QtLogEmitter` alimenté par le
``QtProgressReporter`` côté worker. La timeline est **mode-aware** : sa
séquence d'étapes est dérivée du ``data_mode`` + activation CV via
:func:`app.progress_stages.build_stage_sequence`, et elle avance sur le canal
sémantique ``stage_id`` (plus de matching texte fragile). La barre de
progression suit ``progress`` (garde monotone : jamais en arrière) et bascule
en indéterminé sur ``busy`` (régime large-raster mono-image). Les compteurs
live (« 8/12 dalles ») viennent du canal structuré ``metric``.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from qgis.PyQt.QtCore import Qt, QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QTextCursor
from qgis.PyQt.QtWidgets import (
    QApplication,
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
from ..app.progress_stages import STAGE_LABELS, build_stage_sequence
from .layer_loader import load_result_layers
from .log_bridge import QtLogEmitter, QtLogHandler


def _fmt_hms(seconds: float) -> str:
    """Durée en h:mm:ss (ex. 134 → « 0:02:14 », 3722 → « 1:02:02 », 0 → « 0:00:00 »)."""
    h, rem = divmod(int(max(0.0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_since(seconds: float) -> str:
    """Ancienneté courte (« 12 s », « 3 min »)."""
    s = int(max(0.0, seconds))
    return f"{s} s" if s < 60 else f"{s // 60} min"


def _level_category(levelname: str) -> str:
    """Catégorie de filtre du journal à partir du niveau de log."""
    if levelname == "WARNING":
        return "warn"
    if levelname in ("ERROR", "CRITICAL"):
        return "err"
    return "info"  # USER_INFO, INFO…


class _TimelineStep(QFrame):
    """Pastille numérotée + libellé + sous-libellé (statique · compteur) + chrono."""

    def __init__(self, index: int, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("RunStep")
        self._index = index
        self._static_sub = ""
        self._count = ""
        # Marges uniformes (tous états) → l'encadré de l'étape active n'introduit
        # aucun décalage relatif entre étapes.
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)
        lay.setAlignment(Qt.AlignHCenter)
        self._circle = QLabel(str(index + 1))
        self._circle.setObjectName("RunStepCircle")
        self._circle.setFixedSize(24, 24)
        self._circle.setAlignment(Qt.AlignCenter)
        self._caption = QLabel(label)
        self._caption.setObjectName("RunStepCaption")
        self._caption.setAlignment(Qt.AlignCenter)
        # Sous-libellé + chrono TOUJOURS présents (même vides) → hauteur stable.
        self._subtitle = QLabel("")
        self._subtitle.setObjectName("RunStepSubtitle")
        self._subtitle.setAlignment(Qt.AlignCenter)
        self._timing = QLabel("")
        self._timing.setObjectName("RunStepTiming")
        self._timing.setAlignment(Qt.AlignCenter)
        self._timing.setFixedWidth(64)  # ~"9:59:59" → pas de reflow quand ça défile
        lay.addWidget(self._circle, 0, Qt.AlignHCenter)
        lay.addWidget(self._caption, 0, Qt.AlignHCenter)
        lay.addWidget(self._subtitle, 0, Qt.AlignHCenter)
        lay.addWidget(self._timing, 0, Qt.AlignHCenter)
        self.set_state("pending")

    def set_state(self, state: str) -> None:
        self._circle.setText("✓" if state == "done" else str(self._index + 1))
        for w in (self, self._circle, self._caption, self._subtitle, self._timing):
            w.setProperty("state", state)
            w.style().unpolish(w)
            w.style().polish(w)

    def set_subtitle(self, text: str) -> None:
        self._static_sub = text or ""
        self._compose_sub()

    def set_count(self, text: str) -> None:
        self._count = text or ""
        self._compose_sub()

    def _compose_sub(self) -> None:
        parts = [p for p in (self._static_sub, self._count) if p]
        self._subtitle.setText(" · ".join(parts))

    def set_timing(self, text: str) -> None:
        self._timing.setText(text or "")


class _Timeline(QWidget):
    """Bandeau d'étapes reconstruit selon le mode (``set_stages``).

    Les pastilles ne sont plus figées : ``set_stages`` recrée les
    :class:`_TimelineStep` à partir de la séquence d'IDs sémantiques du mode
    courant, de sorte qu'aucune pastille morte (téléchargement absent,
    détection désactivée…) ne reste affichée.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._steps: List[_TimelineStep] = []
        self._stage_ids: List[str] = []

    def set_stages(self, stage_ids: List[str], labels: Dict[str, str]) -> None:
        # Purge les widgets existants (pastilles + lignes de liaison).
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._steps = []
        self._stage_ids = list(stage_ids)
        for i, sid in enumerate(self._stage_ids):
            step = _TimelineStep(i, labels.get(sid, sid))
            self._steps.append(step)
            self._lay.addWidget(step)
            if i < len(self._stage_ids) - 1:
                line = QFrame()
                line.setObjectName("RunStepLine")
                line.setFixedHeight(2)
                self._lay.addWidget(line, 1)

    def set_active(self, bucket: int) -> None:
        for i, step in enumerate(self._steps):
            if i < bucket:
                step.set_state("done")
            elif i == bucket:
                step.set_state("active")
            else:
                step.set_state("pending")

    def set_timing(self, i: int, text: str) -> None:
        if 0 <= i < len(self._steps):
            self._steps[i].set_timing(text)

    def set_count(self, i: int, text: str) -> None:
        if 0 <= i < len(self._steps):
            self._steps[i].set_count(text)

    def set_step_subtitles(self, subs: Dict[str, str]) -> None:
        """Sous-libellés indexés par ID d'étape (``Stage.PRODUCTS`` …)."""
        for i, step in enumerate(self._steps):
            step.set_subtitle(subs.get(self._stage_ids[i], ""))

    def mark_all_done(self) -> None:
        for step in self._steps:
            step.set_state("done")

    def reset(self) -> None:
        for step in self._steps:
            step.set_state("pending")
            step.set_timing("")
            step.set_count("")


class RunView(QWidget):
    """Vue d'exécution : en-tête + timeline + barre de progression + journal."""

    run_started = pyqtSignal()
    run_finished = pyqtSignal()

    def __init__(self, config_ref: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._config = config_ref or {}
        self._cancel_event = threading.Event()
        self._bucket = -1
        self._running = False
        self._last_stage_text = ""
        self._run_started_at: Optional[float] = None
        # Timeline mode-aware : séquence d'IDs + mapping ID→bucket + sous-libellés.
        self._stage_ids: List[str] = []
        self._stage_bucket_map: Dict[str, int] = {}
        self._subtitles: Dict[str, str] = {}
        # Chronométrage + compteurs par étape (purement UI), dimensionnés sur
        # la séquence d'étapes courante (cf. _apply_stage_sequence).
        self._step_started: List[Optional[float]] = []
        self._step_elapsed: List[Optional[float]] = []
        self._active_started: Optional[float] = None
        self._ui_timer: Optional[QTimer] = None
        # Barre : dernière valeur déterminée (pour restaurer après un passage
        # en indéterminé) + flag d'état indéterminé.
        self._last_progress = 0
        self._indeterminate = False
        # Journal : modèle d'entrées (catégorie, texte) + état transient + filtres.
        self._log_entries: List[list] = []
        self._transient_group: Optional[str] = None
        self._transient_entry: Optional[list] = None
        self._transient_block: Optional[int] = None
        self._log_show = {"info": True, "warn": True, "err": True}

        # ── Logger + pont Qt ──
        self._logger = logging.getLogger("archeologia_pipeline")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._emitter = QtLogEmitter()
        self._emitter.message.connect(self._append_log)
        self._emitter.message_transient.connect(self._update_transient_log)
        self._emitter.progress.connect(self._set_progress)
        self._emitter.stage.connect(self._set_stage)
        self._emitter.stage_id.connect(self._on_stage_id)
        self._emitter.busy.connect(self._on_busy)
        self._emitter.run_enabled.connect(self._on_run_enabled)
        self._emitter.load_layers.connect(self._on_load_layers)
        self._emitter.metric.connect(self._on_metric)
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
        # Timeline initiale (au cas où la vue est affichée avant le 1er run) :
        # dérivée de la config courante, ré-appliquée fermement dans start_run.
        self._apply_stage_sequence()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # En-tête : « Étape N/M · <étape> » (M = nb d'étapes du mode) + sous-ligne
        # (texte + démarré il y a) + compteur live à droite.
        header = QFrame()
        header.setObjectName("RunHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self._run_step_label = QLabel("En attente")
        self._run_step_label.setObjectName("RunHeaderStep")
        self._run_sub_label = QLabel("")
        self._run_sub_label.setObjectName("RunHeaderSub")
        titles.addWidget(self._run_step_label)
        titles.addWidget(self._run_sub_label)
        self._run_metric_label = QLabel("")
        self._run_metric_label.setObjectName("RunHeaderMetric")
        hl.addLayout(titles, 1)
        hl.addWidget(self._run_metric_label, 0, Qt.AlignVCenter)
        root.addWidget(header)

        self._timeline = _Timeline()
        root.addWidget(self._timeline)

        self._progress = QProgressBar()
        self._progress.setObjectName("RunProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        jhead = QHBoxLayout()
        jhead.setSpacing(8)
        jtitle = QLabel("Journal d'exécution")
        jtitle.setObjectName("RunJournalTitle")
        copy_btn = QPushButton("Copier")
        copy_btn.setObjectName("GhostButton")
        copy_btn.clicked.connect(self._copy_journal)
        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("GhostButton")
        clear_btn.clicked.connect(self._clear_journal)
        jhead.addWidget(jtitle)
        jhead.addStretch(1)
        jhead.addWidget(copy_btn)
        jhead.addWidget(clear_btn)
        root.addLayout(jhead)

        self._journal = QPlainTextEdit()
        self._journal.setObjectName("RunJournal")
        self._journal.setReadOnly(True)
        self._journal.setMaximumBlockCount(5000)
        root.addWidget(self._journal, 1)

        actions = QHBoxLayout()
        self._open_dir_btn = QPushButton("📁 Ouvrir le dossier")
        self._open_dir_btn.setObjectName("GhostButton")
        self._open_dir_btn.clicked.connect(self._open_output_dir)
        self._open_log_btn = QPushButton("📄 Log complet")
        self._open_log_btn.setObjectName("GhostButton")
        self._open_log_btn.clicked.connect(self._open_log)
        actions.addWidget(self._open_dir_btn)
        actions.addWidget(self._open_log_btn)
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

    def set_step_subtitles(self, subs: Dict[str, str]) -> None:
        """Sous-libellés statiques de la timeline, indexés par ID d'étape.

        Mémorisés puis ré-appliqués après chaque reconstruction de la timeline
        (``_apply_stage_sequence``), pour survivre au changement de mode.
        """
        self._subtitles = subs or {}
        self._timeline.set_step_subtitles(self._subtitles)

    def _apply_stage_sequence(self) -> None:
        """(Re)construit la timeline selon le mode + activation CV de la config.

        Source de vérité : ``app.files.data_mode`` + ``computer_vision.enabled``.
        Réinitialise les structures de chronométrage à la nouvelle longueur.
        """
        files = (self._config.get("app") or {}).get("files") or {}
        mode = files.get("data_mode") or "ign_laz"
        cv = (self._config.get("computer_vision") or {})
        cv_enabled = bool(cv.get("enabled"))
        try:
            stage_ids = build_stage_sequence(mode, cv_enabled)
        except ValueError:
            stage_ids = build_stage_sequence("ign_laz", cv_enabled)
        self._stage_ids = stage_ids
        self._stage_bucket_map = {sid: i for i, sid in enumerate(stage_ids)}
        self._timeline.set_stages(stage_ids, STAGE_LABELS)
        self._timeline.set_step_subtitles(self._subtitles)
        self._step_started = [None] * len(stage_ids)
        self._step_elapsed = [None] * len(stage_ids)
        self._bucket = -1

    def start_run(self, config: dict) -> None:
        """Valide la config puis lance le pipeline dans un thread worker."""
        if self._running:
            return
        self._config = config or {}
        # Reconstruit la timeline pour le mode du run avant tout affichage.
        self._apply_stage_sequence()

        try:
            from ..app.run_context import build_run_context, validate_run_context
            ctx = build_run_context(self._config)
            errors, warnings = validate_run_context(ctx)
        except Exception as e:  # noqa: BLE001
            self._reset_view()
            self._append_log("ERROR", f"❌ Configuration invalide : {e}")
            return

        if errors:
            self._reset_view()
            self._append_log("ERROR", "❌ Impossible de lancer — corrigez les points suivants :")
            for err in errors:
                self._append_log("ERROR", f"   • {err}")
            return

        self._reset_view()
        for w in warnings:
            self._logger.warning(w)
        self._logger.log(USER_INFO, "Lancement du pipeline…")

        self._running = True
        self._run_started_at = time.monotonic()
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
            from ..pipeline.cancellation import PipelineCancelled

            reporter = QtProgressReporter(self._logger, self._emitter)
            with file_logging(ctx.output_dir, reporter):
                PipelineController().run(
                    ctx=ctx, reporter=reporter, cancel=CancelToken(self._cancel_event)
                )
        except PipelineCancelled:
            # Annulation propre (filet de sécurité) : jamais journalisée comme erreur.
            self._logger.log(USER_INFO, "Pipeline annulé.")
        except Exception:
            self._logger.exception("Erreur pendant l'exécution du pipeline")
        finally:
            self._emitter.run_enabled.emit(True)

    # ------------------------------------------------------------------
    # Slots (signaux émis depuis le worker → thread UI)
    # ------------------------------------------------------------------
    def _reset_view(self) -> None:
        self._bucket = -1
        self._last_stage_text = ""
        self._timeline.reset()  # états + chronos + compteurs (garde les sous-libellés)
        # Barre : retour en mode déterminé, remise à zéro.
        self._indeterminate = False
        self._progress.setRange(0, 100)
        self._last_progress = 0
        self._progress.setValue(0)
        self._run_metric_label.setText("")
        self._update_header()
        self._clear_journal()
        self._step_started = [None] * len(self._stage_ids)
        self._step_elapsed = [None] * len(self._stage_ids)
        self._active_started = None
        if self._ui_timer is not None:
            self._ui_timer.stop()

    def _maybe_scroll(self) -> None:
        # Auto-défilement toujours actif (option UI retirée).
        sb = self._journal.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy_journal(self) -> None:
        QApplication.clipboard().setText(self._journal.toPlainText())

    def _clear_journal(self) -> None:
        self._journal.clear()
        self._log_entries = []
        self._transient_group = None
        self._transient_entry = None
        self._transient_block = None

    def _render_journal(self) -> None:
        """Reconstruit le journal filtré depuis le modèle d'entrées."""
        lines = [t for (c, t) in self._log_entries if self._log_show.get(c, True)]
        self._journal.setPlainText("\n".join(lines))
        # Le pointeur transient devient invalide après reconstruction : la
        # prochaine sous-progression repartira sur une nouvelle ligne.
        self._transient_block = None
        self._transient_group = None
        self._maybe_scroll()

    def _append_log(self, level: str, msg: str) -> None:
        cat = _level_category(level)
        self._log_entries.append([cat, msg])
        self._transient_group = None
        self._transient_entry = None
        self._transient_block = None
        if self._log_show.get(cat, True):
            self._journal.appendPlainText(msg)
            self._maybe_scroll()

    def _update_transient_log(self, group: str, level: str, msg: str) -> None:
        cat = _level_category(level)
        same = group == self._transient_group
        if same and self._transient_entry is not None:
            self._transient_entry[1] = msg
        else:
            self._transient_entry = [cat, msg]
            self._log_entries.append(self._transient_entry)
            self._transient_group = group
            self._transient_block = None
        if not self._log_show.get(cat, True):
            return  # catégorie masquée : entrée mise à jour silencieusement
        doc = self._journal.document()
        if same and self._transient_block is not None:
            block = doc.findBlockByNumber(self._transient_block)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.StartOfBlock)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.insertText(msg)
                self._maybe_scroll()
                return
        self._journal.appendPlainText(msg)
        self._transient_block = doc.blockCount() - 1
        self._maybe_scroll()

    def _set_progress(self, value: int) -> None:
        # Garde monotone : en mode indéterminé on ignore les valeurs ; sinon on
        # borne à [0,100] et on ne redescend jamais (plus de recul de la barre).
        if self._indeterminate:
            return
        v = max(0, min(100, int(value)))
        if v < self._last_progress:
            return
        self._last_progress = v
        self._progress.setValue(v)

    def _set_stage(self, text: str) -> None:
        # ``stage(text)`` ne porte plus que l'étiquette descriptive du header ;
        # l'avancée de la timeline passe par le canal sémantique ``stage_id``.
        self._last_stage_text = text or ""
        self._update_header()

    def _on_stage_id(self, stage_id: str) -> None:
        bucket = self._stage_bucket_map.get(stage_id)
        if bucket is not None and bucket > self._bucket:
            self._enter_bucket(bucket)
        self._update_header()

    def _on_busy(self, active: bool) -> None:
        """Bascule barre indéterminée (phase sans signal fin) / déterminée."""
        if active and not self._indeterminate:
            self._indeterminate = True
            self._progress.setRange(0, 0)  # marquee animé
        elif not active and self._indeterminate:
            self._indeterminate = False
            self._progress.setRange(0, 100)
            self._progress.setValue(self._last_progress)

    def _on_metric(self, current: int, total: int, label: str) -> None:
        if self._bucket < 0:
            return
        text = f"{current}/{total} {label}"
        self._timeline.set_count(self._bucket, text)
        self._run_metric_label.setText(text)

    def _update_header(self) -> None:
        if self._bucket < 0:
            self._run_step_label.setText("Préparation…" if self._running else "En attente")
            self._run_sub_label.setText("")
            return
        label = STAGE_LABELS.get(self._stage_ids[self._bucket], "") if self._stage_ids else ""
        self._run_step_label.setText(
            f"Étape {self._bucket + 1}/{len(self._stage_ids)} · {label}"
        )
        parts = []
        if self._last_stage_text:
            parts.append(self._last_stage_text)
        if self._running and self._run_started_at is not None:
            parts.append(f"démarré il y a {_fmt_since(time.monotonic() - self._run_started_at)}")
        self._run_sub_label.setText(" · ".join(parts))

    def _enter_bucket(self, bucket: int) -> None:
        """Transition d'étape : fige le chrono de la précédente, démarre la
        nouvelle, lance le rafraîchissement live."""
        now = time.monotonic()
        if 0 <= self._bucket < len(self._stage_ids) and self._active_started is not None:
            self._step_elapsed[self._bucket] = now - self._active_started
            self._timeline.set_timing(self._bucket, _fmt_hms(self._step_elapsed[self._bucket]))
        self._bucket = bucket
        self._step_started[bucket] = now
        self._active_started = now
        self._timeline.set_active(bucket)
        self._timeline.set_timing(bucket, "0:00:00")
        self._ensure_timer()

    def _ensure_timer(self) -> None:
        if self._ui_timer is None:
            self._ui_timer = QTimer(self)
            self._ui_timer.setInterval(500)
            self._ui_timer.timeout.connect(self._tick_active)
        if not self._ui_timer.isActive():
            self._ui_timer.start()

    def _tick_active(self) -> None:
        if not self._running or self._active_started is None:
            return
        if 0 <= self._bucket < len(self._stage_ids):
            self._timeline.set_timing(
                self._bucket, _fmt_hms(time.monotonic() - self._active_started)
            )
        self._update_header()  # rafraîchit « démarré il y a X »

    def _on_run_enabled(self, enabled: bool) -> None:
        if not enabled:
            return
        self._running = False
        self._cancel_btn.setEnabled(False)
        self._cancel_event.clear()
        # Filet de sécurité : ne jamais rester bloqué en barre indéterminée.
        if self._indeterminate:
            self._indeterminate = False
            self._progress.setRange(0, 100)
            self._progress.setValue(self._last_progress)
        # Fige le chrono de l'étape active courante puis arrête le rafraîchissement.
        if 0 <= self._bucket < len(self._stage_ids) and self._active_started is not None:
            self._step_elapsed[self._bucket] = time.monotonic() - self._active_started
            self._timeline.set_timing(self._bucket, _fmt_hms(self._step_elapsed[self._bucket]))
        self._active_started = None
        if self._ui_timer is not None:
            self._ui_timer.stop()
        if self._stage_ids and self._bucket >= len(self._stage_ids) - 1:
            self._timeline.mark_all_done()
        self.run_finished.emit()

    def _on_load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list) -> None:
        cv = (self._config or {}).get("computer_vision") or {}
        try:
            conf = float(cv.get("confidence_threshold", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        load_result_layers(vrt_paths, shapefile_paths, class_colors, self._logger, conf)

    # ------------------------------------------------------------------
    # Accès dossier de sortie / log
    # ------------------------------------------------------------------
    def _output_dir(self) -> Optional[Path]:
        if not isinstance(self._config, dict):
            return None
        p = ((self._config.get("app") or {}).get("files") or {}).get("output_dir")
        return Path(p) if p else None

    def _open_output_dir(self) -> None:
        d = self._output_dir()
        if d and d.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _open_log(self) -> None:
        d = self._output_dir()
        if not (d and d.is_dir()):
            return
        logs = sorted(d.glob("pipeline_log_*.txt"))
        if logs:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs[-1])))
