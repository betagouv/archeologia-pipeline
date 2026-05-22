"""Étape 4 — Lancement : récap propre, puis bascule sur la vue d'exécution.

Deux phases dans un ``QStackedWidget`` interne :

- **Récap** (avant lancement) : panneau « État du système » (préflight lancé en
  tâche de fond), récapitulatif des choix des étapes 1-3, et paramètres avancés
  repliés (workers). Aucune estimation de durée/RAM (le pipeline ne les expose
  pas de façon fiable).
- **Exécution** : le :class:`RunView` (timeline + journal). Le bouton « Lancer »
  de la barre d'actions appelle :meth:`start_run`, qui bascule sur cette vue.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List

from qgis.PyQt.QtCore import QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..run_view import RunView
from ..widgets.card import build_card
from ..widgets.collapsible import CollapsibleSection
from ..widgets.no_wheel import NoWheelSpinBox


@dataclass
class RecapSection:
    """Une ligne du récap : libellé + pastilles + valeur + sous-ligne de détail."""

    label: str
    badges: List[str] = field(default_factory=list)
    value: str = ""
    detail: str = ""


class _FlowLayout(QLayout):
    """Layout qui fait passer les widgets à la ligne quand la largeur déborde
    (recette Qt standard). Utilisé pour les pastilles du récap."""

    def __init__(self, parent=None, hspacing=4, vspacing=4):
        super().__init__(parent)
        self._items: list = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 (signature Qt)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspace
            if next_x - self._hspace > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self._vspace
                next_x = x + hint.width() + self._hspace
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class _PreflightEmitter(QObject):
    """Pont thread worker → thread UI pour les résultats préflight."""

    done = pyqtSignal(int, object)  # (génération, List[CheckResult])


class LaunchPage(QWidget):
    run_started = pyqtSignal()
    run_finished = pyqtSignal()
    workers_changed = pyqtSignal(int)

    def __init__(self, plugin_root, config_ref=None, parent=None):
        super().__init__(parent)
        self._recap_rows: List[QWidget] = []
        self._workers = 4
        self._preflight_gen = 0
        self._pf_emitter = _PreflightEmitter(self)
        self._pf_emitter.done.connect(self._on_preflight_done)
        self._build(config_ref)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self, config_ref) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._stack = QStackedWidget()
        self._recap_page = self._build_recap_page()
        self._run_page = self._build_run_page(config_ref)
        self._stack.addWidget(self._recap_page)  # [0]
        self._stack.addWidget(self._run_page)    # [1]
        root.addWidget(self._stack)

    def _build_recap_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("LaunchScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # Bandeau de validation : rouge si la config est inexécutable, vert sinon.
        self._banner = QLabel("")
        self._banner.setObjectName("ValidationBanner")
        self._banner.setProperty("state", "ok")
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        v.addWidget(self._banner)

        # ① État du système (préflight)
        v.addWidget(self._build_preflight_card())

        # ② Récapitulatif du run
        recap_card, rv = build_card("Récapitulatif du run", "2")
        self._recap_host = QWidget()
        self._recap_layout = QVBoxLayout(self._recap_host)
        self._recap_layout.setContentsMargins(0, 0, 0, 0)
        self._recap_layout.setSpacing(6)
        rv.addWidget(self._recap_host)
        v.addWidget(recap_card)

        # ③ Paramètres avancés (replié) — workers
        self._adv = CollapsibleSection(
            "Paramètres avancés",
            preview_getter=lambda: f"workers : {self._workers}",
        )
        self._adv.set_body(self._build_workers_body())
        v.addWidget(self._adv)

        hint = QLabel("Le run produira un projet QGIS consolidé prêt à ouvrir.")
        hint.setObjectName("LaunchHint")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        v.addWidget(hint)

        v.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page

    def _build_preflight_card(self) -> QWidget:
        card, cv = build_card("État du système", "1")

        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 0, 0, 0)
        pill_row.addStretch(1)
        self._preflight_pill = QLabel("●  …")
        self._preflight_pill.setObjectName("PreflightPill")
        self._preflight_pill.setProperty("kind", "ok")
        pill_row.addWidget(self._preflight_pill)
        cv.addLayout(pill_row)

        self._preflight_host = QWidget()
        self._preflight_grid = QGridLayout(self._preflight_host)
        self._preflight_grid.setContentsMargins(0, 0, 0, 0)
        self._preflight_grid.setHorizontalSpacing(10)
        self._preflight_grid.setVerticalSpacing(3)
        self._preflight_grid.setColumnStretch(2, 1)  # le détail prend la largeur restante
        cv.addWidget(self._preflight_host)
        return card

    def _build_workers_body(self) -> QWidget:
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        lbl = QLabel("Workers parallèles")
        lbl.setObjectName("FieldLabel")
        h.addWidget(lbl)

        minus = QPushButton("−")
        minus.setObjectName("WorkersStep")
        minus.setProperty("role", "micro")
        minus.clicked.connect(lambda: self._set_workers(self._workers - 1))
        h.addWidget(minus)

        self._workers_spin = NoWheelSpinBox()
        self._workers_spin.setObjectName("WorkersSpin")
        self._workers_spin.setRange(1, 16)
        self._workers_spin.setValue(self._workers)
        self._workers_spin.setAlignment(Qt.AlignCenter)
        # Flèches natives masquées : on a déjà des boutons −/+ explicites
        # (sinon les flèches Fusion se compriment dans le champ étroit).
        self._workers_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._workers_spin.setFixedWidth(48)
        self._workers_spin.valueChanged.connect(self._set_workers)
        h.addWidget(self._workers_spin)

        plus = QPushButton("+")
        plus.setObjectName("WorkersStep")
        plus.setProperty("role", "micro")
        plus.clicked.connect(lambda: self._set_workers(self._workers + 1))
        h.addWidget(plus)

        note = QLabel(
            "Nombre de dalles traitées en parallèle. Plus de workers accélère le "
            "traitement sur les machines multi-cœurs."
        )
        note.setObjectName("WorkersNote")
        note.setWordWrap(True)
        h.addWidget(note, 1)
        return body

    def _build_run_page(self, config_ref) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)
        run_card, xv = build_card("Exécution")
        self._run_view = RunView(config_ref)
        self._run_view.run_started.connect(self.run_started)
        self._run_view.run_finished.connect(self.run_finished)
        xv.addWidget(self._run_view)
        v.addWidget(run_card, 1)
        return page

    # ------------------------------------------------------------------
    # Récapitulatif (assemblé par le wizard)
    # ------------------------------------------------------------------
    def update_recap(self, sections: List[RecapSection]) -> None:
        """Reconstruit le récap (rebuild propre : deleteLater des anciennes lignes)."""
        for w in self._recap_rows:
            w.deleteLater()
        self._recap_rows = []
        for sec in sections:
            self._recap_layout.addWidget(self._build_section(sec))

    def _build_section(self, sec: RecapSection) -> QWidget:
        row = QWidget()
        col = QVBoxLayout(row)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        key = QLabel(sec.label)
        key.setObjectName("RecapKey")
        key.setFixedWidth(120)
        top.addWidget(key, 0, Qt.AlignTop)

        if sec.badges:
            badge_host = QWidget()
            flow = _FlowLayout(badge_host)
            for b in sec.badges:
                lbl = QLabel(b)
                lbl.setObjectName("RecapBadge")
                flow.addWidget(lbl)
            top.addWidget(badge_host, 1)
            if sec.value:
                val = QLabel(sec.value)
                val.setObjectName("RecapVal")
                top.addWidget(val, 0, Qt.AlignTop)
        else:
            val = QLabel(sec.value or "—")
            val.setObjectName("RecapVal")
            val.setWordWrap(True)
            top.addWidget(val, 1)
        col.addLayout(top)

        if sec.detail:
            det = QLabel(sec.detail)
            det.setObjectName("RecapDetail")
            det.setWordWrap(True)
            det.setContentsMargins(128, 0, 0, 0)  # aligné sous la valeur (clé 120 + spacing 8)
            col.addWidget(det)

        self._recap_rows.append(row)
        return row

    def set_validation(self, errors: list) -> None:
        """Affiche le bandeau : erreurs bloquantes (rouge) ou « prêt » (vert)."""
        if errors:
            lines = "<br>".join(f"• {e}" for e in errors)
            self._banner.setText(f"<b>⚠ Impossible de lancer — corrigez :</b><br>{lines}")
            self._banner.setProperty("state", "error")
        else:
            self._banner.setText("✓ Configuration valide — prêt à lancer.")
            self._banner.setProperty("state", "ok")
        self._banner.setVisible(True)
        self._banner.style().unpolish(self._banner)
        self._banner.style().polish(self._banner)

    # ------------------------------------------------------------------
    # Paramètres avancés — workers
    # ------------------------------------------------------------------
    def set_workers(self, n: int) -> None:
        """Initialise le compteur depuis la config (sans émettre de signal)."""
        n = max(1, min(16, self._safe_int(n, 4)))
        self._workers = n
        self._workers_spin.blockSignals(True)
        self._workers_spin.setValue(n)
        self._workers_spin.blockSignals(False)
        self._adv.refresh_preview()

    def _set_workers(self, n: int) -> None:
        """Réagit à une interaction utilisateur (spin / boutons −+)."""
        n = max(1, min(16, self._safe_int(n, self._workers)))
        # Resynchronise le spin (cas du clamp aux bornes) sans boucler.
        self._workers_spin.blockSignals(True)
        self._workers_spin.setValue(n)
        self._workers_spin.blockSignals(False)
        if n == self._workers:
            return
        self._workers = n
        self._adv.refresh_preview()
        self.workers_changed.emit(n)

    def workers_value(self) -> int:
        return self._workers

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Préflight (panneau « État du système »)
    # ------------------------------------------------------------------
    def refresh_preflight(self, config: dict) -> None:
        """Lance les vérifications préflight en tâche de fond pour ``config``."""
        self._preflight_gen += 1
        gen = self._preflight_gen
        self._show_preflight_loading()
        try:
            from ...app.pipeline_controller import _files_as_dict
            from ...app.run_context import build_run_context

            ctx = build_run_context(config or {})
            kwargs = dict(
                mode=str(ctx.mode),
                cv_config=ctx.cv.raw,
                products=ctx.processing.products.as_dict(),
                files_config=_files_as_dict(ctx),
                output_dir=ctx.output_dir,
            )
        except Exception as e:  # noqa: BLE001
            self._render_preflight_error(str(e))
            return
        threading.Thread(
            target=self._preflight_worker, args=(gen, kwargs), daemon=True
        ).start()

    def _preflight_worker(self, gen: int, kwargs: dict) -> None:
        try:
            from ...pipeline.preflight import collect_preflight_results

            results = collect_preflight_results(**kwargs)
        except Exception:  # noqa: BLE001
            results = []
        self._pf_emitter.done.emit(gen, results)

    def _on_preflight_done(self, gen: int, results: list) -> None:
        if gen != self._preflight_gen:
            return  # résultat périmé (une vérif plus récente est en cours)
        self._render_preflight(results)

    def _show_preflight_loading(self) -> None:
        self._clear_grid()
        self._set_pill("●  …", "ok")
        lbl = QLabel("Vérification de l'environnement…")
        lbl.setObjectName("PreflightDetail")
        self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)

    def _render_preflight_error(self, msg: str) -> None:
        self._clear_grid()
        self._set_pill("●  ✗", "err")
        lbl = QLabel(f"Configuration illisible : {msg}")
        lbl.setObjectName("PreflightDetail")
        lbl.setWordWrap(True)
        self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)

    def _render_preflight(self, results: list) -> None:
        self._clear_grid()
        if not results:
            lbl = QLabel("Aucune vérification applicable pour cette configuration.")
            lbl.setObjectName("PreflightDetail")
            lbl.setWordWrap(True)
            self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)
            self._set_pill("●  —", "ok")
            return

        n_ok = 0
        n_critical_fail = 0
        for row, r in enumerate(results):
            if r.ok:
                status, glyph = "ok", "✓"
                n_ok += 1
            elif not r.critical:
                status, glyph = "warn", "⚠"
            else:
                status, glyph = "err", "✗"
                n_critical_fail += 1

            icon = QLabel(glyph)
            icon.setObjectName("PreflightIcon")
            icon.setProperty("status", status)
            icon.setFixedWidth(16)
            icon.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            self._preflight_grid.addWidget(icon, row, 0)

            name = QLabel(r.name)
            name.setObjectName("PreflightName")
            self._preflight_grid.addWidget(name, row, 1, Qt.AlignTop)

            detail = QLabel(r.details)
            detail.setObjectName("PreflightDetail")
            detail.setWordWrap(True)
            detail.setToolTip(r.details)
            detail.setAlignment(Qt.AlignRight | Qt.AlignTop)
            self._preflight_grid.addWidget(detail, row, 2)

        total = len(results)
        if n_critical_fail:
            kind = "err"
        elif n_ok < total:
            kind = "warn"
        else:
            kind = "ok"
        self._set_pill(f"●  {n_ok}/{total}", kind)

    def _set_pill(self, text: str, kind: str) -> None:
        self._preflight_pill.setText(text)
        self._preflight_pill.setProperty("kind", kind)
        self._preflight_pill.style().unpolish(self._preflight_pill)
        self._preflight_pill.style().polish(self._preflight_pill)

    def _clear_grid(self) -> None:
        while self._preflight_grid.count():
            item = self._preflight_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ------------------------------------------------------------------
    # Bascule récap ↔ exécution
    # ------------------------------------------------------------------
    def show_recap(self) -> None:
        self._stack.setCurrentWidget(self._recap_page)

    def set_step_subtitles(self, subs: Dict[int, str]) -> None:
        self._run_view.set_step_subtitles(subs)

    def start_run(self, config: dict) -> None:
        self._stack.setCurrentWidget(self._run_page)
        self._run_view.start_run(config)

    def is_running(self) -> bool:
        return self._run_view.is_running()
