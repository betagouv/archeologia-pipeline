"""Étape 4 — Lancement : récapitulatif riche + vue d'exécution (timeline + journal).

Le récap reprend les choix des étapes 1-3 sous forme de sections (badges + détail),
assemblées par le wizard. Le bouton « Lancer » de la barre d'actions déclenche
:meth:`start_run`, qui valide puis exécute le pipeline dans le :class:`RunView`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from qgis.PyQt.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QLayout, QVBoxLayout, QWidget

from ..run_view import RunView
from ..widgets.card import build_card


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


class LaunchPage(QWidget):
    run_started = pyqtSignal()
    run_finished = pyqtSignal()

    def __init__(self, plugin_root, config_ref=None, parent=None):
        super().__init__(parent)
        self._recap_rows: List[QWidget] = []
        self._build(config_ref)

    def _build(self, config_ref) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Bandeau de validation : rouge si la config est inexécutable, vert sinon.
        self._banner = QLabel("")
        self._banner.setObjectName("ValidationBanner")
        self._banner.setProperty("state", "ok")
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        root.addWidget(self._banner)

        recap_card, rv = build_card("Récapitulatif", "1")
        self._recap_host = QWidget()
        self._recap_layout = QVBoxLayout(self._recap_host)
        self._recap_layout.setContentsMargins(0, 0, 0, 0)
        self._recap_layout.setSpacing(6)
        rv.addWidget(self._recap_host)
        root.addWidget(recap_card)

        run_card, xv = build_card("Exécution", "2")
        self._run_view = RunView(config_ref)
        self._run_view.run_started.connect(self.run_started)
        self._run_view.run_finished.connect(self.run_finished)
        xv.addWidget(self._run_view)
        root.addWidget(run_card, 1)

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

    def set_step_subtitles(self, subs: Dict[int, str]) -> None:
        self._run_view.set_step_subtitles(subs)

    def start_run(self, config: dict) -> None:
        self._run_view.start_run(config)

    def is_running(self) -> bool:
        return self._run_view.is_running()
