"""Étape 4 — Lancement : récapitulatif + vue d'exécution (timeline + journal).

Le récap reprend les choix des étapes 1-3 (résumés fournis par le wizard). Le
bouton « Lancer » de la barre d'actions déclenche :meth:`start_run`, qui valide
puis exécute le pipeline dans le :class:`RunView`.
"""
from __future__ import annotations

from typing import List, Tuple

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..run_view import RunView
from ..widgets.card import build_card


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
        self._recap_layout.setSpacing(5)
        rv.addWidget(self._recap_host)
        root.addWidget(recap_card)

        run_card, xv = build_card("Exécution", "2")
        self._run_view = RunView(config_ref)
        self._run_view.run_started.connect(self.run_started)
        self._run_view.run_finished.connect(self.run_finished)
        xv.addWidget(self._run_view)
        root.addWidget(run_card, 1)

    # ------------------------------------------------------------------
    def update_recap(self, rows: List[Tuple[str, str]]) -> None:
        """``rows`` = [(libellé, valeur)…] reconstruit le récap."""
        for w in self._recap_rows:
            w.deleteLater()
        self._recap_rows = []
        for label, value in rows:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            key = QLabel(label)
            key.setObjectName("RecapKey")
            key.setFixedWidth(120)
            val = QLabel(value or "—")
            val.setObjectName("RecapVal")
            val.setWordWrap(True)
            h.addWidget(key)
            h.addWidget(val, 1)
            self._recap_layout.addWidget(row)
            self._recap_rows.append(row)

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

    def start_run(self, config: dict) -> None:
        self._run_view.start_run(config)

    def is_running(self) -> bool:
        return self._run_view.is_running()
