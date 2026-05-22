"""Section repliable (header cliquable + corps masquable).

Header en texte riche : chevron ▸/▾ + titre gras + aperçu gris facultatif
(``preview_getter()``), p. ex. « ▸ Paramètres avancés · workers : 4 ». Le style
est piloté par QSS (objectName) — cf. ``theme/v2.qss``.
"""
from __future__ import annotations

from typing import Callable, Optional

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class _ClickableFrame(QFrame):
    """QFrame émettant ``clicked`` au clic gauche (header repliable)."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class CollapsibleSection(QFrame):
    """Carte repliable : header cliquable révélant un corps au clic."""

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        preview_getter: Optional[Callable[[], str]] = None,
        *,
        expanded: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("Collapsible")
        self._title = title
        self._preview_getter = preview_getter
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = _ClickableFrame()
        self._header.setObjectName("CollapsibleHeader")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.setSpacing(0)
        self._header_label = QLabel()
        self._header_label.setObjectName("CollapsibleHeaderLabel")
        self._header_label.setTextFormat(Qt.RichText)
        hl.addWidget(self._header_label)
        hl.addStretch(1)
        root.addWidget(self._header)

        self._sep = QFrame()
        self._sep.setObjectName("CollapsibleSep")
        self._sep.setFrameShape(QFrame.HLine)
        self._sep.setVisible(self._expanded)
        root.addWidget(self._sep)

        self._body = QWidget()
        self._body.setObjectName("CollapsibleBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 10, 14, 12)
        self._body.setVisible(self._expanded)
        root.addWidget(self._body)

        self._refresh_header()

    # ------------------------------------------------------------------
    def set_body(self, widget: QWidget) -> None:
        """Remplace le contenu du corps par ``widget``."""
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._body_layout.addWidget(widget)

    def refresh_preview(self) -> None:
        """À appeler quand l'aperçu change (ex. workers modifié)."""
        self._refresh_header()

    def is_expanded(self) -> bool:
        return self._expanded

    # ------------------------------------------------------------------
    def _refresh_header(self) -> None:
        chevron = "▾" if self._expanded else "▸"
        sep = "&nbsp;&nbsp;·&nbsp;&nbsp;"
        extra = ""
        if not self._expanded:
            if self._preview_getter:
                text = self._preview_getter()
                if text:
                    extra += f"<span style='color:#5a5a5a;'>{sep}{text}</span>"
            extra += (
                f"<span style='color:#5a5a5a; font-style:italic;'>{sep}"
                "Cliquer pour modifier</span>"
            )
        self._header_label.setText(
            f"<span style='color:#5a5a5a;'>{chevron}</span>&nbsp;&nbsp;"
            f"<b>{self._title}</b>{extra}"
        )

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._sep.setVisible(self._expanded)
        self._body.setVisible(self._expanded)
        self._refresh_header()
        self.toggled.emit(self._expanded)
