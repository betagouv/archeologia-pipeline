"""Rail de navigation latéral du wizard V2 (stepper).

Liste verticale d'étapes cliquables (navigation libre) avec pastille d'index :
numéro (à faire), ✓ (faite) ou actif (rempli bleu). Un « fil » vertical relie
les pastilles (aspect chronologique). Sous-libellé dynamique + chip « opt. ».
Style porté par ``theme/v2.qss`` (propriété ``state``).
"""
from __future__ import annotations

from typing import List, Sequence

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter, QPen
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class _RailItem(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, label: str, sub: str, optional: bool = False, parent=None):
        super().__init__(parent)
        self._index = index
        self._is_first = False
        self._is_last = False
        self.setObjectName("RailItem")
        self.setProperty("state", "todo")
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        self._idx = QLabel(str(index))
        self._idx.setObjectName("RailIdx")
        self._idx.setAlignment(Qt.AlignCenter)
        self._idx.setFixedSize(22, 22)

        meta = QVBoxLayout()
        meta.setSpacing(1)
        label_row = QHBoxLayout()
        label_row.setSpacing(5)
        self._label = QLabel(label)
        self._label.setObjectName("RailLabel")
        label_row.addWidget(self._label)
        if optional:
            chip = QLabel("OPT.")
            chip.setObjectName("RailOpt")
            label_row.addWidget(chip)
        label_row.addStretch(1)
        self._sub = QLabel(sub)
        self._sub.setObjectName("RailSub")
        self._sub.setWordWrap(True)
        meta.addLayout(label_row)
        meta.addWidget(self._sub)

        layout.addWidget(self._idx, 0, Qt.AlignTop)
        layout.addLayout(meta, 1)

    def set_state(self, state: str) -> None:
        """state ∈ {'active', 'done', 'todo'}."""
        self.setProperty("state", state)
        self._idx.setText("✓" if state == "done" else str(self._index))
        for w in (self, self._idx, self._label, self._sub):
            try:
                w.style().unpolish(w)
                w.style().polish(w)
            except Exception:
                pass
        self.update()  # redessine le fil

    def set_sub(self, text: str) -> None:
        self._sub.setText(text)

    def paintEvent(self, event):  # noqa: N802 (signature Qt)
        super().paintEvent(event)
        done = self.property("state") == "done"
        painter = QPainter(self)
        pen = QPen(QColor("#418141") if done else QColor("#c4c4c4"))
        pen.setWidth(2)
        painter.setPen(pen)
        rect = self._idx.geometry()
        cx = rect.center().x()
        if not self._is_first:
            painter.drawLine(cx, 0, cx, rect.top())
        if not self._is_last:
            painter.drawLine(cx, rect.bottom(), cx, self.height())
        painter.end()

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class StepperRail(QFrame):
    """Rail vertical d'étapes. Émet ``step_clicked(n)`` au clic sur une étape."""

    step_clicked = pyqtSignal(int)

    def __init__(self, steps: Sequence[dict], parent=None):
        super().__init__(parent)
        self.setObjectName("StepperRail")
        self.setFixedWidth(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)

        self._items: List[_RailItem] = []
        for i, step in enumerate(steps, start=1):
            item = _RailItem(i, step["label"], step.get("sub", ""), step.get("optional", False))
            item.clicked.connect(self._relay_click)
            layout.addWidget(item)
            self._items.append(item)
        layout.addStretch(1)
        if self._items:
            self._items[0]._is_first = True
            self._items[-1]._is_last = True

    def _relay_click(self, n: int) -> None:
        self.step_clicked.emit(n)

    def set_current(self, step: int) -> None:
        """Items avant ``step`` = faits, ``step`` = actif, après = à faire."""
        for idx, item in enumerate(self._items, start=1):
            item.set_state("done" if idx < step else ("active" if idx == step else "todo"))

    def set_sub(self, step: int, text: str) -> None:
        if 1 <= step <= len(self._items):
            self._items[step - 1].set_sub(text)
