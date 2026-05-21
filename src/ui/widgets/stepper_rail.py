"""Rail de navigation latéral du wizard V2 (stepper).

Liste verticale d'étapes cliquables (navigation libre, non linéaire) avec
badge d'état : numéro (à faire), ✓ (fait), ! (erreur) ou actif. Le style
est piloté par ``theme/v2.qss`` via la propriété dynamique ``state``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class _RailItem(QFrame):
    """Un item du rail : badge + libellé + sous-libellé, cliquable."""

    clicked = pyqtSignal(int)  # numéro d'étape (1-based)

    def __init__(self, index: int, label: str, sublabel: str, parent=None):
        super().__init__(parent)
        self._index = index
        self.setObjectName("RailItem")
        self.setProperty("state", "todo")
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self._badge = QLabel(str(index))
        self._badge.setObjectName("RailBadge")
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setFixedSize(26, 26)

        texts = QVBoxLayout()
        texts.setSpacing(0)
        self._label = QLabel(label)
        self._label.setObjectName("RailLabel")
        self._sub = QLabel(sublabel)
        self._sub.setObjectName("RailSub")
        self._sub.setWordWrap(True)
        texts.addWidget(self._label)
        texts.addWidget(self._sub)

        layout.addWidget(self._badge)
        layout.addLayout(texts, 1)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def set_state(self, state: str) -> None:
        """state ∈ {'active', 'done', 'todo', 'error'}."""
        self.setProperty("state", state)
        self._badge.setText({"done": "✓", "error": "!"}.get(state, str(self._index)))
        # Forcer le re-style (propriété dynamique → QSS).
        self.style().unpolish(self)
        self.style().polish(self)
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)


class StepperRail(QFrame):
    """Rail vertical d'étapes. Émet ``step_clicked(n)`` au clic sur une étape."""

    step_clicked = pyqtSignal(int)

    def __init__(self, steps: Sequence[Tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setObjectName("StepperRail")
        self.setFixedWidth(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(6)

        self._items: List[_RailItem] = []
        for i, (label, sub) in enumerate(steps, start=1):
            item = _RailItem(i, label, sub)
            item.clicked.connect(self.step_clicked)
            layout.addWidget(item)
            self._items.append(item)
        layout.addStretch(1)

    def set_current(self, step: int) -> None:
        """Marque l'étape ``step`` active, les autres à 'todo' (sauf erreurs)."""
        for idx, item in enumerate(self._items, start=1):
            item.set_state("active" if idx == step else "todo")

    def set_states(self, current: int, errors: Sequence[int] = ()) -> None:
        """Variante riche : 'active' pour current, 'error' pour les étapes en
        erreur, 'todo' pour le reste. (Utilisé à partir du Jalon 7.)"""
        errors = set(errors)
        for idx, item in enumerate(self._items, start=1):
            if idx == current:
                item.set_state("active")
            elif idx in errors:
                item.set_state("error")
            else:
                item.set_state("todo")
