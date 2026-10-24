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
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout


class _RailItem(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, label: str, sub: str, optional: bool = False, parent=None):
        super().__init__(parent)
        self._index = index
        self._is_first = False
        self._is_last = False
        self._sub_text = sub        # le résumé de l'étape, rendu quand rien n'est à corriger
        self._a_corriger = False
        self.setObjectName("RailItem")
        self.setProperty("state", "todo")
        self.setProperty("error", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        self._idx = QLabel(str(index))
        self._idx.setObjectName("RailIdx")
        self._idx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idx.setFixedSize(22, 22)

        meta = QVBoxLayout()
        meta.setSpacing(1)
        label_row = QHBoxLayout()
        label_row.setSpacing(5)
        self._label = QLabel(label)
        self._label.setObjectName("RailLabel")
        # « Structures à chercher » tient sur deux lignes dans le rail de 210 px, à côté de la puce « OPT. ».
        self._label.setWordWrap(True)
        # Un QLabel qui replie se contente de sa largeur minimale : lui donner toute la
        # largeur disponible, sinon « Images du relief » se coupe sans raison.
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label_row.addWidget(self._label, 1)
        if optional:
            chip = QLabel("OPT.")
            chip.setObjectName("RailOpt")
            label_row.addWidget(chip, 0, Qt.AlignmentFlag.AlignTop)
        # Pas d'étirement en fin de ligne : il partageait la largeur avec le libellé replié,
        # qui se coupait en deux (« Images du / relief ») ; la puce « OPT. » reste à droite.
        self._sub = QLabel(sub)
        self._sub.setObjectName("RailSub")
        self._sub.setWordWrap(True)
        meta.addLayout(label_row)
        meta.addWidget(self._sub)

        layout.addWidget(self._idx, 0, Qt.AlignmentFlag.AlignTop)
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
        self._sub_text = text
        self._rendre_sub()

    def _rendre_sub(self) -> None:
        """« à corriger » remplace le résumé tant que l'étape porte une erreur (le
        résumé reste en infobulle avec les messages) ; sinon le résumé revient."""
        self._sub.setText("à corriger" if self._a_corriger else self._sub_text)
        self._sub.setProperty("error", "true" if self._a_corriger else "false")
        self._sub.style().unpolish(self._sub)
        self._sub.style().polish(self._sub)

    def set_error(self, on: bool, tooltip: str = "") -> None:
        """Marqueur d'erreur indépendant de l'état (active/done/todo) : pastille
        rouge + infobulle listant les problèmes bloquants de cette étape."""
        flag = "true" if on else "false"
        self._a_corriger = bool(on)
        self.setProperty("error", flag)
        self._idx.setProperty("error", flag)
        self._label.setProperty("error", flag)
        self.setToolTip(tooltip if on else "")
        self._rendre_sub()
        for w in (self, self._idx, self._label):
            try:
                w.style().unpolish(w)
                w.style().polish(w)
            except Exception:
                pass

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

    def set_states(self, etats: dict, messages: dict) -> None:
        """``etats`` = {n: 'active' | 'ok' | 'a_corriger' | 'a_venir'} (cf.
        ``app.services.validation_ui.etat_rail``), ``messages`` = {n: [erreurs…]}.
        Une étape « à corriger » garde son aspect (faite ou à venir) et prend la
        pastille rouge, le texte « à corriger » et les messages en infobulle."""
        for idx, item in enumerate(self._items, start=1):
            etat = etats.get(idx, "a_venir")
            item.set_state("active" if etat == "active" else ("done" if etat == "ok" else "todo"))
            msgs = messages.get(idx) or []
            item.set_error(etat == "a_corriger", "\n".join(msgs))

    def set_sub(self, step: int, text: str) -> None:
        if 1 <= step <= len(self._items):
            self._items[step - 1].set_sub(text)
