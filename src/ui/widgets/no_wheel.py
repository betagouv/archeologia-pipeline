"""Spinbox qui ignorent la molette.

Dans une zone défilante, scroller au-dessus d'un spinbox modifie sa valeur par
accident. Ces variantes ignorent l'événement molette (il remonte au scroll)
sauf si le widget a explicitement le focus clavier.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDoubleSpinBox, QSpinBox


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):  # noqa: N802 (signature Qt)
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):  # noqa: N802 (signature Qt)
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
