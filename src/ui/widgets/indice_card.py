"""IndiceCard — une carte du mur visuel (onglet « Visualisation »).

Vignette + libellé métier + nom technique + bouton « Afficher dans QGIS ».
Le titre est le libellé MÉTIER (« Creux & dépressions »), jamais le sigle :
l'utilisateur reconnaît une image, il ne décode pas un acronyme.

La carte est un pur widget de présentation : elle émet des signaux et reçoit son
état par :meth:`set_state`. C'est la page qui sait quelles couches sont ouvertes
dans QGIS — l'état n'est jamais stocké ici (si l'utilisateur supprime la couche
depuis le panneau Couches de QGIS, la page repasse la carte en ``idle``).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QRect, QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

CARD_W = 200
MAX_W = 320            # au-delà, une carte cesse de lire comme une vignette
THUMB_H = 112          # 200x112 = 16:9

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_LOADED = "loaded"

_CHIP_BG = QColor(24, 28, 32, 184)
_BLUE = QColor("#2b79c2")


class _Thumb(QWidget):
    """Vignette + pastilles + voile d'état, peints en une seule passe.

    Tout est dessiné plutôt qu'empilé en widgets : les pastilles doivent
    chevaucher l'image, et un QLabel par pastille coûterait un layout par carte
    pour un résultat moins net.
    """

    remove_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap = QPixmap()
        self._sigle = ""
        self._size_label = ""
        self._state = STATE_IDLE
        self.setMinimumHeight(THUMB_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(CARD_W, THUMB_H)

    def set_content(self, pixmap: QPixmap, sigle: str, size_label: str) -> None:
        self._pixmap = pixmap or QPixmap()
        self._sigle = sigle
        self._size_label = size_label
        self.update()

    def set_state(self, state: str) -> None:
        self._state = state
        self.setCursor(Qt.CursorShape.PointingHandCursor if state == STATE_LOADED
                       else Qt.CursorShape.ArrowCursor)
        self.setToolTip("Retirer la couche de QGIS" if state == STATE_LOADED else "")
        self.update()

    def _close_rect(self) -> QRect:
        return QRect(6, 6, 18, 18)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        if self._state == STATE_LOADED and self._close_rect().contains(event.pos()):
            self.remove_clicked.emit()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        r = self.rect()

        if self._pixmap.isNull():
            p.fillRect(r, QColor("#e3e3e3"))
            p.setPen(QColor("#909090"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "…")
        else:
            scaled = self._pixmap.scaled(
                r.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            # recadrage centré : la vignette remplit la carte sans se déformer
            x = (scaled.width() - r.width()) // 2
            y = (scaled.height() - r.height()) // 2
            p.drawPixmap(r, scaled, QRect(x, y, r.width(), r.height()))

        if self._state == STATE_LOADING:
            p.fillRect(r, QColor(255, 255, 255, 140))

        mono = QFont("Consolas")
        mono.setPixelSize(10)
        p.setFont(mono)
        fm = QFontMetrics(mono)

        def chip(text: str, right: bool, top: int, bg: QColor = _CHIP_BG) -> None:
            w = fm.horizontalAdvance(text) + 12
            x0 = r.width() - w - 6 if right else 6
            box = QRect(x0, top, w, 15)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

        if self._size_label:
            chip(self._size_label, right=True, top=6)
        if self._sigle:
            chip(self._sigle, right=False, top=r.height() - 21)

        if self._state == STATE_LOADING:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_BLUE)
            p.drawRect(QRect(0, r.height() - 4, r.width(), 4))
            f = QFont(mono)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#1d5a96"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "OUVERTURE…")

        if self._state == STATE_LOADED:
            box = self._close_rect()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(24, 28, 32, 200))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QPen(QColor("#ffffff"), 1.4))
            p.drawLine(box.left() + 6, box.top() + 6, box.right() - 6, box.bottom() - 6)
            p.drawLine(box.right() - 6, box.top() + 6, box.left() + 6, box.bottom() - 6)
        p.end()


class IndiceCard(QFrame):
    """Carte d'un indice consultable. États : idle / loading / loaded."""

    open_requested = pyqtSignal(str)     # clé de l'indice
    remove_requested = pyqtSignal(str)

    def __init__(self, key: str, metier: str, technique: str, parent=None):
        super().__init__(parent)
        self._key = key
        self._technique = technique
        self._state = STATE_IDLE
        self.setObjectName("VisuCard")
        self.setProperty("state", STATE_IDLE)
        self.setMinimumWidth(CARD_W)
        self.setMaximumWidth(MAX_W)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._thumb = _Thumb()
        self._thumb.remove_clicked.connect(lambda: self.remove_requested.emit(self._key))
        lay.addWidget(self._thumb)

        body = QVBoxLayout()
        body.setContentsMargins(9, 8, 9, 9)
        body.setSpacing(2)
        self._title = QLabel(metier)
        self._title.setObjectName("VisuCardTitle")
        self._tech = QLabel(technique)
        self._tech.setObjectName("VisuCardTech")
        body.addWidget(self._title)
        body.addWidget(self._tech)
        body.addSpacing(6)

        self._btn = QPushButton("Afficher dans QGIS")
        self._btn.setObjectName("VisuCardBtn")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(lambda: self.open_requested.emit(self._key))
        body.addWidget(self._btn)
        lay.addLayout(body)

    @property
    def key(self) -> str:
        return self._key

    def set_content(self, pixmap: QPixmap, sigle: str, size_label: str) -> None:
        self._thumb.set_content(pixmap, sigle, size_label)

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.setProperty("state", state)
        self._thumb.set_state(state)
        if state == STATE_LOADING:
            self._btn.setEnabled(False)
            self._btn.setText("Ouverture…")
        elif state == STATE_LOADED:
            self._btn.setEnabled(False)
            self._btn.setText("Affichée")
        else:
            self._btn.setEnabled(True)
            self._btn.setText("Afficher dans QGIS")
        # Une propriété QSS ne se répercute pas toute seule : il faut relancer
        # le calcul de style sur la carte ET ses enfants stylés par sélecteur.
        for w in (self, self._title, self._tech, self._btn):
            w.style().unpolish(w)
            w.style().polish(w)

    def resizeEvent(self, event):  # noqa: N802 (signature Qt)
        super().resizeEvent(event)
        fm = QFontMetrics(self._tech.font())
        self._tech.setText(fm.elidedText(
            self._technique, Qt.TextElideMode.ElideRight, max(40, self._tech.width())))
