"""FichePanel — le panneau latéral des fiches (interface rationalisée, 2026-10-24).

Les fiches (image du relief, structure détectable, modèle) s'ouvraient en
dialogues modaux : il fallait fermer la fiche pour cocher la carte qu'elle
décrivait. Le panneau se pose à droite des pages de l'assistant, **sans bloquer
la page** : on lit la fiche, on coche, la fiche reste. Trois onglets, un par
sorte de fiche, chacun gardant la dernière fiche ouverte ; ✕ ou Échap le ferme.

Pur widget : il reçoit un corps de fiche déjà construit (``ouvrir``), il ne
sait rien des données. Styles ``#FichePanel*`` et ``#FicheTab`` dans ``theme/v2.qss``.
"""
from __future__ import annotations

from typing import Dict, Optional

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

LARGEUR = 420
ONGLETS = ("Image", "Structure", "Modèle")


class FichePanel(QFrame):
    ferme = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FichePanel")
        self.setFixedWidth(LARGEUR)
        self._titres: Dict[str, str] = {}
        self._onglet = ONGLETS[0]

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tete = QFrame()
        tete.setObjectName("FichePanelTete")
        tl = QVBoxLayout(tete)
        tl.setContentsMargins(12, 8, 8, 0)
        tl.setSpacing(4)
        ligne = QHBoxLayout()
        ligne.setSpacing(6)
        self._titre = QLabel("")
        self._titre.setObjectName("FichePanelTitre")
        self._titre.setWordWrap(True)
        fermer = QToolButton()
        fermer.setObjectName("FichePanelFermer")
        fermer.setText("✕")
        fermer.setAutoRaise(True)
        fermer.setCursor(Qt.CursorShape.PointingHandCursor)
        fermer.setToolTip("Fermer la fiche (Échap)")
        fermer.clicked.connect(self.fermer)
        ligne.addWidget(self._titre, 1)
        ligne.addWidget(fermer, 0, Qt.AlignmentFlag.AlignTop)
        tl.addLayout(ligne)

        onglets = QHBoxLayout()
        onglets.setSpacing(2)
        self._groupe = QButtonGroup(self)
        self._groupe.setExclusive(True)
        self._boutons: Dict[str, QPushButton] = {}
        for nom in ONGLETS:
            b = QPushButton(nom)
            b.setObjectName("FicheTab")
            b.setCheckable(True)
            b.setEnabled(False)          # actif dès qu'une fiche de cette sorte a été ouverte
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            b.clicked.connect(lambda _c=False, n=nom: self._afficher(n))
            self._groupe.addButton(b)
            self._boutons[nom] = b
            onglets.addWidget(b)
        onglets.addStretch(1)
        tl.addLayout(onglets)
        root.addWidget(tete)

        # Une zone défilante par onglet : chaque fiche garde sa position de lecture,
        # et ``setWidget`` détruit l'ancien corps quand on en pose un nouveau.
        self._pile = QStackedWidget()
        self._zones: Dict[str, QScrollArea] = {}
        for nom in ONGLETS:
            zone = QScrollArea()
            zone.setObjectName("FicheScroll")
            zone.setWidgetResizable(True)
            zone.setFrameShape(QFrame.Shape.NoFrame)
            zone.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._zones[nom] = zone
            self._pile.addWidget(zone)
        root.addWidget(self._pile, 1)
        self.hide()

    # ------------------------------------------------------------------
    def ouvrir(self, onglet: str, widget: QWidget, titre: str) -> None:
        """Pose ``widget`` comme fiche de l'onglet et l'affiche. Le corps précédent de
        cet onglet est détruit (un seul widget par onglet)."""
        if onglet not in self._zones:
            raise ValueError(f"onglet inconnu : {onglet!r}")
        zone = self._zones[onglet]
        ancien = zone.takeWidget()
        if ancien is not None:
            ancien.deleteLater()
        zone.setWidget(widget)
        zone.verticalScrollBar().setValue(0)
        self._titres[onglet] = titre
        self._boutons[onglet].setEnabled(True)
        self._afficher(onglet)
        self.show()
        self._boutons[onglet].setFocus()

    def fermer(self) -> None:
        if self.isVisible():
            self.hide()
            self.ferme.emit()

    def onglet_courant(self) -> str:
        return self._onglet

    def contenu(self, onglet: str) -> Optional[QWidget]:
        """Le corps posé dans ``onglet``, ou ``None``."""
        return self._zones[onglet].widget() if onglet in self._zones else None

    # ------------------------------------------------------------------
    def _afficher(self, onglet: str) -> None:
        self._onglet = onglet
        self._boutons[onglet].setChecked(True)
        self._pile.setCurrentWidget(self._zones[onglet])
        self._titre.setText(self._titres.get(onglet, ""))
