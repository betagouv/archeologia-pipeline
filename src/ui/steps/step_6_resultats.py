"""Étape 6 — Résultats : revoir un traitement de ce poste (lot 9, 2026-10-24).

Une ligne par traitement connu (registre du profil) : date, point de départ,
dossier, images, structures, détections ; et trois gestes : ajouter ses couches
au projet QGIS courant (le même chargement qu'à la fin d'un traitement), ouvrir son
rapport, relancer
avec un seuil changé (la configuration de la trace complète est rechargée dans
l'assistant, qui s'ouvre à l'étape 3). Les données viennent du module pur
:mod:`app.services.resultats_connus`.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import List, Optional

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.services.resultats_connus import (
    Resultat,
    chemins_du_livrable,
    config_du_resultat,
    lister,
)
from ...app.services.vocabulaire import nombre, pluriel
from ..widgets.card import build_card
from ..widgets.toast import show_toast

logger = logging.getLogger("archeologia_pipeline")


def _date_fr(iso: str) -> str:
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%d/%m/%Y à %H:%M")
    except (TypeError, ValueError):
        return iso or "date inconnue"


class _Ligne(QFrame):
    """Un traitement connu : ses faits, puis ses trois gestes."""

    ajouter = pyqtSignal(object)
    ouvrir_rapport = pyqtSignal(object)
    relancer = pyqtSignal(object)

    def __init__(self, r: Resultat, parent=None):
        super().__init__(parent)
        self._r = r
        self.setObjectName("ResultatLigne")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        titre = QLabel(f"{_date_fr(r.date)}" + (f" · {r.mode}" if r.mode else ""))
        titre.setObjectName("ResultatTitre")
        lay.addWidget(titre)
        dossier = QLabel(str(r.dossier))
        dossier.setObjectName("ResultatDossier")
        dossier.setToolTip(str(r.dossier))
        dossier.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(dossier)
        faits = []
        if r.images:
            faits.append("produits : " + ", ".join(r.images))
        if r.structures:
            faits.append("structures : " + ", ".join(r.structures))
        faits.append(f"{nombre(r.n_detections, 0)} détection{'s' if r.n_detections > 1 else ''}"
                     if r.n_detections else "aucune détection")
        detail = QLabel(" · ".join(faits))
        detail.setObjectName("ResultatDetail")
        detail.setWordWrap(True)
        lay.addWidget(detail)

        boutons = QHBoxLayout()
        boutons.setSpacing(6)
        ajouter = QPushButton("Ajouter au projet QGIS courant")
        ajouter.setObjectName("GhostButton")
        ajouter.setToolTip("Charge les mosaïques et les couches de détections de ce traitement, stylées, dans le projet ouvert")
        ajouter.clicked.connect(lambda: self.ajouter.emit(self._r))
        rapport = QPushButton("Rapport")
        rapport.setObjectName("GhostButton")
        rapport.setEnabled(r.rapport.is_file())
        rapport.clicked.connect(lambda: self.ouvrir_rapport.emit(self._r))
        relancer = QPushButton("Relancer avec un seuil changé")
        relancer.setObjectName("GhostButton")
        relancer.setToolTip("Recharge les réglages de ce traitement dans l'assistant, à l'étape 3, pour les modifier et relancer")
        relancer.setEnabled(r.metadata is not None)
        relancer.clicked.connect(lambda: self.relancer.emit(self._r))
        for b in (ajouter, rapport, relancer):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            boutons.addWidget(b)
        boutons.addStretch(1)
        lay.addLayout(boutons)


class ResultatsPage(QWidget):
    relancer = pyqtSignal(dict)   # la configuration à recharger dans l'assistant

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self._iface = iface
        self._resultats: List[Resultat] = []
        self._lignes: List[QWidget] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        titre = QLabel("Les traitements lancés depuis ce poste")
        titre.setObjectName("WizardPageHeading")
        root.addWidget(titre)
        sous = QLabel(
            "Pour chacun : ajouter ses couches au projet QGIS ouvert, ouvrir son rapport, "
            "ou le relancer avec d'autres réglages."
        )
        sous.setObjectName("WizardPageSub")
        sous.setWordWrap(True)
        root.addWidget(sous)
        card, cv = build_card("Résultats", "1")
        self._compteur = card.counter
        self._hote = QWidget()
        self._liste = QVBoxLayout(self._hote)
        self._liste.setContentsMargins(0, 0, 0, 0)
        self._liste.setSpacing(8)
        cv.addWidget(self._hote)
        self._vide = QLabel("Aucun traitement connu sur ce poste.")
        self._vide.setObjectName("WizardPagePlaceholder")
        cv.addWidget(self._vide)
        root.addWidget(card)
        root.addStretch(1)

    # ------------------------------------------------------------------
    def rafraichir(self) -> None:
        """Relit le registre du profil (dossiers disparus écartés)."""
        from ..run_view import profil_archeologia

        try:
            self.set_resultats(lister(profil_archeologia()))
        except Exception:  # noqa: BLE001 — jamais bloquant
            self.set_resultats([])

    def set_resultats(self, resultats: List[Resultat]) -> None:
        for w in self._lignes:
            self._liste.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._lignes = []
        self._resultats = list(resultats)
        for r in self._resultats:
            ligne = _Ligne(r)
            ligne.ajouter.connect(self._ajouter)
            ligne.ouvrir_rapport.connect(self._ouvrir_rapport)
            ligne.relancer.connect(self._relancer)
            self._liste.addWidget(ligne)
            self._lignes.append(ligne)
        self._vide.setVisible(not self._resultats)
        if self._compteur is not None:
            self._compteur.setText(self.summary())

    def summary(self) -> str:
        n = len(self._resultats)
        return f"{pluriel(n, 'traitement')} connu{'s' if n > 1 else ''}" if n else "aucun traitement"

    # ------------------------------------------------------------------
    def _ajouter(self, r: Resultat) -> None:
        """Le même chargement qu'à la fin d'un traitement, depuis le livrable et la
        configuration de la trace (seuils par structure, regroupements)."""
        from ...app.services.finalize_service import build_entity_grouping, build_min_confidence_by_slug
        from ..layer_loader import load_result_layers

        vrt, gpkg, qa = chemins_du_livrable(r)
        if not (vrt or gpkg or qa):
            show_toast(self, "Rien à charger : le livrable de ce traitement est vide ou déplacé.")
            return
        cfg = config_du_resultat(r) or {}
        cv = cfg.get("computer_vision") or {}
        proc = cfg.get("processing") or {}
        try:
            conf = float(cv.get("confidence_threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            cov = float(proc.get("coverage_threshold_percent", 30.0) or 30.0)
        except (TypeError, ValueError):
            cov = 30.0
        entity_labels, derived_slugs = build_entity_grouping(cv.get("runs"))
        min_conf = build_min_confidence_by_slug(cv.get("runs"))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            load_result_layers(
                vrt, gpkg, None, logger, conf,
                entity_labels=entity_labels, derived_slugs=derived_slugs,
                min_conf_by_slug=min_conf, qa_paths=qa, coverage_threshold_percent=cov,
            )
        finally:
            QApplication.restoreOverrideCursor()

    @staticmethod
    def _ouvrir_rapport(r: Resultat) -> None:
        if r.rapport.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(r.rapport)))

    def _relancer(self, r: Resultat) -> None:
        cfg = config_du_resultat(r)
        if not cfg:
            show_toast(self, "La trace de ce traitement ne porte pas sa configuration : relancez-le depuis l'étape 1.")
            return
        self.relancer.emit(cfg)

    def resultats(self) -> List[Resultat]:
        return list(self._resultats)

    @staticmethod
    def dossier_de(r: Resultat) -> Optional[Path]:
        return r.dossier if r.dossier.is_dir() else None
