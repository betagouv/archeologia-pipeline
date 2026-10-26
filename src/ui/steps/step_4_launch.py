"""Étape 4 — Prêt à lancer : bandeau à quatre états, « Environnement » (vérifications
lancées en tâche de fond, en français, détail en infobulle, « Revérifier »),
récapitulatif des choix des étapes 1-3 avec un lien « modifier » par section et
« Ce qui sera produit », puis « Réglages du poste » (dalles en parallèle, carte
graphique et la décision mémorisée, qu'on peut oublier). Aucune estimation de
durée ni de mémoire (le traitement ne les expose pas de façon fiable).

Le bouton « Lancer le traitement » de la barre d'actions démarre le traitement, suivi
à l'étape 5 (``RunView``, posé par l'assistant). Les deux écrans étaient les deux
sous-pages de l'ancienne étape « Lancer » (scindée le 2026-10-09).
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from qgis.PyQt.QtCore import QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...app.services.aide import AIDE_DIRNAME, CLE_DEPANNAGE, resoudre_cible
from ...app.services.carte_graphique import oublier, phrase_decision, registre_path
from ...app.services.validation_ui import classer_erreurs
from ...app.services.vocabulaire import libelle_environnement, pluriel, resume_detail
from ..widgets.card import build_card
from ..widgets.collapsible import CollapsibleSection
from ..widgets.no_wheel import NoWheelSpinBox

@dataclass
class RecapSection:
    """Une ligne du récap : libellé + pastilles + valeur + sous-ligne de détail ;
    ``step`` > 0 pose un lien « modifier » qui ramène à cette étape."""

    label: str
    badges: List[str] = field(default_factory=list)
    value: str = ""
    detail: str = ""
    step: int = 0


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
        return Qt.Orientation(0)

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


class _PreflightEmitter(QObject):
    """Pont thread worker → thread UI pour les résultats des vérifications."""

    done = pyqtSignal(int, object)  # (génération, List[CheckResult])


class LaunchPage(QWidget):
    workers_changed = pyqtSignal(int)
    carte_graphique_changed = pyqtSignal(bool)
    reverifier_demande = pyqtSignal()        # « Revérifier » : l'assistant relance les vérifications
    modifier_demande = pyqtSignal(int)       # « modifier » d'une section du récap : l'étape à ouvrir

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recap_rows: List[QWidget] = []
        self._workers = 3   # = ConfigManager (processing.max_workers)
        self._carte = True   # = ConfigManager (computer_vision.carte_graphique), cochée par défaut (2026-10-26)
        self._preflight_gen = 0
        self._validation_errors: List[str] = []
        # État des vérifications pour le bandeau : pending | ok | warn | err, avec les
        # libellés des contrôles manquants (critiques) et des avertissements. Le bandeau
        # ne doit pas dire « prêt à lancer » (vert) au-dessus d'un contrôle rouge.
        self._preflight_status = "pending"
        self._pf_manquants: List[str] = []
        self._pf_avertissements: List[str] = []
        self._pf_emitter = _PreflightEmitter(self)
        self._pf_emitter.done.connect(self._on_preflight_done)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("LaunchScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # Bandeau : à corriger (rouge, par étape), vérification en cours, outil manquant
        # (rouge, lien vers le Dépannage), avertissement (orange) ou prêt (vert).
        self._banner = QLabel("")
        self._banner.setObjectName("ValidationBanner")
        self._banner.setProperty("state", "ok")
        self._banner.setTextFormat(Qt.TextFormat.RichText)
        self._banner.setWordWrap(True)
        self._banner.setOpenExternalLinks(False)
        self._banner.linkActivated.connect(self._sur_lien)
        self._banner.setVisible(False)
        v.addWidget(self._banner)

        # ① Environnement (vérifications)
        v.addWidget(self._build_preflight_card())

        # ② Récapitulatif des choix, avec « modifier » par section
        recap_card, rv = build_card("Récapitulatif", "2")
        self._recap_host = QWidget()
        self._recap_layout = QVBoxLayout(self._recap_host)
        self._recap_layout.setContentsMargins(0, 0, 0, 0)
        self._recap_layout.setSpacing(6)
        rv.addWidget(self._recap_host)
        v.addWidget(recap_card)

        # ③ Réglages du poste (replié) — dalles en parallèle, carte graphique
        self._adv = CollapsibleSection(
            "Réglages du poste",
            preview_getter=lambda: f"{pluriel(self._workers, 'dalle')} en parallèle · carte graphique : {'oui' if self._carte else 'non'}",
        )
        self._adv.set_body(self._build_workers_body())
        v.addWidget(self._adv)

        hint = QLabel("Le traitement écrit un projet QGIS prêt à ouvrir dans livrable/.")
        hint.setObjectName("LaunchHint")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(hint)

        v.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_preflight_card(self) -> QWidget:
        card, cv = build_card("Environnement", "1")

        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 0, 0, 0)
        pill_row.setSpacing(8)
        self._reverifier_btn = QPushButton("Revérifier")
        self._reverifier_btn.setObjectName("GhostButton")
        self._reverifier_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reverifier_btn.setToolTip("Relancer les vérifications, après avoir installé un outil par exemple")
        self._reverifier_btn.clicked.connect(self.reverifier_demande)
        pill_row.addStretch(1)
        pill_row.addWidget(self._reverifier_btn)
        self._preflight_pill = QLabel("●  …")
        self._preflight_pill.setObjectName("PreflightPill")
        self._preflight_pill.setProperty("kind", "ok")
        pill_row.addWidget(self._preflight_pill)
        cv.addLayout(pill_row)

        self._preflight_host = QWidget()
        self._preflight_grid = QGridLayout(self._preflight_host)
        self._preflight_grid.setContentsMargins(0, 0, 0, 0)
        self._preflight_grid.setHorizontalSpacing(10)
        self._preflight_grid.setVerticalSpacing(3)
        self._preflight_grid.setColumnStretch(2, 1)  # le résumé prend la largeur restante
        cv.addWidget(self._preflight_host)
        return card

    def _build_workers_body(self) -> QWidget:
        body = QWidget()
        colonne = QVBoxLayout(body)
        colonne.setContentsMargins(0, 0, 0, 0)
        colonne.setSpacing(8)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        colonne.addLayout(h)

        lbl = QLabel("Dalles traitées en parallèle")
        lbl.setObjectName("FieldLabel")
        h.addWidget(lbl)

        minus = QPushButton("−")
        minus.setObjectName("WorkersStep")
        minus.setProperty("role", "micro")
        minus.clicked.connect(lambda: self._set_workers(self._workers - 1))
        h.addWidget(minus)

        self._workers_spin = NoWheelSpinBox()
        self._workers_spin.setObjectName("WorkersSpin")
        self._workers_spin.setRange(1, 16)
        self._workers_spin.setValue(self._workers)
        self._workers_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Flèches natives masquées : on a déjà des boutons −/+ explicites
        # (sinon les flèches Fusion se compriment dans le champ étroit).
        self._workers_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._workers_spin.setFixedWidth(48)
        self._workers_spin.valueChanged.connect(self._set_workers)
        h.addWidget(self._workers_spin)

        plus = QPushButton("+")
        plus.setObjectName("WorkersStep")
        plus.setProperty("role", "micro")
        plus.clicked.connect(lambda: self._set_workers(self._workers + 1))
        h.addWidget(plus)

        note = QLabel(
            "Nombre de dalles téléchargées, fusionnées et préparées pour la détection en "
            "même temps ; le MNT et les images sont calculés une dalle après l'autre. "
            "Plus de dalles en parallèle accélère ces étapes mais consomme plus de mémoire."
        )
        note.setObjectName("WorkersNote")
        note.setWordWrap(True)
        h.addWidget(note, 1)

        # Carte graphique (DirectML, Windows) : décochée par défaut ; la sonde par poste et
        # le repli sont faits par le moteur, le journal dit ce qui a été décidé et pourquoi.
        ligne = QHBoxLayout()
        ligne.setContentsMargins(0, 0, 0, 0)
        ligne.setSpacing(8)
        self._carte_check = QCheckBox("Détection sur la carte graphique")
        self._carte_check.setObjectName("CarteGraphiqueCheck")
        self._carte_check.setChecked(True)
        self._carte_check.toggled.connect(self._on_carte_toggled)
        ligne.addWidget(self._carte_check)
        note_carte = QLabel(
            "Au premier usage de chaque modèle sur ce poste, la carte est comparée au processeur sur la "
            "première image : elle n'est gardée que si elle est plus rapide et donne les mêmes détections. "
            "Sinon, ou à la première erreur en cours de traitement, la détection continue sur le processeur "
            "et le journal le dit."
        )
        note_carte.setObjectName("WorkersNote")
        note_carte.setWordWrap(True)
        ligne.addWidget(note_carte, 1)
        colonne.addLayout(ligne)
        if sys.platform != "win32":
            self._carte_check.setEnabled(False)
            self._carte_check.setToolTip("Windows seulement (DirectML).")

        # La décision mémorisée pour ce poste (carte retenue ou écartée, à quelle date),
        # et « Oublier » pour la faire re-sonder au prochain traitement.
        memo = QHBoxLayout()
        memo.setContentsMargins(0, 0, 0, 0)
        memo.setSpacing(8)
        self._carte_memo = QLabel("")
        self._carte_memo.setObjectName("WorkersNote")
        self._carte_memo.setWordWrap(True)
        self._oublier_btn = QPushButton("Oublier")
        self._oublier_btn.setObjectName("GhostButton")
        self._oublier_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._oublier_btn.setToolTip(
            "Efface la décision mémorisée : la carte sera de nouveau comparée au processeur "
            "au prochain traitement"
        )
        self._oublier_btn.clicked.connect(self._on_oublier)
        memo.addWidget(self._carte_memo, 1)
        memo.addWidget(self._oublier_btn, 0, Qt.AlignmentFlag.AlignTop)
        colonne.addLayout(memo)
        self._carte_memo.setVisible(False)
        self._oublier_btn.setVisible(False)
        self._workers_body = body
        return body

    # ------------------------------------------------------------------
    # Récapitulatif (assemblé par l'assistant)
    # ------------------------------------------------------------------
    def update_recap(self, sections: List[RecapSection]) -> None:
        """Reconstruit le récap. Retrait SYNCHRONE des anciennes lignes : deleteLater seul
        les laissait dans la mise en page jusqu'au retour à la boucle d'événements, et deux
        reconstructions rapprochées les montraient en double."""
        for w in self._recap_rows:
            self._recap_layout.removeWidget(w)
            w.setParent(None)
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
        top.addWidget(key, 0, Qt.AlignmentFlag.AlignTop)

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
                top.addWidget(val, 0, Qt.AlignmentFlag.AlignTop)
        elif chr(10) in (sec.value or ""):
            # Une liste (« Ce qui sera produit ») : un libellé d'une ligne par chemin. Un seul
            # libellé de treize lignes se faisait rogner ou distribuait de la hauteur aux
            # autres lignes (hauteur selon largeur dans la zone défilante).
            liste = QWidget()
            lv = QVBoxLayout(liste)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setSpacing(1)
            for ligne in sec.value.split(chr(10)):
                item = QLabel(ligne)
                item.setObjectName("RecapListe")
                item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                lv.addWidget(item)
            top.addWidget(liste, 1, Qt.AlignmentFlag.AlignTop)
        else:
            val = QLabel(sec.value or "—")
            val.setObjectName("RecapVal")
            val.setWordWrap(True)
            top.addWidget(val, 1)
        if sec.step:
            # « modifier » ramène à l'étape de ce choix : on relit, on corrige, on revient.
            lien = QPushButton("modifier")
            lien.setObjectName("SeeLink")
            lien.setFlat(True)
            lien.setCursor(Qt.CursorShape.PointingHandCursor)
            lien.setToolTip(f"Ouvrir l'étape {sec.step}")
            lien.clicked.connect(lambda _c=False, n=sec.step: self.modifier_demande.emit(n))
            top.addWidget(lien, 0, Qt.AlignmentFlag.AlignTop)
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
        """Mémorise les points à corriger (bloquants) et met à jour le bandeau."""
        self._validation_errors = list(errors or [])
        self._update_banner()

    def _update_banner(self) -> None:
        """Bandeau d'état combiné : configuration (bloquante) + vérifications.

        - points à corriger → rouge, rangés par étape ;
        - configuration valide mais vérification en cours → neutre ;
        - un outil manque → rouge, avec le renvoi au Dépannage ;
        - un contrôle en avertissement → orange, en le nommant ;
        - sinon → vert « Prêt à lancer. ».
        """
        if self._validation_errors:
            par_etape = classer_erreurs(self._validation_errors)
            lignes = []
            for etape in sorted(par_etape):
                for msg in par_etape[etape]:
                    lignes.append(f"• Étape {etape} : {msg}" if etape else f"• {msg}")
            self._banner.setText("<b>Lancement impossible :</b><br>" + "<br>".join(lignes))
            self._banner.setProperty("state", "incomplete")
        elif self._preflight_status == "pending":
            self._banner.setText("Vérification de l'environnement…")
            self._banner.setProperty("state", "pending")
        elif self._preflight_status == "err":
            quoi = ", ".join(self._pf_manquants) or "un outil"
            self._banner.setText(
                f"<b>Un outil manque : {quoi}.</b>&nbsp;&nbsp;"
                '<a href="manuel:depannage#les-verifications-prealables-ont-echoue">Voir l\'aide › Dépannage</a>'
            )
            self._banner.setProperty("state", "tool_missing")
        elif self._preflight_status == "warn":
            quoi = ", ".join(self._pf_avertissements) or "un contrôle"
            self._banner.setText(f"Prêt à lancer, avec un avertissement : {quoi}.")
            self._banner.setProperty("state", "warn")
        else:
            self._banner.setText("Prêt à lancer.")
            self._banner.setProperty("state", "ok")
        self._banner.setVisible(True)
        self._banner.style().unpolish(self._banner)
        self._banner.style().polish(self._banner)

    def _sur_lien(self, lien: str) -> None:
        """Lien du bandeau → le manuel, sur la rubrique de Dépannage."""
        if not lien.startswith("manuel:"):
            return
        from ..dialogs.aide_dialog import ouvrir_aide  # différé

        cle, ancre = resoudre_cible(lien[len("manuel:"):])
        racine = Path(__file__).resolve().parents[3]
        ouvrir_aide(racine / AIDE_DIRNAME, parent=self.window(), cle=cle or CLE_DEPANNAGE, ancre=ancre)

    # ------------------------------------------------------------------
    # Réglages du poste — dalles en parallèle
    # ------------------------------------------------------------------
    def set_workers(self, n: int) -> None:
        """Initialise le compteur depuis la configuration (sans émettre de signal)."""
        n = max(1, min(16, self._safe_int(n, 4)))
        self._workers = n
        self._workers_spin.blockSignals(True)
        self._workers_spin.setValue(n)
        self._workers_spin.blockSignals(False)
        self._adv.refresh_preview()

    def _set_workers(self, n: int) -> None:
        """Réagit à une interaction utilisateur (spin / boutons −+)."""
        n = max(1, min(16, self._safe_int(n, self._workers)))
        # Resynchronise le spin (cas du clamp aux bornes) sans boucler.
        self._workers_spin.blockSignals(True)
        self._workers_spin.setValue(n)
        self._workers_spin.blockSignals(False)
        if n == self._workers:
            return
        self._workers = n
        self._adv.refresh_preview()
        self.workers_changed.emit(n)

    def workers_value(self) -> int:
        return self._workers

    # ------------------------------------------------------------------
    # Réglages du poste — carte graphique
    # ------------------------------------------------------------------
    def set_carte_graphique(self, on: bool) -> None:
        """Initialise la case depuis la configuration (sans émettre de signal)."""
        self._carte = bool(on)
        self._carte_check.blockSignals(True)
        self._carte_check.setChecked(self._carte)
        self._carte_check.blockSignals(False)
        self._adv.refresh_preview()

    def _on_carte_toggled(self, on: bool) -> None:
        self._carte = bool(on)
        self._adv.refresh_preview()
        self.carte_graphique_changed.emit(self._carte)

    def carte_graphique_value(self) -> bool:
        return self._carte

    def set_decision_carte(self, decision: Optional[dict]) -> None:
        """La décision mémorisée pour ce poste (``carte_graphique.decision_memorisee``),
        ou ``None`` : la ligne et « Oublier » n'apparaissent que s'il y en a une."""
        if decision:
            self._carte_memo.setText(f"Décision mémorisée pour ce poste : {phrase_decision(decision)}.")
        self._carte_memo.setVisible(bool(decision))
        self._oublier_btn.setVisible(bool(decision))

    def _on_oublier(self) -> None:
        oublier(registre_path())
        self.set_decision_carte(None)

    def set_readonly(self, on: bool) -> None:
        """Pendant un traitement : les réglages du poste du traitement en cours ne se
        changent plus ; « Revérifier » et les liens « modifier » restent."""
        self._workers_body.setEnabled(not on)

    @staticmethod
    def _safe_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Vérifications (carte « Environnement »)
    # ------------------------------------------------------------------
    def refresh_preflight(self, config: dict) -> None:
        """Lance les vérifications en tâche de fond pour ``config``."""
        self._preflight_gen += 1
        gen = self._preflight_gen
        self._show_preflight_loading()
        try:
            from ...app.pipeline_controller import _files_as_dict
            from ...app.run_context import build_run_context

            ctx = build_run_context(config or {})
            kwargs = dict(
                mode=str(ctx.mode),
                cv_config=ctx.cv.raw,
                products=ctx.processing.products.as_dict(),
                files_config=_files_as_dict(ctx),
                output_dir=ctx.output_dir,
            )
        except Exception as e:  # noqa: BLE001
            self._render_preflight_error(str(e))
            return
        threading.Thread(
            target=self._preflight_worker, args=(gen, kwargs), daemon=True
        ).start()

    def _preflight_worker(self, gen: int, kwargs: dict) -> None:
        try:
            from ...pipeline.preflight import collect_preflight_results

            results = collect_preflight_results(**kwargs)
        except Exception:  # noqa: BLE001
            results = []
        self._pf_emitter.done.emit(gen, results)

    def _on_preflight_done(self, gen: int, results: list) -> None:
        if gen != self._preflight_gen:
            return  # résultat périmé (une vérification plus récente est en cours)
        self._render_preflight(results)

    def _show_preflight_loading(self) -> None:
        self._clear_grid()
        # Pastille neutre : le vert est réservé à des vérifications réellement passées.
        self._set_pill("●  …", "neutral")
        self._preflight_status = "pending"
        self._reverifier_btn.setEnabled(False)
        self._update_banner()
        lbl = QLabel("Vérification de l'environnement…")
        lbl.setObjectName("PreflightDetail")
        self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)

    def _render_preflight_error(self, msg: str) -> None:
        self._clear_grid()
        self._set_pill("●  ✗", "err")
        self._preflight_status = "err"
        self._pf_manquants = ["configuration illisible"]
        self._reverifier_btn.setEnabled(True)
        self._update_banner()
        lbl = QLabel(f"Configuration illisible : {msg}")
        lbl.setObjectName("PreflightDetail")
        lbl.setWordWrap(True)
        self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)

    def _render_preflight(self, results: list) -> None:
        self._clear_grid()
        self._reverifier_btn.setEnabled(True)
        self._pf_manquants = []
        self._pf_avertissements = []
        if not results:
            lbl = QLabel("Aucune vérification applicable pour cette configuration.")
            lbl.setObjectName("PreflightDetail")
            lbl.setWordWrap(True)
            self._preflight_grid.addWidget(lbl, 0, 0, 1, 3)
            self._set_pill("●  —", "neutral")
            self._preflight_status = "ok"
            self._update_banner()
            return

        n_ok = 0
        n_critical_fail = 0
        for row, r in enumerate(results):
            libelle = libelle_environnement(r.name)
            if r.ok:
                status, glyph = "ok", "✓"
                n_ok += 1
            elif not r.critical:
                status, glyph = "warn", "⚠"
                self._pf_avertissements.append(libelle)
            else:
                status, glyph = "err", "✗"
                n_critical_fail += 1
                self._pf_manquants.append(libelle)

            icon = QLabel(glyph)
            icon.setObjectName("PreflightIcon")
            icon.setProperty("status", status)
            icon.setFixedWidth(16)
            icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            self._preflight_grid.addWidget(icon, row, 0)

            # Le contrôle en français ; son nom technique en infobulle.
            name = QLabel(libelle)
            name.setObjectName("PreflightName")
            name.setToolTip(r.name)
            self._preflight_grid.addWidget(name, row, 1, Qt.AlignmentFlag.AlignTop)

            # Un mot dans la colonne (« trouvé », « manquant »…) ; le détail brut —
            # chemin, erreur — seulement en infobulle.
            detail = QLabel(resume_detail(r.details))
            detail.setObjectName("PreflightDetail")
            detail.setWordWrap(True)
            detail.setToolTip(r.details)
            detail.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self._preflight_grid.addWidget(detail, row, 2)

        total = len(results)
        if n_critical_fail:
            kind = "err"
        elif n_ok < total:
            kind = "warn"
        else:
            kind = "ok"
        self._set_pill(f"●  {n_ok}/{total}", kind)
        self._preflight_status = kind
        self._update_banner()

    def _set_pill(self, text: str, kind: str) -> None:
        self._preflight_pill.setText(text)
        self._preflight_pill.setProperty("kind", kind)
        self._preflight_pill.style().unpolish(self._preflight_pill)
        self._preflight_pill.style().polish(self._preflight_pill)

    def _clear_grid(self) -> None:
        # Retrait SYNCHRONE : deleteLater seul laissait « Vérification… » dans la grille
        # jusqu'au tour de boucle suivant, sous la première ligne des résultats.
        while self._preflight_grid.count():
            item = self._preflight_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
