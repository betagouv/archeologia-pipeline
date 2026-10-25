"""Étape 3 — Structures à chercher (sélection par structures).

L'utilisateur coche des **structures** ; l'orchestrateur
(:mod:`app.services.model_orchestrator`) résout automatiquement les modèles et
les analyses ``(modèle, image du relief)``. Chaque structure est une ligne sous
sa morphologie (:class:`~ui.widgets.entity_row.EntityRow`) : modèle retenu
(modifiable si plusieurs candidats), option « regrouper en zones » si le modèle
en propose, image lue par le modèle en orange si elle n'est pas cochée à
l'étape 2 (avec « l'ajouter »). Les fiches (structure, modèle) ne s'ouvrent pas
ici : la page émet ``fiche_requested`` / ``modele_requested`` et l'assistant les
pose dans son panneau latéral (interface rationalisée, 2026-10-24).
"""
from __future__ import annotations

import html
from pathlib import Path

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...app.services.model_orchestrator import (
    build_entity_coverage,
    discover_installed_models,
    effective_model_names,
    group_entities_by_morphology,
    layer_name_for_class,
    load_entities_catalog,
    load_model_card,
    resolve_runs_from_entities,
)
from ..widgets.card import build_card
from ..widgets.entity_row import EntityRow
from ..widgets.info_button import InfoButton
from ..widgets.profil_scores import couleur_de_classe
from ...app.services.fiabilite_observee import observer
from ...app.services.profil_scores import profil_pour_classe
from ...app.services.cout_modele import infobulle, libelle_menu
from ...app.services.reglages_defaut import (
    a_des_surcharges,
    effacer_surcharges,
    fusionner_surcharge,
    phrase_entite_reinitialisee,
)
from ...app.services.vocabulaire import pluriel
from ..widgets.toast import show_toast
from ..widgets.toggle_switch import ToggleSwitch


class _CadreInterrupteur(QFrame):
    """Le bandeau de l'interrupteur : un clic n'importe où dessus (titre, phrase,
    badge) bascule la détection — l'interrupteur seul était une cible de 36 px."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DetectionPage(QWidget):
    changed = pyqtSignal()
    activate_rvt = pyqtSignal(str)        # demande de cocher une image du relief à l'étape 2
    fiche_requested = pyqtSignal(str)     # id de structure → le panneau ⓘ de l'assistant
    modele_requested = pyqtSignal(object)  # InstalledModel → le panneau ⓘ de l'assistant

    def __init__(self, plugin_root, parent=None):
        super().__init__(parent)
        self._plugin_root = Path(plugin_root)
        self._catalog = load_entities_catalog(
            self._plugin_root / "data" / "entities_catalog.json"
        )
        self._installed = discover_installed_models(self._plugin_root / "data" / "models")
        self._coverage = {
            ec.entity.id: ec
            for ec in build_entity_coverage(self._catalog, self._installed)
        }
        self._models = {m.name: m for m in self._installed}
        self._model_cards: dict = {}  # nom de modèle -> model_card.yaml parsé (à la demande)
        self._profils_cache: dict = {}  # (modèle, classe) -> Profil ou None (évaluation lue une fois)

        self._enabled = False
        self._selected: dict = {}
        self._overrides: dict = {}
        self._cluster: set = set()
        self._entity_thresholds: dict = {}  # eid -> {confidence_threshold, min_area_m2}
        self._entity_cluster_params: dict = {}  # eid -> {eps_m, min_cluster_size, …}
        self._active_rvts: set = set()
        self._loading = False
        self._readonly = False
        self._rows: dict = {}
        self._run_rows: list = []
        self._content_scroll = None
        self._sel_count = None
        # Analyses d'une configuration ancienne (sans structures) à préserver tant
        # que l'utilisateur ne reprend pas la main par les structures. Cf. load_from.
        self._legacy_runs = None

        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Interrupteur (pilule + libellé + badge), tout le bandeau cliquable ──
        toggle = _CadreInterrupteur()
        toggle.setObjectName("DetectionToggle")
        toggle.setProperty("on", "false")
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.clicked.connect(self._basculer_interrupteur)
        self._toggle_frame = toggle
        tl = QHBoxLayout(toggle)
        tl.setContentsMargins(14, 10, 14, 10)
        tl.setSpacing(12)
        self._enable_check = ToggleSwitch()
        self._enable_check.toggled.connect(self._on_enable_toggled)
        title = QLabel("Chercher des structures automatiquement")
        title.setObjectName("DetectionSwitch")
        sub = QLabel("Cochez les structures à chercher : le modèle est choisi pour vous.")
        sub.setObjectName("WizardPageSub")
        sub.setWordWrap(True)  # sinon ses 869 px deviennent la largeur mini de la fenêtre
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(sub)
        opt = QLabel("FACULTATIF")
        opt.setObjectName("FacultatifTag")
        tl.addWidget(self._enable_check, 0, Qt.AlignmentFlag.AlignVCenter)
        tl.addLayout(text, 1)
        tl.addWidget(opt, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(toggle)

        # ── Détection désactivée ──
        self._empty_state = QFrame()
        self._empty_state.setObjectName("EmptyState")
        es = QVBoxLayout(self._empty_state)
        es.setContentsMargins(24, 24, 24, 24)
        es.setSpacing(6)
        es.setAlignment(Qt.AlignmentFlag.AlignCenter)
        es_title = QLabel("Détection désactivée")
        es_title.setObjectName("EmptyStateTitle")
        es_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        es_desc = QLabel(
            "Passez à l'étape suivante pour calculer les images seules, ou activez "
            "la détection pour y chercher des structures."
        )
        es_desc.setObjectName("WizardPageSub")
        es_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        es_desc.setWordWrap(True)
        self._es_btn = QPushButton("Chercher des structures")
        self._es_btn.clicked.connect(lambda: self._enable_check.setChecked(True))
        es.addWidget(es_title)
        es.addWidget(es_desc)
        es.addWidget(self._es_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # ── Contenu (détection activée) ──
        self._content = QWidget()
        cv = QVBoxLayout(self._content)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(12)

        ent_card, ev = build_card("Structures", "1")
        self._sel_count = ent_card.counter  # « X sur Y cochées » dans l'en-tête
        # Ni réinitialisation globale (demande utilisateur 2026-09-16) ni case « Seuils et
        # réglages par structure » (2026-10-25) : chaque ligne porte son « ▸ Réglages »
        # et son « ↺ », comme une image à l'étape 2.

        # ── Une liste par morphologie : en-tête puis une ligne par structure ──
        for key, label, glyph, ents in group_entities_by_morphology(self._catalog):
            header = QLabel(f"{glyph}  {label}")
            header.setObjectName("EntitySectionHeader")
            ev.addWidget(header)
            for entity in ents:
                ev.addWidget(self._make_entity_row(entity))
        cv.addWidget(ent_card)

        runs_card, rv = build_card("Ce qui sera analysé", "2")
        self._runs_count = runs_card.counter  # compteur dans l'en-tête de la carte
        self._runs_host = QWidget()
        self._runs_layout = QVBoxLayout(self._runs_host)
        self._runs_layout.setContentsMargins(0, 0, 0, 0)
        self._runs_layout.setSpacing(5)
        self._runs_layout.addStretch(1)  # garde les analyses compactes en haut
        rv.addWidget(self._runs_host)
        rv.addLayout(self._build_footer())
        cv.addWidget(runs_card)
        cv.addStretch(1)

        # Le contenu défile au niveau de la PAGE : la section ① a toute la place,
        # et l'utilisateur atteint la section ② en défilant.
        content_scroll = QScrollArea()
        content_scroll.setObjectName("DetectionScroll")
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Un seul sens de défilement : la page descend, elle ne va jamais de
        # côté (demande utilisateur 2026-09-17). La description de chaque ligne
        # s'élide pour que le contenu tienne vraiment dans la largeur — la
        # politique ci-dessous n'est que le garde-fou.
        content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        content_scroll.setWidget(self._content)
        self._content_scroll = content_scroll

        # Détection désactivée et contenu dans un QStackedWidget : sa taille = le MAX
        # des deux → activer/désactiver la détection ne redimensionne plus la fenêtre.
        self._body_stack = QStackedWidget()
        self._body_stack.addWidget(self._empty_state)    # index 0
        self._body_stack.addWidget(content_scroll)       # index 1
        root.addWidget(self._body_stack, 1)

    def _build_footer(self):
        footer = QHBoxLayout()
        footer.setSpacing(10)
        self._annot_check = QCheckBox("Enregistrer aussi les images annotées")
        self._annot_check.toggled.connect(self._on_changed)
        footer.addWidget(self._annot_check)
        footer.addStretch(1)
        return footer

    def _make_entity_row(self, entity) -> EntityRow:
        """Crée et câble la ligne d'une structure du catalogue."""
        ec = self._coverage.get(entity.id)
        row = EntityRow(entity.id, entity.label, entity.description)
        cand_names = list(ec.candidate_models if ec else ())
        candidates = [(name, self._models[name].display_name) for name in cand_names]
        # Coût structurel (fenêtres d'analyse par dalle) : menu « Changer ▾ » et
        # infobulle du modèle — un fait calculé d'args.yaml, jamais une durée.
        couts = {
            name: (
                libelle_menu(m.display_name, m.sahi_slice_px, m.sahi_overlap),
                infobulle(m.display_name, m.sahi_slice_px, m.sahi_overlap),
            )
            for name in cand_names
            for m in (self._models[name],)
        }
        row.set_candidates(candidates, couts=couts)
        vignette, cadrage = self._premiere_vignette(entity.id)
        row.set_fiche(vignette, disponible=bool(cand_names), cadrage=cadrage)
        row.fiche_requested.connect(self.fiche_requested)
        row.reglages_toggled.connect(self._on_reglages_toggled)
        row.modele_requested.connect(self._on_modele_demande)
        row.reset_requested.connect(self._on_reset_entity)
        row.toggled.connect(self._on_entity_toggled)
        row.models_changed.connect(self._on_models_changed)
        row.cluster_toggled.connect(self._on_cluster_toggled)
        row.activate_rvt.connect(self.activate_rvt)
        row.thresholds_changed.connect(self._on_thresholds_changed)
        row.cluster_params_changed.connect(self._on_cluster_params_changed)
        self._rows[entity.id] = row
        return row

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _basculer_interrupteur(self) -> None:
        if self._enable_check.isEnabled():
            self._enable_check.toggle()

    def _on_enable_toggled(self, checked: bool) -> None:
        self._enabled = checked
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_entity_toggled(self, entity_id: str, selected: bool) -> None:
        if entity_id in self._incluses():
            return  # incluse par une cible dérivée cochée : c'est elle qu'on décoche
        self._selected[entity_id] = selected
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _incluses(self) -> dict:
        """``{structure de base: cible dérivée cochée qui l'inclut}``.

        Cocher « Regroupement de cratères » produit déjà la couche Cratères
        (une seule, dans le groupe du regroupement) : « Cratères » est alors
        cochée d'office, non décochable, et son seuil se règle sur la ligne du
        regroupement. ``self._selected`` reste le choix EXPLICITE de
        l'utilisateur ; l'inclusion se recalcule, elle n'est jamais persistée.
        """
        out: dict = {}
        for derived, on in self._selected.items():
            if not on:
                continue
            ec = self._coverage.get(derived)
            for name in (effective_model_names(ec, self._overrides) if ec else []):
                model = self._models.get(name)
                for base, d in (model.implied_entities.items() if model else ()):
                    if d == derived:
                        out[base] = derived
        return out

    def _selection_effective(self) -> list:
        """Structures cochées explicitement + celles incluses par une dérivée cochée."""
        explicites = [e for e, on in self._selected.items() if on]
        return explicites + [b for b in self._incluses() if not self._selected.get(b)]

    def _on_models_changed(self, entity_id: str, model_names: list) -> None:
        # 1..n modèles cochés dans le menu de la ligne (≥2 = comparaison A/B).
        # La surcharge persistée accepte str (legacy) ou liste — on stocke la liste.
        self._overrides[entity_id] = [str(n) for n in model_names]
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_cluster_toggled(self, entity_id: str, on: bool) -> None:
        if on:
            self._cluster.add(entity_id)
        else:
            self._cluster.discard(entity_id)
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_changed(self, *_args) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_modele_demande(self, model_name: str) -> None:
        """Vignette d'une ligne : la fiche du modèle, dans le panneau (onglet Modèle)."""
        model = self._models.get(model_name)
        if model is not None:
            self.modele_requested.emit(model)

    def _on_reglages_toggled(self, _entity_id: str, _on: bool) -> None:
        # Les profils d'une ligne ne sont chargés que dépliée : rafraîchir pour les poser.
        self._refresh()

    def _on_thresholds_changed(self, entity_id: str, changes: dict) -> None:
        """Une seule clé à la fois (C-01) : régler le seuil ne fige pas l'aire minimale,
        qui continue de suivre le modèle. La valeur est stockée telle qu'émise."""
        self._entity_thresholds = fusionner_surcharge(
            self._entity_thresholds, entity_id, dict(changes or {})
        )
        self._armer_reinit(entity_id)
        if not self._loading:
            self.changed.emit()

    def _on_cluster_params_changed(self, entity_id: str, params: dict) -> None:
        self._entity_cluster_params[entity_id] = dict(params or {})
        self._armer_reinit(entity_id)
        if not self._loading:
            self.changed.emit()

    def _armer_reinit(self, entity_id: str) -> None:
        """Le « ↺ » de la ligne s'active au moment même où une surcharge apparaît."""
        row = self._rows.get(entity_id)
        if row is not None and not self._readonly:
            row.set_reinit_possible(True)

    def _cle_surcharge(self, entity_id: str) -> str:
        """Structure sous laquelle les surcharges de ``entity_id`` sont rangées.

        Une structure incluse par une cible dérivée n'a pas de seuil propre : c'est
        celui de la dérivée qui s'applique (règle 2026-09-15). Sa ligne de réglages
        est d'ailleurs désactivée, bouton compris ; cette résolution n'est donc
        qu'une sécurité si le chemin change.
        """
        return self._incluses().get(entity_id) or entity_id

    def _on_reset_entity(self, entity_id: str) -> None:
        """Rétablit les valeurs du modèle pour UNE structure. Les autres sont intactes.

        On retire les entrées plutôt que d'y réécrire les valeurs du modèle : une
        valeur recopiée redeviendrait une surcharge, figée sur l'ancien modèle, et
        survivrait à un changement de modèle.
        """
        cle = self._cle_surcharge(entity_id)
        touches = effacer_surcharges(
            cle, self._entity_thresholds, self._entity_cluster_params
        )
        if not touches:
            return
        # Pas besoin de geler les signaux : EntityRow.update_state met déjà son
        # propre `_loading` autour de ses setValue, et n'émet donc pas pendant
        # qu'on repeuple les cases avec les défauts du modèle.
        self._refresh()
        couverture = self._coverage.get(cle)
        show_toast(self, phrase_entite_reinitialisee(
            couverture.entity.label if couverture else cle, touches
        ))
        if not self._loading:
            self.changed.emit()

    # ------------------------------------------------------------------
    # Rafraîchissement
    # ------------------------------------------------------------------
    @staticmethod
    def _default_conf_entite(model, eid: str) -> float:
        """Défaut de confiance affiché pour UNE structure.

        Si le model_card porte des seuils par classe (mesurés au banc), la structure
        hérite du seuil de SES classes — c'est aussi la valeur vers laquelle le
        « ↺ » de la ligne la ramène. ``min`` si elle couvre plusieurs classes aux
        seuils différents (cohérent avec le plancher de décodage). Sinon, défaut
        global du modèle, comme avant.
        """
        if model is None:
            return 0.3   # = model_config.DEFAULT_CONFIDENCE (0,2 ici jusqu'alors)
        pc = getattr(model, "default_confidence_per_class", None) or {}
        vals = [pc[c] for c in model.coverage.get(eid, ()) if c in pc]
        return float(min(vals)) if vals else float(model.default_confidence)

    @staticmethod
    def _fiabilite_hint(model, eid: str, conf_override) -> str:
        """Aide « Fiabilité affichée — douteux dès 0,29 · … » pour UNE structure : les
        catégories EFFECTIVES de chacune de ses classes au seuil courant (surcharge
        UI si posée, sinon seuil par classe du modèle) — même règle que le traitement."""
        if model is None or not getattr(model, "fiabilite_per_class", None):
            return ""
        try:
            from ...app.services.fiabilite import categories_effectives, hint_etape3
        except ImportError:
            from app.services.fiabilite import categories_effectives, hint_etape3  # type: ignore[no-redef]
        pc = getattr(model, "default_confidence_per_class", None) or {}
        par_classe = {}
        for c in model.coverage.get(eid, ()):
            cats = model.fiabilite_per_class.get(c)
            if not cats:
                continue
            seuil = (float(conf_override) if conf_override is not None
                     else float(pc.get(c, model.default_confidence)))
            par_classe[c] = categories_effectives(
                cats, seuil, sous_seuil=(getattr(model, "fiabilite_sous_seuil", None) or {}).get(c))
        return hint_etape3(par_classe)

    def _refresh(self) -> None:
        self._enable_check.setChecked(self._enabled)
        # Le bandeau ne doit dire « actif » (bleu) que si la détection l'est :
        # partout ailleurs dans le thème, #d6e6f4 + #2b79c2 code l'état coché.
        self._toggle_frame.setProperty("on", "true" if self._enabled else "false")
        self._toggle_frame.style().unpolish(self._toggle_frame)
        self._toggle_frame.style().polish(self._toggle_frame)
        self._body_stack.setCurrentIndex(1 if self._enabled else 0)
        if not self._enabled:
            return

        incluses = self._incluses()
        for eid, row in self._rows.items():
            ec = self._coverage.get(eid)
            model_names = effective_model_names(ec, self._overrides) if ec else []
            # Modèle « primaire » (1ᵉʳ de la liste) : porte les défauts des
            # cases. En comparaison A/B les surcharges de seuils s'appliquent
            # identiquement aux deux analyses (comparaison à seuil égal).
            model = self._models.get(model_names[0]) if model_names else None
            # Garde sur TOUS les modèles cochés : sans elle, l'analyse du modèle
            # secondaire dont l'image n'est pas cochée à l'étape 2 partirait sans
            # avertissement et échouerait à l'exécution.
            rvts: list = []
            for _n in model_names:
                _m = self._models.get(_n)
                if _m and _m.target_rvt not in rvts:
                    rvts.append(_m.target_rvt)
            rvt = " + ".join(rvts) if rvts else "—"
            missing_rvts = [r for r in rvts if r not in self._active_rvts]
            cluster_outputs = model.cluster_options.get(eid, ()) if model else ()
            is_derived = bool(model) and eid in model.derived_entities
            # Structure incluse par une dérivée cochée : cochée d'office, ses seuils
            # sont CEUX de la dérivée (un seul réglage, affiché ici en lecture seule).
            par = incluses.get(eid)
            ov = self._entity_thresholds.get(par or eid, {})
            label_par = self._coverage[par].entity.label if par else ""
            conf_label, conf_tip = "Confiance", ""
            if par:
                conf_tip = f"Réglé sur la ligne « {label_par} », qui inclut cette structure"
            elif is_derived and model.derived_source_classes.get(eid):
                sources = model.derived_source_labels.get(eid) or ", ".join(
                    self._coverage[b].entity.label
                    for b, d in model.implied_entities.items() if d == eid
                ) or "détections sources"
                defaut = f"{self._default_conf_entite(model, eid):.2f}".replace(".", ",")
                conf_label = f"Confiance des {sources.lower()}"
                conf_tip = (
                    f"Seuil appliqué aux {sources.lower()} détectés, dont les zones sont "
                    f"calculées : le relever change les zones (règle calibrée au seuil {defaut})"
                )
            # Défauts des paramètres de regroupement (DBSCAN) pour cette structure :
            # ceux de la 1ʳᵉ sortie de regroupement qu'elle produit (dérivée ou cluster).
            cluster_default_params: dict = {}
            if model:
                if is_derived:
                    src = set(model.derived_source_classes.get(eid, ()))
                    outs = [c for c in model.coverage.get(eid, ()) if c not in src]
                else:
                    outs = list(cluster_outputs)
                for oc in outs:
                    if oc in model.cluster_defaults:
                        cluster_default_params = model.cluster_defaults[oc]
                        break
            # Aide au choix du seuil (réglages) : mini-profils des classes de la
            # structure sur le modèle primaire, lus une fois par (modèle, classe).
            row.set_profils(
                self._profils_entite(model, eid, len(model_names) > 1)
                if (model is not None and row.est_deplie() and not par) else []
            )
            row.update_state(
                selected=bool(self._selected.get(eid) or par),
                implique_par=label_par,
                conf_label=conf_label,
                conf_tip=conf_tip,
                current_models=model_names,
                rvt=rvt,
                rvt_active=(not missing_rvts) if model else True,
                missing_rvt=missing_rvts[0] if missing_rvts else None,
                cluster_outputs=cluster_outputs,
                cluster_on=eid in self._cluster,
                default_confidence=self._default_conf_entite(model, eid),
                default_min_area=model.default_min_area if model else 0.0,
                conf_override=ov.get("confidence_threshold"),
                area_override=ov.get("min_area_m2"),
                is_derived=is_derived,
                cluster_default_params=cluster_default_params,
                cluster_params_override=self._entity_cluster_params.get(eid),
                fiabilite_hint=self._fiabilite_hint(model, eid, ov.get("confidence_threshold")),
                # Le « ↺ » de la ligne ne sert que si CETTE structure a été réglée.
                reinit_possible=not self._readonly and a_des_surcharges(
                    par or eid, self._entity_thresholds, self._entity_cluster_params
                ),
            )
        self._update_selection_count()
        self._rebuild_runs()

    def _update_selection_count(self) -> None:
        if self._sel_count is None:
            return
        total = sum(
            1 for e in self._catalog
            if (self._coverage.get(e.id) and self._coverage[e.id].default_model)
        )
        n = len(self._selection_effective())
        self._sel_count.setText(f"{n} sur {total} cochée{'s' if n != 1 else ''}")

    def _rebuild_runs(self) -> None:
        # Retrait SYNCHRONE des anciennes lignes : deleteLater() seul les laisse
        # dans le layout (espace fantôme) jusqu'au tour de boucle suivant → churn
        # de hauteur + repaint = clignotement. removeWidget + setParent(None)
        # détachent immédiatement ; deleteLater ne sert plus qu'au nettoyage mémoire.
        for w in self._run_rows:
            self._runs_layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._run_rows = []

        selected_ids = self._selection_effective()
        runs = resolve_runs_from_entities(
            selected_ids, self._overrides, self._installed, self._catalog, self._cluster,
            entity_thresholds=self._entity_thresholds,
            entity_cluster_params=self._entity_cluster_params,
        )
        self._runs_count.setText(f"{pluriel(len(runs), 'analyse')} · un modèle lit une image")
        if not runs:
            row = QLabel("Rien à analyser : cochez une structure.")
            row.setObjectName("EntityDesc")
            self._runs_layout.insertWidget(0, row)  # au-dessus du stretch final
            self._run_rows.append(row)
            return
        for idx, run in enumerate(runs):
            model = self._models.get(run["model"])
            disp = model.display_name if model else run["model"]
            # Une phrase par analyse : « <modèle> lit l'image <sigle> : Cratères, Parcellaire »
            # (libellés du catalogue portés par run["entities"] ; repli sur les noms
            # de classes bruts pour une analyse d'une configuration ancienne).
            noms = [
                e["label"] for e in (run.get("entities") or []) if e.get("label")
            ] or [str(c) for c in (run.get("selected_classes") or [])]
            row = QFrame()
            row.setObjectName("RunRow")
            h = QHBoxLayout(row)
            h.setContentsMargins(10, 6, 10, 6)
            h.setSpacing(8)
            texte = QLabel(
                f"<b>{html.escape(str(disp))}</b> lit l'image <b>{html.escape(str(run['target_rvt']))}</b>"
                f" : {html.escape(', '.join(noms))}"
            )
            texte.setObjectName("RunTexte")
            texte.setTextFormat(Qt.TextFormat.RichText)
            texte.setWordWrap(True)  # nom long = largeur mini de la page sinon
            # Le « ⓘ » du modèle : sa fiche dans le panneau latéral. Désactivé si le
            # modèle n'est pas résolu (analyse orpheline, ne devrait pas arriver).
            info_btn = InfoButton("Fiche du modèle : architecture, entraînement, évaluation")
            if model is not None:
                # Capture du modèle dans la signature par défaut pour figer la
                # référence (sans ça, toutes les lambdas pointeraient sur le dernier).
                info_btn.clicked.connect(lambda _checked=False, m=model: self.modele_requested.emit(m))
            else:
                info_btn.setEnabled(False)
            h.addWidget(texte, 1)
            h.addWidget(info_btn, 0, Qt.AlignmentFlag.AlignTop)
            self._runs_layout.insertWidget(idx, row)  # au-dessus du stretch final
            self._run_rows.append(row)

    # ------------------------------------------------------------------
    # Fiche de structure (illustration + provenance des données d'entraînement)
    # ------------------------------------------------------------------
    def _model_card(self, model_name: str):
        """``model_card.yaml`` parsé, mémorisé par modèle.

        ``discover_installed_models`` lit déjà ces fichiers mais n'en garde que
        l'extrait dont l'orchestrateur a besoin ; la fiche veut la carte
        entière. Lecture à la demande, une fois par modèle et par session.
        """
        if model_name in self._model_cards:
            return self._model_cards[model_name]
        model = self._models.get(model_name)
        card = None
        if model is not None and model.model_dir is not None:
            card = load_model_card(model.model_dir)
        self._model_cards[model_name] = card
        return card

    def _fiches_for_entity(self, entity_id: str):
        """Les fiches des classes qui portent ``entity_id``, tous modèles
        effectifs confondus (une comparaison A/B en produit une par modèle)."""
        from ...app.services.class_fiche import fiches_par_entite

        out = []
        ec = self._coverage.get(entity_id)
        for name in (effective_model_names(ec, self._overrides) if ec else []):
            card = self._model_card(name)
            model = self._models.get(name)
            if card is None or model is None:
                continue
            out.extend(fiches_par_entite(card, model.coverage.get(entity_id, ())))
        return out

    def _libelle(self, eid: str) -> str:
        """Le libellé de la structure, celui que portent ses couches (N-01)."""
        couverture = self._coverage.get(eid)
        return couverture.entity.label if couverture else eid

    def _profils_entite(self, model, eid: str, compared: bool) -> list:
        """``[(classe, Profil, couleur), …]`` des classes de ``eid`` dont le modèle
        livre son évaluation. La couleur est celle de la **couche** (qualifiée
        par le modèle en comparaison A/B), comme dans la légende."""
        out = []
        if model is None or model.model_dir is None:
            return out
        for classe in model.coverage.get(eid, ()):
            cats = (getattr(model, "fiabilite_per_class", None) or {}).get(classe)
            if not cats:
                continue
            cle = (model.name, classe)
            if cle not in self._profils_cache:
                self._profils_cache[cle] = profil_pour_classe(Path(model.model_dir), classe, cats)
            profil = self._profils_cache[cle]
            if profil is not None:
                couleur = couleur_de_classe(
                    layer_name_for_class(model, eid, classe, compared, label=self._libelle(eid)),
                    ancien="" if compared else classe,
                )
                out.append((classe, profil, couleur))
        return out

    def _premiere_vignette(self, entity_id: str):
        """``(chemin absolu, cadrage)`` de la vignette d'icône, ou ``(None, None)``.

        La première vignette du premier modèle qui existe réellement sur
        disque ; son ``cadrage`` dit quelle fenêtre l'icône découpe.
        """
        for fiche in self._fiches_for_entity(entity_id):
            model = self._models.get(fiche.modele_id)
            if model is None or model.model_dir is None:
                continue
            for v in fiche.vignettes:
                p = Path(model.model_dir) / v.brut
                if p.is_file():
                    return str(p), v.cadrage
        return None, None

    def corps_fiche(self, entity_id: str):
        """``(corps, titre)`` de la fiche de la structure, à poser dans le panneau ⓘ de
        l'assistant ; ``None`` s'il n'y a rien à montrer. Le corps émet
        ``modele_demande(model)`` pour son lien « Fiche du modèle ». Import différé
        des widgets de fiche : la plupart des sessions n'en ouvrent aucune."""
        fiches = self._fiches_for_entity(entity_id)
        if not fiches:
            return None
        from ..dialogs.class_info_dialog import corps_fiche_entite

        dirs = {
            m.name: m.model_dir for m in self._installed if m.model_dir is not None
        }
        couverture = self._coverage.get(entity_id)
        titre = couverture.entity.label if couverture else entity_id
        # Couleur de la COUCHE (qualifiée par le modèle en comparaison A/B) et
        # seuil effectif de chaque classe (surcharge UI, sinon seuil du modèle) :
        # la figure de la fiche montre la même couleur et le même seuil que le traitement.
        noms = effective_model_names(couverture, self._overrides) if couverture else []
        compared = len(noms) > 1
        ov = self._entity_thresholds.get(self._incluses().get(entity_id) or entity_id, {})
        couleurs, seuils = {}, {}
        for f in fiches:
            model = self._models.get(f.modele_id)
            if model is None:
                continue
            couleurs[(f.modele_id, f.nom)] = couleur_de_classe(
                layer_name_for_class(model, entity_id, f.nom, compared, label=titre),
                ancien="" if compared else f.nom,
            )
            conf = ov.get("confidence_threshold")
            pc = getattr(model, "default_confidence_per_class", None) or {}
            seuils[(f.modele_id, f.nom)] = (
                float(conf) if conf is not None else float(pc.get(f.nom, model.default_confidence))
            )
        # Fiabilité observée : vos verdicts (champ validation) sur les traitements connus.
        from ..run_view import profil_archeologia

        observes = {}
        for f in fiches:
            try:
                observes[(f.modele_id, f.nom)] = observer(profil_archeologia(), f.modele_id, f.nom)
            except Exception:  # noqa: BLE001 — jamais bloquant
                continue
        corps = corps_fiche_entite(
            fiches, dirs, titre, models=self._models,
            couleurs=couleurs, seuils=seuils, observes=observes,
        )
        return corps, titre

    # ------------------------------------------------------------------
    # API page
    # ------------------------------------------------------------------
    def set_active_rvts(self, rvt_keys) -> None:
        self._active_rvts = set(rvt_keys or [])
        self._refresh()

    def set_readonly(self, ro: bool) -> None:
        """Verrouille la saisie pour consultation pendant un traitement (lecture seule).

        Désactive l'interrupteur, les réglages des lignes et la case « images
        annotées » ; garde actifs le défilement, les « ⓘ » (le panneau des fiches
        reste ouvrable) et le « ▸ Réglages » de chaque ligne (révèle les seuils en
        lecture seule). N'agit que sur ``setEnabled`` → aucun ``changed``/autosave.
        """
        self._readonly = bool(ro)
        self._enable_check.setEnabled(not ro)
        self._annot_check.setEnabled(not ro)
        self._es_btn.setEnabled(not ro)
        for row in self._rows.values():
            row.set_readonly(ro)
        if not ro:
            self._refresh()   # le « ↺ » d'une structure réglée redevient actif (reinit_possible)

    def summary(self) -> str:
        if not self._enabled:
            return "désactivée"
        selected_ids = self._selection_effective()
        if not selected_ids:
            if self._legacy_runs:
                n = len(self._legacy_runs)
                return f"{pluriel(n, 'analyse')} (configuration importée)"
            return "aucune structure"
        runs = resolve_runs_from_entities(
            selected_ids, self._overrides, self._installed, self._catalog, self._cluster
        )
        n_ent = len(selected_ids)
        n_mod = len({r["model"] for r in runs})
        return f"{pluriel(n_ent, 'structure')} · {pluriel(n_mod, 'modèle')}"

    def recap_entities(self) -> list:
        """Libellés des structures cochées, dans l'ordre du catalogue (récap étape 4)."""
        if not self._enabled:
            return []
        effectives = set(self._selection_effective())
        return [e.label for e in self._catalog if e.id in effectives]

    def recap_runs(self) -> list:
        """Une chaîne par analyse résolue : « <modèle> sur <image> » (récap étape 4).

        Couvre aussi le cas d'une configuration ancienne (analyses préservées
        sans structure sélectionnée).
        """
        if not self._enabled:
            return []
        selected_ids = self._selection_effective()
        if selected_ids:
            runs = resolve_runs_from_entities(
                selected_ids, self._overrides, self._installed, self._catalog,
                self._cluster, entity_thresholds=self._entity_thresholds,
            )
        else:
            runs = self._legacy_runs or []
        out = []
        for run in runs:
            model = self._models.get(run.get("model"))
            disp = model.display_name if model else (run.get("model") or "?")
            out.append(f"{disp} sur {run.get('target_rvt', 'LD')}")
        return out

    def model_count(self) -> int:
        """Nombre de modèles distincts impliqués (sous-libellé du rail)."""
        selected_ids = self._selection_effective()
        if selected_ids:
            runs = resolve_runs_from_entities(
                selected_ids, self._overrides, self._installed, self._catalog, self._cluster
            )
        else:
            runs = self._legacy_runs or []
        return len({r.get("model") for r in runs if r.get("model")})

    def load_from(self, config: dict) -> None:
        cv = config.get("computer_vision") or {}
        prev = self._loading
        self._loading = True
        try:
            self._enabled = bool(cv.get("enabled", False))
            self._selected = {e: True for e in (cv.get("selected_entities") or [])}
            self._overrides = dict(cv.get("entity_model_overrides") or {})
            self._cluster = set(cv.get("entity_cluster_enabled") or [])
            self._entity_thresholds = dict(cv.get("entity_thresholds") or {})
            self._entity_cluster_params = dict(cv.get("entity_cluster_params") or {})
            self._annot_check.setChecked(bool(cv.get("generate_annotated_images", False)))
            # Rétrocompat : configuration ancienne avec des analyses explicites mais
            # aucune structure → on les préserve (jamais d'écrasement silencieux) tant
            # que l'utilisateur ne sélectionne pas de structure (cf. collect_into).
            legacy = cv.get("runs") or []
            self._legacy_runs = list(legacy) if (legacy and not self._selected) else None
            # Configuration d'avant l'inclusion (2026-09-15) : un seuil posé sur « Cratères »
            # avec le regroupement coché s'appliquait ; il se règle désormais sur la
            # ligne du regroupement → on l'y recopie si elle n'en a pas, sinon le
            # choix enregistré serait ignoré en silence.
            for base, derived in self._incluses().items():
                if base in self._entity_thresholds and derived not in self._entity_thresholds:
                    self._entity_thresholds[derived] = dict(self._entity_thresholds[base])
            self._refresh()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        cv = config.setdefault("computer_vision", {})
        cv["enabled"] = self._enabled
        selected_ids = self._selection_effective()
        if selected_ids:
            self._legacy_runs = None  # l'utilisateur pilote par structures désormais
        # Persisté : le choix EXPLICITE seulement — une structure incluse par une
        # dérivée se recalcule au rechargement (sinon elle resterait cochée
        # après qu'on a décoché la dérivée).
        cv["selected_entities"] = [e for e, on in self._selected.items() if on]
        cv["entity_model_overrides"] = {
            e: m for e, m in self._overrides.items() if self._selected.get(e)
        }
        cv["entity_cluster_enabled"] = sorted(e for e in self._cluster if self._selected.get(e))
        # Seuils et paramètres de regroupement de TOUTES les structures, cochées ou
        # non (C-02, 2026-10-24) : décocher puis recocher ne perd pas un réglage.
        # L'aval ne lit que ceux des structures cochées (resolve_runs_from_entities).
        cv["entity_thresholds"] = dict(self._entity_thresholds)
        cv["entity_cluster_params"] = dict(self._entity_cluster_params)
        cv["generate_annotated_images"] = self._annot_check.isChecked()
        cv["generate_shapefiles"] = True  # toujours : le GeoPackage des détections est produit
        # Analyses pour le traitement (l'aval ne consomme que 'runs') : résolues depuis
        # les structures, ou — configuration ancienne sans structures — préservées telles quelles.
        if not self._enabled:
            cv["runs"] = []
        elif selected_ids:
            cv["runs"] = resolve_runs_from_entities(
                selected_ids, self._overrides, self._installed, self._catalog, self._cluster,
                entity_thresholds=self._entity_thresholds,
                entity_cluster_params=self._entity_cluster_params,
            )
        elif self._legacy_runs is not None:
            cv["runs"] = self._legacy_runs
        else:
            cv["runs"] = []
