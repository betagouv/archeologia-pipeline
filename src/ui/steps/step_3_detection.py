"""Étape 3 — Détection IA (sélection par entités).

L'utilisateur coche des **entités** ; l'orchestrateur
(:mod:`app.services.model_orchestrator`) résout automatiquement les modèles et
les runs ``(modèle, RVT)``. Chaque entité affiche son modèle assigné (modifiable
si plusieurs candidats) et une option « regrouper en zones » (clustering) si le
modèle en propose. Le tag RVT passe en orange si l'indice n'est pas activé à
l'étape 2 (avec un bouton « + Activer »).
"""
from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
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
    group_entities_by_morphology,
    load_entities_catalog,
    resolve_runs_from_entities,
)
from ..widgets.card import build_card
from ..widgets.entity_card import EntityCard
from ..widgets.toggle_switch import ToggleSwitch


class DetectionPage(QWidget):
    changed = pyqtSignal()
    activate_rvt = pyqtSignal(str)  # demande d'activer un indice RVT à l'étape 2

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

        self._enabled = False
        self._advanced = False
        self._selected: dict = {}
        self._overrides: dict = {}
        self._cluster: set = set()
        self._entity_thresholds: dict = {}  # eid -> {confidence_threshold, min_area_m2}
        self._entity_cluster_params: dict = {}  # eid -> {eps_m, min_cluster_size, …}
        self._active_rvts: set = set()
        self._loading = False
        self._cards: dict = {}
        self._filter = "all"            # filtre morphologique courant (affichage only)
        self._filter_buttons: dict = {}
        self._section_widgets: dict = {}  # morpho_key -> (header, container)
        self._run_rows: list = []
        self._content_scroll = None
        self._sel_count = None
        # Runs d'une config legacy (sans entités) à préserver tant que
        # l'utilisateur ne reprend pas la main par les entités. Cf. load_from.
        self._legacy_runs = None

        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Toggle d'activation (interrupteur pilule + libellé + badge) ──
        toggle = QFrame()
        toggle.setObjectName("DetectionToggle")
        tl = QHBoxLayout(toggle)
        tl.setContentsMargins(14, 10, 14, 10)
        tl.setSpacing(12)
        self._enable_check = ToggleSwitch()
        self._enable_check.toggled.connect(self._on_enable_toggled)
        title = QLabel("Détection automatique par IA")
        title.setObjectName("DetectionSwitch")
        sub = QLabel("Sélectionnez les entités à détecter — les modèles sont choisis automatiquement.")
        sub.setObjectName("WizardPageSub")
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(sub)
        opt = QLabel("FACULTATIF")
        opt.setObjectName("FacultatifTag")
        tl.addWidget(self._enable_check, 0, Qt.AlignVCenter)
        tl.addLayout(text, 1)
        tl.addWidget(opt, 0, Qt.AlignVCenter)
        root.addWidget(toggle)

        # ── Empty state (détection désactivée) ──
        self._empty_state = QFrame()
        self._empty_state.setObjectName("EmptyState")
        es = QVBoxLayout(self._empty_state)
        es.setContentsMargins(24, 24, 24, 24)
        es.setSpacing(6)
        es.setAlignment(Qt.AlignCenter)
        es_title = QLabel("Détection IA désactivée")
        es_title.setObjectName("EmptyStateTitle")
        es_title.setAlignment(Qt.AlignCenter)
        es_desc = QLabel(
            "Passez à l'étape 4 pour calculer les rasters seuls, ou activez la "
            "détection pour analyser automatiquement vos indices."
        )
        es_desc.setObjectName("WizardPageSub")
        es_desc.setAlignment(Qt.AlignCenter)
        es_desc.setWordWrap(True)
        es_btn = QPushButton("Activer la détection")
        es_btn.clicked.connect(lambda: self._enable_check.setChecked(True))
        es.addWidget(es_title)
        es.addWidget(es_desc)
        es.addWidget(es_btn, 0, Qt.AlignCenter)

        # ── Contenu (détection activée) ──
        self._content = QWidget()
        cv = QVBoxLayout(self._content)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(12)

        ent_card, ev = build_card("Entités à détecter", "1")
        self._sel_count = ent_card.counter  # « X sur Y sélectionnées » dans l'en-tête
        adv_row = QHBoxLayout()
        adv_row.addStretch(1)
        self._adv_check = QCheckBox("Réglages avancés (seuils par entité)")
        self._adv_check.setObjectName("WizardPageSub")
        self._adv_check.toggled.connect(self._on_advanced_toggled)
        adv_row.addWidget(self._adv_check)
        ev.addLayout(adv_row)

        # ── Filtres morphologiques (organisation seulement, jamais la sélection) ──
        groups = group_entities_by_morphology(self._catalog)
        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(6)
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        self._add_filter_chip(chip_row, "all", "Tout")
        for key, label, glyph, _ents in groups:
            self._add_filter_chip(chip_row, key, f"{glyph} {label.split(' / ')[0]}")
        chip_row.addStretch(1)
        ev.addLayout(chip_row)
        hint = QLabel("Le filtre n'affecte que l'affichage, pas la sélection.")
        hint.setObjectName("EntityDesc")
        ev.addWidget(hint)

        # ── Sections par morphologie (en-tête + cartes en 2 colonnes) ──
        for key, label, glyph, ents in groups:
            header = QLabel(f"{glyph}  {label}")
            header.setObjectName("EntitySectionHeader")
            ev.addWidget(header)
            container = QWidget()
            container.setObjectName("EntitySection")
            cols = QHBoxLayout(container)
            cols.setContentsMargins(0, 0, 0, 0)
            cols.setSpacing(8)
            left = QVBoxLayout()
            left.setContentsMargins(0, 0, 0, 0)
            left.setSpacing(8)
            right = QVBoxLayout()
            right.setContentsMargins(0, 0, 0, 0)
            right.setSpacing(8)
            # Deux colonnes INDÉPENDANTES : développer une carte ne pousse que
            # les cartes sous elle dans la même colonne (pas toute la grille).
            for i, entity in enumerate(ents):
                card = self._make_entity_card(entity)
                (left if i % 2 == 0 else right).addWidget(card)
            left.addStretch(1)
            right.addStretch(1)
            cols.addLayout(left)
            cols.addLayout(right)
            ev.addWidget(container)
            self._section_widgets[key] = (header, container)
        cv.addWidget(ent_card)

        runs_card, rv = build_card("Runs IA programmés", "2")
        self._runs_count = runs_card.counter  # compteur dans l'en-tête de la carte
        self._runs_host = QWidget()
        self._runs_layout = QVBoxLayout(self._runs_host)
        self._runs_layout.setContentsMargins(0, 0, 0, 0)
        self._runs_layout.setSpacing(5)
        self._runs_layout.addStretch(1)  # garde les runs compacts en haut
        rv.addWidget(self._runs_host)
        rv.addLayout(self._build_footer())
        cv.addWidget(runs_card)
        cv.addStretch(1)

        # Le contenu défile au niveau de la PAGE : la section ① a toute la place,
        # et l'utilisateur atteint la section ② en scrollant (pas de sous-zone
        # défilante bornée qui rétrécit ①).
        content_scroll = QScrollArea()
        content_scroll.setObjectName("DetectionScroll")
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.NoFrame)
        content_scroll.setWidget(self._content)
        self._content_scroll = content_scroll

        # Empty-state et contenu dans un QStackedWidget : sa taille = le MAX des
        # deux → activer/désactiver la détection ne redimensionne plus la fenêtre.
        self._body_stack = QStackedWidget()
        self._body_stack.addWidget(self._empty_state)    # index 0
        self._body_stack.addWidget(content_scroll)       # index 1
        root.addWidget(self._body_stack, 1)

    def _build_footer(self):
        footer = QHBoxLayout()
        footer.setSpacing(10)
        self._annot_check = QCheckBox("Générer images annotées")
        self._annot_check.toggled.connect(self._on_changed)
        footer.addWidget(self._annot_check)
        footer.addStretch(1)
        return footer

    def _add_filter_chip(self, row, key: str, text: str) -> None:
        btn = QPushButton(text)
        btn.setObjectName("EntityChip")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setChecked(key == "all")
        btn.clicked.connect(lambda _checked=False, k=key: self._on_filter_changed(k))
        self._chip_group.addButton(btn)
        self._filter_buttons[key] = btn
        row.addWidget(btn)

    def _make_entity_card(self, entity) -> EntityCard:
        """Crée et câble une EntityCard pour une entité du catalogue."""
        ec = self._coverage.get(entity.id)
        card = EntityCard(entity.id, entity.label, entity.description)
        cand_names = list(ec.candidate_models if ec else ())
        candidates = [(name, self._models[name].display_name) for name in cand_names]
        has_cluster = any(
            self._models[name].cluster_options.get(entity.id) for name in cand_names
        )
        is_derived = any(
            entity.id in self._models[name].derived_entities for name in cand_names
        )
        card.set_candidates(candidates, has_cluster=has_cluster, is_derived=is_derived)
        card.toggled.connect(self._on_entity_toggled)
        card.model_changed.connect(self._on_model_changed)
        card.cluster_toggled.connect(self._on_cluster_toggled)
        card.activate_rvt.connect(self.activate_rvt)
        card.thresholds_changed.connect(self._on_thresholds_changed)
        card.cluster_params_changed.connect(self._on_cluster_params_changed)
        self._cards[entity.id] = card
        return card

    def _on_filter_changed(self, key: str) -> None:
        # Filtre d'AFFICHAGE uniquement : ne touche jamais self._selected ni runs.
        self._filter = key
        self._apply_filter()

    def _apply_filter(self) -> None:
        for key, (header, container) in self._section_widgets.items():
            visible = self._filter == "all" or self._filter == key
            header.setVisible(visible)
            container.setVisible(visible)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_enable_toggled(self, checked: bool) -> None:
        self._enabled = checked
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_entity_toggled(self, entity_id: str, selected: bool) -> None:
        self._selected[entity_id] = selected
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_model_changed(self, entity_id: str, model_name: str) -> None:
        self._overrides[entity_id] = model_name
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

    def _on_advanced_toggled(self, on: bool) -> None:
        self._advanced = on
        for card in self._cards.values():
            card.set_advanced(on)
        self._refresh()

    def _on_thresholds_changed(self, entity_id: str, confidence: float, min_area: float) -> None:
        self._entity_thresholds[entity_id] = {
            "confidence_threshold": float(confidence),
            "min_area_m2": float(min_area),
        }
        if not self._loading:
            self.changed.emit()

    def _on_cluster_params_changed(self, entity_id: str, params: dict) -> None:
        self._entity_cluster_params[entity_id] = dict(params or {})
        if not self._loading:
            self.changed.emit()

    # ------------------------------------------------------------------
    # Rafraîchissement
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        self._enable_check.setChecked(self._enabled)
        self._body_stack.setCurrentIndex(1 if self._enabled else 0)
        if not self._enabled:
            return

        for eid, card in self._cards.items():
            ec = self._coverage.get(eid)
            model_name = self._overrides.get(eid) or (ec.default_model if ec else None)
            model = self._models.get(model_name) if model_name else None
            rvt = model.target_rvt if model else "—"
            cluster_outputs = model.cluster_options.get(eid, ()) if model else ()
            is_derived = bool(model) and eid in model.derived_entities
            ov = self._entity_thresholds.get(eid, {})
            # Défauts des paramètres de regroupement (DBSCAN) pour cette entité :
            # ceux de la 1ʳᵉ sortie de clustering qu'elle produit (dérivée ou cluster).
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
            card.update_state(
                selected=bool(self._selected.get(eid)),
                current_model=model_name,
                rvt=rvt,
                rvt_active=(rvt in self._active_rvts) if model else True,
                cluster_outputs=cluster_outputs,
                cluster_on=eid in self._cluster,
                default_confidence=model.default_confidence if model else 0.3,
                default_min_area=model.default_min_area if model else 0.0,
                conf_override=ov.get("confidence_threshold"),
                area_override=ov.get("min_area_m2"),
                is_derived=is_derived,
                cluster_default_params=cluster_default_params,
                cluster_params_override=self._entity_cluster_params.get(eid),
            )
        self._update_selection_count()
        self._rebuild_runs()
        self._apply_filter()

    def _update_selection_count(self) -> None:
        if self._sel_count is None:
            return
        total = sum(
            1 for e in self._catalog
            if (self._coverage.get(e.id) and self._coverage[e.id].default_model)
        )
        n = sum(1 for on in self._selected.values() if on)
        self._sel_count.setText(f"{n} sur {total} sélectionnée{'s' if n != 1 else ''}")

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

        selected_ids = [e for e, on in self._selected.items() if on]
        runs = resolve_runs_from_entities(
            selected_ids, self._overrides, self._installed, self._catalog, self._cluster,
            entity_thresholds=self._entity_thresholds,
            entity_cluster_params=self._entity_cluster_params,
        )
        self._runs_count.setText(f"{len(runs)} run{'s' if len(runs) > 1 else ''} · 1 modèle = 1 indice")
        if not runs:
            row = QLabel("Aucun run — sélectionnez au moins une entité couverte par un modèle.")
            row.setObjectName("EntityDesc")
            self._runs_layout.insertWidget(0, row)  # au-dessus du stretch final
            self._run_rows.append(row)
            return
        for idx, run in enumerate(runs):
            model = self._models.get(run["model"])
            disp = model.display_name if model else run["model"]
            row = QFrame()
            row.setObjectName("RunRow")
            h = QHBoxLayout(row)
            h.setContentsMargins(10, 6, 10, 6)
            h.setSpacing(8)
            name = QLabel(disp)
            name.setObjectName("RunName")
            classes = QLabel(", ".join(run.get("selected_classes") or []))
            classes.setObjectName("EntityDesc")
            rvt_tag = QLabel(f"🔗 {run['target_rvt']}")
            rvt_tag.setObjectName("RunRvtTag")
            h.addWidget(name)
            h.addWidget(classes, 1)
            h.addWidget(rvt_tag)
            self._runs_layout.insertWidget(idx, row)  # au-dessus du stretch final
            self._run_rows.append(row)

    # ------------------------------------------------------------------
    # API page
    # ------------------------------------------------------------------
    def set_active_rvts(self, rvt_keys) -> None:
        self._active_rvts = set(rvt_keys or [])
        self._refresh()

    def summary(self) -> str:
        if not self._enabled:
            return "désactivée"
        selected_ids = [e for e, on in self._selected.items() if on]
        if not selected_ids:
            if self._legacy_runs:
                n = len(self._legacy_runs)
                return f"{n} run{'s' if n > 1 else ''} (config importée)"
            return "aucune entité"
        runs = resolve_runs_from_entities(
            selected_ids, self._overrides, self._installed, self._catalog, self._cluster
        )
        n_ent = len(selected_ids)
        n_mod = len({r["model"] for r in runs})
        return f"{n_ent} entité{'s' if n_ent > 1 else ''} · {n_mod} modèle{'s' if n_mod > 1 else ''}"

    def recap_entities(self) -> list:
        """Libellés des entités cochées, dans l'ordre du catalogue (récap étape 4)."""
        if not self._enabled:
            return []
        return [e.label for e in self._catalog if self._selected.get(e.id)]

    def recap_runs(self) -> list:
        """Une chaîne par run résolu : « <modèle> sur <RVT> » (récap étape 4).

        Couvre aussi le cas legacy (runs préservés sans entité sélectionnée).
        """
        if not self._enabled:
            return []
        selected_ids = [e for e, on in self._selected.items() if on]
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
        """Nombre de modèles distincts impliqués (sous-libellé timeline)."""
        selected_ids = [e for e, on in self._selected.items() if on]
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
            # Rétrocompat : config legacy avec des « runs » explicites mais aucune
            # entité → on préserve ces runs (jamais d'écrasement silencieux) tant
            # que l'utilisateur ne sélectionne pas d'entité (cf. collect_into).
            legacy = cv.get("runs") or []
            self._legacy_runs = list(legacy) if (legacy and not self._selected) else None
            self._refresh()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        cv = config.setdefault("computer_vision", {})
        cv["enabled"] = self._enabled
        selected_ids = [e for e, on in self._selected.items() if on]
        if selected_ids:
            self._legacy_runs = None  # l'utilisateur pilote par entités désormais
        cv["selected_entities"] = selected_ids
        cv["entity_model_overrides"] = {
            e: m for e, m in self._overrides.items() if self._selected.get(e)
        }
        cv["entity_cluster_enabled"] = sorted(e for e in self._cluster if self._selected.get(e))
        cv["entity_thresholds"] = {
            e: t for e, t in self._entity_thresholds.items() if self._selected.get(e)
        }
        cv["entity_cluster_params"] = {
            e: p for e, p in self._entity_cluster_params.items() if self._selected.get(e)
        }
        cv["generate_annotated_images"] = self._annot_check.isChecked()
        cv["generate_shapefiles"] = True  # toujours : le GeoPackage des détections est produit
        # Runs pour le pipeline (l'aval ne consomme que 'runs') : résolus depuis
        # les entités, ou — config legacy sans entités — les runs préservés tels quels.
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
