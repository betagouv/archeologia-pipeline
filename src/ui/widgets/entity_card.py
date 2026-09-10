"""EntityCard — carte d'une entité à détecter (étape 3).

Affiche : coche + libellé + tag RVT (orange si l'indice n'est pas activé à
l'étape 2) + description. Quand l'entité est cochée et couverte par un modèle :
sélecteur de modèle (si plusieurs candidats) + option « regrouper en zones »
(clustering) si le modèle en propose. Si aucun modèle ne couvre l'entité :
carte grisée + avertissement.

La carte est un pur widget de présentation : elle émet des signaux et reçoit
son état via :meth:`update_state` ; toute la logique vit dans la page.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from .no_wheel import NoWheelDoubleSpinBox


class _StayOpenMenu(QMenu):
    """QMenu qui reste ouvert quand on (dé)coche une action checkable.

    Permet de cocher/décocher plusieurs modèles en UNE ouverture (comparaison
    A/B) ; fermeture normale par clic hors menu ou Échap. Compatible Qt5/Qt6
    (``event.position()`` PyQt6, repli ``event.pos()`` PyQt5).
    """

    def mouseReleaseEvent(self, event):  # noqa: N802 (signature Qt)
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        act = self.actionAt(pos)
        if act is not None and act.isCheckable() and act.isEnabled():
            act.setChecked(not act.isChecked())
            return  # ne pas fermer le menu
        super().mouseReleaseEvent(event)


# Paramètres des briques de synthèse éditables, avec explications (tooltips).
# Seuls ceux présents dans les défauts du modèle (règle args.yaml) sont
# affichés : les clés DBSCAN et enclosure ne se croisent donc jamais.
# (clé, libellé, min, max, pas, décimales, est_entier, tooltip)
_CLUSTER_PARAM_SPECS = (
    # — Regroupement (DBSCAN) —
    ("eps_m", "Distance max (m)", 0.0, 1000.0, 5.0, 0, True,
     "Deux détections plus proches que cette distance sont regroupées."),
    ("min_cluster_size", "Nb min de détections", 1.0, 10000.0, 1.0, 0, True,
     "Taille minimale d'un groupe pour créer une zone."),
    ("min_confidence", "Confiance min", 0.0, 1.0, 0.05, 2, False,
     "Ignore les détections sous ce seuil POUR le regroupement (≠ seuil de détection)."),
    ("min_area_m2", "Aire min zone (m²)", 0.0, 1_000_000.0, 50.0, 0, True,
     "Supprime les zones de surface inférieure."),
    ("buffer_m", "Tampon (m)", 0.0, 200.0, 1.0, 0, True,
     "Dilate l'enveloppe de chaque groupe."),
    ("min_samples", "Densité (avancé)", 1.0, 100.0, 1.0, 0, True,
     "min_samples DBSCAN : nb de voisins pour qu'un point soit « cœur » de cluster."),
    # — Enclos (fermeture vectorielle) —
    ("gap_tolerance_m", "Pontage des interruptions (m)", 0.5, 50.0, 0.5, 1, False,
     "Ponte les interruptions du tracé jusqu'à cette largeur. ⚠ Un enclos plus "
     "étroit que cette valeur est rempli par la fermeture (taille min détectable ≈ T)."),
    ("max_area_m2", "Surface max (m²)", 1.0, 1_000_000.0, 500.0, 0, True,
     "Écarte les surfaces encloses plus grandes (limite les mailles de parcellaire)."),
    ("min_closure", "Fermeture min (0–1)", 0.0, 1.0, 0.05, 2, False,
     "Part minimale du contour couverte par de vraies détections (0,6 ≈ 3 côtés sur 4)."),
    ("max_elongation", "Élongation max", 1.0, 20.0, 0.5, 1, False,
     "Rapport longueur/largeur maximal (écarte couloirs et lanières)."),
    ("min_ancrage", "Ancrage des sources min (0–1)", 0.0, 1.0, 0.05, 2, False,
     "Part de l'aire des fragments contributeurs qui reste au voisinage du "
     "contour. Bas = cour incidente entre des lanières qui continuent au loin "
     "(faux positif de parcellaire) ; un vrai enclos est proche de 1."),
    ("max_isolement", "Isolement max (0–1)", 0.0, 1.0, 0.05, 2, False,
     "Part max du contour partagée avec d'autres candidats — écarte les "
     "mailles de trame parcellaire (un enclos isolé ≈ 0)."),
    ("min_rectangularite", "Rectangularité min (0–1)", 0.0, 1.0, 0.05, 2, False,
     "Régularité min de la forme (0 = tout accepter ; un cercle vaut ~0,79)."),
    # — Axes linéaires (bandes directionnelles) —
    ("band_width_m", "Largeur max de la bande (m)", 5.0, 200.0, 5.0, 0, True,
     "Étalement latéral maximal des brins parallèles d'un même axe "
     "(fossés bordiers + agger + tronçons décalés)."),
    ("angle_tolerance_deg", "Tolérance d'orientation (°)", 5.0, 45.0, 1.0, 0, True,
     "Écart d'azimut maximal entre un fragment et la direction de l'axe."),
    ("min_length_m", "Longueur min (m)", 50.0, 50_000.0, 50.0, 0, True,
     "Longueur minimale d'un axe publié — la rectitude kilométrique est la "
     "signature des voies anciennes."),
    ("max_gap_m", "Interruption max (m)", 10.0, 5000.0, 10.0, 0, True,
     "Au-delà de ce trou le long de l'axe, l'enfilade est coupée en deux axes."),
    ("min_coverage", "Couverture min (0–1)", 0.0, 1.0, 0.05, 2, False,
     "Part minimale de l'axe réellement couverte par des détections."),
    ("min_sources", "Nb min de fragments", 2.0, 1000.0, 1.0, 0, True,
     "Nombre minimal de détections constitutives d'un axe."),
)


_THUMB = 44  # côté de la vignette de carte, en px logiques


class EntityCard(QFrame):
    toggled = pyqtSignal(str, bool)        # entity_id, selected
    models_changed = pyqtSignal(str, list)  # entity_id, [model_name, ...] (≥2 = comparaison A/B)
    cluster_toggled = pyqtSignal(str, bool)
    activate_rvt = pyqtSignal(str)         # rvt key
    thresholds_changed = pyqtSignal(str, float, float)  # entity_id, confiance, aire min
    cluster_params_changed = pyqtSignal(str, dict)  # entity_id, {param: valeur}
    fiche_requested = pyqtSignal(str)      # entity_id — ouvrir la fiche de structure

    def __init__(self, entity_id: str, label: str, description: str, parent=None):
        super().__init__(parent)
        self._id = entity_id
        self._has_model = False
        self._selected = False
        self._loading = False
        self._advanced = False
        self._candidates: Dict[str, str] = {}  # name -> display_name
        self._current_models: list = []  # modèles effectifs (1..n ; ≥2 = comparaison)
        self._active_cluster_keys: set = set()  # params de cluster effectivement édités
        self.setObjectName("EntityCard")
        self.setProperty("state", "off")

        # Deux colonnes : la vignette de la structure à gauche, tout le reste à
        # droite. La colonne de gauche a une largeur FIXE, occupée même sans
        # vignette (cadre d'attente) : les libellés restent alignés d'une carte
        # à l'autre, et le gabarit de hauteur verrouillé plus bas n'est pas
        # touché (le contenu réservé dépasse toujours 44 px).
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 7, 10, 7)
        outer.setSpacing(9)

        self._thumb = QPushButton("")
        self._thumb.setObjectName("EntityThumb")
        self._thumb.setFixedSize(_THUMB, _THUMB)
        self._thumb.setFlat(True)
        self._thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thumb.setToolTip("Voir la fiche de la structure")
        self._thumb.clicked.connect(lambda: self.fiche_requested.emit(self._id))
        outer.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        outer.addLayout(layout, 1)

        header = QHBoxLayout()
        header.setSpacing(7)
        self._check = QLabel("")
        self._check.setObjectName("EntityCheck")
        self._check.setFixedSize(15, 15)
        self._check.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel(label)
        self._label.setObjectName("EntityLabel")
        self._fiche_btn = QPushButton("Fiche")
        self._fiche_btn.setObjectName("EntityFicheBtn")
        self._fiche_btn.setFlat(True)
        self._fiche_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fiche_btn.setToolTip(
            "Illustration, provenance des données d'entraînement, hors-cible "
            "et contexte d'utilisation"
        )
        self._fiche_btn.clicked.connect(lambda: self.fiche_requested.emit(self._id))
        self._rvt_tag = QLabel("")
        self._rvt_tag.setObjectName("EntityRvtTag")
        header.addWidget(self._check)
        header.addWidget(self._label)
        header.addStretch(1)
        header.addWidget(self._fiche_btn)
        header.addWidget(self._rvt_tag)
        layout.addLayout(header)

        self._desc = QLabel(description)
        self._desc.setObjectName("EntityDesc")
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        self._nomodel = QLabel("⚠ Aucun modèle disponible — importez un modèle ONNX")
        self._nomodel.setObjectName("EntityNoModel")
        self._nomodel.setWordWrap(True)
        self._nomodel.setVisible(False)
        layout.addWidget(self._nomodel)

        # Ligne RVT manquant : « Indice X non activé » + bouton + Activer
        self._rvt_row = _Row()
        self._rvt_warn = QLabel("")
        self._rvt_warn.setObjectName("EntityRvtWarn")
        self._rvt_warn.setWordWrap(True)
        self._rvt_btn = QPushButton("+ Activer")
        self._rvt_btn.setObjectName("EntityActivateBtn")
        self._rvt_btn.clicked.connect(lambda: self.activate_rvt.emit(self._rvt_key))
        self._rvt_row.addWidget(self._rvt_warn, 1)
        self._rvt_row.addWidget(self._rvt_btn)
        self._rvt_row.setVisible(False)
        self._rvt_key = ""
        layout.addWidget(self._rvt_row)

        # Ligne modèle : « <nom>   Changer ▾ » (discret) / « seul disponible ».
        # Pas de libellé « Modèle » séparé : le display_name le porte déjà
        # (préfixé « Modèle … » dans chaque model_card.yaml) → évite le doublon.
        self._model_row = _Row()
        self._model_name = QLabel("")
        self._model_name.setObjectName("EntityModelName")
        self._change_btn = QPushButton("Changer ▾")
        self._change_btn.setObjectName("EntityChangeBtn")
        self._change_btn.setFlat(True)
        self._change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._change_btn.clicked.connect(self._on_change_clicked)
        self._single_hint = QLabel("seul disponible")
        self._single_hint.setObjectName("EntityModelSingle")
        self._model_row.addWidget(self._model_name, 1)
        self._model_row.addWidget(self._change_btn)
        self._model_row.addWidget(self._single_hint)
        self._model_row.setVisible(False)
        layout.addWidget(self._model_row)

        # Option clustering
        self._cluster_check = QCheckBox("")
        self._cluster_check.setObjectName("EntityCluster")
        self._cluster_check.toggled.connect(self._on_cluster_toggled)
        self._cluster_check.setVisible(False)
        layout.addWidget(self._cluster_check)

        # Badge cible dérivée : le regroupement est intrinsèque à l'entité
        # (remplace la case cluster, jamais affichés ensemble).
        self._derived_badge = QLabel("↳ regroupement automatique en zones")
        self._derived_badge.setObjectName("EntityDerivedBadge")
        self._derived_badge.setWordWrap(True)
        self._derived_badge.setVisible(False)
        layout.addWidget(self._derived_badge)

        # Réglages avancés (mode avancé) : confiance + aire min surchargeables.
        self._adv_row = _Row()
        conf_lbl = QLabel("Confiance")
        conf_lbl.setObjectName("EntityModelLabel")
        self._conf_spin = NoWheelDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setFixedWidth(58)
        self._conf_spin.valueChanged.connect(self._on_thresholds_changed)
        area_lbl = QLabel("Aire min m²")
        area_lbl.setObjectName("EntityModelLabel")
        self._area_spin = NoWheelDoubleSpinBox()
        self._area_spin.setRange(0.0, 1_000_000.0)
        self._area_spin.setSingleStep(50.0)
        self._area_spin.setDecimals(0)
        self._area_spin.setFixedWidth(82)
        self._area_spin.valueChanged.connect(self._on_thresholds_changed)
        self._adv_row.addWidget(conf_lbl)
        self._adv_row.addWidget(self._conf_spin)
        self._adv_row.addWidget(area_lbl)
        self._adv_row.addWidget(self._area_spin)
        self._adv_row.setVisible(False)
        layout.addWidget(self._adv_row)
        # Aide fiabilité (mode avancé) : les coupures effectives des catégories
        # douteux/possible/probable/très probable au seuil courant, par classe.
        self._fiab_hint = QLabel("")
        self._fiab_hint.setObjectName("EntityFiabHint")
        self._fiab_hint.setWordWrap(True)
        self._fiab_hint.setVisible(False)
        layout.addWidget(self._fiab_hint)

        # Paramètres du regroupement (DBSCAN) : éditables en mode avancé pour une
        # entité dérivée / clusterisée. Place NON réservée (apparaît seulement pour
        # ces entités-là) → ne bloate pas les cartes simples.
        self._cluster_params_box = QFrame()
        self._cluster_params_box.setObjectName("EntityClusterParams")
        cpv = QVBoxLayout(self._cluster_params_box)
        cpv.setContentsMargins(0, 2, 0, 0)
        cpv.setSpacing(2)
        cp_title = QLabel("Paramètres du regroupement")
        cp_title.setObjectName("EntityModelLabel")
        cpv.addWidget(cp_title)
        self._cluster_spins: Dict[str, NoWheelDoubleSpinBox] = {}
        self._cluster_int_keys = set()
        for key, lbl, mn, mx, step, dec, is_int, tip in _CLUSTER_PARAM_SPECS:
            r = _Row()
            wl = QLabel(lbl)
            wl.setObjectName("EntityModelLabel")
            sp = NoWheelDoubleSpinBox()
            sp.setRange(mn, mx)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setFixedWidth(92)
            sp.setToolTip(tip)
            wl.setToolTip(tip)
            sp.valueChanged.connect(self._on_cluster_params_changed)
            r.addWidget(wl, 1)
            r.addWidget(sp)
            cpv.addWidget(r)
            self._cluster_spins[key] = sp
            if is_int:
                self._cluster_int_keys.add(key)
        self._cluster_params_box.setVisible(False)
        layout.addWidget(self._cluster_params_box)

        # Pas de hauteur fixe : la carte épouse son contenu MAXIMAL. Les lignes
        # « réglages » (modèle, cluster, avancé) réservent en permanence leur
        # place (retainSizeWhenHidden, cf. set_candidates) → cocher/décocher ou
        # (dés)activer les réglages avancés ne change JAMAIS la hauteur. Une
        # carte sans clustering est donc naturellement plus courte qu'une carte
        # qui en propose : chaque carte est « au plus juste » de son gabarit.

    # ------------------------------------------------------------------
    def set_fiche(
        self,
        vignette_path: Optional[str],
        *,
        disponible: bool = True,
        cadrage: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Alimente la vignette et l'accès à la fiche de structure.

        ``vignette_path`` absent ou illisible → cadre d'attente (la classe
        n'a pas encore de bloc ``fiche.vignettes`` dans son ``model_card``).
        ``disponible=False`` (aucun modèle n'installe cette entité) → l'accès
        à la fiche disparaît : il n'y aurait rien à montrer.

        ``cadrage`` = ``(x, y, côté)`` en fractions de l'image : la vignette
        couvre 324 m de terrain, réduite à 44 px elle serait illisible, donc
        l'icône n'en montre que la fenêtre qui porte la structure. Absent →
        image entière (comportement d'origine).
        """
        self._fiche_btn.setVisible(disponible)
        self._thumb.setEnabled(disponible)

        pix = QPixmap(vignette_path) if vignette_path else QPixmap()
        if not pix.isNull() and cadrage:
            pix = self._decouper(pix, cadrage)
        if pix.isNull():
            self._thumb.setIcon(QIcon())
            self._thumb.setText("◌" if disponible else "")
            self._thumb.setProperty("state", "vide")
        else:
            marge = 2  # laisse respirer le liseré du cadre
            self._thumb.setText("")
            self._thumb.setIcon(QIcon(pix))
            self._thumb.setIconSize(QSize(_THUMB - marge * 2, _THUMB - marge * 2))
            self._thumb.setProperty("state", "plein")
        self._thumb.style().unpolish(self._thumb)
        self._thumb.style().polish(self._thumb)

    @staticmethod
    def _decouper(pix: QPixmap, cadrage: Tuple[float, float, float]) -> QPixmap:
        """Découpe la fenêtre ``(x, y, côté)`` fractionnaire du pixmap.

        Le cadrage est déjà borné à l'image par ``app.services.class_fiche`` ;
        on reborne quand même ici (un pixmap non carré donnerait un rectangle
        hors limites) et on rend l'image entière si le découpage est vide.
        """
        x, y, cote = cadrage
        c = max(1, int(round(cote * pix.width())))
        left = min(max(0, int(round(x * pix.width()))), max(0, pix.width() - c))
        top = min(max(0, int(round(y * pix.height()))), max(0, pix.height() - c))
        decoupe = pix.copy(left, top, c, c)
        return pix if decoupe.isNull() else decoupe

    # ------------------------------------------------------------------
    def set_candidates(
        self,
        candidates: Sequence[Tuple[str, str]],
        *,
        has_cluster: bool = False,
        is_derived: bool = False,
    ) -> None:
        """``candidates`` = [(model_name, display_name)…] ; couvre l'entité.

        ``has_cluster`` : au moins un modèle candidat propose un regroupement
        pour cette entité → la ligne clustering réserve sa place en permanence.
        ``is_derived`` : l'entité est une *cible dérivée* (sortie de clustering
        présentée comme entité) → badge « regroupement automatique » à la place
        de la case cluster ; sa place est réservée en permanence.
        """
        self._candidates = {name: disp for name, disp in candidates}
        self._has_model = bool(candidates)
        # Affordance : toute la carte est cliquable quand un modèle la couvre
        # (cf. mousePressEvent) — le curseur doit le dire, comme à l'étape 2.
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if self._has_model
            else Qt.CursorShape.ArrowCursor
        )
        self._configure_reservations(has_cluster, is_derived)

    def _configure_reservations(self, has_cluster: bool, is_derived: bool = False) -> None:
        """Verrouille le gabarit : les lignes du contenu MAXIMAL réservent leur
        place même cachées → la hauteur ne bouge plus jamais (cf. __init__)."""
        if self._has_model:
            self._retain(self._model_row, True)
            self._retain(self._adv_row, True)
            self._retain(self._cluster_check, has_cluster)
            self._retain(self._derived_badge, is_derived)
            self._retain(self._nomodel, False)
        else:
            self._retain(self._nomodel, True)
            for w in (self._model_row, self._adv_row, self._cluster_check, self._derived_badge):
                self._retain(w, False)

    @staticmethod
    def _retain(w, on: bool) -> None:
        sp = w.sizePolicy()
        sp.setRetainSizeWhenHidden(on)
        w.setSizePolicy(sp)

    def update_state(
        self,
        *,
        selected: bool,
        current_models: Sequence[str],
        rvt: str,
        rvt_active: bool,
        cluster_outputs: Sequence[str],
        cluster_on: bool,
        default_confidence: float = 0.2,
        default_min_area: float = 0.0,
        conf_override: Optional[float] = None,
        area_override: Optional[float] = None,
        is_derived: bool = False,
        cluster_default_params: Optional[Dict[str, float]] = None,
        cluster_params_override: Optional[Dict[str, float]] = None,
        missing_rvt: Optional[str] = None,
        fiabilite_hint: str = "",
    ) -> None:
        self._selected = selected
        # ``rvt`` est un AFFICHAGE (peut joindre plusieurs indices en
        # comparaison, ex. « LD + SVF ») ; la clé d'activation du bouton
        # « + Activer » est le premier indice MANQUANT, pas le libellé.
        self._rvt_key = missing_rvt or rvt
        self._check.setText("✓" if selected else "")

        # Tag RVT (orange si l'indice n'est pas activé à l'étape 2)
        self._rvt_tag.setText(f"⚠ {rvt}" if (selected and self._has_model and not rvt_active) else rvt)
        self._rvt_tag.setProperty("missing", selected and self._has_model and not rvt_active)

        # État global de la carte (couleur)
        if not self._has_model:
            state = "nomodel"
        elif selected and not rvt_active:
            state = "warn"
        elif selected:
            state = "on"
        else:
            state = "off"
        self.setProperty("state", state)

        self._nomodel.setVisible(not self._has_model)

        # Ligne RVT manquant
        show_rvt_warn = selected and self._has_model and not rvt_active
        self._rvt_row.setVisible(show_rvt_warn)
        if show_rvt_warn:
            self._rvt_warn.setText(
                f"Indice {missing_rvt or rvt} non activé à l'étape 2"
            )

        # Ligne modèle : nom du modèle + « Changer ▾ » discret (si plusieurs).
        # Plusieurs modèles cochés (comparaison A/B) → « N modèles (comparaison) »
        # avec la liste complète en tooltip.
        show_model = selected and self._has_model
        self._model_row.setVisible(show_model)
        if show_model:
            self._current_models = list(current_models or [])
            disps = [self._candidates.get(n, n) for n in self._current_models]
            if len(disps) > 1:
                disp = " + ".join(disps)
                short = f"{len(disps)} modèles (comparaison)"
            else:
                disp = disps[0] if disps else ""
                short = disp if len(disp) <= 28 else disp[:27] + "…"
            self._model_name.setText(short)
            self._model_name.setToolTip(disp)
            multi = len(self._candidates) > 1
            self._change_btn.setVisible(multi)
            self._single_hint.setVisible(not multi)

        # Option clustering
        show_cluster = bool(selected and self._has_model and cluster_outputs)
        self._cluster_check.setVisible(show_cluster)
        if show_cluster:
            self._loading = True
            try:
                self._cluster_check.setText("Regrouper en clusters")
                self._cluster_check.setChecked(cluster_on)
            finally:
                self._loading = False

        # Badge cible dérivée (regroupement intrinsèque) — exclusif de la case
        # cluster : pour une cible dérivée, ``cluster_outputs`` est vide.
        self._derived_badge.setVisible(bool(selected and self._has_model and is_derived))

        # Réglages avancés (confiance + aire min) : visibles si mode avancé.
        show_adv = bool(selected and self._has_model and self._advanced)
        self._adv_row.setVisible(show_adv)
        if show_adv:
            self._loading = True
            try:
                self._conf_spin.setValue(
                    float(conf_override if conf_override is not None else default_confidence)
                )
                self._area_spin.setValue(
                    float(area_override if area_override is not None else default_min_area)
                )
                # Comparaison A/B : la valeur affichée est le défaut du modèle
                # primaire, mais tant qu'elle n'est pas modifiée chaque run
                # applique le défaut de SON modèle — le tooltip le dit.
                self._conf_spin.setToolTip(
                    "Comparaison : sans modification, chaque modèle applique "
                    "son propre seuil par défaut ; modifier la valeur impose "
                    "le même seuil à toutes les variantes."
                    if len(self._current_models) > 1 else ""
                )
            finally:
                self._loading = False
        self._fiab_hint.setText(fiabilite_hint or "")
        self._fiab_hint.setVisible(bool(show_adv and fiabilite_hint))

        # Paramètres du regroupement (DBSCAN) — en mode avancé, pour une entité
        # dérivée ou clusterisée disposant de défauts. Pré-remplis (override sinon
        # défaut du modèle). Seuls les paramètres réellement utilisés par le modèle
        # (présents dans les défauts) sont affichés et émis.
        defaults = cluster_default_params or {}
        override = cluster_params_override or {}
        show_cluster_params = bool(
            selected and self._has_model and self._advanced
            and (is_derived or cluster_on) and defaults
        )
        self._cluster_params_box.setVisible(show_cluster_params)
        if show_cluster_params:
            self._active_cluster_keys = set(defaults.keys())
            self._loading = True
            try:
                for key, spin in self._cluster_spins.items():
                    active = key in defaults
                    # cacher la ligne (le parent _Row) d'un paramètre non utilisé
                    spin.parent().setVisible(active)
                    if active:
                        val = override.get(key, defaults.get(key))
                        try:
                            spin.setValue(float(val))
                        except (TypeError, ValueError):
                            pass
            finally:
                self._loading = False

        self._repolish()

    def set_advanced(self, on: bool) -> None:
        """Affiche/masque les champs avancés. Leur place est réservée en
        permanence (retainSizeWhenHidden) → la hauteur ne change pas."""
        self._advanced = bool(on)
        self._adv_row.setVisible(self._advanced and self._selected and self._has_model)

    def _on_thresholds_changed(self, *_args) -> None:
        if not self._loading:
            self.thresholds_changed.emit(
                self._id, float(self._conf_spin.value()), float(self._area_spin.value())
            )

    def _on_cluster_params_changed(self, *_args) -> None:
        if self._loading:
            return
        params: dict = {}
        for key in self._active_cluster_keys:
            val = self._cluster_spins[key].value()
            params[key] = int(round(val)) if key in self._cluster_int_keys else float(val)
        self.cluster_params_changed.emit(self._id, params)

    # ------------------------------------------------------------------
    def _on_change_clicked(self) -> None:
        # Cases NON exclusives : cocher un 2ᵉ modèle = mode comparaison A/B
        # (un run et des sorties par modèle). Le menu RESTE OUVERT pendant les
        # coches (remplacer A par B = cocher B puis décocher A en une seule
        # ouverture) et on émet UNE fois à la fermeture — aucun état transitoire
        # « comparaison » n'est persisté par l'autosave.
        menu = _StayOpenMenu(self)
        # Action désactivée plutôt que addSection : les styles à menus natifs
        # (macOS) rendent une section comme un simple séparateur sans texte.
        hint = menu.addAction("Cocher plusieurs modèles = comparaison")
        hint.setEnabled(False)
        menu.addSeparator()
        for name, disp in self._candidates.items():
            act = menu.addAction(disp)
            act.setData(name)
            act.setCheckable(True)
            act.setChecked(name in self._current_models)
        menu.exec(self._change_btn.mapToGlobal(self._change_btn.rect().bottomLeft()))
        checked = [a.data() for a in menu.actions() if a.isCheckable() and a.isChecked()]
        if not checked:
            return  # une entité garde au moins un modèle : sélection vide ignorée
        # Ordre stable : les modèles déjà sélectionnés gardent leur rang (le
        # « primaire » — RVT/défauts affichés — ne bascule pas quand on en
        # ajoute un), les nouveaux s'ajoutent à la suite.
        ordered = [n for n in self._current_models if n in checked]
        ordered += [n for n in checked if n not in ordered]
        if ordered != list(self._current_models):
            self.models_changed.emit(self._id, ordered)

    def _on_cluster_toggled(self, checked: bool) -> None:
        if not self._loading:
            self.cluster_toggled.emit(self._id, checked)

    def _repolish(self) -> None:
        # _check inclus : sinon le ✓ blanc reste sur fond blanc (invisible) car
        # son fond bleu vient du sélecteur descendant #EntityCard[state="on"].
        for w in (self, self._label, self._rvt_tag, self._desc, self._check):
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        # Bascule la sélection si l'entité a un modèle. Les widgets interactifs
        # (combo, checkbox cluster, bouton activer) consomment leurs propres
        # clics et ne déclenchent donc pas ce handler.
        if self._has_model:
            self.toggled.emit(self._id, not self._selected)
        super().mousePressEvent(event)


class _Row(QFrame):
    """Petite rangée horizontale (QFrame transparent + QHBoxLayout)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EntityRow")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def addWidget(self, w, stretch: int = 0):  # noqa: N802 (API Qt-like)
        self._layout.addWidget(w, stretch)
