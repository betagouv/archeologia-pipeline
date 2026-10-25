"""EntityRow — la ligne d'une structure à chercher (étape 3).

Une ligne par structure, rangée sous sa morphologie (interface rationalisée,
2026-10-24 ; remplace la carte à deux colonnes) : coche · vignette · libellé et
description élidée · image du relief lue par le modèle (orange, avec
« l'ajouter », si elle n'est pas cochée à l'étape 2) · modèle retenu
(« Changer ▾ » quand plusieurs le couvrent) et son « ⓘ » · le « ⓘ » de la
structure. Sous la ligne, quand elle est cochée : la case « Regrouper en
zones » ou le badge de regroupement, puis, si la case « Seuils et réglages par
structure » de la page est cochée : le seuil de confiance à deux décimales
(l'infobulle montre la valeur appliquée à trois), l'aire minimale, le « ↺ » de
cette seule structure, le profil des scores repliable et les paramètres du
regroupement.

La ligne est un pur widget de présentation : elle émet des signaux et reçoit
son état via :meth:`update_state` ; toute la logique vit dans la page.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QPoint, QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...app.services.vocabulaire import nombre
from .collapsible import CollapsibleSection
from .info_button import InfoButton
from .no_wheel import NoWheelDoubleSpinBox
from .profil_scores import ProfilScoresWidget
from .vignette import pixmap_ajuste


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
     "Seuil des détections qui ouvrent une zone (≠ seuil de détection). Si le modèle en prévoit, celles un peu "
     "en dessous s'y rattachent sans en ouvrir ; ce seuil de rattachement suit ce réglage."),
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


_THUMB = 28  # côté de la vignette de ligne, en px logiques


class _Ligne(QWidget):
    """Petite rangée horizontale (QWidget transparent + QHBoxLayout)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def addWidget(self, w, stretch: int = 0):  # noqa: N802 (API Qt-like)
        self._layout.addWidget(w, stretch)

    def addStretch(self, stretch: int = 1):  # noqa: N802 (API Qt-like)
        self._layout.addStretch(stretch)


class _Elide(QLabel):
    """QLabel élidé à droite à la largeur disponible, texte complet en infobulle.

    Un QLabel ordinaire fait de la largeur de son texte une largeur minimale :
    onze descriptions non élidées imposeraient la leur à la page, qui défilerait
    de côté. Ici la taille conseillée est fixe, le texte s'adapte au cadre.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._texte = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.set_texte(text)

    def set_texte(self, text: str) -> None:
        self._texte = text or ""
        self.setToolTip(self._texte)
        self._refaire()

    def _refaire(self) -> None:
        self.setText(self.fontMetrics().elidedText(
            self._texte, Qt.TextElideMode.ElideRight, max(0, self.width() - 2)
        ))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (signature Qt)
        return QSize(40, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802 (signature Qt)
        return QSize(160, super().sizeHint().height())

    def resizeEvent(self, event):  # noqa: N802 (signature Qt)
        super().resizeEvent(event)
        self._refaire()


class EntityRow(QFrame):
    toggled = pyqtSignal(str, bool)        # entity_id, selected
    models_changed = pyqtSignal(str, list)  # entity_id, [model_name, ...] (≥2 = comparaison A/B)
    cluster_toggled = pyqtSignal(str, bool)
    activate_rvt = pyqtSignal(str)         # rvt key
    thresholds_changed = pyqtSignal(str, dict)  # entity_id, {la seule clé modifiée: valeur} (C-01)
    cluster_params_changed = pyqtSignal(str, dict)  # entity_id, {param: valeur}
    fiche_requested = pyqtSignal(str)      # entity_id — ouvrir la fiche de structure
    modele_requested = pyqtSignal(str)     # nom du modèle primaire — ouvrir sa fiche
    reset_requested = pyqtSignal(str)      # entity_id — rétablir SES valeurs de modèle

    def __init__(self, entity_id: str, label: str, description: str, parent=None):
        super().__init__(parent)
        self._id = entity_id
        self._has_model = False
        self._selected = False
        self._loading = False
        self._advanced = False
        self._readonly = False
        self._candidates: Dict[str, str] = {}  # name -> display_name
        self._couts: Dict[str, Tuple[str, str]] = {}
        self._current_models: list = []  # modèles effectifs (1..n ; ≥2 = comparaison)
        self._active_cluster_keys: set = set()  # params de cluster effectivement édités
        self._implique = False                  # incluse par une dérivée : seuils ailleurs
        self._profils: list = []                # [figure mini, …]
        self._profils_cle = None
        self._rvt_key = ""
        self._seuil_effectif: Optional[float] = None
        self._seuil_regle = False
        self.setObjectName("EntityLine")
        self.setProperty("state", "off")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 5, 8, 5)
        root.setSpacing(4)

        # ── La ligne ──
        tete = QHBoxLayout()
        tete.setSpacing(8)
        self._check = QLabel("")
        self._check.setObjectName("EntityCheck")
        self._check.setFixedSize(15, 15)
        self._check.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Vignette inerte (demande utilisateur 2026-10-08) : un clic dessus coche ou
        # décoche la ligne ; seul le « ⓘ » ouvre la fiche.
        self._thumb = QLabel("")
        self._thumb.setObjectName("EntityThumb")
        self._thumb.setFixedSize(_THUMB, _THUMB)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._label = QLabel(label)
        self._label.setObjectName("EntityLabel")
        self._desc = _Elide(description)
        self._desc.setObjectName("EntityDesc")
        self._nomodel = QLabel("⚠ Aucun modèle installé")
        self._nomodel.setObjectName("EntityNoModel")
        self._nomodel.setToolTip(
            "Aucun modèle installé ne couvre cette structure : importez un modèle ONNX "
            "dans data/models"
        )
        self._nomodel.setVisible(False)
        self._rvt_tag = QLabel("")
        self._rvt_tag.setObjectName("EntityRvtTag")
        self._rvt_btn = QPushButton("l'ajouter")
        self._rvt_btn.setObjectName("EntityChangeBtn")
        self._rvt_btn.setFlat(True)
        self._rvt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rvt_btn.clicked.connect(lambda: self.activate_rvt.emit(self._rvt_key))
        self._rvt_btn.setVisible(False)
        # Pas de libellé « Modèle » séparé : le display_name le porte déjà
        # (préfixé « Modèle … » dans chaque model_card.yaml) → évite le doublon.
        self._model_name = QLabel("")
        self._model_name.setObjectName("EntityModelName")
        self._model_name.setVisible(False)
        self._change_btn = QPushButton("Changer ▾")
        self._change_btn.setObjectName("EntityChangeBtn")
        self._change_btn.setFlat(True)
        self._change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._change_btn.clicked.connect(self._on_change_clicked)
        self._change_btn.setVisible(False)
        self._model_info = InfoButton("Fiche du modèle : architecture, entraînement, évaluation")
        self._model_info.clicked.connect(self._demander_modele)
        self._model_info.setVisible(False)
        # Un seul patron de fiche : le « ⓘ » rond (interface rationalisée, 2026-10-24).
        self._fiche_btn = InfoButton(
            "Fiche de la structure : à quoi elle ressemble, où le modèle l'a apprise, "
            "ce qu'il ne détecte pas"
        )
        self._fiche_btn.clicked.connect(lambda: self.fiche_requested.emit(self._id))
        for w, stretch in (
            (self._check, 0), (self._thumb, 0), (self._label, 0), (self._desc, 1),
            (self._nomodel, 0), (self._rvt_tag, 0), (self._rvt_btn, 0), (self._model_name, 0),
            (self._change_btn, 0), (self._model_info, 0), (self._fiche_btn, 0),
        ):
            tete.addWidget(w, stretch, Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(tete)

        # ── Sous la ligne, dès qu'elle est cochée : regroupement ──
        self._options = _Ligne()
        self._cluster_check = QCheckBox("Regrouper en zones")
        self._cluster_check.setObjectName("EntityCluster")
        self._cluster_check.toggled.connect(self._on_cluster_toggled)
        self._cluster_check.setVisible(False)
        # Badge cible dérivée : le regroupement est intrinsèque à la structure
        # (remplace la case, jamais affichés ensemble) ; autre texte pour une
        # structure INCLUSE par une dérivée cochée.
        self._derived_badge = QLabel("")
        self._derived_badge.setObjectName("EntityDerivedBadge")
        self._derived_badge.setWordWrap(True)
        self._derived_badge.setVisible(False)
        self._options.addWidget(self._cluster_check)
        self._options.addWidget(self._derived_badge, 1)
        self._options.setVisible(False)
        root.addWidget(self._options)

        # ── Seuils et réglages (case de la page) : confiance + aire min surchargeables ──
        self._adv_row = _Ligne()
        conf_lbl = QLabel("Confiance")
        conf_lbl.setObjectName("EntityModelLabel")
        self._conf_lbl = conf_lbl  # relibellé « Confiance des cratères » sur une dérivée
        self._conf_spin = NoWheelDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        # Deux décimales (D-07, 2026-10-24) : c'est la précision d'un réglage à la main.
        # La valeur du modèle garde ses trois décimales (0,245) : le rafraîchissement
        # n'émet rien (garde ``_loading``) et l'infobulle la montre telle quelle.
        self._conf_spin.setDecimals(2)
        self._conf_spin.setFixedWidth(62)
        self._conf_spin.valueChanged.connect(self._on_conf_changed)
        area_lbl = QLabel("Aire min m²")
        area_lbl.setObjectName("EntityModelLabel")
        self._area_spin = NoWheelDoubleSpinBox()
        self._area_spin.setRange(0.0, 1_000_000.0)
        self._area_spin.setSingleStep(50.0)
        self._area_spin.setDecimals(0)
        self._area_spin.setFixedWidth(82)
        self._area_spin.valueChanged.connect(self._on_area_changed)
        # Réinitialisation de CETTE structure seulement (demande utilisateur
        # 2026-09-16). Désactivée avec toute la ligne quand la structure est incluse
        # par une dérivée : ses seuils sont ceux de la ligne parente.
        self._reinit_btn = QPushButton("↺")
        self._reinit_btn.setObjectName("EntityResetBtn")
        self._reinit_btn.setFlat(True)
        self._reinit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reinit_btn.setToolTip("Revenir au seuil du modèle")
        self._reinit_btn.clicked.connect(lambda: self.reset_requested.emit(self._id))
        self._adv_row.addWidget(conf_lbl)
        self._adv_row.addWidget(self._conf_spin)
        self._adv_row.addWidget(area_lbl)
        self._adv_row.addWidget(self._area_spin)
        self._adv_row.addWidget(self._reinit_btn)
        self._adv_row.addStretch(1)
        self._adv_row.setVisible(False)
        root.addWidget(self._adv_row)
        # Aide fiabilité : les coupures effectives des niveaux au seuil courant,
        # par classe — seulement quand le modèle ne livre pas son profil.
        self._fiab_hint = QLabel("")
        self._fiab_hint.setObjectName("EntityFiabHint")
        self._fiab_hint.setWordWrap(True)
        self._fiab_hint.setVisible(False)
        root.addWidget(self._fiab_hint)
        # Aide au choix du seuil (2026-10-08) : le profil des scores de chaque classe
        # de la structure, en mini, dont la ligne du seuil suit la case « Confiance ».
        # Replié par défaut (2026-10-24) : onze lignes tiennent sans défiler.
        self._profil_box = QWidget()
        pv = QVBoxLayout(self._profil_box)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(2)
        self._profil_section = CollapsibleSection("Profil des scores", invite="")
        self._profil_section.set_body(self._profil_box)
        self._profil_section.setVisible(False)
        root.addWidget(self._profil_section)

        # Paramètres du regroupement (DBSCAN) : éditables pour une structure dérivée
        # ou regroupée. Place NON réservée (apparaît seulement pour celles-là).
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
            r = _Ligne()
            wl = QLabel(lbl)
            wl.setObjectName("EntityModelLabel")
            wl.setWordWrap(True)
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
        root.addWidget(self._cluster_params_box)

    # ------------------------------------------------------------------
    def set_fiche(
        self,
        vignette_path: Optional[str],
        *,
        disponible: bool = True,
        cadrage: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Alimente la vignette et l'accès à la fiche de structure.

        ``vignette_path`` absent ou illisible → cadre d'attente (la classe n'a pas
        encore de bloc ``fiche.vignettes`` dans son ``model_card``). ``disponible=False``
        (aucun modèle n'installe cette structure) → le « ⓘ » disparaît : il n'y
        aurait rien à montrer. ``cadrage`` = ``(x, y, côté)`` en fractions de l'image :
        réduite à 28 px la vignette serait illisible, l'icône n'en montre que la
        fenêtre qui porte la structure. Absent → image entière.
        """
        self._fiche_btn.setVisible(disponible)
        cote = _THUMB - 4   # laisse respirer le liseré du cadre
        pix = pixmap_ajuste(
            vignette_path, cote, dpr=self.devicePixelRatioF(), cadrage=cadrage
        ) if vignette_path else QPixmap()
        if pix.isNull():
            self._thumb.setPixmap(QPixmap())
            self._thumb.setText("◌" if disponible else "")
            self._thumb.setProperty("state", "vide")
        else:
            self._thumb.setText("")
            self._thumb.setPixmap(pix)
            self._thumb.setProperty("state", "plein")
        self._thumb.style().unpolish(self._thumb)
        self._thumb.style().polish(self._thumb)

    # ------------------------------------------------------------------
    def set_candidates(
        self,
        candidates: Sequence[Tuple[str, str]],
        *,
        has_cluster: bool = False,
        is_derived: bool = False,
        implicable: bool = False,
        couts: Optional[Mapping[str, Tuple[str, str]]] = None,
    ) -> None:
        """``candidates`` = [(model_name, display_name)…] ; couvre la structure.

        ``couts`` : ``{model_name: (libellé de menu, infobulle)}`` — le coût
        structurel du modèle (fenêtres d'analyse par dalle, calculé par
        ``app.services.cout_modele``), affiché là où le choix se fait : entrées
        du menu « Changer ▾ » et infobulle du nom de modèle. Jamais une durée.
        ``has_cluster`` / ``is_derived`` / ``implicable`` ne réservent plus de
        place : la ligne grandit sous elle quand elle est cochée.
        """
        del has_cluster, is_derived, implicable   # la ligne épouse son contenu
        self._candidates = {name: disp for name, disp in candidates}
        self._couts = dict(couts or {})
        self._has_model = bool(candidates)
        self._maj_curseur()

    def _maj_curseur(self) -> None:
        # Affordance : toute la ligne est cliquable quand un modèle la couvre
        # (cf. mousePressEvent) — le curseur doit le dire, comme à l'étape 2.
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if (self._has_model and not self._readonly)
            else Qt.CursorShape.ArrowCursor
        )

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
        implique_par: str = "",
        conf_label: str = "Confiance",
        conf_tip: str = "",
        reinit_possible: bool = False,
    ) -> None:
        """``implique_par`` : libellé de la cible dérivée cochée qui INCLUT cette
        structure (2026-09-15) → cochée d'office, badge « inclus dans », seuils en
        lecture seule (ce sont ceux de la dérivée). ``conf_label`` / ``conf_tip`` :
        libellé et aide du seuil de confiance (une dérivée dit « Confiance des
        cratères » : son seuil est celui des détections sources)."""
        self._selected = selected
        # Inactif quand il n'y a rien à rétablir : proposer d'annuler un réglage
        # qui n'a jamais été fait ne dit rien à l'utilisateur.
        self._reinit_btn.setEnabled(bool(reinit_possible))   # la page y met déjà la lecture seule
        self._reinit_btn.setToolTip(
            f"Revenir au seuil du modèle ({nombre(float(default_confidence), 2)}) et à ses "
            "réglages d'aire et de regroupement ; les autres structures ne bougent pas"
        )
        self.setToolTip(
            f"Inclus dans « {implique_par} » — décochez cette ligne-là pour l'exclure"
            if implique_par else ""
        )
        # ``rvt`` est un AFFICHAGE (peut joindre plusieurs images en comparaison,
        # ex. « LD + SVF ») ; la clé du bouton « l'ajouter » est la première image
        # MANQUANTE, pas le libellé.
        self._rvt_key = missing_rvt or rvt
        self._check.setText("✓" if selected else "")

        # Image du relief lue par le modèle (orange si elle n'est pas cochée à l'étape 2)
        missing = bool(selected and self._has_model and not rvt_active)
        self._rvt_tag.setText(f"⚠ {rvt}" if missing else rvt)
        self._rvt_tag.setProperty("missing", missing)
        self._rvt_tag.setToolTip(
            f"Image {missing_rvt or rvt} non cochée à l'étape 2" if missing
            else f"Image du relief lue par le modèle : {rvt}"
        )
        self._rvt_btn.setVisible(missing)
        self._rvt_btn.setToolTip(f"Cocher l'image {missing_rvt or rvt} à l'étape 2")

        # État global de la ligne (couleur)
        if not self._has_model:
            state = "nomodel"
        elif missing:
            state = "warn"
        elif selected:
            state = "on"
        else:
            state = "off"
        self.setProperty("state", state)
        self._nomodel.setVisible(not self._has_model)

        # Modèle : nom + « Changer ▾ » discret (si plusieurs) + son « ⓘ ».
        # Plusieurs modèles cochés (comparaison A/B) → « N modèles (comparaison) »
        # avec la liste complète en infobulle.
        show_model = bool(selected and self._has_model)
        self._model_name.setVisible(show_model)
        self._model_info.setVisible(show_model)
        multi = len(self._candidates) > 1
        self._change_btn.setVisible(show_model and multi)
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
            # Infobulle : le coût structurel de chaque modèle courant (une
            # phrase par modèle) ; sans coût connu, la liste des noms complets.
            tips = [self._couts.get(n, ("", ""))[1] for n in self._current_models]
            tips = [t for t in tips if t] or [disp]
            if not multi:
                tips.append("Seul modèle installé pour cette structure.")
            self._model_name.setToolTip("\n\n".join(tips))

        # Case « Regrouper en zones » / badge de regroupement, dès que la ligne est cochée
        show_cluster = bool(selected and self._has_model and cluster_outputs)
        self._cluster_check.setVisible(show_cluster)
        if show_cluster:
            self._loading = True
            try:
                self._cluster_check.setChecked(cluster_on)
            finally:
                self._loading = False
        self._derived_badge.setText(
            f"↳ inclus dans « {implique_par} » : une seule couche, seuil réglé sur cette ligne-là"
            if implique_par else "↳ regroupement automatique en zones"
        )
        show_badge = bool(selected and self._has_model and (is_derived or implique_par))
        self._derived_badge.setVisible(show_badge)
        self._options.setVisible(show_cluster or show_badge)

        # Seuils et réglages : visibles si la case de la page est cochée ; en lecture
        # seule pour une structure incluse (ses seuils sont ceux de la dérivée).
        show_adv = bool(selected and self._has_model and self._advanced)
        self._adv_row.setVisible(show_adv)
        self._adv_row.setEnabled(not implique_par)
        self._conf_lbl.setText(conf_label)
        self._conf_lbl.setToolTip(conf_tip)
        self._seuil_effectif = float(conf_override if conf_override is not None else default_confidence)
        self._seuil_regle = conf_override is not None
        if show_adv:
            self._loading = True
            try:
                self._conf_spin.setValue(self._seuil_effectif)
                self._area_spin.setValue(
                    float(area_override if area_override is not None else default_min_area)
                )
            finally:
                self._loading = False
        self._conf_tip = conf_tip
        self._maj_infobulle_seuil()
        self._fiab_hint.setText(fiabilite_hint or "")
        self._fiab_hint.setVisible(bool(show_adv and fiabilite_hint))
        self._implique = bool(implique_par)
        self._maj_visibilite_profils()
        self._maj_profils()

        # Paramètres du regroupement (DBSCAN) — pour une structure dérivée ou
        # regroupée disposant de défauts. Pré-remplis (surcharge sinon défaut du
        # modèle). Seuls les paramètres réellement utilisés par le modèle
        # (présents dans les défauts) sont affichés et émis.
        defaults = cluster_default_params or {}
        override = cluster_params_override or {}
        show_cluster_params = bool(
            selected and self._has_model and self._advanced
            and (is_derived or cluster_on) and defaults
        )
        self._cluster_params_box.setVisible(show_cluster_params)
        if show_cluster_params:
            # Seules les clés qui ont un champ : min_confidence_extend (lu
            # d'args.yaml pour décaler l'extension) n'en a pas.
            self._active_cluster_keys = {k for k in defaults if k in self._cluster_spins}
            self._loading = True
            try:
                for key, spin in self._cluster_spins.items():
                    active = key in defaults
                    # cacher la ligne (le parent _Ligne) d'un paramètre non utilisé
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

    def _maj_infobulle_seuil(self) -> None:
        """L'infobulle de la case dit le seuil APPLIQUÉ, à trois décimales : la case
        n'en affiche que deux (0,245 s'y lit 0,24 ou 0,25) et, en comparaison A/B,
        chaque modèle garde le sien tant que la valeur n'est pas modifiée."""
        v = self._seuil_effectif
        lignes = []
        if v is not None:
            origine = "réglé" if self._seuil_regle else "seuil du modèle"
            lignes.append(f"Seuil appliqué : {nombre(v, 3)} ({origine})")
        if len(self._current_models) > 1:
            lignes.append(
                "Comparaison : sans modification, chaque modèle applique son propre "
                "seuil par défaut ; modifier la valeur impose le même seuil à toutes les variantes."
            )
        if getattr(self, "_conf_tip", ""):
            lignes.append(self._conf_tip)
        self._conf_spin.setToolTip("\n".join(lignes))

    def set_reinit_possible(self, on: bool) -> None:
        """Active le « ↺ » dès qu'un réglage est saisi : sans ça il restait grisé
        jusqu'au prochain rafraîchissement de la page (constat utilisateur 2026-10-08)."""
        self._reinit_btn.setEnabled(bool(on) and not self._readonly)

    def set_advanced(self, on: bool) -> None:
        """Affiche/masque les seuils et réglages de la ligne."""
        self._advanced = bool(on)
        self._adv_row.setVisible(self._advanced and self._selected and self._has_model)
        self._maj_visibilite_profils()

    def set_readonly(self, ro: bool) -> None:
        """Lecture seule pendant un traitement : plus rien ne se règle ni ne se coche,
        mais les deux « ⓘ » restent cliquables — on consulte, on ne modifie pas."""
        self._readonly = bool(ro)
        for w in (self._conf_spin, self._area_spin, self._cluster_check, self._change_btn,
                  self._rvt_btn, *self._cluster_spins.values()):
            w.setEnabled(not ro)
        if ro:
            self._reinit_btn.setEnabled(False)
        self._maj_curseur()

    def deplier_profil(self, on: bool = True) -> None:
        """Déplie (ou replie) le profil des scores — rendu hors écran et recette."""
        self._profil_section.set_expanded(on)

    # ------------------------------------------------------------------
    # Aide au choix du seuil : profils mini
    # ------------------------------------------------------------------
    def set_profils(self, profils: Sequence[tuple]) -> None:
        """``[(classe, Profil, couleur RGB ou None), …]`` — un mini-profil par
        classe de la structure dont l'évaluation est livrée ; vide = rien d'affiché.
        Les widgets ne sont reconstruits que si la liste change. Pas de texte sous
        la figure (demande utilisateur 2026-10-08) : les niveaux sont écrits sous
        l'axe, le bilan du seuil est dans l'infobulle."""
        cle = tuple((c, id(p), couleur) for c, p, couleur in profils)
        if cle == self._profils_cle:
            return
        self._profils_cle = cle
        lay = self._profil_box.layout()
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._profils = []
        for classe, profil, couleur in profils:
            if len(profils) > 1:
                titre = QLabel(classe)
                titre.setObjectName("EntityModelLabel")
                lay.addWidget(titre)
            fig = ProfilScoresWidget(profil, couleur, mini=True)
            lay.addWidget(fig)
            self._profils.append(fig)
        self._maj_profils()
        self._maj_visibilite_profils()

    def _maj_profils(self, *_args) -> None:
        """La ligne du seuil suit le seuil APPLIQUÉ (trois décimales, D-07), pas la valeur
        arrondie de la case ; le bilan est recalculé."""
        seuil = float(self._seuil_effectif if self._seuil_effectif is not None else self._conf_spin.value())
        for fig in getattr(self, "_profils", []):
            fig.set_seuil(seuil)   # la ligne suit la case ; le bilan est dans l'infobulle

    def _maj_visibilite_profils(self) -> None:
        section = getattr(self, "_profil_section", None)
        if section is not None:
            visible = bool(
                self._advanced and self._selected and self._has_model
                and self._profils and not self._implique
            )
            section.setVisible(visible)
            if visible:
                # La figure nomme les niveaux sous son axe : l'aide textuelle
                # « Fiabilité affichée — douteux dès 0,29 · … » ferait doublon.
                self._fiab_hint.setVisible(False)

    def _on_conf_changed(self, valeur: float) -> None:
        if self._loading:
            return
        self._seuil_effectif = float(valeur)
        self._seuil_regle = True
        self._maj_infobulle_seuil()
        self._maj_profils()
        self.thresholds_changed.emit(self._id, {"confidence_threshold": float(valeur)})

    def _on_area_changed(self, valeur: float) -> None:
        if not self._loading:
            self.thresholds_changed.emit(self._id, {"min_area_m2": float(valeur)})

    def _on_cluster_params_changed(self, *_args) -> None:
        if self._loading:
            return
        params: dict = {}
        for key in self._active_cluster_keys:
            val = self._cluster_spins[key].value()
            params[key] = int(round(val)) if key in self._cluster_int_keys else float(val)
        self.cluster_params_changed.emit(self._id, params)

    def _demander_modele(self) -> None:
        if self._current_models:
            self.modele_requested.emit(str(self._current_models[0]))

    # ------------------------------------------------------------------
    def _on_change_clicked(self) -> None:
        # Cases NON exclusives : cocher un 2ᵉ modèle = mode comparaison A/B
        # (une analyse et des sorties par modèle). Le menu RESTE OUVERT pendant les
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
            act = menu.addAction(self._couts.get(name, (disp, ""))[0] or disp)
            act.setData(name)
            act.setCheckable(True)
            act.setChecked(name in self._current_models)
        menu.exec(self._change_btn.mapToGlobal(self._change_btn.rect().bottomLeft()))
        checked = [a.data() for a in menu.actions() if a.isCheckable() and a.isChecked()]
        if not checked:
            return  # une structure garde au moins un modèle : sélection vide ignorée
        # Ordre stable : les modèles déjà sélectionnés gardent leur rang (le
        # « primaire » — image/défauts affichés — ne bascule pas quand on en
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
        # son fond bleu vient du sélecteur descendant #EntityLine[state="on"].
        for w in (self, self._label, self._rvt_tag, self._desc, self._check):
            w.style().unpolish(w)
            w.style().polish(w)

    def _bande_reglages(self) -> Tuple[int, int]:
        """``(haut, bas)``, en px de la ligne, de la bande des réglages visibles :
        case ou badge de regroupement, seuils, profil, paramètres du regroupement.
        Un clic dans cette bande règle, il ne coche pas (demande utilisateur
        2026-10-08) ; partout ailleurs sur la ligne, il coche ou décoche.
        ``(0, -1)`` : aucune bande."""
        ys = []
        for w in (self._options, self._adv_row, self._fiab_hint,
                  self._profil_section, self._cluster_params_box):
            if w.isVisible():
                haut = w.mapTo(self, QPoint(0, 0)).y()
                ys += [haut, haut + w.height()]
        return (min(ys), max(ys)) if ys else (0, -1)

    def _basculer(self) -> None:
        if self._has_model and not self._readonly:
            self.toggled.emit(self._id, not self._selected)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        # Bascule la sélection si la structure a un modèle ET que le clic tombe hors
        # de la bande des réglages. Les widgets interactifs (« ⓘ », Changer ▾,
        # l'ajouter) consomment leurs propres clics.
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        haut, bas = self._bande_reglages()
        if not (haut - 3 <= pos.y() <= bas + 3):
            self._basculer()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):  # noqa: N802 (signature Qt)
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Select):
            self._basculer()
            event.accept()
            return
        super().keyPressEvent(event)
