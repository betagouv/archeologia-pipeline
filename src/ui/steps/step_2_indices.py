"""Étape 2 — Produits à calculer.

Sélection des produits : MNT / Densité / Couverture (modèle de base) + indices
RVT, tous sur le même gabarit de carte (``_IndexCard``). Le MNT
est verrouillé tant qu'un indice RVT est coché (clic → toast explicatif). Selon
le mode de données, les sections déjà fournies sont masquées (existing_mnt /
existing_rvt). La logique pure vient de :mod:`app.services.indices_model`.

Trois sections (interface rationalisée, 2026-10-24) : « Recommandé pour commencer »
(MNT, SVF, LD, cochés d'avance), « Autres indices de visualisation », « Qualité de la
donnée ». Chaque carte = cocher + « ⓘ » (la fiche), et, une fois cochée, un lien
« ▸ Réglages (n) » qui déplie SES réglages dans la carte : plus de vue séparée à
onglets. La marge entre dalles est une carte de la page. Les réglages sont
persistés dans ``rvt_params`` et ``processing`` du config (consommés tels quels
par le pipeline).
"""
from __future__ import annotations

import copy

import logging
from pathlib import Path

from qgis.PyQt.QtCore import QPoint, Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...pipeline.ign.products.rvt_naming import PENTE_POURCENT, slope_unit as unite_pente
from ...app.services.indices_metres import en_px, format_m, migrer_px_vers_m
from ...app.services.rvt_kernel_context import (
    message_choix_inconnu,
    message_ecretage,
    mstp_scale_errors,
    tile_margin_px,
    tiled_context_warnings,
)
from ...app.services.indice_fiche import (
    build_all_fiches,
    build_comparaison,
    load_indices_fiches,
)
from ...app.services.indices_model import (
    all_products,
    count_selected,
    default_products,
    other_rvt_keys,
    product,
    products_unavailable_in_mode,
    quality_keys,
    recommended_keys,
    requires_mnt,
    rvt_keys,
    toggle,
)
from ...app.services.reglages_defaut import (
    TUILAGE,
    champs_du_produit,
    phrase_produit_reinitialise,
    produit_de_section,
)
from ...app.services.vocabulaire import pluriel
from ..icons import colored_pixmap
from ..widgets.card import build_card
from ..widgets.info_button import InfoButton
from ..widgets.no_wheel import NoWheelDoubleSpinBox, NoWheelSpinBox
from ..widgets.toast import show_toast
from ..widgets.vignette import FicheThumb

logger = logging.getLogger(__name__)

#: Dossier ``data/`` du plugin — les chemins de vignettes du JSON des fiches y
#: sont relatifs. Ce module vit dans ``src/ui/steps/``.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Filtre PDAL par défaut (identique à ConfigManager.default_config).
DEFAULT_FILTER = (
    "Classification = 2 OR Classification = 6 OR Classification = 66 "
    "OR Classification = 67 OR Classification = 9"
)

# Aides communes à plusieurs indices.
_HELP_VE = "Exagération verticale (1 = aucune)."
#: Bruit : rvt ignore max(arrondi(rayon × 10 / 20 / 40 %), 1) px du rayon ; au niveau 1, sous 15 px de rayon, cela
#: fait 1 px, comme sans suppression (mesuré identique au pixel à 10 px).
_HELP_BRUIT = ("Ignore le début du rayon pour gommer le bruit du nuage (0 = aucun, 3 = fort). "
               "« 1 » (faible) n'agit qu'à partir d'un rayon de 15 px.")
_HELP_8BIT = "Fichier plus léger, suffisant pour l'affichage."


class _IndexCard(QFrame):
    """Carte d'un produit : vignette + tag + nom + description + coche,
    « ⓘ » pour la fiche, et, cochée, « ▸ Réglages (n) » qui déplie ses réglages
    dans la carte (accordéon). Un clic n'importe où ailleurs coche ou décoche ;
    la carte prend le focus clavier et se coche à Espace.

    Sert AUSSI aux produits de base (MNT, Densité, Couverture) : il n'y a qu'un
    gabarit de carte à l'étape 2. Le MNT se verrouille (cadenas + infobulle) tant
    qu'un produit est coché : plus de toast.
    """

    clicked = pyqtSignal(str)
    fiche_requested = pyqtSignal(str)   # clé de produit — ouvrir sa fiche

    def __init__(self, key: str, tag: str, full_name: str, desc: str, parent=None):
        super().__init__(parent)
        self._key = key
        self._locked = False
        self._readonly = False
        self._reglages: QWidget | None = None
        self.setObjectName("IndexCard")
        self.setProperty("checked", False)
        self.setProperty("locked", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Colonne de gauche à largeur FIXE, occupée même sans vignette (cadre
        # d'attente) : les libellés restent alignés d'une carte à l'autre. La
        # vignette ne consomme plus le clic : un seul geste pour cocher, un seul
        # bouton (« ⓘ ») pour la fiche.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(9)
        self._thumb = FicheThumb("")
        self._thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._thumb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        outer.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        outer.addLayout(layout, 1)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._tag = QLabel(tag)
        self._tag.setObjectName("IndexTag")
        self._info_btn = InfoButton(
            "Ce que montre le produit, à quoi il sert, ce qu'il ne montre pas, réglages et sources"
        )
        self._info_btn.clicked.connect(lambda: self.fiche_requested.emit(self._key))
        self._check = QLabel("")
        self._check.setObjectName("IndexCheck")
        self._check.setFixedSize(15, 15)
        self._check.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._tag)
        top.addStretch(1)
        top.addWidget(self._info_btn)
        top.addWidget(self._check)
        self._name = QLabel(full_name)
        self._name.setObjectName("IndexName")
        # Sans retour à la ligne, « Multi-Scale Topographic Position » élargirait
        # sa colonne et déformerait la grille.
        self._name.setWordWrap(True)
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("IndexDesc")
        desc_lbl.setWordWrap(True)
        layout.addLayout(top)
        layout.addWidget(self._name)
        layout.addWidget(desc_lbl)

        # Accordéon des réglages : lien « ▸ Réglages (n) » visible quand la carte
        # est cochée, le widget des réglages en dessous, replié par défaut.
        self._toggle = QPushButton("")
        self._toggle.setObjectName("CardToggle")
        self._toggle.setFlat(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setVisible(False)
        self._toggle.clicked.connect(self._basculer_reglages)
        layout.addWidget(self._toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self._reglages_host = QVBoxLayout()
        self._reglages_host.setContentsMargins(0, 2, 0, 0)
        layout.addLayout(self._reglages_host)
        self._n_reglages = 0
        self._deplie = False

    def set_vignette(self, chemin, cadrage=None) -> None:
        """Pose la vignette de la fiche ; absente → cadre d'attente."""
        self._thumb.set_vignette(chemin, cadrage=cadrage)

    def set_reglages(self, widget: QWidget, n_champs: int) -> None:
        """Les réglages de cette image, dépliables dans la carte."""
        self._reglages = widget
        self._n_reglages = n_champs
        widget.setVisible(False)
        self._reglages_host.addWidget(widget)
        self._rendre_toggle()

    def _rendre_toggle(self) -> None:
        self._toggle.setText(f"{'▾' if self._deplie else '▸'} Réglages ({self._n_reglages})")
        self._toggle.setVisible(bool(self._reglages) and bool(self.property("checked")))

    def _basculer_reglages(self) -> None:
        self._deplie = not self._deplie
        if self._reglages is not None:
            self._reglages.setVisible(self._deplie)
        self._rendre_toggle()

    def set_checked(self, on: bool) -> None:
        self.setProperty("checked", on)
        if not on and self._deplie:
            self._basculer_reglages()   # décochée : ses réglages se replient
        self._rendre_check()
        self._rendre_toggle()
        # Re-polish la carte ET ses enfants stylés (tag/nom/coche) sinon les
        # sélecteurs descendants #IndexCard[checked] #IndexTag restent figés.
        for w in (self, self._check, self._tag, self._name):
            w.style().unpolish(w)
            w.style().polish(w)

    def set_locked(self, on: bool, tooltip: str = "") -> None:
        """Verrou (MNT nécessaire aux images cochées) : cadenas et infobulle, pas de toast."""
        self._locked = bool(on)
        self.setProperty("locked", self._locked)
        self.setToolTip(tooltip if on else "")
        self._rendre_check()
        self.style().unpolish(self)
        self.style().polish(self)

    def _rendre_check(self) -> None:
        on = bool(self.property("checked"))
        self._check.setProperty("checked", on)
        if self._locked and on:
            self._check.setText("")
            self._check.setPixmap(colored_pixmap("lock", "#ffffff", 10, dpr=self.devicePixelRatioF()))
        else:
            self._check.setPixmap(QPixmap())
            self._check.setText("✓" if on else "")
        self._check.style().unpolish(self._check)
        self._check.style().polish(self._check)

    def _dans_les_reglages(self, pos: QPoint) -> bool:
        """Un clic dans les réglages dépliés règle, il ne coche pas."""
        w = self._reglages
        if w is None or not w.isVisible():
            return False
        haut = w.mapTo(self, QPoint(0, 0)).y()
        return haut - 3 <= pos.y() <= haut + w.height() + 3

    def set_readonly(self, ro: bool) -> None:
        """Pendant un traitement : la carte ne se coche plus, mais son « ⓘ » et son lien
        « ▸ Réglages » restent cliquables (consulter, pas modifier) ; les champs des
        réglages sont grisés par la page."""
        self._readonly = bool(ro)
        self.setCursor(Qt.CursorShape.ArrowCursor if ro else Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if not self._readonly and not self._dans_les_reglages(pos):
            self.clicked.emit(self._key)
        super().mousePressEvent(event)

    def keyPressEvent(self, event):  # noqa: N802 (signature Qt)
        # Espace seulement : Entrée dans un champ des réglages dépliés remonte ici (les
        # QAbstractSpinBox, QComboBox et QLineEdit l'ignorent) et décochait la carte.
        if event.key() == Qt.Key.Key_Space and not self._readonly:
            self.clicked.emit(self._key)
            return
        super().keyPressEvent(event)


class IndicesPage(QWidget):
    changed = pyqtSignal()
    fiche_requested = pyqtSignal(str)   # clé d'image (ou « __comparaison__ ») → le panneau ⓘ

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "ign_laz"
        self._products = default_products()
        self._loading = False
        self._index_cards: dict = {}
        # descripteurs (produit, section, key, widget, kind, default) — le produit
        # porte la PORTÉE de la réinitialisation, cf. app/services/reglages_defaut.
        self._adv_fields: list = []
        self._reset_btns: dict = {}       # clé de produit → bouton « ↺ »
        self._fiches_cache = None         # fiches des produits, chargées à la demande
        self._comparaison_cache = None    # tableaux comparatifs, lus en même temps
        self._reglages: dict = {}         # clé de produit → widget de ses réglages (accordéon)
        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._build_reglages()   # les champs d'abord : les cartes les reçoivent
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        heading = QLabel("Quels produits voulez-vous ?")
        heading.setObjectName("WizardPageHeading")
        root.addWidget(heading)
        self._sub = QLabel("")
        self._sub.setObjectName("WizardPageSub")
        self._sub.setWordWrap(True)
        root.addWidget(self._sub)

        # « Comparer les images » depuis la page : les deux tableaux de la
        # documentation RVT, en tête des fiches.
        self._comparer_btn = QPushButton("Comparer les produits ▸")
        self._comparer_btn.setObjectName("SeeLink")
        self._comparer_btn.setFlat(True)
        self._comparer_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._comparer_btn.setToolTip("Quel produit pour quelle forme, et ce que chacun sait faire")
        self._comparer_btn.clicked.connect(lambda: self._open_fiche("__comparaison__"))
        root.addWidget(self._comparer_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self._mode_banner = QLabel("")
        self._mode_banner.setObjectName("InfoBanner")
        self._mode_banner.setWordWrap(True)
        self._mode_banner.setVisible(False)
        root.addWidget(self._mode_banner)

        # ── Trois sections, un seul gabarit de carte ──
        self._reco_card, rv = build_card("Recommandé pour commencer", "1")
        rv.addLayout(self._grille(recommended_keys(), 3))
        root.addWidget(self._reco_card)

        self._rvt_card, av = build_card("Autres indices de visualisation", "2")
        av.addLayout(self._grille(other_rvt_keys(), 3))
        footer = QHBoxLayout()
        self._count_label = QLabel("")
        self._count_label.setObjectName("WizardPageSub")
        footer.addWidget(self._count_label)
        footer.addStretch(1)
        av.addLayout(footer)
        root.addWidget(self._rvt_card)

        self._quality_card, qv = build_card("Qualité de la donnée", "3")
        qv.addLayout(self._grille(quality_keys(), 3))
        root.addWidget(self._quality_card)

        # ── Marge entre dalles : une carte de la page, avec son diagnostic ──
        root.addWidget(self._marge_card)

        root.addStretch(1)
        self._comparer_btn.setVisible(self._comparaison() is not None)

    def _grille(self, keys, colonnes: int) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, key in enumerate(keys):
            grid.addWidget(self._make_card(key), i // colonnes, i % colonnes, Qt.AlignmentFlag.AlignTop)
        for c in range(colonnes):
            grid.setColumnStretch(c, 1)
        return grid

    # ------------------------------------------------------------------
    # Réglages par image (accordéon de chaque carte — valeurs d'origine de rvt-qgis)
    # ------------------------------------------------------------------
    def _build_reglages(self) -> None:
        """Construit tous les champs de réglage et, pour chaque image, le widget
        que sa carte déplie (``self._reglages[clé]``), plus la carte de la marge.

        Pas de réinitialisation globale : elle emportait tous les produits d'un
        coup (demande utilisateur 2026-09-16). Chaque image porte la sienne.
        """
        # — Produits de base : un produit = ses seuls paramètres. La résolution
        #   MNT se règle ici et nulle part ailleurs.
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Ex: Classification = 2 OR Classification = 6")
        self._filter_edit.textChanged.connect(self._on_changed)
        self._res_spin = self._mk_dspin(0.1, 10.0, 0.5)
        self._density_spin = self._mk_dspin(0.01, 100.0, 1.0)
        self._cov_thr_spin = self._mk_spin(5, 95, 30)
        # ⚠ Ces quatre champs partagent la section ``processing`` : le produit ne
        # peut PAS s'en déduire, il faut le nommer — sans quoi les trois onglets de
        # base se réinitialiseraient ensemble.
        self._reg(("processing",), "filter_expression", self._filter_edit, "text",
                  DEFAULT_FILTER, produit="MNT")
        self._reg(("processing",), "mnt_resolution", self._res_spin, "float", 0.5,
                  produit="MNT")
        self._reg(("processing",), "density_resolution", self._density_spin, "float", 1.0,
                  produit="DENSITE")
        self._reg(("processing",), "coverage_threshold_percent", self._cov_thr_spin, "int",
                  30, produit="COUVERTURE")
        base_tabs = [
            ("MNT", [
                ("Filtre PDAL", self._filter_edit,
                 "Classes LiDAR conservées pour reconstruire le sol (codes ASPRS)."),
                ("Résolution MNT (m/pixel)", self._res_spin,
                 "Taille de cellule du MNT, base de tous les autres produits."),
            ]),
            ("DENSITE", [
                ("Résolution densité (m)", self._density_spin,
                 "Taille de cellule du raster de densité de points."),
            ]),
            ("COUVERTURE", [
                ("Seuil zones mal couvertes (%)", self._cov_thr_spin,
                 "En-dessous de ce % de cellules avec points sol dans le "
                 "voisinage, la zone est marquée « mal couverte ». Le calcul part "
                 "du raster de densité, dans sa résolution (onglet Densité)."),
            ]),
        ]
        for _key, _rows in base_tabs:
            self._reglages[_key] = self._make_reglages(_key, _rows)

        # — HS (hs) —
        hs_az = self._mk_spin(0, 360, 315)
        hs_sun = self._mk_spin(0, 90, 35)
        hs_ve = self._mk_spin(1, 100, 1)
        hs_8 = self._mk_check()
        self._reg(("rvt_params", "hs"), "sun_azimuth", hs_az, "int", 315)
        self._reg(("rvt_params", "hs"), "sun_elevation", hs_sun, "int", 35)
        self._reg(("rvt_params", "hs"), "ve_factor", hs_ve, "int", 1)
        self._reg(("rvt_params", "hs"), "save_as_8bit", hs_8, "bool", True)
        self._reglages["HS"] = self._make_reglages("HS", [
            ("Azimut solaire (°)", hs_az,
             "Direction de la lumière (0 = N, 90 = E, 180 = S, 270 = O)."),
            ("Élévation solaire (°)", hs_sun,
             "Hauteur du soleil au-dessus de l'horizon."),
            ("Facteur VE", hs_ve, _HELP_VE),
            ("", hs_8, _HELP_8BIT),
        ])

        # — M-HS (mdh) —
        # Bornes des spins = celles du moteur (moteur_indices.BORNES) : au-delà, la dalle échouait au calcul.
        mdh_dirs = self._mk_spin(2, 64, 16)
        mdh_sun = self._mk_spin(0, 90, 35)
        mdh_ve = self._mk_spin(1, 100, 1)
        mdh_8 = self._mk_check()
        self._reg(("rvt_params", "mdh"), "num_directions", mdh_dirs, "int", 16)
        self._reg(("rvt_params", "mdh"), "sun_elevation", mdh_sun, "int", 35)
        self._reg(("rvt_params", "mdh"), "ve_factor", mdh_ve, "int", 1)
        self._reg(("rvt_params", "mdh"), "save_as_8bit", mdh_8, "bool", True)
        self._reglages["M_HS"] = self._make_reglages("M_HS", [
            ("Nombre de directions", mdh_dirs,
             "Angles d'éclairage simulés. 16 = bon compromis qualité/temps."),
            ("Élévation solaire (°)", mdh_sun,
             "Hauteur du soleil au-dessus de l'horizon."),
            ("Facteur VE", mdh_ve, _HELP_VE),
            ("", mdh_8, _HELP_8BIT),
        ])
        # 8 bits : l'image compose trois éclairages (315°, 22,5°, 90°) ; le nombre de directions ne sert qu'au
        # fichier flottant (décision 0.15.0 ; nom de dossier inchangé, la valeur reste enregistrée).
        cellule_directions = mdh_dirs.parentWidget()
        mdh_8.toggled.connect(lambda huit: cellule_directions.setVisible(not huit))
        cellule_directions.setVisible(not mdh_8.isChecked())

        # — SVF (svf) —
        # Bornes : directions et bruit = bornes DURES de rvt:rvt_svf, comme pour OPNS. Le rayon se règle en
        # MÈTRES (2026-10-31, indices_metres.BORNES_M = anciennes bornes 10-50 px × 0,5 m), converti en pixels à la
        # résolution du MNT ; hors des bornes du calcul à cette résolution, refusé avant le lancement.
        svf_noise = self._mk_spin(0, 3, 0)
        svf_dirs = self._mk_spin(8, 64, 16)
        svf_radius = self._mk_mspin(5.0, 25.0, 5.0)
        svf_ve = self._mk_spin(1, 100, 1)
        svf_8 = self._mk_check()
        self._reg(("rvt_params", "svf"), "noise_remove", svf_noise, "int", 0)
        self._reg(("rvt_params", "svf"), "num_directions", svf_dirs, "int", 16)
        self._reg(("rvt_params", "svf"), "radius_m", svf_radius, "float", 5.0)
        self._reg(("rvt_params", "svf"), "ve_factor", svf_ve, "int", 1)
        self._reg(("rvt_params", "svf"), "save_as_8bit", svf_8, "bool", True)
        self._reglages["SVF"] = self._make_reglages("SVF", [
            ("Suppression du bruit", svf_noise, _HELP_BRUIT),
            ("Nombre de directions", svf_dirs,
             "Directions d'horizon échantillonnées autour de chaque pixel."),
            ("Rayon (m)", svf_radius,
             "Distance de recherche de l'horizon, au sol. Convertie en pixels à la résolution du MNT "
             "(5 m = 10 px à 0,5 m, 5 px à 1 m)."),
            ("Facteur VE", svf_ve, _HELP_VE),
            ("", svf_8, _HELP_8BIT),
        ])

        # — OPNS (opns) : openness, un seul réglage de type comme rvt-qgis —
        # Bornes des spins = bornes DURES de rvt:rvt_opns (directions 8-64, bruit 0-3) ; rayon en mètres, comme
        # le SVF.
        opns_type = QComboBox()
        opns_type.addItem("Positive — les saillies", 0)
        opns_type.addItem("Négative — les creux", 1)
        opns_type.currentIndexChanged.connect(self._on_changed)
        opns_noise = self._mk_spin(0, 3, 0)
        opns_dirs = self._mk_spin(8, 64, 16)
        opns_radius = self._mk_mspin(5.0, 25.0, 5.0)
        opns_ve = self._mk_spin(1, 100, 1)
        opns_8 = self._mk_check()
        self._reg(("rvt_params", "opns"), "opns_type", opns_type, "combo", 0)
        self._reg(("rvt_params", "opns"), "noise_remove", opns_noise, "int", 0)
        self._reg(("rvt_params", "opns"), "num_directions", opns_dirs, "int", 16)
        self._reg(("rvt_params", "opns"), "radius_m", opns_radius, "float", 5.0)
        self._reg(("rvt_params", "opns"), "ve_factor", opns_ve, "int", 1)
        self._reg(("rvt_params", "opns"), "save_as_8bit", opns_8, "bool", True)
        self._reglages["OPNS"] = self._make_reglages("OPNS", [
            ("Type d'ouverture", opns_type,
             "Positive : ce qui dépasse du voisinage (tertres, crêtes, lèvres de "
             "cratère). Négative : ce qui s'y enfonce (fossés, chemins creux). "
             "Un traitement = un type ; l'autre s'obtient en relançant, dans un dossier "
             "distinct."),
            ("Suppression du bruit", opns_noise, _HELP_BRUIT),
            ("Nombre de directions", opns_dirs,
             "Directions d'horizon échantillonnées autour de chaque pixel."),
            ("Rayon (m)", opns_radius,
             "Distance de recherche de l'horizon, au sol. Borne la taille des formes vues ; "
             "convertie en pixels à la résolution du MNT."),
            ("Facteur VE", opns_ve, _HELP_VE),
            ("", opns_8, _HELP_8BIT),
        ])

        # — Slope (slope) —
        slope_unit = QComboBox()
        slope_unit.addItem("Degrés", 0)
        slope_unit.addItem("Pourcentage", PENTE_POURCENT)   # code « percent » de rvt (1 = radians)
        slope_unit.currentIndexChanged.connect(self._on_changed)
        slope_ve = self._mk_spin(1, 100, 1)
        slope_8 = self._mk_check()
        self._reg(("rvt_params", "slope"), "unit", slope_unit, "combo", 0)
        self._reg(("rvt_params", "slope"), "ve_factor", slope_ve, "int", 1)
        self._reg(("rvt_params", "slope"), "save_as_8bit", slope_8, "bool", True)
        self._reglages["SLO"] = self._make_reglages("SLO", [
            ("Unité", slope_unit, "Pente exprimée en degrés ou en pourcentage."),
            ("Facteur VE", slope_ve, _HELP_VE),
            ("", slope_8, _HELP_8BIT),
        ])

        # — LD (ldo) —
        ld_ang = self._mk_spin(4, 32, 15)
        # Anneau en MÈTRES (2026-10-31) : 5–10 m, celui sur lequel tous les modèles ont appris, quelle que soit la
        # résolution (10–20 px à 0,5 m, 5–10 px à 1 m). Bornes = anciennes bornes 1-100 px × 0,5 m.
        ld_rmin = self._mk_mspin(0.5, 50.0, 5.0)   # 0 : division par une distance nulle
        ld_rmax = self._mk_mspin(0.5, 50.0, 10.0)
        ld_obs = self._mk_dspin(0.5, 20.0, 1.7)
        # rayon min ≤ rayon max : sinon aucune distance, image sans valeur. Le champ modifié POUSSE l'autre (et non des
        # bornes croisées, qui ne se posaient qu'au premier changement du voisin et dépendaient de l'ordre de
        # chargement d'une configuration) : vrai dès l'ouverture, quelle que soit la valeur chargée.
        ld_rmin.valueChanged.connect(lambda v, hi=ld_rmax: hi.setValue(v) if v > hi.value() else None)
        ld_rmax.valueChanged.connect(lambda v, lo=ld_rmin: lo.setValue(v) if v < lo.value() else None)
        ld_ve = self._mk_spin(1, 100, 1)
        ld_8 = self._mk_check()
        self._reg(("rvt_params", "ldo"), "angular_res", ld_ang, "int", 15)
        self._reg(("rvt_params", "ldo"), "min_radius_m", ld_rmin, "float", 5.0)
        self._reg(("rvt_params", "ldo"), "max_radius_m", ld_rmax, "float", 10.0)
        self._reg(("rvt_params", "ldo"), "observer_h", ld_obs, "float", 1.7)
        self._reg(("rvt_params", "ldo"), "ve_factor", ld_ve, "int", 1)
        self._reg(("rvt_params", "ldo"), "save_as_8bit", ld_8, "bool", True)
        self._reglages["LD"] = self._make_reglages("LD", [
            ("Résolution angulaire (°)", ld_ang,
             "Pas angulaire du balayage. Plus petit = plus précis mais plus lent."),
            ("Rayon min (m)", ld_rmin,
             "Distance minimale prise en compte autour du pixel, au sol. Les modèles de détection "
             "ont appris sur l'anneau 5–10 m."),
            ("Rayon max (m)", ld_rmax,
             "Distance maximale prise en compte autour du pixel, au sol."),
            ("Hauteur observateur (m)", ld_obs,
             "Hauteur de l'œil virtuel au-dessus du sol."),
            ("Facteur VE", ld_ve, _HELP_VE),
            ("", ld_8, _HELP_8BIT),
        ])

        # — SLRM (slrm) —
        slrm_radius = self._mk_mspin(5.0, 25.0, 10.0)   # mètres (anciennes bornes 10-50 px × 0,5 m)
        slrm_ve = self._mk_spin(1, 100, 1)
        slrm_8 = self._mk_check()
        self._reg(("rvt_params", "slrm"), "radius_m", slrm_radius, "float", 10.0)
        self._reg(("rvt_params", "slrm"), "ve_factor", slrm_ve, "int", 1)
        self._reg(("rvt_params", "slrm"), "save_as_8bit", slrm_8, "bool", True)
        self._reglages["SLRM"] = self._make_reglages("SLRM", [
            ("Rayon (m)", slrm_radius,
             "Rayon du lissage, au sol : sépare le micro-relief du relief général."),
            ("Facteur VE", slrm_ve, _HELP_VE),
            ("", slrm_8, _HELP_8BIT),
        ])

        # — VAT (vat) —
        vat_terrain = QComboBox()
        vat_terrain.addItem("Général", 0)
        vat_terrain.addItem("Plat", 1)
        vat_terrain.addItem("Pentu", 2)
        vat_terrain.currentIndexChanged.connect(self._on_changed)
        vat_8 = self._mk_check()
        self._reg(("rvt_params", "vat"), "terrain_type", vat_terrain, "combo", 0)
        self._reg(("rvt_params", "vat"), "save_as_8bit", vat_8, "bool", True)
        self._reglages["VAT"] = self._make_reglages("VAT", [
            ("Type de terrain", vat_terrain,
             "Préréglage adapté au relief dominant de la zone."),
            ("", vat_8, _HELP_8BIT),
        ])

        # — MSTP (mstp) : position topographique multi-échelle —
        # Échelles en MÈTRES (2026-10-31) : anciens défauts et bornes en pixels × 0,5 m.
        mstp_lmin = self._mk_mspin(0.5, 50000.0, 1.5)
        mstp_lmax = self._mk_mspin(0.5, 50000.0, 10.5)
        mstp_lstep = self._mk_mspin(0.5, 50000.0, 1.0)
        mstp_mmin = self._mk_mspin(0.5, 50000.0, 11.5)
        mstp_mmax = self._mk_mspin(0.5, 50000.0, 101.5)
        mstp_mstep = self._mk_mspin(0.5, 50000.0, 9.0)
        # Échelle large : défauts du plugin (100/400/60), pas ceux de RVT
        # (223/2023/180) — un rayon de 2023 px dépasse largement le contexte que
        # la marge de tuilage fournit. Cf. app.services.rvt_kernel_context.
        mstp_bmin = self._mk_mspin(0.5, 50000.0, 50.0)
        mstp_bmax = self._mk_mspin(0.5, 50000.0, 200.0)
        mstp_bstep = self._mk_mspin(0.5, 50000.0, 30.0)
        mstp_light = self._mk_dspin(0.1, 5.0, 0.9)
        # Pas de « Facteur VE » : le MSTP divise l'écart par l'écart type local, l'exagération s'y annule (mesuré :
        # médiane de l'écart 5·10⁻⁶ ; tests/unit/test_effet_reglages.py). Le champ ne faisait rien.
        mstp_8 = self._mk_check()
        self._reg(("rvt_params", "mstp"), "local_scale_min_m", mstp_lmin, "float", 1.5)
        self._reg(("rvt_params", "mstp"), "local_scale_max_m", mstp_lmax, "float", 10.5)
        self._reg(("rvt_params", "mstp"), "local_scale_step_m", mstp_lstep, "float", 1.0)
        self._reg(("rvt_params", "mstp"), "meso_scale_min_m", mstp_mmin, "float", 11.5)
        self._reg(("rvt_params", "mstp"), "meso_scale_max_m", mstp_mmax, "float", 101.5)
        self._reg(("rvt_params", "mstp"), "meso_scale_step_m", mstp_mstep, "float", 9.0)
        self._reg(("rvt_params", "mstp"), "broad_scale_min_m", mstp_bmin, "float", 50.0)
        self._reg(("rvt_params", "mstp"), "broad_scale_max_m", mstp_bmax, "float", 200.0)
        self._reg(("rvt_params", "mstp"), "broad_scale_step_m", mstp_bstep, "float", 30.0)
        self._reg(("rvt_params", "mstp"), "lightness", mstp_light, "float", 0.9)
        self._reg(("rvt_params", "mstp"), "save_as_8bit", mstp_8, "bool", True)
        self._reglages["MSTP"] = self._make_reglages("MSTP", [
            ("Échelle locale — rayon min (m)", mstp_lmin,
             "Détails fins (canal bleu de la composition RGB)."),
            ("Échelle locale — rayon max (m)", mstp_lmax, "Borne haute de l'échelle locale."),
            ("Échelle locale — pas (m)", mstp_lstep, "Pas d'échantillonnage de l'échelle locale."),
            ("Échelle méso — rayon min (m)", mstp_mmin,
             "Structures intermédiaires (canal vert)."),
            ("Échelle méso — rayon max (m)", mstp_mmax, "Borne haute de l'échelle méso."),
            ("Échelle méso — pas (m)", mstp_mstep, "Pas d'échantillonnage de l'échelle méso."),
            ("Échelle large — rayon min (m)", mstp_bmin,
             "Grandes formes du paysage (canal rouge)."),
            ("Échelle large — rayon max (m)", mstp_bmax, "Borne haute de l'échelle large."),
            ("Échelle large — pas (m)", mstp_bstep, "Pas d'échantillonnage de l'échelle large."),
            ("Luminosité", mstp_light, "Clarté globale de l'image composite (0.1–5)."),
            ("", mstp_8, _HELP_8BIT),
        ])

        # — CVAT (cvat) : VAT combiné (general + flat), composition figée —
        cvat_8 = self._mk_check()
        self._reg(("rvt_params", "cvat"), "save_as_8bit", cvat_8, "bool", True)
        self._reglages["CVAT"] = self._make_reglages("CVAT", [
            ("", cvat_8, _HELP_8BIT),
        ])

        # — PRISM (prism) : ouverture prismatique, combinaison 1 du blender —
        # Même forme que le VAT : le préréglage de terrain vaut pour TOUTE
        # combinaison du blender, il change l'élévation solaire et les
        # étirements sous-jacents.
        # Chaque ``_reg`` est écrit en toutes lettres (et non dans une boucle) :
        # tests/unit/test_reset_defauts_ui.py relit ces appels par AST pour
        # vérifier que tout champ est rattaché à un produit. Une section passée
        # par variable y devient illisible, et le champ orphelin.
        prism_terrain = QComboBox()
        prism_terrain.addItem("Général", 0)
        prism_terrain.addItem("Plat", 1)
        prism_terrain.addItem("Pentu", 2)
        prism_terrain.currentIndexChanged.connect(self._on_changed)
        prism_8 = self._mk_check()
        self._reg(("rvt_params", "prism"), "terrain_type", prism_terrain, "combo", 0)
        self._reg(("rvt_params", "prism"), "save_as_8bit", prism_8, "bool", True)
        self._reglages["PRISM"] = self._make_reglages("PRISM", [
            ("Type de terrain", prism_terrain,
             "Préréglage adapté au relief dominant de la zone."),
            ("", prism_8, _HELP_8BIT),
        ])

        # — CRIM (crim) : relief coloré par la pente —
        crim_cmap = QComboBox()
        # Colormaps matplotlib séquentielles, lisibles en relief. « OrRd » est
        # le défaut de rvt.blend.color_relief_image_map : fond sable, pentes en
        # rouge. Changer de colormap change entièrement l'image, d'où le dossier
        # de sortie distinct (cf. rvt_naming.get_rvt_param_suffix).
        for _libelle, _valeur in (("Orangé → rouge (OrRd)", "OrRd"),
                                  ("Rouge inversé (Reds_r)", "Reds_r"),
                                  ("Jaune → orangé → rouge (YlOrRd)", "YlOrRd"),
                                  ("Gris inversé (Greys_r)", "Greys_r"),
                                  ("Terrain (gist_earth)", "gist_earth")):
            crim_cmap.addItem(_libelle, _valeur)
        crim_cmap.currentIndexChanged.connect(self._on_changed)
        crim_cut_min = self._mk_dspin(0.0, 1.0, 0.0)
        crim_cut_max = self._mk_dspin(0.0, 1.0, 1.0)
        # Les deux bornes ne peuvent pas se croiser : RVT lève dès que la coupe
        # basse atteint la haute, et le garde-fou de crim.py avale l'exception —
        # l'indice serait simplement absent, sans rien dire à l'utilisateur.
        crim_cut_min.valueChanged.connect(
            lambda v, hi=crim_cut_max: hi.setMinimum(min(1.0, v + 0.01)))
        crim_cut_max.valueChanged.connect(
            lambda v, lo=crim_cut_min: lo.setMaximum(max(0.0, v - 0.01)))
        crim_8 = self._mk_check()
        self._reg(("rvt_params", "crim"), "colormap", crim_cmap, "combo", "OrRd")
        self._reg(("rvt_params", "crim"), "min_colormap_cut", crim_cut_min, "float", 0.0)
        self._reg(("rvt_params", "crim"), "max_colormap_cut", crim_cut_max, "float", 1.0)
        self._reg(("rvt_params", "crim"), "save_as_8bit", crim_8, "bool", True)
        self._reglages["CRIM"] = self._make_reglages("CRIM", [
            ("Palette de couleurs", crim_cmap,
             "Palette appliquée à la PENTE du terrain, jamais à l'altitude."),
            ("Coupe basse de la palette", crim_cut_min,
             "Retire le bas de la palette (0 = palette entière)."),
            ("Coupe haute de la palette", crim_cut_max,
             "Retire le haut de la palette (1 = palette entière)."),
            ("", crim_8, _HELP_8BIT),
        ])

        # — Marge entre dalles (vaut pour toutes les images) —
        ov_card, ovv = build_card("Marge entre dalles")
        self._overlap_spin = self._mk_spin(0, 100, 20)
        self._reg(("processing",), "tile_overlap", self._overlap_spin, "int", 20,
                  produit=TUILAGE)
        # La marge ne sert qu'aux modes qui fusionnent des dalles voisines (LAZ) : masquée ailleurs (set_mode), où elle
        # ne faisait rien. Le diagnostic de contexte (_ctx_hint) reste, il vaut aussi pour un MNT existant.
        self._ov_marge = QWidget()
        ov_row = QHBoxLayout(self._ov_marge)
        ov_row.setContentsMargins(0, 0, 0, 0)
        ov_row.setSpacing(8)
        ov_lbl = QLabel("Marge entre dalles")
        ov_lbl.setObjectName("FieldLabel")
        ov_row.addWidget(ov_lbl)
        ov_row.addWidget(self._overlap_spin)
        ov_unit = QLabel("%")
        ov_unit.setObjectName("FieldUnit")
        ov_row.addWidget(ov_unit)
        ov_row.addStretch(1)
        # Le tuilage n'a pas d'onglet — il vaut pour tous les indices — donc il
        # porte son bouton ici. Sans lui, ce réglage serait le seul à n'avoir
        # plus aucune réinitialisation depuis le retrait du bouton global.
        ov_row.addWidget(self._mk_reset_btn(TUILAGE, "Marge entre dalles"))
        ov_hint = QLabel(
            "Chevauchement entre dalles voisines pour le calcul des produits. Évite "
            "les artefacts aux bordures. Vaut pour tous les produits."
        )
        ov_hint.setObjectName("MntHint")
        ov_hint.setWordWrap(True)
        self._ov_hint = ov_hint
        # Diagnostic live : la marge fournit-elle assez de contexte aux noyaux
        # RVT demandés ? C'est ici que la marge, la résolution et les rayons se
        # règlent, donc c'est ici que la réponse doit s'afficher.
        self._ctx_hint = QLabel("")
        self._ctx_hint.setObjectName("MntHint")
        self._ctx_hint.setWordWrap(True)
        ovv.addWidget(self._ov_marge)
        ovv.addWidget(ov_hint)
        ovv.addWidget(self._ctx_hint)
        self._marge_card = ov_card

    # — fabriques de widgets (numériques wheel-safe) —
    def _mk_spin(self, lo: int, hi: int, val: int) -> NoWheelSpinBox:
        s = NoWheelSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(self._on_changed)
        return s

    def _mk_dspin(self, lo: float, hi: float, val: float) -> NoWheelDoubleSpinBox:
        s = NoWheelDoubleSpinBox()
        s.setDecimals(2)
        s.setRange(lo, hi)
        s.setSingleStep(0.1)
        s.setValue(val)
        s.valueChanged.connect(self._on_changed)
        return s

    def _mk_mspin(self, lo: float, hi: float, val: float) -> NoWheelDoubleSpinBox:
        """Champ d'une distance au sol (m), au pas de 0,5 m — la résolution du LiDAR HD : à 0,5 m chaque
        valeur proposée tombe sur un nombre entier de pixels (règle 2026-10-31, indices_metres)."""
        s = NoWheelDoubleSpinBox()
        s.setDecimals(1)
        s.setRange(lo, hi)
        s.setSingleStep(0.5)
        s.setSuffix(" m")
        s.setValue(val)
        s.valueChanged.connect(self._on_changed)
        return s

    def _mk_check(self) -> QCheckBox:
        c = QCheckBox("Sauver en 8 bits")
        c.setChecked(True)
        c.toggled.connect(self._on_changed)
        return c

    def _reg(self, section: tuple, key: str, widget, kind: str, default,
             produit: str = "") -> None:
        """Enregistre un champ avancé, rattaché à SON produit.

        Le produit se déduit de la section pour les indices RVT ; il doit être
        nommé explicitement pour ``("processing",)``, que MNT, Densité,
        Couverture et le tuilage se partagent.
        """
        produit = produit or produit_de_section(section)
        assert produit, f"champ {key} sans image : la réinitialisation le raterait"
        self._adv_fields.append((produit, section, key, widget, kind, default))

    def _make_reglages(self, key: str, rows) -> QWidget:
        """Les réglages d'une image, dépliés dans sa carte : grille 2 colonnes de
        champs avec aide, et le « ↺ » qui ne rend que SES valeurs par défaut.

        ``rows`` = liste de ``(label, widget, help_text)``. Un ``QLineEdit``
        occupe les deux colonnes (champ texte large, ex. filtre PDAL).
        """
        tab = QWidget()
        tab.setObjectName("CardReglages")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 4, 0, 2)
        outer.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        r = c = 0
        for label, widget, help_text in rows:
            cell = self._param_cell(label, widget, help_text)
            if isinstance(widget, QLineEdit):
                if c != 0:
                    r += 1
                    c = 0
                grid.addWidget(cell, r, 0, 1, 2)
                r += 1
            else:
                grid.addWidget(cell, r, c)
                c += 1
                if c == 2:
                    c = 0
                    r += 1
        outer.addLayout(grid)
        pied = QHBoxLayout()
        pied.addStretch(1)
        pied.addWidget(self._mk_reset_btn(key, product(key).full_name), 0,
                       Qt.AlignmentFlag.AlignTop)
        outer.addLayout(pied)
        return tab

    @staticmethod
    def _param_cell(label: str, widget, help_text: str) -> QWidget:
        """Cellule de paramètre : ligne [label · valeur] (ou case à cocher seule)
        surmontant un sous-texte d'aide."""
        cell = QVBoxLayout()
        cell.setContentsMargins(0, 0, 0, 0)
        cell.setSpacing(3)
        if isinstance(widget, QCheckBox):
            cell.addWidget(widget)
        elif isinstance(widget, QLineEdit):
            if label:
                lbl = QLabel(label)
                lbl.setObjectName("FieldLabel")
                cell.addWidget(lbl)
            cell.addWidget(widget)
        else:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(label)
            lbl.setObjectName("FieldLabel")
            if isinstance(widget, QComboBox):
                widget.setMinimumWidth(130)
            else:
                widget.setFixedWidth(96)
            # Label et champ accolés (stretch APRÈS) : avec le stretch entre
            # les deux, le champ se retrouvait visuellement plus proche du
            # label du paramètre voisin que du sien (loi de proximité).
            row.addWidget(lbl)
            row.addWidget(widget)
            row.addStretch(1)
            cell.addLayout(row)
        if help_text:
            hint = QLabel(help_text)
            hint.setObjectName("FieldHint")
            hint.setWordWrap(True)
            cell.addWidget(hint)
        wrap = QWidget()
        wrap.setLayout(cell)
        return wrap

    # ------------------------------------------------------------------
    # Cartes de produits et fiches
    # ------------------------------------------------------------------
    def _make_card(self, key: str) -> _IndexCard:
        """Une carte de produit, câblée et illustrée.

        Produits de base et indices passent par ici : un seul gabarit, donc un
        seul endroit où brancher la sélection et l'accès à la fiche.
        """
        p = product(key)
        card = _IndexCard(key, p.tag, p.full_name, p.description)
        card.clicked.connect(self._on_product_clicked)
        card.fiche_requested.connect(self._open_fiche)
        chemin, cadrage = self._vignette_de(key)
        card.set_vignette(chemin, cadrage)
        reglages = self._reglages.get(key)
        if reglages is not None:
            card.set_reglages(reglages, len(champs_du_produit(self._adv_fields, key)))
        self._index_cards[key] = card
        return card

    def _comparaison(self):
        """Les tableaux comparatifs, ou ``None`` s'il n'y en a pas."""
        self._fiches()
        c = self._comparaison_cache
        return c if (c is not None and not getattr(c, "est_vide", True)) else None

    def _fiches(self):
        """Les fiches de tous les produits, lues une fois par session.

        Le JSON est petit et la lecture est tolérante (fichier absent → fiches
        dégradées) : l'étape 2 s'ouvre même si le fichier manque.
        """
        if self._fiches_cache is None:
            donnees = load_indices_fiches()
            self._fiches_cache = build_all_fiches(donnees)
            self._comparaison_cache = build_comparaison(donnees)
        return self._fiches_cache

    def _vignette_de(self, key: str):
        """``(chemin absolu, cadrage)`` de la vignette d'icône, ou ``(None, None)``.

        La première vignette déclarée qui existe réellement sur disque ; son
        ``cadrage`` dit quelle fenêtre l'icône de 44 px découpe.
        """
        for fiche in self._fiches():
            if fiche.cle != key:
                continue
            for v in fiche.vignettes:
                chemin = _DATA_DIR / v.image
                if chemin.is_file():
                    return str(chemin), v.cadrage
        return None, None

    def _open_fiche(self, key: str) -> None:
        """Demande la fiche de ``key`` (ou « __comparaison__ ») : l'assistant la pose
        dans son panneau latéral, à côté de la page."""
        self.fiche_requested.emit(key)

    def corps_fiche(self, key: str):
        """``(corps, titre)`` des fiches des images, ouvertes sur ``key``. Import
        différé des widgets de fiche : ils ne se chargent que si on la demande."""
        from ..dialogs.indice_info_dialog import corps_fiche_indice

        fiches = self._fiches()   # remplit aussi le cache de comparaison
        return (
            corps_fiche_indice(fiches, _DATA_DIR, key, comparaison=self._comparaison()),
            "Produits",
        )

    # ------------------------------------------------------------------
    # Logique
    # ------------------------------------------------------------------
    def _on_product_clicked(self, key: str) -> None:
        new, verrou = toggle(self._products, key)
        if verrou:
            return   # MNT nécessaire aux images cochées : la carte le dit (cadenas, infobulle)
        self._products = new
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_changed(self, *_args) -> None:
        self._update_context_hint()
        if not self._loading:
            self.changed.emit()

    def _update_context_hint(self) -> None:
        """Rafraîchit le diagnostic « contexte disponible vs noyaux RVT demandés ».

        Le calcul est délégué à ``app.services.rvt_kernel_context`` (pur, testé) ;
        les paramètres courants sont relus via le sérialiseur existant plutôt que
        de redupliquer la lecture des widgets.
        """
        scratch: dict = {}
        self._collect_advanced(scratch)
        rvt_params = scratch.get("rvt_params", {})

        if self._mode not in ("ign_laz", "local_laz"):
            self._ctx_hint.setText(
                "Aucune dalle n'est fusionnée dans ce mode : le contexte disponible "
                "dépend de la taille de chaque raster fourni, vérifiée au lancement."
            )
            return

        overlap = self._overlap_spin.value()
        resolution = self._res_spin.value()
        rvt_params = en_px(rvt_params, resolution)   # rayons réglés en mètres, contrôlés en pixels
        messages = tiled_context_warnings(
            self._products, rvt_params,
            tile_overlap_percent=overlap, mnt_resolution=resolution,
        ) + mstp_scale_errors(self._products, rvt_params)
        if messages:
            self._ctx_hint.setText("⚠  " + "\n⚠  ".join(messages))
        else:
            marge = tile_margin_px(overlap, resolution)
            self._ctx_hint.setText(
                f"✓  Contexte fourni aux noyaux RVT : {format_m(marge * resolution)} ({marge} px) "
                "de chaque côté de la dalle — suffisant pour les produits cochés."
            )

    def _refresh(self) -> None:
        for key, card in self._index_cards.items():
            card.set_checked(self._products.get(key, False))
        mnt = self._index_cards.get("MNT")
        if mnt is not None:
            actifs = [product(k).tag for k in rvt_keys() if self._products.get(k)]
            mnt.set_locked(
                requires_mnt(self._products),
                f"Nécessaire aux produits cochés : {', '.join(actifs)}. Décochez-les pour retirer le MNT."
                if actifs else "",
            )
        n = count_selected(self._products)
        self._count_label.setText(
            f"{pluriel(n, 'produit')} coché{'s' if n > 1 else ''}"
        )
        self._sub.setText(
            "Cliquez un produit pour le cocher. Le MNT est la base de tous les autres."
            if self._mode not in ("existing_mnt", "existing_rvt")
            else "Vous fournissez le MNT : cochez les produits à calculer."
        )
        self._update_context_hint()

    # ------------------------------------------------------------------
    # Mode (neutralisation des sections fournies)
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        laz = mode in ("ign_laz", "local_laz")
        # Depuis un MNT existant, le relief nu est la donnée d'entrée et la qualité
        # de la donnée n'existe pas ; depuis des images existantes, rien à calculer.
        self._index_cards["MNT"].setVisible(laz)
        self._quality_card.setVisible(laz)
        self._reco_card.setVisible(mode != "existing_rvt")
        self._rvt_card.setVisible(mode != "existing_rvt")
        self._marge_card.setVisible(mode != "existing_rvt")
        if mode == "existing_mnt":
            self._mode_banner.setText(
                "ℹ  Vous fournissez le MNT : le relief nu et la qualité de la donnée ne sont pas recalculés."
            )
            self._mode_banner.setVisible(True)
        elif mode == "existing_rvt":
            self._mode_banner.setText(
                "ℹ  Vous fournissez les indices de visualisation : cette étape est sans objet, passez aux structures."
            )
            self._mode_banner.setVisible(True)
        else:
            self._mode_banner.setVisible(False)
        # Marge entre dalles : seuls les modes LAZ fusionnent des voisines (sans effet ailleurs, mesuré au pixel).
        self._ov_marge.setVisible(mode in ("ign_laz", "local_laz"))
        self._ov_hint.setVisible(mode in ("ign_laz", "local_laz"))
        # Purge des produits dérivés du nuage (densité/couverture) dans les modes
        # sans nuage : la carte « Modèle de base » est seulement masquée, jamais
        # réinitialisée — sans ça une sélection héritée d'un run LAZ persiste dans
        # self._products et se resérialise (faux « produit demandé », avertissement
        # « Couverture indisponible… — ignoré » côté runner).
        purged = False
        for key in products_unavailable_in_mode(mode):
            if self._products.get(key):
                self._products[key] = False
                purged = True
        self._refresh()
        # Persiste la purge (autosave nettoie last_ui_config). Garde anti-boucle
        # identique à activate_product ; au démarrage le signal n'est pas encore
        # connecté, donc l'émission est sans effet (cohérent).
        if purged and not self._loading:
            self.changed.emit()

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def active_rvt_keys(self) -> set:
        """Indices RVT actuellement cochés (pour l'étape 3)."""
        return {k for k in rvt_keys() if self._products.get(k)}

    def recap_products(self) -> list:
        """Tags des produits actifs, dans l'ordre du catalogue (récap étape 4)."""
        return [p.tag for p in all_products() if self._products.get(p.key)]

    def resolution(self) -> float:
        """Résolution MNT (m/pixel), pour le récap de l'étape 4."""
        return float(self._res_spin.value())

    def activate_product(self, key: str) -> None:
        """Active un produit (utilisé par « + Activer » de l'étape 3)."""
        if self._products.get(key):
            return
        self._products[key] = True
        if key in rvt_keys():
            self._products["MNT"] = True
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def summary(self) -> str:
        n_rvt = sum(1 for k in rvt_keys() if self._products.get(k))
        base = "MNT" if self._products.get("MNT") else "—"
        if n_rvt:
            return f"{base} + {pluriel(n_rvt, 'indice')}"   # sous-libellé du rail : « MNT + 2 indices »
        return base

    def set_readonly(self, ro: bool) -> None:
        """Verrouille la saisie pour consultation pendant un run (lecture seule).

        Les cartes ne se cochent plus et leurs champs sont grisés ; « ⓘ », « ▸ Réglages »
        et le défilement restent, pour tout consulter. N'agit que sur
        ``setEnabled``/``setReadOnly`` → aucun signal ``changed`` ni autosave.
        """
        for btn in self._reset_btns.values():
            btn.setEnabled(not ro)
        for card in self._index_cards.values():
            card.set_readonly(ro)
        for _p, _section, _key, widget, kind, _default in self._adv_fields:
            widget.setEnabled(not ro)
            if kind == "text":
                widget.setReadOnly(ro)

    def load_from(self, config: dict) -> None:
        proc = config.get("processing") or {}
        prods = proc.get("products") or {}
        prev = self._loading
        self._loading = True
        try:
            self._products = {p.key: bool(prods.get(p.key, False)) for p in all_products()}
            self._load_advanced(config)  # mnt_resolution incluse (champ avancé)
            self._refresh()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        proc = config.setdefault("processing", {})
        products = proc.setdefault("products", {})
        for p in all_products():
            products[p.key] = bool(self._products.get(p.key, False))
        self._collect_advanced(config)

    # ------------------------------------------------------------------
    # Réglages avancés — chargement / collecte / reset (data-driven)
    # ------------------------------------------------------------------
    def _load_advanced(self, config: dict) -> None:
        # Valeurs hors bornes écrêtées par le champ : dites (journal + toast de l'assistant), plus en silence.
        self.ecretages = []
        # Rayons encore en pixels (configuration nommée d'avant le 2026-10-31) : convertis en mètres sur une
        # copie (0,5 m/px supposé), et dits de même.
        if isinstance(config.get("rvt_params"), dict):
            config = {**config, "rvt_params": copy.deepcopy(config["rvt_params"])}
            for msg in migrer_px_vers_m(config["rvt_params"]):
                logger.warning(msg)
                self.ecretages.append(msg)
        for produit, section, key, widget, kind, default in self._adv_fields:
            container = config
            for part in section:
                container = container.get(part) if isinstance(container, dict) else None
                if container is None:
                    break
            raw = container.get(key, default) if isinstance(container, dict) else default
            if (section, key) == (("rvt_params", "slope"), "unit"):
                raw = unite_pente({"slope": {"unit": raw}})   # l'ancien 1 (« Pourcentage ») devient 2
            msg = None
            if kind in ("int", "float"):
                msg = message_ecretage(produit, key, raw, widget.minimum(), widget.maximum(), kind == "float")
            elif kind == "combo":
                msg = message_choix_inconnu(produit, key, raw, [widget.itemData(i) for i in range(widget.count())],
                                            default)
            if msg:
                logger.warning(msg)
                self.ecretages.append(msg)
            self._apply_field(widget, kind, raw, default)

    def _collect_advanced(self, config: dict) -> None:
        for _p, section, key, widget, kind, _default in self._adv_fields:
            container = config
            for part in section:
                container = container.setdefault(part, {})
            container[key] = self._read_field(widget, kind)

    def _mk_reset_btn(self, produit: str, libelle: str) -> QPushButton:
        """Bouton « ↺ Défauts » d'un produit, mémorisé pour le mode lecture seule."""
        btn = QPushButton("↺  Défauts")
        btn.setObjectName("GhostButton")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(
            f"Revenir aux valeurs par défaut de « {libelle} ». "
            "Les autres produits ne sont pas touchés."
        )
        btn.clicked.connect(lambda _c=False, p=produit, lib=libelle: self._reset_produit(p, lib))
        self._reset_btns[produit] = btn
        return btn

    def _reset_produit(self, produit: str, libelle: str) -> None:
        """Rétablit les défauts d'UN produit. Les onze autres sont intacts.

        La portée vient de ``app.services.reglages_defaut.champs_du_produit`` :
        c'est le produit déclaré à l'enregistrement qui fait foi, jamais la
        section de configuration — MNT, Densité et Couverture partagent
        ``processing`` et seraient sinon réinitialisés ensemble.
        """
        prev = self._loading
        self._loading = True
        modifies = 0
        try:
            for _p, _section, _key, widget, kind, default in champs_du_produit(
                self._adv_fields, produit
            ):
                avant = self._read_field(widget, kind)
                self._apply_field(widget, kind, default, default)
                if self._read_field(widget, kind) != avant:
                    modifies += 1
        finally:
            self._loading = prev
        # Confirmation qui NOMME le produit : c'est ce qui dit à l'utilisateur
        # qu'il n'a pas perdu le reste de ses réglages.
        show_toast(self, phrase_produit_reinitialise(libelle, modifies)
                   or f"↺  {libelle} : déjà aux valeurs par défaut")
        self._update_context_hint()
        if not self._loading:
            self.changed.emit()

    @staticmethod
    def _apply_field(widget, kind: str, raw, default) -> None:
        if kind == "int":
            try:
                widget.setValue(int(raw))
            except (TypeError, ValueError):
                widget.setValue(int(default))
        elif kind == "float":
            try:
                widget.setValue(float(raw))
            except (TypeError, ValueError):
                widget.setValue(float(default))
        elif kind == "bool":
            widget.setChecked(bool(raw))
        elif kind == "combo":
            idx = widget.findData(raw)
            if idx < 0:
                idx = widget.findData(default)
            widget.setCurrentIndex(max(0, idx))
        elif kind == "text":
            widget.setText("" if raw is None else str(raw))

    @staticmethod
    def _read_field(widget, kind: str):
        if kind == "int":
            return int(widget.value())
        if kind == "float":
            return float(widget.value())
        if kind == "bool":
            return bool(widget.isChecked())
        if kind == "combo":
            return widget.currentData()
        return widget.text()  # kind == "text"
