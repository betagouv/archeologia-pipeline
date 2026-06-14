"""Contexte d'exécution typé du pipeline.

Avant ce module, ``RunContext`` était un transport de ``Dict[str, Any]``
non typés : chaque consommateur faisait son propre ``.get("X", default)``
avec sa propre validation, et l'IDE ne pouvait rien proposer en
auto-complétion.

Cette version expose des dataclasses gelées par catégorie
(``FilesConfig``, ``ProductsConfig``, ``ProcessingConfig``,
``CvConfig``) — les runners et services consomment des champs typés
(``ctx.products.svf``, ``ctx.processing.max_workers``) au lieu de
``ctx.products_cfg.get("SVF", False)``.

Deux champs restent volontairement en ``Dict[str, Any]`` :

- ``rvt_params`` — consommé tel quel par les générateurs RVT (qui
  attendent un dict de paramètres SAGA bruts) ;
- ``ui_config`` — re-sérialisé tel quel dans ``metadata.json`` à la
  fin du pipeline.

Les typer apporterait de la friction sans bénéfice tant que les
modules en aval continuent d'attendre des dicts.

``CvConfig`` expose à la fois ``enabled``, ``runs`` (liste typée
extraite) et ``raw`` (dict d'origine) car les utilitaires
``pipeline.cv.class_utils`` consomment encore le dict brut. ``raw``
servira de seam à ``ModelProfile`` (V2.2) pour l'éliminer
progressivement.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Sous-configurations typées
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FilesConfig:
    """Chemins d'entrée/sortie du run.

    ``data_mode`` est dupliqué dans ``RunContext.mode`` (alias de
    convenance pour la dispatch sur les runners) — la source de vérité
    reste ce champ.
    """

    data_mode: str = ""
    output_dir: Optional[Path] = None
    input_file: Optional[Path] = None       # mode ign_laz (URLs ou polygone)
    local_laz_dir: Optional[Path] = None    # mode local_laz
    existing_mnt_dir: Optional[Path] = None  # mode existing_mnt
    existing_rvt_dir: Optional[Path] = None  # mode existing_rvt
    # CRS déclaré par l'utilisateur (authid, ex. "EPSG:2154"), utilisé comme repli
    # pour les entrées sans CRS dans les métadonnées (ex. .asc). None = non déclaré.
    declared_crs: Optional[str] = None

    def input_path_for_mode(self) -> Optional[Path]:
        """Renvoie le chemin d'entrée pertinent pour le mode courant."""
        return {
            "ign_laz": self.input_file,
            "local_laz": self.local_laz_dir,
            "existing_mnt": self.existing_mnt_dir,
            "existing_rvt": self.existing_rvt_dir,
        }.get(self.data_mode)


# Codes des indices de visualisation RVT (tous dérivés du MNT). Source unique :
# toute logique « au moins un indice » / « faut-il un MNT » doit s'appuyer
# dessus, sinon on réintroduit le bug d'un indice oublié dans une liste codée
# en dur (HS absent de la validation, SLRM absent de needs_mnt…).
_VISUALIZATION_PRODUCTS: Tuple[str, ...] = (
    "HS", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT", "MSTP", "CVAT",
)
_ALL_PRODUCTS: Tuple[str, ...] = ("MNT", "DENSITE", "COUVERTURE", *_VISUALIZATION_PRODUCTS)


@dataclass(frozen=True)
class ProductsConfig:
    """Drapeaux d'activation des produits visualisation.

    Les noms suivent le code court historique (``M_HS``, ``SVF``…).
    Pour un wording humain, voir :data:`app.user_narrator.PRODUCT_LABELS`.
    """

    MNT: bool = True
    DENSITE: bool = False
    COUVERTURE: bool = False
    HS: bool = False
    M_HS: bool = False
    SVF: bool = False
    SLO: bool = False
    LD: bool = False
    SLRM: bool = False
    VAT: bool = False
    MSTP: bool = False
    CVAT: bool = False

    def active(self) -> List[str]:
        """Liste des produits activés (pour les logs/metadata)."""
        return [k for k in _ALL_PRODUCTS if getattr(self, k)]

    def has_visualization_index(self) -> bool:
        """Vrai si au moins un indice de visualisation RVT est actif."""
        return any(getattr(self, k) for k in _VISUALIZATION_PRODUCTS)

    def needs_mnt(self) -> bool:
        """Vrai si on doit calculer un MNT (soit demandé directement,
        soit comme dépendance d'un indice de visualisation)."""
        return self.MNT or self.has_visualization_index()

    def needs_tile_processing(self) -> bool:
        """Vrai si la boucle de traitement par dalle a du travail.

        Garde du runner LAZ : ``needs_mnt()`` seul sauterait silencieusement
        tout le traitement pour une config « DENSITE/COUVERTURE seule ».
        """
        return self.needs_mnt() or self.DENSITE or self.COUVERTURE

    def as_dict(self) -> Dict[str, bool]:
        """Vue dict (pour les call-sites qui en attendent encore un)."""
        return {k: getattr(self, k) for k in _ALL_PRODUCTS}


@dataclass(frozen=True)
class ProcessingConfig:
    """Paramètres de traitement (résolutions, parallélisme, formats)."""

    products: ProductsConfig = field(default_factory=ProductsConfig)
    max_workers: int = 4
    tile_overlap: float = 5.0
    mnt_resolution: float = 0.5
    density_resolution: float = 1.0
    coverage_threshold_percent: float = 30.0
    filter_expression: str = (
        "Classification = 2 OR Classification = 6 OR Classification = 66 "
        "OR Classification = 67 OR Classification = 9"
    )
    # Conservés en dict : leurs sous-champs (booléens, listes) sont lus
    # tels quels par les generators downstream sans validation interne.
    output_structure: Dict[str, Any] = field(default_factory=dict)
    output_formats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CvConfig:
    """Configuration Computer Vision.

    ``raw`` est conservé pour les utilitaires (``resolve_cv_runs``,
    ``_build_global_class_color_map``…) qui consomment encore le dict
    brut. Il sera supprimé une fois ``ModelProfile`` (V2.2) totalement
    déployé sur les call-sites concernés.
    """

    enabled: bool = False
    runs: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Contexte d'exécution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RunContext:
    """Contexte typé d'un run du pipeline.

    Construit une seule fois par :func:`build_run_context` à partir du
    ``config.json`` (ou de la persistance UI). Frozen : un runner ne
    peut pas le modifier en cours de route.

    Champs ``mode`` et ``output_dir`` sont des alias de
    ``files.data_mode`` / ``files.output_dir`` — gardés en niveau
    racine pour la dispatch ``get_runner(ctx.mode)``.
    """

    mode: str
    output_dir: Optional[Path]
    files: FilesConfig
    processing: ProcessingConfig
    cv: CvConfig
    rvt_params: Dict[str, Any]
    ui_config: Dict[str, Any]


# ----------------------------------------------------------------------
# Constructeur
# ----------------------------------------------------------------------
def _to_path(value: Any) -> Optional[Path]:
    """Coerce une valeur arbitraire en ``Path`` ou ``None``."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    return Path(s) if s else None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# Helpers de coercition + clipping
# ----------------------------------------------------------------------
# Motivation : ``dict.get(key, default)`` ne renvoie le défaut QUE si la
# clé est absente. Si elle existe avec valeur ``""``, ``0``, ``-1``, le
# pipeline reçoit cette valeur dégénérée et :
#
# - PDAL avec ``RESOLUTION=0`` produit un raster invalide ;
# - ``ThreadPoolExecutor(max_workers=0)`` lève ``ValueError`` ;
# - ``confidence_threshold=-1`` accepte tout, génère des faux positifs ;
# - ``tile_overlap=-5`` calcule des bounds spatiales inversées.
#
# Ces helpers garantissent que la valeur sortante est *saine* — saine
# au sens "ne casse pas le pipeline en aval" — sans rien logger : un
# défaut sain est silencieux et idempotent. Les valeurs vraiment hors
# normes (mais non-cassantes) sont signalées séparément par
# :func:`validate_run_context` en warnings.

def _coerce_positive_float(value: Any, default: float, *, exclusive: bool = True) -> float:
    """Force un float strictement positif (par défaut), sinon ``default``.

    ``exclusive=True`` : la valeur doit être > 0. ``exclusive=False`` :
    la valeur doit être >= 0. Adapté pour les résolutions PDAL qui
    refusent une valeur nulle.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if exclusive:
        return v if v > 0.0 else default
    return v if v >= 0.0 else default


def _coerce_unit_interval(value: Any, default: float) -> float:
    """Force un float dans ``[0.0, 1.0]``, défaut sinon.

    Adapté aux probabilités / seuils CV (``confidence_threshold``,
    ``iou_threshold``).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _coerce_int_min(value: Any, default: int, min_value: int) -> int:
    """Force un int >= ``min_value``, défaut si cast échoue ou < min."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return default
    return v if v >= min_value else default


def _build_files_config(files_dict: Dict[str, Any]) -> FilesConfig:
    return FilesConfig(
        data_mode=str(files_dict.get("data_mode") or "").strip(),
        output_dir=_to_path(files_dict.get("output_dir")),
        input_file=_to_path(files_dict.get("input_file")),
        local_laz_dir=_to_path(files_dict.get("local_laz_dir")),
        existing_mnt_dir=_to_path(files_dict.get("existing_mnt_dir")),
        existing_rvt_dir=_to_path(files_dict.get("existing_rvt_dir")),
        declared_crs=(str(files_dict.get("declared_crs")).strip() or None)
        if files_dict.get("declared_crs")
        else None,
    )


def _build_products_config(products_dict: Dict[str, Any]) -> ProductsConfig:
    """Construit ``ProductsConfig`` à partir d'un dict ``{"MNT": True, …}``.

    Les clés inconnues sont silencieusement ignorées (compat ascendante
    si ``products_cfg`` contient des champs hérités).
    """
    if not isinstance(products_dict, dict):
        products_dict = {}
    return ProductsConfig(
        MNT=bool(products_dict.get("MNT", True)),
        DENSITE=bool(products_dict.get("DENSITE", False)),
        COUVERTURE=bool(products_dict.get("COUVERTURE", False)),
        HS=bool(products_dict.get("HS", False)),
        M_HS=bool(products_dict.get("M_HS", False)),
        SVF=bool(products_dict.get("SVF", False)),
        SLO=bool(products_dict.get("SLO", False)),
        LD=bool(products_dict.get("LD", False)),
        SLRM=bool(products_dict.get("SLRM", False)),
        VAT=bool(products_dict.get("VAT", False)),
        MSTP=bool(products_dict.get("MSTP", False)),
        CVAT=bool(products_dict.get("CVAT", False)),
    )


def _build_processing_config(processing_dict: Dict[str, Any]) -> ProcessingConfig:
    products_dict = processing_dict.get("products") or {}

    # ``dict.get(key, default)`` ne renvoie le défaut QUE si la clé est
    # absente, pas si elle vaut "". Or last_ui_config peut sérialiser un
    # filter_expression vidé par l'utilisateur — auquel cas PDAL ne
    # filtre RIEN et le MNT inclut toute la canopée végétale (DSM au
    # lieu d'un DTM bare-earth). On applique donc explicitement le
    # défaut quand la valeur est vide ou blanche.
    DEFAULT_FILTER = (
        "Classification = 2 OR Classification = 6 OR Classification = 66 "
        "OR Classification = 67 OR Classification = 9"
    )
    filter_expr = str(processing_dict.get("filter_expression") or "").strip()
    if not filter_expr:
        filter_expr = DEFAULT_FILTER

    # Seuil « zones mal couvertes » (produit COUVERTURE) : borné 5–95 %.
    cov_thr = _coerce_positive_float(
        processing_dict.get("coverage_threshold_percent", 30.0), 30.0
    )
    cov_thr = min(95.0, max(5.0, cov_thr))

    return ProcessingConfig(
        products=_build_products_config(products_dict),
        # max_workers >= 1 : 0 ferait planter ThreadPoolExecutor.
        max_workers=_coerce_int_min(processing_dict.get("max_workers", 4), 4, min_value=1),
        # tile_overlap >= 0 : négatif inverserait les bounds spatiales.
        # On accepte 0 (warning émis par validate_run_context).
        tile_overlap=_coerce_positive_float(
            processing_dict.get("tile_overlap", 5), 5.0, exclusive=False
        ),
        # Résolutions PDAL : doivent être > 0 (pas de raster à RESOLUTION=0).
        mnt_resolution=_coerce_positive_float(processing_dict.get("mnt_resolution", 0.5), 0.5),
        density_resolution=_coerce_positive_float(processing_dict.get("density_resolution", 1.0), 1.0),
        coverage_threshold_percent=cov_thr,
        filter_expression=filter_expr,
        output_structure=dict(processing_dict.get("output_structure") or {}),
        output_formats=dict(processing_dict.get("output_formats") or {}),
    )


def _normalize_cv_run(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce les paramètres numériques d'un run CV vers des plages saines.

    Modifie une COPIE du dict — on garde l'original intact pour ne pas
    perturber les call-sites qui consomment encore ``cv.raw["runs"]``.
    Les autres clés (``model``, ``target_rvt``, ``selected_classes``…)
    sont préservées tel quel.
    """
    out = dict(run_dict)
    if "confidence_threshold" in out:
        out["confidence_threshold"] = _coerce_unit_interval(out["confidence_threshold"], 0.3)
    if "iou_threshold" in out:
        out["iou_threshold"] = _coerce_unit_interval(out["iou_threshold"], 0.5)
    if "min_area_m2" in out:
        # min_area_m2 = 0 reste valide (= pas de filtre d'aire).
        out["min_area_m2"] = _coerce_positive_float(out["min_area_m2"], 0.0, exclusive=False)
    return out


def _build_cv_config(cv_dict: Dict[str, Any]) -> CvConfig:
    """Construit ``CvConfig`` à partir du dict ``computer_vision``.

    Convention ``selected_classes`` (par run) :

    - ``None`` ou clé absente : toutes les classes du modèle sont actives.
    - ``[]`` (liste vide) : court-circuit explicite, le run est ignoré
      (cf. :func:`pipeline.cv.runner.run_cv_on_folder`).
    - ``["class_a", …]`` : filtre explicite sur les classes nommées.
    """
    if not isinstance(cv_dict, dict):
        cv_dict = {}
    runs = cv_dict.get("runs")
    if not isinstance(runs, list):
        runs = []
    typed_runs = [_normalize_cv_run(r) for r in runs if isinstance(r, dict)]

    # Aussi normaliser les seuils globaux dans cv.raw, pour les
    # consommateurs qui lisent encore le dict brut (ex. external_runner,
    # conversion_shp).
    raw = dict(cv_dict)
    if "confidence_threshold" in raw:
        raw["confidence_threshold"] = _coerce_unit_interval(raw["confidence_threshold"], 0.3)
    if "iou_threshold" in raw:
        raw["iou_threshold"] = _coerce_unit_interval(raw["iou_threshold"], 0.5)

    return CvConfig(
        enabled=bool(cv_dict.get("enabled", False)),
        runs=typed_runs,
        raw=raw,
    )


def validate_run_context(ctx: RunContext) -> Tuple[List[str], List[str]]:
    """Vérifie que ``ctx`` est exécutable pour son ``mode``.

    Retourne ``(errors, warnings)`` :

    - **errors** (bloquantes) : configuration inexécutable. L'UI
      grise le bouton « Lancer », ``PipelineController.run`` abort.
      Liste **complète** sans short-circuit pour que l'utilisateur
      corrige tout d'un coup.
    - **warnings** (non bloquantes) : config exécutable mais avec
      des choix qui peuvent surprendre — un seuil clustering qui
      masque toutes les détections, un overlap à 0 qui peut créer
      des artefacts en bordure, un run CV explicitement court-circuité…
      Le pipeline continue mais le ``slog`` les trace.

    Cette fonction concentre les vérifications qui étaient dupliquées
    dans chaque runner (``if output_dir is None``, ``if existing_X_dir
    is None``…). Les vérifications de dépendances système (CLI tools,
    QGIS Processing) restent du ressort de :func:`pipeline.preflight`.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not ctx.mode:
        errors.append("Aucun mode d'acquisition (data_mode) configuré")

    if ctx.output_dir is None:
        errors.append("Aucun dossier de sortie n'est configuré")

    if ctx.mode == "ign_laz":
        if ctx.files.input_file is None:
            errors.append("Mode IGN sélectionné mais aucun fichier de zone/liste n'est configuré")
        elif not ctx.files.input_file.exists():
            errors.append(f"Fichier IGN introuvable : {ctx.files.input_file}")

    elif ctx.mode == "local_laz":
        if ctx.files.local_laz_dir is None:
            errors.append("Mode LAZ local sélectionné mais aucun dossier nuages locaux n'est configuré")
        elif not ctx.files.local_laz_dir.exists():
            errors.append(f"Dossier LAZ local introuvable : {ctx.files.local_laz_dir}")

    elif ctx.mode == "existing_mnt":
        if ctx.files.existing_mnt_dir is None:
            errors.append("Mode MNT existant sélectionné mais aucun dossier MNT n'est configuré")
        elif not ctx.files.existing_mnt_dir.exists():
            errors.append(f"Dossier MNT introuvable : {ctx.files.existing_mnt_dir}")

    elif ctx.mode == "existing_rvt":
        if ctx.files.existing_rvt_dir is None:
            errors.append("Mode RVT existant sélectionné mais aucun dossier RVT n'est configuré")
        elif not ctx.files.existing_rvt_dir.exists():
            errors.append(f"Dossier RVT introuvable : {ctx.files.existing_rvt_dir}")

    elif ctx.mode:  # mode renseigné mais inconnu
        errors.append(f"Mode d'acquisition inconnu : {ctx.mode!r}")

    # Règle métier transverse : il faut au moins un produit actif, sauf
    # en mode existing_rvt qui ne calcule rien (lance juste la CV sur
    # les RVT déjà fournis).
    if ctx.mode in ("ign_laz", "local_laz", "existing_mnt"):
        if ctx.mode == "existing_mnt":
            # En existing_mnt le MNT est déjà fourni : seul un indice RVT
            # constitue un calcul à faire (cf. _VISUALIZATION_PRODUCTS, source
            # unique — ne jamais réénumérer les indices à la main ici).
            if not ctx.processing.products.has_visualization_index():
                errors.append("Cochez au moins un indice de visualisation")
        elif not ctx.processing.products.active():
            errors.append("Cochez au moins un produit à générer")

    # ── Warnings (non bloquants) ─────────────────────────────────────
    # tile_overlap=0 : pas de marge entre dalles. Calcul réussit mais
    # les bords de chaque dalle peuvent montrer des artefacts (effets
    # de bord PDAL/RVT sur les pixels en bordure).
    if ctx.processing.tile_overlap == 0 and ctx.mode in ("ign_laz", "local_laz"):
        warnings.append(
            "Tile overlap à 0 : pas de marge entre dalles, possibles artefacts en bordure."
        )

    # CV runs : signaler ceux qui seront court-circuités (sélection
    # explicite de zéro classe) — pratique pour ne pas se demander
    # pourquoi rien n'est détecté.
    for i, run in enumerate(ctx.cv.runs, start=1):
        sel = run.get("selected_classes")
        if isinstance(sel, list) and len(sel) == 0:
            model_name = run.get("model") or f"run #{i}"
            warnings.append(
                f"Run CV « {model_name} » : aucune classe sélectionnée — sera court-circuité."
            )

    # Cohérence cross-config : si le seuil global de confiance est
    # supérieur aux ``min_confidence`` configurés dans args.yaml du
    # modèle (clustering), aucune détection n'atteindra le clustering.
    # Détection best-effort — si on ne peut pas charger args.yaml, on
    # ne lève pas l'alerte (silencieux).
    if ctx.cv.enabled:
        for i, run in enumerate(ctx.cv.runs, start=1):
            runtime_conf = float(run.get("confidence_threshold", 0.3) or 0.3)
            cluster_min = _peek_clustering_min_confidence(run)
            if cluster_min is not None and runtime_conf > cluster_min:
                model_name = run.get("model") or f"run #{i}"
                warnings.append(
                    f"Run CV « {model_name} » : seuil de confiance global "
                    f"({runtime_conf:.2f}) > min_confidence clustering ({cluster_min:.2f}) — "
                    "le clustering ne verra aucune détection."
                )

    return errors, warnings


def _peek_clustering_min_confidence(run_dict: Dict[str, Any]) -> Optional[float]:
    """Extrait le ``min_confidence`` clustering depuis le ``args.yaml``
    du modèle, ou ``None`` si non chargeable.

    Lookup best-effort pour la cohérence cross-config — on ne fait pas
    planter ``validate_run_context`` si le modèle n'a pas d'args.yaml
    (modèle externe, fichier manquant, etc.).
    """
    try:
        from pathlib import Path as _Path
        from ..pipeline.cv.model_config import load_clustering_config_from_model

        weights = run_dict.get("weights_path") or run_dict.get("model")
        if not weights:
            return None
        weights_path = _Path(weights)
        if not weights_path.exists():
            return None
        configs = load_clustering_config_from_model(weights_path)
        if not configs:
            return None
        # On retourne le SEUIL LE PLUS BAS configuré : si une seule
        # règle clustering accepte des confidences faibles, le bug
        # "rien n'arrive au clustering" ne s'applique pas.
        mins = [
            float(c.get("min_confidence"))
            for c in configs
            if isinstance(c, dict) and c.get("min_confidence") is not None
        ]
        return min(mins) if mins else None
    except Exception:
        return None


def build_run_context(config: Dict[str, Any]) -> RunContext:
    """Construit un :class:`RunContext` typé à partir du ``config.json``.

    Tolère les configs partielles ou malformées : les valeurs absentes
    sont remplacées par les défauts des dataclasses, les valeurs de
    mauvais type tombent sur le défaut via ``_safe_int`` / ``_safe_float``.
    """
    cfg = config if isinstance(config, dict) else {}
    app_cfg = cfg.get("app") or {}
    files_dict = (app_cfg.get("files") or {}) if isinstance(app_cfg, dict) else {}
    if not isinstance(files_dict, dict):
        files_dict = {}

    processing_dict = cfg.get("processing") or {}
    if not isinstance(processing_dict, dict):
        processing_dict = {}

    cv_dict = cfg.get("computer_vision") or {}

    rvt_params = cfg.get("rvt_params") or {}
    if not isinstance(rvt_params, dict):
        rvt_params = {}

    files = _build_files_config(files_dict)
    processing = _build_processing_config(processing_dict)
    cv = _build_cv_config(cv_dict)

    # COUVERTURE exige le nuage de points : un MNT livré est déjà interpolé,
    # l'information « où étaient les points » n'existe plus (vrai pour les
    # trois layouts standard/small/large). Neutralisé hors modes LAZ — le
    # runner existing_mnt logge l'indisponibilité depuis la config brute.
    if processing.products.COUVERTURE and files.data_mode not in ("ign_laz", "local_laz"):
        processing = replace(
            processing, products=replace(processing.products, COUVERTURE=False)
        )

    return RunContext(
        mode=files.data_mode,
        output_dir=files.output_dir,
        files=files,
        processing=processing,
        cv=cv,
        rvt_params=rvt_params,
        # Snapshot indépendant : le worker pipeline tourne dans un thread
        # séparé du main Qt qui peut continuer à muter le `_config` du
        # dialog (autosave, signaux). Sans cette deepcopy, ctx.ui_config
        # observe ces mutations en cours de run.
        ui_config=copy.deepcopy(cfg),
    )
