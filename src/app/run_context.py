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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def input_path_for_mode(self) -> Optional[Path]:
        """Renvoie le chemin d'entrée pertinent pour le mode courant."""
        return {
            "ign_laz": self.input_file,
            "local_laz": self.local_laz_dir,
            "existing_mnt": self.existing_mnt_dir,
            "existing_rvt": self.existing_rvt_dir,
        }.get(self.data_mode)


@dataclass(frozen=True)
class ProductsConfig:
    """Drapeaux d'activation des produits visualisation.

    Les noms suivent le code court historique (``M_HS``, ``SVF``…).
    Pour un wording humain, voir :data:`app.user_narrator.PRODUCT_LABELS`.
    """

    MNT: bool = True
    DENSITE: bool = False
    M_HS: bool = False
    SVF: bool = False
    SLO: bool = False
    LD: bool = False
    SLRM: bool = False
    VAT: bool = False

    def active(self) -> List[str]:
        """Liste des produits activés (pour les logs/metadata)."""
        return [k for k in ("MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT") if getattr(self, k)]

    def needs_mnt(self) -> bool:
        """Vrai si on doit calculer un MNT (soit demandé directement,
        soit comme dépendance d'un indice de visualisation)."""
        return self.MNT or self.M_HS or self.SVF or self.SLO or self.LD or self.VAT

    def as_dict(self) -> Dict[str, bool]:
        """Vue dict (pour les call-sites qui en attendent encore un)."""
        return {k: getattr(self, k) for k in ("MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT")}


@dataclass(frozen=True)
class ProcessingConfig:
    """Paramètres de traitement (résolutions, parallélisme, formats)."""

    products: ProductsConfig = field(default_factory=ProductsConfig)
    max_workers: int = 4
    tile_overlap: float = 5.0
    mnt_resolution: float = 0.5
    density_resolution: float = 1.0
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


def _build_files_config(files_dict: Dict[str, Any]) -> FilesConfig:
    return FilesConfig(
        data_mode=str(files_dict.get("data_mode") or "").strip(),
        output_dir=_to_path(files_dict.get("output_dir")),
        input_file=_to_path(files_dict.get("input_file")),
        local_laz_dir=_to_path(files_dict.get("local_laz_dir")),
        existing_mnt_dir=_to_path(files_dict.get("existing_mnt_dir")),
        existing_rvt_dir=_to_path(files_dict.get("existing_rvt_dir")),
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
        M_HS=bool(products_dict.get("M_HS", False)),
        SVF=bool(products_dict.get("SVF", False)),
        SLO=bool(products_dict.get("SLO", False)),
        LD=bool(products_dict.get("LD", False)),
        SLRM=bool(products_dict.get("SLRM", False)),
        VAT=bool(products_dict.get("VAT", False)),
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

    return ProcessingConfig(
        products=_build_products_config(products_dict),
        max_workers=_safe_int(processing_dict.get("max_workers", 4), 4),
        tile_overlap=_safe_float(processing_dict.get("tile_overlap", 5), 5.0),
        mnt_resolution=_safe_float(processing_dict.get("mnt_resolution", 0.5), 0.5),
        density_resolution=_safe_float(processing_dict.get("density_resolution", 1.0), 1.0),
        filter_expression=filter_expr,
        output_structure=dict(processing_dict.get("output_structure") or {}),
        output_formats=dict(processing_dict.get("output_formats") or {}),
    )


def _build_cv_config(cv_dict: Dict[str, Any]) -> CvConfig:
    if not isinstance(cv_dict, dict):
        cv_dict = {}
    runs = cv_dict.get("runs")
    if not isinstance(runs, list):
        runs = []
    return CvConfig(
        enabled=bool(cv_dict.get("enabled", False)),
        runs=[r for r in runs if isinstance(r, dict)],
        raw=cv_dict,
    )


def validate_run_context(ctx: RunContext) -> List[str]:
    """Vérifie que ``ctx`` est exécutable pour son ``mode``.

    Retourne la liste **complète** des erreurs trouvées (pas de
    short-circuit) : l'utilisateur voit d'un coup tout ce qu'il doit
    corriger plutôt que de relancer après chaque correction.

    Cette fonction concentre les vérifications qui étaient dupliquées
    dans chaque runner (``if output_dir is None``, ``if existing_X_dir
    is None``…). Les vérifications de dépendances système (CLI tools,
    QGIS Processing) restent du ressort de :func:`pipeline.preflight`.
    """
    errors: List[str] = []

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
    # les RVT déjà fournis). En existing_mnt on ne peut pas demander
    # DENSITE ni MNT (les LAZ ne sont pas là), mais au moins un index
    # de visualisation doit être coché.
    if ctx.mode in ("ign_laz", "local_laz", "existing_mnt"):
        if ctx.mode == "existing_mnt":
            # MNT et DENSITE n'ont pas de sens (pas de LAZ) — on
            # n'exige qu'un index de visualisation.
            visu_active = (
                ctx.processing.products.M_HS
                or ctx.processing.products.SVF
                or ctx.processing.products.SLO
                or ctx.processing.products.LD
                or ctx.processing.products.SLRM
                or ctx.processing.products.VAT
            )
            if not visu_active:
                errors.append("Cochez au moins un indice de visualisation")
        elif not ctx.processing.products.active():
            errors.append("Cochez au moins un produit à générer")

    return errors


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

    return RunContext(
        mode=files.data_mode,
        output_dir=files.output_dir,
        files=files,
        processing=processing,
        cv=cv,
        rvt_params=rvt_params,
        ui_config=cfg,
    )
