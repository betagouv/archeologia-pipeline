"""Source unique de vérité pour la configuration d'un modèle CV.

Avant ce module, les paramètres d'un modèle (SAHI, clustering, post-processing,
classes, couleurs, type RF-DETR vs YOLO, métadonnées du sidecar ``.json``)
étaient lus indépendamment par 4-5 fonctions, chacune relisant ``args.yaml``
ou le ``.json`` à la volée. Cela conduisait à :

- de multiples lectures du même ``args.yaml`` par run ;
- une priorité floue entre ``cv_config``, ``args.yaml`` et le ``.json`` du
  modèle pour ``confidence_threshold`` notamment ;
- un éparpillement de la définition de la « configuration d'un modèle »
  dans deux modules ``class_utils`` / ``model_config`` qui se ré-exportent.

:class:`ModelProfile` charge tout d'un coup à partir du chemin des poids
et expose des sous-dataclasses immuables. Les anciens loaders dans
``model_config.py`` et ``class_utils.py`` continuent d'exister pour la
rétrocompatibilité — la migration des call sites se fait progressivement.

Le module est pure-Python (pas de QGIS, pas de shapely). La lecture YAML
utilise un import différé pour ne pas pénaliser les usages qui ne
chargent pas effectivement de profil.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

# Tâches « segmentation » du sidecar ``task`` — même vocabulaire que le gate
# de ``conversion_shp.create_shapefile_from_detections`` (une tâche connue hors
# de cet ensemble est un modèle bbox / object detection).
_SEGMENTATION_TASKS = frozenset(
    {"instance_segmentation", "semantic_segmentation", "segment"}
)


@dataclass(frozen=True)
class SahiConfig:
    """Paramètres SAHI (slicing à l'inférence)."""
    slice_height: int = 640
    slice_width: int = 640
    overlap_ratio: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_height": self.slice_height,
            "slice_width": self.slice_width,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(frozen=True)
class ClusteringRule:
    """Une règle de clustering DBSCAN spatial post-détection (type: dbscan)."""
    target_classes: Tuple[str, ...]
    min_confidence: float
    min_confidence_extend: float
    min_cluster_size: int
    min_samples: int
    eps_m: float
    output_class_name: str
    output_geometry: str
    buffer_m: float
    min_area_m2: float
    concave_ratio: float
    confidence_weight: float
    type: str = "dbscan"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target_classes": list(self.target_classes),
            "min_confidence": self.min_confidence,
            "min_confidence_extend": self.min_confidence_extend,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "eps_m": self.eps_m,
            "output_class_name": self.output_class_name,
            "output_geometry": self.output_geometry,
            "buffer_m": self.buffer_m,
            "min_area_m2": self.min_area_m2,
            "concave_ratio": self.concave_ratio,
            "confidence_weight": self.confidence_weight,
        }


@dataclass(frozen=True)
class EnclosureRule:
    """Règle « enclosure » : fermeture vectorielle (buffer±T/2) + scoring.

    Détecte des enclos (circuits fermés/quasi fermés) à partir des détections
    des ``target_classes`` — voir ``pipeline.cv.enclosure``. Distances en
    mètres (Lambert-93 métrique).
    """
    target_classes: Tuple[str, ...]
    output_class_name: str
    gap_tolerance_m: float
    min_area_m2: float
    max_area_m2: float
    min_closure: float
    max_elongation: float
    min_ancrage: float
    min_confidence: float
    max_isolement: float = 0.5
    min_rectangularite: float = 0.0
    generator: str = "auto"
    mode_calibration: bool = False
    type: str = "enclosure"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target_classes": list(self.target_classes),
            "output_class_name": self.output_class_name,
            "generator": self.generator,
            "gap_tolerance_m": self.gap_tolerance_m,
            "min_area_m2": self.min_area_m2,
            "max_area_m2": self.max_area_m2,
            "min_closure": self.min_closure,
            "max_elongation": self.max_elongation,
            "min_ancrage": self.min_ancrage,
            "max_isolement": self.max_isolement,
            "min_rectangularite": self.min_rectangularite,
            "min_confidence": self.min_confidence,
            "mode_calibration": self.mode_calibration,
        }


@dataclass(frozen=True)
class AlignmentRule:
    """Règle « alignment » : bandes directionnelles à brins multiples.

    Détecte les axes linéaires (voies anciennes…) — enfilades de détections
    co-orientées dans une bande étroite — voir ``pipeline.cv.alignment``.
    Distances en mètres, angles en degrés (azimut modulo 180°).
    """
    target_classes: Tuple[str, ...]
    output_class_name: str
    band_width_m: float
    angle_tolerance_deg: float
    min_length_m: float
    max_gap_m: float
    min_coverage: float
    min_sources: int
    min_confidence: float
    type: str = "alignment"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target_classes": list(self.target_classes),
            "output_class_name": self.output_class_name,
            "band_width_m": self.band_width_m,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "min_length_m": self.min_length_m,
            "max_gap_m": self.max_gap_m,
            "min_coverage": self.min_coverage,
            "min_sources": self.min_sources,
            "min_confidence": self.min_confidence,
        }


@dataclass(frozen=True)
class PostprocessConfig:
    """Activation des étapes de post-traitement géométrique.

    ``merge_buffer_m`` est la VRAIE distance (mètres) en deçà de laquelle deux
    polygones de même classe sont fusionnés (cf. ``postprocess_geo_detections``,
    prédicat ``dwithin``). 0,5 m par défaut (~1 px à 0,5 m/px).

    ``overlap_strategy`` pilote l'étape de suppression des superpositions
    (``remove_overlaps``) :

    - ``"difference"`` (défaut segmentation, historique) : découpe le polygone
      le moins confiant le long du contour de l'autre (``geom.difference``).
      Pour un modèle mono-classe (cratères) ce découpage FABRIQUE des
      artefacts — anneau troué (petit imbriqué) ou arête droite partagée
      (accolés). Il ne supprime JAMAIS un doublon, d'où le défaut
      ``"relation"`` pour les modèles bbox (cf. :func:`_parse_postprocess`) :
      le halo inter-dalles fait détecter le même objet par plusieurs dalles.
    - ``"relation"`` : pour les détections de MÊME classe, on raisonne en
      confinement (IoS = aire intersection / aire du plus petit) — si
      IoS ≥ ``overlap_ios_threshold`` on FUSIONNE par union (l'union absorbe le
      petit dans le grand sans anneau, et soude deux fragments fortement
      chevauchants sans arête). Les superpositions INTER-classes restent gérées
      par ``difference`` (un objet d'une classe rogne celui d'une autre).
    """
    merge_adjacent: bool = True
    remove_overlaps: bool = True
    merge_buffer_m: float = 0.5
    overlap_strategy: str = "difference"
    overlap_ios_threshold: float = 0.5
    overlap_min_area_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merge_adjacent": self.merge_adjacent,
            "remove_overlaps": self.remove_overlaps,
            "merge_buffer_m": self.merge_buffer_m,
            "overlap_strategy": self.overlap_strategy,
            "overlap_ios_threshold": self.overlap_ios_threshold,
            "overlap_min_area_ratio": self.overlap_min_area_ratio,
        }


@dataclass(frozen=True)
class ModelProfile:
    """Configuration complète d'un modèle CV, chargée une seule fois.

    Champs principaux :

    - ``weights_path`` : chemin vers ``best.onnx`` (ou ``.pt`` si pas
      d'ONNX disponible).
    - ``model_dir`` : dossier racine du modèle (parent de ``weights/``).
    - ``class_names`` : noms des classes (0-indexés). ``None`` si fichier
      absent.
    - ``class_colors`` : indices de couleurs par classe (depuis
      ``args.yaml:class_colors``). ``None`` si absent.
    - ``sahi`` : configuration SAHI (slice + overlap).
    - ``clustering`` : tuple de :class:`ClusteringRule` extraits de
      ``args.yaml:clustering``.
    - ``postprocess`` : flags merge/remove du post-traitement géométrique.
    - ``is_rfdetr`` : True si ``args.yaml:model`` contient ``rf-detr`` /
      ``rfdetr`` (impacte le décalage des class IDs).
    - ``args_yaml`` : contenu brut du ``args.yaml`` (vide si absent).
    - ``metadata`` : contenu brut du ``.json`` sidecar (vide si absent).

    Pour récupérer le seuil de confiance effectif d'un run, voir
    :meth:`effective_confidence_threshold` qui implémente la priorité
    historique : ``run_override > metadata > arg_default``.
    """

    weights_path: Path
    model_dir: Path
    class_names: Optional[Tuple[str, ...]]
    class_colors: Optional[Tuple[int, ...]]
    sahi: SahiConfig
    clustering: Tuple[ClusteringRule, ...]
    postprocess: PostprocessConfig
    is_rfdetr: bool
    args_yaml: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, weights_path: Union[str, Path]) -> "ModelProfile":
        """Charge un profil depuis le chemin des poids du modèle.

        Lit en une seule passe :

        - le ``args.yaml`` du dossier modèle (config SAHI, clustering,
          post-traitement, ``class_colors``, type ``model``).
        - le ``.json`` sidecar des poids (métadonnées du modèle :
          ``model_type``, ``task``, ``bg_bias``, ``confidence_threshold``,
          etc.).
        - les noms de classes depuis ``classes.txt`` / ``class_names.txt``
          / ``classes.json`` (fallback en cascade).

        Lève :class:`FileNotFoundError` si ``weights_path`` n'existe pas.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")

        model_dir = _resolve_model_dir(weights_path)
        args_yaml = _load_args_yaml(model_dir)
        metadata = _load_sidecar_metadata(weights_path)

        sahi = _parse_sahi(args_yaml)
        clustering = _parse_clustering(args_yaml)
        _task = metadata.get("task")
        postprocess = _parse_postprocess(
            args_yaml, task=str(_task) if _task is not None else None
        )
        class_colors = _parse_class_colors(args_yaml)
        is_rfdetr = _parse_is_rfdetr(args_yaml)
        class_names = _load_class_names(model_dir)

        return cls(
            weights_path=weights_path,
            model_dir=model_dir,
            class_names=tuple(class_names) if class_names is not None else None,
            class_colors=tuple(class_colors) if class_colors is not None else None,
            sahi=sahi,
            clustering=clustering,
            postprocess=postprocess,
            is_rfdetr=is_rfdetr,
            args_yaml=args_yaml,
            metadata=metadata,
        )

    def effective_confidence_threshold(self, run_default: float) -> float:
        """Retourne le seuil de confiance effectif pour un run.

        Priorité (de la plus forte à la plus faible) :

        1. ``metadata.confidence_threshold`` (sidecar ``.json`` du modèle)
        2. ``run_default`` (passé par l'appelant : cv_config, default…)

        Cette priorité reproduit la logique historique observée dans
        ``runner_inference.run_fallback_inference`` : les modèles peuvent
        forcer un seuil par défaut via leur sidecar même si l'UI a une
        valeur globale.
        """
        meta_conf = self.metadata.get("confidence_threshold")
        if meta_conf is not None:
            try:
                return float(meta_conf)
            except (TypeError, ValueError):
                pass
        return float(run_default)

    @property
    def task(self) -> Optional[str]:
        """Type de tâche déclaré dans le sidecar (``detection``,
        ``semantic_segmentation``, ``instance_segmentation``…)."""
        t = self.metadata.get("task")
        return str(t) if t is not None else None

    @property
    def model_type(self) -> Optional[str]:
        """Type de modèle déclaré dans le sidecar (``yolo``, ``rfdetr``,
        ``segformer``, ``smp``…)."""
        t = self.metadata.get("model_type")
        return str(t) if t is not None else None


# ----------------------------------------------------------------------
# Helpers de chargement (privés)
# ----------------------------------------------------------------------
def _resolve_model_dir(model_path: Path) -> Path:
    """Remonte au dossier racine du modèle (parent de ``weights/``).

    Convention Ultralytics / RF-DETR : ``<model>/weights/best.onnx``.
    """
    if model_path.is_file():
        if model_path.parent.name == "weights":
            return model_path.parent.parent
        return model_path.parent
    return model_path


def _load_args_yaml(model_dir: Path) -> Dict[str, Any]:
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML indisponible, args.yaml ignoré")
        return {}
    try:
        with open(args_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Erreur lecture {args_file}: {e}")
        return {}


def _load_sidecar_metadata(weights_path: Path) -> Dict[str, Any]:
    sidecar = weights_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Erreur lecture sidecar {sidecar}: {e}")
        return {}


def _parse_sahi(args_yaml: Dict[str, Any]) -> SahiConfig:
    sahi = args_yaml.get("sahi")
    if not isinstance(sahi, dict):
        return SahiConfig()
    try:
        raw_h = int(sahi.get("slice_height", 640))
        raw_w = int(sahi.get("slice_width", 640))
        raw_ov = float(sahi.get("overlap_ratio", 0.2))
        # Bornes (AUDIT v2 PARSE-12) : slice ≥ 32, overlap ∈ [0, 0.9] — un
        # overlap ≥ 1 ou un slice ≤ 0 gèle l'inférence en boucle infinie.
        cfg = SahiConfig(
            slice_height=max(32, raw_h),
            slice_width=max(32, raw_w),
            overlap_ratio=min(max(raw_ov, 0.0), 0.9),
        )
        if (cfg.slice_height, cfg.slice_width, cfg.overlap_ratio) != (raw_h, raw_w, raw_ov):
            logger.warning(
                f"SAHI hors bornes ({raw_h}x{raw_w}, overlap {raw_ov}) — "
                f"clampé à {cfg.slice_height}x{cfg.slice_width}, {cfg.overlap_ratio}"
            )
        return cfg
    except (TypeError, ValueError) as e:
        logger.warning(f"SAHI config invalide ({e}), valeurs par défaut")
        return SahiConfig()


def _parse_clustering(args_yaml: Dict[str, Any]) -> Tuple[Any, ...]:
    """Règles de synthèse typées : ClusteringRule (dbscan) ou EnclosureRule."""
    raw = args_yaml.get("clustering")
    if not raw:
        return tuple()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return tuple()

    rules: List[Any] = []
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        target = cfg.get("target_classes", cfg.get("target_class"))
        if isinstance(target, str):
            target = [target]
        if not isinstance(target, list) or not target:
            logger.warning("Clustering rule ignorée : target_classes manquant/invalide")
            continue
        rule_type = str(cfg.get("type", "dbscan")).strip().lower() or "dbscan"
        if rule_type == "enclosure":
            try:
                from .clustering_bounds import sanitize_clustering_rule

                # Défauts V3 calibrés campagne Bretagne (131 GT à parcellaire,
                # générateur auto) : F1 0,264, R 0,39, sentinelles fid30/fid34
                # publiées, plafond 1 ha (95 % des GT < 1,2 ha — les géants
                # sont des parcelles modernes, verdict terrain V2).
                sane = sanitize_clustering_rule(
                    {
                        "gap_tolerance_m": float(cfg.get("gap_tolerance_m", 15.0)),
                        "min_area_m2": float(cfg.get("min_area_m2", 200.0)),
                        "max_area_m2": float(cfg.get("max_area_m2", 10000.0)),
                        "min_closure": float(cfg.get("min_closure", 0.5)),
                        "max_elongation": float(cfg.get("max_elongation", 2.0)),
                        "min_ancrage": float(cfg.get("min_ancrage", 0.2)),
                        "max_isolement": float(cfg.get("max_isolement", 0.5)),
                        "min_rectangularite": float(cfg.get("min_rectangularite", 0.0)),
                        "min_confidence": float(cfg.get("min_confidence", 0.0)),
                    },
                    warn=logger.warning,
                    rule_type="enclosure",
                )
                if sane["max_area_m2"] < sane["min_area_m2"]:
                    sane["max_area_m2"] = sane["min_area_m2"]
                # Générateur de candidats : "auto" (V3 — anneaux ∪ cours ∪
                # blobs, dédoublonné), "hull" (enveloppe seule) ou "dilation"
                # (fermeture V1 seule).
                gen = str(cfg.get("generator", "auto")).strip().lower()
                if gen not in ("auto", "hull", "dilation"):
                    logger.warning(f"Enclosure: generator {gen!r} inconnu — auto utilisé")
                    gen = "auto"
                output_class = str(cfg.get("output_class_name", "")) or f"enclos_{'_'.join(target)}"
                rules.append(EnclosureRule(
                    target_classes=tuple(str(t) for t in target),
                    output_class_name=output_class,
                    generator=gen,
                    mode_calibration=bool(cfg.get("mode_calibration", False)),
                    **sane,
                ))
            except (TypeError, ValueError) as e:
                logger.warning(f"Règle enclosure ignorée : {e}")
            continue
        if rule_type == "alignment":
            try:
                from .clustering_bounds import sanitize_clustering_rule

                sane = sanitize_clustering_rule(
                    {
                        "band_width_m": float(cfg.get("band_width_m", 40.0)),
                        "angle_tolerance_deg": float(cfg.get("angle_tolerance_deg", 20.0)),
                        "min_length_m": float(cfg.get("min_length_m", 500.0)),
                        "max_gap_m": float(cfg.get("max_gap_m", 200.0)),
                        "min_coverage": float(cfg.get("min_coverage", 0.25)),
                        "min_sources": int(cfg.get("min_sources", 5)),
                        "min_confidence": float(cfg.get("min_confidence", 0.0)),
                    },
                    warn=logger.warning,
                    rule_type="alignment",
                )
                output_class = str(cfg.get("output_class_name", "")) or f"axe_{'_'.join(target)}"
                rules.append(AlignmentRule(
                    target_classes=tuple(str(t) for t in target),
                    output_class_name=output_class,
                    **sane,
                ))
            except (TypeError, ValueError) as e:
                logger.warning(f"Règle alignment ignorée : {e}")
            continue
        if rule_type != "dbscan":
            logger.warning(f"Règle de synthèse ignorée : type inconnu {rule_type!r}")
            continue
        try:
            from .clustering_bounds import sanitize_clustering_rule

            min_conf = float(cfg.get("min_confidence", 0.0))
            sane = sanitize_clustering_rule(
                {
                    "min_confidence": min_conf,
                    "min_confidence_extend": float(
                        cfg.get("min_confidence_extend", min_conf)
                    ),
                    "min_cluster_size": int(cfg.get("min_cluster_size", 5)),
                    "min_samples": int(cfg.get("min_samples", 3)),
                    "eps_m": float(cfg.get("eps_m", 30.0)),
                    "buffer_m": float(cfg.get("buffer_m", 10.0)),
                    "min_area_m2": float(cfg.get("min_area_m2", 0.0)),
                    "concave_ratio": float(cfg.get("concave_ratio", 0.3)),
                    "confidence_weight": float(cfg.get("confidence_weight", 0.0)),
                },
                warn=logger.warning,
            )
            output_class = str(cfg.get("output_class_name", "")) or f"cluster_{'_'.join(target)}"
            rules.append(
                ClusteringRule(
                    target_classes=tuple(str(t) for t in target),
                    min_confidence=sane["min_confidence"],
                    min_confidence_extend=sane["min_confidence_extend"],
                    min_cluster_size=sane["min_cluster_size"],
                    min_samples=sane["min_samples"],
                    eps_m=sane["eps_m"],
                    output_class_name=output_class,
                    output_geometry=str(cfg.get("output_geometry", "convex_hull")),
                    buffer_m=sane["buffer_m"],
                    min_area_m2=sane["min_area_m2"],
                    concave_ratio=sane["concave_ratio"],
                    confidence_weight=sane["confidence_weight"],
                )
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Clustering rule ignorée : {e}")
            continue
    return tuple(rules)


def _parse_postprocess(
    args_yaml: Dict[str, Any], task: Optional[str] = None
) -> PostprocessConfig:
    # Défaut de stratégie de superposition selon la tâche : pour un modèle bbox
    # (object detection), « difference » ne supprime jamais un doublon (elle
    # rogne le perdant) — or le halo inter-dalles fait détecter le même objet
    # par 2–4 dalles voisines. Seule « relation » (IoS) déduplique réellement,
    # donc c'est le défaut bbox ; args.yaml peut toujours surcharger.
    default_strategy = (
        "relation"
        if task is not None and str(task) not in _SEGMENTATION_TASKS
        else "difference"
    )
    pp = args_yaml.get("postprocess")
    if not isinstance(pp, dict):
        return PostprocessConfig(overlap_strategy=default_strategy)
    try:
        buffer_m = float(pp.get("merge_buffer_m", 0.5))
    except (TypeError, ValueError):
        buffer_m = 0.5
    if not (0 < buffer_m < float("inf")):  # ≤ 0, NaN ou inf → défaut
        buffer_m = 0.5

    strategy = str(pp.get("overlap_strategy", default_strategy)).strip().lower()
    if strategy not in ("difference", "relation"):
        strategy = default_strategy

    try:
        ios = float(pp.get("overlap_ios_threshold", 0.5))
    except (TypeError, ValueError):
        ios = 0.5
    if not (0 < ios <= 1):  # hors ]0, 1], NaN → défaut
        ios = 0.5

    try:
        ratio = float(pp.get("overlap_min_area_ratio", 0.0))
    except (TypeError, ValueError):
        ratio = 0.0
    if not (0 <= ratio <= 1):  # hors [0, 1], NaN → garde-fou désactivé
        ratio = 0.0

    return PostprocessConfig(
        merge_adjacent=bool(pp.get("merge_adjacent", True)),
        remove_overlaps=bool(pp.get("remove_overlaps", True)),
        merge_buffer_m=buffer_m,
        overlap_strategy=strategy,
        overlap_ios_threshold=ios,
        overlap_min_area_ratio=ratio,
    )


def _parse_class_colors(args_yaml: Dict[str, Any]) -> Optional[List[int]]:
    colors = args_yaml.get("class_colors")
    if not isinstance(colors, list):
        return None
    result: List[int] = []
    for c in colors:
        try:
            result.append(int(c))
        except (TypeError, ValueError):
            result.append(0)
    return result


def _parse_is_rfdetr(args_yaml: Dict[str, Any]) -> bool:
    model_type = str(args_yaml.get("model", "")).lower().strip()
    return "rf-detr" in model_type or "rfdetr" in model_type


def _load_class_names(model_dir: Path) -> Optional[List[str]]:
    """Cherche le fichier de classes dans le modèle (ordre cascade)."""
    candidates = [
        model_dir / "classes.txt",
        model_dir / "class_names.txt",
        model_dir / "classes.json",
        model_dir / "class_names.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix.lower() == ".json":
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(parsed, list) and parsed:
                    return [str(c).strip() for c in parsed]
                if isinstance(parsed, dict) and parsed:
                    # Clés JSON = strings → indexer str(i) (AUDIT v2 PARSE-08).
                    max_key = max(int(k) for k in parsed.keys())
                    return [str(parsed.get(str(i), f"classe_{i}")).strip() for i in range(max_key + 1)]
            else:
                lines = [ln.strip() for ln in candidate.read_text(encoding="utf-8-sig").splitlines()]
                lines = [ln for ln in lines if ln]
                if lines:
                    return lines
        except Exception as e:
            logger.warning(f"Erreur lecture {candidate}: {e}")
            continue
    return None
