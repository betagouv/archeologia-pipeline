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
    """Une règle de clustering DBSCAN spatial post-détection."""
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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
class PostprocessConfig:
    """Activation des étapes de post-traitement géométrique."""
    merge_adjacent: bool = True
    remove_overlaps: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "merge_adjacent": self.merge_adjacent,
            "remove_overlaps": self.remove_overlaps,
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
        postprocess = _parse_postprocess(args_yaml)
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
        return SahiConfig(
            slice_height=int(sahi.get("slice_height", 640)),
            slice_width=int(sahi.get("slice_width", 640)),
            overlap_ratio=float(sahi.get("overlap_ratio", 0.2)),
        )
    except (TypeError, ValueError) as e:
        logger.warning(f"SAHI config invalide ({e}), valeurs par défaut")
        return SahiConfig()


def _parse_clustering(args_yaml: Dict[str, Any]) -> Tuple[ClusteringRule, ...]:
    raw = args_yaml.get("clustering")
    if not raw:
        return tuple()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return tuple()

    rules: List[ClusteringRule] = []
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        target = cfg.get("target_classes", cfg.get("target_class"))
        if isinstance(target, str):
            target = [target]
        if not isinstance(target, list) or not target:
            logger.warning("Clustering rule ignorée : target_classes manquant/invalide")
            continue
        try:
            min_conf = float(cfg.get("min_confidence", 0.0))
            min_conf_extend = float(cfg.get("min_confidence_extend", min_conf))
            output_class = str(cfg.get("output_class_name", "")) or f"cluster_{'_'.join(target)}"
            rules.append(
                ClusteringRule(
                    target_classes=tuple(str(t) for t in target),
                    min_confidence=min_conf,
                    min_confidence_extend=min_conf_extend,
                    min_cluster_size=int(cfg.get("min_cluster_size", 5)),
                    min_samples=int(cfg.get("min_samples", 3)),
                    eps_m=float(cfg.get("eps_m", 30.0)),
                    output_class_name=output_class,
                    output_geometry=str(cfg.get("output_geometry", "convex_hull")),
                    buffer_m=float(cfg.get("buffer_m", 10.0)),
                    min_area_m2=float(cfg.get("min_area_m2", 0.0)),
                    concave_ratio=float(cfg.get("concave_ratio", 0.3)),
                    confidence_weight=float(cfg.get("confidence_weight", 0.0)),
                )
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Clustering rule ignorée : {e}")
            continue
    return tuple(rules)


def _parse_postprocess(args_yaml: Dict[str, Any]) -> PostprocessConfig:
    pp = args_yaml.get("postprocess")
    if not isinstance(pp, dict):
        return PostprocessConfig()
    return PostprocessConfig(
        merge_adjacent=bool(pp.get("merge_adjacent", True)),
        remove_overlaps=bool(pp.get("remove_overlaps", True)),
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
                    max_key = max(int(k) for k in parsed.keys())
                    return [str(parsed.get(i, f"classe_{i}")).strip() for i in range(max_key + 1)]
            else:
                lines = [ln.strip() for ln in candidate.read_text(encoding="utf-8-sig").splitlines()]
                lines = [ln for ln in lines if ln]
                if lines:
                    return lines
        except Exception as e:
            logger.warning(f"Erreur lecture {candidate}: {e}")
            continue
    return None
