"""
Résolution et configuration des modèles CV (chemins, SAHI, type de modèle, runs).

Extrait de class_utils.py pour séparer les responsabilités :
- model_config.py : résolution des chemins et configuration des modèles
- class_utils.py  : gestion des classes (noms, couleurs, indexing)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _resolve_model_dir(model_path: Union[str, Path]) -> Path:
    """
    Résout le dossier racine du modèle à partir d'un chemin de fichier weights
    ou d'un dossier modèle.
    
    Structure typique : model_name/weights/best.pt → model_name/
    """
    model_path = Path(model_path)
    if model_path.is_file():
        if model_path.parent.name == "weights":
            return model_path.parent.parent
        return model_path.parent
    return model_path


def resolve_model_weights_path(cv_config: Dict) -> Optional[Path]:
    """
    Résout le chemin complet vers le fichier weights du modèle CV
    à partir de la configuration.
    
    Cherche dans l'ordre :
    1. Chemin direct si le fichier existe
    2. models_dir / selected_model / weights / best.onnx
    3. models_dir / selected_model / weights / best.pt
    
    Returns:
        Path vers le fichier weights ou None si non trouvé
    """
    selected_model = str((cv_config or {}).get("selected_model", "")).strip()
    if not selected_model:
        return None
    
    model_path = Path(selected_model)
    if model_path.exists() and model_path.is_file():
        return model_path
    
    models_dir = Path((cv_config or {}).get("models_dir", "models"))
    
    # Chercher best.onnx en priorité, puis best.pt
    for ext in ("best.onnx", "best.pt"):
        candidate = models_dir / selected_model / "weights" / ext
        if candidate.exists():
            return candidate
    
    # Fallback : chemin par défaut (même s'il n'existe pas encore)
    return models_dir / selected_model / "weights" / "best.pt"


def _resolve_model_path_for_sahi(model: str, cv_config: Dict) -> Optional[Path]:
    """
    Résout le chemin du modèle pour charger sa config SAHI.
    Utilise la même logique que resolve_model_weights_path.
    """
    model_p = Path(model)
    if model_p.exists():
        return model_p
    models_dir = Path((cv_config or {}).get("models_dir", "models"))
    candidate = models_dir / model
    if candidate.exists():
        return candidate
    # Essayer avec weights/best.onnx
    for ext in ("best.onnx", "best.pt"):
        w = models_dir / model / "weights" / ext
        if w.exists():
            return w
    return None


def load_sahi_config_from_model(model_path: Union[str, Path]) -> Dict:
    """
    Charge la configuration SAHI depuis le args.yaml du modèle.
    
    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle
        
    Returns:
        Dict avec slice_height, slice_width, overlap_ratio.
        Valeurs par défaut (640, 640, 0.2) si non trouvé.
    """
    defaults = {"slice_height": 640, "slice_width": 640, "overlap_ratio": 0.2}
    model_dir = _resolve_model_dir(model_path)
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        logger.debug(f"Pas de args.yaml dans {model_dir}, SAHI par défaut")
        return defaults
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        if isinstance(args, dict):
            sahi = args.get("sahi")
            if isinstance(sahi, dict):
                result = {
                    "slice_height": int(sahi.get("slice_height", defaults["slice_height"])),
                    "slice_width": int(sahi.get("slice_width", defaults["slice_width"])),
                    "overlap_ratio": float(sahi.get("overlap_ratio", defaults["overlap_ratio"])),
                }
                logger.info(f"SAHI config chargée depuis {args_file.name}: {result}")
                return result
    except Exception as e:
        logger.warning(f"Erreur lecture SAHI depuis args.yaml: {e}")
    return defaults


def load_clustering_config_from_model(model_path: Union[str, Path]) -> Optional[List[Dict]]:
    """
    Charge la configuration de clustering depuis le args.yaml du modèle.
    
    Format attendu dans args.yaml:
        clustering:
          - target_classes: ["cratere_obus"]
            min_confidence: 0.5            # seuil core : initie/étend un cluster
            min_confidence_extend: 0.3     # (optionnel) seuil bas hystérésis :
                                           # absorbé comme "border" mais ne crée
                                           # pas de cluster. Défaut = min_confidence
                                           # (DBSCAN classique).
            min_cluster_size: 10
            min_samples: 5
            eps_m: 30
            output_class_name: "zone_crateres"
            output_geometry: "convex_hull"   # ou "concave_hull" / "bounding_box"
            concave_ratio: 0.3               # (optionnel, si concave_hull)
                                             # 0 = très concave, 1 = ~ convex_hull
            buffer_m: 10
            min_area_m2: 500
    
    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle
        
    Returns:
        Liste de configs clustering, ou None si non défini / désactivé
    """
    model_dir = _resolve_model_dir(model_path)
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return None
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        if not isinstance(args, dict):
            return None
        clustering_raw = args.get("clustering")
        if not clustering_raw:
            return None
        # Accepter un dict unique ou une liste
        if isinstance(clustering_raw, dict):
            clustering_raw = [clustering_raw]
        if not isinstance(clustering_raw, list):
            return None
        configs = []
        for cfg in clustering_raw:
            if not isinstance(cfg, dict):
                continue
            # Accepter target_classes (liste) ou target_class (str)
            target = cfg.get("target_classes", cfg.get("target_class"))
            if isinstance(target, str):
                target = [target]
            if not isinstance(target, list) or not target:
                logger.warning("Clustering config ignorée: target_classes manquant ou invalide")
                continue
            min_confidence_val = float(cfg.get("min_confidence", 0.0))
            # Hystérésis (Approche 1) : seuil bas pour absorber des détections
            # faibles dans un cluster existant sans qu'elles puissent l'initier.
            # Défaut = min_confidence → DBSCAN classique (rétro-compat).
            min_confidence_extend_val = float(
                cfg.get("min_confidence_extend", min_confidence_val)
            )
            parsed = {
                "target_classes": target,
                "min_confidence": min_confidence_val,
                "min_confidence_extend": min_confidence_extend_val,
                "min_cluster_size": int(cfg.get("min_cluster_size", 5)),
                "min_samples": int(cfg.get("min_samples", 3)),
                "eps_m": float(cfg.get("eps_m", 30.0)),
                "output_class_name": str(cfg.get("output_class_name", "")),
                "output_geometry": str(cfg.get("output_geometry", "convex_hull")),
                "buffer_m": float(cfg.get("buffer_m", 10.0)),
                "min_area_m2": float(cfg.get("min_area_m2", 0.0)),
                "concave_ratio": float(cfg.get("concave_ratio", 0.3)),
                "confidence_weight": float(cfg.get("confidence_weight", 0.0)),
            }
            # Nom par défaut basé sur les classes cibles
            if not parsed["output_class_name"]:
                parsed["output_class_name"] = f"cluster_{'_'.join(target)}"
            configs.append(parsed)
        if configs:
            logger.info(f"Clustering config chargée depuis {args_file.name}: {len(configs)} config(s)")
            return configs
    except Exception as e:
        logger.warning(f"Erreur lecture clustering depuis args.yaml: {e}")
    return None


def load_postprocess_config_from_model(model_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Charge la configuration de post-traitement géométrique depuis ``args.yaml``.

    Format attendu dans ``args.yaml`` :

    .. code-block:: yaml

        postprocess:
          merge_adjacent: false   # cratères disjoints, skip fusion intra-classe
          remove_overlaps: false  # skip suppression des superpositions inter-classes

    Si la section ``postprocess`` est absente, retourne les valeurs par défaut
    historiques (``merge_adjacent=True``, ``remove_overlaps=True``) pour ne pas
    casser les modèles existants qui dépendent de ces étapes (formes linéaires
    notamment, où la fusion entre dalles est essentielle).

    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle.

    Returns:
        Dict ``{"merge_adjacent": bool, "remove_overlaps": bool}``.
    """
    defaults = {"merge_adjacent": True, "remove_overlaps": True}
    model_dir = _resolve_model_dir(model_path)
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return defaults
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        if not isinstance(args, dict):
            return defaults
        pp = args.get("postprocess")
        if not isinstance(pp, dict):
            return defaults
        result = {
            "merge_adjacent": bool(pp.get("merge_adjacent", defaults["merge_adjacent"])),
            "remove_overlaps": bool(pp.get("remove_overlaps", defaults["remove_overlaps"])),
        }
        logger.info(f"Postprocess config chargée depuis {args_file.name}: {result}")
        return result
    except Exception as e:
        logger.warning(f"Erreur lecture postprocess depuis args.yaml: {e}")
    return defaults


def is_rfdetr_model(model_path: Union[str, Path]) -> bool:
    """
    Détecte si le modèle est un modèle RF-DETR en lisant args.yaml.
    
    RF-DETR utilise des class IDs 1-indexés, contrairement à YOLO (0-indexé).
    
    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle
        
    Returns:
        True si RF-DETR, False sinon (YOLO par défaut)
    """
    model_dir = _resolve_model_dir(model_path)
    
    # Chercher args.yaml
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return False
    
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        if isinstance(args, dict):
            model_type = str(args.get("model", "")).lower().strip()
            if "rf-detr" in model_type or "rfdetr" in model_type:
                logger.info(f"Modèle RF-DETR détecté via args.yaml: {model_type}")
                return True
    except Exception as e:
        logger.warning(f"Erreur lecture args.yaml: {e}")
    
    return False


def resolve_cv_runs(cv_config: Dict) -> List[Dict]:
    """
    Résout la liste des runs CV depuis la configuration.
    
    Chaque run est un dict cv_config complet avec son propre
    ``selected_model`` et ``target_rvt``.
    
    Rétrocompatible : si ``runs`` est absent ou vide, utilise
    ``selected_model`` + ``target_rvt`` comme unique run.
    """
    if not isinstance(cv_config, dict):
        return []

    runs_raw = cv_config.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        # Ancien format mono-modèle
        model = str(cv_config.get("selected_model") or "").strip()
        rvt = str(cv_config.get("target_rvt") or "LD").strip()
        if not model:
            return []
        run_cfg = dict(cv_config, selected_model=model, target_rvt=rvt)
        model_path = _resolve_model_path_for_sahi(model, cv_config)
        if model_path:
            run_cfg["sahi"] = load_sahi_config_from_model(model_path)
        return [run_cfg]

    result = []
    for run in runs_raw:
        if not isinstance(run, dict):
            continue
        model = str(run.get("model") or "").strip()
        if not model:
            continue
        rvt = str(run.get("target_rvt") or "LD").strip()
        # Construire un cv_config complet pour ce run
        run_cfg = dict(cv_config, selected_model=model, target_rvt=rvt)
        # Propager les champs spécifiques au run
        if "selected_classes" in run:
            run_cfg["selected_classes"] = run["selected_classes"]
        if "min_area_m2" in run:
            run_cfg["min_area_m2"] = float(run["min_area_m2"])
        # Charger la config SAHI depuis le dossier du modèle
        model_path = _resolve_model_path_for_sahi(model, cv_config)
        if model_path:
            run_cfg["sahi"] = load_sahi_config_from_model(model_path)
        result.append(run_cfg)
    return result
