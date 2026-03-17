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
