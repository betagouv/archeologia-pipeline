"""
Utilitaires centralisés pour la gestion des classes de détection CV.

Ce module fournit une source unique de vérité pour:
- Chargement des noms de classes depuis le modèle
- Normalisation des class IDs (gestion 0-indexé vs 1-indexé)
- Mapping class_id <-> class_name
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def is_rfdetr_model(model_path: Union[str, Path]) -> bool:
    """
    Détecte si le modèle est un modèle RF-DETR en lisant args.yaml.
    
    RF-DETR utilise des class IDs 1-indexés, contrairement à YOLO (0-indexé).
    
    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle
        
    Returns:
        True si RF-DETR, False sinon (YOLO par défaut)
    """
    model_path = Path(model_path)
    
    # Déterminer le dossier du modèle
    if model_path.is_file():
        if model_path.parent.name == "weights":
            model_dir = model_path.parent.parent
        else:
            model_dir = model_path.parent
    else:
        model_dir = model_path
    
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


def load_class_names_from_model(model_path: Union[str, Path]) -> Optional[List[str]]:
    """
    Charge les noms de classes depuis le dossier du modèle.
    
    Cherche dans l'ordre:
    - classes.txt
    - class_names.txt
    - classes.json
    - class_names.json
    
    Args:
        model_path: Chemin vers le fichier weights (best.pt) ou le dossier du modèle
        
    Returns:
        Liste des noms de classes (0-indexée) ou None si non trouvé
    """
    model_path = Path(model_path)
    
    # Déterminer le dossier du modèle
    if model_path.is_file():
        # Si c'est un fichier (best.pt), remonter au dossier parent du modèle
        # Structure typique: model_name/weights/best.pt
        if model_path.parent.name == "weights":
            model_dir = model_path.parent.parent
        else:
            model_dir = model_path.parent
    else:
        model_dir = model_path
    
    if not model_dir.exists():
        logger.warning(f"Dossier modèle introuvable: {model_dir}")
        return None
    
    # Candidats pour le fichier de classes
    candidates = [
        model_dir / "classes.txt",
        model_dir / "class_names.txt",
        model_dir / "classes.json",
        model_dir / "class_names.json",
    ]
    
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
            
        try:
            if candidate.suffix.lower() == ".json":
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(parsed, list) and parsed:
                    logger.info(f"Classes chargées depuis {candidate.name}: {len(parsed)} classes")
                    return [str(c).strip() for c in parsed]
                elif isinstance(parsed, dict) and parsed:
                    # Dict {0: "class0", 1: "class1", ...}
                    max_key = max(int(k) for k in parsed.keys())
                    result = [str(parsed.get(i, f"classe_{i}")).strip() for i in range(max_key + 1)]
                    logger.info(f"Classes chargées depuis {candidate.name}: {len(result)} classes")
                    return result
            else:
                # Fichier texte: une classe par ligne
                lines = [ln.strip() for ln in candidate.read_text(encoding="utf-8-sig").splitlines()]
                lines = [ln for ln in lines if ln]
                if lines:
                    logger.info(f"Classes chargées depuis {candidate.name}: {len(lines)} classes")
                    return lines
        except Exception as e:
            logger.warning(f"Erreur lecture {candidate}: {e}")
            continue
    
    logger.warning(f"Aucun fichier de classes trouvé dans {model_dir}")
    return None


def get_num_classes_from_model(model_path: Union[str, Path]) -> Optional[int]:
    """
    Retourne le nombre de classes du modèle.
    
    Args:
        model_path: Chemin vers le modèle
        
    Returns:
        Nombre de classes ou None si non déterminable
    """
    class_names = load_class_names_from_model(model_path)
    if class_names:
        return len(class_names)
    
    # Fallback: essayer de lire depuis le modèle YOLO directement
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        if hasattr(model, 'names') and model.names:
            return len(model.names)
    except Exception:
        pass
    
    return None


def normalize_class_id(
    class_id: int,
    num_classes: int,
    *,
    detected_min_id: Optional[int] = None,
    detected_max_id: Optional[int] = None,
) -> int:
    """
    Normalise un class_id pour garantir qu'il est 0-indexé.
    
    Détecte automatiquement si les IDs sont 1-indexés et les corrige.
    
    Args:
        class_id: L'ID de classe brut depuis YOLO
        num_classes: Nombre total de classes dans le modèle
        detected_min_id: ID minimum détecté dans le batch (pour détection 1-indexé)
        detected_max_id: ID maximum détecté dans le batch (pour détection 1-indexé)
        
    Returns:
        class_id normalisé (0-indexé, borné à [0, num_classes-1])
    """
    # Détection des IDs 1-indexés:
    # Si min >= 1, pas de 0, et max >= num_classes → probablement 1-indexé
    if detected_min_id is not None and detected_max_id is not None:
        if detected_min_id >= 1 and detected_max_id >= num_classes:
            # Décaler de -1
            class_id = class_id - 1
    
    # Borner à [0, num_classes - 1]
    if class_id < 0:
        class_id = 0
    elif class_id >= num_classes:
        class_id = num_classes - 1
    
    return class_id


def detect_indexing_offset(class_ids: List[int], num_classes: int) -> int:
    """
    Détecte si les class_ids sont 1-indexés et retourne l'offset à appliquer.
    
    Args:
        class_ids: Liste des class_ids détectés
        num_classes: Nombre de classes dans le modèle
        
    Returns:
        0 si déjà 0-indexé, -1 si 1-indexé (à soustraire)
    """
    if not class_ids or num_classes <= 0:
        return 0
    
    min_id = min(class_ids)
    max_id = max(class_ids)
    has_zero = 0 in class_ids
    
    # Heuristique: si min >= 1, pas de 0, et max >= num_classes → 1-indexé
    if min_id >= 1 and not has_zero and max_id >= num_classes:
        logger.warning(
            f"Class IDs semblent 1-indexés (min={min_id}, max={max_id}, "
            f"nb_classes={num_classes}). Décalage de -1 appliqué."
        )
        return -1
    
    return 0


def class_id_to_name(
    class_id: int,
    class_names: Optional[List[str]],
    *,
    offset: int = 0,
) -> str:
    """
    Convertit un class_id en nom de classe.
    
    Args:
        class_id: ID de classe (après normalisation si nécessaire)
        class_names: Liste des noms de classes (0-indexée)
        offset: Offset à appliquer (ex: -1 si 1-indexé)
        
    Returns:
        Nom de la classe ou fallback "classe_N"
    """
    adjusted_id = class_id + offset
    
    if class_names and 0 <= adjusted_id < len(class_names):
        return class_names[adjusted_id].strip()
    
    return f"classe_{class_id + 1}"


def create_class_mapping(class_names: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Crée des mappings bidirectionnels class_id <-> class_name.
    
    Args:
        class_names: Liste des noms de classes (0-indexée)
        
    Returns:
        Tuple (id_to_name, name_to_id)
    """
    id_to_name = {i: name.strip() for i, name in enumerate(class_names)}
    name_to_id = {name.strip(): i for i, name in enumerate(class_names)}
    return id_to_name, name_to_id


# Palette de couleurs de base (12 couleurs numérotées 0-11)
# Chaque couleur a 5 variantes de luminosité pour les gammes de confiance
# Format: (R, G, B) - couleur de base (confiance haute)
BASE_COLOR_PALETTE = [
    (255, 59, 59),    # 0: Rouge vif
    (50, 205, 50),    # 1: Vert lime
    (30, 144, 255),   # 2: Bleu dodger
    (255, 215, 0),    # 3: Or/Jaune
    (255, 0, 255),    # 4: Magenta
    (0, 206, 209),    # 5: Turquoise
    (255, 140, 0),    # 6: Orange
    (138, 43, 226),   # 7: Violet
    (0, 250, 154),    # 8: Vert printemps
    (255, 20, 147),   # 9: Rose profond
    (173, 255, 47),   # 10: Vert-jaune
    (65, 105, 225),   # 11: Bleu royal
]


def _lighten_color(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Éclaircit une couleur RGB par un facteur (0=original, 1=blanc)."""
    r, g, b = rgb
    return (
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


def get_color_for_confidence(base_color_index: int, confidence: float) -> Tuple[int, int, int]:
    """
    Retourne une couleur RGB basée sur l'index de couleur et la confiance.
    
    Plus la confiance est haute, plus la couleur est saturée (foncée).
    Plus la confiance est basse, plus la couleur est claire.
    
    Args:
        base_color_index: Index de la couleur de base (0-11)
        confidence: Valeur de confiance (0.0-1.0)
        
    Returns:
        Tuple (R, G, B)
    """
    base_color = BASE_COLOR_PALETTE[base_color_index % len(BASE_COLOR_PALETTE)]
    
    # Normaliser la confiance
    if confidence > 1.0:
        confidence = confidence / 10.0 if confidence <= 10.0 else 1.0
    confidence = max(0.0, min(1.0, confidence))
    
    # Facteur d'éclaircissement inversé (haute confiance = couleur saturée)
    # Max 85% d'éclaircissement pour une variation plus importante
    lighten_factor = (1.0 - confidence) * 0.85
    
    return _lighten_color(base_color, lighten_factor)


def load_class_colors_from_model(model_path: Union[str, Path]) -> Optional[List[int]]:
    """
    Charge les indices de couleurs par classe depuis args.yaml du modèle.
    
    Format attendu dans args.yaml:
        class_colors: [0, 1, 2]  # Index de couleur pour chaque classe
    
    Args:
        model_path: Chemin vers le fichier weights ou le dossier du modèle
        
    Returns:
        Liste des indices de couleurs (0-11) par classe, ou None si non défini
    """
    model_path = Path(model_path)
    print(f"[CLASS_UTILS] load_class_colors_from_model: model_path={model_path}", flush=True)
    
    # Déterminer le dossier du modèle
    if model_path.is_file():
        if model_path.parent.name == "weights":
            model_dir = model_path.parent.parent
        else:
            model_dir = model_path.parent
    else:
        model_dir = model_path
    
    # Chercher args.yaml
    args_file = model_dir / "args.yaml"
    print(f"[CLASS_UTILS] args_file={args_file}, exists={args_file.exists()}", flush=True)
    if not args_file.exists():
        print(f"[CLASS_UTILS] args.yaml not found, returning None", flush=True)
        return None
    
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        print(f"[CLASS_UTILS] args.yaml loaded, keys={list(args.keys()) if isinstance(args, dict) else 'not a dict'}", flush=True)
        
        if isinstance(args, dict) and "class_colors" in args:
            colors = args["class_colors"]
            print(f"[CLASS_UTILS] class_colors found in args.yaml: {colors}", flush=True)
            if isinstance(colors, list):
                # Valider que ce sont des entiers dans la plage valide
                result = []
                for c in colors:
                    try:
                        idx = int(c)
                        result.append(idx % len(BASE_COLOR_PALETTE))
                    except (ValueError, TypeError):
                        result.append(0)
                print(f"[CLASS_UTILS] class_colors validated: {result}", flush=True)
                logger.info(f"Couleurs de classes chargées depuis args.yaml: {result}")
                return result
        else:
            print(f"[CLASS_UTILS] class_colors NOT found in args.yaml", flush=True)
    except Exception as e:
        print(f"[CLASS_UTILS] Error reading args.yaml: {e}", flush=True)
        logger.warning(f"Erreur lecture class_colors depuis args.yaml: {e}")
    
    return None


def get_class_color(class_id: int, class_colors: Optional[List[int]] = None) -> Tuple[int, int, int]:
    """
    Retourne la couleur RGB de base pour une classe.
    
    Args:
        class_id: ID de la classe (0-indexé)
        class_colors: Liste optionnelle des indices de couleurs par classe
        
    Returns:
        Tuple (R, G, B)
    """
    if class_colors and 0 <= class_id < len(class_colors):
        color_index = class_colors[class_id]
    else:
        color_index = class_id
    
    return BASE_COLOR_PALETTE[color_index % len(BASE_COLOR_PALETTE)]


def get_confidence_color_name(base_color_index: int, confidence: float) -> str:
    """
    Retourne un nom de couleur pour les shapefiles basé sur l'index et la confiance.
    
    Format: "color{index}_{bucket}" où bucket est l'intervalle de confiance.
    
    Args:
        base_color_index: Index de la couleur de base (0-11)
        confidence: Valeur de confiance (0.0-1.0)
        
    Returns:
        Nom de couleur (ex: "color0_high", "color1_medium")
    """
    # Normaliser
    if confidence > 1.0:
        confidence = confidence / 10.0 if confidence <= 10.0 else 1.0
    
    if confidence >= 0.8:
        bucket = "high"
    elif confidence >= 0.6:
        bucket = "medium_high"
    elif confidence >= 0.4:
        bucket = "medium"
    elif confidence >= 0.2:
        bucket = "medium_low"
    else:
        bucket = "low"
    
    return f"color{base_color_index}_{bucket}"
