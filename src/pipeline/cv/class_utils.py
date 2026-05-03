"""
Utilitaires centralisés pour la gestion des classes de détection CV.

Ce module fournit une source unique de vérité pour:
- Chargement des noms de classes depuis le modèle
- Normalisation des class IDs (gestion 0-indexé vs 1-indexé)
- Palette de couleurs et mapping class_id <-> couleur

La résolution des chemins et configuration des modèles (SAHI, runs, etc.)
est dans model_config.py.  Les symboles sont réexportés ici pour
rétrocompatibilité.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Réexportation depuis model_config (rétrocompatibilité : tous les imports
# existants ``from .class_utils import resolve_cv_runs`` continuent de fonctionner).
from .model_config import (  # noqa: F401
    resolve_cv_runs,
    resolve_model_weights_path,
    load_sahi_config_from_model,
    is_rfdetr_model,
    _resolve_model_dir,
)

logger = logging.getLogger(__name__)


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
    model_dir = _resolve_model_dir(model_path)
    
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


def _darken_color(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Assombrit une couleur RGB par un facteur (0=original, 1=noir)."""
    r, g, b = rgb
    return (
        int(r * (1 - factor)),
        int(g * (1 - factor)),
        int(b * (1 - factor)),
    )


def get_color_for_confidence(base_color_index: int, confidence: float) -> Tuple[int, int, int]:
    """
    Retourne une couleur RGB basée sur l'index de couleur et la confiance.
    
    Plus la confiance est haute, plus la couleur est saturée (foncée).
    Plus la confiance est basse, plus la couleur est claire.
    
    Écart accentué:
    - Confiance >= 0.8: couleur assombrie (30%)
    - Confiance 0.6-0.8: couleur assombrie légèrement (15%)
    - Confiance 0.4-0.6: couleur de base
    - Confiance 0.2-0.4: couleur éclaircie (50%)
    - Confiance < 0.2: couleur très claire (90%)
    
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
    
    # Écart accentué avec assombrissement pour haute confiance et éclaircissement pour basse
    if confidence >= 0.8:
        # Haute confiance: couleur foncée (assombrie de 30%)
        return _darken_color(base_color, 0.30)
    elif confidence >= 0.6:
        # Confiance moyenne-haute: légèrement assombrie (15%)
        return _darken_color(base_color, 0.15)
    elif confidence >= 0.4:
        # Confiance moyenne: couleur de base
        return base_color
    elif confidence >= 0.2:
        # Confiance moyenne-basse: éclaircie (35%)
        return _lighten_color(base_color, 0.35)
    else:
        # Basse confiance: claire (65%)
        return _lighten_color(base_color, 0.65)


# --- Tranches de confiance (conf_bin) ------------------------------------
# Bornes supérieures standard de la grille 0.2 utilisées pour les tranches
# (cohérentes avec `get_color_for_confidence`).
_CONFIDENCE_UPPER_BOUNDS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

# Suffixes textuels utilisés dans le champ `conf_color` (ex. "color0_medium").
_CONFIDENCE_SUFFIXES: Tuple[str, ...] = (
    "low", "medium_low", "medium", "medium_high", "high",
)


def _format_conf_bound(x: float) -> str:
    """Formate une borne de confiance à la manière Python ``f"{x:g}"``.

    Exemples : 0.0→"0", 0.2→"0.2", 0.35→"0.35", 1.0→"1".
    """
    # clamp
    x = max(0.0, min(1.0, float(x)))
    return f"{x:g}"


def _suffix_for_confidence(value: float) -> str:
    """Retourne le suffixe textuel (low / medium_low / medium / medium_high / high)
    utilisé dans `conf_color`, basé sur la même grille 0.2 que
    :func:`get_color_for_confidence`.
    """
    v = max(0.0, min(1.0, float(value)))
    if v >= 0.8:
        return "high"
    if v >= 0.6:
        return "medium_high"
    if v >= 0.4:
        return "medium"
    if v >= 0.2:
        return "medium_low"
    return "low"


def compute_confidence_bins(min_confidence: float = 0.0) -> List[Dict[str, float]]:
    """Calcule les tranches de confiance utilisées pour le champ ``conf_bin``
    et la symbologie QGIS catégorisée.

    Les tranches suivent la grille standard 0.2 ``{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}``,
    sauf la première qui est **tronquée à ``min_confidence``** si celui-ci est
    strictement supérieur à 0 et ne tombe pas sur une borne. Ceci évite
    d'afficher une catégorie vide (ex. ``[0:0.2[``) quand l'utilisateur a
    positionné un seuil de confiance ≥ 0.2 dans les paramètres avancés :
    toutes les détections inférieures à ce seuil sont filtrées en amont.

    Exemples :

    - ``min_confidence=0``    → ``[0:0.2[, [0.2:0.4[, [0.4:0.6[, [0.6:0.8[, [0.8:1]``
    - ``min_confidence=0.3``  → ``[0.3:0.4[, [0.4:0.6[, [0.6:0.8[, [0.8:1]``
    - ``min_confidence=0.5``  → ``[0.5:0.6[, [0.6:0.8[, [0.8:1]``
    - ``min_confidence=0.9``  → ``[0.9:1]``

    Chaque tranche est retournée sous la forme d'un dict::

        {"label": str, "lower": float, "upper": float, "repr": float, "suffix": str}

    où ``repr`` est le point médian de la tranche (utilisé pour calculer la
    couleur via :func:`get_color_for_confidence`) et ``suffix`` est le libellé
    descriptif (low/medium_low/medium/medium_high/high).
    """
    m = max(0.0, min(1.0, float(min_confidence)))
    bins: List[Dict[str, float]] = []
    lower = m
    EPS = 1e-9
    uppers = list(_CONFIDENCE_UPPER_BOUNDS)
    for i, upper in enumerate(uppers):
        if upper <= lower + EPS:
            continue  # tranche vide si seuil aligné sur cette borne
        is_last = (i == len(uppers) - 1)
        closing = "]" if is_last else "["
        label = f"[{_format_conf_bound(lower)}:{_format_conf_bound(upper)}{closing}"
        mid = (lower + upper) / 2.0
        bins.append({
            "label": label,
            "lower": float(lower),
            "upper": float(upper),
            "repr": float(mid),
            "suffix": _suffix_for_confidence(mid),
        })
        lower = upper
    return bins


def assign_confidence_bin(
    confidence_value: Optional[float],
    color_index: int = 0,
    min_confidence: float = 0.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Assigne la tranche (``conf_bin``) et le nom de couleur (``conf_color``)
    d'une détection, en tenant compte d'un seuil de confiance minimal.

    Retourne ``(None, None)`` si ``confidence_value`` est absente ou invalide.
    Retourne ``(None, None)`` si la valeur est strictement inférieure au seuil
    (cas défensif : normalement filtré en amont).
    """
    if confidence_value is None:
        return None, None
    try:
        c = float(confidence_value)
    except Exception:
        return None, None

    # Normaliser si la confiance semble être sur [0,10]
    if c > 1.0 and c <= 10.0:
        c = c / 10.0
    c = max(0.0, min(1.0, c))

    bins = compute_confidence_bins(min_confidence)
    if not bins:
        return None, None

    # Si la valeur est en dessous du premier bin (donc < seuil), pas de bin.
    if c < bins[0]["lower"] - 1e-9:
        return None, None

    for i, b in enumerate(bins):
        is_last = (i == len(bins) - 1)
        if b["lower"] - 1e-9 <= c and (
            c < b["upper"] - 1e-9 or (is_last and c <= b["upper"] + 1e-9)
        ):
            return b["label"], f"color{color_index}_{b['suffix']}"

    # fallback : dernier bin
    last = bins[-1]
    return last["label"], f"color{color_index}_{last['suffix']}"


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
    model_dir = _resolve_model_dir(model_path)
    
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return None
    
    try:
        import yaml
        with open(args_file, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        if isinstance(args, dict) and "class_colors" in args:
            colors = args["class_colors"]
            if isinstance(colors, list):
                result = []
                for c in colors:
                    try:
                        idx = int(c)
                        result.append(idx % len(BASE_COLOR_PALETTE))
                    except (ValueError, TypeError):
                        result.append(0)
                logger.info(f"Couleurs de classes chargées depuis args.yaml: {result}")
                return result
    except Exception as e:
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
