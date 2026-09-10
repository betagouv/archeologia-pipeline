"""Humanisation des codes techniques d'un ``model_card`` (codes → libellés FR).

Vit dans ``app`` et non dans ``ui`` parce que deux consommateurs s'en servent :
le dialog d'info modèle (``ui/dialogs/_model_info_data.py``, qui les ré-exporte)
et la fiche de classe (``app/services/class_fiche.py``, module pur importé par
le pipeline). Une dépendance app → ui casserait l'import standalone dès que
``src/ui/`` se met à importer Qt au chargement — et ``src/ui/`` n'est pas
couvert par pytest, donc la casse ne se verrait qu'au runtime dans QGIS.
"""
from __future__ import annotations

from typing import Dict

_RVT_LONG_NAMES: Dict[str, str] = {
    "LD": "Local Dominance (LD)",
    "SVF": "Sky View Factor (SVF)",
    "M_HS": "Hillshade multi-directionnel (M-HS)",
    "HS": "Hillshade simple (HS)",
    "SLO": "Pente (SLO)",
    "SLRM": "Simple Local Relief Model (SLRM)",
    "VAT": "Visualisation Archéologique Totale (VAT)",
    "MSTP": "Multi-Scale Topographic Position (MSTP)",
    "CVAT": "Combined VAT (CVAT)",
}

_TASK_LABELS: Dict[str, str] = {
    "object_detection": "Détection d'objets",
    "instance_segmentation": "Segmentation d'instances",
    "semantic_segmentation": "Segmentation sémantique",
}


def pretty_rvt_name(code: str) -> str:
    """Code court (``LD``, ``SVF``, …) → libellé long FR. Repli : code brut."""
    raw = code or ""
    return _RVT_LONG_NAMES.get(raw.upper(), raw)


def pretty_task(code: str) -> str:
    """Code de tâche (``object_detection``…) → libellé FR. Repli : valeur brute."""
    raw = code or ""
    return _TASK_LABELS.get(raw, raw)
