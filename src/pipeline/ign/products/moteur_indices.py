"""Moteur d'indices intégré : les indices de visualisation calculés dans le plugin avec la bibliothèque
rvt embarquée (``rvt_embarque/`` : copie figée de celle de rvt-qgis 1.0.1, voir son NOTICE), au lieu des
algorithmes Processing de l'extension rvt-qgis.

Règle : chaque produit appelle les MÊMES fonctions de rvt, avec les MÊMES arguments, que l'algorithme
Processing qu'il remplace (``processing_provider/rvt_*.py`` de rvt-qgis 1.0.1). Tout écart est nommé là où il
est fait et prouvé au pixel près (``tests/unit/test_moteur_indices.py``, bancs N2).

Imports de numpy et de rvt différés : ce module s'importe sans eux (tests, préflight).
"""
from __future__ import annotations

import os
from pathlib import Path

#: « processing » : chaque produit qui a un algorithme Processing passe par l'extension rvt-qgis, comme avant
#: la 0.16.0 (diagnostic d'un écart ; l'extension est alors requise). CVAT et CRIM n'en ont pas.
VARIABLE_MOTEUR = "ARCHEO_INDICES_MOTEUR"

#: Produits calculés par le moteur intégré ; les autres passent encore par Processing (rvt-qgis).
PRODUITS_MOTEUR = frozenset({"CVAT", "CRIM"})

#: Produits sans algorithme Processing : toujours calculés ici, même forcé.
SANS_PROCESSING = frozenset({"CVAT", "CRIM"})


def par_le_moteur(produit: str) -> bool:
    """Vrai si ``produit`` est calculé ici, faux s'il passe par Processing (rvt-qgis)."""
    if produit in SANS_PROCESSING:
        return True
    if os.environ.get(VARIABLE_MOTEUR, "").strip().lower() == "processing":
        return False
    return produit in PRODUITS_MOTEUR


def chemin_reglages(nom: str) -> Path:
    """Fichier de réglages de rvt embarqué (``rvt_embarque/settings/``)."""
    return Path(__file__).resolve().parent / "rvt_embarque" / "settings" / nom


def moteur_disponible() -> str:
    """'' si la bibliothèque rvt embarquée se charge (numpy, scipy, matplotlib, GDAL), sinon la raison."""
    try:
        from .rvt_embarque import blend, default, vis  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — toute raison est à montrer au préflight
        return f"{type(exc).__name__}: {exc}"
    return ""
