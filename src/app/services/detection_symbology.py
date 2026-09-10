"""Prédicat partagé : une couche de détection est-elle une SORTIE de brique
de synthèse (zone de clusters, enclos, axe linéaire) ?

Les zones de clusters DBSCAN (champ ``nb_detect``) gardent le style hachuré
historique. Les sorties enclos/axes (``nb_sources``) portent depuis la
confiance composite un ``conf_bin`` cohérent avec les tranches du run →
elles sont rendues en catégories de confiance, comme les détections
(« même granularité que les parcellaires »).

⚠ Ne PAS clef sur ``cluster_id``/``enclos_id``/``axe_id`` : les couches
SOURCES les portent aussi (traçabilité membre → synthèse).

Module pur (aucun import) → testable hors QGIS ; consommé par
``ui/layer_loader.build_detection_vector_layer`` (chargement live + ``.qgs``).
"""
from __future__ import annotations

from typing import Iterable

_SYNTHESIS_COUNT_FIELDS = ("nb_detect",)


def is_synthesis_layer(field_names: Iterable[str]) -> bool:
    """True si la couche doit recevoir le style zone hachuré (clusters DBSCAN)."""
    names = set(field_names)
    return any(f in names for f in _SYNTHESIS_COUNT_FIELDS)
