"""Prédicat partagé : une couche de détection est-elle une SORTIE de brique
de synthèse (zone de clusters, enclos, axe linéaire) ?

Les sorties de synthèse n'ont pas de ``conf_bin`` exploitable (les tranches
sont calculées AVANT les briques) — un rendu catégorisé par confiance ne
matcherait aucune entité et la couche serait invisible (bug constaté sur la
couche « Enclos »). Elles portent en revanche un champ de comptage qui
n'existe QUE sur elles : ``nb_detect`` (dbscan) ou ``nb_sources``
(enclosure, alignment).

⚠ Ne PAS clef sur ``cluster_id``/``enclos_id``/``axe_id`` : les couches
SOURCES les portent aussi (traçabilité membre → synthèse).

Module pur (aucun import) → testable hors QGIS ; consommé par
``ui/layer_loader.build_detection_vector_layer`` (chargement live + ``.qgs``).
"""
from __future__ import annotations

from typing import Iterable

_SYNTHESIS_COUNT_FIELDS = ("nb_detect", "nb_sources")


def is_synthesis_layer(field_names: Iterable[str]) -> bool:
    """True si la couche est une sortie de brique de synthèse (style zone)."""
    names = set(field_names)
    return any(f in names for f in _SYNTHESIS_COUNT_FIELDS)
