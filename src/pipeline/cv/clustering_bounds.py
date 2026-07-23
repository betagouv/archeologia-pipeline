"""Bornes et coercition tolérante des paramètres de clustering.

Source unique pour les trois consommateurs (AUDIT v1 PARSE-02/03, corrigés
en v2) :

- ``model_config.load_clustering_config_from_model`` (args.yaml, loader legacy)
- ``model_profile._parse_clustering`` (args.yaml, ModelProfile)
- ``runner_shapefiles`` (surcharges UI ``clustering_overrides``)

Sans bornes, une valeur aberrante (``eps_m: -10``, ``min_samples: 0``)
traversait jusqu'à scipy/DBSCAN → clustering silencieusement vide ou crash.
Module pur (logging/typing uniquement) → testable hors QGIS.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

WarnFn = Callable[[str], None]

# clé → (lo, hi, is_int). hi=None : pas de borne haute.
NUMERIC_BOUNDS: Dict[str, tuple] = {
    "min_confidence": (0.0, 1.0, False),
    "min_confidence_extend": (0.0, 1.0, False),
    "min_cluster_size": (1, None, True),
    "min_samples": (1, None, True),
    "eps_m": (0.1, None, False),
    "buffer_m": (0.0, None, False),
    "min_area_m2": (0.0, None, False),
    "concave_ratio": (0.0, 1.0, False),
    "confidence_weight": (0.0, None, False),
}

ALLOWED_GEOMETRIES = ("convex_hull", "concave_hull", "bounding_box")

# Bornes des règles « enclosure » (fermeture vectorielle + scoring).
ENCLOSURE_BOUNDS: Dict[str, tuple] = {
    "gap_tolerance_m": (0.5, 50.0, False),
    "min_area_m2": (1.0, None, False),
    "max_area_m2": (1.0, None, False),
    "min_closure": (0.0, 1.0, False),
    "max_elongation": (1.0, None, False),
    "min_confidence": (0.0, 1.0, False),
}

# Registre par type de règle de synthèse (args.yaml:clustering → champ ``type``).
BOUNDS_BY_TYPE: Dict[str, Dict[str, tuple]] = {
    "dbscan": NUMERIC_BOUNDS,
    "enclosure": ENCLOSURE_BOUNDS,
}


def _clamp(
    key: str, value: Any, warn: Optional[WarnFn], bounds: Dict[str, tuple] = NUMERIC_BOUNDS
) -> Optional[Any]:
    """Caste puis borne ``value`` pour ``key``. None si non castable."""
    lo, hi, is_int = bounds[key]
    try:
        num = int(float(value)) if is_int else float(value)
    except (TypeError, ValueError):
        if warn:
            warn(f"Clustering: {key}={value!r} non numérique — ignoré")
        return None
    clamped = max(lo, num) if hi is None else min(max(lo, num), hi)
    if clamped != num and warn:
        warn(f"Clustering: {key}={num} hors bornes — clampé à {clamped}")
    return clamped


def sanitize_clustering_rule(
    rule: Dict[str, Any], warn: Optional[WarnFn] = None, rule_type: str = "dbscan"
) -> Dict[str, Any]:
    """Borne les champs numériques d'une règle (les autres clés sont gardées).

    Un champ non castable retombe sur la borne basse (la règle reste
    utilisable) — l'isolation par règle est gérée par l'appelant.
    ``rule_type`` sélectionne le jeu de bornes (dbscan par défaut).
    """
    warn = warn or logger.warning
    bounds = BOUNDS_BY_TYPE.get(rule_type, NUMERIC_BOUNDS)
    out = dict(rule)
    for key in bounds:
        if key in out:
            clamped = _clamp(key, out[key], warn, bounds)
            out[key] = clamped if clamped is not None else bounds[key][0]
    return out


def sanitize_clustering_overrides(
    overrides: Dict[str, Any], warn: Optional[WarnFn] = None, rule_type: str = "dbscan"
) -> Dict[str, Any]:
    """Filtre des surcharges UI : clés CONNUES uniquement, castées et bornées.

    Contrairement à la config modèle (castée au chargement), les overrides
    étaient fusionnés tels quels par ``cc.update(ov)`` (AUDIT PARSE-03) —
    types/plages non garantis, clés arbitraires injectables.
    """
    warn = warn or logger.warning
    bounds = BOUNDS_BY_TYPE.get(rule_type, NUMERIC_BOUNDS)
    out: Dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        if key in bounds:
            clamped = _clamp(key, value, warn, bounds)
            if clamped is not None:
                out[key] = clamped
        elif key == "output_geometry" and rule_type == "dbscan":
            geom = str(value)
            if geom in ALLOWED_GEOMETRIES:
                out[key] = geom
            elif warn:
                warn(f"Clustering: output_geometry={value!r} inconnue — ignorée")
        elif warn:
            warn(f"Clustering: surcharge inconnue {key!r} — ignorée")
    return out
