"""Attribution déterministe d'une couleur de base PAR CLASSE.

Source unique de vérité pour la couleur d'une classe de détection, utilisée
identiquement à la génération du GeoPackage et à l'affichage (live + .qgs).

Remplace l'ancien système (palette fixe de 12 couleurs + mapping par index
avec replis ``class_id`` / fuzzy) qui produisait des collisions — deux classes
de modèles différents au même ``class_id`` recevaient la même couleur.

Principe (cf. docs/superpowers/specs/2026-06-12-couleurs-detections-design.md) :

- La couleur dérive d'un **hash stable** du nom de classe (``hashlib``, jamais
  ``hash()`` qui varie d'un process à l'autre — ce qui désynchroniserait le
  ``.qgs`` du chargement live).
- La **teinte** parcourt le cercle chromatique ; saturation et luminosité de
  base sont fixes → couleurs vives et lisibles. Avec un hash équiréparti, des
  classes distinctes obtiennent des teintes réparties, **sans limite de nombre
  de classes** et **sans état** (ajouter une classe ne déplace aucune autre).
- La **confiance** module la luminosité (5 paliers), comme avant.

Module pur (``hashlib``, ``colorsys``, ``math``) → testable hors QGIS.
"""
from __future__ import annotations

import colorsys
import hashlib
from typing import Tuple

RGB = Tuple[int, int, int]

# Luminosité de base (espace HLS) : fixe, centrale, pour laisser de la marge à
# la modulation par confiance (claircir/assombrir) — la confiance ne doit pas
# saturer en blanc/noir. Garder la luminosité constante préserve aussi la
# lisibilité (pas de classe « presque blanche » ou « presque noire »).
_BASE_LIGHTNESS = 0.50

# Saturation sur paliers (2ᵉ dimension) : un hash de teinte seul peut placer
# deux classes à ~1° l'une de l'autre (quasi-collision). En séparant aussi la
# saturation, deux classes à teintes voisines restent perceptiblement
# distinctes — sans aucun état ni réattribution (toujours fonction pure du nom).
_SATURATION_LEVELS = (0.45, 0.70, 0.95)

# Décalage par le nombre d'or conjugué : sur un hash déjà équiréparti il est
# neutre, mais garantit l'équidistribution même si la source de hash changeait.
_GOLDEN_CONJUGATE = 0.6180339887498949


def _normalize(class_name: str) -> str:
    return str(class_name or "").strip().lower()


def _class_digest(class_name: str) -> bytes:
    return hashlib.md5(_normalize(class_name).encode("utf-8")).digest()


def _hls_to_rgb255(h: float, s: float) -> RGB:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, _BASE_LIGHTNESS, s)
    return (round(r * 255), round(g * 255), round(b * 255))


def base_color_for_rank(rank: int) -> RGB:
    """Couleur de base RGB (0-255) pour un **rang** stable de classe.

    Teinte répartie par le nombre d'or sur le rang (les rangs consécutifs sont
    maximalement écartés sur le cercle chromatique → distinction garantie pour
    des classes ajoutées séquentiellement) × saturation cyclique (sépare encore
    les rares paires de rangs aux teintes voisines). C'est la voie nominale,
    alimentée par le registre ``class_name → rang``.
    """
    rank = max(0, int(rank))
    h = (rank * _GOLDEN_CONJUGATE) % 1.0
    s = _SATURATION_LEVELS[rank % len(_SATURATION_LEVELS)]
    return _hls_to_rgb255(h, s)


def hue_for_class(class_name: str) -> float:
    """Teinte [0.0, 1.0) stable et déterministe pour un nom de classe (fallback)."""
    n = int.from_bytes(_class_digest(class_name)[:8], "big") / float(1 << 64)
    return (n + _GOLDEN_CONJUGATE) % 1.0


def base_color_for_class(class_name: str) -> RGB:
    """Repli **sans registre** : couleur dérivée directement du nom (hash).

    Utilisé quand aucun rang n'est disponible (contexte sans profil). Stable et
    déterministe, mais sans garantie de distinction entre classes — la voie
    nominale est :func:`base_color_for_rank` via le registre.
    """
    h = hue_for_class(class_name)
    s = _SATURATION_LEVELS[_class_digest(class_name)[8] % len(_SATURATION_LEVELS)]
    return _hls_to_rgb255(h, s)


def _lighten(rgb: RGB, factor: float) -> RGB:
    r, g, b = rgb
    return (
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


def _darken(rgb: RGB, factor: float) -> RGB:
    r, g, b = rgb
    return (int(r * (1 - factor)), int(g * (1 - factor)), int(b * (1 - factor)))


def apply_confidence(base_rgb: RGB, confidence: float) -> RGB:
    """Module la luminosité d'une couleur de base selon la confiance.

    Paliers identiques à l'ancien ``get_color_for_confidence`` :
    haute confiance → plus sombre, basse confiance → plus clair.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    # Tolère une confiance exprimée en pourcentage (ex. 85 → 0.85).
    if conf > 1.0:
        conf = conf / 10.0 if conf <= 10.0 else 1.0
    conf = max(0.0, min(1.0, conf))

    if conf >= 0.8:
        return _darken(base_rgb, 0.30)
    if conf >= 0.6:
        return _darken(base_rgb, 0.15)
    if conf >= 0.4:
        return tuple(base_rgb)  # type: ignore[return-value]
    if conf >= 0.2:
        return _lighten(base_rgb, 0.35)
    return _lighten(base_rgb, 0.65)
