"""Couleurs de couche pour le rapport de traitement — module pur (lot 6, 2026-10-24).

Le rapport dessinait ses barres de bilan en bleu quel que soit la couche ; dans QGIS
chaque couche a sa couleur (registre ``class_color_registry.json`` du profil, rang
stable par nom de couche) déclinée par niveau de fiabilité. Ici : la lecture du
registre en JSON pur (rang = position dans la liste, même normalisation que le
registre), la palette et la déclinaison du pipeline importées **dans les fonctions**
(``pipeline.cv`` tire shapely) et remplaçables par injection pour les tests, avec un
repli en clarté quand le pipeline n'est pas importable.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .fiabilite import STYLE_SPEC

RGB = Tuple[int, int, int]
Palette = Callable[[int], RGB]
Declinaison = Callable[[RGB, float], RGB]


def lire_registre(chemin: Path) -> Dict[str, Any]:
    """Le registre ``{"classes": [...]}`` ; ``{}`` s'il manque ou est illisible."""
    try:
        d = json.loads(Path(chemin).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _normaliser(nom: str) -> str:
    return str(nom or "").strip().lower()     # même règle que class_color_registry._normalize


def rang_de_couche(registre: Mapping[str, Any], couche: str) -> Optional[int]:
    """Le rang (position) de ``couche`` dans le registre, ``None`` si elle n'y est pas."""
    classes = registre.get("classes") if isinstance(registre, Mapping) else None
    if not isinstance(classes, list):
        return None
    voulu = _normaliser(couche)
    if not voulu:
        return None
    for i, nom in enumerate(classes):
        if _normaliser(nom) == voulu:
            return i
    return None


def hex_rgb(rgb: RGB) -> str:
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _palette_defaut(rang: int) -> RGB:
    try:
        from ...pipeline.cv.color_palette import base_color_for_rank
    except ImportError:
        from pipeline.cv.color_palette import base_color_for_rank  # type: ignore[no-redef]
    return base_color_for_rank(rang)


def _declinaison_clarte(base: RGB, repr_: float) -> RGB:
    """Repli sans le pipeline : la règle de ``profil_scores._teinte`` hors QGIS (plus sûr
    = plus sombre, douteux = plus clair), en clarté HSV comme ``QColor.darker/lighter``."""
    r, g, b = (max(0.0, min(1.0, v / 255.0)) for v in base)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if repr_ >= 0.5:
        v = v / ((100 + (repr_ - 0.5) * 80) / 100.0)
    else:
        v = min(1.0, v * ((100 + (0.5 - repr_) * 120) / 100.0))
    return tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, s, v))  # type: ignore[return-value]


def _declinaison_defaut(base: RGB, repr_: float) -> RGB:
    try:
        try:
            from ...pipeline.cv.color_palette import apply_confidence
        except ImportError:
            from pipeline.cv.color_palette import apply_confidence  # type: ignore[no-redef]
        return tuple(int(v) for v in apply_confidence(base, repr_))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 — pipeline non importable (shapely, osgeo) : repli
        return _declinaison_clarte(base, repr_)


def couleur_hex_de_couche(registre: Mapping[str, Any], couche: str,
                          palette: Optional[Palette] = None) -> Optional[str]:
    """La couleur de base de ``couche`` (``#rrggbb``) d'après son rang, ``None`` si inconnue."""
    rang = rang_de_couche(registre, couche)
    if rang is None:
        return None
    return hex_rgb((palette or _palette_defaut)(rang))


def teintes_par_niveau(base: RGB, declinaison: Optional[Declinaison] = None) -> Dict[str, str]:
    """``{catégorie: #rrggbb}`` : la couleur de base déclinée par niveau, comme la légende
    de QGIS (``STYLE_SPEC[...]["repr"]`` : très probable assombri, douteux éclairci)."""
    decliner = declinaison or _declinaison_defaut
    return {cat: hex_rgb(decliner(base, float(spec.get("repr", 0.5)))) for cat, spec in STYLE_SPEC.items()}


def couleurs_du_bilan(registre: Mapping[str, Any], couches: Iterable[str],
                      palette: Optional[Palette] = None,
                      declinaison: Optional[Declinaison] = None) -> Dict[str, Dict[str, str]]:
    """``{couche: {catégorie: #rrggbb}}`` pour les couches connues du registre ; une couche
    absente n'a pas d'entrée (le rapport garde alors ses bleus)."""
    out: Dict[str, Dict[str, str]] = {}
    peindre = palette or _palette_defaut
    for couche in couches:
        rang = rang_de_couche(registre, couche)
        if rang is None:
            continue
        out[couche] = teintes_par_niveau(peindre(rang), declinaison)
    return out
