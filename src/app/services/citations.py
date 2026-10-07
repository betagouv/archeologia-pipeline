"""Détecteur de citations restées en anglais (règle utilisateur 2026-09-21).

Les fiches et le manuel sont lus par des archéologues francophones : une citation
laissée en anglais est un trou dans la lecture. On traduit, on garde les guillemets.
Un nom propre laissé en VO (l'algorithme QGIS « Export to raster », la page RVT
« Choosing a visualization ») reste : il n'atteint jamais deux mots-outils distincts.

Même détecteur que ``scripts/validate_models_metadata.py`` (script autonome, qui
garde sa copie) ; les tests du plugin passent par ici.
"""
from __future__ import annotations

import re

_MOTS_OUTILS_EN = re.compile(
    r"\b(the|and|is|are|of|it|that|with|for|to|in|does|not|can|be|which|by|as"
    r"|from|on|you|your|this|these|all|more|than|its|was|were|has|have|but"
    r"|because|while|such|they|their|an|a)\b",
    re.IGNORECASE,
)


def citations_anglaises(texte: str) -> list[str]:
    """Citations « … » de ``texte`` qui contiennent ≥ 2 mots-outils anglais distincts."""
    return [
        m.group(1)
        for m in re.finditer(r"«\s*([^»]{3,}?)\s*»", texte)
        if len({w.lower() for w in _MOTS_OUTILS_EN.findall(m.group(1))}) >= 2
    ]
