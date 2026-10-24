"""Rangement des erreurs de configuration sur l'étape et le champ qui les corrigent.

Pur : les messages viennent de :func:`app.run_context.validate_run_context` (et des
contrôles de bornes de :mod:`app.services.rvt_kernel_context`) ; le rail de
l'assistant pose sa pastille « à corriger » sur l'étape rendue ici, et l'étape 4
peut dire « Étape 2 : … ». Un message qu'aucune règle ne reconnaît va sous la
clé ``0`` : il reste listé à l'étape 4, sans pastille — le test
``tests/unit/test_validation_ui.py`` exige qu'il n'y en ait aucun.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence, Tuple

ETAPE_DONNEES = 1
ETAPE_IMAGES = 2
ETAPE_STRUCTURES = 3

#: (motif, étape, champ) — la première règle qui reconnaît le message gagne.
_REGLES: Tuple[Tuple[str, int, str], ...] = (
    (r"^Indiquez le dossier de sortie", ETAPE_DONNEES, "sortie"),
    (r"^Le dossier de sortie est dans", ETAPE_DONNEES, "sortie"),
    (r"^Choisissez un point de départ", ETAPE_DONNEES, "source"),
    (r"^Point de départ inconnu", ETAPE_DONNEES, "source"),
    (r"^Indiquez (la zone d'étude|le dossier)", ETAPE_DONNEES, "source"),
    (r"^Ce (fichier|dossier) est introuvable", ETAPE_DONNEES, "source"),
    (r"^Ce fichier \.dbf", ETAPE_DONNEES, "source"),
    (r"^Cochez au moins une image", ETAPE_IMAGES, "images"),
    (r"^(Image|Indice) [A-Z0-9-]+ / «", ETAPE_IMAGES, "images"),
    (r"^MSTP / «", ETAPE_IMAGES, "images"),
    (r"^Marge entre dalles", ETAPE_IMAGES, "images"),
    (r"^(Analyse|Modèle|Structure) ", ETAPE_STRUCTURES, "structures"),
)


def _regle(message: str) -> Tuple[int, str]:
    for motif, etape, champ in _REGLES:
        if re.search(motif, message):
            return etape, champ
    return 0, ""


def classer_erreurs(errors: Sequence[str]) -> Dict[int, List[str]]:
    """``{étape: [messages…]}`` ; ``0`` = non reconnu."""
    out: Dict[int, List[str]] = {}
    for message in errors:
        out.setdefault(_regle(message)[0], []).append(message)
    return out


def champ_de_l_erreur(message: str) -> str:
    """``"source"`` | ``"sortie"`` | ``"images"`` | ``"structures"`` | ``""``."""
    return _regle(message)[1]


def etat_rail(etape: int, courante: int, erreurs: Mapping[int, Sequence[str]]) -> str:
    """État d'une étape du rail : ``active`` (affichée), ``a_corriger`` (porte une
    erreur), ``ok`` (déjà parcourue, sans erreur), ``a_venir``."""
    if etape == courante:
        return "active"
    if erreurs.get(etape):
        return "a_corriger"
    return "ok" if etape < courante else "a_venir"
