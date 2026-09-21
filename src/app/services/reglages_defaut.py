"""Portée d'une réinitialisation aux valeurs par défaut — module pur.

Les deux étapes réglables offraient un bouton **global** : « Réinit. val. par
défaut » remettait d'un coup les paramètres de tous les produits (étape 2), et
« Réinit. val. défaut du modèle » effaçait les surcharges de **toutes** les
entités (étape 3). C'est trop brutal : on affine le rayon du Sky-View Factor, on
veut annuler ce réglage-là, et on perd au passage l'élévation solaire de
l'ombrage et le seuil de la couverture. Depuis le 2026-09-16 (demande
utilisateur), la portée d'une réinitialisation est **un produit**, ou **une
entité**.

Ce module porte cette portée, et rien d'autre : pas de Qt, pas de QGIS. Le
câblage vit dans ``ui/steps/step_2_indices.py``, ``ui/steps/step_3_detection.py``
et ``ui/widgets/entity_card.py``, qui ne sont pas collectés par pytest —
``tests/unit/test_reset_defauts_ui.py`` vérifie par AST qu'ils appellent bien
d'ici.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

#: Section de ``rvt_params`` → clé de produit. L'étape 2 range les paramètres
#: d'un indice sous un nom de section qui n'est PAS son code produit : celle du
#: Local Dominance est ``ldo`` et non ``ld``, celle du multi-hillshade ``mdh``.
#: Un test vérifie que chaque indice du pipeline est couvert.
SECTION_PRODUIT = {
    "hs": "HS",
    "mdh": "M_HS",
    "svf": "SVF",
    "opns": "OPNS",
    "slope": "SLO",
    "ldo": "LD",
    "slrm": "SLRM",
    "vat": "VAT",
    "mstp": "MSTP",
    "cvat": "CVAT",
    "prism": "PRISM",
    "crim": "CRIM",
}

#: Pseudo-produit du réglage de tuilage. Il vaut pour tous les indices à la fois
#: et n'a donc pas d'onglet : il porte son propre bouton sur sa carte. Sans cette
#: clé, il n'aurait plus aucune réinitialisation une fois le bouton global retiré.
TUILAGE = "_TUILAGE"


def produit_de_section(section: Sequence[str]) -> str:
    """Clé de produit d'une section de configuration. ``""`` si indéterminable.

    ``("rvt_params", "svf")`` → ``"SVF"``. La section ``("processing",)`` sert à
    MNT, Densité, Couverture ET au tuilage : elle ne détermine rien, et
    l'appelant doit alors nommer le produit lui-même.
    """
    if not section or len(section) < 2:
        return ""
    return SECTION_PRODUIT.get(section[1], "")


def champs_du_produit(
    champs: Iterable[Sequence[Any]], produit: str
) -> Tuple[Sequence[Any], ...]:
    """Les descripteurs de champ déclarés pour ``produit``, dans leur ordre.

    Un descripteur est une séquence dont le **premier** élément est la clé de
    produit ; le reste ne regarde pas ce module (l'étape 2 y met section, clé,
    widget, type et défaut).

    ⚠ C'est le produit déclaré qui fait la portée, **jamais la section de
    configuration** : MNT, Densité et Couverture écrivent tous les trois dans
    ``processing``, et les séparer par section rendrait les trois réinitialisations
    identiques.

    L'ordre de déclaration est celui de l'onglet : le préserver fait que les
    valeurs sont réécrites dans l'ordre où l'utilisateur les voit.
    """
    if not produit:
        return ()
    return tuple(c for c in champs if c and c[0] == produit)


def produits_declares(champs: Iterable[Sequence[Any]]) -> Tuple[str, ...]:
    """Les produits présents dans ``champs``, dédoublonnés, dans l'ordre vu.

    Sert au test de contrat : tout produit du pipeline doit avoir au moins un
    champ, sans quoi son onglet afficherait un bouton qui ne ferait rien.
    """
    vus: List[str] = []
    for c in champs:
        if c and c[0] and c[0] not in vus:
            vus.append(c[0])
    return tuple(vus)


def effacer_surcharges(cle: str, *dicos: MutableMapping[str, Any]) -> int:
    """Retire ``cle`` de chaque dictionnaire de surcharges. Rend le nombre touché.

    Les dictionnaires sont modifiés **sur place** : ce sont ceux que l'étape 3
    relit pour construire ses runs. On retire l'entrée plutôt que d'y réécrire la
    valeur du modèle — une valeur recopiée redeviendrait une surcharge, figée sur
    l'ancien modèle, et survivrait à un changement de modèle.

    Le compte rendu sert au message de confirmation : il dit ce qui a réellement
    bougé, et non le nombre de dictionnaires examinés.

    Une ``cle`` vide ne touche à rien : une entité sans identifiant viderait
    sinon une entrée ``""`` au hasard.
    """
    if not cle:
        return 0
    touches = 0
    for d in dicos:
        if isinstance(d, MutableMapping) and cle in d:
            del d[cle]
            touches += 1
    return touches


def a_des_surcharges(cle: str, *dicos: Mapping[str, Any]) -> bool:
    """Vrai si ``cle`` figure dans au moins un dictionnaire.

    C'est ce qui décide si le bouton d'une carte est actif : proposer de
    réinitialiser ce qui n'a jamais été réglé n'apprend rien à l'utilisateur.
    """
    if not cle:
        return False
    return any(isinstance(d, Mapping) and cle in d for d in dicos)


def phrase_produit_reinitialise(libelle: str, combien: int) -> str:
    """Confirmation d'une réinitialisation de produit. ``""`` si rien n'a bougé.

    Le message **nomme le produit** : c'est ce qui distingue une réinitialisation
    ciblée d'un retour global, et ce qui rassure qu'on n'a pas tout perdu.
    """
    if combien <= 0:
        return ""
    reglages = f"{combien} réglage{'s' if combien > 1 else ''}"
    return f"↺  {libelle} : {reglages} remis aux valeurs par défaut"


def phrase_entite_reinitialisee(libelle: str, combien: int) -> str:
    """Confirmation d'une réinitialisation d'entité. ``""`` si rien n'a bougé."""
    if combien <= 0:
        return ""
    return f"↺  {libelle} : valeurs du modèle rétablies"
