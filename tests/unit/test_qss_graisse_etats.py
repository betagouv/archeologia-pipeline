"""Le QSS ne doit pas changer la POLICE sur un état (:checked, :selected…).

Qt dimensionne un widget avec la police de son état au repos, puis peint le
texte avec celle de son état courant. Une règle QSS du type
``QTabBar::tab:selected { font-weight: bold; }`` élargit donc le texte peint
sans élargir la boîte : le libellé se fait rogner des deux côtés.

Ça a été livré deux fois — « Nouveau traitement » affiché « ouveau traiteme »,
puis « Tous (10) » affiché « ous (10 ». Un contrôle au rendu ne l'attrape pas
(``sizeHint()`` ignore la graisse venue du QSS), donc on verrouille la cause :
la déclaration elle-même.

Changer la COULEUR ou le FOND sur un état reste évidemment permis — c'est comme
ça qu'on marque un état sans toucher aux métriques.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

QSS = Path(__file__).resolve().parents[2] / "src" / "ui" / "theme" / "v2.qss"

#: Pseudo-états qui n'existent pas au moment où Qt calcule la taille du widget.
ETATS = (":checked", ":selected", ":hover", ":pressed", ":focus", ":on",
         ":open", ":active")

#: Propriétés qui changent les métriques du texte.
POLICE = re.compile(r"\bfont(-weight|-size|-family|-style)?\s*:", re.IGNORECASE)

#: Un commentaire /* … */ ne doit pas être lu comme une règle.
COMMENTAIRES = re.compile(r"/\*.*?\*/", re.DOTALL)


def _regles():
    """(sélecteur, corps) de chaque règle du QSS, commentaires retirés."""
    texte = COMMENTAIRES.sub("", QSS.read_text(encoding="utf-8"))
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", texte):
        yield " ".join(m.group(1).split()), m.group(2)


def test_qss_present():
    assert QSS.is_file(), f"feuille de style introuvable : {QSS}"


def test_aucune_police_conditionnee_a_un_etat():
    fautifs = []
    for selecteur, corps in _regles():
        if not any(etat in selecteur for etat in ETATS):
            continue
        for decl in corps.split(";"):
            if POLICE.search(decl):
                fautifs.append(f"{selecteur} {{ {decl.strip()} }}")

    assert not fautifs, (
        "Une règle QSS change la police sur un état : le texte sera peint plus "
        "large que la boîte calculée, donc rogné. Marquer l'état par la couleur "
        "ou le fond.\n  - " + "\n  - ".join(fautifs))


@pytest.mark.parametrize("etat", ETATS)
def test_le_controle_attrape_bien_la_faute(etat):
    """Le test ci-dessus doit échouer sur la faute qu'il prétend interdire."""
    faux_qss = f"#Bouton{etat} {{ background: #fff; font-weight: bold; }}"
    trouve = [
        decl for sel, corps in [(faux_qss.split("{")[0].strip(),
                                 faux_qss.split("{")[1].rstrip("}"))]
        if any(e in sel for e in ETATS)
        for decl in corps.split(";") if POLICE.search(decl)
    ]
    assert trouve, f"la règle sur {etat} aurait dû être signalée"


def test_la_couleur_sur_un_etat_reste_permise():
    """Marquer un état par la couleur ne touche pas aux métriques : autorisé."""
    corps = " background: #d6e6f4; color: #1d5a96; border-color: #2b79c2; "
    assert not [d for d in corps.split(";") if POLICE.search(d)]
