"""Réglage hors bornes dans une configuration chargée : l'étape 2 l'écrête (setValue), désormais en le disant ;
et une erreur de bornes_errors se range sur l'étape 2 (pastille du rail), pas sur l'étape 1."""
import ast
import os

import pytest

from app.services.rvt_kernel_context import bornes_errors, message_ecretage

WIZARD = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ui", "wizard_dialog.py")


def test_ecretage_nomme_le_reglage_la_valeur_lue_et_la_valeur_retenue():
    m = message_ecretage("LD", "angular_res", 2, 4, 32, False)
    assert "LD" in m and "Résolution angulaire" in m and "2" in m and "4 retenu" in m
    assert "150" in message_ecretage("LD", "max_radius", 150, 1, 100, False)
    assert "100 retenu" in message_ecretage("LD", "max_radius", 150, 1, 100, False)
    assert "« radius »" in message_ecretage("SVF", "radius", 0, 1, 50, False)   # sans libellé connu : la clé


@pytest.mark.parametrize("lue", [4, 32, "10", 12.9, None, "abc"])
def test_pas_de_message_dans_les_bornes_ou_sans_nombre(lue):
    assert message_ecretage("LD", "angular_res", lue, 4, 32, False) is None


def _classify_error_step():
    arbre = ast.parse(open(WIZARD, encoding="utf-8").read())
    fn = next(n for n in arbre.body if isinstance(n, ast.FunctionDef) and n.name == "_classify_error_step")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), WIZARD, "exec"), ns)  # noqa: S102
    return ns["_classify_error_step"]


@pytest.mark.parametrize("produit,section,valeurs", [
    ("LD", "ldo", {"min_radius": 30, "max_radius": 20}),
    ("LD", "ldo", {"angular_res": 2}),
    ("SLRM", "slrm", {"radius": 80}),
    ("MSTP", "mstp", {"lightness": 6.0}),
])
def test_erreur_de_bornes_rangee_sur_l_etape_2(produit, section, valeurs):
    classer = _classify_error_step()
    erreurs = bornes_errors({produit: True}, {section: valeurs})
    assert erreurs and all(classer(e) == 2 for e in erreurs)
