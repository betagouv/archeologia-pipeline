"""dev/fiches/README.md décrit bien les outils de dev/fiches (même garde que test_doc_cli de training-models).

Deux étages, par AST, sans importer les outils (ils lisent des chemins locaux) :
1. couverture : chaque dev/fiches/*.py qui a un ArgumentParser ou lit sys.argv est cité dans le README ;
2. doc -> code : chaque ``--option`` documentée dans le README existe dans l'argparse de l'outil cité.
"""
import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FICHES = os.path.join(ROOT, "dev", "fiches")
README = os.path.join(FICHES, "README.md")


def _outils():
    for nom in sorted(os.listdir(FICHES)):
        if nom.endswith(".py"):
            src = open(os.path.join(FICHES, nom), encoding="utf-8").read()
            if "ArgumentParser" in src or "sys.argv" in src:
                yield nom, ast.parse(src)


def _options_argparse(arbre):
    opts = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
            for a in n.args:
                if isinstance(a, ast.Constant) and str(a.value).startswith("--"):
                    opts.add(a.value)
    return opts


@pytest.mark.parametrize("nom,arbre", list(_outils()))
def test_outil_cite_et_options_reelles(nom, arbre):
    readme = open(README, encoding="utf-8").read()
    assert nom in readme, f"{nom} absent de dev/fiches/README.md"
    reelles = _options_argparse(arbre)
    # options documentées dans les blocs de commande qui appellent cet outil
    documentees = set()
    for bloc in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL):
        if nom in bloc:
            documentees |= set(re.findall(r"(--[a-z][a-z0-9-]*)", bloc))
    fantomes = sorted(documentees - reelles)
    assert not fantomes, f"{nom} : options documentées inexistantes {fantomes} (réelles : {sorted(reelles)})"
