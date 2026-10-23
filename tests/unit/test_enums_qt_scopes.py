"""Énumérations Qt scopées dans ``src/`` (CLAUDE.md § Qt5/Qt6 ; lot 5 du chantier d'outillage, 2026-10-20).

L'UI doit tourner sous Qt5 (QGIS 3.34+) et Qt6 (QGIS 4.x) : Qt6 a retiré l'accès plat ``Qt.AlignCenter``,
seule la forme scopée ``Qt.AlignmentFlag.AlignCenter`` existe des deux côtés. ``src/ui`` n'étant pas
importé par pytest (pas de QGIS), un énuméré plat ne se voyait qu'au lancement dans QGIS 4. Ce test
reprend les trois balayages de CLAUDE.md, par AST (les docstrings et commentaires ne comptent pas) :

- racines ``Q[A-Z]…`` (QFrame, QEvent, QScrollArea…), ``Qt``, ``Qgs…`` et ``Qgis`` ;
- tout accès terminal ``QClasse.Valeur`` (majuscule, non suivi d'un autre ``.Valeur``, pas un appel) est
  un énuméré plat ;
- ``.exec_()`` et ``.print_()`` sont interdits (``.exec()`` / ``.print()``) ; ``.raise_()`` et
  ``.lower_()`` restent (mots-clés Python).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_RACINES = re.compile(r"^(Q[A-Z]\w*|Qt|Qgs\w*|Qgis)$")
_APPELS_INTERDITS = {"exec_": "exec", "print_": "print"}


def acces_plats(source: str, nom: str = "<source>") -> list[str]:
    """Les énumérés plats et appels interdits d'un source, ``"ligne: Qt.AlignCenter"``."""
    arbre = ast.parse(source, nom)
    parents = {enfant: noeud for noeud in ast.walk(arbre) for enfant in ast.iter_child_nodes(noeud)}
    constats = []
    for n in ast.walk(arbre):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _APPELS_INTERDITS:
            constats.append(f"{n.lineno}: .{n.func.attr}() → .{_APPELS_INTERDITS[n.func.attr]}()")
        if not (isinstance(n, ast.Attribute) and n.attr[:1].isupper()
                and isinstance(n.value, ast.Name) and _RACINES.match(n.value.id)):
            continue
        p = parents.get(n)
        if isinstance(p, ast.Attribute) and p.value is n:     # Qt.AlignmentFlag.AlignCenter : scopé
            continue
        if isinstance(p, ast.Call) and p.func is n:            # QClasse.Methode(...) : un appel
            continue
        constats.append(f"{n.lineno}: {n.value.id}.{n.attr}")
    return constats


def test_aucun_enumere_plat_dans_src():
    constats = []
    for fichier in sorted(_SRC.rglob("*.py")):
        for c in acces_plats(fichier.read_text(encoding="utf-8"), str(fichier)):
            constats.append(f"{fichier.relative_to(_SRC).as_posix()}:{c}")
    assert constats == [], "énumérés plats (Qt6 les refuse) — scoper QClasse.Type.Valeur :\n" + "\n".join(constats)


def test_le_detecteur_voit_le_plat_et_tolere_le_scope():
    source = (
        "from qgis.PyQt.QtCore import Qt, QEvent\n"
        "from qgis.PyQt.QtWidgets import QFrame, QDialog\n"
        "from qgis.core import Qgis, QgsWkbTypes, QgsProject\n"
        "a = Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter\n"   # scopé
        "b = Qt.AlignCenter\n"                                                 # plat
        "c = QFrame.Shape.HLine\n"                                             # scopé
        "d = QEvent.Resize\n"                                                  # plat
        "e = Qgis.MessageLevel.Critical\n"                                     # scopé
        "f = Qgis.Critical\n"                                                  # plat
        "g = QgsWkbTypes.PolygonGeometry\n"                                    # plat
        "h = QgsProject.instance()\n"                                          # appel : toléré
        "i = QDialog.DialogCode.Accepted\n"                                    # scopé
        "dlg = QDialog()\n"
        "dlg.exec_()\n"                                                        # interdit
        "dlg.exec()\n"
        "dlg.raise_()\n"                                                       # toléré
        "'''docstring : Qt.AlignCenter ne compte pas'''\n"
        "# commentaire : Qt.AlignLeft ne compte pas\n"
    )
    assert acces_plats(source) == ["5: Qt.AlignCenter", "7: QEvent.Resize", "9: Qgis.Critical",
                                   "10: QgsWkbTypes.PolygonGeometry", "14: .exec_() → .exec()"]
