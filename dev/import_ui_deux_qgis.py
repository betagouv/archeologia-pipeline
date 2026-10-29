#!/usr/bin/env python3
"""Importe chaque module de ``src/ui`` sous le Python de QGIS courant, Qt5 ou Qt6, et écrit le bilan.

Les deux QGIS du poste (le ``.bat`` avale stdout : tout va dans ``--sortie``) :

    C:/OSGeo4W/bin/python-qgis.bat dev/import_ui_deux_qgis.py --sortie <dossier>/import_ui_qt5.txt
    "C:/Program Files/QGIS 4.0.3/bin/python-qgis.bat" dev/import_ui_deux_qgis.py --sortie <dossier>/import_ui_qt6.txt

Complément du test AST ``tests/unit/test_enums_qt_scopes.py`` : l'import exécute le code de niveau
module (énumérés dans des constantes, signatures, décorateurs), ce que l'AST ne juge pas. Code de
retour 0 si tous les modules s'importent, 1 sinon ; la liste des échecs (avec trace) est dans le fichier.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
import types
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sortie", type=Path, required=True, help="fichier du bilan")
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Le plugin s'importe comme un paquet (imports relatifs ``...app``) : alias sur sa racine.
    paquet = types.ModuleType("archeo")
    paquet.__path__ = [str(RACINE)]
    sys.modules["archeo"] = paquet

    from qgis.core import Qgis
    from qgis.PyQt.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

    modules = sorted(p.relative_to(RACINE).with_suffix("").as_posix().replace("/", ".")
                     for p in (RACINE / "src" / "ui").rglob("*.py"))
    lignes = [f"QGIS {Qgis.version()} | Qt {QT_VERSION_STR} | PyQt {PYQT_VERSION_STR} | Python {sys.version.split()[0]}"]
    echecs = []
    t0 = time.perf_counter()
    for nom in modules:
        try:
            importlib.import_module(f"archeo.{nom}")
            lignes.append(f"OK     {nom}")
        except Exception:  # noqa: BLE001 — chaque échec est un constat, pas un arrêt
            echecs.append(nom)
            lignes.append(f"ECHEC  {nom}\n" + traceback.format_exc())
    lignes.append(f"{len(modules) - len(echecs)}/{len(modules)} modules importés en {time.perf_counter() - t0:.2f} s"
                  f" — RESULTAT : {'OK' if not echecs else 'ECHEC ' + ', '.join(echecs)}")
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return 0 if not echecs else 1


if __name__ == "__main__":
    sys.exit(main())
