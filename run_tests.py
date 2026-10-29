#!/usr/bin/env python3
"""
Point d'entrée unique pour lancer tous les tests du projet.

Usage :
    python run_tests.py                 # Tous les tests, en parallèle (-n auto) si pytest-xdist est installé
    python run_tests.py --serie         # En série (un seul processus)
    python run_tests.py unit            # Tests unitaires uniquement
    python run_tests.py integration     # Tests d'intégration uniquement
    python run_tests.py -k helpers      # Tests dont le nom contient "helpers"
"""

import importlib.util
import sys
from pathlib import Path

# S'assurer que src/ est dans le PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def arguments(argv: list, xdist: bool) -> list:
    """Arguments passés à pytest (pur, testé par tests/unit/test_run_tests.py).

    ``--serie`` ou un ``-n`` explicite désactivent le ``-n auto`` par défaut ;
    ``-q`` demandé → pas de ``-v`` ajouté (sinon les deux s'annulent)."""
    args = [a for a in argv if a != "--serie"]
    serie = len(args) != len(argv)

    # Raccourcis : "unit" → tests/unit, "integration" → tests/integration
    if args and args[0] in ("unit", "integration"):
        args[0] = f"tests/{args[0]}"

    # Arguments par défaut si aucun n'est fourni
    if not args:
        args = ["tests/"]

    if not any(a in ("-v", "--verbose", "-q", "--quiet") for a in args):
        args.append("-v")

    n_donne = any(a == "-n" or a.startswith("-n") or a.startswith("--numprocesses") for a in args)
    if xdist and not serie and not n_donne:
        args += ["-n", "auto"]
    return args


def main() -> int:
    try:
        import pytest
    except ImportError:
        print("[ERROR] pytest n'est pas installé.")
        print("        pip install -r dev/requirements/test.txt")
        return 1

    return pytest.main(arguments(sys.argv[1:], xdist=importlib.util.find_spec("xdist") is not None))


if __name__ == "__main__":
    sys.exit(main())
