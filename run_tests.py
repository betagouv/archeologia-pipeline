#!/usr/bin/env python3
"""
Point d'entrée unique pour lancer tous les tests du projet.

Usage :
    python run_tests.py                 # Tous les tests
    python run_tests.py unit            # Tests unitaires uniquement
    python run_tests.py integration     # Tests d'intégration uniquement
    python run_tests.py -k helpers      # Tests dont le nom contient "helpers"
"""

import sys
from pathlib import Path

# S'assurer que src/ est dans le PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    try:
        import pytest
    except ImportError:
        print("[ERROR] pytest n'est pas installé.")
        print("        pip install -r dev/requirements/test.txt")
        return 1

    args = sys.argv[1:]

    # Raccourcis : "unit" → tests/unit, "integration" → tests/integration
    if args and args[0] in ("unit", "integration"):
        args[0] = f"tests/{args[0]}"

    # Arguments par défaut si aucun n'est fourni
    if not args:
        args = ["tests/"]

    # Toujours ajouter -v pour la lisibilité
    if "-v" not in args and "--verbose" not in args:
        args.append("-v")

    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
