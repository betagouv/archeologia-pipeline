"""Hook Claude Code « Stop » : la suite de tests avant de rendre la main.

Lancé par `.claude/settings.json` à chaque fin de tour. Si des fichiers de `src/` ou `tests/`
ont changé (suivis ou non), lance `python run_tests.py -q -x` avec le Python du dépôt ; une
suite rouge bloque la fin de tour (code 2, sortie sur stderr) pour que l'agent corrige au lieu
d'affirmer que « les tests passent ». Rien à faire : code 0, silencieux.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
VENV = RACINE / ".venv_dev" / "Scripts" / "python.exe"
DOSSIERS = ("src", "tests")


def _python() -> str:
    return str(VENV) if VENV.exists() else sys.executable


def _fichiers_modifies() -> list[str]:
    sortie = subprocess.run(
        ["git", "status", "--porcelain", "--", *DOSSIERS],
        cwd=RACINE, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return [ligne[3:] for ligne in sortie.stdout.splitlines() if ligne.strip()]


def main() -> int:
    try:
        entree = json.load(sys.stdin)
    except Exception:  # noqa: BLE001 — stdin vide ou non JSON : on fait sans
        entree = {}
    if entree.get("stop_hook_active"):
        return 0  # déjà relancé par ce hook : pas de boucle
    modifies = _fichiers_modifies()
    if not modifies:
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    resultat = subprocess.run(
        [_python(), "run_tests.py", "-q", "-x", "-p", "no:cacheprovider"],
        cwd=RACINE, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    if resultat.returncode == 0:
        return 0
    lignes = (resultat.stdout + resultat.stderr).splitlines()
    print("Suite de tests rouge (hook Stop) après modification de "
          + ", ".join(modifies[:6]) + (" …" if len(modifies) > 6 else "") + " :", file=sys.stderr)
    print("\n".join(lignes[-40:]), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
