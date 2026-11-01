"""Garde : un import relatif vers ``app`` doit remonter exactement jusqu'à ``src``.

Dans QGIS, le plugin est un paquet : ``from ...app`` écrit un dossier trop bas pointe vers un module
inexistant. Les tests ne le voient pas (``src/`` est sur le chemin, l'import de secours ``from app``
réussit), d'où ce contrôle statique (incident du 2026-11-01 : « No module named 'app' » sur chaque dalle
en existing_mnt, ``results.retirer_publication_remplacee``).
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
_IMPORT_APP = re.compile(r"from (\.+)app[. ]")


def test_profondeur_des_imports_relatifs_vers_app():
    fautifs = []
    for f in SRC.rglob("*.py"):
        attendu = len(f.relative_to(SRC).parts)  # dossiers entre src et le fichier, + 1
        for i, ligne in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = _IMPORT_APP.search(ligne)
            if m and len(m.group(1)) != attendu:
                fautifs.append(f"{f.relative_to(SRC)}:{i} {len(m.group(1))} points au lieu de {attendu}")
    assert not fautifs, fautifs
