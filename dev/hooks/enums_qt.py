"""Hook Claude Code « PostToolUse » (Edit|Write) : énumérations Qt plates et ruff sur le fichier touché.

Un énum plat (`Qt.AlignCenter`, `QImage.Format_ARGB32`, `Qgis.Critical`) ne casse qu'à l'exécution
sous Qt6, hors de portée de pytest : ce hook reprend les trois balayages de CLAUDE.md § Qt5/Qt6
sur chaque fichier Python édité sous `src/` et bloque (code 2) tant qu'il en reste, ainsi que
`.exec_(` et `.print_(`. `.raise_()` et `.lower_()` restent permis (mots-clés Python). Puis `ruff`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
VENV = RACINE / ".venv_dev" / "Scripts" / "python.exe"
MOTIFS = (
    re.compile(r"\bQ[A-Z]\w*\.[A-Z]\w+"),            # classes Qt (QImage.Format…)
    re.compile(r"\bQt\.[A-Z]\w+"),                    # espace de noms Qt
    re.compile(r"\bQgs\w*\.[A-Z]\w+|\bQgis\.[A-Z]\w+"),  # classes QGIS
)
INTERDITS = re.compile(r"\.(exec_|print_)\(")


def _dans_chaine(ligne: str, position: int) -> bool:
    """Vrai si ``position`` est entre guillemets (approximation : parité des guillemets avant)."""
    avant = ligne[:position]
    return avant.count('"') % 2 == 1 or avant.count("'") % 2 == 1


def enums_plats(texte: str) -> list[tuple[int, str]]:
    trouves: list[tuple[int, str]] = []
    for numero, ligne in enumerate(texte.splitlines(), 1):
        if ligne.lstrip().startswith("#"):
            continue
        for motif in MOTIFS:
            for m in motif.finditer(ligne):
                if _dans_chaine(ligne, m.start()):
                    continue
                suite = ligne[m.end():m.end() + 2]
                # Forme scopée (`QClass.EnumType.Value`) ou appel (`QClass.methode(`) : correct.
                if re.match(r"\.[A-Za-z_]", suite) or suite.startswith("("):
                    continue
                trouves.append((numero, m.group(0)))
        for m in INTERDITS.finditer(ligne):
            if not _dans_chaine(ligne, m.start()):
                trouves.append((numero, m.group(0)))
    return trouves


def main() -> int:
    try:
        entree = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0
    chemin = (entree.get("tool_input") or {}).get("file_path") or ""
    if not chemin:
        return 0
    fichier = Path(chemin)
    try:
        relatif = fichier.resolve().relative_to(RACINE)
    except ValueError:
        return 0
    if relatif.parts[:1] != ("src",) or fichier.suffix != ".py" or not fichier.exists():
        return 0
    texte = fichier.read_text(encoding="utf-8", errors="replace")
    problemes = enums_plats(texte)
    messages = [f"  {relatif}:{n} : {token}" for n, token in problemes]
    python = str(VENV) if VENV.exists() else sys.executable
    ruff = subprocess.run(
        [python, "-m", "ruff", "check", str(fichier)],
        cwd=RACINE, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if ruff.returncode not in (0, None) and ruff.stdout.strip():
        messages.append(ruff.stdout.strip())
    if not messages:
        return 0
    if problemes:
        print("Énumération Qt plate ou appel retiré de Qt6 (forme scopée attendue : QClass.EnumType.Value) :",
              file=sys.stderr)
    print("\n".join(messages), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
