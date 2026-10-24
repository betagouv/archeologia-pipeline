"""Le lexique de l'interface (2026-10-24) tient : aucun mot de développeur dans
les textes que l'utilisateur lit. Vérifié par AST sur les chaînes des modules
d'interface et des modules purs qui lui fournissent ses phrases (narrateur,
validation, modes, étapes) — ``src/ui`` n'est pas importable hors QGIS."""
import ast
import re
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
FICHIERS = sorted((RACINE / "src" / "ui").rglob("*.py")) + [
    RACINE / "src" / "app" / "user_narrator.py",
    RACINE / "src" / "app" / "run_context.py",
    RACINE / "src" / "app" / "services" / "source_modes.py",
    RACINE / "src" / "app" / "progress_stages.py",
]

#: (motif interdit, mot du lexique qui le remplace)
INTERDITS = [
    (r"\bpipeline\b", "traitement"),
    (r"\binjecter\b", "cliquer sur ce que vous possédez"),
    (r"\bruns?\b", "traitement / analyse"),
    (r"\bworkers?\b", "dalles traitées en parallèle"),
    (r"\blog\b", "journal"),
    (r"\btuiles?\b", "dalle"),
    (r"\bYes\b|\bCancel\b", "bouton nommé par le verbe de l'action"),
    (r"\bconfig\b", "configuration"),
    (r"\bindices?\b", "image du relief"),
    (r"\bproduits?\b", "image du relief"),
    (r"\bentités?\b", "structure"),
    (r"\bIA\b", "détection automatique"),
    (r"\bPATH\b", "libellé français (vocabulaire.ENVIRONNEMENT)"),
]

#: Textes légitimes qui contiennent un motif interdit (regex sur la chaîne entière).
ALLOWLIST = [
    r"^Ce qui sera produit",                 # participe, pas le nom
    r"index_",                               # noms de fichiers de mosaïques
    r"Classification = ",                    # filtre PDAL, texte technique attendu
    r"^Indice de confiance",                 # champ d'attribut des détections (QGIS)
    r"Archéolog'IA",                         # le nom du plugin
    r"=%s",                                  # journal de débogage (logging), jamais affiché
]


def _docstrings(arbre: ast.AST) -> set:
    ids = set()
    for node in ast.walk(arbre):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def chaines_visibles(fichier: Path):
    """``(ligne, texte)`` des constantes chaîne hors docstrings, avec au moins une espace."""
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    docs = _docstrings(arbre)
    for node in ast.walk(arbre):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            texte = node.value
            if " " in texte.strip():
                yield node.lineno, texte


def test_aucun_mot_de_developpeur_dans_les_textes_visibles():
    fautes = []
    for fichier in FICHIERS:
        for ligne, texte in chaines_visibles(fichier):
            if any(re.search(a, texte) for a in ALLOWLIST):
                continue
            for motif, remplacant in INTERDITS:
                if re.search(motif, texte, flags=re.IGNORECASE):
                    fautes.append(f"{fichier.relative_to(RACINE)}:{ligne}: {texte!r} → {remplacant}")
                    break
    if fautes:
        pytest.fail(f"{len(fautes)} texte(s) hors lexique :\n" + "\n".join(fautes))


def messages_de_validation():
    arbre = ast.parse((RACINE / "src" / "app" / "run_context.py").read_text(encoding="utf-8"))
    for node in ast.walk(arbre):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "append" and getattr(node.func.value, "id", "") == "errors":
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                yield arg.value
            elif isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
                yield arg.values[0].value


def test_messages_de_validation_disent_le_geste():
    msgs = list(messages_de_validation())
    assert msgs, "aucun errors.append(...) trouvé"
    fautes = [m for m in msgs if not m[0].isupper()
              or re.search(r"n'est configuré|sélectionné mais|\(data_mode\)|à générer", m)]
    assert not fautes, "messages encore passifs ou techniques :\n" + "\n".join(fautes)
