"""Skills partagés entre le plugin et training-models : vérifier ou recopier.

Les skills de l'installation d'un modèle dans le plugin (`installer-modele-plugin`,
`fiche-classe-plugin`) vivent, à l'identique, dans `.claude/skills/` des DEUX
dépôts : celui qui entraîne (C:/projets/Archeologia/training-models) et celui qui
consomme (ce plugin). Modifier l'un sans l'autre est une régression que les deux
suites de tests signalent (`tests/unit/test_skills_partages.py` ici,
`tests/test_skills_partages.py` là-bas). La comparaison ignore les fins de ligne
(git autocrlf réécrit CRLF côté training-models).

Usage ::

    python dev/sync_skills_training.py --check            # diff, code 2 si divergence
    python dev/sync_skills_training.py --vers-training    # plugin -> training-models
    python dev/sync_skills_training.py --depuis-training  # training-models -> plugin
    [--training <racine training-models>]
"""
import argparse
import os
import shutil
import sys

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = "C:/projets/Archeologia/training-models"
PARTAGES = ("installer-modele-plugin", "fiche-classe-plugin")


def _normalise(chemin):
    with open(chemin, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def chemins(racine, skill):
    d = os.path.join(racine, ".claude", "skills", skill)
    return [os.path.join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []


def comparer(training):
    """Liste (skill, fichier, état) ; état ∈ identique / différent / absent côté X."""
    etats = []
    for skill in PARTAGES:
        a = {os.path.basename(p): p for p in chemins(PLUGIN, skill)}
        b = {os.path.basename(p): p for p in chemins(training, skill)}
        for nom in sorted(set(a) | set(b)):
            if nom not in a:
                etats.append((skill, nom, "absent côté plugin"))
            elif nom not in b:
                etats.append((skill, nom, "absent côté training-models"))
            elif _normalise(a[nom]) == _normalise(b[nom]):
                etats.append((skill, nom, "identique"))
            else:
                etats.append((skill, nom, "différent"))
    return etats


def copier(src_racine, dst_racine):
    for skill in PARTAGES:
        dst = os.path.join(dst_racine, ".claude", "skills", skill)
        os.makedirs(dst, exist_ok=True)
        for p in chemins(src_racine, skill):
            shutil.copy(p, os.path.join(dst, os.path.basename(p)))
            print(f"copié {skill}/{os.path.basename(p)} -> {dst_racine}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--vers-training", action="store_true")
    g.add_argument("--depuis-training", action="store_true")
    ap.add_argument("--training", default=TRAINING)
    a = ap.parse_args()
    if not os.path.isdir(a.training):
        sys.exit(f"training-models introuvable : {a.training}")
    if a.vers_training:
        copier(PLUGIN, a.training)
    elif a.depuis_training:
        copier(a.training, PLUGIN)
    etats = comparer(a.training)
    for skill, nom, etat in etats:
        print(f"{skill}/{nom:12s} {etat}")
    if any(e != "identique" for _, _, e in etats):
        print("DIVERGENCE : recopier avec --vers-training ou --depuis-training")
        sys.exit(2)
    print("skills partagés identiques dans les deux dépôts")


if __name__ == "__main__":
    main()
