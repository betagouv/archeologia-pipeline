"""Compare les model_card du plugin et de l'archive Drive, hors bloc `fiche`.

Lecture seule. Sert à décider si le report des fiches vers G: peut être purement
additif (les deux copies disent la même chose par ailleurs) ou si les deux ont
divergé et qu'il faut le signaler avant d'écrire.
"""
import json
import os

import yaml

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(RACINE, "data", "models")
DRIVE = r"G:/Mon Drive/Archeologia/Archeologia_Shared/model-training"

BUNDLES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "bundles_drive.json"), encoding="utf-8"))


def sans_fiche(card):
    """Copie profonde du model_card, bloc `fiche` retiré de chaque classe."""
    c = json.loads(json.dumps(card, default=str))
    for cl in c.get("classes") or []:
        if isinstance(cl, dict):
            cl.pop("fiche", None)
    return c


def chemins_differents(a, b, prefixe=""):
    """Liste des chemins de clés où les deux structures divergent."""
    if type(a) is not type(b):
        return [prefixe or "(racine)"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            # a = copie Drive, b = copie plugin (cf. l'appel plus bas).
            if k not in a:
                out.append(f"{prefixe}{k} (seulement dans le plugin)")
            elif k not in b:
                out.append(f"{prefixe}{k} (seulement sur Drive)")
            else:
                out += chemins_differents(a[k], b[k], f"{prefixe}{k}.")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{prefixe[:-1]} (longueurs {len(a)} vs {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += chemins_differents(x, y, f"{prefixe[:-1]}[{i}].")
        return out
    return [] if a == b else [prefixe[:-1] or "(valeur)"]


for nom, rel in BUNDLES.items():
    p_plug = os.path.join(PLUGIN, nom, "model_card.yaml")
    p_drive = os.path.join(DRIVE, rel, "model_card.yaml")
    print("##", nom)
    if not os.path.isfile(p_drive):
        print("    Drive : ABSENT")
        continue
    plug = yaml.safe_load(open(p_plug, encoding="utf-8"))
    drive = yaml.safe_load(open(p_drive, encoding="utf-8"))

    classes_plug = [c.get("name") for c in plug.get("classes") or []]
    classes_drive = [c.get("name") for c in drive.get("classes") or []]
    if classes_plug != classes_drive:
        print(f"    CLASSES DIFFERENTES  plugin={classes_plug}  drive={classes_drive}")
        continue

    deja = [c.get("name") for c in drive.get("classes") or [] if c.get("fiche")]
    if deja:
        print(f"    fiche deja presente sur Drive pour : {deja}")

    ecarts = chemins_differents(sans_fiche(drive), sans_fiche(plug))
    if not ecarts:
        print(f"    identiques hors fiche — report additif sur {len(classes_plug)} classe(s)")
    else:
        print(f"    {len(ecarts)} ecart(s) hors fiche :")
        for e in ecarts[:12]:
            print("       -", e)
        if len(ecarts) > 12:
            print(f"       … et {len(ecarts) - 12} autre(s)")
