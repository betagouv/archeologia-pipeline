"""Ajoute des entrées ``vignettes[]`` à une fiche déjà écrite d'un model_card.

``injecter_fiches.py`` refuse d'écraser une fiche existante ; ce script complète
seulement sa liste ``vignettes`` (fiche qui n'avait qu'une image, cadres ajoutés
par ``appliquer_choix.py --rang-depart 2``). Chirurgie TEXTUELLE, comme pour les
fiches et les cadrages : les commentaires du model_card sont préservés.

Usage ::

    python dev/fiches/ajouter_vignettes.py <ajouts.json>

Le JSON est une liste de ``{modele, classe, vignettes: [{brut, annote, zone, legende}]}``,
``classe`` = ``name`` d'une classe ou ``output_class`` d'une cible dérivée. Une
vignette dont le ``brut`` est déjà déclaré est ignorée.
"""
import json
import os
import re
import sys

import yaml

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(RACINE, "data", "models")


def _indent(s):
    return len(s) - len(s.lstrip(" "))


def _yaml_str(v):
    v = str(v)
    if re.search(r"[:#'\"]|^\s|\s$", v) or v.lower() in ("yes", "no", "true", "false", "null"):
        return "'" + v.replace("'", "''") + "'"
    return v


def _bloc_classe(lignes, classe):
    """(début, fin) de l'entrée ``name: classe`` (ou ``output_class:``), clé sur la ligne du tiret ou dessous."""
    motif = re.compile(r"^(\s*)(-\s*)?(name|output_class):\s*['\"]?" + re.escape(classe) + r"['\"]?\s*$")
    for i, ligne in enumerate(lignes):
        m = motif.match(ligne)
        if not m:
            continue
        champs = len(m.group(1)) + (len(m.group(2)) if m.group(2) else 0)   # indentation des champs de l'entrée
        debut = i
        if not m.group(2):   # remonter jusqu'au tiret de l'entrée
            while debut > 0 and not (lignes[debut].lstrip().startswith("- ") or lignes[debut].strip() == "-"):
                debut -= 1
        tiret = _indent(lignes[debut])
        j = i + 1
        while j < len(lignes):
            s = lignes[j]
            if s.strip() and (_indent(s) < champs or (_indent(s) == tiret and s.lstrip().startswith("-"))):
                break
            j += 1
        return debut, j
    raise SystemExit(f"classe {classe!r} introuvable")


def ajouter(chemin, classe, vignettes):
    lignes = open(chemin, encoding="utf-8").read().split("\n")
    a, b = _bloc_classe(lignes, classe)
    bloc = "\n".join(lignes[a:b])
    vignettes = [v for v in vignettes if v["brut"] not in bloc]
    if not vignettes:
        return 0
    k = next((i for i in range(a, b) if re.match(r"^\s*vignettes:\s*$", lignes[i])), None)
    if k is None:
        raise SystemExit(f"{classe} : pas de liste 'vignettes' dans la fiche")
    ind = _indent(lignes[k])
    fin = k + 1
    while fin < b:
        s = lignes[fin]
        dans = (not s.strip()) or _indent(s) > ind or (_indent(s) == ind and s.lstrip().startswith("- "))
        if not dans:
            break
        fin += 1
    while fin > k + 1 and not lignes[fin - 1].strip():   # insérer avant les lignes vides de fin
        fin -= 1
    item_ind = next((_indent(lignes[i]) for i in range(k + 1, fin) if lignes[i].lstrip().startswith("- ")), ind)
    pad = " " * item_ind
    nouvelles = []
    for v in vignettes:
        nouvelles.append(f"{pad}- brut: {v['brut']}")
        for cle in ("annote", "zone", "legende"):
            if v.get(cle):
                nouvelles.append(f"{pad}  {cle}: {_yaml_str(v[cle])}")
    lignes[fin:fin] = nouvelles
    open(chemin, "w", encoding="utf-8", newline="\n").write("\n".join(lignes))
    return len(vignettes)


def verifier(chemin, classe, vignettes):
    card = yaml.safe_load(open(chemin, encoding="utf-8"))
    for rub, cle in (("classes", "name"), ("derived_targets", "output_class")):
        for c in card.get(rub) or []:
            if c.get(cle) == classe:
                bruts = [v["brut"] for v in c["fiche"]["vignettes"]]
                manque = [v["brut"] for v in vignettes if v["brut"] not in bruts]
                if manque:
                    raise SystemExit(f"{classe} : relecture KO, absents {manque}")
                for v in c["fiche"]["vignettes"]:
                    for cle_f in ("brut", "annote"):
                        p = os.path.join(os.path.dirname(chemin), v.get(cle_f, ""))
                        if v.get(cle_f) and not os.path.isfile(p):
                            raise SystemExit(f"{classe} : fichier absent {v[cle_f]}")
                return len(bruts)
    raise SystemExit(f"{classe} : introuvable à la relecture")


def main():
    entrees = json.load(open(sys.argv[1], encoding="utf-8"))
    for e in entrees:
        chemin = os.path.join(MODELS, e["modele"], "model_card.yaml")
        n = ajouter(chemin, e["classe"], e["vignettes"])
        total = verifier(chemin, e["classe"], e["vignettes"])
        print(f"{e['modele']}/{e['classe']:30s} +{n} vignette(s) -> {total} au total, relecture OK")


if __name__ == "__main__":
    main()
