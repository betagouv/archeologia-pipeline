"""Écrit ``vignettes[].cadrage`` dans les model_card, et rend les icônes 44 px.

Le cadrage est la fenêtre carrée que découpe l'icône de la carte d'entité : une
vignette couvre 324 m de terrain, réduite telle quelle à 44 px elle est
illisible. Coordonnées en FRACTIONS de l'image, donc indépendantes de sa
résolution.

Usage ::

    python dev/fiches/injecter_cadrages.py <cadrages.json> [planche.png]

Le JSON est une liste d'entrées ::

    [{"modele": "tranchees_seg_ld_v1",
      "vignette": "vignettes/tranchees_et_boyaux_01_brut.jpg",
      "cadrage": {"x": 0.39, "y": 0.27, "cote": 0.26}}]

Insertion TEXTUELLE, comme pour les blocs ``fiche`` : les model_card portent des
commentaires (choix des seuils, calibrage de la fiabilité) qu'un round-trip
PyYAML effacerait. Rejouer le script sur une vignette déjà cadrée remplace la
clé au lieu d'en empiler une seconde.

La planche de contrôle rend chaque icône à sa taille réelle (44 px) puis
agrandie : c'est le seul moyen de voir si le cadrage tient la route.
"""
import json
import os
import sys

import yaml
from PIL import Image

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(RACINE, "data", "models")

ICONE = 44          # taille réelle de la vignette sur la carte d'entité
AGRANDI = 132       # ×3, pour juger le cadrage à l'œil


def _ligne(indent, c):
    return f"{' ' * indent}cadrage: {{x: {c['x']}, y: {c['y']}, cote: {c['cote']}}}\n"


def injecter(chemin, vignette_rel, cadrage):
    """Ajoute (ou remplace) la clé ``cadrage`` de la vignette visée."""
    lignes = open(chemin, encoding="utf-8").read().splitlines(keepends=True)
    cible = f"- brut: {vignette_rel}"
    for i, li in enumerate(lignes):
        if li.strip() != cible:
            continue
        indent = len(li) - len(li.lstrip(" ")) + 2
        j = i + 1
        while (j < len(lignes) and lignes[j].startswith(" " * indent)
               and not lignes[j].lstrip().startswith("- ")):
            if lignes[j].lstrip().startswith("cadrage:"):
                lignes[j] = _ligne(indent, cadrage)
                open(chemin, "w", encoding="utf-8", newline="\n").write("".join(lignes))
                return "remplacé"
            j += 1
        lignes.insert(i + 1, _ligne(indent, cadrage))
        open(chemin, "w", encoding="utf-8", newline="\n").write("".join(lignes))
        return "inséré"
    return f"ERREUR : ligne '{cible}' introuvable dans {os.path.basename(chemin)}"


def decouper(chemin_image, cadrage, cote=ICONE):
    """Découpe la fenêtre fractionnaire et réduit à la taille de l'icône."""
    im = Image.open(chemin_image).convert("RGB")
    n = im.size[0]
    c = round(cadrage["cote"] * n)
    x, y = round(cadrage["x"] * n), round(cadrage["y"] * n)
    return im.crop((x, y, x + c, y + c)).resize((cote, cote), Image.LANCZOS)


def planche(icones, sortie, colonnes=5):
    """Contact sheet : taille réelle en haut, agrandissement dessous."""
    if not icones:
        return
    marge, pas = 10, AGRANDI + 12
    lignes = (len(icones) + colonnes - 1) // colonnes
    img = Image.new("RGB", (colonnes * pas + marge,
                            lignes * (pas + ICONE + marge) + marge), "white")
    for i, (_, ic) in enumerate(icones):
        x = marge + (i % colonnes) * pas
        y = marge + (i // colonnes) * (pas + ICONE + marge)
        img.paste(ic, (x, y))
        img.paste(ic.resize((AGRANDI, AGRANDI), Image.NEAREST), (x, y + ICONE + marge // 2))
    img.save(sortie)
    print("planche :", sortie)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    entrees = json.load(open(sys.argv[1], encoding="utf-8"))
    sortie = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RACINE, "planche_icones.png")

    icones, echecs = [], 0
    for e in entrees:
        chemin = os.path.join(MODELS, e["modele"], "model_card.yaml")
        etat = injecter(chemin, e["vignette"], e["cadrage"])
        print(f"  {e['modele']}/{e['vignette']:52s} {etat}")
        if etat.startswith("ERREUR"):
            echecs += 1
            continue
        icones.append((e["vignette"],
                       decouper(os.path.join(MODELS, e["modele"], e["vignette"]), e["cadrage"])))

    planche(icones, sortie)

    # relecture : la clé doit se relire telle qu'on l'a écrite
    for e in entrees:
        card = yaml.safe_load(
            open(os.path.join(MODELS, e["modele"], "model_card.yaml"), encoding="utf-8")
        )
        trouve = any(
            v.get("brut") == e["vignette"] and v.get("cadrage")
            for cl in card.get("classes", [])
            for v in ((cl.get("fiche") or {}).get("vignettes") or [])
        )
        if not trouve:
            print(f"  ! {e['modele']}/{e['vignette']} : cadrage absent après écriture")
            echecs += 1
    if not echecs:
        print("relecture YAML : tous les cadrages présents")
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main()
