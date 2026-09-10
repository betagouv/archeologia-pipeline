"""Insère un bloc ``fiche`` sous la bonne classe de chaque model_card.yaml.

Insertion TEXTUELLE et non round-trip PyYAML : les model_card portent des
commentaires qui documentent le choix des seuils et le calibrage de la
fiabilité (``seuils_provenance``, bloc ``fiabilite``). Un dump PyYAML les
effacerait tous. On repère donc l'entrée de la classe dans le texte et on y
ajoute le bloc, puis on relit le fichier avec PyYAML pour vérifier que le
résultat parse et contient exactement ce qu'on voulait écrire.
"""
import json
import os
import sys

import yaml

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(RACINE, "data", "models")


def _str_presenter(dumper, data):
    """Prose longue en scalaire plié (``>``) : lisible dans le fichier."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    if len(data) > 90:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)


def bloc_yaml(fiche: dict, indent: str) -> str:
    """``fiche`` sérialisé, réindenté sous l'entrée de classe."""
    brut = yaml.dump(
        {"fiche": fiche},
        allow_unicode=True, default_flow_style=False, sort_keys=False, width=88,
    )
    return "".join(indent + ligne if ligne.strip() else ligne
                   for ligne in brut.splitlines(keepends=True))


def indent_de(ligne: str) -> int:
    return len(ligne) - len(ligne.lstrip(" "))


def injecter(chemin: str, classe: str, fiche: dict) -> str:
    """Ajoute ``fiche`` à l'entrée ``classe``. Renvoie un message d'état."""
    lignes = open(chemin, encoding="utf-8").read().splitlines(keepends=True)

    # 1. la section classes:
    i_classes = next(
        (i for i, li in enumerate(lignes) if li.rstrip("\n") == "classes:"), None
    )
    if i_classes is None:
        return f"ERREUR {classe} : pas de section 'classes:'"

    # 2. les entrées de la liste (tirets au premier niveau sous classes:)
    debut = i_classes + 1
    tiret = None
    entrees = []  # (i_debut, i_fin_exclu)
    i = debut
    while i < len(lignes):
        li = lignes[i]
        if not li.strip() or li.lstrip().startswith("#"):
            i += 1
            continue
        ind = indent_de(li)
        if li.lstrip().startswith("- "):
            if tiret is None:
                tiret = ind
            if ind == tiret:
                if entrees:
                    entrees[-1] = (entrees[-1][0], i)
                entrees.append((i, len(lignes)))
                i += 1
                continue
        if tiret is not None and ind <= tiret and not li.lstrip().startswith("- "):
            if entrees:
                entrees[-1] = (entrees[-1][0], i)
            break
        i += 1

    # 3. l'entrée qui porte name: <classe>
    cible = None
    for a, b in entrees:
        texte = "".join(lignes[a:b])
        for ligne in texte.splitlines():
            if ligne.strip().rstrip() in (f"name: {classe}", f"name: '{classe}'",
                                          f'name: "{classe}"'):
                cible = (a, b)
                break
        if cible:
            break
    if cible is None:
        return f"ERREUR {classe} : classe absente de {os.path.basename(chemin)}"

    a, b = cible
    if any("fiche:" in ligne for ligne in lignes[a:b]):
        return f"IGNORE {classe} : bloc fiche déjà présent"

    # indentation des clés DANS l'entrée (celle de « name: »)
    interne = next(
        indent_de(li) for li in lignes[a:b] if li.strip().startswith("name:")
    )

    # fin réelle de l'entrée : dernière ligne non vide
    fin = b
    while fin > a and not lignes[fin - 1].strip():
        fin -= 1

    lignes[fin:fin] = [bloc_yaml(fiche, " " * interne)]
    open(chemin, "w", encoding="utf-8", newline="\n").write("".join(lignes))
    return f"OK     {classe} : bloc fiche inséré dans {os.path.basename(chemin)}"


def verifier(chemin: str, classe: str, fiche: dict) -> str:
    data = yaml.safe_load(open(chemin, encoding="utf-8"))
    bloc = next(
        (c.get("fiche") for c in data.get("classes", [])
         if isinstance(c, dict) and c.get("name") == classe),
        None,
    )
    if bloc is None:
        return f"       {classe} : RELECTURE KO — fiche absente après écriture"
    ecarts = [k for k in fiche if json.dumps(bloc.get(k), sort_keys=True, ensure_ascii=False)
              != json.dumps(fiche[k], sort_keys=True, ensure_ascii=False)]
    if ecarts:
        return f"       {classe} : RELECTURE KO — champs altérés : {ecarts}"
    return f"       {classe} : relecture OK ({len(bloc)} clés)"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage : python dev/fiches/injecter_fiches.py <fiches.json>\n"
                 "  le JSON est une liste de {modele, classe, fiche}")
    fiches = json.load(open(sys.argv[1], encoding="utf-8"))
    codes = 0
    for entree in fiches:
        chemin = os.path.join(MODELS, entree["modele"], "model_card.yaml")
        msg = injecter(chemin, entree["classe"], entree["fiche"])
        print(msg)
        if msg.startswith("OK"):
            print(verifier(chemin, entree["classe"], entree["fiche"]))
        else:
            codes += 1
    sys.exit(1 if codes else 0)


if __name__ == "__main__":
    main()
