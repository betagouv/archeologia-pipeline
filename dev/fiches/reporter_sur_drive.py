"""Reporte les blocs `fiche` et les vignettes du plugin vers l'archive Drive.

Les model_card du plugin sont la copie de travail ; celles de
``…/runs/training/<modèle>/package/`` sont la source dont un ré-export
repartira. Sans ce report, les fiches seraient effacées au prochain export.

Purement ADDITIF : on lit le bloc `fiche` déjà écrit côté plugin (donc corrections
manuelles comprises), on l'insère dans la carte Drive par la même chirurgie
textuelle que côté plugin — aucun round-trip PyYAML, aucun commentaire perdu — et
on copie le dossier ``vignettes/``. Une carte qui porte déjà un bloc `fiche` est
laissée telle quelle et signalée.
"""
import importlib.util
import json
import os
import shutil

import yaml

SP = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(RACINE, "data", "models")
DRIVE = r"G:/Mon Drive/Archeologia/Archeologia_Shared/model-training"

BUNDLES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "bundles_drive.json"), encoding="utf-8"))

spec = importlib.util.spec_from_file_location("inj", os.path.join(SP, "injecter_fiches.py"))
inj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inj)

def retirer_fiches(chemin):
    """Supprime tous les blocs ``fiche:`` du fichier. Renvoie True si ça a mordu.

    Un bloc court de sa ligne ``fiche:`` jusqu'à la première ligne non vide
    d'indentation inférieure ou égale.
    """
    lignes = open(chemin, encoding="utf-8").read().splitlines(keepends=True)
    sortie, i, touche = [], 0, False
    while i < len(lignes):
        li = lignes[i]
        if li.lstrip().startswith("fiche:"):
            ind = len(li) - len(li.lstrip(" "))
            i += 1
            while i < len(lignes) and (
                not lignes[i].strip()
                or len(lignes[i]) - len(lignes[i].lstrip(" ")) > ind
            ):
                i += 1
            touche = True
            continue
        sortie.append(li)
        i += 1
    if touche:
        open(chemin, "w", encoding="utf-8", newline="\n").write("".join(sortie))
    return touche


resume = []
for nom, rel in BUNDLES.items():
    src_card = os.path.join(PLUGIN, nom, "model_card.yaml")
    dst_dir = os.path.join(DRIVE, rel)
    dst_card = os.path.join(dst_dir, "model_card.yaml")
    print("##", nom)
    if not os.path.isfile(dst_card):
        print("    Drive : model_card.yaml ABSENT — rien fait")
        continue

    plug = yaml.safe_load(open(src_card, encoding="utf-8"))
    fiches = {c["name"]: c["fiche"] for c in plug.get("classes") or []
              if isinstance(c, dict) and c.get("fiche")}

    # La copie plugin fait foi : on retire d'abord TOUS les blocs `fiche` de la
    # carte Drive (une seule passe, sinon la boucle par classe se marcherait
    # dessus), puis on réinjecte les blocs à jour.
    if retirer_fiches(dst_card):
        print("     blocs fiche antérieurs retirés")

    for classe, fiche in fiches.items():
        msg = inj.injecter(dst_card, classe, fiche)
        print("   ", msg)
        if msg.startswith("OK"):
            print("   ", inj.verifier(dst_card, classe, fiche).strip())
            resume.append((nom, classe))

    # vignettes/ : copie à l'identique, écrase un fichier de même nom
    src_v = os.path.join(PLUGIN, nom, "vignettes")
    if os.path.isdir(src_v):
        dst_v = os.path.join(dst_dir, "vignettes")
        os.makedirs(dst_v, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(src_v)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy2(os.path.join(src_v, f), os.path.join(dst_v, f))
                n += 1
        print(f"     vignettes : {n} fichier(s) copié(s)")

print(f"\n{len(resume)} bloc(s) fiche reporté(s) :")
for nom, classe in resume:
    print(f"   {nom} / {classe}")
