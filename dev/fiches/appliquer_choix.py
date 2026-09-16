"""Applique le choix fait dans le navigateur (page de ``page_choix.py``) au modèle.

Entrée : le document ``choix/<classe>`` relu dans la base de l'artefact
(``read_db``), enregistré tel quel en JSON ::

    {"classe": "zone_crateres", "modele": "crateres_seg_ld_v1",
     "retenues": [3, 1], "cadrages": {"3": {"x": 0.3, "y": 0.33, "cote": 0.45}, ...}}

Sortie : les cadres retenus copiés dans ``data/models/<modele>/vignettes/`` sous
``<classe>_0k_{brut,annote}.jpg`` (k = rang du clic, 1 = icône), et un
``cadrages_<classe>.json`` prêt pour ``injecter_cadrages.py`` — seule la première
vignette porte un cadrage (l'icône), les autres s'affichent entières dans la
fiche. Les légendes des ``vignettes[]`` de la fiche restent à écrire à la main.

Usage ::

    python dev/fiches/appliquer_choix.py <choix.json> <dossier_candidats> [--sortie cadrages.json]
"""
import argparse
import json
import os
import shutil

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(RACINE, "data", "models")


def appliquer(choix, dossier, sortie, rang_depart=1):
    classe, modele = choix["classe"], choix["modele"]
    retenues = [int(n) for n in choix["retenues"]]
    if not retenues:
        raise SystemExit("aucun cadre retenu dans le choix")
    dest = os.path.join(MODELS, modele, "vignettes")
    os.makedirs(dest, exist_ok=True)
    vignettes = []
    for k, n in enumerate(retenues, rang_depart):
        for kind in ("brut", "annote"):
            src = os.path.join(dossier, f"{n}_{kind}.jpg")
            if not os.path.isfile(src):
                if kind == "annote":
                    continue
                raise SystemExit(f"{src} absent")
            shutil.copy(src, os.path.join(dest, f"{classe}_{k:02d}_{kind}.jpg"))
        vignettes.append({"candidat": n, "brut": f"vignettes/{classe}_{k:02d}_brut.jpg",
                          "annote": f"vignettes/{classe}_{k:02d}_annote.jpg"})
    for v in vignettes:
        print(f"candidat {v['candidat']} -> {v['brut']}")
    if rang_depart > 1:     # complément : l'icône existante garde son cadrage, rien à injecter
        print(f"complément à partir du rang {rang_depart:02d} : pas de cadrage")
        return vignettes
    c = choix["cadrages"][str(retenues[0])]
    cadrages = [{"modele": modele, "vignette": vignettes[0]["brut"],
                 "cadrage": {"x": round(float(c["x"]), 3), "y": round(float(c["y"]), 3),
                             "cote": round(float(c["cote"]), 3)}}]
    with open(sortie, "w", encoding="utf-8") as f:
        json.dump(cadrages, f, indent=1, ensure_ascii=False)
    print(f"cadrages : {sortie} (icône = candidat {retenues[0]}) ; "
          f"puis python dev/fiches/injecter_cadrages.py {sortie} planche.png")
    return vignettes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("choix", help="document choix/<classe> relu par read_db, en JSON")
    ap.add_argument("dossier", help="dossier des candidats (N_brut.jpg, N_annote.jpg)")
    ap.add_argument("--sortie", default=None, help="cadrages JSON (défaut : <dossier>/cadrages_<classe>.json)")
    ap.add_argument("--rang-depart", type=int, default=1,
                    help="numéro de la première vignette écrite (2 = complément d'une fiche qui a déjà son icône)")
    a = ap.parse_args()
    with open(a.choix, encoding="utf-8") as f:
        choix = json.load(f)
    appliquer(choix, a.dossier, a.sortie or os.path.join(a.dossier, f"cadrages_{choix['classe']}.json"), a.rang_depart)


if __name__ == "__main__":
    main()
