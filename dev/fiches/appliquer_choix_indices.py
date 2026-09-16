"""Installe les vignettes des produits choisies dans le navigateur.

Dernière étape de la chaîne des fiches de produits (étape 2) :
``candidats_indices.py`` → ``page_choix_indices.py`` → **ici**.

Lit le document ``choix/indices`` relu depuis la base de l'artefact
(``{produits: {clé: n°}, cadrages: {n°: {x, y, cote}}}``), copie le cadre
retenu de chaque produit dans ``data/indices_vignettes/<CLÉ>.jpg`` et déclare
la vignette dans ``data/indices_fiches.json`` — avec son cadrage, sa provenance
et sa licence.

Usage ::

    python dev/fiches/appliquer_choix_indices.py choix_indices.json \\
        D:/brouillons/vignettes_indices [--legendes legendes.json] [--verifier]

``--legendes`` est un JSON ``{clé: "phrase de lecture"}`` : la légende est le
seul texte éditorial de la vignette, elle se rédige à la main en regardant
l'image. Sans elle, la vignette est installée sans légende (la fiche reste
valide) et l'outil dit lesquelles restent à écrire.

``--verifier`` n'écrit rien : il dit ce qui serait fait.

Contrairement aux fiches de classes, il n'y a pas de chirurgie textuelle à
faire : ``indices_fiches.json`` est du JSON pur, sans commentaires à préserver.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RACINE, "src"))

from app.services.indices_model import all_products  # noqa: E402

DATA = os.path.join(_RACINE, "data")
FICHES = os.path.join(DATA, "indices_fiches.json")
VIGNETTES = os.path.join(DATA, "indices_vignettes")

#: Chemin déclaré dans la fiche, relatif à ``data/``.
REL = "indices_vignettes"

#: Licence des vignettes que le plugin produit lui-même.
LICENCE = ("Produite par le plugin sur données IGN LiDAR HD "
           "(licence ouverte Etalab)")


def _charger(chemin):
    return json.loads(io.open(chemin, encoding="utf-8").read())


def _source(cand):
    """Phrase de provenance : dalle, emprise et run — vérifiable, pas décorative."""
    dalle = cand.get("dalle", "?")
    emprise = cand.get("emprise_m")
    bout = f"{dalle}, {emprise} m de côté" if emprise else dalle
    return f"Dalle {bout}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("choix", help="JSON du document choix/indices relu de la base")
    ap.add_argument("dossier", help="dossier produit par candidats_indices.py")
    ap.add_argument("--legendes", help="JSON {clé: légende}")
    ap.add_argument("--verifier", action="store_true", help="n'écrit rien")
    a = ap.parse_args()

    choix = _charger(a.choix)
    meta = _charger(os.path.join(a.dossier, "candidats.json"))
    legendes = _charger(a.legendes) if a.legendes else {}
    fiches = _charger(FICHES)

    par_num = {c["candidat"]: c for c in meta["candidats"]}
    connues = {p.key for p in all_products()}
    produits = choix.get("produits") or {}
    cadrages = choix.get("cadrages") or {}

    inconnues = sorted(set(produits) - connues)
    if inconnues:
        raise SystemExit(f"Clés inconnues dans le choix : {', '.join(inconnues)}")

    poses, sans_legende, manquants = [], [], []
    for cle in [p.key for p in all_products()]:
        if cle not in produits:
            manquants.append(cle)
            continue
        num = produits[cle]
        cand = par_num.get(num)
        if cand is None:
            raise SystemExit(f"{cle} : fenêtre {num} absente de candidats.json")
        src = os.path.join(a.dossier, cand["images"][cle])
        if not os.path.isfile(src):
            raise SystemExit(f"{cle} : cadre introuvable {src}")
        dst_rel = f"{REL}/{cle}.jpg"
        cadrage = cadrages.get(str(num)) or cadrages.get(num)
        legende = legendes.get(cle, "")
        if not legende:
            sans_legende.append(cle)
        poses.append((cle, src, dst_rel, cadrage, legende, cand))

    if manquants:
        print(f"⚠ produits sans choix, laissés en l'état : {', '.join(manquants)}")

    if a.verifier:
        for cle, src, dst_rel, cadrage, legende, cand in poses:
            c = (f" cadrage {cadrage['x']:.2f},{cadrage['y']:.2f},{cadrage['cote']:.2f}"
                 if cadrage else " (pas de cadrage)")
            print(f"  {cle:11s} ← {os.path.basename(src)}  {_source(cand)}{c}")
        print(f"\n{len(poses)} vignette(s) seraient installées. Rien n'a été écrit.")
        return

    os.makedirs(VIGNETTES, exist_ok=True)
    for cle, src, dst_rel, cadrage, legende, cand in poses:
        shutil.copy2(src, os.path.join(DATA, dst_rel))
        bloc = {"image": dst_rel, "source": _source(cand), "licence": LICENCE}
        if legende:
            bloc["legende"] = legende
        if cadrage:
            bloc["cadrage"] = {
                "x": round(float(cadrage["x"]), 4),
                "y": round(float(cadrage["y"]), 4),
                "cote": round(float(cadrage["cote"]), 4),
            }
        fiches.setdefault(cle, {})["vignettes"] = [bloc]
        print(f"  {cle:11s} → data/{dst_rel}")

    io.open(FICHES, "w", encoding="utf-8", newline="").write(
        json.dumps(fiches, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"\n{len(poses)} vignette(s) installées, data/indices_fiches.json mis à jour.")
    if sans_legende:
        print(f"Légendes à rédiger : {', '.join(sans_legende)} "
              "(--legendes, une phrase de lecture par produit).")
    print("Vérifier ensuite : python run_tests.py unit -k indice_fiche")


if __name__ == "__main__":
    main()
