"""Tire des cadres candidats (brut + vérité terrain) d'un corpus COCO pour la fiche d'une classe.

Étape 1 du skill ``fiche-classe-plugin`` : depuis les tuiles PNG d'un corpus et
son ``_annotations.coco.json``, choisit ``--n`` tuiles porteuses de la classe en
variant les zones (celle de la vignette déjà installée passe en dernier), cadre
une fenêtre carrée qui contient le plus d'objets entiers, écarte le NoData et les
tuiles plates, puis écrit ``N_brut.jpg`` / ``N_annote.jpg`` (512 px, polygones de
la classe en jaune), ``candidats.json`` (format attendu par ``page_choix.py``) et
une planche de contrôle ``planche_candidats.png``.

Usage ::

    python dev/fiches/candidats_corpus.py <corpus_dir> --categorie four --sortie <dossier> \\
        [--n 6] [--splits train] [--eviter-zone 30_ales_garrigues_ne] [--gsd 0.5]

Le corpus est celui de ``C:/projets/Archeologia/training-models/corpus/<corpus>/``
(``<split>/_annotations.coco.json`` + PNG 648 px). Regarder la planche avant de
publier la page : l'algorithme ne juge pas la lisibilité, seulement les effectifs.
"""
import argparse
import collections
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw

SECTEURS = {
    "57_fenetrange": "Fénétrange (57)", "70_vosges_saonoises": "Vosges saônoises (70)", "41_blois": "Blois (41)",
    "25_besancon_chailluz": "Forêt de Chailluz, Besançon (25)", "25_haut_doubs": "Haut-Doubs (25)",
    "30_ales_garrigues_ne": "Alès, garrigues nord-est (30)", "30_la_capelle_et_masmolene": "La Capelle-et-Masmolène (30)",
    "78_rambouillet": "Forêt de Rambouillet (78)", "54_foret_de_haye": "Forêt de Haye (54)",
    "77_fontainebleau": "Forêt de Fontainebleau (77)", "78_saint_germain_marly": "Saint-Germain-en-Laye et Marly (78)",
    "55_verdun": "Verdun (55)", "ie_sligo": "Sligo (Irlande)", "ie_kerry": "Kerry (Irlande)",
    "ie_boyne_valley": "Vallée de la Boyne (Irlande)", "ie_galway_01": "Galway secteur 01 (Irlande)",
    "ie_galway_02": "Galway secteur 02 (Irlande)", "ie_galway_b_01": "Galway secteur b_01 (Irlande)",
    "ie_galway_b_02": "Galway secteur b_02 (Irlande)", "ie_roscommon": "Roscommon (Irlande)",
    "ie_cork_03": "Cork secteur 03 (Irlande)",
}
COTES = (384, 448, 512, 648)
MARGE = 6           # px entre un objet et le bord de la fenêtre pour le compter « entier »
NODATA_MAX = 0.003  # part de pixels saturés (NoData des PNG LD) tolérée
CONTRASTE_MIN = 10  # écart-type minimal des niveaux de gris
TAILLE = 512


def zone_de(nom):
    return re.sub(r"_r\d+_c\d+\.png$", "", nom)


def rc_de(nom):
    m = re.search(r"_r(\d+)_c(\d+)\.png$", nom)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def charger(corpus, split, categorie):
    d = json.load(open(os.path.join(corpus, split, "_annotations.coco.json"), encoding="utf-8"))
    cid = next(c["id"] for c in d["categories"] if c["name"] == categorie)
    imgs = {i["id"]: i for i in d["images"]}
    par = collections.defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] == cid:
            par[a["image_id"]].append(a)
    return [(imgs[i], anns) for i, anns in par.items()]


def fenetre(anns, W, H):
    """(n_entiers, cote, ox, oy) : la plus petite fenêtre qui capture le maximum d'objets entiers."""
    boites = [a["bbox"] for a in anns]
    cx = np.mean([b[0] + b[2] / 2 for b in boites])
    cy = np.mean([b[1] + b[3] / 2 for b in boites])
    meilleur = None
    for s in COTES:
        s = min(s, W, H)
        best = None
        for oy in range(0, H - s + 1, 32):
            for ox in range(0, W - s + 1, 32):
                n = sum(1 for x, y, w, h in boites
                        if x >= ox + MARGE and y >= oy + MARGE and x + w <= ox + s - MARGE and y + h <= oy + s - MARGE)
                dist = abs(ox + s / 2 - cx) + abs(oy + s / 2 - cy)
                if best is None or (n, -dist) > (best[0], -best[1]):
                    best = (n, dist, ox, oy)
        n, dist, ox, oy = best
        if meilleur is None or n > meilleur[0]:
            meilleur = (n, s, ox, oy)
        if s >= min(W, H):
            break
    return meilleur


def qualite(chemin, ox, oy, s):
    arr = np.asarray(Image.open(chemin).convert("L"))[oy:oy + s, ox:ox + s]
    return float(((arr >= 254) | (arr <= 1)).mean()), float(arr.std())


def rendre(chemin, anns, ox, oy, s, brut, annote):
    im = Image.open(chemin).convert("RGB").crop((ox, oy, ox + s, oy + s)).resize((TAILLE, TAILLE), Image.LANCZOS)
    im.save(brut, quality=88)
    k = TAILLE / s
    an = im.copy()
    g = ImageDraw.Draw(an)
    for a in anns:
        segs = a.get("segmentation") or []
        dessine = False
        for seg in segs:
            if isinstance(seg, list) and len(seg) >= 6:
                pts = [((seg[i] - ox) * k, (seg[i + 1] - oy) * k) for i in range(0, len(seg) - 1, 2)]
                g.line(pts + [pts[0]], fill=(255, 225, 0), width=2, joint="curve")
                dessine = True
        if not dessine:
            x, y, w, h = a["bbox"]
            g.rectangle([(x - ox) * k, (y - oy) * k, (x + w - ox) * k, (y + h - oy) * k], outline=(255, 225, 0), width=2)
    an.save(annote, quality=88)


def choisir(corpus, categorie, splits, n, eviter, gsd, sortie, par_zone=2, presel=14):
    rows = []
    for split in splits:
        for im, anns in charger(corpus, split, categorie):
            ne, s, ox, oy = fenetre(anns, im["width"], im["height"])
            if ne == 0:
                continue
            rows.append(dict(split=split, tuile=im["file_name"], zone=zone_de(im["file_name"]), anns=anns,
                             objets=ne, total=len(anns), cote=s, ox=ox, oy=oy))
    # présélection par zone sur les effectifs, puis lecture des pixels
    par = collections.defaultdict(list)
    for r in rows:
        par[r["zone"]].append(r)
    retenus = collections.defaultdict(list)
    for z, lst in par.items():
        lst.sort(key=lambda r: (-min(r["objets"], 8), r["cote"], r["tuile"]))
        for r in lst[:presel]:
            nd, std = qualite(os.path.join(corpus, r["split"], r["tuile"]), r["ox"], r["oy"], r["cote"])
            if nd > NODATA_MAX or std < CONTRASTE_MIN:
                continue
            r["nodata"] = round(nd, 4)
            r["contraste"] = round(std, 1)
            r["score"] = min(r["objets"], 8) * 10 + min(std, 40)
            retenus[z].append(r)
    zones = sorted(retenus, key=lambda z: (z in eviter, -len(retenus[z])))
    for z in zones:
        retenus[z].sort(key=lambda r: -r["score"])
    choix, pris = [], collections.Counter()
    while len(choix) < n:
        avance = False
        for z in zones:
            if len(choix) >= n or pris[z] >= par_zone:
                continue
            for r in retenus[z]:
                if r in choix:
                    continue
                rr, cc = rc_de(r["tuile"])
                if any(c["zone"] == z and abs(rc_de(c["tuile"])[0] - rr) <= 1 and abs(rc_de(c["tuile"])[1] - cc) <= 1
                       for c in choix):
                    continue
                choix.append(r)
                pris[z] += 1
                avance = True
                break
        if not avance:
            if par_zone >= 6:
                break
            par_zone += 1     # pas assez de zones : on relâche la diversité
    os.makedirs(sortie, exist_ok=True)
    cands = []
    for k, r in enumerate(choix, 1):
        rendre(os.path.join(corpus, r["split"], r["tuile"]), r["anns"], r["ox"], r["oy"], r["cote"],
               os.path.join(sortie, f"{k}_brut.jpg"), os.path.join(sortie, f"{k}_annote.jpg"))
        secteur = SECTEURS.get(r["zone"], r["zone"])
        emprise = int(round(r["cote"] * gsd))
        entier = "tuile entière" if r["cote"] >= 648 else "recadrée"
        cands.append({"candidat": k, "tuile": r["tuile"], "split": r["split"], "zone_id": r["zone"], "zone": secteur,
                      "objets": r["objets"], "objets_tuile": r["total"], "emprise_m": emprise,
                      "fenetre_px": [r["ox"], r["oy"], r["cote"]], "contraste": r["contraste"],
                      "profil": f"{r['objets']} objet(s) entier(s) sur {r['total']} dans la tuile, fenêtre de {emprise} m",
                      "pourquoi": f"{secteur}, split {r['split']} : {r['objets']} objet(s) entier(s) dans une fenêtre "
                                  f"de {emprise} m ({entier})."})
        print(f"{k}: {r['tuile']:44s} {secteur:38s} objets={r['objets']}/{r['total']} cote={r['cote']} std={r['contraste']}")
    json.dump(cands, open(os.path.join(sortie, "candidats.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    planche(sortie, cands)
    return cands


def planche(sortie, cands, cote=340):
    cols = 3
    lignes = (len(cands) + cols - 1) // cols
    sh = Image.new("RGB", (cols * (cote + 10), lignes * (cote + 30)), (24, 24, 24))
    g = ImageDraw.Draw(sh)
    for j, c in enumerate(cands):
        im = Image.open(os.path.join(sortie, f"{c['candidat']}_annote.jpg")).resize((cote, cote), Image.LANCZOS)
        x, y = (j % cols) * (cote + 10) + 5, (j // cols) * (cote + 30) + 5
        sh.paste(im, (x, y))
        g.text((x, y + cote + 4), f"{c['candidat']} - {c['zone']} - {c['objets']} obj - {c['emprise_m']} m",
               fill=(255, 255, 255))
    sh.save(os.path.join(sortie, "planche_candidats.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--categorie", required=True, help="nom de catégorie COCO")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--splits", default="train", help="splits séparés par des virgules")
    ap.add_argument("--eviter-zone", default="", help="zones (ids, virgules) à ne prendre qu'en dernier recours")
    ap.add_argument("--gsd", type=float, default=0.5, help="m/px des tuiles")
    a = ap.parse_args()
    choisir(a.corpus, a.categorie, a.splits.split(","), a.n, set(filter(None, a.eviter_zone.split(","))), a.gsd, a.sortie)
    print(f"planche : {os.path.join(a.sortie, 'planche_candidats.png')}")


if __name__ == "__main__":
    main()
