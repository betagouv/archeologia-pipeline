"""Tire des fenêtres candidates pour les vignettes des fiches de produits (étape 2).

Pendant, côté produits, de ``candidats_corpus.py``. La différence tient à ce
qu'on illustre : une classe se montre sur des cadres différents, un produit se
montre sur **le même terrain que les onze autres**. C'est en comparant la même
parcelle en Sky-View Factor et en Local Dominance qu'on comprend lequel cocher.
L'outil choisit donc des *fenêtres de terrain*, et rend chacune dans les douze
produits d'un run de référence.

Usage ::

    python dev/fiches/candidats_indices.py D:/pipeline_results/demo_comite \\
        --sortie D:/brouillons/vignettes_indices/ [--n 6] [--cote-m 324]

``<run>`` est un dossier de sortie du pipeline qui contient ``indices/`` avec
les douze produits — c'est la seule exigence. La fenêtre fait 324 m de côté par
défaut, soit 648 px au pas de 0,5 m : le même cadre que les vignettes de classes.

Écrit, dans ``--sortie`` : ``NN_<CLÉ>.jpg`` (douze par fenêtre),
``planche_candidats.png`` (une ligne par fenêtre, les douze produits côte à
côte) et ``candidats.json``. Regarder la planche fait partie du travail :
l'algorithme note le micro-relief et la densité de contours, il ne juge pas
l'intérêt archéologique.

Le choix se fait ensuite dans le navigateur avec ``page_choix_indices.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RACINE, "src"))

from app.services.indices_model import all_products  # noqa: E402
from app.services.visu_catalogue import indice_info  # noqa: E402

#: Côté de la fenêtre, en MÈTRES. 324 m fait 648 px au pas de 0,5 m, soit le
#: même cadre que les vignettes de classes. En mètres et non en pixels parce que
#: les produits n'ont pas tous le même pas (cf. :func:`_lire`).
COTE_M = 324

#: Côté du JPEG écrit pour chaque produit.
TAILLE = 512

#: Part de pixels saturés (0 ou 255) au-delà de laquelle une fenêtre est tenue
#: pour bâtie : sur un modèle de terrain, un toit sature le micro-relief et
#: écrase tout le reste de l'image.
SATURATION_BATI = 0.020

#: Seuil du produit Couverture, repris de la configuration par défaut.
SEUIL_COUVERTURE = 30


# ----------------------------------------------------------------------
# Lecture raster (rasterio, repli osgeo — même stratégie que pipeline/coverage)
# ----------------------------------------------------------------------
def _geo(chemin):
    """``(xmin, ymax, res_x, res_y, largeur, hauteur)`` du raster."""
    try:
        import rasterio

        with rasterio.open(chemin) as s:
            t = s.transform
            return t.c, t.f, abs(t.a), abs(t.e), s.width, s.height
    except ImportError:
        from osgeo import gdal

        ds = gdal.Open(chemin)
        if ds is None:
            raise IOError(f"raster illisible : {chemin}")
        g = ds.GetGeoTransform()
        return g[0], g[3], abs(g[1]), abs(g[5]), ds.RasterXSize, ds.RasterYSize


def _lire(chemin, xmin_m=None, ymax_m=None, cote_m=None):
    """``(tableau, nodata)`` — fenêtre exprimée en COORDONNÉES TERRAIN.

    Les produits n'ont pas tous le même pas : les indices RVT sont au pas du
    modèle d'altitude (0,5 m), la densité et la couverture au pas de densité
    (1 m). Une fenêtre exprimée en pixels couvrirait donc deux fois plus de
    terrain sur ces deux produits — les vignettes ne seraient plus comparables,
    ce qui est précisément ce que la planche doit permettre. La fenêtre est donc
    en mètres, et chaque raster la convertit avec sa propre géoréférence.

    Tableau 2D, ou 3D ``(bande, y, x)`` si multibande.
    """
    x0, y0, rx, ry, w, h = _geo(chemin)
    if xmin_m is None:
        col = row = 0
        nc, nr = w, h
    else:
        col = int(round((xmin_m - x0) / rx))
        row = int(round((y0 - ymax_m) / ry))
        nc = max(1, int(round(cote_m / rx)))
        nr = max(1, int(round(cote_m / ry)))
        col = min(max(0, col), max(0, w - nc))
        row = min(max(0, row), max(0, h - nr))
    try:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(chemin) as s:
            arr = s.read(window=Window(col, row, nc, nr))
            nodata = s.nodata
    except ImportError:
        from osgeo import gdal

        ds = gdal.Open(chemin)
        arr = np.asarray(ds.ReadAsArray(col, row, nc, nr))
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        if arr.ndim == 2:
            arr = arr[None, ...]
    return (arr[0] if arr.shape[0] == 1 else arr), nodata


# ----------------------------------------------------------------------
# Découverte des produits du run
# ----------------------------------------------------------------------
def produits_du_run(run_dir):
    """``{clé: dossier tif}`` pour les produits présents dans ``<run>/indices/``.

    Les dossiers d'indices portent le code du produit suivi d'un suffixe de
    paramètres (``SVF_R10_D16_V1_N0``) : on apparie sur le code exact ou le code
    suivi d'un ``_``, jamais sur un simple préfixe — sans quoi ``SLO`` capterait
    ``SLRM``.
    """
    racine = os.path.join(run_dir, "indices")
    if not os.path.isdir(racine):
        raise SystemExit(f"Pas de dossier indices/ dans {run_dir}")
    dossiers = sorted(os.listdir(racine))
    out = {}
    for p in all_products():
        for d in dossiers:
            if d == p.key or d.startswith(p.key + "_"):
                tif = os.path.join(racine, d, "tif")
                if os.path.isdir(tif) and any(f.endswith(".tif") for f in os.listdir(tif)):
                    out[p.key] = tif
                    break
    return out


def _dalles(tif_dir, cle):
    """``{nom de dalle: chemin}`` — le nom est le préfixe ``LHD_FXX_xxxx_yyyy``."""
    out = {}
    for f in sorted(os.listdir(tif_dir)):
        if not f.endswith(".tif") or f.startswith("index"):
            continue
        parts = f.split("_")
        if len(parts) >= 4:
            out["_".join(parts[:4])] = os.path.join(tif_dir, f)
    return out


# ----------------------------------------------------------------------
# Notation des fenêtres
# ----------------------------------------------------------------------
def _note(win):
    """``(contours, saturation, amplitude)`` d'une fenêtre de micro-relief.

    ``contours`` : gradient moyen — les structures en font monter la valeur.
    ``saturation`` : part de pixels collés aux bornes, signature du bâti.
    ``amplitude`` : écart-type, pour écarter les fenêtres plates.
    """
    a = np.nan_to_num(win.astype(np.float32))
    gy, gx = np.gradient(a)
    contours = float(np.abs(gy).mean() + np.abs(gx).mean())
    saturation = float(((a <= 2) | (a >= 253)).mean())
    return contours, saturation, float(a.std())


def fenetres_candidates(tif_dir, cle, cote_m):
    """Fenêtres notées du produit de référence, triées par intérêt.

    ``cote_m`` est en mètres ; la position retenue l'est aussi (coin haut-gauche
    en Lambert-93), pour que tous les produits cadrent le même terrain.
    """
    out = []
    for dalle, chemin in _dalles(tif_dir, cle).items():
        x0, y0, rx, ry, w, h = _geo(chemin)
        arr, nodata = _lire(chemin)
        if arr.ndim != 2:
            arr = arr[0]
        nc, nr = int(round(cote_m / rx)), int(round(cote_m / ry))
        for row in range(0, h - nr + 1, nr):
            for col in range(0, w - nc + 1, nc):
                win = arr[row:row + nr, col:col + nc]
                if nodata is not None and float((win == nodata).mean()) > 0.01:
                    continue
                contours, saturation, amplitude = _note(win)
                out.append({
                    "dalle": dalle,
                    "xmin": round(x0 + col * rx, 2),
                    "ymax": round(y0 - row * ry, 2),
                    "contours": round(contours, 4),
                    "saturation": round(saturation, 4),
                    "amplitude": round(amplitude, 2),
                })
    return out


def choisir(fenetres, n, couverture_par_dalle):
    """Sélection diversifiée : du bocage d'abord, plus un bâti et une lacune.

    - **bocage** : contours marqués mais peu de saturation — limites de
      parcelles, chemins creux, talus. C'est ce qui illustre le mieux les
      indices de micro-relief, et c'est le régime que la note brute rate,
      puisque le bâti sature toujours plus fort qu'un talus.
    - **bâti** : une fenêtre pour illustrer honnêtement ce que le filtre par
      défaut laisse dans le modèle de terrain (les toits), et pour la densité.
    - **lacune** : la fenêtre la moins bien couverte, seule façon d'illustrer
      le produit Couverture avec autre chose qu'un aplat vert.

    Une seule fenêtre par dalle, pour varier les terrains.
    """
    bocage = [f for f in fenetres if f["saturation"] < SATURATION_BATI]
    bocage.sort(key=lambda f: -f["contours"])

    choix, vues = [], set()

    def prendre(f, pourquoi):
        if f is None or f["dalle"] in vues:
            return False
        vues.add(f["dalle"])
        choix.append({**f, "pourquoi": pourquoi})
        return True

    # La fenêtre la moins couverte, si elle est vraiment lacunaire.
    lacune = min(couverture_par_dalle.items(), key=lambda kv: kv[1], default=(None, 100))
    if lacune[0] is not None and lacune[1] < 99:
        cand = [f for f in fenetres if f["dalle"] == lacune[0]]
        if cand:
            prendre(max(cand, key=lambda f: f["contours"]),
                    f"dalle la moins couverte ({lacune[1]:.0f} % en moyenne) — "
                    "illustre Couverture et Densité")

    bati = sorted(fenetres, key=lambda f: -f["saturation"])
    if bati and bati[0]["saturation"] > SATURATION_BATI:
        prendre(bati[0], "bâti dense — illustre ce que le filtre par défaut "
                         "laisse dans le modèle de terrain")

    for f in bocage:
        if len(choix) >= n:
            break
        prendre(f, "micro-relief marqué, sans bâti — limites de parcelles, "
                   "chemins creux, talus")
    return choix[:n]


# ----------------------------------------------------------------------
# Rendu d'un produit sur une fenêtre
# ----------------------------------------------------------------------
def _etirer(a, masque_valide, bas=2, haut=98):
    """Étirement par percentiles → uint8. Utilisé pour les produits non 8 bits."""
    v = a[masque_valide]
    if v.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(v, [bas, haut])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def _couverture_sur_ombrage(cov, nodata, ombrage):
    """Rendu de production du produit Couverture : 0 % rouge → seuil orange →
    100 % transparent, par-dessus l'ombrage. Reprend la rampe de
    ``ui/layer_loader.apply_coverage_raster_symbology``.

    L'ombrage et la couverture cadrent le même terrain mais pas au même pas
    (0,5 m contre 1 m) : le fond est ramené sur la grille de la couverture
    avant fusion, comme le ferait QGIS à l'affichage.
    """
    if ombrage.shape != cov.shape:
        ombrage = np.asarray(
            Image.fromarray(ombrage).resize(
                (cov.shape[1], cov.shape[0]), Image.BILINEAR
            )
        )
    fond = np.stack([ombrage] * 3, axis=-1).astype(np.float32)
    v = cov.astype(np.float32)
    valide = v != (255 if nodata is None else nodata)
    seuil = float(SEUIL_COUVERTURE)
    # Couleurs et opacités de la rampe de production.
    c0, a0 = np.array([178, 24, 43], np.float32), 1.0
    c1, a1 = np.array([244, 165, 130], np.float32), 180 / 255.0
    c2, a2 = np.array([255, 255, 255], np.float32), 0.0
    couleur = np.zeros(v.shape + (3,), np.float32)
    alpha = np.zeros(v.shape, np.float32)
    bas = valide & (v <= seuil)
    t = np.clip(np.where(seuil > 0, v / max(seuil, 1e-6), 0.0), 0, 1)[..., None]
    couleur[bas] = ((1 - t) * c0 + t * c1)[bas]
    alpha[bas] = ((1 - t[..., 0]) * a0 + t[..., 0] * a1)[bas]
    haut = valide & (v > seuil)
    t2 = np.clip((v - seuil) / max(100 - seuil, 1e-6), 0, 1)[..., None]
    couleur[haut] = ((1 - t2) * c1 + t2 * c2)[haut]
    alpha[haut] = ((1 - t2[..., 0]) * a1 + t2[..., 0] * a2)[haut]
    a = alpha[..., None]
    return np.clip(fond * (1 - a) + couleur * a, 0, 255).astype(np.uint8)


def rendre(cle, arr, nodata, ombrage=None):
    """Image PIL d'un produit sur une fenêtre, telle que l'archéologue la voit.

    Les produits RVT sont déjà en 8 bits étirés par RVT : on les rend tels
    quels, sinon la vignette mentirait sur ce que donne le pipeline. Le modèle
    d'altitude et la densité sont des mesures : ils reçoivent un étirement par
    percentiles. La couverture reçoit sa symbologie de production.
    """
    if arr.ndim == 3:                     # M-HS, MSTP : déjà en RVB
        return Image.fromarray(np.moveaxis(arr[:3], 0, -1).astype(np.uint8), "RGB")
    if cle == "COUVERTURE" and ombrage is not None:
        return Image.fromarray(_couverture_sur_ombrage(arr, nodata, ombrage), "RGB")
    a = arr.astype(np.float32)
    valide = np.isfinite(a)
    if nodata is not None:
        valide &= a != nodata
    if cle in ("MNT", "DENSITE", "COUVERTURE"):
        gris = _etirer(a, valide)
    else:
        gris = np.clip(np.where(valide, a, 0), 0, 255).astype(np.uint8)
    return Image.fromarray(gris, "L").convert("RGB")


# ----------------------------------------------------------------------
# Programme
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="dossier de sortie du pipeline (contient indices/)")
    ap.add_argument("--sortie", required=True, help="dossier des candidats")
    ap.add_argument("--n", type=int, default=6, help="nombre de fenêtres (défaut 6)")
    ap.add_argument("--cote-m", type=int, default=COTE_M,
                    help=f"côté de la fenêtre en mètres (défaut {COTE_M})")
    ap.add_argument("--reference", default="SLRM",
                    help="produit servant à noter les fenêtres (défaut SLRM)")
    args = ap.parse_args()

    prods = produits_du_run(args.run)
    manquants = [p.key for p in all_products() if p.key not in prods]
    print(f"Produits trouvés : {len(prods)}/{len(all_products())}")
    if manquants:
        print(f"  ⚠ absents du run : {', '.join(manquants)}")
    if args.reference not in prods:
        raise SystemExit(f"Produit de référence {args.reference} absent du run.")

    # Couverture moyenne par dalle, pour repérer la fenêtre lacunaire.
    couverture = {}
    if "COUVERTURE" in prods:
        for dalle, chemin in _dalles(prods["COUVERTURE"], "COUVERTURE").items():
            a, nd = _lire(chemin)
            v = a[a != (255 if nd is None else nd)].astype(np.float32)
            if v.size:
                couverture[dalle] = float(v.mean())

    cote_m = float(args.cote_m)
    fenetres = fenetres_candidates(prods[args.reference], args.reference, cote_m)
    print(f"{len(fenetres)} fenetres de {cote_m:.0f} m notees")
    choix = choisir(fenetres, args.n, couverture)
    print(f"{len(choix)} retenues")

    os.makedirs(args.sortie, exist_ok=True)
    # Ordre des cartes de l'étape 2 (base puis indices RVT) : la planche et la
    # page de choix se lisent comme le sélecteur, et comme la liste de gauche
    # des fiches. Pas l'ordre du mur de l'onglet Visualisation.
    ordre = [p.key for p in all_products() if p.key in prods]

    fiches = []
    for i, f in enumerate(choix):
        # L'ombrage sert de fond à la couverture : même fenêtre, même dalle.
        ombrage = None
        if "HS" in prods:
            hs_path = _dalles(prods["HS"], "HS").get(f["dalle"])
            if hs_path:
                a, _ = _lire(hs_path, f["xmin"], f["ymax"], cote_m)
                ombrage = np.clip(np.nan_to_num(a.astype(np.float32)), 0, 255).astype(np.uint8)

        images = {}
        for cle in ordre:
            chemin = _dalles(prods[cle], cle).get(f["dalle"])
            if not chemin:
                continue
            arr, nodata = _lire(chemin, f["xmin"], f["ymax"], cote_m)
            im = rendre(cle, arr, nodata, ombrage).resize((TAILLE, TAILLE), Image.LANCZOS)
            nom = f"{i:02d}_{cle}.jpg"
            im.save(os.path.join(args.sortie, nom), quality=88, optimize=True)
            images[cle] = nom
        fiches.append({
            "candidat": i, "dalle": f["dalle"],
            "xmin": f["xmin"], "ymax": f["ymax"],
            "emprise_m": int(cote_m), "pourquoi": f["pourquoi"],
            "contours": f["contours"], "saturation": f["saturation"],
            "amplitude": f["amplitude"], "images": images,
        })
        print(f"  #{i} {f['dalle']} - {f['pourquoi'][:60]}")

    meta = {
        "run": os.path.abspath(args.run), "emprise_m": int(cote_m),
        "reference": args.reference,
        "ordre": ordre, "candidats": fiches,
    }
    with open(os.path.join(args.sortie, "candidats.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    _planche(args.sortie, meta)
    print(f"\nÉcrit dans {args.sortie} : {len(fiches)} × {len(ordre)} JPEG, "
          "candidats.json, planche_candidats.png")
    print("Regarder la planche AVANT de publier la page de choix.")


def _police(taille=12):
    """Police lisible pour la planche.

    La police par défaut de PIL est une bitmap ASCII : elle rend « Densité » en
    « Densit□ ». On prend donc une TrueType du système, avec repli sur la
    bitmap si aucune n'est trouvée.
    """
    for nom in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nom, taille)
        except OSError:
            continue
    return ImageFont.load_default()


def _planche(sortie, meta):
    """Contact sheet : une ligne par fenêtre, les douze produits côte à côte."""
    ordre, cands = meta["ordre"], meta["candidats"]
    if not cands:
        return
    t, marge_g, marge_h = 150, 118, 22
    largeur = marge_g + t * len(ordre)
    hauteur = marge_h + (t + 16) * len(cands)
    sheet = Image.new("RGB", (largeur, hauteur), "white")
    dr = ImageDraw.Draw(sheet)
    dr.font = _police(12)
    for j, cle in enumerate(ordre):
        dr.text((marge_g + j * t + 3, 6), f"{indice_info(cle).sigle}", fill="black")
    for i, c in enumerate(cands):
        y = marge_h + i * (t + 16)
        dr.text((4, y + 4), f"#{c['candidat']}", fill="black")
        dr.text((4, y + 18), c["dalle"][-9:], fill="#555555")
        dr.text((4, y + 32), c["pourquoi"][:16], fill="#777777")
        for j, cle in enumerate(ordre):
            nom = c["images"].get(cle)
            if not nom:
                continue
            im = Image.open(os.path.join(sortie, nom)).resize((t, t), Image.LANCZOS)
            sheet.paste(im, (marge_g + j * t, y))
    sheet.save(os.path.join(sortie, "planche_candidats.png"))


if __name__ == "__main__":
    main()
