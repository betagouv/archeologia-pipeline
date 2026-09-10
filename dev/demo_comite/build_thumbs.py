"""Fabrique les vignettes du mur visuel (onglet Visualisation) depuis de vraies dalles.

Une seule fenetre geographique, rendue par chaque indice : c'est ce qui permet de
reconnaitre un indice a son rendu, principe du mur visuel. La dalle est choisie
automatiquement (la plus texturee du bloc plein), pas au hasard.

    python dev/demo_comite/build_thumbs.py [--src D:/pipeline_results] [--out data/demo_catalogue/thumbs]

Sortie : un PNG 400x224 par indice + un thumbs.json decrivant ce qui a ete rendu.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# cle produit du plugin (indices_model) -> dossier sous indices/.
# MEME sens que DEMO_INDICES dans build_catalogue.py : deux dictionnaires
# d'orientation opposee dans le meme dossier, c'est une erreur qui attend.
# Le suffixe de parametres des dossiers RVT vient de rvt_naming.get_rvt_folder_name :
# il change si on change les parametres RVT du run.
DEMO = {
    "MNT": "MNT",
    "M_HS": "M_HS_D16_E35_V1",
    "SVF": "SVF_R10_D16_V1_N0",
    "LD": "LD_A15_Rmin10_Rmax20_H1p7_V1",
    "SLRM": "SLRM_R20_V1",
    "VAT": "VAT_T0_B0",
    "CVAT": "CVAT",
    "MSTP": "MSTP_L3-21_M23-203_B223-2023_Li1p2",
    "HS": "HS_Az315_E35_V1",
    "SLO": "SLO_U0_V1",
    "DENSITE": "DENSITE",
    "COUVERTURE": "COUVERTURE",
}

THUMB_W, THUMB_H = 400, 224
# Fenetre exprimee en METRES, pas en pixels : tous les produits ne sont pas a la
# meme resolution (DENSITE et COUVERTURE sortent a 1 m, les indices RVT a 0,5 m).
# En pixels, ces deux-la montreraient deux fois plus de terrain que les autres —
# et, la dalle ne faisant que 1000 px de cote, l'image sortirait deformee.
WIN_M_W, WIN_M_H = 800.0, 448.0   # 16:9, comme la vignette


def _tile_key(path: Path) -> str:
    """LHD_FXX_0357_6794_SVF_... -> '0357_6794' (identifie la dalle, pas l'indice)."""
    parts = path.stem.split("_")
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit() and len(parts[i]) == 4:
            return f"{parts[i]}_{parts[i + 1]}"
    return path.stem


def _index_dir(run: Path, folder: str) -> Path:
    return run / "indices" / folder / "tif"


def _read_window(path: Path, largeur_m: float = WIN_M_W,
                 hauteur_m: float = WIN_M_H) -> np.ndarray | None:
    """Lit une fenetre centree de ``largeur_m`` x ``hauteur_m`` METRES.

    La conversion en pixels se fait avec la resolution reelle du raster, donc
    tous les produits cadrent la meme etendue de terrain quelle que soit leur
    resolution. Si la dalle est trop petite, on reduit en gardant le rapport
    16:9 plutot que de rogner un seul cote (ce qui deformerait la vignette).
    """
    with rasterio.open(path) as ds:
        res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
        w = int(round(largeur_m / res_x))
        h = int(round(hauteur_m / res_y))
        if w > ds.width:
            facteur = ds.width / w
            w, h = ds.width, max(1, int(round(h * facteur)))
        if h > ds.height:
            facteur = ds.height / h
            h, w = ds.height, max(1, int(round(w * facteur)))
        col = (ds.width - w) // 2
        row = (ds.height - h) // 2
        arr = ds.read(window=rasterio.windows.Window(col, row, w, h)).astype("float32")
        nod = ds.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    if not np.isfinite(arr).any():
        return None
    return arr


def _texture_score(path: Path) -> float:
    """Ecart-type normalise d'une fenetre : proxy de 'il se passe quelque chose'."""
    arr = _read_window(path, 400.0, 400.0)
    if arr is None:
        return -1.0
    band = arr[0]
    finite = np.isfinite(band)
    if finite.mean() < 0.98:          # dalle trouee : mauvaise vignette
        return -1.0
    vals = band[finite]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo <= 0:
        return -1.0
    return float(np.std(np.clip(vals, lo, hi)) / (hi - lo))


def pick_tile(run: Path, indices: dict[str, str], scorer_folder: str) -> str:
    """Choisit la dalle la plus texturee PARMI celles presentes dans tous les indices."""
    per_index = {}
    for key, folder in indices.items():
        d = _index_dir(run, folder)
        per_index[key] = {_tile_key(p) for p in d.glob("*.tif")} if d.is_dir() else set()
    common = set.intersection(*per_index.values()) if per_index else set()
    if not common:
        vides = [k for k, v in per_index.items() if not v]
        detail = f" — indices sans aucune dalle : {', '.join(vides)}" if vides else ""
        raise SystemExit(f"aucune dalle commune a tous les indices de {run}{detail}")

    scorer = _index_dir(run, scorer_folder)
    scored = []
    for p in sorted(scorer.glob("*.tif")):
        key = _tile_key(p)
        if key in common:
            scored.append((_texture_score(p), key))
    scored.sort(reverse=True)
    print(f"  {len(common)} dalles communes, top 5 par texture :")
    for s, k in scored[:5]:
        print(f"    {k}  score {s:.3f}")
    return scored[0][1]


def _to_image(arr: np.ndarray) -> Image.Image:
    """Rend une fenetre en image : RGB si 3 bandes, gris etire 2-98 % sinon."""
    if arr.shape[0] >= 3:
        chans = []
        for b in arr[:3]:
            finite = np.isfinite(b)
            vals = b[finite]
            lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
            if hi - lo <= 0:
                hi = lo + 1.0
            c = np.clip((b - lo) / (hi - lo), 0, 1)
            chans.append(np.nan_to_num(c, nan=0.0))
        rgb = (np.dstack(chans) * 255).astype("uint8")
        return Image.fromarray(rgb, "RGB")

    band = arr[0]
    finite = np.isfinite(band)
    vals = band[finite]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo <= 0:
        hi = lo + 1.0
    g = np.clip((band - lo) / (hi - lo), 0, 1)
    g = np.nan_to_num(g, nan=0.0)
    return Image.fromarray((g * 255).astype("uint8"), "L").convert("RGB")


def build(run: Path, indices: dict[str, str], tile: str, out: Path, suffix: str = "") -> dict:
    made = {}
    for key, folder in indices.items():
        d = _index_dir(run, folder)
        match = [p for p in d.glob("*.tif") if _tile_key(p) == tile]
        if not match:
            print(f"  ! {key:6s} : dalle {tile} absente de {folder}")
            continue
        arr = _read_window(match[0])
        if arr is None:
            print(f"  ! {key:6s} : fenetre vide")
            continue
        img = _to_image(arr).resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
        name = f"{key}{suffix}.png"
        img.save(out / name, "PNG", optimize=True)
        made[key] = name
        print(f"  + {key:6s} -> {name}  ({arr.shape[0]} bande(s), {match[0].name})")
    return made


def planche_contact(run: Path, indices: dict[str, str], cle_index: str,
                    sortie: Path, cote: int = 260) -> Path:
    """Rend TOUTES les dalles d'un indice sur une planche, pour choisir a l'oeil.

    Le score de texture ne sait pas distinguer un lotissement d'un site : le bati
    sature le LD et gagne a tous les coups (verifie deux fois sur deux). La seule
    facon fiable de choisir une dalle de vignette est de les regarder.

    La planche respecte la geometrie du bloc : une case par kilometre, le nord en
    haut — un trou dans la couverture se voit donc comme un trou.
    """
    from PIL import ImageDraw

    dossier = _index_dir(run, indices[cle_index])
    dalles = {}
    for f in sorted(dossier.glob("*.tif")):
        cle = _tile_key(f)
        m = re.match(r"(\d{4})_(\d{4})$", cle)
        if m:
            dalles[(int(m.group(1)), int(m.group(2)))] = f
    if not dalles:
        raise SystemExit(f"aucune dalle dans {dossier}")

    xs = sorted({x for x, _ in dalles})
    ys = sorted({y for _, y in dalles}, reverse=True)     # nord en haut
    img = Image.new("RGB", (len(xs) * cote, len(ys) * cote), "black")
    dessin = ImageDraw.Draw(img)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            f = dalles.get((x, y))
            px, py = i * cote, j * cote
            if f is None:
                continue
            arr = _read_window(f, 1000.0, 1000.0)
            if arr is None:
                continue
            img.paste(_to_image(arr).resize((cote, cote), Image.Resampling.LANCZOS),
                      (px, py))
            dessin.text((px + 4, py + 3), f"{x:04d}_{y}", fill="yellow")
    img.save(sortie)
    print(f"planche {len(xs)}x{len(ys)} ({len(dalles)} dalles) -> {sortie}")
    return sortie


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/pipeline_results")
    ap.add_argument("--run", default="demo_comite",
                    help="dossier de run sous --src (defaut : le bloc dense 12 produits)")
    ap.add_argument("--out", default="data/demo_catalogue/thumbs")
    ap.add_argument("--tile", default="",
                    help="dalle a rendre ; vide = choix automatique au score de texture")
    ap.add_argument("--planche", default="",
                    help="ne rend qu'une planche-contact de toutes les dalles, a ce chemin")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run = src / args.run
    if not (run / "indices").is_dir():
        raise SystemExit(f"run introuvable : {run}\n(lancer d'abord run_pipeline.py)")
    print(f"{run} :")

    if args.planche:
        planche_contact(run, DEMO, "LD", Path(args.planche))
        return 0

    # Le score de texture pointe les bourgs (le bati sature le LD), pas les
    # structures archeologiques : il donne un point de depart, pas un verdict.
    # Passer --tile pour imposer une dalle regardee et choisie.
    tile = args.tile or pick_tile(run, DEMO, "LD_A15_Rmin10_Rmax20_H1p7_V1")
    print(f"  dalle : {tile}")
    thumbs = build(run, DEMO, tile, out)

    manifest = {
        "source": args.run,
        "tile": tile,
        "window_m": [WIN_M_W, WIN_M_H],
        "thumbs": thumbs,
    }
    (out / "thumbs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(thumbs)} vignettes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
