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
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# indices/<dossier> -> cle produit du plugin (indices_model)
BRETAGNE = {
    "MNT": "MNT",
    "HS_Az315_E35_V1": "HS",
    "M_HS_D16_E35_V1": "M_HS",
    "SVF_R10_D16_V1_N0": "SVF",
    "SLO_U0_V1": "SLO",
    "LD_A15_Rmin10_Rmax20_H1p7_V1": "LD",
    "SLRM_R20_V1": "SLRM",
    "VAT_T0_B0": "VAT",
    "MSTP_L3-21_M23-203_B223-2023_Li1p2": "MSTP",
    "CVAT": "CVAT",
}
# Dalle retenue pour tout le mur : une etoile forestiere (layons convergeant sur
# un rond-point) avec des rayures de labour ancien, et presque pas de bati. Elle
# se lit en vignette et elle est archeologique — deux choses qu'un score de
# texture ne sait pas juger : le maximum de texture tombait sur un bourg.
# Le run Saint-Germain, lui, couvre des villes : il sert de source raster reelle
# pour les Yvelines, pas de vignette.
DEFAULT_TILE = "0362_6800"

THUMB_W, THUMB_H = 400, 224
WIN_W, WIN_H = 1600, 896          # fenetre lue puis reduite (800 x 448 m a 0,5 m)


def _tile_key(path: Path) -> str:
    """LHD_FXX_0357_6794_SVF_... -> '0357_6794' (identifie la dalle, pas l'indice)."""
    parts = path.stem.split("_")
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit() and len(parts[i]) == 4:
            return f"{parts[i]}_{parts[i + 1]}"
    return path.stem


def _index_dir(run: Path, folder: str) -> Path:
    return run / "indices" / folder / "tif"


def _read_window(path: Path, w: int, h: int) -> np.ndarray | None:
    """Lit une fenetre centree de w x h pixels. Retourne (bandes, h, w) en float."""
    with rasterio.open(path) as ds:
        w = min(w, ds.width)
        h = min(h, ds.height)
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
    arr = _read_window(path, 800, 800)
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


def pick_tile(run: Path, folders: dict[str, str], scorer_folder: str) -> str:
    """Choisit la dalle la plus texturee PARMI celles presentes dans tous les indices."""
    per_index = {}
    for folder in folders:
        d = _index_dir(run, folder)
        per_index[folder] = {_tile_key(p) for p in d.glob("*.tif")} if d.is_dir() else set()
    common = set.intersection(*per_index.values()) if per_index else set()
    if not common:
        raise SystemExit(f"aucune dalle commune a tous les indices de {run}")

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


def build(run: Path, folders: dict[str, str], tile: str, out: Path, suffix: str = "") -> dict:
    made = {}
    for folder, key in folders.items():
        d = _index_dir(run, folder)
        match = [p for p in d.glob("*.tif") if _tile_key(p) == tile]
        if not match:
            print(f"  ! {key:6s} : dalle {tile} absente de {folder}")
            continue
        arr = _read_window(match[0], WIN_W, WIN_H)
        if arr is None:
            print(f"  ! {key:6s} : fenetre vide")
            continue
        img = _to_image(arr).resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
        name = f"{key}{suffix}.png"
        img.save(out / name, "PNG", optimize=True)
        made[key] = name
        print(f"  + {key:6s} -> {name}  ({arr.shape[0]} bande(s), {match[0].name})")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/pipeline_results")
    ap.add_argument("--out", default="data/demo_catalogue/thumbs")
    ap.add_argument("--tile", default=DEFAULT_TILE,
                    help="dalle a rendre (defaut : l'etoile forestiere retenue)")
    ap.add_argument("--pick", action="store_true",
                    help="rechoisir la dalle au score de texture (attention : vise les bourgs)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bret = src / "output_bretagne"
    print(f"Bretagne ({bret}) :")
    tile = pick_tile(bret, BRETAGNE, "LD_A15_Rmin10_Rmax20_H1p7_V1") if args.pick else args.tile
    print(f"  dalle : {tile}")
    thumbs = build(bret, BRETAGNE, tile, out)

    manifest = {
        "source": "output_bretagne",
        "tile": tile,
        "window_m": [WIN_W // 2, WIN_H // 2],
        "thumbs": thumbs,
    }
    (out / "thumbs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(thumbs)} vignettes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
