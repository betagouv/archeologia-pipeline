"""Fabrique le catalogue de démonstration de l'onglet « Visualisation ».

Produit ``data/demo_catalogue/catalogue.json`` : 101 départements, une couverture
déterministe (même hash d'un rendu à l'autre, donc la démo ne bouge pas entre
deux répétitions), et pour chaque indice un ``source`` que GDAL sait ouvrir.

Ce qui est VRAI dans ce catalogue :
  - les vignettes sont des rendus réels de dalles LiDAR HD (cf. build_thumbs.py) ;
  - les ``source`` pointent des mosaïques VRT réellement calculées, donc
    « Afficher dans QGIS » charge une vraie couche ;
  - deux départements portent leurs propres données : 35 (jeu demo_comite,
    calculé pour cette démo) et 78 (run forêt de Saint-Germain) ;
  - toutes les emprises visées sont des rectangles SANS TROU (cf. plus bas).

Ce qui est SIMULÉ : la couverture des 83 autres départements et les volumes.
Ils réutilisent les rasters du jeu demo_comite — c'est une maquette de parcours, pas un
catalogue de production. Le champ ``streamed: false`` le dit à l'interface, qui
affiche « fichier local » au lieu de « flux distant ».

    python dev/demo_comite/build_catalogue.py [--src D:/pipeline_results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Emprises Lambert-93.
#
# Le jeu « demo_comite » a ete calcule pour cette demo (dev/demo_comite/
# run_pipeline.py) : 30 dalles JOINTIVES, 6 x 5 km, avec les 12 produits. Il
# leve le compromis des anciens jeux, qui etaient soit denses avec 1 indice,
# soit riches en indices mais disperses (les 201 dalles bretonnes ne
# remplissent que 1 % de leur emprise).
DEMO_BLOC = [388000, 6815000, 394000, 6820000]          # 6x5 km, 30 dalles, sans trou
SAINT_GERMAIN_BLOC = [629000, 6860000, 635000, 6865000]  # 6x5 km, 30 dalles

DEMO_RUN = "demo_comite"
SAINT_GERMAIN_RUN = "output_78_Foret_domaniale_Saint-Germain_lidar_HD"

# cle produit -> dossier sous indices/, par run
DEMO_INDICES = {
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
SAINT_GERMAIN_INDICES = {
    "MNT": "MNT",
    "LD": "LD_A15_Rmin10_Rmax20_H1p7_V1",
    "SLRM": "SLRM",
}

# Ordre de remplissage d'un département simulé : le socle d'abord.
FILL_ORDER = ["MNT", "M_HS", "SVF", "LD", "SLRM", "VAT", "CVAT", "MSTP", "HS", "SLO",
              "DENSITE", "COUVERTURE"]

DEPTS = [
    ("01", "Ain"), ("02", "Aisne"), ("03", "Allier"), ("04", "Alpes-de-Hte-Provence"),
    ("05", "Hautes-Alpes"), ("06", "Alpes-Maritimes"), ("07", "Ardèche"), ("08", "Ardennes"),
    ("09", "Ariège"), ("10", "Aube"), ("11", "Aude"), ("12", "Aveyron"),
    ("13", "Bouches-du-Rhône"), ("14", "Calvados"), ("15", "Cantal"), ("16", "Charente"),
    ("17", "Charente-Maritime"), ("18", "Cher"), ("19", "Corrèze"), ("2A", "Corse-du-Sud"),
    ("2B", "Haute-Corse"), ("21", "Côte-d'Or"), ("22", "Côtes-d'Armor"), ("23", "Creuse"),
    ("24", "Dordogne"), ("25", "Doubs"), ("26", "Drôme"), ("27", "Eure"),
    ("28", "Eure-et-Loir"), ("29", "Finistère"), ("30", "Gard"), ("31", "Haute-Garonne"),
    ("32", "Gers"), ("33", "Gironde"), ("34", "Hérault"), ("35", "Ille-et-Vilaine"),
    ("36", "Indre"), ("37", "Indre-et-Loire"), ("38", "Isère"), ("39", "Jura"),
    ("40", "Landes"), ("41", "Loir-et-Cher"), ("42", "Loire"), ("43", "Haute-Loire"),
    ("44", "Loire-Atlantique"), ("45", "Loiret"), ("46", "Lot"), ("47", "Lot-et-Garonne"),
    ("48", "Lozère"), ("49", "Maine-et-Loire"), ("50", "Manche"), ("51", "Marne"),
    ("52", "Haute-Marne"), ("53", "Mayenne"), ("54", "Meurthe-et-Moselle"), ("55", "Meuse"),
    ("56", "Morbihan"), ("57", "Moselle"), ("58", "Nièvre"), ("59", "Nord"),
    ("60", "Oise"), ("61", "Orne"), ("62", "Pas-de-Calais"), ("63", "Puy-de-Dôme"),
    ("64", "Pyrénées-Atlantiques"), ("65", "Hautes-Pyrénées"), ("66", "Pyrénées-Orientales"),
    ("67", "Bas-Rhin"), ("68", "Haut-Rhin"), ("69", "Rhône"), ("70", "Haute-Saône"),
    ("71", "Saône-et-Loire"), ("72", "Sarthe"), ("73", "Savoie"), ("74", "Haute-Savoie"),
    ("75", "Paris"), ("76", "Seine-Maritime"), ("77", "Seine-et-Marne"), ("78", "Yvelines"),
    ("79", "Deux-Sèvres"), ("80", "Somme"), ("81", "Tarn"), ("82", "Tarn-et-Garonne"),
    ("83", "Var"), ("84", "Vaucluse"), ("85", "Vendée"), ("86", "Vienne"),
    ("87", "Haute-Vienne"), ("88", "Vosges"), ("89", "Yonne"), ("90", "Terr. de Belfort"),
    ("91", "Essonne"), ("92", "Hauts-de-Seine"), ("93", "Seine-St-Denis"), ("94", "Val-de-Marne"),
    ("95", "Val-d'Oise"), ("971", "Guadeloupe"), ("972", "Martinique"), ("973", "Guyane"),
    ("974", "La Réunion"), ("976", "Mayotte"),
]


def _hash(s: str) -> int:
    """FNV-1a 32 bits — un hash stable, pour que la démo soit identique demain."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _seeded(code: str, salt: str) -> float:
    return (_hash(f"{code}·{salt}") % 1000) / 1000.0


def vrt_path(src: Path, run: str, folder: str) -> str:
    p = src / run / "indices" / folder / "tif" / f"index_{folder}.vrt"
    if not p.is_file():
        raise SystemExit(f"VRT manquant : {p}\n(relance le pipeline ou corrige --src)")
    return p.as_posix()


def _item(key: str, source: str, extent, size: float, streamed: bool) -> dict:
    return {
        "key": key,
        "source": source,
        "thumbnail": f"thumbs/{key}.png",
        "size_go": round(size, 1),
        "extent": extent,
        "streamed": streamed,
    }


def build(src: Path) -> dict:
    demo = {k: vrt_path(src, DEMO_RUN, f) for k, f in DEMO_INDICES.items()}
    sg = {k: vrt_path(src, SAINT_GERMAIN_RUN, f) for k, f in SAINT_GERMAIN_INDICES.items()}

    departments = []
    for code, name in DEPTS:
        if code == "35":                       # le département vitrine : 10 indices, vue pleine
            items = [_item(k, demo[k], DEMO_BLOC, _size(code, k), False) for k in FILL_ORDER]
            updated = "2026-07"
        elif code == "78":                     # run forêt de Saint-Germain
            items = [_item(k, sg[k], SAINT_GERMAIN_BLOC, _size(code, k), False)
                     for k in ("MNT", "LD", "SLRM")]
            updated = "2026-05"
        else:
            r = _seeded(code, "cov")
            if r < 0.18:                       # ~18 % du territoire pas encore traité
                continue
            n = 3 + int(_seeded(code, "n") * 10)         # 3 à 12 indices
            month = 1 + int(_seeded(code, "m") * 9)      # janvier → septembre 2026
            items = [_item(k, demo[k], DEMO_BLOC, _size(code, k), False)
                     for k in FILL_ORDER[:n]]
            updated = f"2026-{month:02d}"
        entry = {
            "code": code, "name": name, "updated": updated,
            "resolution": 0.5, "items": items,
        }
        if code == "35":
            # Le mur s'ouvre ici : 10 indices et une vue pleine.
            entry["featured"] = True
        departments.append(entry)

    return {"updated": "2026-09", "resolution": 0.5, "departments": departments}


def _size(code: str, key: str) -> float:
    """Volume plausible : le MNT (float64) pèse bien plus que les indices uint8."""
    base = 3.0 + _seeded(code + key, "sz") * 4.0 if key == "MNT" else 0.6 + _seeded(code + key, "sz") * 3.5
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/pipeline_results")
    ap.add_argument("--out", default="data/demo_catalogue/catalogue.json")
    args = ap.parse_args()

    cat = build(Path(args.src))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")

    covered = len(cat["departments"])
    products = sum(len(d["items"]) for d in cat["departments"])
    print(f"{covered} départements couverts sur {len(DEPTS)}, {products} indices consultables")
    print(f"-> {out}  ({out.stat().st_size // 1024} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
