#!/usr/bin/env python3
"""Situe les zones d'apprentissage des fiches de classe → ``data/zones_corpus.json``.

Le bandeau « Appris sur » de la fiche d'une classe (bloc « Ce que le modèle a appris »)
montre sur une carte où la classe a été apprise et mesurée. Ce fichier, versionné et
livré avec le plugin, lui donne :

- ``zones`` : l'emprise de chaque zone de corpus (``[lon0, lat0, lon1, lat1]`` en WGS84)
  et son pays, calculée comme l'union des tuiles de ses ``split_manifest.yaml`` dans
  training-models (``datasets/*/``) ;
- ``noms`` : le nom de zone tel que l'écrit une fiche (``classes[].fiche.entrainement.zones[].nom``,
  « Haut-Doubs (25) », « Sligo (Irlande) ») → identifiant de zone, apparié par le numéro
  de département puis les mots du nom ;
- ``contours`` : France métropolitaine, Irlande et Irlande du Nord, simplifiés, tirés de
  la carte du monde livrée avec QGIS (Natural Earth, domaine public).

À relancer après l'installation d'un modèle dont une fiche cite une zone nouvelle :
``tests/unit/test_carte_zones.py`` échoue tant qu'une zone d'une fiche installée n'est
pas située. À lancer avec le Python de QGIS (GDAL/OGR pour les reprojections) :

    C:/OSGeo4W/bin/python-qgis.bat dev/fiches/zones_corpus.py [--training-models C:/projets/Archeologia/training-models] [--sortie data/zones_corpus.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "src"))


def _mots(texte: str) -> set:
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower()
    return {m for m in re.split(r"[^a-z0-9]+", t) if m and m not in {"de", "la", "le", "les", "et", "secteur", "foret", "forets", "irlande", "en", "d"}}


def apparier(nom: str, ids: list) -> str | None:
    """Nom de fiche → identifiant de zone : même département (ou Irlande), puis le plus de
    mots communs (« Galway secteur b_02 (Irlande) » → ``irlande/ie_galway_b_02``)."""
    m = re.search(r"\((\d{2,3})\)", nom)
    if m:
        candidats = [z for z in ids if z.rsplit("/", 1)[-1].startswith(m.group(1) + "_")]
    elif "irlande" in nom.lower():
        candidats = [z for z in ids if z.startswith("irlande/")]
    else:
        candidats = list(ids)
    if not candidats:
        return None
    mots = _mots(re.sub(r"\(.*?\)", "", nom))
    if len(candidats) == 1:
        return candidats[0]

    def cle(z):
        mz = _mots(z.rsplit("/", 1)[-1]) - {"ie"}
        return (len(mots & mz), -len(mz - mots))     # le plus de mots communs, puis le moins en trop
    score = sorted(((cle(z), z) for z in candidats), reverse=True)
    if score[0][0][0] == 0 or (len(score) > 1 and score[0][0] == score[1][0]):
        return None
    return score[0][1]


def noms_des_fiches() -> list:
    import yaml

    from app.services.class_fiche import build_all_fiches

    noms = []
    for card_path in sorted((RACINE / "data" / "models").glob("*/model_card.yaml")):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
        for f in build_all_fiches(card):
            if f.entrainement is not None:
                noms += [z.nom for z in f.entrainement.zones]
    return sorted(set(noms))


def emprises(training_models: Path) -> dict:
    import yaml
    from osgeo import osr

    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    out: dict = {}
    for m in glob.glob(str(training_models / "datasets" / "*" / "split_manifest.yaml")):
        try:
            d = yaml.safe_load(open(m, encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — manifeste illisible : ignoré
            continue
        zone = d.get("zone") or (d.get("config") or {}).get("zone")
        crs = (d.get("grille") or {}).get("crs")
        tuiles = d.get("tuiles") or []
        if not zone or not crs or not tuiles:
            continue
        x0 = min(t["bounds"][0] for t in tuiles)
        y0 = min(t["bounds"][1] for t in tuiles)
        x1 = max(t["bounds"][2] for t in tuiles)
        y1 = max(t["bounds"][3] for t in tuiles)
        src = osr.SpatialReference()
        src.SetFromUserInput(crs)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(src, wgs)
        coins = [tr.TransformPoint(x, y)[:2] for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
        e = [min(c[0] for c in coins), min(c[1] for c in coins), max(c[0] for c in coins), max(c[1] for c in coins)]
        if zone in out:
            a = out[zone]["emprise"]
            e = [min(a[0], e[0]), min(a[1], e[1]), max(a[2], e[2]), max(a[3], e[3])]
        out[zone] = {"pays": "irlande" if zone.startswith("irlande/") else "france", "emprise": [round(v, 4) for v in e]}
    return out


def contours(tolerance: float = 0.03) -> dict:
    from osgeo import ogr

    world = Path(sys.prefix).parent / "qgis" / "resources" / "data" / "world_map.gpkg"
    if not world.is_file():
        world = Path(r"C:\OSGeo4W\apps\qgis\resources\data\world_map.gpkg")

    def anneaux(geom, bbox=None):
        if geom is None or geom.IsEmpty():
            return []
        if bbox:
            b = ogr.CreateGeometryFromWkt(
                f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))")
            geom = geom.Intersection(b)
        if geom is None or geom.IsEmpty():
            return []
        geom = geom.SimplifyPreserveTopology(tolerance)
        polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] if geom.GetGeometryName() == "MULTIPOLYGON" else [geom]
        out = []
        for poly in polys:
            r = poly.GetGeometryRef(0)
            if r is not None and r.GetPointCount() > 3:
                out.append([[round(r.GetX(i), 3), round(r.GetY(i), 3)] for i in range(r.GetPointCount())])
        return out

    ds = ogr.Open(str(world))          # garder le jeu de données : sinon la couche est invalide
    lyr = ds.GetLayerByName("countries")
    res = {"france": [], "irlande": []}
    for feat in lyr:
        iso = str(feat.GetField("ISO_A3"))
        if iso == "FRA":
            res["france"] += anneaux(feat.GetGeometryRef(), (-5.6, 41.2, 9.8, 51.3))
        elif iso == "IRL":
            res["irlande"] += anneaux(feat.GetGeometryRef())
        elif iso == "GBR":
            res["irlande"] += anneaux(feat.GetGeometryRef(), (-8.3, 53.9, -5.3, 55.4))   # Irlande du Nord : l'île entière
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-models", type=Path, default=Path("C:/projets/Archeologia/training-models"),
                    help="dépôt training-models (datasets/*/split_manifest.yaml)")
    ap.add_argument("--sortie", type=Path, default=RACINE / "data" / "zones_corpus.json")
    args = ap.parse_args(argv)
    zones = emprises(args.training_models)
    noms = {}
    manquants = []
    for nom in noms_des_fiches():
        zid = apparier(nom, sorted(zones))
        if zid:
            noms[nom] = zid
        else:
            manquants.append(nom)
    utiles = sorted(set(noms.values()))
    data = {
        "source": "Emprises : union des tuiles des split_manifest de training-models. Contours : Natural Earth "
                  "(domaine public), via la carte du monde livrée avec QGIS, simplifiés.",
        "zones": {z: zones[z] for z in utiles},
        "noms": dict(sorted(noms.items())),
        "contours": contours(),
    }
    args.sortie.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rapport = RACINE / "dev" / "docs" / "_local" / "zones_corpus.log"
    rapport.parent.mkdir(parents=True, exist_ok=True)
    rapport.write_text(
        f"{len(noms)} noms situés sur {len(utiles)} zones ; non situés : {manquants or 'aucun'}\n"
        + "\n".join(f"  {n} → {z}" for n, z in sorted(noms.items())) + "\n", encoding="utf-8")
    return 1 if manquants else 0


if __name__ == "__main__":
    sys.exit(main())
