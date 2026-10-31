"""Harmonise l'échelle des fiches de modèles (règle 2026-10-31) : ``gsd_m`` + anneau du LD en mètres explicites.

Les 7 modèles écrivaient l'anneau du LD de deux façons — ``min_radius``/``max_radius`` (pixels à 1 m, enclos) ou
``rmin_px``/``rmax_px`` + ``gsd_m`` (pixels à 0,5 m, les autres) —, le même anneau physique 5–10 m. Ce script
ajoute, sans rien retirer (rétrocompatible), ``gsd_m``, ``rmin_m`` et ``rmax_m`` aux paramètres d'indice de
``training_params.json`` (``rvt.params``) et de ``model_card.yaml`` (``preferred_rvt.params``, édité ligne à ligne
pour garder les commentaires). Valeurs déduites par ``app.services.indices_metres`` ; un modèle dont l'échelle
est indécidable est laissé tel quel, et dit.

    python dev/harmoniser_echelle_modeles.py [<data/models>]            # liste ce qui changerait
    python dev/harmoniser_echelle_modeles.py [<data/models>] --ecrire   # écrit (puis valider :
                                                                         # python scripts/validate_models_metadata.py)

``data/models`` est gitignoré : à lancer dans le dossier qui porte les modèles (le clone du profil QGIS).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from app.services.indices_metres import anneau_m, gsd_de_fiche  # noqa: E402


def cles_a_ajouter(params: dict, tp: dict) -> dict:
    """``{gsd_m, rmin_m, rmax_m}`` manquants de ``params`` (LD), déduits des pixels et de la résolution."""
    gsd = gsd_de_fiche(params, tp)
    a = anneau_m(params, gsd)
    if gsd is None or a is None:
        return {}
    voulu = {"gsd_m": gsd, "rmin_m": a[0], "rmax_m": a[1]}
    return {k: v for k, v in voulu.items() if k not in params}


def _nombre(v: float) -> str:
    return f"{v:g}" if v != int(v) else f"{v:.1f}"


def harmoniser_card(texte: str, ajout: dict) -> str:
    """Ajoute ``ajout`` sous ``preferred_rvt.params`` d'un model_card.yaml (bloc ou ``{...}`` sur une ligne)."""
    if not ajout:
        return texte
    lignes = texte.splitlines(keepends=True)
    dans_pref = False
    for i, ligne in enumerate(lignes):
        if re.match(r"^preferred_rvt:\s*$", ligne):
            dans_pref = True
            continue
        if dans_pref and re.match(r"^\S", ligne):
            break
        m = re.match(r"^(\s+)params:\s*(\{.*\})?\s*(#.*)?$", ligne.rstrip("\r\n")) if dans_pref else None
        if not m:
            continue
        nl = "\r\n" if ligne.endswith("\r\n") else "\n"
        if m.group(2):   # forme en ligne : { a: 1, b: 2 }
            interieur = m.group(2)[1:-1].strip()
            sup = ", ".join(f"{k}: {_nombre(v)}" for k, v in ajout.items())
            lignes[i] = f"{m.group(1)}params: {{ {interieur}, {sup} }}{nl}"
        else:
            j = i + 1
            retrait = re.match(r"^(\s+)", lignes[j]).group(1) if j < len(lignes) and lignes[j].strip() else m.group(1) + "  "
            lignes[i + 1:i + 1] = [f"{retrait}{k}: {_nombre(v)}{nl}" for k, v in ajout.items()]
        return "".join(lignes)
    raise ValueError("preferred_rvt.params introuvable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modeles", nargs="?", default=str(RACINE / "data" / "models"))
    ap.add_argument("--ecrire", action="store_true", help="écrit les fichiers (sinon : liste seulement)")
    args = ap.parse_args(argv)
    for d in sorted(p for p in Path(args.modeles).iterdir() if (p / "training_params.json").is_file()):
        tp_f, mc_f = d / "training_params.json", d / "model_card.yaml"
        brut = tp_f.read_bytes().decode("utf-8")   # octets : fins de ligne d'origine gardées
        tp = json.loads(brut)
        rvt = tp.get("rvt") or {}
        if str(rvt.get("type") or "").upper() != "LD":
            print(f"{d.name} : indice {rvt.get('type')!r}, pas d'anneau — laissé")
            continue
        aj_tp = cles_a_ajouter(rvt.get("params") or {}, tp)
        texte = mc_f.read_bytes().decode("utf-8") if mc_f.is_file() else ""
        import yaml
        card = yaml.safe_load(texte) or {}
        aj_mc = cles_a_ajouter(((card.get("preferred_rvt") or {}).get("params") or {}), tp)
        if not aj_tp and not aj_mc:
            print(f"{d.name} : déjà harmonisé (ou échelle indécidable)")
            continue
        print(f"{d.name} : training_params.json + {aj_tp} ; model_card.yaml + {aj_mc}")
        if args.ecrire:
            if aj_tp:
                rvt.setdefault("params", {}).update(aj_tp)
                nl = "\r\n" if "\r\n" in brut else "\n"
                sortie = json.dumps(tp, indent=2, ensure_ascii=False).replace("\n", nl)
                tp_f.write_bytes((sortie + (nl if brut.endswith("\n") else "")).encode("utf-8"))
            if aj_mc:
                mc_f.write_bytes(harmoniser_card(texte, aj_mc).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
