"""« Ce qui sera produit » (étape 4) : les chemins du livrable que le traitement écrira.

Pur : lu dans la configuration, rendu en chemins **relatifs** au dossier de sortie
(arborescence v3, ``livrable/``). Le nom d'un dossier d'image dépend de ses réglages
(``rvt_naming.get_rvt_folder_name``, sous ``pipeline.ign.products`` dont l'import tire
QGIS) : il est importé dans la fonction, ou injecté par ``nom_dossier_indice`` pour les
tests.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .indices_model import all_products
from .rapport_run import NOM_RAPPORT

DIR_LIVRABLE = "livrable"
NOM_PROJET = "projet.qgs"
NOM_TRAITEMENT = "traitement.json"


def _nom_dossier_indice(key: str, rvt_params: Dict[str, Any]) -> str:
    from ...pipeline.ign.products.rvt_naming import get_rvt_folder_name  # différé : QGIS

    return get_rvt_folder_name(key, rvt_params)


def sorties_prevues(
    config: dict,
    *,
    nom_dossier_indice: Optional[Callable[[str, Dict[str, Any]], str]] = None,
) -> List[str]:
    """Chemins relatifs du livrable, dans l'ordre où on les cherche : le projet QGIS,
    le rapport, la trace, puis un dossier par image cochée et un par structure.
    Depuis des indices de visualisation existants, aucun produit n'est calculé."""
    files = (config.get("app") or {}).get("files") or {}
    mode = str(files.get("data_mode") or "ign_laz")
    proc = config.get("processing") or {}
    cv = config.get("computer_vision") or {}
    nommer = nom_dossier_indice or _nom_dossier_indice
    out = [
        f"{DIR_LIVRABLE}/{NOM_PROJET}",
        f"{DIR_LIVRABLE}/{NOM_RAPPORT}",
        f"{DIR_LIVRABLE}/{NOM_TRAITEMENT}",
    ]
    if mode != "existing_rvt":
        produits = proc.get("products") or {}
        params = config.get("rvt_params") or {}
        for p in all_products():
            if produits.get(p.key):
                out.append(f"{DIR_LIVRABLE}/indices/{nommer(p.key, params)}/tif/")
    if cv.get("enabled"):
        for run in cv.get("runs") or []:
            for e in run.get("entities") or []:
                slug = str(e.get("slug") or e.get("id") or "").strip()
                if slug:
                    out.append(f"{DIR_LIVRABLE}/detections/{slug}/")
    return out
