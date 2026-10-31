"""« Ce qui sera produit » (étape 4) : les chemins du livrable que le traitement écrira.

Pur : lu dans la configuration, rendu en chemins **relatifs** au dossier de sortie
(arborescence v3, ``livrable/``). Le nom d'un dossier d'image dépend de ses réglages
(``rvt_naming.get_rvt_folder_name``, sous ``pipeline.ign.products`` dont l'import tire
QGIS) : il est importé dans la fonction, ou injecté par ``nom_dossier_indice`` pour les
tests.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .indices_metres import RESOLUTION_REFERENCE_M, en_px
from .indices_model import all_products
from .rapport_run import NOM_RAPPORT

DIR_LIVRABLE = "livrable"
NOM_PROJET = "projet.qgs"
NOM_TRAITEMENT = "traitement.json"


def _nom_dossier_indice(key: str, rvt_params: Dict[str, Any]) -> str:
    from ...pipeline.ign.products.rvt_naming import get_rvt_folder_name  # différé : QGIS

    return get_rvt_folder_name(key, rvt_params)


def _resolution(mode: str, proc: dict, files: dict) -> float:
    """Résolution du MNT dont partiront les indices : réglée (modes LiDAR) ou lue sur le premier MNT fourni."""
    if mode in ("ign_laz", "local_laz"):
        try:
            return float(proc.get("mnt_resolution") or RESOLUTION_REFERENCE_M)
        except (TypeError, ValueError):
            return RESOLUTION_REFERENCE_M
    dossier = str(files.get("existing_mnt_dir") or "").strip()
    if not dossier:
        return RESOLUTION_REFERENCE_M
    from ...pipeline.ingest_plan import resolution_des_entrees   # différé : rasterio
    return resolution_des_entrees(dossier, ("tif", "tiff", "asc"))


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
        # Rayons réglés en mètres : le nom du dossier porte les pixels reçus à la résolution du MNT.
        params = en_px(config.get("rvt_params") or {}, _resolution(mode, proc, files))
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
