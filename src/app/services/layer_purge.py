"""Sélection des couches QGIS à retirer avant un re-run dans le même ``output_dir`` (pur).

Au re-run d'un pipeline dans le **même** dossier de sortie, les couches ``index_<produit>.vrt``
du run précédent restent chargées dans le projet QGIS. Deux conséquences :
QGIS continue d'afficher le VRT périmé (la couche n'est jamais relue), et il
sérialise même sa version en mémoire (avec ses overviews/stats) **par-dessus** le
VRT que le worker vient de régénérer → la/les nouvelle(s) dalle(s) disparaissent.

La parade (côté UI, au lancement du run, sur le thread principal) est de retirer ces
couches périmées. Ce module isole la **décision pure** — « quelles couches retirer,
à partir d'un ``{id: source}`` et du ``output_dir`` visé » — afin de la tester hors
QGIS. Le retrait lui-même (``QgsProject.removeMapLayers``) reste dans ``src/ui``.

Module pur (aucun import QGIS) → collectable et testable en standalone.
"""
from __future__ import annotations

import os
from typing import List, Mapping

# Sous-arbres livrables régénérés à chaque run (source de vérité :
# ``pipeline/output_paths.py`` → ``DIR_INDICES`` / ``DIR_DETECTIONS``). Gardés en
# littéraux locaux pour ne pas importer ``pipeline`` (qui tire QGIS) depuis un
# module devant rester pur.
_PURGE_SUBTREES = ("indices", "detections")


def _norm_parts(path: str | os.PathLike) -> List[str]:
    """Composants normalisés d'un chemin, indépendamment de l'OS.

    Unifie ``\\``/``/``, minuscule (casse Windows + robustesse), et écarte les
    segments vides/``.`` (slashs multiples, slash final). On n'utilise PAS
    ``os.path`` (dépendant de l'OS : sur POSIX il ignore ``\\`` et les lettres de
    lecteur) ni ``Path.resolve`` (toucherait le disque, casserait la pureté).
    """
    s = str(path).replace("\\", "/")
    return [seg.lower() for seg in s.split("/") if seg not in ("", ".")]


def _strip_decoration(source: str) -> str:
    """Chemin nu d'une source de couche : tronque la décoration OGR/GDAL.

    Les sources GeoPackage portent un suffixe ``|layername=…`` (convention déjà
    utilisée ailleurs). Les rasters/VRT sont des chemins nus → inchangés.
    """
    return (source or "").split("|", 1)[0].strip()


def select_layers_to_purge(
    layer_sources: Mapping[str, str],
    output_dir: str | os.PathLike,
) -> List[str]:
    """IDs des couches dont la source est sous ``<output_dir>/indices`` ou ``…/detections``.

    Conservateur : ne retient QUE les deux sous-arbres livrables régénérés à chaque
    run — jamais ``sources/`` ni ``intermediaires/``, ni une couche hors
    ``output_dir`` (fond de carte, polygone d'emprise, couche quadrillage), ni une
    source non-filesystem (mémoire, postgres). Un ``output_dir`` différent (ou vide)
    renvoie ``[]`` (rien à purger). Le rapprochement se fait par **préfixe de
    composants** (pas de chaîne) → ``…/output_bretagne`` ne capture pas
    ``…/output_bretagne_old``.
    """
    target = _norm_parts(output_dir)
    if not target:
        return []

    to_remove: List[str] = []
    for layer_id, source in layer_sources.items():
        path = _strip_decoration(source)
        if not path:
            continue
        parts = _norm_parts(path)
        if len(parts) <= len(target):
            continue
        if parts[: len(target)] != target:
            continue
        if parts[len(target)] in _PURGE_SUBTREES:
            to_remove.append(layer_id)
    return to_remove
