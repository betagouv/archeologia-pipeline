"""Invalidation du cache d'intermédiaires entre deux runs d'un même ``output_dir``.

Les caches « le fichier existe → on saute » du pipeline LAZ (LAZ fusionnés,
``<dalle>_MNT.tif``, produits rognés…) n'encodent pas les paramètres de
traitement dans leurs noms : relancer dans le même dossier de sortie après
avoir changé la résolution MNT réutilisait silencieusement les produits du run
précédent (bug SRA HDF 2026-08-31, « Terminé en 0.0s »). Seuls les paramètres
des indices RVT sont encodés dans les noms (suffixe ``rvt_naming``) — les
autres passent par la signature comparée ici au sidecar
``intermediaires/run_params.json``, écrit en début de traitement.

Périmètre STRICT (revue adversariale 2026-09-02) : seule ``intermediaires/``
(cache) est purgée. ``indices/`` est le **livrable accumulé** (multi-zones,
recette §22) — jamais touché : la re-publication des finaux périmés est
assurée par la fraîcheur des mtimes côté ``results.needs_refresh`` (un
intermédiaire recalculé est plus récent que le TIF publié → re-copié).
``sources/`` et ``detections/`` ne sont jamais touchés non plus.

Sidecar absent (dossier vierge, antérieur au correctif, ou nettoyé à la
main) : la signature courante est adoptée comme référence SANS purge — pas de
recalcul forcé de plusieurs heures sur un simple re-run legacy à paramètres
identiques ; un dossier legacy dont les paramètres avaient déjà changé garde
une fois son cache périmé (comportement d'avant le correctif, pas une
régression), l'invalidation joue dès le run suivant.

Aucun import QGIS (pathlib/shutil/json) : testable en standalone
(``tests/unit/test_cache_guard.py``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict

SIDECAR_NAME = "run_params.json"


def build_signature(
    *,
    mnt_resolution: float,
    density_resolution: float,
    tile_overlap: float,
    filter_expression: str,
) -> Dict[str, Any]:
    """Signature des paramètres de traitement NON encodés dans les noms de fichiers."""
    return {
        "mnt_resolution": float(mnt_resolution),
        "density_resolution": float(density_resolution),
        "tile_overlap": float(tile_overlap),
        "filter_expression": str(filter_expression),
    }


def ensure_cache_matches(
    *,
    signature: Dict[str, Any],
    intermediaires: Path,
    log: Callable[[str], None] = lambda _m: None,
) -> bool:
    """Compare ``signature`` au sidecar ; purge ``intermediaires/`` si elle diffère.

    Renvoie ``True`` si une purge a eu lieu. Écrit toujours le sidecar (sauf
    purge en échec) : même un run interrompu laisse une signature fidèle à ses
    intermédiaires. À signature identique, rien n'est purgé — le motif
    exists→skip continue de servir de reprise après annulation.
    """
    sidecar = intermediaires / SIDECAR_NAME
    sidecar_exists = sidecar.exists()
    previous: Dict[str, Any] | None = None
    if sidecar_exists:
        try:
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            previous = None  # sidecar illisible → provenance douteuse → purge

    if sidecar_exists and previous == signature:
        return False

    purged = False
    if sidecar_exists:
        # Provenance prouvée différente (ou illisible) → le cache est périmé.
        log(
            "⚠️ Paramètres de traitement modifiés → cache des intermédiaires "
            "invalidé : les dalles de ce run seront recalculées et re-publiées."
        )
        try:
            # Enfant par enfant, SIDECAR EN DERNIER : un rmtree du dossier
            # entier peut supprimer run_params.json ('r') avant de buter sur un
            # fichier verrouillé qui trie après — le run suivant adopterait
            # alors la nouvelle signature sur un cache périmé, définitivement.
            # Ici, tout échec laisse l'ancien sidecar en place → re-purge.
            for child in intermediaires.iterdir():
                if child.name == SIDECAR_NAME:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        except OSError as e:
            # Verrou Windows (couche QGIS chargée depuis ce dossier, projet
            # detections_validation.qgs ouvert dans une autre instance…) : on
            # avorte SANS réécrire le sidecar → le run suivant re-purgera.
            locked = getattr(e, "filename", None) or str(intermediaires)
            raise RuntimeError(
                f"Purge du cache impossible : « {locked} » est verrouillé — "
                "fermez les couches QGIS chargées depuis ce dossier de sortie "
                "(intermediaires/, indices/) et tout projet "
                "detections_validation.qgs ouvert dans une autre instance, "
                "puis relancez."
            ) from e
        purged = True

    intermediaires.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return purged
