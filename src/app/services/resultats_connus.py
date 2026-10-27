"""« Revoir un résultat » (sixième écran de l'assistant, lot 9, 2026-10-24) — module pur.

Les traitements lancés depuis ce poste sont dans le registre ``runs_connus.json`` du
profil (``fiabilite_observee``) ; chacun est décrit par son ``livrable/traitement.json``
(date, images, structures, bilan) et, pour le relancer, par la trace complète
``technique/journaux/metadata_<ts>.json`` qui porte la configuration de l'assistant.
Un dossier disparu, ou sans trace, n'est pas un résultat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fiabilite_observee import runs_connus
from .source_modes import mode_info

DIR_LIVRABLE = "livrable"
NOM_TRAITEMENT = "traitement.json"
NOM_PROJET = "projet.qgs"
NOM_RAPPORT = "rapport.html"
DIR_JOURNAUX = Path("technique") / "journaux"
QA_COUVERTURE = Path("indices") / "COUVERTURE" / "zones_mal_couvertes.gpkg"


@dataclass(frozen=True)
class Resultat:
    dossier: Path
    date: str                       # ISO, telle que la trace l'écrit
    mode: str                       # libellé du point de départ, "" si la trace complète manque
    images: Tuple[str, ...]         # codes des images calculées (MNT, SVF, …)
    structures: Tuple[str, ...]     # libellés des structures détectées
    n_detections: int               # total du bilan de fiabilité
    projet: Path                    # livrable/projet.qgs
    rapport: Path                   # livrable/rapport.html
    metadata: Optional[Path]        # technique/journaux/metadata_<ts>.json le plus récent


def _json(chemin: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _dernier_metadata(dossier: Path) -> Optional[Path]:
    try:
        traces = sorted((dossier / DIR_JOURNAUX).glob("metadata_*.json"))
    except OSError:
        return None
    return traces[-1] if traces else None


def _mode(metadata: Optional[Path]) -> str:
    d = _json(metadata) if metadata else None
    app = (((d or {}).get("ui_config") or {}).get("app")) or {}
    files = (app.get("files") if isinstance(app, dict) else None) or {}
    code = str(files.get("data_mode") or "")
    if not code:
        return ""
    try:
        return mode_info(code).banner_label
    except Exception:  # noqa: BLE001 — mode inconnu d'une version plus récente
        return code


def _resultat(dossier: Path) -> Optional[Resultat]:
    trace = _json(dossier / DIR_LIVRABLE / NOM_TRAITEMENT)
    if trace is None:
        return None
    structures = tuple(
        str(e.get("label") or e.get("slug") or "")
        for e in (trace.get("detections_entities") or []) if isinstance(e, dict)
    )
    n = 0
    for ligne in trace.get("bilan_fiabilite") or []:
        if isinstance(ligne, dict):
            try:
                n += int(ligne.get("total") or 0)
            except (TypeError, ValueError):
                pass
    metadata = _dernier_metadata(dossier)
    return Resultat(
        dossier=dossier,
        date=str(trace.get("date") or ""),
        mode=_mode(metadata),
        images=tuple(str(p) for p in (trace.get("active_products") or [])),
        structures=tuple(s for s in structures if s),
        n_detections=n,
        projet=dossier / DIR_LIVRABLE / NOM_PROJET,
        rapport=dossier / DIR_LIVRABLE / NOM_RAPPORT,
        metadata=metadata,
    )


def lister(profil_dir: Path) -> List[Resultat]:
    """Les traitements connus de ce poste, du plus récent au plus ancien ; un dossier
    disparu ou sans ``livrable/traitement.json`` est ignoré."""
    out = [r for r in (_resultat(d) for d in runs_connus(profil_dir)) if r is not None]
    return sorted(out, key=lambda r: r.date, reverse=True)


def config_du_resultat(r: Resultat) -> Optional[Dict[str, Any]]:
    """La configuration de l'assistant qui a produit ``r`` (``ui_config`` de la trace
    complète), pour la relancer ; ``None`` sans trace lisible."""
    if r.metadata is None:
        return None
    d = _json(r.metadata)
    cfg = (d or {}).get("ui_config")
    return dict(cfg) if isinstance(cfg, dict) and cfg else None


def chemins_du_livrable(r: Resultat) -> Tuple[List[str], List[str], List[str]]:
    """``(mosaïques VRT, GeoPackages de détections, GeoPackages de contrôle)`` du
    livrable, dans l'arborescence v3 — ce que « Ajouter au projet QGIS courant » charge."""
    livrable = r.dossier / DIR_LIVRABLE
    vrt = sorted(str(p) for p in livrable.glob("indices/*/tif/index_*.vrt"))
    gpkg = sorted(str(p) for p in livrable.glob("detections/*/*.gpkg"))
    qa_fichier = livrable / QA_COUVERTURE
    qa = [str(qa_fichier)] if qa_fichier.is_file() else []
    return vrt, gpkg, qa
