"""Nom de l'analyse (0.19.0) — module PUR.

Champ de l'étape 1 (``app.files.nom_analyse``), proposé d'après le dossier de sortie
(« <dossier parent> · <dossier de sortie> ») tant que l'utilisateur ne l'a pas écrit
lui-même ; une configuration sans le champ (antérieure, ou champ vidé) retombe sur la
proposition. Le nom suivi de la date du traitement est le **groupe parent** des couches
dans QGIS (« Ajouter au projet QGIS courant », étape 6, ``livrable/projet.qgs``) ; il titre
le rapport et la trace (``traitement.json`` / ``metadata_*.json`` : ``nom_analyse``).

Les fonctions d'arbre ne connaissent que l'interface de ``QgsLayerTreeGroup`` qu'elles
appellent (``findGroup``, ``insertGroup``, ``addGroup``, ``children``, ``removeChildNode``,
``customProperty``) : ``ui/layer_loader`` et ``ui/qgs_writer`` leur passent la racine de leur
projet, d'où le même arbre dans les deux, testé ici sans QGIS.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .finalize_service import GROUPE_STRUCTURES

SEPARATEUR = " · "
GROUPE_INDICES = "Indices"
#: Marque posée sur le groupe d'une analyse : son titre d'origine. Retrouve le groupe même
#: renommé par l'utilisateur, et distingue un groupe du plugin d'un groupe de même nom.
PROPRIETE = "archeologia/analyse"


def nom_par_defaut(output_dir: Any) -> str:
    """« <parent> · <dossier> » (``…\\be_vl\\indices`` → « be_vl · indices ») ; le dossier
    seul sous la racine d'un lecteur ; "" sans dossier."""
    parts = [p for p in re.split(r"[\\/]+", str(output_dir or "").strip()) if p and not p.endswith(":")]
    return SEPARATEUR.join(parts[-2:])


def est_automatique(texte: str, output_dir: Any) -> bool:
    """Le nom suit-il encore le dossier ? Oui s'il est vide ou égal à la proposition."""
    t = str(texte or "").strip()
    return not t or t == nom_par_defaut(output_dir)


def nom_de_config(config: Optional[dict]) -> str:
    files = ((config or {}).get("app") or {}).get("files") or {}
    return str(files.get("nom_analyse") or "").strip() or nom_par_defaut(files.get("output_dir"))


def titre_groupe(nom: str, date_iso: str) -> str:
    """« <nom> — JJ/MM/AAAA HH:MM » (date du traitement, celle de la trace)."""
    nom = str(nom or "").strip() or "Analyse"
    try:
        return f"{nom} — {_dt.datetime.fromisoformat(str(date_iso)).strftime('%d/%m/%Y %H:%M')}"
    except ValueError:
        return nom


def titre_du_livrable(output_dir: Any) -> str:
    """Titre du groupe d'un traitement, lu dans ``livrable/traitement.json`` : la vue de fin et
    l'étape 6 nomment ainsi le même traitement de la même façon (recharger ne duplique rien)."""
    try:
        trace = json.loads((Path(output_dir) / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        trace = {}
    if not isinstance(trace, dict):
        trace = {}
    return titre_groupe(str(trace.get("nom_analyse") or "") or nom_par_defaut(output_dir), str(trace.get("date") or ""))


# ── arbre des couches ───────────────────────────────────────────────
def groupe_analyse(root, titre: str):
    """Le groupe de l'analyse ``titre`` (retrouvé par sa marque), sinon créé en haut de l'arbre.
    Sans titre : la racine (arbre à plat d'avant 0.19.0)."""
    if not titre:
        return root
    for n in root.children():
        if n.customProperty(PROPRIETE) == titre:
            return n
    g = root.insertGroup(0, titre)
    g.setCustomProperty(PROPRIETE, titre)
    return g


def groupe_structures(parent, sous_groupe: str = ""):
    """« Structures » (en haut du groupe), et le sous-groupe d'une dérivée ou d'une comparaison."""
    s = parent.findGroup(GROUPE_STRUCTURES) or parent.insertGroup(0, GROUPE_STRUCTURES)
    return (s.findGroup(sous_groupe) or s.insertGroup(0, sous_groupe)) if sous_groupe else s


def groupe_indices(parent):
    """« Indices » (sous les structures) dans le groupe d'une analyse ; à la racine, les rasters
    restent à plat."""
    if parent.customProperty(PROPRIETE) is None:
        return parent
    return parent.findGroup(GROUPE_INDICES) or parent.addGroup(GROUPE_INDICES)


def _vider(groupe, est_groupe: Callable[[Any], bool]) -> None:
    for n in list(groupe.children()):
        if est_groupe(n) and not n.children():
            groupe.removeChildNode(n)


def retirer_groupes_vides(root, est_groupe: Callable[[Any], bool]) -> None:
    """Après la purge (par chemin) : retire les groupes du plugin qu'elle a vidés — groupe d'une
    analyse (marqué), ses « Structures » (et leurs sous-groupes) et « Indices », l'ancien
    « Structures » de la racine. Jamais un groupe de l'utilisateur, jamais un groupe peuplé."""
    for g in [n for n in root.children() if est_groupe(n)]:
        if g.customProperty(PROPRIETE) is not None:
            for n in list(g.children()):
                if est_groupe(n) and n.name() == GROUPE_STRUCTURES:
                    _vider(n, est_groupe)
                if est_groupe(n) and n.name() in (GROUPE_STRUCTURES, GROUPE_INDICES) and not n.children():
                    g.removeChildNode(n)
        elif g.name() == GROUPE_STRUCTURES:
            _vider(g, est_groupe)
        else:
            continue
        if not g.children():
            root.removeChildNode(g)
