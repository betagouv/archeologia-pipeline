"""Arborescence du dossier de sortie (v3, 2026-10-08) — module PUR.

Deux racines, une règle pour l'utilisateur : ``livrable/`` se garde et se transmet
(projet QGIS, rapport, indices, détections, trace du traitement), ``technique/`` se
supprime (sources LiDAR re-téléchargeables, intermédiaires, images d'inférence,
sorties brutes de détection, journaux). Les chemins eux-mêmes sont dans
``pipeline/output_paths.py`` ; ce module ne connaît que la **migration** d'un dossier
écrit par une version précédente (v2 : ``indices/``, ``detections/``,
``intermediaires/``, ``sources/`` et quelques fichiers à la racine).

La migration est proposée, jamais faite en silence : :func:`plan_migration` liste
les déplacements, :func:`decrire` les explique pour le dialogue, :func:`appliquer`
les exécute par renommage (même disque, sans copie) et rend les échecs. Ce qui
n'est pas reconnu (``MNT/`` et ``RVT/`` de la toute première arborescence, un
fichier posé par l'utilisateur) n'est jamais touché : :func:`non_reconnus` le liste.
Un ancien ``detections_validation.qgs`` suit son dossier ``detections/`` et reste
valide : ses chemins relatifs (``./<entité>/…``, ``../indices/…``) le restent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

VERSION = 3
LIVRABLE = "livrable"
TECHNIQUE = "technique"

#: Racines et fichiers de l'arborescence v2, à la racine du dossier de sortie.
_RACINES_V2 = ("indices", "detections", "intermediaires", "sources")
_FICHIERS_V2 = ("metadata.json", "rapport.html", "rapport_vignette.png", "dalles_urls.txt")


@dataclass(frozen=True)
class Deplacement:
    source: Path
    destination: Path


def _ancien(nom: str) -> bool:
    return nom in _RACINES_V2 or nom in _FICHIERS_V2 or (nom.startswith("pipeline_log_") and nom.endswith(".txt"))


def etat(output_dir: Path) -> str:
    """``"nouvelle"`` (``livrable/`` ou ``technique/`` présent), ``"ancienne"``
    (une racine ou un fichier v2 à la racine, sans les nouvelles), sinon ``"vide"``
    (dossier absent, vide, ou sans rien que le plugin reconnaisse)."""
    p = Path(output_dir)
    if not p.is_dir():
        return "vide"
    noms = {e.name for e in p.iterdir()}
    if LIVRABLE in noms or TECHNIQUE in noms:
        return "nouvelle"
    return "ancienne" if any(_ancien(n) for n in noms) else "vide"


def plan_migration(output_dir: Path) -> List[Deplacement]:
    """Les déplacements qui font passer un dossier v2 en v3, dans l'ordre où ils
    doivent être faits (les enfants avant leur dossier). Une destination déjà
    occupée est laissée telle quelle : la source reste en place et sera listée par
    :func:`non_reconnus`. Vide pour un dossier déjà en v3 ou sans rien de v2."""
    p = Path(output_dir)
    if not p.is_dir():
        return []
    liv, tech = p / LIVRABLE, p / TECHNIQUE
    plan: List[Deplacement] = []

    def deplacer(src: Path, dst: Path) -> None:
        if src.exists() and not dst.exists():
            plan.append(Deplacement(src, dst))

    indices = p / "indices"
    if indices.is_dir():
        for produit in sorted(x for x in indices.iterdir() if x.is_dir()):
            deplacer(produit / "png", tech / "png" / produit.name)
        deplacer(indices, liv / "indices")
    detections = p / "detections"
    if detections.is_dir():
        technique_v2 = detections / "_technique"
        if technique_v2.is_dir():
            for modele in sorted(x for x in technique_v2.iterdir() if x.is_dir()):
                deplacer(modele, tech / "detection" / modele.name)
        deplacer(detections, liv / "detections")
    deplacer(p / "intermediaires", tech / "intermediaires")
    deplacer(p / "sources", tech / "sources")
    deplacer(p / "dalles_urls.txt", tech / "sources" / "dalles_urls.txt")
    journaux = sorted(p.glob("pipeline_log_*.txt"))
    for journal in journaux:
        deplacer(journal, tech / "journaux" / journal.name)
    horodatage = journaux[-1].stem[len("pipeline_log_"):] if journaux else "ancien"
    deplacer(p / "metadata.json", tech / "journaux" / f"metadata_{horodatage}.json")
    deplacer(p / "rapport.html", liv / "rapport.html")
    deplacer(p / "rapport_vignette.png", liv / "rapport_vignette.png")
    return plan


def non_reconnus(output_dir: Path, plan: Sequence[Deplacement]) -> List[str]:
    """Entrées de la racine que la migration ne touche pas : ni ``livrable/`` ni
    ``technique/``, ni source d'un déplacement (``MNT/``, ``RVT/``, un fichier de
    l'utilisateur, une racine v2 dont la destination était déjà prise)."""
    p = Path(output_dir)
    if not p.is_dir():
        return []
    sources = {d.source for d in plan}
    return sorted(e.name for e in p.iterdir() if e.name not in (LIVRABLE, TECHNIQUE) and e not in sources)


def _rel(chemin: Path, racine: Path) -> str:
    try:
        return chemin.relative_to(racine).as_posix()
    except ValueError:
        return chemin.as_posix()


def decrire(plan: Sequence[Deplacement], output_dir: Path, inconnus: Sequence[str] = ()) -> str:
    """Le texte du dialogue de confirmation : ce que deviennent les deux racines,
    les déplacements de premier niveau, ce qui ne bouge pas."""
    racine = Path(output_dir)
    premiers = [d for d in plan if d.source.parent == racine]
    lignes = [f"    {_rel(d.source, racine)}  →  {_rel(d.destination, racine)}" for d in premiers]
    interieurs = len(plan) - len(premiers)
    texte = (
        "Ce dossier de sortie a été écrit par une version précédente du plugin.\n"
        "Pour continuer, il est réorganisé en deux dossiers :\n"
        f"  • {LIVRABLE}/ : indices, détections, projet QGIS, rapport — ce que vous gardez et transmettez ;\n"
        f"  • {TECHNIQUE}/ : sources, intermédiaires, images d'inférence, journaux — ce que vous pouvez supprimer.\n\n"
        f"{len(plan)} déplacement{'s' if len(plan) > 1 else ''} sur le même disque, sans copie :\n"
        + "\n".join(lignes)
    )
    if interieurs:
        texte += f"\n    … et {interieurs} à l'intérieur de ces dossiers (images d'inférence, sorties brutes des modèles)"
    if inconnus:
        texte += "\n\nNon déplacé, inconnu du plugin : " + ", ".join(inconnus)
    texte += "\n\nLes couches QGIS chargées depuis ce dossier sont retirées du projet, puis rechargées en fin de traitement."
    return texte


def appliquer(plan: Sequence[Deplacement], log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Exécute les déplacements par renommage ; rend la liste des échecs (fichier
    verrouillé, autre disque…), un par déplacement, sans s'arrêter au premier.
    Retire ensuite le ``_technique/`` vidé de ses modèles."""
    erreurs: List[str] = []
    racines = {d.source.parent for d in plan}
    racine = min(racines, key=lambda r: len(r.parts)) if racines else None
    for d in plan:
        try:
            d.destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(d.source, d.destination)
            if log is not None and racine is not None and d.source.parent == racine:
                log(f"Réorganisation : {_rel(d.source, racine)} → {_rel(d.destination, racine)}")
        except OSError as e:
            erreurs.append(f"{d.source.name} → {_rel(d.destination, racine) if racine else d.destination} : {e}")
    if racine is not None:
        for vide in (racine / "detections" / "_technique", racine / LIVRABLE / "detections" / "_technique"):
            try:
                if vide.is_dir() and not any(vide.iterdir()):
                    vide.rmdir()
            except OSError:
                pass
    return erreurs
