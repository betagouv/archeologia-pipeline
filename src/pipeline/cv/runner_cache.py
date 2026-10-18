"""Helpers de cache et de résolution de chemins pour le runner CV.

Ce module contient les utilitaires qui décident *où* écrire les sorties
d'inférence et *si* une image a déjà été traitée. Il est volontairement
isolé pour rester sans dépendances vers le reste du runner (pas de cycle
d'import) et facilement testable.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import LogFn


def get_model_slug(cv_config: Dict[str, Any]) -> str:
    """Retourne un slug court et sûr pour le nom du modèle (sous-dossiers)."""
    selected = cv_config.get("selected_model", "")
    if not selected:
        return "unknown"
    p = Path(selected)
    # Remonter au dossier modèle (weights/best.onnx -> model_name)
    model_dir = p.parent
    if model_dir.name == "weights":
        model_dir = model_dir.parent
    slug = model_dir.name or p.stem
    # Nettoyer pour un nom de dossier sûr
    slug = re.sub(r'[^\w\-.]', '_', slug)
    return slug or "model"


def prepare_model_workdir(
    rvt_base_dir: Optional[Path],
    model_slug: str,
    log: LogFn,
) -> Path:
    """Crée le dossier raw_detections/ pour stocker les JSON/TXT d'inférence.

    Les images PNG restent dans indices/<RVT>/png/ et ne sont pas copiées.
    Retourne model_raw_dir.
    """
    model_raw_dir = (rvt_base_dir or Path(".")) / model_slug / "raw_detections"
    model_raw_dir.mkdir(parents=True, exist_ok=True)
    return model_raw_dir


def has_cached_detection(raw_dir: Path, png_stem: str) -> bool:
    """Renvoie True si une détection (txt ou json) existe déjà pour ce PNG.

    Le runner ONNX externe comme le fallback Python écrivent
    ``{stem}.txt`` (format YOLO) et ``{stem}.json`` (payload complet) dans
    ``raw_detections/``. La présence de l'un des deux suffit à considérer
    l'image traitée — un run précédent peut n'avoir écrit que le .json si
    aucune détection n'a passé le seuil de confiance (fichier .txt vide
    possible). On considère aussi les fichiers vides (0 détection) comme
    un résultat légitime.
    """
    return (raw_dir / f"{png_stem}.txt").exists() or (raw_dir / f"{png_stem}.json").exists()


def purge_stale_cached_detections(raw_dir: Path, candidate_pngs: List[Path]) -> int:
    """Supprime les détections en cache plus anciennes que leur PNG d'inférence.

    Le cache (``{stem}.txt``/``{stem}.json``) est indexé par stem : régénérer
    un PNG sous le même nom (ex. bascule vers l'image à marge de l'option B,
    ou changement de paramètres RVT) laisserait le binaire externe comme le
    fallback réutiliser des coordonnées normalisées calculées sur l'ancienne
    géométrie → détections décalées. Un couple est périmé si le PLUS RÉCENT
    de ses deux fichiers date d'avant le PNG ; il est alors supprimé pour
    forcer la ré-inférence (le binaire saute par simple existence).

    Renvoie le nombre d'images dont le cache a été purgé.
    """
    if not raw_dir.is_dir():
        return 0
    purged = 0
    for png in candidate_pngs:
        try:
            png_mtime = png.stat().st_mtime
        except OSError:
            continue
        cached = [p for p in (raw_dir / f"{png.stem}.txt", raw_dir / f"{png.stem}.json") if p.exists()]
        if not cached:
            continue
        if max(p.stat().st_mtime for p in cached) >= png_mtime:
            continue
        for p in cached:
            try:
                p.unlink()
            except OSError:
                pass
        purged += 1
    return purged


def list_candidate_pngs(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    single_jpg: Optional[Path],
) -> List[Path]:
    """Liste les PNG que le runner va traiter, en respectant single_jpg/scan_all.

    Réplique le choix fait par le fallback inference pour que le
    short-circuit amont voit exactement le même périmètre que l'inférence.
    """
    if single_jpg is not None:
        return [single_jpg] if Path(single_jpg).exists() else []
    all_pngs = sorted(jpg_dir.glob("*.png"))
    scan_all = bool(cv_config.get("scan_all", False))
    return all_pngs if scan_all else all_pngs[:1]


#: Clé du cache des détections, à côté des ``.txt``/``.json`` qu'elle couvre.
FICHIER_CLE_CACHE = "cle_cache.json"

#: Fichiers du dossier d'un modèle lus par l'inférence, en plus de ses poids.
_FICHIERS_DU_MODELE = ("args.yaml", "classes.txt")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def version_moteur(ext: Optional[Path]) -> str:
    """Le moteur qui produira les détections : le binaire, identifié par le
    ``sources_sha256`` du ``build_info.json`` posé à côté de lui par
    ``dev/runner_onnx/build.py`` ; ``"python"`` pour l'inférence dans QGIS."""
    if ext is None:
        return "python"
    try:
        info = json.loads((Path(ext).parent / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "binaire:inconnu"
    return f"binaire:{info.get('sources_sha256') or info.get('commit') or 'inconnu'}"


def cle_cache_detections(
    cv_config: Dict[str, Any], weights_path: Optional[Path], moteur: str
) -> Dict[str, Any]:
    """Tout ce dont dépendent les détections brutes d'un modèle : ses fichiers
    (poids ``.onnx``, métadonnées ``.json``, ``args.yaml``, ``classes.txt``), le
    SAHI résolu, les seuils et l'IoU envoyés au moteur, le moteur lui-même.

    Passée par un aller-retour JSON : elle se compare telle quelle au fichier relu.
    """
    fichiers: Dict[str, str] = {}
    if weights_path is not None and Path(weights_path).is_file():
        poids = Path(weights_path)
        dossier = poids.parent.parent if poids.parent.name == "weights" else poids.parent
        for p in [poids, poids.with_suffix(".json"), *(dossier / n for n in _FICHIERS_DU_MODELE)]:
            if p.is_file():
                fichiers[p.name] = _sha256(p)
    cle = {
        "modele": fichiers,
        "sahi": cv_config.get("sahi") or {},
        "confidence_threshold": cv_config.get("confidence_threshold"),
        "confidence_per_class": cv_config.get("confidence_per_class") or {},
        "iou_threshold": cv_config.get("iou_threshold"),
        "moteur": moteur,
    }
    return json.loads(json.dumps(cle, sort_keys=True))


def aligner_cache_detections(raw_dir: Path, cle: Dict[str, Any]) -> int:
    """Purge les détections en cache de ``raw_dir`` calculées sous une autre clé
    — ou sous aucune : dossier d'une version précédente —, puis enregistre la
    clé courante. Renvoie le nombre d'images purgées.

    Le binaire saute toute image dont le ``.txt``/``.json`` existe : sans cette
    purge, un seuil abaissé, un autre modèle ou un autre moteur resservaient les
    détections du lancement précédent. ``classes.txt`` part aussi (le plugin le
    réécrit aussitôt). Une suppression impossible laisse l'ancienne clé : le
    lancement suivant purge de nouveau.
    """
    fichier = raw_dir / FICHIER_CLE_CACHE
    try:
        if json.loads(fichier.read_text(encoding="utf-8")) == cle:
            return 0
    except (OSError, ValueError):
        pass
    images = set()
    complet = True
    for p in sorted(raw_dir.glob("*.txt")) + sorted(raw_dir.glob("*.json")):
        if p.name == FICHIER_CLE_CACHE:
            continue
        try:
            p.unlink()
        except OSError:
            complet = False
            continue
        if p.name != "classes.txt":
            images.add(p.stem)
    if complet:
        raw_dir.mkdir(parents=True, exist_ok=True)
        fichier.write_text(json.dumps(cle, sort_keys=True, indent=1), encoding="utf-8")
    return len(images)


def image_deja_traitee(raw_dir: Path, stem: str, force_reprocess: bool) -> bool:
    """Vrai si l'image a ses détections en cache et qu'on ne force pas le
    re-traitement. Règle unique du plugin, du binaire (CLI) et du repli Python :
    le binaire sautait toute image au cache sans lire ``force_reprocess``, et une
    image annotée seule suffisait à le faire sauter."""
    return not force_reprocess and has_cached_detection(raw_dir, stem)
