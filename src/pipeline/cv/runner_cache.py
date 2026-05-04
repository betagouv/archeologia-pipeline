"""Helpers de cache et de résolution de chemins pour le runner CV.

Ce module contient les utilitaires qui décident *où* écrire les sorties
d'inférence et *si* une image a déjà été traitée. Il est volontairement
isolé pour rester sans dépendances vers le reste du runner (pas de cycle
d'import) et facilement testable.
"""
from __future__ import annotations

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
