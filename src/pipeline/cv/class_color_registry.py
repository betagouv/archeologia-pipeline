"""Registre de rangs stables pour les couleurs de classes.

Chaque nom de classe reçoit un **rang** à sa première apparition, figé ensuite
(append-only). Le rang alimente :func:`color_palette.base_color_for_rank` →
couleurs réparties (nombre d'or) et **stables dans le temps** : ajouter une
classe n'en déplace aucune autre.

Persisté en JSON dans le dossier de profil QGIS (cohérent avec
``last_ui_config.json``), pour survivre aux mises à jour du plugin. Module pur
(I/O fichier, pas d'API QGIS) → testable hors QGIS et utilisable depuis le
thread worker comme depuis l'UI (la cohérence live/.qgs vient du fichier partagé).

Voir docs/superpowers/specs/2026-06-12-couleurs-detections-design.md.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .color_palette import base_color_for_rank

logger = logging.getLogger(__name__)

RGB = Tuple[int, int, int]


def _normalize(class_name: str) -> str:
    return str(class_name or "").strip().lower()


class ClassColorRegistry:
    """Mappe ``class_name → rang stable`` et en dérive la couleur de base."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ranks: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            classes = data.get("classes") if isinstance(data, dict) else data
            if isinstance(classes, list):
                for i, name in enumerate(classes):
                    norm = _normalize(name)
                    if norm and norm not in self._ranks:
                        self._ranks[norm] = i
        except Exception as e:  # noqa: BLE001 — registre best-effort
            logger.warning(f"Registre couleurs illisible ({self.path}): {e} — réinitialisé")
            self._ranks = {}

    def _persist(self) -> None:
        # Ordre = rang : on sérialise la liste triée par rang.
        ordered: List[str] = [
            name for name, _ in sorted(self._ranks.items(), key=lambda kv: kv[1])
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                json.dumps({"classes": ordered}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)  # écriture atomique
        except Exception as e:  # noqa: BLE001 — la persistance ne doit pas bloquer
            logger.warning(f"Registre couleurs non persisté ({self.path}): {e}")

    def rank_for(self, class_name: str) -> int:
        norm = _normalize(class_name)
        with self._lock:
            rank = self._ranks.get(norm)
            if rank is None:
                rank = len(self._ranks)
                self._ranks[norm] = rank
                self._persist()
            return rank

    def color_for(self, class_name: str) -> RGB:
        return base_color_for_rank(self.rank_for(class_name))


# ---------------------------------------------------------------------------
# Instance par défaut (résolue paresseusement vers le dossier de profil).
# Les sites de génération et d'affichage l'utilisent → source unique partagée.
# ---------------------------------------------------------------------------
_DEFAULT: Optional[ClassColorRegistry] = None
_DEFAULT_LOCK = threading.Lock()


def _resolve_default_path() -> Path:
    """Chemin du registre dans le profil QGIS, avec repli hors QGIS."""
    try:
        from qgis.core import QgsApplication
        base = Path(QgsApplication.qgisSettingsDirPath()) / "archeologia"
    except Exception:
        # Hors QGIS : à côté du plugin (repli ; les tests passent un chemin explicite).
        base = Path(__file__).resolve().parents[3]
    return base / "class_color_registry.json"


def default_registry() -> ClassColorRegistry:
    """Registre partagé (profil). Source unique des couleurs live/.qgs/gpkg."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = ClassColorRegistry(_resolve_default_path())
        return _DEFAULT


def set_default_registry(registry: Optional[ClassColorRegistry]) -> None:
    """Override de l'instance par défaut (tests / réinitialisation)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = registry


def color_for_class(class_name: str) -> RGB:
    """Couleur de base d'une classe via le registre partagé (point d'accès unique)."""
    return default_registry().color_for(class_name)


def rank_for_class(class_name: str) -> int:
    """Rang stable d'une classe via le registre partagé."""
    return default_registry().rank_for(class_name)
