"""Registre de rangs stables pour les couleurs de classes.

Chaque nom de classe reçoit un **rang** à sa première apparition, figé ensuite
(append-only). Le rang alimente :func:`color_palette.base_color_for_rank` →
couleurs réparties (nombre d'or) et **stables dans le temps** : ajouter une
classe n'en déplace aucune autre.

Persisté en JSON dans le dossier de profil QGIS (cohérent avec
``last_ui_config.json``), pour survivre aux mises à jour du plugin. Module pur
(I/O fichier, pas d'API QGIS) → testable hors QGIS et utilisable depuis le
thread worker comme depuis l'UI (la cohérence live/.qgs vient du fichier partagé).

Depuis le 2026-10-25 (N-01) la couche d'une structure porte le libellé de la structure
(« Cratères ») ; la clé du registre étant le nom de couche, un **alias** nom de couche →
classe lui garde le rang (donc la couleur) que la classe avait. Append-only lui aussi.

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
        self._alias: Dict[str, str] = {}   # nom de couche renommée → classe dont elle garde le rang
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
            alias = data.get("alias") if isinstance(data, dict) else None
            if isinstance(alias, dict):
                for k, v in alias.items():
                    nk, nv = _normalize(k), _normalize(v)
                    if nk and nk not in self._ranks and nv in self._ranks:
                        self._alias[nk] = nv
        except Exception as e:  # noqa: BLE001 — registre best-effort
            logger.warning(f"Registre couleurs illisible ({self.path}): {e} — réinitialisé")
            self._ranks = {}
            self._alias = {}

    def _persist(self) -> None:
        # Ordre = rang : on sérialise la liste triée par rang.
        ordered: List[str] = [
            name for name, _ in sorted(self._ranks.items(), key=lambda kv: kv[1])
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            payload: Dict[str, object] = {"classes": ordered}
            if self._alias:
                payload["alias"] = dict(self._alias)
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)  # écriture atomique
        except Exception as e:  # noqa: BLE001 — la persistance ne doit pas bloquer
            logger.warning(f"Registre couleurs non persisté ({self.path}): {e}")

    def rank_for(self, class_name: str) -> int:
        norm = _normalize(class_name)
        with self._lock:
            norm = self._alias.get(norm, norm)
            rank = self._ranks.get(norm)
            if rank is None:
                rank = len(self._ranks)
                self._ranks[norm] = rank
                self._persist()
            return rank

    def color_for(self, class_name: str) -> RGB:
        return base_color_for_rank(self.rank_for(class_name))

    def alias(self, nouveau: str, ancien: str) -> bool:
        """``nouveau`` (une couche renommée) prend le rang d'``ancien`` (sa classe) ; vrai si
        l'alias est posé. Rien si ``nouveau`` a déjà un rang ou un alias, si les deux
        noms se confondent une fois normalisés, ou si ``ancien`` est inconnu."""
        n, a = _normalize(nouveau), _normalize(ancien)
        with self._lock:
            if not n or not a or n == a or n in self._ranks or n in self._alias or a not in self._ranks:
                return False
            self._alias[n] = a
            self._persist()
            return True


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


def aliaser_depuis_runs(runs) -> int:
    """Chaque couche nommée par sa structure (``entities[].layer_names`` : classe → nom de
    couche) prend la couleur de sa classe, pour que « Cratères » garde celle que « cratere »
    avait. Les variantes d'une comparaison A/B (``group_label``) gardent leur clé propre,
    donc leur couleur propre. Renvoie le nombre d'alias posés."""
    reg = default_registry()
    n = 0
    for r in runs or []:
        if not isinstance(r, dict):
            continue
        for ent in r.get("entities") or []:
            if not isinstance(ent, dict) or ent.get("group_label"):
                continue
            for classe, couche in (ent.get("layer_names") or {}).items():
                n += int(reg.alias(str(couche), str(classe)))
    return n
