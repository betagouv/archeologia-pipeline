"""Bilan de fiabilité en fin de run — module PUR (sans Qt, sans OGR).

Quand la détection a tourné, chaque entité a un GeoPackage dont les détections
portent le champ ``fiabilite`` (douteux / possible / probable / très probable).
La conversion écrit **les effectifs par niveau** dans le sidecar ``fiabilite.json``
(clé ``effectifs``, par libellé), à côté des catégories : ce module les relit
pour répondre à « par où je commence ? » — une ligne par entité, du niveau le
plus sûr au plus douteux, dans le journal, dans ``metadata.json`` et, côté UI,
en barres sous le bandeau de fin (``ui/widgets/bilan_fiabilite``).

Aucune lecture de GeoPackage : tout vient du sidecar, écrit au moment où les
détections le sont. Un sidecar d'un run antérieur (sans ``effectifs``) est
ignoré : pas de bilan, jamais d'exception.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .fiabilite import CATEGORIES, LABELS_FR, Categorie, categories_sidecar

#: Ordre d'affichage : du plus sûr au plus douteux.
NIVEAUX_DU_PLUS_SUR: Tuple[str, ...] = tuple(reversed(CATEGORIES))

#: Accord avec « détection » (féminin) : (singulier, pluriel).
_ADJECTIFS: Dict[str, Tuple[str, str]] = {
    "quasi_certain": ("très probable", "très probables"),
    "probable": ("probable", "probables"),
    "possible": ("possible", "possibles"),
    "douteux": ("douteuse", "douteuses"),
}


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


@dataclass(frozen=True)
class LigneBilan:
    slug: str                              # dossier detections/<slug>/
    label: str                             # libellé de l'entité (qualifié « · couche » si plusieurs)
    couche: str                            # nom de la couche = clé du registre de couleurs
    classe: str
    modele: str
    categories: Tuple[Categorie, ...]      # niveaux effectifs du run (seuils croissants)
    effectifs: Dict[str, int]              # catégorie → nombre de détections écrites

    @property
    def total(self) -> int:
        return sum(self.effectifs.values())

    def par_niveau(self) -> List[Tuple[str, int]]:
        """``[(catégorie, n), …]`` du plus sûr au plus douteux, niveaux présents seulement."""
        return [(c, self.effectifs.get(c, 0)) for c in NIVEAUX_DU_PLUS_SUR if c in self.effectifs]

    def phrase(self) -> str:
        """« Parcellaire : 103 détections — 12 très probables, 30 probables, 41 possibles, 20 douteuses »."""
        if self.total == 0:
            return f"{self.label} : aucune détection"
        morceaux = []
        for cat, n in self.par_niveau():
            sing, plur = _ADJECTIFS.get(cat, (cat, cat))
            morceaux.append(f"{_nb(n)} {sing if n == 1 else plur}")
        total = f"{_nb(self.total)} détection{'s' if self.total > 1 else ''}"
        return f"{self.label} : {total} — {', '.join(morceaux)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug, "label": self.label, "couche": self.couche, "classe": self.classe,
            "modele": self.modele, "total": self.total,
            "effectifs": {c: n for c, n in self.par_niveau()},
        }


def effectifs_par_categorie(entree: Optional[Mapping[str, Any]]) -> Optional[Dict[str, int]]:
    """Les ``effectifs`` du sidecar (par libellé, tel qu'écrit dans le champ) ramenés
    aux identifiants de catégorie ; ``None`` si le sidecar n'en porte pas (run ancien)."""
    if not isinstance(entree, Mapping) or not isinstance(entree.get("effectifs"), Mapping):
        return None
    par_label = {lab.lower(): cat for cat, lab in LABELS_FR.items()}
    out: Dict[str, int] = {}
    for libelle, n in entree["effectifs"].items():
        cat = par_label.get(str(libelle).strip().lower())
        if cat is None:
            continue
        try:
            out[cat] = out.get(cat, 0) + int(n)
        except (TypeError, ValueError):
            continue
    return out


def _labels_par_slug(runs: Optional[Sequence[Any]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for run in runs or []:
        if not isinstance(run, Mapping):
            continue
        for ent in run.get("entities") or []:
            if isinstance(ent, Mapping) and ent.get("slug"):
                labels.setdefault(str(ent["slug"]), str(ent.get("label") or ent["slug"]))
    return labels


def collecter(det_dir: Path, runs: Optional[Sequence[Any]] = None) -> List[LigneBilan]:
    """Une ligne par couche de détections qui porte des effectifs, dans l'ordre des
    entités des runs puis des dossiers restants. ``det_dir`` = ``<sortie>/detections``."""
    det_dir = Path(det_dir)
    if not det_dir.is_dir():
        return []
    labels = _labels_par_slug(runs)
    slugs = [s for s in labels if (det_dir / s / "fiabilite.json").is_file()]
    slugs += sorted(p.parent.name for p in det_dir.glob("*/fiabilite.json") if p.parent.name not in slugs)
    lignes: List[LigneBilan] = []
    for slug in slugs:
        try:
            data = json.loads((det_dir / slug / "fiabilite.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, Mapping):
            continue
        entrees = [(str(k), v) for k, v in data.items() if effectifs_par_categorie(v) is not None]
        for couche, entree in entrees:
            label = labels.get(slug, slug)
            if len(entrees) > 1:
                label = f"{label} · {couche}"
            lignes.append(LigneBilan(
                slug=slug, label=label, couche=couche,
                classe=str(entree.get("classe") or couche), modele=str(entree.get("modele") or ""),
                categories=categories_sidecar(entree),
                effectifs=effectifs_par_categorie(entree) or {},
            ))
    return lignes
