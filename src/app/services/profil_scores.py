"""Profil des scores d'une classe — module PUR (sans Qt) : les bandes de l'évaluation.

Les quatre niveaux de fiabilité (douteux / possible / probable / très probable) sont
des bandes de score définies par la part de vrais objets mesurée à l'évaluation. Ce
module relit les **bandes** de cette évaluation (``metriques_eval.json`` →
``modeles[<id>]…etude_seuil.bandes``, pas de 0,01, ``tp``/``fp``) et les agrège par
pas de 0,05 : c'est la matière de la figure « profil des scores » (``ui/widgets/
profil_scores``), qui montre d'où viennent les coupures — sous le seuil presque tout
est faux, au-dessus de la dernière coupure presque tout est vrai.

Même règle de provenance que le validateur des modèles : ``thresholds.fiabilite.source``
désigne l'évaluation (par défaut ``entrainement/evaluation/metriques_eval.json``) ;
``thresholds.fiabilite.zones`` (liste, ou ``{classe: [zones]}``) restreint aux zones à
annotation exhaustive, en sommant ``par_zone_classe[zone][classe].bandes`` — c'est le
cas des classes linéaires, calibrées sur le critère de couverture. Sinon
``par_classe[classe].etude_seuil.bandes``, sinon ``global.etude_seuil.bandes``.

Tout est tolérant : fichier absent, clé manquante, modèle absent de ``modeles`` →
``None`` / liste vide, jamais d'exception (le texte de la fiche reste affiché seul).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .fiabilite import Categorie

PAS_DEFAUT = 0.05
EVAL_DEFAUT = "entrainement/evaluation/metriques_eval.json"


@dataclass(frozen=True)
class Bande:
    lo: float
    hi: float
    tp: int
    fp: int

    @property
    def total(self) -> int:
        return self.tp + self.fp

    @property
    def part_vrais(self) -> Optional[float]:
        return self.tp / self.total if self.total else None


@dataclass(frozen=True)
class Profil:
    classe: str
    bandes: Tuple[Bande, ...]              # agrégées (pas de 0,05), croissantes
    categories: Tuple[Categorie, ...]      # niveaux de la classe, seuils croissants
    n_sous_seuil: int = 0                  # détections écartées, comptées sur les bandes fines

    @property
    def seuil(self) -> float:
        return self.categories[0].seuil if self.categories else 0.0

    @property
    def coupures(self) -> Tuple[float, ...]:
        return tuple(c.seuil for c in self.categories)

    @property
    def total(self) -> int:
        return sum(b.total for b in self.bandes)

    def categorie_de(self, score: float) -> Optional[Categorie]:
        """Le niveau qui contient ``score`` ; ``None`` sous le seuil."""
        courante = None
        for c in self.categories:
            if score >= c.seuil - 1e-9:
                courante = c
        return courante


# ----------------------------------------------------------------------
# Lecture
# ----------------------------------------------------------------------
def _bandes_depuis(liste: Any) -> List[Bande]:
    out: List[Bande] = []
    for b in liste or []:
        try:
            out.append(Bande(float(b["lo"]), float(b["hi"]), int(b["tp"]), int(b["fp"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: x.lo)


def _bloc_modele(evaluation: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    """``modeles[<id>]`` ; si l'id du dossier n'y est pas (poids renommés à
    l'installation), le seul modèle évalué ; sinon le premier."""
    modeles = evaluation.get("modeles")
    if not isinstance(modeles, Mapping) or not modeles:
        return {}
    if model_id in modeles:
        return modeles[model_id] or {}
    return next(iter(modeles.values())) or {}


def _sommer(listes: Sequence[Sequence[Bande]]) -> List[Bande]:
    acc: dict = {}
    for liste in listes:
        for b in liste:
            k = (round(b.lo, 4), round(b.hi, 4))
            tp, fp = acc.get(k, (0, 0))
            acc[k] = (tp + b.tp, fp + b.fp)
    return [Bande(lo, hi, tp, fp) for (lo, hi), (tp, fp) in sorted(acc.items())]


def charger_bandes(
    model_dir: Path,
    classe: str,
    source_rel: Optional[str] = None,
    zones: Optional[Sequence[str]] = None,
) -> List[Bande]:
    """Bandes (pas de l'évaluation) de ``classe`` ; liste vide si rien de lisible."""
    model_dir = Path(model_dir)
    chemin = model_dir / (source_rel or EVAL_DEFAUT)
    try:
        evaluation = json.loads(chemin.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    if not isinstance(evaluation, Mapping):
        return []
    bloc = _bloc_modele(evaluation, model_dir.name)
    if zones:
        pzc = bloc.get("par_zone_classe") or {}
        listes = [
            _bandes_depuis(((pzc.get(z) or {}).get(classe) or {}).get("bandes"))
            for z in zones
        ]
        somme = _sommer([liste for liste in listes if liste])
        if somme:
            return somme
    par_classe = ((bloc.get("par_classe") or {}).get(classe) or {}).get("etude_seuil") or {}
    bandes = _bandes_depuis(par_classe.get("bandes"))
    if bandes:
        return bandes
    return _bandes_depuis(((bloc.get("global") or {}).get("etude_seuil") or {}).get("bandes"))


def agreger(
    bandes: Sequence[Bande], pas: float = PAS_DEFAUT, coupures: Sequence[float] = ()
) -> List[Bande]:
    """Regroupe des bandes fines en bandes de largeur ``pas``, **coupées aussi aux
    coupures** des niveaux : aucune barre n'est à cheval sur un seuil, sinon la
    barre [0,25 ; 0,30[ mélangeait des détections écartées (sous 0,26) et des
    détections gardées au niveau possible. Une bande fine va dans l'intervalle
    qui contient son ``lo``."""
    if not bandes:
        return []
    lo_min = min(b.lo for b in bandes)
    hi_max = max(b.hi for b in bandes)
    plancher = round(math.floor(lo_min / pas + 1e-9) * pas, 6)   # case de la grille qui contient lo_min
    plafond = round(math.ceil(hi_max / pas - 1e-9) * pas, 6)     # … et celle qui contient hi_max
    bornes = {round(plancher + i * pas, 6) for i in range(int(round((plafond - plancher) / pas)) + 1)}
    bornes |= {round(float(c), 6) for c in coupures if plancher < c < plafond}
    limites = sorted(bornes)
    acc: dict = {}
    for b in bandes:
        i = max(k for k, lim in enumerate(limites[:-1]) if b.lo >= lim - 1e-9)
        tp, fp = acc.get(i, (0, 0))
        acc[i] = (tp + b.tp, fp + b.fp)
    return [
        Bande(limites[i], limites[i + 1], tp, fp) for i, (tp, fp) in sorted(acc.items())
    ]


def _source_et_zones(model_dir: Path, classe: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """``(fiabilite.source, zones de la classe)`` lus dans ``model_card.yaml``."""
    try:
        import yaml  # import différé : le module reste importable sans PyYAML

        card = yaml.safe_load((Path(model_dir) / "model_card.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — model_card absent/illisible : pas de profil
        return None, None
    fiab = ((card or {}).get("thresholds") or {}).get("fiabilite") or {}
    if not isinstance(fiab, Mapping):
        return None, None
    source = fiab.get("source")
    zones_raw = fiab.get("zones")
    zones: Optional[List[str]] = None
    if isinstance(zones_raw, list):
        zones = [str(z) for z in zones_raw]
    elif isinstance(zones_raw, Mapping) and isinstance(zones_raw.get(classe), list):
        zones = [str(z) for z in zones_raw[classe]]
    return (str(source) if source else None), zones


def profil_pour_classe(
    model_dir: Path, classe: str, categories: Sequence[Categorie], pas: float = PAS_DEFAUT
) -> Optional[Profil]:
    """Le profil prêt à dessiner, ou ``None`` si l'évaluation n'est pas livrée."""
    if not categories:
        return None
    source, zones = _source_et_zones(model_dir, classe)
    bandes = charger_bandes(model_dir, classe, source, zones)
    if not bandes:
        return None
    cats = tuple(sorted(categories, key=lambda c: c.seuil))
    seuil = cats[0].seuil
    n_sous = sum(b.total for b in bandes if b.hi <= seuil + 1e-9)
    return Profil(
        classe=classe,
        bandes=tuple(agreger(bandes, pas, [c.seuil for c in cats])),
        categories=cats,
        n_sous_seuil=n_sous,
    )
