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

Depuis 2026-10-08 le profil porte aussi ses **bandes fines** (pour le bilan à un seuil
quelconque : ce que ce seuil garde des vrais objets et écarte des fausses détections,
cf. :func:`bilan_au_seuil`), le **seuil d'équilibre précision-rappel** de l'évaluation
(``seuil_f1max``, par classe sinon global — le seuil déployé est choisi en dessous,
règle 2026-09-09) et peut être calculé **par zone d'évaluation** (:func:`profils_par_zone`).

Tout est tolérant : fichier absent, clé manquante, modèle absent de ``modeles`` →
``None`` / liste vide, jamais d'exception (le texte de la fiche reste affiché seul).
"""
from __future__ import annotations

import json
import math
import re
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
    fines: Tuple[Bande, ...] = ()          # bandes de l'évaluation (pas fin) : bilan à un seuil
    seuil_f1max: Optional[float] = None    # point d'équilibre précision-rappel de l'évaluation
    tableau: Tuple[Tuple[float, float, float], ...] = ()   # (seuil, précision, rappel) de l'évaluation, pas de 0,05
    zone: str = ""                         # identifiant de zone (profil par zone), "" = toutes

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


@dataclass(frozen=True)
class Bilan:
    """Ce qu'un seuil garde sur le banc (bandes fines de l'évaluation).

    La phrase se dit **par rapport au seuil du modèle**, jamais en part d'un total :
    sur une classe linéaire évaluée au critère de couverture, les bandes sous le
    seuil comptent des milliers de fragments « corrects » (parcellaire : 26 000 sur
    29 000), et « garde 11 % des vrais objets » aurait été un contresens. Ce que
    l'on achète en montant ou en baissant le seuil, lui, se lit pareil pour tous.
    """
    seuil: float
    vrais_gardes: int
    fausses_gardees: int
    vrais_total: int
    fausses_total: int

    @property
    def ecartees(self) -> int:
        return self.vrais_total + self.fausses_total - self.vrais_gardes - self.fausses_gardees

    def phrase(self, reference: Optional["Bilan"] = None) -> str:
        """Au seuil du modèle (ou sans référence) : les effectifs gardés et écartés ;
        à un autre seuil : la différence avec le seuil du modèle, en nombre et en
        pour cent. Chiffres du banc, dits comme tels : sur le terrain ils varient."""
        if not (self.vrais_total or self.fausses_total):
            return ""
        if reference is None or abs(reference.seuil - self.seuil) < 1e-9:
            return (
                f"Au seuil {_v(self.seuil)}, le banc garde {_nb(self.vrais_gardes)} détections "
                f"correctes et {_nb(self.fausses_gardees)} fausses ; {_nb(self.ecartees)} sont écartées."
            )
        sens = "En montant" if self.seuil > reference.seuil else "En baissant"
        dv = self.vrais_gardes - reference.vrais_gardes
        df = self.fausses_gardees - reference.fausses_gardees
        return (
            f"{sens} le seuil à {_v(self.seuil)} : {_signe(dv)} détections correctes"
            f"{_pct(dv, reference.vrais_gardes)} et {_signe(df)} fausses{_pct(df, reference.fausses_gardees)} "
            f"par rapport au seuil du modèle ({_v(reference.seuil)}), sur le banc."
        )


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _v(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _signe(d: int) -> str:
    return f"{'+' if d >= 0 else '−'}{_nb(abs(d))}"


def _pct(d: int, reference: int) -> str:
    return f" ({'+' if d >= 0 else '−'}{round(abs(d) / reference * 100)} %)" if reference else ""


def bilan_au_seuil(bandes: Sequence[Bande], seuil: float) -> Bilan:
    """Bilan d'un seuil sur des bandes (fines de préférence : exact au pas de
    l'évaluation). Une bande à cheval sur le seuil compte avec son ``lo``."""
    s = float(seuil)
    gardees = [b for b in bandes if b.lo >= s - 1e-9]
    return Bilan(
        s, sum(b.tp for b in gardees), sum(b.fp for b in gardees),
        sum(b.tp for b in bandes), sum(b.fp for b in bandes),
    )


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


def _charger_bloc(model_dir: Path, source_rel: Optional[str]) -> Mapping[str, Any]:
    """Le bloc ``modeles[<id>]`` de l'évaluation livrée ; ``{}`` si rien de lisible."""
    model_dir = Path(model_dir)
    chemin = model_dir / (source_rel or EVAL_DEFAUT)
    try:
        evaluation = json.loads(chemin.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(evaluation, Mapping):
        return {}
    bloc = _bloc_modele(evaluation, model_dir.name)
    return bloc if isinstance(bloc, Mapping) else {}


def _bandes_zone(bloc: Mapping[str, Any], zone: str, classe: str) -> List[Bande]:
    pzc = bloc.get("par_zone_classe") or {}
    return _bandes_depuis(((pzc.get(zone) or {}).get(classe) or {}).get("bandes"))


def charger_bandes(
    model_dir: Path,
    classe: str,
    source_rel: Optional[str] = None,
    zones: Optional[Sequence[str]] = None,
) -> List[Bande]:
    """Bandes (pas de l'évaluation) de ``classe`` ; liste vide si rien de lisible."""
    bloc = _charger_bloc(model_dir, source_rel)
    if not bloc:
        return []
    if zones:
        somme = _sommer([liste for liste in (_bandes_zone(bloc, z, classe) for z in zones) if liste])
        if somme:
            return somme
    par_classe = ((bloc.get("par_classe") or {}).get(classe) or {}).get("etude_seuil") or {}
    bandes = _bandes_depuis(par_classe.get("bandes"))
    if bandes:
        return bandes
    return _bandes_depuis(((bloc.get("global") or {}).get("etude_seuil") or {}).get("bandes"))


def charger_bandes_par_zone(
    model_dir: Path,
    classe: str,
    source_rel: Optional[str] = None,
    zones: Optional[Sequence[str]] = None,
) -> List[Tuple[str, List[Bande]]]:
    """``[(zone, bandes), …]`` de ``classe``, une entrée par zone d'évaluation qui
    porte des bandes — restreint aux ``zones`` déclarées quand il y en a (même
    périmètre que la mesure de fiabilité), dans l'ordre du fichier."""
    bloc = _charger_bloc(model_dir, source_rel)
    pzc = bloc.get("par_zone_classe") or {}
    if not isinstance(pzc, Mapping):
        return []
    cles = [z for z in zones if z in pzc] if zones else list(pzc.keys())
    out = []
    for z in cles:
        bandes = _bandes_zone(bloc, str(z), classe)
        if bandes:
            out.append((str(z), bandes))
    return out


def seuil_f1max(model_dir: Path, classe: str, source_rel: Optional[str] = None) -> Optional[float]:
    """Seuil d'équilibre précision-rappel : par classe, sinon global — lu dans
    l'évaluation de **référence** (``EVAL_DEFAUT``), celle où le seuil déployé est
    choisi (fenêtre [bas du plateau 95 % ; F1-max], règle 2026-09-09), et non dans
    l'évaluation de couverture des linéaires, dont le F1-max (0,185) est sous le
    seuil déployé (0,26) et contredirait la légende « choisi en dessous »."""
    bloc = _charger_bloc(model_dir, source_rel or EVAL_DEFAUT)
    for conteneur in ((bloc.get("par_classe") or {}).get(classe) or {}, bloc.get("global") or {}):
        v = conteneur.get("seuil_f1max") if isinstance(conteneur, Mapping) else None
        try:
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def tableau_precision_rappel(
    model_dir: Path, classe: str, source_rel: Optional[str] = None
) -> List[Tuple[float, float, float]]:
    """``[(seuil, P, R), …]`` de ``etude_seuil.tableau`` (pas de 0,05) — par classe,
    sinon global — dans la **même évaluation que les bandes** (``fiabilite.source``) :
    le rappel y est celui du critère de la classe (couverture pour les linéaires)."""
    bloc = _charger_bloc(model_dir, source_rel)
    for conteneur in ((bloc.get("par_classe") or {}).get(classe) or {}, bloc.get("global") or {}):
        if not isinstance(conteneur, Mapping):
            continue
        lignes = (conteneur.get("etude_seuil") or {}).get("tableau")
        out: List[Tuple[float, float, float]] = []
        for ligne in lignes or []:
            try:
                out.append((float(ligne["seuil"]), float(ligne["P"]), float(ligne["R"])))
            except (KeyError, TypeError, ValueError):
                continue
        if out:
            return sorted(out)
    return []


def _interpoler_rappel(tableau: Sequence[Tuple[float, float, float]], seuil: float) -> Optional[float]:
    if not tableau:
        return None
    if seuil <= tableau[0][0]:
        return tableau[0][2]
    if seuil >= tableau[-1][0]:
        return tableau[-1][2]
    for (s0, _p0, r0), (s1, _p1, r1) in zip(tableau, tableau[1:]):
        if s0 <= seuil <= s1:
            return r0 if s1 == s0 else r0 + (r1 - r0) * (seuil - s0) / (s1 - s0)
    return None


def precision_rappel(profil: Profil, seuil: float) -> Tuple[Optional[float], Optional[float]]:
    """``(précision, rappel)`` au banc pour ``seuil`` : la précision est comptée sur
    les bandes fines (vraies / gardées, même provenance que la figure), le rappel lu
    dans la table de l'évaluation par interpolation linéaire entre deux pas de 0,05
    (les bandes ne connaissent pas les objets manqués). ``None`` quand la donnée
    manque — un profil par zone n'a pas de table, donc pas de rappel."""
    b = bilan_au_seuil(profil.fines or profil.bandes, seuil)
    gardees = b.vrais_gardes + b.fausses_gardees
    precision = b.vrais_gardes / gardees if gardees else None
    return precision, _interpoler_rappel(profil.tableau, float(seuil))


def phrase_precision_rappel(precision: Optional[float], rappel: Optional[float]) -> str:
    """« précision 65 % · rappel 72 % » ; une seule partie si l'autre manque ; ``""`` sans rien."""
    morceaux = []
    if precision is not None:
        morceaux.append(f"précision {round(precision * 100)} %")
    if rappel is not None:
        morceaux.append(f"rappel {round(rappel * 100)} %")
    return " · ".join(morceaux)


_PREFIXE_ZONE = re.compile(r"^(\d+|ie)_")


def libelle_zone(zone: str) -> str:
    """Un libellé lisible depuis l'identifiant de zone de l'évaluation :
    ``grand_est/54_foret_de_haye`` → « Foret de haye », ``irlande/ie_galway_01`` →
    « Galway 01 ». Les accents ne sont pas restitués (l'identifiant ne les porte
    pas) ; l'identifiant complet reste en infobulle."""
    nom = zone.rsplit("/", 1)[-1]
    nom = _PREFIXE_ZONE.sub("", nom).replace("_", " ").strip()
    return nom[:1].upper() + nom[1:] if nom else zone


def disposer_etiquettes(
    elements: Sequence[Tuple[float, float]], largeur_totale: float, ecart: float = 4.0
) -> List[Tuple[int, float]]:
    """Range des étiquettes ``(centre, largeur)``, données de gauche à droite, sur le
    moins de rangées possible **sans chevauchement** : chacune prend la première
    rangée où elle tient à droite de la précédente (``ecart`` de marge), sinon une
    rangée de plus. Une étiquette est d'abord centrée sur sa bande, puis ramenée dans
    ``[0, largeur_totale]``. Renvoie ``(rangée, x gauche)`` par étiquette.

    C'est ce qui empêche « possible » et « probable » de se superposer sous le
    profil des scores quand deux niveaux sont étroits (constat utilisateur
    2026-10-08), en mini comme en complet.
    """
    fins: List[float] = []
    out: List[Tuple[int, float]] = []
    for cx, largeur in elements:
        gauche = min(max(cx - largeur / 2, 0.0), max(0.0, largeur_totale - largeur))
        droite = gauche + largeur
        for r, fin in enumerate(fins):
            if gauche >= fin + ecart:
                fins[r] = droite
                out.append((r, gauche))
                break
        else:
            fins.append(droite)
            out.append((len(fins) - 1, gauche))
    return out


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
    bins = [Bande(limites[i], limites[i + 1], tp, fp) for i, (tp, fp) in sorted(acc.items())]
    return _fusionner_eclats(bins, pas, coupures)


def _fusionner_eclats(bins: List[Bande], pas: float, coupures: Sequence[float]) -> List[Bande]:
    """Une bande plus étroite qu'une demi-case, coincée entre la grille et une coupure
    (0,25–0,26 pour un seuil à 0,26), se dessinait en puce flottante : elle rejoint
    la bande voisine **du même côté de la coupure**, jamais de l'autre côté."""
    coupes = {round(float(c), 6) for c in coupures}
    out: List[Bande] = []
    i = 0
    while i < len(bins):
        b = bins[i]
        eclat = (b.hi - b.lo) < pas / 2 - 1e-9
        if eclat and round(b.hi, 6) in coupes and out and round(out[-1].hi, 6) == round(b.lo, 6):
            prev = out.pop()                     # éclat à gauche d'une coupure → bande précédente
            out.append(Bande(prev.lo, b.hi, prev.tp + b.tp, prev.fp + b.fp))
        elif eclat and round(b.lo, 6) in coupes and i + 1 < len(bins):
            nxt = bins[i + 1]                    # éclat à droite d'une coupure → bande suivante
            bins[i + 1] = Bande(b.lo, nxt.hi, b.tp + nxt.tp, b.fp + nxt.fp)
        else:
            out.append(b)
        i += 1
    return out


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


def _profil(classe: str, bandes: Sequence[Bande], categories: Sequence[Categorie], pas: float,
            seuil_eq: Optional[float], zone: str = "",
            tableau: Sequence[Tuple[float, float, float]] = ()) -> Profil:
    cats = tuple(sorted(categories, key=lambda c: c.seuil))
    seuil = cats[0].seuil
    return Profil(
        classe=classe,
        bandes=tuple(agreger(bandes, pas, [c.seuil for c in cats])),
        categories=cats,
        n_sous_seuil=sum(b.total for b in bandes if b.hi <= seuil + 1e-9),
        fines=tuple(bandes),
        seuil_f1max=seuil_eq,
        zone=zone,
        tableau=tuple(tableau),
    )


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
    return _profil(classe, bandes, categories, pas, seuil_f1max(model_dir, classe),
                   tableau=tableau_precision_rappel(model_dir, classe, source))


def profils_par_zone(
    model_dir: Path, classe: str, categories: Sequence[Categorie], pas: float = PAS_DEFAUT
) -> List[Profil]:
    """Un profil par zone d'évaluation (petits multiples), avec les coupures de la
    classe ; vide s'il y a moins de deux zones — une seule n'apprend rien de plus
    que le profil global."""
    if not categories:
        return []
    source, zones = _source_et_zones(model_dir, classe)
    par_zone = charger_bandes_par_zone(model_dir, classe, source, zones)
    if len(par_zone) < 2:
        return []
    return [_profil(classe, bandes, categories, pas, None, zone=z) for z, bandes in par_zone]
