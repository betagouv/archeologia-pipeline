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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    n_gt: Optional[int] = None             # objets annotés (zone ou classe) : rappel = vraies gardées / n_gt…
    critere: str = ""                      # … seulement au critère objet (« iou ») ; « couverture » = fragments
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


def _charger_eval(model_dir: Path, source_rel: Optional[str]) -> Mapping[str, Any]:
    """L'évaluation livrée entière (``critere`` en tête, ``modeles``) ; ``{}`` si illisible."""
    chemin = Path(model_dir) / (source_rel or EVAL_DEFAUT)
    try:
        evaluation = json.loads(chemin.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return evaluation if isinstance(evaluation, Mapping) else {}


def _charger_bloc(model_dir: Path, source_rel: Optional[str]) -> Mapping[str, Any]:
    """Le bloc ``modeles[<id>]`` de l'évaluation livrée ; ``{}`` si rien de lisible."""
    evaluation = _charger_eval(model_dir, source_rel)
    if not evaluation:
        return {}
    bloc = _bloc_modele(evaluation, Path(model_dir).name)
    return bloc if isinstance(bloc, Mapping) else {}


def critere_evaluation(model_dir: Path, source_rel: Optional[str] = None) -> str:
    """``critere`` en tête de l'évaluation : « iou » (objets appariés un à un) ou
    « couverture » (linéaires : les vraies sont des fragments, pas des objets) ; ``""`` inconnu.

    Les évaluations antérieures à l'ajout du champ (enclos, dépressions, ponctuelles)
    ne l'écrivent pas : on le déduit alors du nom de l'évaluation (« couverture » dans
    le chemin) ou de la présence d'un appariement IoU (``iou`` / ``appariement`` en tête).
    """
    evaluation = _charger_eval(model_dir, source_rel)
    explicite = str(evaluation.get("critere") or "").strip().lower()
    if explicite:
        return explicite
    if "couverture" in (source_rel or "").lower():
        return "couverture"
    if evaluation and ("iou" in evaluation or "appariement" in evaluation):
        return "iou"
    return ""


def _ngt_zone(bloc: Mapping[str, Any], zone: str, classe: str) -> Optional[int]:
    v = ((bloc.get("par_zone_classe") or {}).get(zone) or {}).get(classe) or {}
    try:
        return int(v["n_gt"]) if isinstance(v, Mapping) and v.get("n_gt") is not None else None
    except (TypeError, ValueError):
        return None


def charger_ngt_par_zone(
    model_dir: Path, classe: str, source_rel: Optional[str] = None, zones: Optional[Sequence[str]] = None
) -> Dict[str, int]:
    """``{zone: n_gt}`` (objets annotés de ``classe``), zones déclarées ou toutes."""
    bloc = _charger_bloc(model_dir, source_rel)
    pzc = bloc.get("par_zone_classe") or {}
    if not isinstance(pzc, Mapping):
        return {}
    cles = [z for z in zones if z in pzc] if zones else list(pzc.keys())
    out: Dict[str, int] = {}
    for z in cles:
        n = _ngt_zone(bloc, str(z), classe)
        if n is not None:
            out[str(z)] = n
    return out


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
    (les bandes ne connaissent pas les objets manqués). Sans table — un profil par
    zone — le rappel est **vraies gardées / objets annotés** (``n_gt``), exact au
    critère objet (« iou » : une vraie = un objet apparié) et impossible au critère
    de couverture (une vraie = un fragment, il y en a plus que d'objets) → ``None``."""
    b = bilan_au_seuil(profil.fines or profil.bandes, seuil)
    gardees = b.vrais_gardes + b.fausses_gardees
    precision = b.vrais_gardes / gardees if gardees else None
    rappel = _interpoler_rappel(profil.tableau, float(seuil))
    if rappel is None and profil.n_gt and profil.critere == "iou":
        rappel = min(1.0, b.vrais_gardes / profil.n_gt)
    return precision, rappel


def f1max_depuis_bandes(bandes: Sequence[Bande], n_gt: Optional[int]) -> Optional[float]:
    """Le seuil qui maximise F1 sur des bandes fines, au critère objet : à chaque borne
    basse ``s``, F1(s) = 2·TP(s) / (TP(s) + FP(s) + n_gt) (les manqués valent n_gt − TP).
    C'est le « équilibre (F1) » d'une zone d'évaluation, que le fichier ne donne pas
    par zone. ``None`` sans bandes ni objets annotés ; au critère de couverture,
    l'appelant ne doit pas s'en servir (les vraies sont des fragments)."""
    if not bandes or not n_gt:
        return None
    meilleur, f1_max = None, -1.0
    for b in sorted(bandes, key=lambda x: x.lo):
        tp = sum(x.tp for x in bandes if x.lo >= b.lo - 1e-9)
        fp = sum(x.fp for x in bandes if x.lo >= b.lo - 1e-9)
        f1 = 2 * tp / (tp + fp + n_gt) if (tp + fp + n_gt) else 0.0
        if f1 > f1_max + 1e-12:
            meilleur, f1_max = b.lo, f1
    return meilleur


def phrase_precision_rappel(precision: Optional[float], rappel: Optional[float]) -> str:
    """« précision 65 % · rappel 72 % · F1 68 % » ; le F1 dès que les deux existent, une
    seule partie si l'autre manque ; ``""`` sans rien."""
    morceaux = []
    if precision is not None:
        morceaux.append(f"précision {round(precision * 100)} %")
    if rappel is not None:
        morceaux.append(f"rappel {round(rappel * 100)} %")
    if precision is not None and rappel is not None and (precision + rappel) > 0:
        morceaux.append(f"F1 {round(2 * precision * rappel / (precision + rappel) * 100)} %")
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


def _placer_glouton(
    libelles: Sequence[Tuple[float, Sequence[Tuple[str, float]], Sequence[int]]],
    nb_rangees: int,
    largeur_totale: float,
    ecart: float,
    marge: float,
    premier_a_gauche: bool,
) -> List[Optional[Tuple[int, float, str, int]]]:
    occupe: List[List[Tuple[float, float]]] = [[] for _ in range(max(0, nb_rangees))]
    out: List[Optional[Tuple[int, float, str, int]]] = []
    for k, (x_ligne, variantes, rangees) in enumerate(libelles):
        place: Optional[Tuple[int, float, str, int]] = None
        for iv, (texte, largeur) in enumerate(variantes):
            cotes = (x_ligne + marge, x_ligne - marge - largeur)
            if k == 0 and premier_a_gauche:
                cotes = cotes[::-1]
            for r in rangees:
                if not 0 <= r < len(occupe):
                    continue
                for gauche in cotes:
                    if gauche < 0 or gauche + largeur > largeur_totale:
                        continue
                    if all(gauche + largeur + ecart <= a or b + ecart <= gauche for a, b in occupe[r]):
                        place = (r, gauche, texte, iv)
                        occupe[r].append((gauche, gauche + largeur))
                        break
                if place:
                    break
            if place:
                break
        out.append(place)
    return out


def placer_libelles_lignes(
    libelles: Sequence[Tuple[float, Sequence[Tuple[str, float]], Sequence[int]]],
    nb_rangees: int,
    largeur_totale: float,
    ecart: float = 6.0,
    marge: float = 3.0,
) -> List[Optional[Tuple[int, float, str]]]:
    """Place les libellés attachés à une ligne verticale (seuil, coupures, équilibre),
    donnés **par ordre de priorité** : ``(x de la ligne, [(texte, largeur), …] variantes
    de la plus longue à la plus courte, rangées essayées)``. Chacun prend la première
    variante qui tient, sur la première rangée essayée libre, à droite de sa ligne
    sinon à gauche, sans toucher un libellé déjà posé (``ecart``) ni sortir du cadre ;
    sinon il est omis (``None``) — sa ligne reste, son nom est dans l'infobulle.
    Le premier libellé (le seuil) passe à **gauche** de sa ligne quand cela permet de
    placer davantage de libellés, ou des formes moins abrégées : à 0,29, « seuil 0,29 »
    à droite de sa ligne occupait toute la place de « équilibre (F1) 0,37 ».
    Renvoie ``(rangée, x gauche, texte)`` par libellé, dans l'ordre d'entrée.

    Option A retenue par l'utilisateur (2026-10-08) : la rangée du haut porte les
    seules mesures (précision, rappel, F1) ; seuil, coupures et équilibre se rangent
    ici, en dessous — « équilibre (F1) » et les mesures se recouvraient sur une carte
    étroite.
    """
    def score(res):
        return (res[0] is None if res else False,          # le seuil d'abord
                sum(1 for r in res if r is None),            # puis le moins d'omis
                sum(r[3] for r in res if r is not None))     # puis le moins d'abrégés
    essais = [_placer_glouton(libelles, nb_rangees, largeur_totale, ecart, marge, a_gauche)
              for a_gauche in (False, True)]
    meilleur = min(essais, key=score)                        # à égalité : seuil à droite
    return [None if r is None else (r[0], r[1], r[2]) for r in meilleur]


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
            tableau: Sequence[Tuple[float, float, float]] = (),
            n_gt: Optional[int] = None, critere: str = "") -> Profil:
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
        n_gt=n_gt,
        critere=critere,
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
    ngt = charger_ngt_par_zone(model_dir, classe, source, zones) if zones else {}
    return _profil(classe, bandes, categories, pas, seuil_f1max(model_dir, classe),
                   tableau=tableau_precision_rappel(model_dir, classe, source),
                   n_gt=sum(ngt.values()) if ngt else None, critere=critere_evaluation(model_dir, source))


def zones_sans_objet(model_dir: Path, classe: str) -> List[str]:
    """Les zones d'évaluation où ``classe`` n'a **aucun objet annoté** (``n_gt`` = 0) :
    on n'y mesure que des fausses détections, ni rappel ni fiabilité — elles ne
    figurent pas parmi les petits multiples, la fiche dit lesquelles."""
    source, zones = _source_et_zones(model_dir, classe)
    return [z for z, n in charger_ngt_par_zone(model_dir, classe, source, zones).items() if n == 0]


def profils_par_zone(
    model_dir: Path, classe: str, categories: Sequence[Categorie], pas: float = PAS_DEFAUT
) -> List[Profil]:
    """Un profil par zone d'évaluation (petits multiples), avec les coupures de la
    classe ; sans les zones où la classe n'a aucun objet annoté (``n_gt`` = 0 :
    rien à mesurer, cf. :func:`zones_sans_objet`) ; vide s'il reste moins de deux
    zones — une seule n'apprend rien de plus que le profil global."""
    if not categories:
        return []
    source, zones = _source_et_zones(model_dir, classe)
    ngt = charger_ngt_par_zone(model_dir, classe, source, zones)
    par_zone = [(z, b) for z, b in charger_bandes_par_zone(model_dir, classe, source, zones) if ngt.get(z) != 0]
    if len(par_zone) < 2:
        return []
    critere = critere_evaluation(model_dir, source)
    # Point d'équilibre F1 de la zone : recalculé depuis ses bandes et ses objets annotés
    # (critère objet seulement — le fichier ne le donne pas par zone).
    return [_profil(classe, bandes, categories, pas,
                    f1max_depuis_bandes(bandes, ngt.get(z)) if critere == "iou" else None,
                    zone=z, n_gt=ngt.get(z), critere=critere)
            for z, bandes in par_zone]
