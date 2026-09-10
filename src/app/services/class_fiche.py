"""Fiche d'une classe détectable — construction depuis ``model_card.yaml``.

Ce que l'archéologue doit pouvoir lire AVANT de cocher une entité à l'étape 3 :
à quoi ressemble la structure sur un RVT, **où** et **en quelle quantité** le
modèle l'a apprise, ce qu'elle n'est explicitement pas, et dans quelle optique
de prospection s'en servir.

Le contrat vit dans ``model_card.yaml``, par classe, sous la clé ``fiche`` —
même endroit que ``label_fr``, donc versionné avec le modèle et produit une
fois à l'évaluation ::

    classes:
      - id: 0
        name: depression_circulaire_grande
        label_fr: Dépression circulaire grande
        description: Grande dépression circulaire en cuvette.
        fiche:
          resume: Mardelles, dolines et cuvettes de 14 m et plus.
          reconnaitre: Cuvette sombre à contour net, souvent en grappe.
          usage: Plateaux et massifs forestiers, LD 0,5 m.
          hors_cible:
            - Dépressions de moins de 14 m
          vignettes:
            - brut: vignettes/depression_00_brut.jpg
              annote: vignettes/depression_00_annote.jpg
              zone: Chailluz (25)
              legende: 15 cuvettes sur une dalle de test
          entrainement:
            corpus: depressions_grandes_648_v1
            annotation: masques SAM 2.1 sur boîtes revues à la main
            zones:
              - {nom: Fénétrange (57), tuiles: 620, objets: 1834}
            splits:
              train: {tuiles: 1285, objets: 3577}
              valid: {tuiles: 330, objets: 948}
              test:  {tuiles: 219, objets: 817}

**Tout est optionnel.** Un modèle sans bloc ``fiche`` produit une fiche
dégradée (repli sur ``description``), jamais une exception : les cinq modèles
installés n'ont pas encore le bloc, et l'interface doit rester utilisable
pendant que les fiches se remplissent modèle par modèle. ``est_complete`` /
``manques`` disent ce qui reste à écrire — c'est ce que l'UI affiche en creux
et ce que le validateur de métadonnées relaie.

Module PUR : pas de Qt, pas de QGIS, pas d'I/O. Les chemins de vignettes sont
rendus tels qu'écrits (relatifs au dossier du modèle) ; c'est l'appelant qui
les résout. Testable hors-QGIS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .fiabilite import Categorie, parse_fiabilite
from .vocabulaire_modele import pretty_rvt_name, pretty_task

_SPLITS_ORDRE = ("train", "valid", "test")


# ----------------------------------------------------------------------
# Dataclasses de présentation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Vignette:
    """Un cadre illustratif. ``brut`` est le RVT seul, ``annote`` le même
    cadre avec la vérité terrain dessinée (bascule dans l'UI)."""
    brut: str
    annote: str = ""
    zone: str = ""
    legende: str = ""


@dataclass(frozen=True)
class Effectif:
    """Un compte de tuiles et d'objets, sous un nom (zone ou split)."""
    nom: str
    tuiles: int = 0
    objets: int = 0


@dataclass(frozen=True)
class Entrainement:
    """Où, en quelle quantité et comment la classe a été apprise."""
    corpus: str = ""
    annotation: str = ""
    zones: Tuple[Effectif, ...] = ()
    splits: Tuple[Effectif, ...] = ()

    @property
    def total_tuiles(self) -> int:
        """Total sur les splits ; à défaut, sur les zones."""
        if self.splits:
            return sum(s.tuiles for s in self.splits)
        return sum(z.tuiles for z in self.zones)

    @property
    def total_objets(self) -> int:
        if self.splits:
            return sum(s.objets for s in self.splits)
        return sum(z.objets for z in self.zones)


@dataclass(frozen=True)
class ClassFiche:
    """Tout ce qu'on affiche d'une classe détectable."""
    nom: str                 # nom technique (classes[].name)
    label: str               # label_fr, repli sur le nom technique
    modele_id: str
    modele: str              # display_name du modèle
    resume: str = ""
    reconnaitre: str = ""
    usage: str = ""
    hors_cible: Tuple[str, ...] = ()
    vignettes: Tuple[Vignette, ...] = ()
    entrainement: Optional[Entrainement] = None
    rvt: str = ""
    rvt_label: str = ""
    resolution_m: Optional[float] = None
    seuil: Optional[float] = None
    task_label: str = ""
    statut: str = ""
    fiabilite: Tuple[Categorie, ...] = ()
    limites: Tuple[str, ...] = ()

    @property
    def manques(self) -> Tuple[str, ...]:
        """Les blocs de la fiche qui restent à écrire, dans l'ordre d'écriture."""
        out: List[str] = []
        if not self.resume:
            out.append("resume")
        if not self.usage:
            out.append("usage")
        if not self.vignettes:
            out.append("vignettes")
        if self.entrainement is None:
            out.append("entrainement")
        return tuple(out)

    @property
    def est_complete(self) -> bool:
        return not self.manques


# ----------------------------------------------------------------------
# Coercitions tolérantes
# ----------------------------------------------------------------------
def _txt(v: Any) -> str:
    """Texte affichable. Une valeur absente/non scalaire → chaîne vide."""
    if v is None or isinstance(v, (dict, list, tuple)):
        return ""
    return str(v).strip()


def _entier(v: Any) -> int:
    """Effectif. Non numérique → 0 (on n'invente pas un chiffre)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _nombre(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _liste_txt(v: Any) -> Tuple[str, ...]:
    """Liste de textes. Une chaîne seule devient un élément unique."""
    if v is None:
        return ()
    if isinstance(v, str):
        s = v.strip()
        return (s,) if s else ()
    if not isinstance(v, (list, tuple)):
        return ()
    return tuple(s for s in (_txt(x) for x in v) if s)


def _dict(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


# ----------------------------------------------------------------------
# Sous-blocs
# ----------------------------------------------------------------------
def _vignettes(raw: Any) -> Tuple[Vignette, ...]:
    """Une entrée sans ``brut`` n'a rien à montrer : elle est écartée."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Vignette] = []
    for item in raw:
        d = _dict(item)
        brut = _txt(d.get("brut"))
        if not brut:
            continue
        out.append(Vignette(
            brut=brut,
            annote=_txt(d.get("annote")),
            zone=_txt(d.get("zone")),
            legende=_txt(d.get("legende")),
        ))
    return tuple(out)


def _effectifs_zones(raw: Any) -> Tuple[Effectif, ...]:
    """Une zone sans nom n'est pas citable : elle est écartée."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Effectif] = []
    for item in raw:
        d = _dict(item)
        nom = _txt(d.get("nom"))
        if not nom:
            continue
        out.append(Effectif(nom, _entier(d.get("tuiles")), _entier(d.get("objets"))))
    return tuple(out)


def _effectifs_splits(raw: Any) -> Tuple[Effectif, ...]:
    """``{train: {...}, valid: {...}}`` → tuple ordonné train/valid/test."""
    d = _dict(raw)
    out: List[Effectif] = []
    connus = [s for s in _SPLITS_ORDRE if s in d]
    autres = [s for s in d if s not in _SPLITS_ORDRE]
    for nom in connus + sorted(autres):
        bloc = _dict(d.get(nom))
        out.append(Effectif(nom, _entier(bloc.get("tuiles")), _entier(bloc.get("objets"))))
    return tuple(out)


def _entrainement(raw: Any) -> Optional[Entrainement]:
    """``None`` si le bloc n'apporte rien — mieux vaut une absence franche
    qu'un cadre vide dans l'interface."""
    d = _dict(raw)
    if not d:
        return None
    ent = Entrainement(
        corpus=_txt(d.get("corpus")),
        annotation=_txt(d.get("annotation")),
        zones=_effectifs_zones(d.get("zones")),
        splits=_effectifs_splits(d.get("splits")),
    )
    if not (ent.corpus or ent.annotation or ent.zones or ent.splits):
        return None
    return ent


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def _classes(card: Any) -> List[Mapping[str, Any]]:
    raw = _dict(card).get("classes")
    if not isinstance(raw, (list, tuple)):
        return []
    return [c for c in raw if isinstance(c, Mapping) and _txt(c.get("name"))]


def _seuil(card: Mapping[str, Any], nom: str) -> Optional[float]:
    """Seuil PAR CLASSE, repli sur ``confidence_default`` (même règle qu'à
    l'inférence : une classe absente du dict est décodée au défaut)."""
    th = _dict(card.get("thresholds"))
    par_classe = _dict(th.get("confidence_per_class"))
    if nom in par_classe:
        v = _nombre(par_classe.get(nom))
        if v is not None:
            return v
    return _nombre(th.get("confidence_default"))


def build_class_fiche(card: Any, class_name: str) -> Optional[ClassFiche]:
    """Fiche de la classe ``class_name`` du ``model_card`` parsé.

    ``None`` si la carte est vide ou ne déclare pas cette classe. Ne lève
    jamais sur une donnée mal formée : la clé fautive est simplement ignorée.
    """
    c = _dict(card)
    if not c or not class_name:
        return None
    bloc = next((k for k in _classes(c) if _txt(k.get("name")) == class_name), None)
    if bloc is None:
        return None

    fiche = _dict(bloc.get("fiche"))
    par_classe, _prov = parse_fiabilite(c.get("thresholds"))
    rvt = _txt(_dict(c.get("preferred_rvt")).get("type")).upper()

    return ClassFiche(
        nom=class_name,
        label=_txt(bloc.get("label_fr")) or class_name,
        modele_id=_txt(c.get("id")),
        modele=_txt(c.get("display_name")) or _txt(c.get("id")),
        resume=_txt(fiche.get("resume")) or _txt(bloc.get("description")),
        reconnaitre=_txt(fiche.get("reconnaitre")),
        usage=_txt(fiche.get("usage")),
        hors_cible=_liste_txt(fiche.get("hors_cible")),
        vignettes=_vignettes(fiche.get("vignettes")),
        entrainement=_entrainement(fiche.get("entrainement")),
        rvt=rvt,
        rvt_label=pretty_rvt_name(rvt) if rvt else "",
        resolution_m=_nombre(_dict(c.get("mnt")).get("resolution")),
        seuil=_seuil(c, class_name),
        task_label=pretty_task(_txt(c.get("task"))),
        statut=_txt(c.get("status")),
        fiabilite=tuple(par_classe.get(class_name, ())),
        limites=_liste_txt(c.get("known_limitations")),
    )


def build_all_fiches(card: Any) -> Tuple[ClassFiche, ...]:
    """Une fiche par classe déclarée, dans l'ordre du ``model_card``.

    Un nom de classe dupliqué (accident d'édition) ne produit qu'une fiche.
    """
    vues: set = set()
    noms: List[str] = []
    for bloc in _classes(card):
        nom = _txt(bloc.get("name"))
        if nom not in vues:
            vues.add(nom)
            noms.append(nom)
    fiches = (build_class_fiche(card, n) for n in noms)
    return tuple(f for f in fiches if f is not None)


def fiches_par_entite(
    card: Any, classes_de_l_entite: Iterable[str]
) -> Tuple[ClassFiche, ...]:
    """Les fiches des classes qui portent une entité donnée (ordre du card).

    Une entité de catalogue peut être couverte par plusieurs classes d'un même
    modèle (cible dérivée avec ``include_source``, par exemple) : l'UI affiche
    alors une fiche par classe.
    """
    voulues = {c for c in classes_de_l_entite if c}
    return tuple(f for f in build_all_fiches(card) if f.nom in voulues)


def resume_manques(fiches: Sequence[ClassFiche]) -> str:
    """Phrase de suivi pour le validateur : quelles fiches restent à écrire."""
    incompletes = [f for f in fiches if not f.est_complete]
    if not incompletes:
        return ""
    details = ", ".join(f"{f.nom} ({'/'.join(f.manques)})" for f in incompletes)
    return f"{len(incompletes)}/{len(fiches)} fiche(s) incomplète(s) : {details}"
