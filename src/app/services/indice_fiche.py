"""Fiche d'un produit de l'étape 2 — construction depuis ``indices_fiches.json``.

Ce que l'archéologue doit pouvoir lire AVANT de cocher un produit : à quoi
ressemble l'image, ce qu'elle montre et ce qu'elle ne montre **pas**, dans quelle
optique de prospection s'en servir, comment elle est calculée, avec quels
réglages — et d'après quelles sources. C'est le pendant, côté étape 2, de la
fiche de classe détectable de l'étape 3 (:mod:`class_fiche`).

Le contrat vit dans ``data/indices_fiches.json``, une entrée par clé de produit
d':mod:`indices_model` (``MNT``, ``SVF``, ``COUVERTURE``…) ::

    {
      "SVF": {
        "resume":     "Part du ciel visible depuis chaque point…",
        "lire":       "Plat et crêtes en clair, fossés et fosses en sombre…",
        "usage":      ["Creux de toute forme, sans biais d'orientation", "…"],
        "limites":    ["En terrain plat, seules les formes en creux ressortent", "…"],
        "methode":    ["Horizon mesuré dans 16 directions jusqu'à 10 px", "…"],
        "parametres": [{"cle": "svf.radius", "label": "Rayon (px)",
                        "defaut": 10, "sens": "Distance de recherche de l'horizon."}],
        "references": [{"citation": "Zakšek, Oštir, Kokalj 2011, Remote Sensing 3(2)",
                        "url": "https://doi.org/10.3390/rs3020398"}],
        "vignettes":  [{"image": "indices_vignettes/SVF.jpg", "legende": "…",
                        "source": "…", "licence": "…",
                        "cadrage": {"x": 0.25, "y": 0.5, "cote": 0.25}}]
      }
    }

**L'identité du produit n'est pas dans ce fichier** : sigle, nom technique,
titre métier et famille restent ceux d':mod:`indices_model` et
:mod:`visu_catalogue`, source unique déjà partagée par l'étape 2 et l'onglet
Visualisation. Le JSON ne porte que ce qui s'écrit à la main.

**Tout est optionnel**, comme pour les fiches de classes : un produit sans
entrée produit une fiche dégradée (repli sur la description du catalogue),
jamais une exception — l'interface doit rester utilisable pendant que les
fiches se remplissent. ``est_complete`` / ``manques`` disent ce qui reste à
écrire, et un test de contrat vérifie que le fichier livré, lui, est complet.

Module PUR : pas de Qt, pas de QGIS. Les chemins de vignettes sont rendus tels
qu'écrits (relatifs à ``data/``) ; c'est l'appelant qui les résout. La seule
I/O est la lecture du JSON livré, comme :func:`visu_catalogue.load_catalogue`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from .class_fiche import cadrage_fractions
from .indices_model import all_products
from .visu_catalogue import IndiceInfo, indice_info

#: Nom du fichier livré, sous ``data/``.
FICHIER = "indices_fiches.json"

#: Clé du bloc de comparaison dans ce fichier. Les clés préfixées ``_`` ne sont
#: pas des produits : :func:`build_all_fiches` n'itère que sur
#: :func:`indices_model.all_products`, elles sont donc ignorées d'office.
CLE_COMPARAISON = "_comparaison"

#: Valeurs admises dans une case de tableau, de la plus favorable à la moins.
#: ``""`` = sans objet (une colonne de structure pour un produit de qualité).
VERDICTS = ("oui", "partiel", "non")


def default_fiches_path() -> Path:
    """``<racine du plugin>/data/indices_fiches.json``.

    Ce module vit dans ``src/app/services/`` : trois niveaux au-dessus de la
    racine du plugin, qui porte ``data/``.
    """
    return Path(__file__).resolve().parents[3] / "data" / FICHIER


# ----------------------------------------------------------------------
# Dataclasses de présentation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Parametre:
    """Un réglage de l'étape 2, cité avec son défaut et ce qu'il change.

    ``cle`` est le chemin dans la config écrite par l'étape 2
    (``svf.radius``, ``processing.mnt_resolution``) : c'est ce qui permet de
    vérifier qu'une fiche ne documente pas un réglage fantôme.
    """
    cle: str
    label: str = ""
    defaut: str = ""
    sens: str = ""


@dataclass(frozen=True)
class Reference:
    """Une source. ``citation`` est obligatoire, l'``url`` est un confort."""
    citation: str
    url: str = ""


@dataclass(frozen=True)
class VignetteIndice:
    """Un cadre illustratif du produit.

    ``source`` dit d'où vient l'image (dalle, zone) et ``licence`` sous quelles
    conditions elle est livrée — une fiche embarque des images qui ne sont pas
    toujours les nôtres. ``cadrage`` = ``(x, y, côté)`` en fractions, la fenêtre
    que découpe l'icône de la carte, même contrat que les fiches de classes.
    """
    image: str
    legende: str = ""
    source: str = ""
    licence: str = ""
    cadrage: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class Colonne:
    """Une colonne de tableau comparatif : un critère, avec son explication."""
    cle: str
    libelle: str = ""
    aide: str = ""


@dataclass(frozen=True)
class Tableau:
    """Un tableau comparatif : des produits en lignes, des critères en colonnes.

    ``cases[cle_produit][cle_colonne]`` vaut ``"oui"``, ``"partiel"``, ``"non"``
    ou ``""`` (sans objet). Une valeur inconnue est ramenée à ``""`` : mieux
    vaut une case vide qu'une affirmation inventée.
    """
    titre: str
    colonnes: Tuple[Colonne, ...] = ()
    cases: Mapping[str, Mapping[str, str]] = None
    source: str = ""

    def verdict(self, produit: str, colonne: str) -> str:
        """Case du tableau, ``""`` si absente."""
        return (self.cases or {}).get(produit, {}).get(colonne, "")


@dataclass(frozen=True)
class Comparaison:
    """Ce qui permet de choisir ENTRE les produits, et non de lire l'un d'eux.

    Une fiche répond « que montre ce produit » ; ces tableaux répondent « lequel
    prendre pour ce que je cherche », qui est la question qu'on se pose à
    l'étape 2 devant la liste des produits.
    """
    tableaux: Tuple[Tableau, ...] = ()
    legende: Tuple[Tuple[str, str], ...] = ()
    note: str = ""

    @property
    def est_vide(self) -> bool:
        return not self.tableaux


@dataclass(frozen=True)
class IndiceFiche:
    """Tout ce qu'on affiche d'un produit de l'étape 2."""
    info: IndiceInfo
    resume: str = ""
    lire: str = ""
    usage: Tuple[str, ...] = ()
    limites: Tuple[str, ...] = ()
    methode: Tuple[str, ...] = ()
    parametres: Tuple[Parametre, ...] = ()
    references: Tuple[Reference, ...] = ()
    vignettes: Tuple[VignetteIndice, ...] = ()

    # -- identité, déléguée au catalogue --------------------------------
    @property
    def cle(self) -> str:
        return self.info.key

    @property
    def tag(self) -> str:
        """Badge court affiché sur la carte (``SVF``, ``M-HS``…)."""
        return self.info.sigle

    @property
    def nom(self) -> str:
        """Nom technique (``Sky-View Factor``)."""
        return self.info.name

    @property
    def titre(self) -> str:
        """Titre métier (``Creux & dépressions``)."""
        return self.info.metier

    @property
    def famille(self) -> str:
        return self.info.family

    # -- suivi de rédaction ---------------------------------------------
    @property
    def manques(self) -> Tuple[str, ...]:
        """Les blocs qui restent à écrire, dans l'ordre d'écriture.

        ``resume`` n'y figure pas : il se replie toujours sur la description du
        catalogue, donc il n'est jamais vide. ``vignettes`` non plus : une fiche
        se lit sans image, et les vignettes se produisent en fin de chaîne.
        """
        out: List[str] = []
        if not self.lire:
            out.append("lire")
        if not self.usage:
            out.append("usage")
        if not self.limites:
            out.append("limites")
        if not self.methode:
            out.append("methode")
        return tuple(out)

    @property
    def est_complete(self) -> bool:
        return not self.manques


# ----------------------------------------------------------------------
# Coercitions tolérantes (mêmes règles que class_fiche)
# ----------------------------------------------------------------------
def _txt(v: Any) -> str:
    """Texte affichable. Une valeur absente/non scalaire → chaîne vide.

    Un nombre est rendu tel quel : un défaut de réglage s'écrit ``10`` dans le
    JSON et s'affiche « 10 » — la fiche présente, elle ne calcule pas.
    """
    if v is None or isinstance(v, (dict, list, tuple, bool)):
        return ""
    return str(v).strip()


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
def _parametres(raw: Any) -> Tuple[Parametre, ...]:
    """Un réglage sans ``cle`` n'est pas vérifiable : il est écarté."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Parametre] = []
    for item in raw:
        d = _dict(item)
        cle = _txt(d.get("cle"))
        if not cle:
            continue
        out.append(Parametre(
            cle=cle,
            label=_txt(d.get("label")),
            defaut=_txt(d.get("defaut")),
            sens=_txt(d.get("sens")),
        ))
    return tuple(out)


def _references(raw: Any) -> Tuple[Reference, ...]:
    """Une référence sans citation n'est pas une source : elle est écartée."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Reference] = []
    for item in raw:
        d = _dict(item)
        citation = _txt(d.get("citation"))
        if not citation:
            continue
        out.append(Reference(citation=citation, url=_txt(d.get("url"))))
    return tuple(out)


def _vignettes(raw: Any) -> Tuple[VignetteIndice, ...]:
    """Une entrée sans ``image`` n'a rien à montrer : elle est écartée."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[VignetteIndice] = []
    for item in raw:
        d = _dict(item)
        image = _txt(d.get("image"))
        if not image:
            continue
        out.append(VignetteIndice(
            image=image,
            legende=_txt(d.get("legende")),
            source=_txt(d.get("source")),
            licence=_txt(d.get("licence")),
            cadrage=cadrage_fractions(d.get("cadrage")),
        ))
    return tuple(out)


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def _colonnes(raw: Any) -> Tuple[Colonne, ...]:
    """Une colonne sans clé n'est pas adressable : elle est écartée."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Colonne] = []
    for item in raw:
        d = _dict(item)
        cle = _txt(d.get("cle"))
        if not cle:
            continue
        out.append(Colonne(cle, _txt(d.get("libelle")) or cle, _txt(d.get("aide"))))
    return tuple(out)


def _cases(raw: Any, colonnes: Tuple[Colonne, ...]) -> Mapping[str, Mapping[str, str]]:
    """Cases nettoyées : seules les colonnes déclarées et les verdicts connus.

    Une faute de frappe dans un verdict laisse la case vide plutôt que de faire
    afficher un symbole arbitraire.
    """
    valides = {c.cle for c in colonnes}
    out: dict = {}
    for produit, ligne in _dict(raw).items():
        propre = {
            k: v for k, v in (
                (k, _txt(v)) for k, v in _dict(ligne).items() if k in valides
            ) if v in VERDICTS
        }
        if propre:
            out[produit] = propre
    return out


def build_comparaison(donnees: Any) -> Comparaison:
    """Tableaux comparatifs du fichier livré. Bloc absent → comparaison vide.

    L'interface doit alors simplement ne pas proposer l'entrée, jamais échouer.
    """
    bloc = _dict(_dict(donnees).get(CLE_COMPARAISON))
    tableaux: List[Tableau] = []
    raw = bloc.get("tableaux")
    if isinstance(raw, (list, tuple)):
        for t in raw:
            d = _dict(t)
            colonnes = _colonnes(d.get("colonnes"))
            titre = _txt(d.get("titre"))
            if not (titre and colonnes):
                continue
            tableaux.append(Tableau(
                titre=titre,
                colonnes=colonnes,
                cases=_cases(d.get("cases"), colonnes),
                source=_txt(d.get("source")),
            ))
    legende = tuple(
        (k, _txt(v)) for k, v in _dict(bloc.get("legende")).items()
        if k in VERDICTS and _txt(v)
    )
    return Comparaison(
        tableaux=tuple(tableaux), legende=legende, note=_txt(bloc.get("note"))
    )


def build_indice_fiche(donnees: Any, cle: str) -> IndiceFiche:
    """Fiche du produit ``cle``, montée sur le catalogue + le JSON livré.

    Rend **toujours** une fiche : un produit non documenté se présente avec la
    description du catalogue, une clé inconnue avec elle-même. Ne lève jamais
    sur une donnée mal formée — la clé fautive est simplement ignorée.
    """
    info = indice_info(cle)
    bloc = _dict(_dict(donnees).get(cle))
    return IndiceFiche(
        info=info,
        resume=_txt(bloc.get("resume")) or info.description,
        lire=_txt(bloc.get("lire")),
        usage=_liste_txt(bloc.get("usage")),
        limites=_liste_txt(bloc.get("limites")),
        methode=_liste_txt(bloc.get("methode")),
        parametres=_parametres(bloc.get("parametres")),
        references=_references(bloc.get("references")),
        vignettes=_vignettes(bloc.get("vignettes")),
    )


def build_all_fiches(donnees: Any) -> Tuple[IndiceFiche, ...]:
    """Une fiche par produit du pipeline, dans l'ordre des cartes de l'étape 2.

    C'est-à-dire l'ordre d':func:`indices_model.all_products` — les produits de
    base puis les indices RVT, exactement comme ``base_keys() + rvt_keys()`` les
    dispose à l'écran. La liste de gauche de la fiche doit se lire comme le
    sélecteur qu'on vient de quitter (demande utilisateur 2026-09-16) ; toute
    autre séquence oblige à chercher.

    ⚠ Ne pas reprendre :data:`visu_catalogue.DISPLAY_ORDER` : c'est l'ordre du
    mur de l'onglet Visualisation, qui range par intérêt de consultation et non
    par ordre de sélection. Les deux sont légitimes, dans leur contexte.
    """
    return tuple(build_indice_fiche(donnees, p.key) for p in all_products())


def load_indices_fiches(path: Optional[Path] = None) -> Mapping[str, Any]:
    """Lit le JSON livré. Fichier absent ou illisible → ``{}``.

    Une fiche manquante dégrade l'affichage, elle ne doit jamais empêcher
    l'étape 2 de s'ouvrir : le plugin peut être installé à la main, ou le
    fichier abîmé par une édition. Le test de contrat, lui, échoue bruyamment
    si le fichier livré n'est pas complet.
    """
    p = Path(path) if path is not None else default_fiches_path()
    try:
        brut = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return brut if isinstance(brut, dict) else {}
