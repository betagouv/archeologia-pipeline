"""Catalogue de consultation (onglet « Visualisation »), pur et testable.

Le catalogue est un JSON **statique** — aucun serveur applicatif : une racine,
une entrée par département, une entrée par indice consultable. C'est ce que lit
le mur visuel de l'onglet Visualisation pour proposer d'ouvrir un raster déjà
produit, sans relancer de pipeline.

Ce module ne connaît ni Qt ni QGIS : il modélise, il charge, il filtre. Toute
la partie widgets vit dans ``src/ui/visualisation_tab.py``.

**Le vocabulaire métier est ici la clé de lisibilité.** Un archéologue reconnaît
« Creux & dépressions » ; il ne décode pas « SVF ». Le nom technique reste
affiché en seconde ligne, jamais en titre. Les noms techniques et descriptions
viennent de :mod:`indices_model` (source de vérité des produits du pipeline) —
ce module n'ajoute que le libellé métier et la famille.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .indices_model import ProductInfo, product

# Familles affichées comme filtres au-dessus du mur.
FAMILY_BASE = "Base"
FAMILY_RVT = "RVT"
FAMILY_QUALITE = "Qualité"
FAMILY_ORDER = (FAMILY_BASE, FAMILY_RVT, FAMILY_QUALITE)

# Libellé métier + famille, par clé de produit du pipeline. L'ordre de ce
# dictionnaire est l'ordre d'affichage des cartes sur le mur : le socle
# (relief nu, ombrage, creux) d'abord, la qualité de la donnée en dernier.
_METIER: Dict[str, tuple] = {
    "MNT":        ("Le relief nu",            FAMILY_BASE),
    "M_HS":       ("Ombrage du relief",       FAMILY_RVT),
    "SVF":        ("Creux & dépressions",     FAMILY_RVT),
    "LD":         ("Structures en relief",    FAMILY_RVT),
    "SLRM":       ("Micro-reliefs",           FAMILY_RVT),
    "VAT":        ("Vue archéo optimisée",    FAMILY_RVT),
    "CVAT":       ("Vue archéo renforcée",    FAMILY_RVT),
    "MSTP":       ("Bosses & cuvettes",       FAMILY_RVT),
    "HS":         ("Ombrage simple",          FAMILY_RVT),
    "SLO":        ("Pente du terrain",        FAMILY_RVT),
    "DENSITE":    ("Qualité de la donnée",    FAMILY_QUALITE),
    "COUVERTURE": ("Fiabilité du sol mesuré", FAMILY_QUALITE),
}

#: Ordre d'affichage des indices sur le mur.
DISPLAY_ORDER: tuple = tuple(_METIER)


@dataclass(frozen=True)
class IndiceInfo:
    """Un indice tel que le mur le présente : métier d'abord, technique ensuite."""

    key: str
    sigle: str          # badge de la vignette (MNT, SVF, M-HS…)
    name: str           # nom technique, seconde ligne de la carte
    metier: str         # TITRE de la carte
    description: str
    family: str

    @property
    def rank(self) -> int:
        try:
            return DISPLAY_ORDER.index(self.key)
        except ValueError:
            return len(DISPLAY_ORDER)


def indice_info(key: str) -> IndiceInfo:
    """Fiche d'affichage d'un produit, montée depuis :mod:`indices_model`.

    Une clé inconnue reste affichable (elle se présente sous sa propre clé) :
    un catalogue publié plus tard ne doit pas casser une version installée.
    """
    metier, family = _METIER.get(key, (key, FAMILY_RVT))
    try:
        info: ProductInfo = product(key)
        return IndiceInfo(key, info.tag, info.full_name, metier, info.description, family)
    except KeyError:
        return IndiceInfo(key, key, key, metier, "", family)


@dataclass(frozen=True)
class CatalogItem:
    """Un indice consultable pour un département donné."""

    key: str
    source: str                  # tout ce que GDAL sait ouvrir : chemin, /vsicurl/…, URL WMTS
    thumbnail: str = ""          # chemin relatif au dossier du catalogue
    size_go: Optional[float] = None
    extent: Optional[Sequence[float]] = None   # [xmin, ymin, xmax, ymax] en Lambert-93
    streamed: bool = True        # False = fichier local (démo) ; pilote le libellé affiché

    @property
    def info(self) -> IndiceInfo:
        return indice_info(self.key)


@dataclass(frozen=True)
class Department:
    code: str
    name: str
    items: List[CatalogItem] = field(default_factory=list)
    updated: str = ""
    resolution: Optional[float] = None
    #: Département sur lequel ouvrir le mur — celui dont la couverture est la
    #: plus complète. Sans ce drapeau on ouvre sur le mieux pourvu.
    featured: bool = False

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def families(self) -> set:
        return {it.info.family for it in self.items}

    def sorted_items(self) -> List[CatalogItem]:
        return sorted(self.items, key=lambda it: it.info.rank)

    def items_in_family(self, family: Optional[str]) -> List[CatalogItem]:
        """``None`` (ou « Tous ») = pas de filtre."""
        items = self.sorted_items()
        if not family:
            return items
        return [it for it in items if it.info.family == family]


@dataclass(frozen=True)
class Catalogue:
    departments: List[Department] = field(default_factory=list)
    updated: str = ""

    @property
    def covered(self) -> List[Department]:
        """Départements réellement consultables — les seuls que le rail liste."""
        return [d for d in self.departments if d.count > 0]

    def by_code(self, code: str) -> Optional[Department]:
        for d in self.departments:
            if d.code == code:
                return d
        return None

    @property
    def product_count(self) -> int:
        return sum(d.count for d in self.departments)


def _fold(text: str) -> str:
    """Minuscule sans accent — « Côtes-d'Armor » doit sortir sur « cotes »."""
    norm = unicodedata.normalize("NFD", text or "")
    return "".join(c for c in norm if unicodedata.category(c) != "Mn").lower()


def filter_departments(departments: Sequence[Department], query: str) -> List[Department]:
    """Filtre sur le nom ET le code, insensible à la casse et aux accents."""
    q = _fold(query).strip()
    if not q:
        return list(departments)
    return [d for d in departments if q in _fold(d.name) or q in _fold(d.code)]


def load_catalogue(path: Path) -> Catalogue:
    """Lit un catalogue JSON. Lève ``FileNotFoundError`` / ``ValueError`` si illisible.

    Les chemins de vignette sont laissés **relatifs** : c'est l'appelant qui les
    résout contre le dossier du catalogue (le même JSON doit pouvoir être servi
    depuis un dossier local ou depuis une URL).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("catalogue : objet JSON attendu à la racine")

    default_res = raw.get("resolution")
    departments = []
    for d in raw.get("departments") or []:
        items = []
        for it in d.get("items") or []:
            key = it.get("key")
            source = it.get("source") or it.get("cog_url") or ""
            if not key or not source:
                continue  # entrée inexploitable : on l'ignore plutôt que de planter le mur
            extent = it.get("extent")
            if extent is not None and len(extent) != 4:
                extent = None
            items.append(CatalogItem(
                key=key,
                source=source,
                thumbnail=it.get("thumbnail") or "",
                size_go=it.get("size_go"),
                extent=extent,
                streamed=bool(it.get("streamed", True)),
            ))
        code = str(d.get("code") or "")
        if not code:
            continue
        departments.append(Department(
            code=code,
            name=d.get("name") or code,
            items=items,
            updated=d.get("updated") or "",
            resolution=d.get("resolution", default_res),
            featured=bool(d.get("featured", False)),
        ))
    return Catalogue(departments=departments, updated=raw.get("updated") or "")
