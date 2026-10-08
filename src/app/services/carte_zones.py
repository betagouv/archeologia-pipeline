"""Carte des zones d'apprentissage d'une classe — module PUR (sans Qt).

Le bandeau « Appris sur » du bloc « Ce que le modèle a appris » d'une fiche de classe
montre où la classe a été apprise et mesurée (mêmes zones : le découpage
apprentissage / validation / test se fait par blocs de 2 km dans chaque zone). Les
données viennent de ``data/zones_corpus.json``, produit par
``dev/fiches/zones_corpus.py`` et livré avec le plugin :

- ``noms`` : nom de zone tel que l'écrit une fiche → identifiant de zone ;
- ``zones`` : identifiant → ``{pays, emprise: [lon0, lat0, lon1, lat1]}`` (WGS84) ;
- ``contours`` : ``{"france": [anneaux], "irlande": [anneaux]}``, chaque anneau une liste
  de ``[lon, lat]`` (Natural Earth, simplifié).

Tout est tolérant : fichier absent ou zone inconnue → la zone n'est simplement pas
située, et la fiche retombe sur sa liste de zones en texte.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

FICHIER = Path("data") / "zones_corpus.json"
#: Latitude de référence de la projection (équirectangulaire corrigée) par pays.
_LAT_REF = {"france": 46.6, "irlande": 53.4}


def normaliser(nom: str) -> str:
    """Clé de rapprochement d'un nom de zone : sans accents, minuscules, espaces simples."""
    t = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class ZoneSituee:
    nom: str
    tuiles: int
    objets: int
    pays: str
    lon: float          # centre de l'emprise
    lat: float


def charger(racine_plugin: Path) -> Dict[str, Any]:
    """Le contenu de ``data/zones_corpus.json`` ; ``{}`` s'il manque ou est illisible."""
    try:
        data = json.loads((Path(racine_plugin) / FICHIER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def situer(zones: Sequence[Any], data: Mapping[str, Any]) -> List[ZoneSituee]:
    """Les zones d'une fiche (``Effectif`` : ``nom``, ``tuiles``, ``objets``) qu'on sait
    placer, dans leur ordre ; les autres sont omises."""
    noms = data.get("noms") or {}
    index = {normaliser(k): v for k, v in noms.items()} if isinstance(noms, Mapping) else {}
    emprises = data.get("zones") or {}
    out: List[ZoneSituee] = []
    for z in zones:
        nom = str(getattr(z, "nom", "") or "")
        zid = noms.get(nom) if isinstance(noms, Mapping) else None
        zid = zid or index.get(normaliser(nom))
        e = emprises.get(zid) if zid and isinstance(emprises, Mapping) else None
        try:
            lon0, lat0, lon1, lat1 = (float(v) for v in e["emprise"])
        except (TypeError, KeyError, ValueError):
            continue
        out.append(ZoneSituee(nom, int(getattr(z, "tuiles", 0) or 0), int(getattr(z, "objets", 0) or 0),
                              str(e.get("pays") or "france"), (lon0 + lon1) / 2, (lat0 + lat1) / 2))
    return out


def pays_presents(zones: Sequence[ZoneSituee]) -> List[str]:
    """Pays à dessiner, dans un ordre stable (France puis Irlande)."""
    presents = {z.pays for z in zones}
    return [p for p in ("france", "irlande") if p in presents] + sorted(presents - {"france", "irlande"})


def projection(
    anneaux: Sequence[Sequence[Sequence[float]]], pays: str, largeur: float, hauteur: float, marge: float = 4.0,
) -> Tuple[Callable[[float, float], Tuple[float, float]], float, float]:
    """Projection équirectangulaire corrigée par cos(latitude), calée sur les contours du
    pays dans un cadre ``largeur`` × ``hauteur``. Renvoie ``(f(lon, lat) → (x, y), largeur
    utile, hauteur utile)`` ; ``f`` centre le pays dans le cadre."""
    k = math.cos(math.radians(_LAT_REF.get(pays, 47.0)))
    pts = [(lon * k, lat) for a in anneaux for lon, lat in a]
    if not pts:
        return (lambda lon, lat: (largeur / 2, hauteur / 2)), 0.0, 0.0
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    s = min((largeur - 2 * marge) / max(x1 - x0, 1e-9), (hauteur - 2 * marge) / max(y1 - y0, 1e-9))
    ox = (largeur - (x1 - x0) * s) / 2
    oy = (hauteur - (y1 - y0) * s) / 2

    def f(lon: float, lat: float) -> Tuple[float, float]:
        return ox + (lon * k - x0) * s, oy + (y1 - lat) * s

    return f, (x1 - x0) * s, (y1 - y0) * s


def rayon(objets: int, maximum: int, cote: float) -> float:
    """Rayon d'un disque de zone : aire proportionnelle aux objets annotés, de 3 px à
    ~7 % du côté de la carte."""
    if maximum <= 0:
        return 3.0
    return 3.0 + 0.07 * cote * math.sqrt(max(0, objets) / maximum)


def zone_sous(disques: Sequence[Tuple[Any, float, float, float]], x: float, y: float, marge: float = 2.0) -> Any:
    """La zone dont le disque ``(zone, cx, cy, rayon)`` contient le point, ou ``None``.
    Les petits disques sont dessinés par-dessus les grands : ils passent d'abord, sans
    quoi une zone posée au centre d'une plus grande serait inatteignable."""
    for zone, cx, cy, r in sorted(disques, key=lambda t: t[3]):
        if (x - cx) ** 2 + (y - cy) ** 2 <= (r + marge) ** 2:
            return zone
    return None


def phrase_resume(zones: Sequence[Any]) -> str:
    """« 3 zones · 2 186 objets annotés » (sur toutes les zones de la fiche, situées ou non)."""
    n = len(zones)
    objets = sum(int(getattr(z, "objets", 0) or 0) for z in zones)
    texte = f"{n} zone{'s' if n > 1 else ''}"
    if objets:
        texte += " · " + f"{objets:,}".replace(",", " ") + " objets annotés"
    return texte


def anneaux_du_pays(data: Mapping[str, Any], pays: str) -> List[List[Tuple[float, float]]]:
    brut = (data.get("contours") or {}).get(pays) if isinstance(data.get("contours"), Mapping) else None
    out: List[List[Tuple[float, float]]] = []
    for a in brut or []:
        try:
            out.append([(float(p[0]), float(p[1])) for p in a])
        except (TypeError, ValueError, IndexError):
            continue
    return out

