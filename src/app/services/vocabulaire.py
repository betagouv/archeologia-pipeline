"""Lexique de l'interface — un seul mot par objet, repris par les écrans, le
journal, le rapport et les noms de couches (interface rationalisée, 2026-10-24).

Pur : aucun import Qt ni QGIS. Les écrans n'écrivent plus ces mots en dur ;
ils les prennent ici, ce qui garde le même vocabulaire partout.
"""
from __future__ import annotations

import re
from typing import Dict

from .fiabilite import LABELS_FR
from .indices_model import all_products as _all_products
from .indices_model import product as _product

# 2026-10-25 : l'étape 2 « Visualisation du relief » calcule des PRODUITS — le MNT, les produits de
# qualité de la donnée (densité, couverture) et les INDICES DE VISUALISATION (SVF, LD, M-HS…).
VISUALISATION = "Visualisation du relief"
IMAGE = "produit"
IMAGES = "produits"
INDICE = "indice de visualisation"
INDICES = "indices de visualisation"
STRUCTURE = "structure"
STRUCTURES = "structures"
TRAITEMENT = "traitement"
DALLE = "dalle"

#: Libellé français de chaque contrôle de l'environnement (étape 4), par le
#: ``name`` que ``pipeline/preflight.py`` donne à son ``CheckResult``.
ENVIRONNEMENT: Dict[str, str] = {
    "pdal (CLI)": "Lecteur de nuages de points (pdal)",
    "QGIS processing": "Boîte à outils de QGIS",
    "gdalwarp": "Reprojection des rasters (gdalwarp)",
    "gdal_translate": "Conversion des rasters (gdal_translate)",
    "gdaladdo": "Aperçus des rasters (gdaladdo)",
    "Moteur d'indices intégré": "Calcul des indices de visualisation (moteur intégré)",
    "Algorithmes RVT (Processing)": "Indices de visualisation par l'extension rvt-qgis",
    "cv_runner_onnx (external)": "Moteur de détection",
    "python:onnxruntime": "Bibliothèque de détection (onnxruntime)",
    "python:PIL": "Bibliothèque d'images (Pillow)",
    "python:geopandas": "Bibliothèque de géométries (geopandas)",
    "CRS des rasters": "Système de coordonnées des rasters",
    "CRS du dossier de sortie": "Système de coordonnées du dossier de sortie",
    "Dalles dégénérées": "Dalles trop petites",
    "Dossier de sortie": "Dossier de sortie",
    # Les entrées de l'étape 1, nommées par ``_check_input_path`` (recette 65.5).
    "Zone d'étude (vecteur)": "Zone d'étude (fichier vecteur)",
    "Fichier liste URLs IGN": "Liste des dalles IGN (fichier texte)",
    "Dossier LAZ locaux": "Dossier des nuages de points (LAZ)",
    "Dossier MNT existants": "Dossier des MNT existants",
    "Dossier RVT existants": "Dossier des indices existants",
}


def titre_image(key: str) -> str:
    """``"LD · Structures en relief"`` : le sigle puis ce que l'image montre.
    Une clé inconnue est rendue telle quelle."""
    try:
        p = _product(key)
    except KeyError:
        return key
    return f"{p.tag} · {p.metier}" if p.metier else p.tag


def titre_dossier_image(nom_dossier: str) -> str:
    """Nom de la mosaïque dans QGIS depuis son dossier d'image (N-03, 2026-10-25) :
    ``SVF_R10_D16_V1_N0`` → « SVF · Creux & dépressions · R10_D16_V1_N0 », le titre de
    l'étape 2 puis les réglages du dossier. Un dossier inconnu (``RVT``) reste tel
    quel ; vide → « index »."""
    nom = str(nom_dossier or "")
    if not nom:
        return "index"
    for p in sorted(_all_products(), key=lambda p: -len(p.key)):   # M_HS avant HS
        if nom == p.key:
            return titre_image(p.key)
        if nom.startswith(p.key + "_"):
            return f"{titre_image(p.key)} · {nom[len(p.key) + 1:]}"
    return nom


def nombre(x: float, decimales: int = 2) -> str:
    """Nombre à la française : virgule décimale, espace fine insécable aux milliers."""
    texte = f"{x:,.{decimales}f}"
    return texte.replace(",", " ").replace(".", ",")


def niveau(cat: str) -> str:
    """Niveau de fiabilité en minuscules, au masculin singulier (« très probable »)."""
    return LABELS_FR.get(cat, cat).lower() if cat in LABELS_FR else cat


def pluriel(n: int, singulier: str, pluriel_: str = "") -> str:
    """``"1 dalle"``, ``"12 dalles"``, ``"2 journaux"``."""
    mot = singulier if n in (0, 1) else (pluriel_ or singulier + "s")
    return f"{n} {mot}"


_MANQUANT = ("not found", "introuvable", "manquant", "absent", "no module", "modulenotfound", "error")
_CHEMIN = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")


def resume_detail(details: str) -> str:
    """Le détail d'un contrôle de l'environnement en un mot pour la colonne de l'étape 4 ;
    le texte complet (chemin, erreur) reste en infobulle. Un chemin → « trouvé » (avec
    son complément « · 120 Go libres » s'il y en a un) ; une absence ou une erreur →
    « manquant » ; un contrôle sauté → « non requis » ; le reste (une version, un
    système de coordonnées) est rendu tel quel."""
    texte = (details or "").strip()
    bas = texte.lower()
    if not texte:
        return ""
    if bas.startswith("skipped"):
        return "non requis"
    if any(m in bas for m in _MANQUANT):
        return "manquant"
    if bas.startswith("import ok"):
        version = re.search(r"version=([^,\s)]+)", texte)
        return f"trouvé · version {version.group(1)}" if version else "trouvé"
    if _CHEMIN.match(texte):
        complement = texte.split(" · ", 1)[1] if " · " in texte else ""
        return f"trouvé · {complement}" if complement else "trouvé"
    return texte


def libelle_environnement(name: str) -> str:
    """Libellé français d'un contrôle de l'environnement ; une bibliothèque
    Python inconnue reste nommée ; tout autre nom inconnu est rendu tel quel."""
    if name in ENVIRONNEMENT:
        return ENVIRONNEMENT[name]
    if name.startswith("python:"):
        return f"Bibliothèque Python {name[len('python:'):]}"
    return name
