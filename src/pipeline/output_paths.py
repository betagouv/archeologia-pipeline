"""
Centralise toutes les résolutions de chemins du dossier de sortie du pipeline.

Nouvelle arborescence (v2) :
    <output_dir>/
    ├── indices/            # ex-results/ – rasters finaux (MNT, SVF, LD…)
    │   ├── MNT/tif/
    │   ├── SVF/tif/
    │   └── LD/
    │       ├── tif/
    │       └── png/        # images PNG pour l'inférence
    ├── detections/         # résultats par modèle CV
    │   └── <model_slug>/
    │       ├── shapefiles/
    │       ├── par_dalle/          # JSON/TXT bruts par dalle
    │       ├── raw_detections/     # JSON/TXT inférence (toujours présent)
    │       └── annotated_images/   # images annotées (si option activée)
    ├── sources/            # données d'entrée (dalles LAZ, urls…)
    │   ├── dalles/
    │   └── dalles_urls.txt
    ├── intermediaires/     # fichiers temporaires
    └── metadata.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# ------------------------------------------------------------------ #
#  Constantes – noms de dossiers racine                                #
# ------------------------------------------------------------------ #

DIR_INDICES = "indices"
DIR_DETECTIONS = "detections"
DIR_SOURCES = "sources"
DIR_INTERMEDIAIRES = "intermediaires"


# ------------------------------------------------------------------ #
#  Indices (rasters)                                                    #
# ------------------------------------------------------------------ #

def indices_dir(output_dir: Path) -> Path:
    """Racine des indices raster : ``<output>/indices/``."""
    return output_dir / DIR_INDICES


def indice_tif_dir(output_dir: Path, product_name: str) -> Path:
    """Dossier TIF pour un indice : ``<output>/indices/<PRODUCT>/tif/``."""
    return indices_dir(output_dir) / product_name / "tif"


def indice_png_dir(output_dir: Path, product_name: str) -> Path:
    """Dossier PNG pour un indice (images d'inférence) : ``<output>/indices/<PRODUCT>/png/``."""
    return indices_dir(output_dir) / product_name / "png"


def indice_jpg_dir(output_dir: Path, product_name: str) -> Path:
    """Alias rétrocompat → indice_png_dir."""
    return indice_png_dir(output_dir, product_name)


def indice_base_dir(output_dir: Path, product_name: str) -> Path:
    """Dossier de base d'un indice : ``<output>/indices/<PRODUCT>/``."""
    return indices_dir(output_dir) / product_name


# ------------------------------------------------------------------ #
#  Détections (CV)                                                      #
# ------------------------------------------------------------------ #

def detections_dir(output_dir: Path) -> Path:
    """Racine des détections CV : ``<output>/detections/``."""
    return output_dir / DIR_DETECTIONS


def detection_model_dir(output_dir: Path, model_slug: str) -> Path:
    """Dossier d'un modèle de détection : ``<output>/detections/<model>/``."""
    return detections_dir(output_dir) / model_slug


def detection_shapefiles_dir(output_dir: Path, model_slug: str) -> Path:
    """Shapefiles d'un modèle : ``<output>/detections/<model>/shapefiles/``."""
    return detection_model_dir(output_dir, model_slug) / "shapefiles"


def detection_par_dalle_dir(output_dir: Path, model_slug: str) -> Path:
    """Labels bruts par dalle : ``<output>/detections/<model>/par_dalle/``."""
    return detection_model_dir(output_dir, model_slug) / "par_dalle"


def detection_raw_dir(output_dir: Path, model_slug: str) -> Path:
    """JSON/TXT bruts d'inférence : ``<output>/detections/<model>/raw_detections/``."""
    return detection_model_dir(output_dir, model_slug) / "raw_detections"


def detection_annotated_dir(output_dir: Path, model_slug: str) -> Path:
    """Images annotées : ``<output>/detections/<model>/annotated_images/``."""
    return detection_model_dir(output_dir, model_slug) / "annotated_images"


def detection_jpg_dir(output_dir: Path, model_slug: str) -> Path:
    """Alias rétrocompat → detection_raw_dir."""
    return detection_raw_dir(output_dir, model_slug)


# ------------------------------------------------------------------ #
#  Sources                                                              #
# ------------------------------------------------------------------ #

def sources_dir(output_dir: Path) -> Path:
    """Racine des données source : ``<output>/sources/``."""
    return output_dir / DIR_SOURCES


def dalles_dir(output_dir: Path) -> Path:
    """Dossier des dalles LAZ : ``<output>/sources/dalles/``."""
    return sources_dir(output_dir) / "dalles"


# ------------------------------------------------------------------ #
#  Intermédiaires                                                       #
# ------------------------------------------------------------------ #

def intermediaires_dir(output_dir: Path) -> Path:
    """Racine des fichiers intermédiaires : ``<output>/intermediaires/``."""
    return output_dir / DIR_INTERMEDIAIRES


# ------------------------------------------------------------------ #
#  Rétrocompatibilité : résolution indice par nom de produit             #
# ------------------------------------------------------------------ #

def resolve_rvt_tif_dir(
    output_dir: Path,
    target_rvt: str,
    output_structure: Dict[str, Any] | None = None,
    rvt_params: Dict[str, Any] | None = None,
) -> Path:
    """Construit ``indices/<PRODUCT>/tif`` — remplace l'ancien ``results/RVT/<type><suffix>/tif``.

    Les paramètres ``output_structure`` et ``rvt_params`` sont conservés dans la
    signature pour compatibilité mais ne sont plus utilisés pour construire le chemin.
    Le nom du dossier est désormais toujours le nom court du produit (ex: ``LD``, ``SVF``).
    """
    return indice_tif_dir(output_dir, target_rvt)
