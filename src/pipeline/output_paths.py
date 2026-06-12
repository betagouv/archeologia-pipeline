"""
Centralise toutes les résolutions de chemins du dossier de sortie du pipeline.

Nouvelle arborescence (v2) :
    <output_dir>/
    ├── indices/            # ex-results/ – rasters finaux (MNT, SVF, LD…)
    │   ├── MNT/tif/                    # MNT/DENSITE : pas de paramètres → code brut
    │   ├── SVF_R10_D16_V1_N0/tif/      # nom = code indice + suffixe de paramètres RVT
    │   └── LD_A15_Rmin10_Rmax20_H1p7_V1/
    │       ├── tif/
    │       └── png/        # images PNG pour l'inférence
    ├── detections/         # résultats CV, organisés par ENTITÉ (vocabulaire utilisateur)
    │   ├── detections_validation.qgs   # projet consolidé (point d'entrée), couches groupées par entité
    │   ├── <entity_slug>/              # ex. parcellaire/, chemins_creux/…
    │   │   └── <entity_slug>.gpkg      # détections de l'entité
    │   └── _technique/                 # échafaudage non-livrable (traçabilité/debug)
    │       └── <model_slug>/
    │           ├── raw_detections/     # JSON/TXT inférence
    │           └── annotated_images/   # images annotées (si option activée)
    ├── sources/            # données d'entrée (dalles LAZ, urls…)
    │   ├── dalles/
    │   └── dalles_urls.txt
    ├── intermediaires/     # fichiers temporaires
    └── metadata.json
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict


# ------------------------------------------------------------------ #
#  Constantes – noms de dossiers racine                                #
# ------------------------------------------------------------------ #

DIR_INDICES = "indices"
DIR_DETECTIONS = "detections"
DIR_TECHNIQUE = "_technique"   # sous-dossier de detections/ : échafaudage non-livrable
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
#  Détections – arborescence entité-centrée (v2)                        #
# ------------------------------------------------------------------ #

def detection_entity_dir(output_dir: Path, entity_slug: str) -> Path:
    """Dossier livrable d'une entité : ``<output>/detections/<entity_slug>/``.

    1er niveau **entité-centré** (vocabulaire utilisateur). Contient le(s)
    GeoPackage(s) de l'entité (ex. ``parcellaire/parcellaire.gpkg``).
    """
    return detections_dir(output_dir) / entity_slug


def detection_technique_dir(output_dir: Path, model_slug: str) -> Path:
    """Échafaudage technique d'un modèle : ``<output>/detections/_technique/<model_slug>/``.

    Regroupe le **non-livrable** (dumps d'inférence, images annotées) hors de la
    vue entité-centrée, sans le perdre (traçabilité / debug).
    """
    return detections_dir(output_dir) / DIR_TECHNIQUE / model_slug


def detection_technique_raw_dir(output_dir: Path, model_slug: str) -> Path:
    """JSON/TXT bruts d'inférence : ``…/_technique/<model_slug>/raw_detections/``."""
    return detection_technique_dir(output_dir, model_slug) / "raw_detections"


def detection_technique_annotated_dir(output_dir: Path, model_slug: str) -> Path:
    """Images annotées : ``…/_technique/<model_slug>/annotated_images/``."""
    return detection_technique_dir(output_dir, model_slug) / "annotated_images"


def build_entity_class_targets(output_dir: Path, entities: Any):
    """Routage ``classe → [(gpkg, nom_de_couche)]`` pour la sortie entité-centrée.

    ``entities`` : la liste posée par l'orchestrateur dans le run
    (``[{id, label, slug, classes, is_derived, layer_names}, …]``). ``layer_names``
    (``classe → nom_de_couche``) porte le renommage décidé par l'orchestrateur
    (libellés ``output_label`` / ``source_label`` du model_card, ou repli) ;
    absent → la couche garde le nom de sa classe.

    **Déduplication (décision C)** : si une classe est revendiquée à la fois par
    une couche **canonique** (``nom_de_couche == classe``, entité de base) et par
    une copie **renommée** (constituant d'une entité dérivée, via ``layer_names``),
    on ne garde que la copie renommée — les éléments individuels n'apparaissent
    qu'une fois, dans le groupe de la zone. Cas concret : ``cratere`` cochée à la
    fois comme « Cratères » (base) et comme constituant de « Regroupement de
    cratères » → une seule couche « Cratères » dans le dossier du regroupement.
    Sinon (pas de conflit), la couche canonique reste en tête.
    """
    targets: Dict[str, list] = {}
    for ent in (entities or []):
        ent = ent or {}
        slug = str(ent.get("slug") or ent.get("id") or "").strip()
        # Défense en profondeur (AUDIT PARSE-05) : le repli ``id`` n'est pas
        # garanti slugifié — il entre dans un Path, on neutralise tout
        # caractère de chemin/interdit Windows avant usage.
        slug = re.sub(r"[^a-z0-9_-]+", "_", slug.lower()).strip("_")
        if not slug:
            continue
        gpkg = str(detection_entity_dir(output_dir, slug) / f"{slug}.gpkg")
        layer_names = ent.get("layer_names") or {}
        for cls in (ent.get("classes") or []):
            cls = str(cls)
            layer = str(layer_names.get(cls, cls))
            targets.setdefault(cls, []).append((gpkg, layer))
    deduped: Dict[str, list] = {}
    for cls, lst in targets.items():
        has_canonical = any(layer == cls for _g, layer in lst)
        has_renamed = any(layer != cls for _g, layer in lst)
        if has_canonical and has_renamed:
            # conflit base/dérivée → on ne garde que la couche du groupe (renommée)
            deduped[cls] = [t for t in lst if t[1] != cls]
        else:
            # couche canonique (layer == classe) en tête : chemin d'écriture principal
            deduped[cls] = sorted(lst, key=lambda t: 0 if t[1] == cls else 1)
    return deduped


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
    """Construit ``indices/<CODE><suffixe_params>/tif`` (ex: ``LD_A15_Rmin10_.../tif``).

    Le suffixe de paramètres (issu de ``rvt_params``) permet à deux exécutions de
    paramètres différents de viser des dossiers distincts plutôt que de s'écraser.
    Il doit être identique à celui utilisé à la création (``copy_final_products_to_results``),
    d'où l'invariant : passer le *même* ``rvt_params`` des deux côtés.

    ``output_structure`` est conservé dans la signature pour compatibilité mais
    n'est plus utilisé.
    """
    # Import différé : importer ``ign.products.rvt_naming`` au top-level
    # déclencherait ``ign/products/__init__.py`` → QGIS, alors que ce module doit
    # rester importable en standalone (tests). ``rvt_naming`` n'importe jamais
    # ``output_paths`` → pas de cycle.
    from .ign.products.rvt_naming import get_rvt_folder_name

    folder_name = get_rvt_folder_name(target_rvt, rvt_params or {})
    return indice_tif_dir(output_dir, folder_name)
