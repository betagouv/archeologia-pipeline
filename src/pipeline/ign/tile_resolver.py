"""
Résolution des dalles IGN LiDAR HD à partir d'un polygone de zone d'étude.

Intersecte le polygone utilisateur avec le quadrillage France
(``data/quadrillage_france/`` — GeoPackage slim à R-tree de préférence, sinon
shapefile legacy ; cf. :func:`pipeline.ign.quadrillage_paths.resolve_quadrillage_path`)
pour déterminer quelles dalles télécharger.

Le quadrillage contient :
  - nom_pkk   : nom de la dalle (ex: LHD_FXX_0946_6744_PTS_C_...)
  - url_telech : URL de téléchargement de la dalle

Le CRS du quadrillage est Lambert 93 (EPSG:2154).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..types import CancelFn, LogFn
from .quadrillage_paths import resolve_quadrillage_path

logger = logging.getLogger(__name__)


def _default_log(_: str) -> None:
    return


def _default_cancel() -> bool:
    return False


def _get_plugin_root() -> Path:
    """Retourne la racine du plugin (3 niveaux au-dessus de ce fichier)."""
    return Path(__file__).resolve().parents[3]


def resolve_tiles_from_polygon(
    polygon_path: Path,
    output_file: Path,
    *,
    quadrillage_path: Optional[Path] = None,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> int:
    """
    Résout les dalles IGN LiDAR HD intersectant un polygone.

    Charge **toutes les couches** du fichier (un GeoPackage peut en contenir
    plusieurs, p. ex. exporté depuis un groupe de couches QGIS), unionne leurs
    géométries en les reprojetant en Lambert 93 si besoin, puis effectue une
    intersection spatiale avec le quadrillage France.
    Écrit le résultat dans ``output_file`` au format ``filename,url`` par ligne,
    compatible avec ``parse_ign_input_file()`` / ``download_ign_dalles()``.

    Args:
        polygon_path: Chemin vers le shapefile/GeoJSON de la zone d'étude.
        output_file: Fichier de sortie (dalles_urls.txt).
        quadrillage_path: Chemin vers le shapefile du quadrillage France.
            Si None, utilise le chemin par défaut dans data/.
        log: Fonction de logging.
        cancel: Fonction d'annulation.

    Returns:
        Nombre de dalles trouvées.

    Raises:
        FileNotFoundError: Si le polygone ou le quadrillage n'existe pas.
        RuntimeError: Si GDAL/OGR n'est pas disponible ou si l'intersection échoue.
    """
    try:
        from osgeo import ogr, osr
    except ImportError:
        raise RuntimeError(
            "GDAL/OGR n'est pas disponible. "
            "Installez GDAL ou exécutez le plugin dans QGIS."
        )

    ogr.UseExceptions()

    # ── Résolution du quadrillage (shapefile + index .qix ; .gpkg si présent) ──
    if quadrillage_path is None:
        quadrillage_path = resolve_quadrillage_path(_get_plugin_root())
    if not quadrillage_path.exists():
        raise FileNotFoundError(
            f"Quadrillage France introuvable : {quadrillage_path}\n"
            "Placez le quadrillage IGN (TA_diff_pkk_lidarhd_classe.shp + son index "
            ".qix) dans data/quadrillage_france/."
        )

    # ── Chargement du polygone utilisateur ──
    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygone de zone d'étude introuvable : {polygon_path}")

    log(f"Chargement de la zone d'étude : {polygon_path.name}")
    user_ds = ogr.Open(str(polygon_path), 0)
    if user_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir le fichier : {polygon_path}")
    n_layers = user_ds.GetLayerCount()
    if n_layers == 0:
        raise RuntimeError(f"Aucune couche trouvée dans : {polygon_path}")
    if n_layers > 1:
        log(f"{n_layers} couches dans le fichier — union de toutes les entités")

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(2154)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # ── Union des géométries de TOUTES les couches, reprojetées en Lambert 93 ──
    # La reprojection est faite par couche (et non sur l'union finale) : un
    # GeoPackage issu d'un groupe de couches QGIS peut mélanger les CRS.
    union_geom = None
    for i in range(n_layers):
        user_layer = user_ds.GetLayer(i)
        if user_layer is None:
            continue

        user_srs = user_layer.GetSpatialRef()
        transform = None
        if user_srs is not None:
            user_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not user_srs.IsSame(target_srs):
                log(f"Reprojection de « {user_layer.GetName()} » vers Lambert 93 (EPSG:2154)")
                transform = osr.CoordinateTransformation(user_srs, target_srs)
        else:
            log(f"⚠️ CRS de « {user_layer.GetName()} » non défini — on suppose Lambert 93")

        for feat in user_layer:
            if cancel():
                log("Annulation demandée")
                return 0
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            geom = geom.Clone()
            if transform is not None:
                geom.Transform(transform)
            union_geom = geom if union_geom is None else union_geom.Union(geom)
        user_layer.ResetReading()

    if union_geom is None:
        raise RuntimeError(f"Aucune géométrie valide dans : {polygon_path}")

    # ── Ouverture du quadrillage et filtre spatial ──
    log(f"Chargement du quadrillage : {quadrillage_path.name}")
    grid_ds = ogr.Open(str(quadrillage_path), 0)
    if grid_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir le quadrillage : {quadrillage_path}")
    grid_layer = grid_ds.GetLayer(0)
    if grid_layer is None:
        raise RuntimeError(f"Aucune couche dans le quadrillage : {quadrillage_path}")

    grid_layer.SetSpatialFilter(union_geom)

    # ── Extraction des dalles intersectantes ──
    tiles: List[Tuple[str, str]] = []
    skipped = 0

    for feat in grid_layer:
        if cancel():
            log("Annulation demandée")
            return 0

        nom_pkk = (feat.GetField("nom_pkk") or "").strip()
        url_telech = (feat.GetField("url_telech") or "").strip()

        if not url_telech:
            skipped += 1
            continue

        # Déduire le nom de fichier depuis l'URL ou utiliser nom_pkk
        filename = nom_pkk
        if url_telech.startswith(("http://", "https://")):
            try:
                import urllib.parse
                url_path = urllib.parse.urlparse(url_telech).path
                url_filename = Path(url_path).name
                if url_filename:
                    filename = url_filename
            except Exception:
                pass

        if not filename:
            skipped += 1
            continue

        tiles.append((filename, url_telech))

    grid_layer.ResetReading()

    if skipped > 0:
        log(f"⚠️ {skipped} dalle(s) ignorée(s) (URL manquante)")

    if not tiles:
        log("⚠️ Aucune dalle trouvée pour la zone sélectionnée")
        # Fermer les datasources
        user_ds = None
        grid_ds = None
        return 0

    # ── Écriture du fichier de sortie ──
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for filename, url in tiles:
            f.write(f"{filename},{url}\n")

    log(f"✅ {len(tiles)} dalle(s) identifiée(s) pour la zone d'étude")

    # Fermer les datasources
    user_ds = None
    grid_ds = None

    return len(tiles)
