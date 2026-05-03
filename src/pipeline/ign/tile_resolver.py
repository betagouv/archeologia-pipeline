"""
Résolution des dalles IGN LiDAR HD à partir d'un polygone de zone d'étude.

Intersecte le polygone utilisateur avec le quadrillage France
(data/quadrillage_france/TA_diff_pkk_lidarhd_classe.shp) pour déterminer
quelles dalles télécharger.

Le shapefile de quadrillage contient :
  - nom_pkk   : nom de la dalle (ex: LHD_FXX_0946_6744_PTS_C_...)
  - url_telech : URL de téléchargement de la dalle

Le CRS du quadrillage est Lambert 93 (EPSG:2154).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..types import CancelFn, LogFn

logger = logging.getLogger(__name__)

# ── Quadrillage path (relative to plugin root) ──
_QUADRILLAGE_RELPATH = Path("data") / "quadrillage_france" / "TA_diff_pkk_lidarhd_classe.shp"


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

    Charge le polygone utilisateur, le reprojette en Lambert 93 si besoin,
    puis effectue une intersection spatiale avec le quadrillage France.
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

    # ── Résolution du quadrillage ──
    if quadrillage_path is None:
        quadrillage_path = _get_plugin_root() / _QUADRILLAGE_RELPATH
    if not quadrillage_path.exists():
        raise FileNotFoundError(
            f"Quadrillage France introuvable : {quadrillage_path}\n"
            "Placez le shapefile dans data/quadrillage_france/."
        )

    # ── Chargement du polygone utilisateur ──
    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygone de zone d'étude introuvable : {polygon_path}")

    log(f"Chargement du polygone : {polygon_path.name}")
    user_ds = ogr.Open(str(polygon_path), 0)
    if user_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir le fichier : {polygon_path}")
    user_layer = user_ds.GetLayer(0)
    if user_layer is None:
        raise RuntimeError(f"Aucune couche trouvée dans : {polygon_path}")

    # ── Construction de la géométrie d'union du polygone utilisateur ──
    user_srs = user_layer.GetSpatialRef()
    union_geom = None
    for feat in user_layer:
        if cancel():
            log("Annulation demandée")
            return 0
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        if union_geom is None:
            union_geom = geom.Clone()
        else:
            union_geom = union_geom.Union(geom)
    user_layer.ResetReading()

    if union_geom is None:
        raise RuntimeError(f"Aucune géométrie valide dans : {polygon_path}")

    # ── Reprojection vers Lambert 93 (EPSG:2154) si nécessaire ──
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(2154)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if user_srs is not None:
        user_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not user_srs.IsSame(target_srs):
            log("Reprojection du polygone vers Lambert 93 (EPSG:2154)")
            transform = osr.CoordinateTransformation(user_srs, target_srs)
            union_geom.Transform(transform)
    else:
        log("⚠️ CRS du polygone non défini — on suppose Lambert 93 (EPSG:2154)")

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
