"""Abstraction de placement spatial unique pour toutes les entités du pipeline.

``TileSpec`` capture, pour une entité traitée (dalle LAZ, raster MNT/densité/RVT),
la **vérité spatiale lue depuis les métadonnées géospatiales** : emprise réelle,
taille de pixel, dimensions, CRS, NoData, géotransformée — plus un identifiant
unique stable (``uid``). C'est ce qui remplace la déduction de coordonnées à partir
du nom de fichier (fragile). Le nom de fichier ne sert plus qu'à dériver un préfixe
*cosmétique* (``LHD_FXX_{x}_{y}``) et l'``uid``.

Ce module doit rester importable **sans QGIS** (testable hors plugin) : tous les
imports lourds (``rasterio``, ``osgeo``) sont différés à l'intérieur des fonctions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

# Ordre GDAL : (x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height)
GeoTransform = Tuple[float, float, float, float, float, float]
Bounds = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def _sanitize_token(stem: str) -> str:
    """Réduit un stem à un token alphanumérique sûr pour GDAL/QGIS et les noms de fichiers."""
    safe = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    return safe or "EXT"


def make_uid(source_path, *, max_len: int = 48) -> str:
    """Identifiant unique lisible dérivé du nom de fichier source.

    - Retire un éventuel suffixe ``_MNT`` (pour ne pas produire ``*_MNT_MNT``).
    - Assainit les caractères non alphanumériques.
    - Plafonne la longueur (compatibilité Windows MAX_PATH une fois combiné en aval).

    L'unicité **au sein d'un dossier d'entrée** est garantie par l'unicité des noms
    de fichiers. Pour les rares collisions (stems différents qui s'assainissent
    identiquement, ou deux dossiers fusionnés), utiliser :func:`disambiguate`.
    """
    stem = Path(source_path).stem
    if stem.lower().endswith("_mnt"):
        stem = stem[:-4]
    safe = _sanitize_token(stem)
    if len(safe) > max_len:
        safe = safe[:max_len].strip("_") or "EXT"
    return safe


def disambiguate(uid: str, seen: Set[str]) -> str:
    """Renvoie un ``uid`` garanti absent de ``seen`` (ajoute ``_2``, ``_3``… si besoin).

    Filet anti-collision : ``seen`` est muté pour mémoriser la valeur retenue.
    """
    if uid not in seen:
        seen.add(uid)
        return uid
    n = 2
    while f"{uid}_{n}" in seen:
        n += 1
    resolved = f"{uid}_{n}"
    seen.add(resolved)
    return resolved


def crs_is_projected(crs: Optional[str]) -> Optional[bool]:
    """Indique si un CRS (authid ou WKT) est projeté.

    Renvoie ``True`` (projeté/mètres), ``False`` (géographique/degrés), ou ``None``
    si le CRS est absent, illisible, ou local/engineering (``LOCAL_CS``).
    Utilisé par l'ingest planning pour exiger un CRS projeté (décision 2).
    """
    if not crs:
        return None
    try:
        from rasterio.crs import CRS  # type: ignore

        c = CRS.from_user_input(crs)
        if not (c.is_geographic or c.is_projected):
            return None  # local / engineering
        return bool(c.is_projected)
    except Exception:
        return None


@dataclass(frozen=True)
class TileSpec:
    """Placement spatial d'une entité, lu depuis les métadonnées (pas le nom)."""

    source_path: Path
    bounds: Bounds                 # xmin, ymin, xmax, ymax (monde, dans le CRS de l'entité)
    pixel_size_x: float
    pixel_size_y: float            # signé (négatif pour un raster « nord en haut »)
    width_px: int
    height_px: int
    crs: Optional[str]             # authid ("EPSG:2154") ou WKT ; None si absent/local
    geotransform: GeoTransform     # ordre GDAL
    uid: str                       # identifiant unique stable, sûr pour noms de fichiers
    declared_crs: Optional[str] = None  # CRS déclaré par l'utilisateur (fallback si crs absent)
    nodata: Optional[float] = None      # valeur NoData lue dans les métadonnées (jamais devinée)

    @property
    def effective_crs(self) -> Optional[str]:
        """CRS réel s'il existe, sinon le CRS déclaré par l'utilisateur."""
        return self.crs or self.declared_crs

    @property
    def width_m(self) -> float:
        return abs(self.bounds[2] - self.bounds[0])

    @property
    def height_m(self) -> float:
        return abs(self.bounds[3] - self.bounds[1])

    def cosmetic_xy(self) -> Tuple[int, int]:
        """Indice km du coin nord-ouest — **cosmétique** (préfixe ``LHD_FXX_{x}_{y}``).

        Ne sert jamais au placement : c'est seulement un libellé lisible. Réutilise
        la logique de snap km de :mod:`pipeline.coords`.
        """
        from .coords import _infer_xy_from_bounds  # type: ignore[attr-defined]

        xmin, _ymin, _xmax, ymax = self.bounds
        xy = _infer_xy_from_bounds(xmin, ymax)
        if xy is not None:
            return int(xy.x_km), int(xy.y_km)
        return int(math.floor(xmin / 1000.0)), int(math.floor(ymax / 1000.0))

    @classmethod
    def from_values(
        cls,
        *,
        source_path,
        bounds: Bounds,
        pixel_size_x: float,
        pixel_size_y: float,
        width_px: int,
        height_px: int,
        crs: Optional[str] = None,
        geotransform: Optional[GeoTransform] = None,
        declared_crs: Optional[str] = None,
        nodata: Optional[float] = None,
        uid: Optional[str] = None,
    ) -> "TileSpec":
        """Constructeur pur (sans I/O) — testable directement."""
        source_path = Path(source_path)
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
        if geotransform is None:
            geotransform = (xmin, float(pixel_size_x), 0.0, ymax, 0.0, float(pixel_size_y))
        if uid is None:
            uid = make_uid(source_path)
        return cls(
            source_path=source_path,
            bounds=(xmin, ymin, xmax, ymax),
            pixel_size_x=float(pixel_size_x),
            pixel_size_y=float(pixel_size_y),
            width_px=int(width_px),
            height_px=int(height_px),
            crs=crs,
            geotransform=tuple(float(v) for v in geotransform),  # type: ignore[arg-type]
            uid=uid,
            declared_crs=declared_crs,
            nodata=None if nodata is None else float(nodata),
        )

    @classmethod
    def from_raster(cls, path, *, declared_crs: Optional[str] = None) -> Optional["TileSpec"]:
        """Construit un ``TileSpec`` en lisant les métadonnées d'un raster.

        Essaie ``rasterio`` puis les bindings ``osgeo.gdal``. Renvoie ``None`` si
        aucun backend ne peut lire le fichier (illisible/corrompu). Le CRS est lu
        tel quel (authid si dispo, sinon WKT) ; un CRS local/engineering est traité
        comme **absent**.
        """
        path = Path(path)

        # 1) rasterio
        try:
            import rasterio  # type: ignore

            with rasterio.open(str(path)) as ds:
                t = ds.transform
                gt: GeoTransform = (t.c, t.a, t.b, t.f, t.d, t.e)
                b = ds.bounds
                return cls.from_values(
                    source_path=path,
                    bounds=(float(b.left), float(b.bottom), float(b.right), float(b.top)),
                    pixel_size_x=float(t.a),
                    pixel_size_y=float(t.e),
                    width_px=int(ds.width),
                    height_px=int(ds.height),
                    crs=_crs_from_rasterio(ds.crs),
                    geotransform=gt,
                    declared_crs=declared_crs,
                    nodata=(None if ds.nodata is None else float(ds.nodata)),
                )
        except Exception:
            pass

        # 2) GDAL bindings
        try:
            from osgeo import gdal  # type: ignore

            ds = gdal.Open(str(path))
            if ds is not None:
                gt_gdal = ds.GetGeoTransform()
                width = int(ds.RasterXSize)
                height = int(ds.RasterYSize)
                crs = _crs_from_gdal(ds)
                nodata = None
                try:
                    band = ds.GetRasterBand(1)
                    nd = band.GetNoDataValue() if band is not None else None
                    nodata = None if nd is None else float(nd)
                except Exception:
                    nodata = None
                ds = None
                if gt_gdal:
                    x_origin, px_w, _rr, y_origin, _cr, px_h = (float(v) for v in gt_gdal)
                    xmax = x_origin + width * px_w
                    y_other = y_origin + height * px_h
                    ymin, ymax = sorted((y_origin, y_other))
                    return cls.from_values(
                        source_path=path,
                        bounds=(x_origin, ymin, xmax, ymax),
                        pixel_size_x=px_w,
                        pixel_size_y=px_h,
                        width_px=width,
                        height_px=height,
                        crs=crs,
                        geotransform=tuple(float(v) for v in gt_gdal),  # type: ignore[arg-type]
                        declared_crs=declared_crs,
                        nodata=nodata,
                    )
        except Exception:
            pass

        return None


def _crs_from_rasterio(crs_obj) -> Optional[str]:
    """Convertit un objet CRS rasterio en authid ('EPSG:2154') ou WKT, ou None.

    Un CRS local/engineering (ni géographique ni projeté, ex. ``LOCAL_CS``) est
    traité comme **absent** (None) — sinon il masquerait le fallback ``declared_crs``.
    """
    if crs_obj is None:
        return None
    try:
        if not (crs_obj.is_geographic or crs_obj.is_projected):
            return None
    except Exception:
        pass
    try:
        auth = crs_obj.to_authority()  # ('EPSG', '2154') ou None
        if auth:
            return f"{auth[0]}:{auth[1]}"
    except Exception:
        pass
    try:
        return crs_obj.to_wkt() or None
    except Exception:
        return None


def _crs_from_gdal(ds) -> Optional[str]:
    """Extrait le CRS d'un dataset GDAL en authid ou WKT, ou None s'il est absent/local."""
    try:
        srs = ds.GetSpatialRef()
    except Exception:
        srs = None
    if srs is None:
        return None
    try:
        if srs.IsLocal():
            return None
    except Exception:
        pass
    try:
        name = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if name and code:
            return f"{name}:{code}"
        return srs.ExportToWkt() or None
    except Exception:
        return None


def assign_crs_if_missing(path, fallback_authid: str = "EPSG:2154") -> Optional[str]:
    """Affecte ``fallback_authid`` au raster s'il n'a **pas de CRS exploitable**.

    *Assignation* (jamais de reprojection) : les coordonnées sont supposées déjà
    exprimées dans ce CRS — on n'écrit que l'étiquette manquante. Corrige les MNT
    que ``pdal:exportraster*`` émet parfois en CRS local « unnamed » (dalles
    LiDAR HD à CRS compound) ; sans ça ce CRS se propage jusqu'aux couches QGIS
    et casse la transformation au zoom de finalisation.

    Renvoie l'authid affecté, ou ``None`` si le raster avait déjà un CRS réel,
    s'il est illisible, ou si aucun backend n'est disponible. Essaie rasterio
    puis ``osgeo.gdal`` (même stratégie que :meth:`TileSpec.from_raster`).
    """
    path = Path(path)
    spec = TileSpec.from_raster(path)
    if spec is None or spec.crs is not None:
        return None  # illisible/inexistant, ou CRS réel déjà présent (local → crs=None)

    try:  # rasterio (présent en test ; parfois dans QGIS)
        import rasterio
        from rasterio.crs import CRS

        with rasterio.open(str(path), "r+") as ds:
            ds.crs = CRS.from_user_input(fallback_authid)
        return fallback_authid
    except Exception:
        pass

    try:  # osgeo.gdal (toujours présent côté QGIS)
        from osgeo import gdal, osr

        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            return None
        sr = osr.SpatialReference()
        sr.SetFromUserInput(fallback_authid)
        ds.SetProjection(sr.ExportToWkt())
        ds = None  # flush sur disque
        return fallback_authid
    except Exception:
        return None
