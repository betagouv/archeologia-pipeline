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

# Seuils de détection des dalles « dégénérées » (placeholder 1×1 px / posées à
# l'origine du CRS). Incluses dans un ``gdalbuildvrt``, elles étirent l'emprise de
# la mosaïque jusqu'à (0,0) → couches QGIS quasi vides, données réelles invisibles.
DEGENERATE_MIN_PX = 2          # une vraie dalle fait des centaines/milliers de px
DEGENERATE_ORIGIN_RADIUS_M = 10.0  # coin NO à < 10 m de (0,0) = placeholder (la donnée
#                                    réelle est à des centaines de km de l'origine)


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

    Double backend rasterio → ``osgeo.osr`` (même stratégie que :meth:`TileSpec.from_raster`
    et :func:`assign_crs_if_missing`) : rasterio est souvent absent dans QGIS alors
    qu'``osgeo`` y est toujours présent. Sans ce repli, ``EPSG:2154`` — pourtant projeté —
    était classé ``None`` côté QGIS (ImportError avalé) → faux « CRS non projeté ».
    Ne renvoie ``None`` que si **aucun** backend ne sait interpréter le CRS.
    """
    if not crs:
        return None
    # 1) rasterio (présent en test ; souvent absent dans QGIS)
    try:
        from rasterio.crs import CRS  # type: ignore

        c = CRS.from_user_input(crs)
        if c.is_geographic:
            return False
        if c.is_projected:
            return True
        # ni géographique ni projeté (local) → tenter GDAL avant d'abandonner
    except Exception:
        pass
    # 2) osgeo.osr (toujours présent côté QGIS)
    try:
        from osgeo import osr

        srs = osr.SpatialReference()
        if srs.SetFromUserInput(str(crs)) == 0 and not srs.IsLocal():  # 0 = OGRERR_NONE
            if srs.IsGeographic():
                return False
            if srs.IsProjected():
                return True
    except Exception:
        pass
    return None


def same_crs_geometry(
    a: Optional[str],
    b: Optional[str],
    *,
    ref_lonlat: Tuple[float, float] = (2.5, 47.0),
    tol_m: float = 1.0,
) -> Optional[bool]:
    """``True``/``False`` si ``a`` et ``b`` placent un même point lon/lat au même
    endroit projeté, ``None`` si indéterminable (entrée vide, CRS non projeté, ou
    aucun backend ``osgeo``).

    Mesure le **placement** (ce qui cause un mauvais géoréférencement), pas
    l'identité stricte du datum : un Lambert-93 *custom* (WKT sans code EPSG, datum
    « unnamed » sur GRS80) est ainsi reconnu équivalent à ``EPSG:2154`` (RGF93 ≈
    WGS84), tandis qu'un autre CRS projeté (UTM, Lambert-II…) est distingué.
    ``ref_lonlat`` par défaut au centre de la France métropolitaine.
    """
    if not a or not b:
        return None
    try:
        from osgeo import osr
    except Exception:
        return None
    try:
        sa = osr.SpatialReference()
        sb = osr.SpatialReference()
        if sa.SetFromUserInput(str(a)) != 0 or sb.SetFromUserInput(str(b)) != 0:
            return None
        if not (sa.IsProjected() and sb.IsProjected()):
            return None
        wgs = osr.SpatialReference()
        wgs.SetWellKnownGeogCS("WGS84")
        for s in (sa, sb, wgs):
            try:
                s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            except Exception:
                pass
        lon, lat = ref_lonlat
        xa, ya, *_ = osr.CoordinateTransformation(wgs, sa).TransformPoint(lon, lat)
        xb, yb, *_ = osr.CoordinateTransformation(wgs, sb).TransformPoint(lon, lat)
        return (abs(xa - xb) <= tol_m) and (abs(ya - yb) <= tol_m)
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


def tag_byte_nodata(path, value: int = 255) -> bool:
    """Déclare ``value`` comme NoData sur un raster 8 bits qui n'en a pas d'exploitable.

    rvt-qgis écrit **255** dans les cellules sans donnée de ses rendus 8 bits
    (``rvt/vis.py:byte_scale`` — « change no_data to 255 ») mais étiquette la
    bande avec ``nan``, ce qui n'a aucun sens sur du Byte : ni GDAL ni QGIS ne
    peuvent masquer. Conséquences observées sur un MNT à large NoData (dalle
    étrangère reprojetée) : les zones sans donnée sortent en **blanc opaque**
    dans la dalle, et en **noir** dans les trous de la mosaïque (gdalbuildvrt
    hérite du ``nan``, initialise le tampon Byte à 0). ``compute_cvat`` ne pose
    aucune étiquette : même symptôme, même correctif.

    *Assignation d'étiquette uniquement* (aucun pixel n'est réécrit). ⚠️ 255
    contient AUSSI les pixels valides écrêtés en haut de plage par ``byte_scale``
    (tout x ≥ ~99,6 % de la plage → 255) : c'est :func:`reclass_rvt_nodata`,
    appelée à la génération quand le MNT source est encore sous la main, qui
    sépare les deux classes (saturés → 254). Ce tag reste le filet de sécurité
    pour les rasters orphelins (mode ``existing_rvt``, sorties antérieures).

    No-op si le raster n'est pas 8 bits ou porte déjà un NoData exploitable.
    Renvoie True si l'étiquette a été écrite. Essaie rasterio puis ``osgeo.gdal``
    (même stratégie que :func:`assign_crs_if_missing`).
    """
    def _unusable(nd) -> bool:
        return nd is None or nd != nd  # None ou NaN

    try:  # rasterio (présent en test ; parfois dans QGIS)
        import rasterio

        with rasterio.open(str(path), "r+") as ds:
            if set(ds.dtypes) != {"uint8"} or not _unusable(ds.nodata):
                return False
            ds.nodata = value
        return True
    except ImportError:
        pass
    except Exception:
        return False

    try:  # osgeo.gdal (toujours présent côté QGIS)
        from osgeo import gdal

        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            return False
        bands = [ds.GetRasterBand(i + 1) for i in range(ds.RasterCount)]
        if not bands or any(b.DataType != gdal.GDT_Byte for b in bands):
            return False
        if any(not _unusable(b.GetNoDataValue()) for b in bands):
            return False
        for b in bands:
            b.SetNoDataValue(value)
        ds = None  # flush sur disque
        return True
    except Exception:
        return False


#: Convention des rendus RVT 8 bits reclassés : 255 = NoData réel, 254 = valide saturé.
RVT_BYTE_NODATA = 255
RVT_BYTE_SATURATED = 254


def reclass_rvt_byte(byte_arr, dem_invalid):
    """Reclasse (copie) un rendu RVT 8 bits d'après le masque NoData de son MNT.

    ``byte_scale`` (rvt-qgis) écrit 255 pour DEUX classes indiscernables après
    coup : les nan (emprise NoData du MNT — vérifié dans rvt/vis.py : slrm,
    local_dominance, hillshade et svf sont nan-aware, le nan ne se propage PAS
    aux voisins, le masque du produit = exactement celui du MNT) et les valeurs
    valides écrêtées en haut de plage. Séparation exacte, sans dilatation :

    - MNT invalide → 255 (NoData) ;
    - 255 sur MNT valide → 254 (saturé, visuellement quasi identique).

    ``byte_arr`` : uint8 ``(H, W)`` ou ``(bandes, H, W)`` ; ``dem_invalid`` :
    bool ``(H, W)``. Pure numpy, idempotente.
    """
    import numpy as np

    out = np.array(byte_arr, copy=True)
    dem_valid = ~np.asarray(dem_invalid, dtype=bool)
    if out.ndim == 2:
        out[(out == RVT_BYTE_NODATA) & dem_valid] = RVT_BYTE_SATURATED
        out[~dem_valid] = RVT_BYTE_NODATA
    else:
        out[(out == RVT_BYTE_NODATA) & dem_valid[None, :, :]] = RVT_BYTE_SATURATED
        out[:, ~dem_valid] = RVT_BYTE_NODATA
    return out


def reclass_rvt_nodata(rvt_path, dem_path) -> bool:
    """Applique :func:`reclass_rvt_byte` à un fichier RVT 8 bits, d'après son MNT.

    À appeler **à la génération**, tant que le MNT source est sous la main —
    après coup les deux classes de 255 sont indiscernables. Pose aussi
    l'étiquette NoData=255. Idempotent. Renvoie False (no-op) si le raster
    n'est pas 8 bits ou si les grilles diffèrent. Essaie rasterio puis
    ``osgeo.gdal`` (même stratégie que :func:`tag_byte_nodata`).
    """
    import numpy as np

    def _dem_invalid(arr, nd):
        invalid = np.isnan(arr)
        if nd is not None and not (nd != nd):  # nodata défini et pas nan
            invalid |= arr == nd
        return invalid

    try:  # rasterio (présent en test ; parfois dans QGIS)
        import rasterio

        with rasterio.open(str(dem_path)) as dem:
            dem_arr = dem.read(1)
            invalid = _dem_invalid(dem_arr, dem.nodata)
        with rasterio.open(str(rvt_path), "r+") as ds:
            if set(ds.dtypes) != {"uint8"} or (ds.height, ds.width) != invalid.shape:
                return False
            ds.write(reclass_rvt_byte(ds.read(), invalid))
            ds.nodata = RVT_BYTE_NODATA
        return True
    except ImportError:
        pass
    except Exception:
        return False

    try:  # osgeo.gdal (toujours présent côté QGIS)
        from osgeo import gdal

        dem_ds = gdal.Open(str(dem_path))
        if dem_ds is None:
            return False
        dem_band = dem_ds.GetRasterBand(1)
        invalid = _dem_invalid(
            dem_band.ReadAsArray().astype(np.float64), dem_band.GetNoDataValue()
        )
        dem_ds = None

        ds = gdal.Open(str(rvt_path), gdal.GA_Update)
        if ds is None:
            return False
        bands = [ds.GetRasterBand(i + 1) for i in range(ds.RasterCount)]
        if any(b.DataType != gdal.GDT_Byte for b in bands) or (
            ds.RasterYSize, ds.RasterXSize
        ) != invalid.shape:
            return False
        arr = ds.ReadAsArray()  # (H, W) mono-bande, (bandes, H, W) sinon
        out = reclass_rvt_byte(arr, invalid)
        for i, b in enumerate(bands):
            b.WriteArray(out if out.ndim == 2 else out[i])
            b.SetNoDataValue(RVT_BYTE_NODATA)
        ds = None  # flush sur disque
        return True
    except Exception:
        return False


def is_degenerate_tile(
    spec: "TileSpec",
    *,
    min_px: int = DEGENERATE_MIN_PX,
    origin_radius_m: float = DEGENERATE_ORIGIN_RADIUS_M,
) -> bool:
    """Vrai si ``spec`` est une dalle « placeholder » à exclure de la mosaïque.

    Deux signaux indépendants (OU) :

    - **(A) taille triviale** : ``width_px < min_px`` ou ``height_px < min_px`` —
      attrape les dalles 1×1 px (et toute dimension 0/corrompue).
    - **(B) origine au point (0,0)** : le coin nord-ouest ``(xmin, ymax)`` est à
      moins de ``origin_radius_m`` de l'origine du CRS — une dalle sans
      géoréférencement réel (géotransformée ≈ identité) atterrit là.

    Conçu pur (aucune I/O) et donc testable directement.
    """
    if spec.width_px < min_px or spec.height_px < min_px:
        return True
    xmin, _ymin, _xmax, ymax = spec.bounds
    if abs(xmin) <= origin_radius_m and abs(ymax) <= origin_radius_m:
        return True
    return False


def partition_degenerate(specs):
    """Sépare ``specs`` en ``(kept, degenerate)`` en préservant l'ordre.

    ``kept`` = dalles exploitables, ``degenerate`` = placeholders détectés par
    :func:`is_degenerate_tile`.
    """
    kept = []
    degenerate = []
    for spec in specs:
        (degenerate if is_degenerate_tile(spec) else kept).append(spec)
    return kept, degenerate
