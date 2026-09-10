"""Halo inter-dalles fabriqué depuis les dalles voisines du run.

En ``ign_laz`` / ``local_laz`` l'inférence tourne sur les TIF non rognés
d'``intermediaires/`` (option B). Les modes sans fusion de voisins
(``existing_rvt``, ``existing_mnt``) inféraient chaque dalle 1 km seule : un
objet à cheval sur une frontière sortait coupé au bord — ou pas du tout quand
la moitié visible ne passait plus le seuil. Ici la marge est découpée dans la
mosaïque des dalles FOURNIES (VRT GDAL) : vraie donnée là où un voisin existe,
aplat 0 ailleurs (bruit supprimé en aval par ``valid_region_bounds``).

Fraîcheur comme le LAZ fusionné (``preprocess.merge_tiles``) : un halo est
réutilisé si son jeu de voisins (sidecar ``.inputs.json``) est inchangé et
qu'aucune entrée n'est plus récente que lui ; sinon il est re-découpé et son
mtime frais ré-arme la chaîne aval (PNG → purge du cache CV → ré-inférence).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..types import LogFn

Bounds = Tuple[float, float, float, float]
ExtractFn = Callable[[Sequence[Path], Bounds, Path], None]

#: Marge (m) autour de chaque dalle. Un objet à cheval est vu entier par la
#: dalle qui contient son centre dès que marge ≥ taille/2 → 50 m couvre 100 m.
#: Et c'est GRATUIT à l'inférence : 2 000 + 2 × 100 px = 2 200 px reste sous
#: 648 + 3 × 518 = 2 202 px, la grille SAHI 4×4 des modèles 648/672 px
#: (recouvrement 0,2) n'ajoute aucune tuile ; à 200 m (2 800 px) on passe à
#: 6×6 = × 2,25.
DEFAULT_HALO_MARGIN_M = 50.0


def halo_window(bounds: Bounds, margin_m: float) -> Bounds:
    xmin, ymin, xmax, ymax = bounds
    return (xmin - margin_m, ymin - margin_m, xmax + margin_m, ymax + margin_m)


def halo_inputs(target: Path, tiles: Dict[Path, Bounds], margin_m: float) -> List[Path]:
    """Dalles (cible comprise) dont l'emprise intersecte la fenêtre étendue."""
    wx0, wy0, wx1, wy1 = halo_window(tiles[target], margin_m)
    return sorted(
        p for p, (x0, y0, x1, y1) in tiles.items()
        if x0 < wx1 and x1 > wx0 and y0 < wy1 and y1 > wy0
    )


def _gdal_extract(inputs: Sequence[Path], window: Bounds, dst: Path) -> None:
    """Mosaïque VRT des entrées, découpée à ``window`` → GeoTIFF ``dst``."""
    from osgeo import gdal

    vrt_path = f"/vsimem/halo_{dst.stem}.vrt"
    # Handler silencieux : des RVT 8 bits déclarent parfois NoData=nan (« Band
    # data type of Byte cannot represent… »), un avertissement par entrée et
    # par dalle qui noierait le journal QGIS sans rien changer au résultat.
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        vrt = gdal.BuildVRT(vrt_path, [str(p) for p in inputs], outputBounds=window)
        if vrt is None:
            raise RuntimeError(f"gdal.BuildVRT a échoué ({gdal.GetLastErrorMsg()})")
        try:
            out = gdal.Translate(str(dst), vrt, creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
            if out is None:
                raise RuntimeError(f"gdal.Translate a échoué ({gdal.GetLastErrorMsg()})")
            out = None
        finally:
            vrt = None
            gdal.Unlink(vrt_path)
    finally:
        gdal.PopErrorHandler()


class NeighborHalo:
    """Résout, pour une dalle du run, son TIF « dalle + marge » fabriqué à la demande."""

    def __init__(
        self,
        tiles: Dict[Path, Bounds],
        halo_dir: Path,
        margin_m: float = DEFAULT_HALO_MARGIN_M,
        *,
        log: LogFn = lambda _: None,
        extract_fn: Optional[ExtractFn] = None,
    ) -> None:
        self._tiles = dict(tiles)
        self._halo_dir = Path(halo_dir)
        self._margin = float(margin_m)
        self._log = log
        self._extract = extract_fn
        self.built = 0
        self.reused = 0
        self.skipped = 0

    @property
    def margin_m(self) -> float:
        return self._margin

    def resolve(self, tif: Path) -> Optional[Path]:
        bounds = self._tiles.get(Path(tif))
        if bounds is None:
            return None
        inputs = halo_inputs(Path(tif), self._tiles, self._margin)
        if len(inputs) < 2:  # aucun voisin : la marge ne serait qu'un aplat
            self.skipped += 1
            return None
        dst = self._halo_dir / Path(tif).name
        sidecar = dst.with_suffix(".inputs.json")
        expected = sorted(p.name for p in inputs)
        if dst.exists() and _is_fresh(dst, sidecar, expected, inputs):
            self.reused += 1
            return dst
        try:
            self._halo_dir.mkdir(parents=True, exist_ok=True)
            (self._extract or _gdal_extract)(inputs, halo_window(bounds, self._margin), dst)
            sidecar.write_text(json.dumps(expected), encoding="utf-8")
        except Exception as exc:
            self._log(
                f"Halo inter-dalles impossible pour {Path(tif).name} ({exc}) "
                "— inférence sur la dalle seule"
            )
            return None
        self.built += 1
        return dst


def _is_fresh(dst: Path, sidecar: Path, expected: List[str], inputs: Sequence[Path]) -> bool:
    try:
        if json.loads(sidecar.read_text(encoding="utf-8")) != expected:
            return False
        dst_mtime = dst.stat().st_mtime
        return all(p.stat().st_mtime <= dst_mtime for p in inputs)
    except (OSError, ValueError):
        return False
