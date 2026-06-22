"""Planification de l'ingestion des entrées raster (placement métadonnées-first).

Construit un :class:`~pipeline.tilespec.TileSpec` par raster, valide le CRS
(un seul CRS **projeté** par run — décision 2), avertit sur les résolutions
hétérogènes (décision 4), ignore les rasters illisibles/vides avec un résumé
(décision 6), et décide si l'entrée est **mosaïcable** (chemin primaire) ou doit
passer par le repli par dalle.

Importable sans QGIS (imports ``rasterio`` différés). La logique de validation
pure (:func:`plan_from_specs`) est testable directement avec des ``TileSpec``
construits via ``from_values`` ; :func:`plan_raster_inputs` lit le disque.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .tilespec import (
    TileSpec,
    crs_is_projected,
    partition_degenerate,
    same_crs_geometry,
)

RASTER_EXTS = {".tif", ".tiff", ".asc"}

# Raison de mise à l'écart des dalles « placeholder » (1×1 px / posées à l'origine
# du CRS) — exploitée par le preflight pour les compter et avertir l'utilisateur.
DEGENERATE_SKIP_REASON = "placeholder dégénéré (1×1 px / origine ≈ 0,0)"


def _epsg_code(crs: Optional[str]) -> Optional[str]:
    """Extrait le code EPSG d'un authid (« EPSG:2154 » → « 2154 »), sinon None."""
    m = re.match(r"^\s*EPSG:(\d+)\s*$", str(crs or ""), re.IGNORECASE)
    return m.group(1) if m else None


def _short_crs(crs: Optional[str]) -> str:
    """Libellé court d'un CRS pour les messages utilisateur : authid si présent,
    sinon le nom du PROJCS/PROJCRS du WKT, sinon « CRS projeté custom »."""
    code = _epsg_code(crs)
    if code:
        return f"EPSG:{code}"
    m = re.match(r'\s*PROJ(?:CS|CRS)\[\s*"([^"]+)"', str(crs or ""))
    return m.group(1) if m else "CRS projeté custom"


def same_epsg(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """``True``/``False`` si les deux CRS ont un code EPSG comparable, ``None``
    si l'un est un WKT non identifiable (comparaison indécidable)."""
    ca, cb = _epsg_code(a), _epsg_code(b)
    if ca is None or cb is None:
        return None
    return ca == cb


class IngestValidationError(ValueError):
    """Entrées invalides (CRS absent/mélangé/géographique, aucune dalle) — abandon préflight."""


@dataclass(frozen=True)
class IngestPlan:
    tiles: List[TileSpec]               # dalles valides retenues
    crs: Optional[str]                  # CRS unifié du run (effective)
    mosaicable: bool                    # CRS + résolution uniformes et > 1 dalle
    skipped: List[Tuple[Path, str]]     # (chemin, raison)
    warnings: List[str]
    crs_verified: bool = True           # False si CRS accepté faute de pouvoir le classer

    @property
    def summary(self) -> str:
        parts = [f"{len(self.tiles)} dalle(s) retenue(s)"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} ignorée(s)")
        parts.append(f"CRS={self.crs}")
        parts.append("mosaïque" if self.mosaicable else "par-dalle")
        return ", ".join(parts)


def plan_from_specs(
    tiles: Sequence[TileSpec],
    *,
    skipped: Optional[Sequence[Tuple[Path, str]]] = None,
    model_resolution: Optional[float] = None,
    resolution_tol: float = 1e-6,
    expected_crs: Optional[str] = None,
) -> IngestPlan:
    """Validation pure (sans I/O) à partir de ``TileSpec`` déjà construits.

    Lève :class:`IngestValidationError` pour les erreurs dures (CRS) ; accumule
    les problèmes souples (résolution) dans ``warnings``.

    ``expected_crs`` (ex. ``"EPSG:2154"``) : si fourni, le CRS du run doit lui
    correspondre. Sinon (AUDIT v2 GEO-02), un raster projeté mais dans un autre
    CRS — ex. Lambert-II EPSG:27572 — serait traité comme si ses coordonnées
    étaient en 2154 (étiquetage GeoPackage en dur) → détections mal placées.
    """
    skipped_list: List[Tuple[Path, str]] = list(skipped or [])
    tiles = list(tiles)
    warnings: List[str] = []

    # Retirer les dalles « dégénérées » (placeholder 1×1 px / posées à l'origine du
    # CRS) AVANT toute validation : incluses dans un gdalbuildvrt, elles étirent
    # l'emprise de la mosaïque jusqu'à (0,0) → couches QGIS quasi vides. Filtrées
    # avant le contrôle CRS pour qu'un placeholder sans CRS ne déclenche pas
    # l'erreur dure « CRS introuvable ».
    tiles, degenerate = partition_degenerate(tiles)
    if degenerate:
        skipped_list.extend((d.source_path, DEGENERATE_SKIP_REASON) for d in degenerate)
        warnings.append(
            f"{len(degenerate)} dalle(s) dégénérée(s) ignorée(s) "
            "(1×1 px / origine ≈ 0,0) — exclues de la mosaïque."
        )

    if not tiles:
        raise IngestValidationError(
            "Aucune dalle raster exploitable (toutes illisibles ou vides)."
        )

    # --- CRS : un seul CRS projeté par run ---
    effective = [t.effective_crs for t in tiles]
    if any(c is None for c in effective):
        missing = [t.source_path.name for t in tiles if t.effective_crs is None]
        shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        raise IngestValidationError(
            f"CRS introuvable et non déclaré pour : {shown}. "
            "Déclarez un CRS projeté (ex. EPSG:2154) pour ces entrées."
        )
    distinct = sorted(set(effective))
    if len(distinct) > 1:
        raise IngestValidationError(
            "Mélange de CRS détecté (" + ", ".join(distinct[:4]) + ") — un seul CRS par run. "
            "Reprojetez vos entrées dans un CRS commun avant traitement."
        )
    run_crs = distinct[0]
    crs_verified = True

    # CRS attendu (GEO-02) : refuser un CRS projeté mais ≠ celui du pipeline (ex. Lambert-II
    # EPSG:27572). Comparaison purement textuelle (codes EPSG) → indépendante de tout backend.
    if expected_crs is not None and same_epsg(run_crs, expected_crs) is False:
        raise IngestValidationError(
            f"CRS « {run_crs} » différent du CRS attendu « {expected_crs} » (Lambert-93). "
            "Le pipeline étiquette toutes les sorties en EPSG:2154 : un autre CRS placerait "
            "les détections au mauvais endroit. Reprojetez vos entrées en EPSG:2154."
        )

    # Garde-fou : si le code EPSG est celui attendu (CRS projeté connu), c'est valide par
    # définition — inutile d'exiger un backend géo pour le confirmer (corrige le faux
    # « EPSG:2154 non interprétable » quand rasterio manque dans QGIS).
    confirmed_by_epsg = expected_crs is not None and same_epsg(run_crs, expected_crs) is True

    if not confirmed_by_epsg:
        projected = crs_is_projected(run_crs)
        if projected is False:
            raise IngestValidationError(
                f"CRS géographique (degrés) « {run_crs} » non supporté : les indices RVT exigent "
                "un CRS projeté en mètres. Reprojetez vos entrées (ex. EPSG:2154)."
            )
        if projected is None:
            # Ni interprétable par un backend, ni confirmé par correspondance EPSG : on n'est
            # pas SÛR que ce soit faux → avertir au lieu de bloquer (le pipeline ré-affecte 2154).
            crs_verified = False
            warnings.append(
                f"CRS « {run_crs} » non vérifiable (aucun backend de lecture raster disponible) — "
                "vérifiez que vos entrées sont en EPSG:2154, sinon les détections seront mal placées."
            )
        elif expected_crs is not None:
            # Garde-fou (utilisateur : « avertir, ne pas forcer ») : CRS projeté SANS code
            # EPSG comparable (WKT). On mesure géométriquement s'il place les coordonnées
            # comme le CRS attendu — un Lambert-93 custom (datum « unnamed ») passe ; un
            # autre CRS projeté (UTM, Lambert-II…) déclenche un avertissement NON bloquant.
            geom = same_crs_geometry(run_crs, expected_crs)
            if geom is False:
                crs_verified = False
                warnings.append(
                    f"CRS projeté « {_short_crs(run_crs)} » différent géométriquement du CRS "
                    f"attendu « {expected_crs} » (Lambert-93) — le pipeline étiquette les sorties "
                    "en EPSG:2154 : vos indices/détections seront mal placés. "
                    "Reprojetez vos entrées en EPSG:2154."
                )

    # --- Résolution : avertir seulement (décision 4) ---
    resolutions = {
        (round(abs(t.pixel_size_x), 6), round(abs(t.pixel_size_y), 6)) for t in tiles
    }
    uniform_res = len(resolutions) == 1
    if not uniform_res:
        listed = ", ".join(f"{rx}×{ry} m" for rx, ry in sorted(resolutions))
        warnings.append(
            f"Résolutions hétérogènes ({listed}) — RVT à la résolution native ; "
            "pas de mosaïque inter-résolution."
        )
    if model_resolution is not None:
        off = [
            t for t in tiles
            if abs(abs(t.pixel_size_x) - model_resolution) > resolution_tol
        ]
        if off:
            warnings.append(
                f"{len(off)} dalle(s) à une résolution ≠ résolution d'entraînement du modèle "
                f"({model_resolution} m) — détections possiblement dégradées."
            )

    mosaicable = (len(tiles) > 1) and uniform_res

    return IngestPlan(
        tiles=tiles,
        crs=run_crs,
        mosaicable=mosaicable,
        skipped=skipped_list,
        warnings=warnings,
        crs_verified=crs_verified,
    )


def plan_raster_inputs(
    paths: Sequence[Path],
    *,
    declared_crs: Optional[str] = None,
    model_resolution: Optional[float] = None,
    expected_crs: Optional[str] = None,
) -> IngestPlan:
    """Lit chaque raster (métadonnées), ignore illisibles/vides, puis valide.

    ``declared_crs`` sert de CRS de repli pour les entrées sans CRS (ex. ``.asc``).
    ``expected_crs`` : CRS exigé pour le run (cf. :func:`plan_from_specs`).
    """
    tiles: List[TileSpec] = []
    skipped: List[Tuple[Path, str]] = []

    for p in paths:
        p = Path(p)
        spec = TileSpec.from_raster(p, declared_crs=declared_crs)
        if spec is None:
            skipped.append((p, "illisible/corrompu"))
            continue
        if not _raster_has_valid_data(p, spec.nodata):
            skipped.append((p, "vide / tout-NoData"))
            continue
        tiles.append(spec)

    return plan_from_specs(
        tiles, skipped=skipped, model_resolution=model_resolution,
        expected_crs=expected_crs,
    )


def _raster_has_valid_data(path, nodata, *, sample: int = 64) -> bool:
    """Vrai s'il existe au moins un pixel valide (lecture décimée).

    Conservateur : renvoie ``True`` en cas d'incertitude (on n'exclut jamais une
    dalle sur un doute). Ne devine jamais le NoData : si ``nodata`` est ``None`` et
    qu'aucun masque n'est défini, la dalle est considérée valide.
    """
    try:
        import numpy as np  # type: ignore
        import rasterio  # type: ignore

        with rasterio.open(str(path)) as ds:
            h = min(sample, ds.height)
            w = min(sample, ds.width)
            arr = ds.read(1, out_shape=(h, w), masked=True)
            mask = np.ma.getmaskarray(arr)
            if mask.all():
                return False
            valid = ~mask
            if nodata is not None:
                valid = valid & (np.ma.getdata(arr) != nodata)
            return bool(valid.any())
    except Exception:
        return True
