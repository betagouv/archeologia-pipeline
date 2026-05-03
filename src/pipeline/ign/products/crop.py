from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ...coords import extract_xy_from_tile_name as _extract_xy_from_tile_name
from ...subprocess_utils import subprocess_kwargs_no_window
from .rvt_naming import get_rvt_source_and_dest_filenames
from ...types import LogFn


# _extract_xy_from_tile_name importé depuis coords
# _subprocess_kwargs_no_window importé depuis subprocess_utils


def crop_final_products(
    *,
    temp_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    gdalwarp_path: Optional[str] = None,
) -> Dict[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)

    x, y = _extract_xy_from_tile_name(current_tile_name)

    gdalwarp = gdalwarp_path or shutil.which("gdalwarp")  # type: ignore[name-defined]
    if not gdalwarp:
        raise FileNotFoundError("gdalwarp executable not found in PATH")

    target_xmin = int(x) * 1000
    target_xmax = (int(x) + 1) * 1000
    target_ymin = (int(y) - 1) * 1000
    target_ymax = int(y) * 1000

    xmin_r = str(target_xmin)
    ymin_r = str(target_ymin)
    xmax_r = str(target_xmax)
    ymax_r = str(target_ymax)

    # Utiliser la fonction utilitaire pour générer les noms de fichiers avec paramètres
    cropped: Dict[str, Path] = {}

    for product_name in ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]:
        if not products.get(product_name, False):
            continue

        src_name, dst_name = get_rvt_source_and_dest_filenames(
            product_name, current_tile_name, x, y, rvt_params
        )
        src_path = temp_dir / src_name
        dst_path = temp_dir / dst_name

        if dst_path.exists():
            cropped[product_name] = dst_path
            continue

        if not src_path.exists():
            log(f"Fichier source introuvable pour {product_name}: {src_path.name}")
            continue
        
        # Vérifier que le fichier n'est pas vide/corrompu
        if src_path.stat().st_size == 0:
            log(f"Fichier source vide pour {product_name}: {src_path.name}")
            continue

        # Compression spécifique pour MNT: LERC_ZSTD avec tolérance 1cm
        # (précision LiDAR HD: 0-10cm absolu, 0-5cm relatif)
        if product_name == "MNT":
            cmd = [
                gdalwarp,
                "-te",
                xmin_r,
                ymin_r,
                xmax_r,
                ymax_r,
                str(src_path),
                str(dst_path),
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LERC_ZSTD",
                "-co",
                "MAX_Z_ERROR=0.01",
            ]
        else:
            cmd = [
                gdalwarp,
                "-te",
                xmin_r,
                ymin_r,
                xmax_r,
                ymax_r,
                str(src_path),
                str(dst_path),
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=ZSTD",
                "-co",
                "PREDICTOR=2",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, **subprocess_kwargs_no_window())
        if result.returncode != 0:
            raise RuntimeError(f"Erreur gdalwarp ({product_name}): {result.stderr or result.stdout}")

        if dst_path.exists():
            cropped[product_name] = dst_path

    return cropped


def copy_products_without_crop(
    *,
    temp_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
) -> Dict[str, Path]:
    """Variante de ``crop_final_products`` qui préserve l'emprise native du MNT.

    Utilisée quand le MNT fourni est plus petit qu'une dalle 1 km (ou non
    aligné sur la grille IGN). Au lieu de découper à une cellule 1 km (ce qui
    introduirait du NoData autour), on renomme simplement les produits RVT
    source vers leur nom "cropped" attendu par ``copy_final_products_to_results``.

    Le géoréférencement du TIF (et du fichier world .pgw/.jgw généré ensuite)
    reste celui du MNT d'origine : la chaîne de conversion détections → shapefile
    fonctionne indépendamment de la taille de l'image.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    x, y = _extract_xy_from_tile_name(current_tile_name)

    copied: Dict[str, Path] = {}

    for product_name in ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]:
        if not products.get(product_name, False):
            continue

        src_name, dst_name = get_rvt_source_and_dest_filenames(
            product_name, current_tile_name, x, y, rvt_params
        )
        src_path = temp_dir / src_name
        dst_path = temp_dir / dst_name

        if dst_path.exists():
            copied[product_name] = dst_path
            continue

        if not src_path.exists() or src_path.stat().st_size == 0:
            log(f"Fichier source introuvable/vide pour {product_name}: {src_path.name}")
            continue

        shutil.copy2(str(src_path), str(dst_path))
        log(f"{product_name}: copie sans crop (emprise native MNT) -> {dst_path.name}")
        copied[product_name] = dst_path

    return copied
