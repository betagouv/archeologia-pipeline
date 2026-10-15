"""Calcul de l'indice CVAT (*Combined Visualization for Archaeological Topography*).

CVAT n'est pas exposé par le framework Processing de QGIS : l'enum ``BLEND_COMBINATION`` de
``rvt:rvt_blender`` ne liste que les combinaisons de ``settings/default_blender_combinations.json``
(VAT / Prismatic openness / City) ; CVAT est câblé en dur dans la boîte de dialogue de rvt-qgis.

On le reproduit avec la bibliothèque rvt intégrée au plugin (``rvt_embarque``, voir son NOTICE) : deux rendus de
VAT (terrains *general* et *flat*) fusionnés en mode *Normal*, opacités 50 / 100, exactement comme la branche CVAT
de ``rvt_blender.py``. Imports de rvt différés (module importable sans numpy ni GDAL).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ...types import LogFn


def _find_rendered_tif(out_base: Path, save_as_8bit: bool) -> Optional[Path]:
    """Localise le .tif produit par ``render_all_images`` (suffixe variable).

    RVT ajoute ``_8bit`` (ou rien) avant l'extension selon le mode de sortie ;
    on tente le nom attendu puis on retombe sur un glob (même logique que le bloc
    VAT de ``indices.py``).
    """
    suffix = "_8bit.tif" if save_as_8bit else ".tif"
    expected = Path(str(out_base) + suffix)
    if expected.exists():
        return expected
    matches = sorted(out_base.parent.glob(f"{out_base.name}*.tif"))
    return matches[0] if matches else None


def compute_cvat(
    *,
    input_path: Path,
    output_path: Path,
    couches,
    save_as_8bit: bool = True,
    log: LogFn = lambda _: None,
) -> Optional[Path]:
    """Calcule CVAT pour ``input_path`` (MNT) et écrit le raster dans ``output_path``.

    ``couches`` : fonction qui rend le MNT de la dalle et ses couches partagées (``moteur_indices.Couches``, lu au
    premier appel), appelée dans le ``try`` comme l'était la lecture du MNT ; les deux VAT reçoivent leurs couches
    déjà calculées (``moteur_indices.vat_cvat``) au lieu de les recalculer.

    Renvoie ``output_path`` en cas de succès, ``None`` si le rendu échoue (l'indice est alors
    simplement ignoré, le reste du pipeline continue).
    """
    try:
        c = couches()
        from .moteur_indices import vat_cvat
        from .rvt_embarque import blend

        dem_path = str(input_path)
        no_data = c.no_data
        save_float = not save_as_8bit
        vat_arr_general = vat_cvat(c, "general")
        vat_arr_flat = vat_cvat(c, "flat")

        combination = blend.BlenderCombination()
        combination.create_layer(
            vis_method="VAT general", image=vat_arr_general, normalization="Value",
            minimum=0, maximum=1, blend_mode="Normal", opacity=50,
        )
        combination.create_layer(
            vis_method="VAT flat", image=vat_arr_flat, normalization="Value",
            minimum=0, maximum=1, blend_mode="Normal", opacity=100,
        )
        combination.add_dem_path(dem_path)

        out_base = output_path.with_suffix("").with_name(output_path.stem + "_cvat_outputs")
        combination.render_all_images(
            save_render_path=str(out_base), save_visualizations=False,
            save_float=save_float, save_8bit=save_as_8bit, no_data=no_data,
        )
    except Exception as exc:  # pragma: no cover - dépend de numpy/scipy/GDAL
        log(f"CVAT : échec du calcul in-process ({exc!r}) — indice ignoré.")
        return None

    produced = _find_rendered_tif(out_base, save_as_8bit)
    if produced is None:
        log("CVAT : aucun raster produit par RVT — indice ignoré.")
        return None

    shutil.copy2(str(produced), str(output_path))
    return output_path
