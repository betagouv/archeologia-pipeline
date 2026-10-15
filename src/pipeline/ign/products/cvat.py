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
from .moteur_indices import chemin_reglages


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
    save_as_8bit: bool = True,
    log: LogFn = lambda _: None,
) -> Optional[Path]:
    """Calcule CVAT pour ``input_path`` (MNT) et écrit le raster dans ``output_path``.

    Renvoie ``output_path`` en cas de succès, ``None`` si le rendu échoue (l'indice est alors
    simplement ignoré, le reste du pipeline continue).
    """
    try:
        from .rvt_embarque import blend, default

        dem_path = str(input_path)
        dict_arr = default.get_raster_arr(dem_path)
        dem_arr = dict_arr["array"]
        resolution = dict_arr["resolution"]  # (x_res, y_res)
        no_data = dict_arr["no_data"]

        save_float = not save_as_8bit

        default_general = default.DefaultValues()
        default_flat = default.DefaultValues()

        vat_general = blend.BlenderCombination()
        vat_flat = blend.BlenderCombination()
        vat_general.read_from_file(str(chemin_reglages("blender_VAT.json")))
        vat_flat.read_from_file(str(chemin_reglages("blender_VAT.json")))

        terrains = blend.TerrainsSettings()
        terrains.read_from_file(str(chemin_reglages("default_terrains_settings.json")))
        terrains.select_terrain_settings_by_name("general").apply_terrain(
            default=default_general, combination=vat_general
        )
        terrains.select_terrain_settings_by_name("flat").apply_terrain(
            default=default_flat, combination=vat_flat
        )

        vat_general.add_dem_arr(dem_arr=dem_arr, dem_resolution=resolution[0])
        vat_arr_general = vat_general.render_all_images(default=default_general, no_data=no_data)
        vat_flat.add_dem_arr(dem_arr=dem_arr, dem_resolution=resolution[0])
        vat_arr_flat = vat_flat.render_all_images(default=default_flat, no_data=no_data)

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
