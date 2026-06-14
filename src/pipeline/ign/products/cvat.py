"""Calcul de l'indice CVAT (*Combined Visualization for Archaeological Topography*).

Contrairement aux autres indices RVT, CVAT n'est **pas** exposé par le framework
Processing de QGIS : l'algorithme ``rvt:rvt_blender`` ne propose dans son enum
``BLEND_COMBINATION`` que les combinaisons listées dans
``settings/default_blender_combinations.json`` (VAT / Prismatic openness / City).
CVAT est une combinaison « avancée » câblée en dur dans le plugin RVT, accessible
uniquement via sa boîte de dialogue.

On le reproduit donc *in-process* en réutilisant le paquet ``rvt`` fourni par le
plugin tiers **rvt-qgis** (déjà requis pour tous les autres indices) : on rend deux
variantes de VAT (terrains *general* et *flat*) puis on les fusionne (opacités
50 / 100), exactement comme la branche CVAT de ``rvt_blender.py``.

Tout le couplage aux internes de RVT est isolé ici, avec des imports **différés**
(le module reste importable en standalone/tests sans QGIS ni rvt-qgis).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...types import LogFn


@dataclass(frozen=True)
class RvtPaths:
    """Chemins utiles dans le plugin rvt-qgis installé."""

    plugin_dir: Path
    blender_vat_json: Path
    terrains_json: Path


def _settings_paths(rvt_plugin_dir: Path) -> Optional[RvtPaths]:
    """Construit :class:`RvtPaths` si les fichiers de settings RVT existent."""
    blender_vat = rvt_plugin_dir / "settings" / "blender_VAT.json"
    terrains = rvt_plugin_dir / "settings" / "default_terrains_settings.json"
    if blender_vat.is_file() and terrains.is_file():
        return RvtPaths(plugin_dir=rvt_plugin_dir, blender_vat_json=blender_vat, terrains_json=terrains)
    return None


def _find_rvt_dir(start: Path) -> Optional[Path]:
    """Cherche le dossier du plugin rvt-qgis parmi les plugins frères.

    Remonte les ancêtres de ``start`` (typiquement ``__file__``) et, à chaque
    niveau, inspecte les dossiers enfants dont le nom commence par ``rvt`` (le
    plugin officiel s'installe sous ``rvt-qgis``). Un candidat est validé s'il
    contient à la fois le paquet ``rvt/blend.py`` et ``settings/blender_VAT.json``.

    Fonction pure (pas d'import de ``rvt``) → testable sans QGIS.
    """
    try:
        start = start.resolve()
    except OSError:
        pass
    for parent in start.parents:
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for cand in children:
            try:
                if not cand.is_dir() or not cand.name.lower().startswith("rvt"):
                    continue
                if (cand / "rvt" / "blend.py").is_file() and (
                    cand / "settings" / "blender_VAT.json"
                ).is_file():
                    return cand
            except OSError:
                continue
    return None


def _locate_rvt_package() -> Optional[RvtPaths]:
    """Rend le paquet ``rvt`` importable et renvoie les chemins de settings.

    1. Tente ``import rvt`` (rvt-qgis est souvent déjà chargé comme plugin actif) ;
    2. sinon, localise le dossier rvt-qgis voisin et l'ajoute à ``sys.path``.

    Renvoie ``None`` si rvt-qgis est introuvable ou non importable.
    """
    # 1) déjà importable ?
    try:
        import rvt  # type: ignore  # noqa: F401

        rvt_dir = Path(rvt.__file__).resolve().parent.parent
        paths = _settings_paths(rvt_dir)
        if paths is not None:
            return paths
    except Exception:
        pass

    # 2) recherche parmi les plugins frères
    rvt_dir = _find_rvt_dir(Path(__file__))
    if rvt_dir is None:
        return None
    if str(rvt_dir) not in sys.path:
        sys.path.insert(0, str(rvt_dir))
    try:
        import rvt  # type: ignore  # noqa: F401
    except Exception:
        return None
    return _settings_paths(rvt_dir)


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

    Réplique fidèlement la recette de la branche CVAT de ``rvt_blender.py`` :
    deux rendus VAT (terrains *general* et *flat*) fusionnés en mode *Normal*
    avec opacités 50 / 100.

    Renvoie ``output_path`` en cas de succès, ``None`` si rvt-qgis est absent ou
    si le rendu échoue (l'indice est alors simplement ignoré, le reste du
    pipeline continue).
    """
    paths = _locate_rvt_package()
    if paths is None:
        log(
            "CVAT ignoré : paquet RVT introuvable. Le plugin QGIS « Relief "
            "Visualization Toolbox » (rvt-qgis) doit être installé."
        )
        return None

    try:
        import rvt.blend  # type: ignore
        import rvt.default  # type: ignore

        dem_path = str(input_path)
        dict_arr = rvt.default.get_raster_arr(dem_path)
        dem_arr = dict_arr["array"]
        resolution = dict_arr["resolution"]  # (x_res, y_res)
        no_data = dict_arr["no_data"]

        save_float = not save_as_8bit

        default_general = rvt.default.DefaultValues()
        default_flat = rvt.default.DefaultValues()

        vat_general = rvt.blend.BlenderCombination()
        vat_flat = rvt.blend.BlenderCombination()
        vat_general.read_from_file(str(paths.blender_vat_json))
        vat_flat.read_from_file(str(paths.blender_vat_json))

        terrains = rvt.blend.TerrainsSettings()
        terrains.read_from_file(str(paths.terrains_json))
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

        combination = rvt.blend.BlenderCombination()
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
    except Exception as exc:  # pragma: no cover - dépend de l'environnement RVT
        log(f"CVAT : échec du calcul in-process ({exc!r}) — indice ignoré.")
        return None

    produced = _find_rendered_tif(out_base, save_as_8bit)
    if produced is None:
        log("CVAT : aucun raster produit par RVT — indice ignoré.")
        return None

    import shutil

    shutil.copy2(str(produced), str(output_path))
    return output_path
