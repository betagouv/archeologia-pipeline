"""Calcul de l'indice CRIM (*Color Relief Image Map*).

Comme le CVAT, CRIM n'est **pas** atteignable par le framework Processing de
QGIS : l'enum ``BLEND_COMBINATION`` de ``rvt:rvt_blender`` est construit
uniquement depuis ``settings/default_blender_combinations.json``, qui ne liste
que *VAT*, *Prismatic openness* et *City*. CRIM n'existe que comme fonction du
paquet ``rvt`` (``rvt.blend.color_relief_image_map``), sans branche dédiée dans
l'algorithme Processing — donc inatteignable par ``processing.run``.

On l'appelle donc *in-process*, en réutilisant le paquet ``rvt`` fourni par le
plugin tiers **rvt-qgis** (déjà requis pour tous les autres indices). La
localisation du paquet est partagée avec :mod:`cvat` — un seul point de
détection pour les deux indices in-process, et le préflight n'en interroge qu'un.

Recette, telle que numérotée par la docstring de ``color_relief_image_map`` :

1. ``openness positive − openness négative``, mode *overlay*, 50 % d'opacité ;
2. la même différence, mode *luminosity*, 50 % d'opacité ;
3. la **pente**, colorée par une colormap matplotlib (défaut ``OrRd``).

⚠ Cette numérotation va de la surface vers le FOND, et non l'inverse :
``BlenderCombination.render_all_images`` compose par ``range(len(layers) - 1,
-1, -1)``, donc la **dernière** couche est le fond. Le fond de CRIM est la
pente colorée, et les deux couches d'openness sont posées par-dessus — c'est
pour cela que la couleur porte la pente et le modelé l'openness.

La sortie est donc **RGB (3 bandes)**, comme MSTP et contrairement aux indices
en niveaux de gris : la couleur y porte l'information de pente.

Tout le couplage aux internes de RVT est isolé ici, avec des imports **différés**
(le module reste importable en standalone/tests sans QGIS ni rvt-qgis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...types import LogFn
from .cvat import _locate_rvt_package

#: Défauts du plugin, alignés sur la signature de ``color_relief_image_map``.
DEFAUT_COLORMAP = "OrRd"
DEFAUT_CUT_MIN = 0.0
DEFAUT_CUT_MAX = 1.0


def compute_crim(
    *,
    input_path: Path,
    output_path: Path,
    colormap: str = DEFAUT_COLORMAP,
    min_colormap_cut: float = DEFAUT_CUT_MIN,
    max_colormap_cut: float = DEFAUT_CUT_MAX,
    save_as_8bit: bool = True,
    log: LogFn = lambda _: None,
) -> Optional[Path]:
    """Calcule CRIM pour ``input_path`` (MNT) et écrit le raster dans ``output_path``.

    ``colormap`` est un nom de colormap matplotlib appliqué à la **pente**
    (et non à l'altitude) ; ``min_colormap_cut`` / ``max_colormap_cut`` rognent
    les extrémités de cette colormap, entre 0 et 1.

    Renvoie ``output_path`` en cas de succès, ``None`` si rvt-qgis est absent ou
    si le rendu échoue (l'indice est alors simplement ignoré, le reste du
    pipeline continue — même contrat que :func:`cvat.compute_cvat`).
    """
    if _locate_rvt_package() is None:
        log(
            "CRIM ignoré : paquet RVT introuvable. Le plugin QGIS « Relief "
            "Visualization Toolbox » (rvt-qgis) doit être installé."
        )
        return None

    try:
        import numpy as np
        import rvt.blend  # type: ignore
        import rvt.default  # type: ignore
        import rvt.vis  # type: ignore

        dem_path = str(input_path)
        dict_arr = rvt.default.get_raster_arr(dem_path)
        # ``color_relief_image_map`` écrit des NaN DANS le tableau qu'on lui
        # passe (``dem[dem == no_data] = np.nan``) : on lui donne une copie,
        # sinon on abîmerait un tableau que l'appelant pourrait réutiliser.
        dem_arr = np.array(dict_arr["array"], copy=True)
        resolution = dict_arr["resolution"][0]
        no_data = dict_arr["no_data"]

        crim_arr = rvt.blend.color_relief_image_map(
            dem=dem_arr,
            resolution=resolution,
            default=rvt.default.DefaultValues(),
            colormap=colormap,
            min_colormap_cut=min_colormap_cut,
            max_colormap_cut=max_colormap_cut,
            no_data=no_data,
        )

        # Écriture en deux temps : RVT écrit dans un voisin temporaire, qu'on
        # bascule seulement une fois le rendu complet. Écrire directement dans
        # ``output_path`` laisserait, en cas d'interruption (disque plein, GDAL),
        # un TIF partiel — et l'appelant ne teste que ``out.exists()``, il le
        # prendrait donc pour un succès.
        temporaire = output_path.with_name(output_path.name + ".tmp")
        # Même convention que la branche e3MSTP de rvt_blender.py : 8 bits via
        # byte_scale sur [0, 1] (e_type=1), sinon flottant brut (e_type=6).
        if save_as_8bit:
            rvt.default.save_raster(
                src_raster_path=dem_path, out_raster_path=str(temporaire),
                out_raster_arr=rvt.vis.byte_scale(crim_arr, c_min=0.0, c_max=1.0),
                no_data=np.nan, e_type=1,
            )
        else:
            rvt.default.save_raster(
                src_raster_path=dem_path, out_raster_path=str(temporaire),
                out_raster_arr=crim_arr, no_data=np.nan, e_type=6,
            )
        if not temporaire.exists():
            log("CRIM : aucun raster produit par RVT — indice ignoré.")
            return None
        temporaire.replace(output_path)
    except Exception as exc:  # pragma: no cover - dépend de l'environnement RVT
        log(f"CRIM : échec du calcul in-process ({exc!r}) — indice ignoré.")
        return None

    if not output_path.exists():
        log("CRIM : aucun raster produit par RVT — indice ignoré.")
        return None
    return output_path
