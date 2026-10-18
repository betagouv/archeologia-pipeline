"""Calcul de l'indice CRIM (*Color Relief Image Map*).

Comme le CVAT, CRIM n'est pas atteignable par le framework Processing de QGIS : l'enum ``BLEND_COMBINATION`` de
``rvt:rvt_blender`` est construit depuis ``settings/default_blender_combinations.json``, qui ne liste que *VAT*,
*Prismatic openness* et *City*. CRIM n'existe que comme fonction de la bibliothèque rvt
(``blend.color_relief_image_map``), dont ``moteur_indices.crim`` reprend le corps avec la copie intégrée au plugin
(``rvt_embarque``) : mêmes couches, ses trois couches de calcul partagées avec les autres produits de la dalle.

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

Imports de rvt différés (module importable sans numpy ni GDAL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...types import LogFn

#: Défauts du plugin, alignés sur la signature de ``color_relief_image_map``.
DEFAUT_COLORMAP = "OrRd"
DEFAUT_CUT_MIN = 0.0
DEFAUT_CUT_MAX = 1.0


def compute_crim(
    *,
    input_path: Path,
    output_path: Path,
    couches,
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

    ``couches`` : fonction qui rend le MNT de la dalle et ses couches partagées (``moteur_indices.Couches``, lu au
    premier appel), appelée dans le ``try`` comme l'était la lecture du MNT : l'openness ± et la pente sont reprises
    si un autre produit les a déjà calculées (``moteur_indices.crim``).

    Renvoie ``output_path`` en cas de succès, ``None`` si le rendu échoue (l'indice est
    alors simplement ignoré, le reste du pipeline continue — même contrat que :func:`cvat.compute_cvat`).
    """
    try:
        c = couches()
        import numpy as np
        from .moteur_indices import crim
        from .rvt_embarque import default, vis

        dem_path = str(input_path)
        crim_arr = crim(c, colormap=colormap, min_colormap_cut=min_colormap_cut,
                        max_colormap_cut=max_colormap_cut)

        # Écriture en deux temps : RVT écrit dans un voisin temporaire, qu'on
        # bascule seulement une fois le rendu complet. Écrire directement dans
        # ``output_path`` laisserait, en cas d'interruption (disque plein, GDAL),
        # un TIF partiel — et l'appelant ne teste que ``out.exists()``, il le
        # prendrait donc pour un succès.
        temporaire = output_path.with_name(output_path.name + ".tmp")
        # Même convention que la branche e3MSTP de rvt_blender.py : 8 bits via
        # byte_scale sur [0, 1] (e_type=1), sinon flottant brut (e_type=6).
        if save_as_8bit:
            default.save_raster(
                src_raster_path=dem_path, out_raster_path=str(temporaire),
                out_raster_arr=vis.byte_scale(crim_arr, c_min=0.0, c_max=1.0),
                no_data=np.nan, e_type=1,
            )
        else:
            default.save_raster(
                src_raster_path=dem_path, out_raster_path=str(temporaire),
                out_raster_arr=crim_arr, no_data=np.nan, e_type=6,
            )
        if not temporaire.exists():
            log("CRIM : aucun raster produit par RVT — indice ignoré.")
            return None
        temporaire.replace(output_path)
    except Exception as exc:  # pragma: no cover - dépend de numpy/scipy/GDAL
        log(f"CRIM : échec du calcul in-process ({exc!r}) — indice ignoré.")
        return None

    if not output_path.exists():
        log("CRIM : aucun raster produit par RVT — indice ignoré.")
        return None
    return output_path
