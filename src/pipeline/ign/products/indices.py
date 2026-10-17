from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import moteur_indices
from .crim import compute_crim
from .cvat import compute_cvat
from .qgis_processing import run_qgis_algorithm
from .results import needs_refresh
from .rvt_naming import get_rvt_temp_filename, mstp_settings, opns_settings, slope_unit, svf_settings
from ...tilespec import reclass_rvt_nodata
from ...types import LogFn, format_params_line


@dataclass(frozen=True)
class IndicesResult:
    outputs: Dict[str, Path]


def _as_int(value: Any, default: int) -> int:
    """Convertit une valeur en int, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "10,5" -> 10)
            value = value.replace(",", ".")
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    """Convertit une valeur en float, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "1,7" -> 1.7)
            value = value.replace(",", ".")
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    try:
        return bool(value)
    except Exception:
        return bool(default)


#: Calculs dont le résultat a changé en 0.15.0 à nom de fichier égal (même dossier) : MSTP (luminosité appliquée
#: telle quelle, rvt-qgis l'arrondissait à l'entier) et M-HS 8 bits (élévation et exagération appliquées). Une note
#: ``<intermédiaire>.calcul`` porte la version ; sans elle — calculé par une version précédente, ou par Processing —
#: l'intermédiaire est recalculé une fois (needs_refresh ne voit que les dates). Le rognage et la publication
#: suivent par fraîcheur (results.needs_refresh).
VERSION_CALCUL = {"MSTP": "luminosite-reelle-1", "M_HS": "elevation-reelle-1",
                  # openness NÉGATIVE : sans-donnée exclu avant la négation (rvt-qgis le comptait comme un relief)
                  "OPNS": "negative-sans-donnee-exclu-1"}


def _note_calcul(out: Path) -> Path:
    return out.with_name(out.name + ".calcul")


def _a_recalculer(produit: str, source: Path, out: Path) -> bool:
    if needs_refresh(source, out):
        return True
    try:
        return _note_calcul(out).read_text(encoding="utf-8") != VERSION_CALCUL[produit]
    except OSError:
        return True


def _noter_calcul(produit: str, out: Path, par_le_moteur: bool) -> None:
    note = _note_calcul(out)
    if par_le_moteur and out.exists():
        note.write_text(VERSION_CALCUL[produit], encoding="utf-8")
    elif note.exists():
        note.unlink()


def create_visualization_products(
    *,
    temp_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    feedback: Optional[Any] = None,
    context: Optional[Any] = None,
) -> IndicesResult:
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_file = f"{current_tile_name}_MNT.tif"
    input_path = temp_dir / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"MNT introuvable pour indices: {input_path}")

    outputs: Dict[str, Path] = {}
    couches: Optional[moteur_indices.Couches] = None

    def _couches() -> moteur_indices.Couches:
        """Le MNT de la dalle, lu une fois pour tous les produits du moteur intégré."""
        nonlocal couches
        if couches is None:
            couches = moteur_indices.Couches.lire(input_path)
            if not couches.partage:
                log(f"Indices : raster de {couches.taille[0]} × {couches.taille[1]} px, au-delà d'une dalle "
                    "standard : chaque produit recalcule ses couches (pas de partage, mémoire inchangée).")
        return couches

    if products.get("HS", False):
        hs = (rvt_params or {}).get("hs", {})
        sun_azimuth = _as_int(hs.get("sun_azimuth", 315), 315)
        sun_elevation = _as_int(hs.get("sun_elevation", 35), 35)
        ve_factor = _as_int(hs.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(hs.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("HS", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("HS")
            log(format_params_line("RVT/HS", {
                "tile": current_tile_name,
                "sun_azimuth": sun_azimuth,
                "sun_elevation": sun_elevation,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_hs(_couches(), out, azimut=sun_azimuth, elevation=sun_elevation,
                                           ve=ve_factor, huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "SUN_AZIMUTH": sun_azimuth,
                    "SUN_ELEVATION": sun_elevation,
                    "VE_FACTOR": ve_factor,
                    "SAVE_AS_8BIT": save_as_8bit,
                }
                run_qgis_algorithm("rvt:rvt_hillshade", params, feedback=feedback, context=context)
        else:
            log(f"HS réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["HS"] = out
        else:
            log(f"HS non créé: {out.name}")

    if products.get("M_HS", False):
        mdh = (rvt_params or {}).get("mdh", {})
        num_directions = _as_int(mdh.get("num_directions", 16), 16)
        if num_directions < 2:
            log(
                f"RVT multi-hillshade: NUM_DIRECTIONS={num_directions} invalide, utilisation de 16 (minimum=2)"
            )
            num_directions = 16
        sun_elevation = _as_int(mdh.get("sun_elevation", 35), 35)
        ve_factor = _as_int(mdh.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(mdh.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("M_HS", current_tile_name, rvt_params)
        if _a_recalculer("M_HS", input_path, out):
            moteur = moteur_indices.par_le_moteur("M_HS")
            log(format_params_line("RVT/M_HS", {
                "tile": current_tile_name,
                "num_directions": num_directions,
                "sun_elevation": sun_elevation,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_mhs(_couches(), out, directions=num_directions, elevation=sun_elevation,
                                            ve=ve_factor, huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "NUM_DIRECTIONS": num_directions,
                    "SAVE_AS_8BIT": save_as_8bit,
                    "SUN_ELEVATION": sun_elevation,
                    "VE_FACTOR": ve_factor,
                }
                run_qgis_algorithm("rvt:rvt_multi_hillshade", params, feedback=feedback, context=context)
            _noter_calcul("M_HS", out, moteur)
        else:
            log(f"M_HS réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["M_HS"] = out
        else:
            log(f"M_HS non créé: {out.name}")

    if products.get("SVF", False):
        # Bornes dures de rvt:rvt_svf (rayon 10-50, directions 8-64, bruit 0-3),
        # appliquées par ``svf_settings`` — la MÊME fonction que le nom de
        # dossier, sans quoi le nom mentirait sur l'image.
        svf = svf_settings(rvt_params)
        brut = (rvt_params or {}).get("svf", {}) or {}
        for cle in ("radius", "num_directions", "noise_remove"):
            if cle in brut and _as_int(brut[cle], svf[cle]) != svf[cle]:
                log(
                    f"RVT SVF: {cle.upper()}={brut[cle]} hors des bornes de "
                    f"rvt:rvt_svf, ramené à {svf[cle]}"
                )
        save_as_8bit = _as_bool(brut.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SVF", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("SVF")
            log(format_params_line("RVT/SVF", {
                "tile": current_tile_name,
                "num_directions": svf["num_directions"],
                "radius": svf["radius"],
                "noise_remove": svf["noise_remove"],
                "ve_factor": svf["ve_factor"],
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_svf(_couches(), out, rayon=svf["radius"], directions=svf["num_directions"],
                                            bruit=svf["noise_remove"], ve=svf["ve_factor"], huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "NOISE_REMOVE": svf["noise_remove"],
                    "NUM_DIRECTIONS": svf["num_directions"],
                    "RADIUS": svf["radius"],
                    "SAVE_AS_8BIT": save_as_8bit,
                    "VE_FACTOR": svf["ve_factor"],
                }
                run_qgis_algorithm("rvt:rvt_svf", params, feedback=feedback, context=context)
        else:
            log(f"SVF réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["SVF"] = out
        else:
            log(f"SVF non créé: {out.name}")

    if products.get("OPNS", False):
        # Un seul produit, un réglage de type (positive / négative), comme
        # l'algorithme rvt:rvt_opns. Les bornes dures de RVT (rayon 10-50,
        # directions 8-64) sont appliquées par ``opns_settings`` — la MÊME
        # fonction que le nom de dossier, sans quoi le nom mentirait.
        opns = opns_settings(rvt_params)
        brut = (rvt_params or {}).get("opns", {}) or {}
        for cle in ("radius", "num_directions", "noise_remove"):
            if cle in brut and _as_int(brut[cle], opns[cle]) != opns[cle]:
                log(
                    f"RVT OPNS: {cle.upper()}={brut[cle]} hors des bornes de "
                    f"rvt:rvt_opns, ramené à {opns[cle]}"
                )
        save_as_8bit = _as_bool(brut.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("OPNS", current_tile_name, rvt_params)
        negative = opns["opns_type"] == 1
        if _a_recalculer("OPNS", input_path, out) if negative else needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("OPNS")
            log(format_params_line("RVT/OPNS", {
                "tile": current_tile_name,
                "type": "negative" if opns["opns_type"] == 1 else "positive",
                "radius": opns["radius"],
                "num_directions": opns["num_directions"],
                "noise_remove": opns["noise_remove"],
                "ve_factor": opns["ve_factor"],
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_opns(_couches(), out, negative=opns["opns_type"] == 1, rayon=opns["radius"],
                                             directions=opns["num_directions"], bruit=opns["noise_remove"],
                                             ve=opns["ve_factor"], huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "NOISE_REMOVE": opns["noise_remove"],
                    "NUM_DIRECTIONS": opns["num_directions"],
                    "OPNS_TYPE": opns["opns_type"],
                    "RADIUS": opns["radius"],
                    "SAVE_AS_8BIT": save_as_8bit,
                    "VE_FACTOR": opns["ve_factor"],
                }
                run_qgis_algorithm("rvt:rvt_opns", params, feedback=feedback, context=context)
            if negative:
                _noter_calcul("OPNS", out, moteur)
        else:
            log(f"OPNS réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["OPNS"] = out
        else:
            log(f"OPNS non créé: {out.name}")

    if products.get("SLO", False):
        slope = (rvt_params or {}).get("slope", {})
        unit = slope_unit(rvt_params)   # 0 degrés, 2 pourcentage (l'ancien 1 est relu comme un pourcentage)
        ve_factor = _as_int(slope.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(slope.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SLO", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("SLO")
            log(format_params_line("RVT/SLO", {
                "tile": current_tile_name,
                "unit": moteur_indices.UNITES_PENTE[unit],
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_slo(_couches(), out, unite=unit, ve=ve_factor, huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "UNIT": unit,
                    "VE_FACTOR": ve_factor,
                    "SAVE_AS_8BIT": save_as_8bit,
                }
                run_qgis_algorithm("rvt:rvt_slope", params, feedback=feedback, context=context)
        else:
            log(f"SLO réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["SLO"] = out
        else:
            log(f"SLO non créé: {out.name}")

    if products.get("LD", False):
        ldo = (rvt_params or {}).get("ldo", {})
        angular_res = _as_int(ldo.get("angular_res", 15), 15)
        min_radius = _as_int(ldo.get("min_radius", 10), 10)
        max_radius = _as_int(ldo.get("max_radius", 20), 20)
        # OBSERVER_H: utiliser _as_float pour gérer la virgule française
        observer_h = _as_float(ldo.get("observer_h", 1.7), 1.7)
        # RVT LD requiert OBSERVER_H > 0
        if observer_h <= 0:
            log(f"LD: OBSERVER_H={observer_h} invalide (RVT requiert > 0), utilisation de 1.7")
            observer_h = 1.7
        ve_factor = _as_int(ldo.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(ldo.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("LD", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("LD")
            log(format_params_line("RVT/LD", {
                "tile": current_tile_name,
                "angular_res": angular_res,
                "min_radius": min_radius,
                "max_radius": max_radius,
                "observer_h_m": observer_h,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_ld(_couches(), out, angular_res=angular_res, min_radius=min_radius,
                                           max_radius=max_radius, observer_h=observer_h, ve=ve_factor,
                                           huit_bits=save_as_8bit)
            else:
                # RVT QGIS: passer un entier si la valeur est entière, sinon float
                observer_h_param = int(observer_h) if observer_h == int(observer_h) else observer_h
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "ANGULAR_RES": angular_res,
                    "MIN_RADIUS": min_radius,
                    "MAX_RADIUS": max_radius,
                    "OBSERVER_H": observer_h_param,
                    "VE_FACTOR": ve_factor,
                    "SAVE_AS_8BIT": save_as_8bit,
                }
                run_qgis_algorithm("rvt:rvt_ld", params, feedback=feedback, context=context)
        else:
            log(f"LD réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["LD"] = out
        else:
            log(f"LD non créé: {out.name}")

    if products.get("SLRM", False):
        slrm = (rvt_params or {}).get("slrm", {})
        radius = _as_int(slrm.get("radius", 20), 20)
        ve_factor = _as_int(slrm.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(slrm.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SLRM", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            moteur = moteur_indices.par_le_moteur("SLRM")
            log(format_params_line("RVT/SLRM", {
                "tile": current_tile_name,
                "radius": radius,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_slrm(_couches(), out, rayon=radius, ve=ve_factor, huit_bits=save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "RADIUS": radius,
                    "VE_FACTOR": ve_factor,
                    "SAVE_AS_8BIT": save_as_8bit,
                }
                run_qgis_algorithm("rvt:rvt_slrm", params, feedback=feedback, context=context)
        else:
            log(f"SLRM réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["SLRM"] = out
        else:
            log(f"SLRM non créé: {out.name}")

    if products.get("VAT", False):
        vat = (rvt_params or {}).get("vat", {})
        terrain_type = _as_int(vat.get("terrain_type", 0), 0)
        blend_combination = _as_int(vat.get("blend_combination", 0), 0)
        save_as_8bit = _as_bool(vat.get("save_as_8bit", True), True)

        # Utiliser get_rvt_temp_filename pour le nom final
        standard_vat_tif = temp_dir / get_rvt_temp_filename("VAT", current_tile_name, rvt_params)
        # Le nom de base pour les outputs intermédiaires de rvt_blender
        vat_output_base = standard_vat_tif.with_suffix("").with_name(standard_vat_tif.stem + "_outputs")

        if needs_refresh(input_path, standard_vat_tif):
            moteur = moteur_indices.par_le_moteur("VAT")
            log(format_params_line("RVT/VAT", {
                "tile": current_tile_name,
                "terrain_type": {0: "general", 1: "flat", 2: "steep"}.get(terrain_type, str(terrain_type)),
                "blend_combination": blend_combination,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.rendre_melange(_couches(), "VAT", blend_combination, terrain_type, vat_output_base,
                                              save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "distance_units": "meters",
                    "area_units": "m2",
                    "ellipsoid": "EPSG:7019",
                    "BLEND_COMBINATION": blend_combination,
                    "TERRAIN_TYPE": terrain_type,
                    "SAVE_AS_8BIT": save_as_8bit,
                    "SAVE_AS_FLOAT": False,
                    "OUTPUT": str(vat_output_base),
                }
                run_qgis_algorithm("rvt:rvt_blender", params, feedback=feedback, context=context)

            rendu = moteur_indices.sortie_rendue(vat_output_base, save_as_8bit)
            if rendu is not None:
                shutil.copy2(str(rendu), str(standard_vat_tif))
        else:
            log(f"VAT réutilisé (cache intermédiaire) : {standard_vat_tif.name}")

        if standard_vat_tif.exists():
            outputs["VAT"] = standard_vat_tif
        else:
            log(f"VAT non créé: {standard_vat_tif.name}")

    if products.get("MSTP", False):
        mstp = (rvt_params or {}).get("mstp", {})
        m = mstp_settings(rvt_params)   # même source que le nom du dossier (rvt_naming)
        local_scale_min, local_scale_max, local_scale_step = (
            m["local_scale_min"], m["local_scale_max"], m["local_scale_step"])
        meso_scale_min, meso_scale_max, meso_scale_step = m["meso_scale_min"], m["meso_scale_max"], m["meso_scale_step"]
        broad_scale_min, broad_scale_max, broad_scale_step = (
            m["broad_scale_min"], m["broad_scale_max"], m["broad_scale_step"])
        lightness, ve_factor = m["lightness"], m["ve_factor"]
        save_as_8bit = _as_bool(mstp.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("MSTP", current_tile_name, rvt_params)
        if _a_recalculer("MSTP", input_path, out):
            moteur = moteur_indices.par_le_moteur("MSTP")
            log(format_params_line("RVT/MSTP", {
                "tile": current_tile_name,
                "local_scale": f"{local_scale_min}-{local_scale_max}/{local_scale_step}",
                "meso_scale": f"{meso_scale_min}-{meso_scale_max}/{meso_scale_step}",
                "broad_scale": f"{broad_scale_min}-{broad_scale_max}/{broad_scale_step}",
                "lightness": lightness,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.calculer_mstp(
                    _couches(), out, local=(local_scale_min, local_scale_max, local_scale_step),
                    meso=(meso_scale_min, meso_scale_max, meso_scale_step),
                    large=(broad_scale_min, broad_scale_max, broad_scale_step), luminosite=lightness,
                    ve=ve_factor, huit_bits=save_as_8bit)
            else:   # rvt:rvt_mstp arrondit la luminosité à l'entier (int(parameterAsDouble))
                params = {
                    "INPUT": str(input_path),
                    "OUTPUT": str(out),
                    "LOCAL_SCALE_MIN": local_scale_min,
                    "LOCAL_SCALE_MAX": local_scale_max,
                    "LOCAL_SCALE_STEP": local_scale_step,
                    "MESO_SCALE_MIN": meso_scale_min,
                    "MESO_SCALE_MAX": meso_scale_max,
                    "MESO_SCALE_STEP": meso_scale_step,
                    "BROAD_SCALE_MIN": broad_scale_min,
                    "BROAD_SCALE_MAX": broad_scale_max,
                    "BROAD_SCALE_STEP": broad_scale_step,
                    "LIGHTNESS": lightness,
                    "VE_FACTOR": ve_factor,
                    "SAVE_AS_8BIT": save_as_8bit,
                }
                run_qgis_algorithm("rvt:rvt_mstp", params, feedback=feedback, context=context)
            _noter_calcul("MSTP", out, moteur)
        else:
            log(f"MSTP réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["MSTP"] = out
        else:
            log(f"MSTP non créé: {out.name}")

    if products.get("CVAT", False):
        cvat = (rvt_params or {}).get("cvat", {})
        save_as_8bit = _as_bool(cvat.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("CVAT", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            log(format_params_line("RVT/CVAT", {
                "tile": current_tile_name,
                "save_as_8bit": save_as_8bit,
                "moteur": "interne",
            }))
            # CVAT n'est pas exposé via Processing : calcul in-process (cf. cvat.py). Le MNT est lu dans son
            # try (_couches, pas _couches()) : MNT ou bibliothèque en défaut → indice sauté, comme avant.
            compute_cvat(
                input_path=input_path,
                output_path=out,
                couches=_couches,
                save_as_8bit=save_as_8bit,
                log=log,
            )
        else:
            log(f"CVAT réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["CVAT"] = out
        else:
            log(f"CVAT non créé: {out.name}")

    # PRISM : combinaison 1 du MÊME algorithme que le VAT
    # (settings/default_blender_combinations.json, dans l'ordre VAT / Prismatic
    # openness / City). Seul l'indice de combinaison change.
    if products.get("PRISM", False):
        prism = (rvt_params or {}).get("prism", {})
        # Le préréglage de terrain s'applique à TOUTE combinaison du blender
        # (rvt_blender.py appelle apply_terrain avant le rendu), pas seulement
        # au VAT : il change l'élévation solaire et les étirements sous-jacents.
        terrain_type = _as_int(prism.get("terrain_type", 0), 0)
        save_as_8bit = _as_bool(prism.get("save_as_8bit", True), True)

        cible = temp_dir / get_rvt_temp_filename("PRISM", current_tile_name, rvt_params)
        base_sorties = cible.with_suffix("").with_name(cible.stem + "_outputs")

        if needs_refresh(input_path, cible):
            moteur = moteur_indices.par_le_moteur("PRISM")
            log(format_params_line("RVT/PRISM", {
                "tile": current_tile_name,
                "blend_combination": 1,
                "terrain_type": {0: "general", 1: "flat", 2: "steep"}.get(
                    terrain_type, str(terrain_type)),
                "save_as_8bit": save_as_8bit,
                "moteur": "interne" if moteur else "processing",
            }))
            if moteur:
                moteur_indices.rendre_melange(_couches(), "PRISM", 1, terrain_type, base_sorties, save_as_8bit)
            else:
                params = {
                    "INPUT": str(input_path),
                    "distance_units": "meters",
                    "area_units": "m2",
                    "ellipsoid": "EPSG:7019",
                    "BLEND_COMBINATION": 1,
                    "TERRAIN_TYPE": terrain_type,
                    "SAVE_AS_8BIT": save_as_8bit,
                    "SAVE_AS_FLOAT": False,
                    "OUTPUT": str(base_sorties),
                }
                run_qgis_algorithm("rvt:rvt_blender", params, feedback=feedback, context=context)

            rendu = moteur_indices.sortie_rendue(base_sorties, save_as_8bit)
            if rendu is not None:
                shutil.copy2(str(rendu), str(cible))
        else:
            log(f"PRISM réutilisé (cache intermédiaire) : {cible.name}")

        if cible.exists():
            outputs["PRISM"] = cible
        else:
            log(f"PRISM non créé: {cible.name}")

    if products.get("CRIM", False):
        crim = (rvt_params or {}).get("crim", {})
        colormap = str(crim.get("colormap", "OrRd")).strip() or "OrRd"
        cut_min = _as_float(crim.get("min_colormap_cut", 0.0), 0.0)
        cut_max = _as_float(crim.get("max_colormap_cut", 1.0), 1.0)
        save_as_8bit = _as_bool(crim.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("CRIM", current_tile_name, rvt_params)
        if needs_refresh(input_path, out):
            log(format_params_line("RVT/CRIM", {
                "tile": current_tile_name,
                "colormap": colormap,
                "colormap_cut": f"{cut_min}-{cut_max}",
                "save_as_8bit": save_as_8bit,
                "moteur": "interne",
            }))
            # CRIM n'est pas exposé via Processing : calcul in-process (cf. crim.py).
            compute_crim(
                input_path=input_path,
                output_path=out,
                couches=_couches,
                colormap=colormap,
                min_colormap_cut=cut_min,
                max_colormap_cut=cut_max,
                save_as_8bit=save_as_8bit,
                log=log,
            )
        else:
            log(f"CRIM réutilisé (cache intermédiaire) : {out.name}")
        if out.exists():
            outputs["CRIM"] = out
        else:
            log(f"CRIM non créé: {out.name}")

    # Rendus 8 bits : séparer NoData réel (→255, étiqueté) et valides saturés
    # (→254) tant que le MNT est sous la main — après coup les deux classes de
    # 255 sont indiscernables (cf. tilespec.reclass_rvt_nodata). Sans cela, un
    # MNT à large NoData (dalle étrangère reprojetée) rend le sans-donnée en
    # blanc opaque, et masquer 255 en aval troue les zones saturées (talus).
    # Idempotent, best-effort, no-op sur un produit float.
    for key, out_path in outputs.items():
        try:
            reclass_rvt_nodata(out_path, input_path)
        except Exception as exc:
            log(f"Reclass NoData {key} ignoré ({exc!r})")

    return IndicesResult(outputs=outputs)
