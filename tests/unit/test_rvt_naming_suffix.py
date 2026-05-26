from __future__ import annotations

from pathlib import Path

from pipeline.coords import extract_xy_from_tile_name
from pipeline.ign.products.rvt_naming import (
    get_rvt_folder_name,
    get_rvt_param_suffix,
    get_rvt_source_and_dest_filenames,
)
from pipeline.output_paths import indice_tif_dir

TILE = "LHD_FXX_0624_6864"
X, Y = "0624", "6864"
LD_PARAMS = {"ldo": {"angular_res": 15, "min_radius": 10, "max_radius": 20,
                     "observer_h": 1.7, "ve_factor": 1}}
HS_PARAMS = {"hs": {"sun_azimuth": 315, "sun_elevation": 35, "ve_factor": 1}}
ALL_PRODUCTS = ["MNT", "DENSITE", "HS", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]


def _dest(product, params=None, **kw):
    _src, dest = get_rvt_source_and_dest_filenames(product, TILE, X, Y, params or {}, **kw)
    return dest


class TestRegressionDefaultUnchanged:
    """name_suffix='' (défaut) ⇒ noms strictement identiques à l'historique."""

    def test_mnt(self):
        assert _dest("MNT") == "LHD_FXX_0624_6864_MNT_A_0M50_LAMB93_IGN69.tif"

    def test_densite(self):
        assert _dest("DENSITE") == "LHD_FXX_0624_6864_densite_A_LAMB93.tif"

    def test_ld_with_params(self):
        # Format observé sur les vraies sorties du run_78.
        assert _dest("LD", LD_PARAMS) == \
            "LHD_FXX_0624_6864_LD_A15_Rmin10_Rmax20_H1p7_V1_A_LAMB93.tif"

    def test_hs_with_params(self):
        # Hillshade simple : azimut + élévation + VE (clé "HS", pas de cas spécial).
        assert _dest("HS", HS_PARAMS) == \
            "LHD_FXX_0624_6864_HS_Az315_E35_V1_A_LAMB93.tif"

    def test_explicit_empty_suffix_equals_default(self):
        for product in ["MNT", "DENSITE", "HS", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]:
            assert _dest(product, LD_PARAMS) == _dest(product, LD_PARAMS, name_suffix="")


class TestUniquenessSuffix:
    def test_suffix_inserted_before_extension(self):
        assert _dest("MNT", name_suffix="_abc") == \
            "LHD_FXX_0624_6864_MNT_A_0M50_LAMB93_IGN69_abc.tif"
        assert _dest("LD", LD_PARAMS, name_suffix="_abc") == \
            "LHD_FXX_0624_6864_LD_A15_Rmin10_Rmax20_H1p7_V1_A_LAMB93_abc.tif"

    def test_distinct_suffixes_give_distinct_names(self):
        a = _dest("LD", LD_PARAMS, name_suffix="_tileA")
        b = _dest("LD", LD_PARAMS, name_suffix="_tileB")
        assert a != b

    def test_xy_still_parseable_after_suffix(self):
        dest = _dest("LD", LD_PARAMS, name_suffix="_05_MNT_6240_68640")
        # Downstream still reads the cosmetic km coords at parts[2]/parts[3].
        x, y = extract_xy_from_tile_name(dest.replace(".tif", ""))
        assert (x, y) == ("0624", "6864")


class TestRvtFolderName:
    """Nom de dossier d'indice = code produit + suffixe de paramètres RVT."""

    def test_mnt_and_densite_have_no_suffix(self):
        # Pas de paramètres → code brut, même si on passe un dict de params.
        assert get_rvt_folder_name("MNT", {}) == "MNT"
        assert get_rvt_folder_name("DENSITE", {}) == "DENSITE"
        assert get_rvt_folder_name("MNT", LD_PARAMS) == "MNT"
        assert get_rvt_folder_name("DENSITE", LD_PARAMS) == "DENSITE"

    def test_svf_default_suffix(self):
        # dict vide ⇒ valeurs par défaut (pas de suffixe vide).
        assert get_rvt_folder_name("SVF", {}) == "SVF_R10_D16_V1_N0"

    def test_ld_with_params(self):
        assert get_rvt_folder_name("LD", LD_PARAMS) == \
            "LD_A15_Rmin10_Rmax20_H1p7_V1"

    def test_hs_with_params(self):
        assert get_rvt_folder_name("HS", HS_PARAMS) == "HS_Az315_E35_V1"

    def test_folder_equals_product_plus_suffix(self):
        for product in ALL_PRODUCTS:
            assert get_rvt_folder_name(product, LD_PARAMS) == \
                product + get_rvt_param_suffix(product, LD_PARAMS)

    def test_distinct_params_give_distinct_folders(self):
        svf_r10 = get_rvt_folder_name("SVF", {"svf": {"radius": 10}})
        svf_r20 = get_rvt_folder_name("SVF", {"svf": {"radius": 20}})
        assert svf_r10 != svf_r20
        assert svf_r10 == "SVF_R10_D16_V1_N0"
        assert svf_r20 == "SVF_R20_D16_V1_N0"


class TestFolderPathSymmetry:
    """Le dossier dérivé doit correspondre au chemin résolu côté consommation CV.

    On teste au niveau des helpers purs (sans appeler resolve_rvt_tif_dir, qui
    importe le paquet QGIS ign.products en différé).
    """

    def test_tif_dir_matches_folder_name(self):
        out = Path("/tmp/out")
        folder = get_rvt_folder_name("LD", LD_PARAMS)
        assert indice_tif_dir(out, folder) == \
            out / "indices" / "LD_A15_Rmin10_Rmax20_H1p7_V1" / "tif"

    def test_mnt_dir_unchanged(self):
        out = Path("/tmp/out")
        assert indice_tif_dir(out, get_rvt_folder_name("MNT", LD_PARAMS)) == \
            out / "indices" / "MNT" / "tif"
