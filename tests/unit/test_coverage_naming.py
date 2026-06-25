"""Nommage du produit COUVERTURE + cohérence de la liste produits centralisée."""
from __future__ import annotations

from app.run_context import _ALL_PRODUCTS
from pipeline.ign.products.rvt_naming import (
    PRODUCT_ORDER,
    get_rvt_folder_name,
    get_rvt_source_and_dest_filenames,
    get_rvt_temp_filename,
)

TILE = "LHD_FXX_0624_6864"


class TestProductOrder:
    def test_contient_couverture_apres_densite(self):
        i = PRODUCT_ORDER.index
        assert i("COUVERTURE") == i("DENSITE") + 1

    def test_miroir_de_run_context(self):
        # Deux tuples (app-side ne peut pas importer ign.products au top-level) :
        # ce test verrouille leur synchronisation.
        assert tuple(PRODUCT_ORDER) == tuple(_ALL_PRODUCTS)


class TestCouvertureNaming:
    def test_dossier_sans_suffixe(self):
        assert get_rvt_folder_name("COUVERTURE", {}) == "COUVERTURE"
        assert get_rvt_folder_name("COUVERTURE", {"svf": {"radius": 99}}) == "COUVERTURE"

    def test_nom_temporaire(self):
        assert get_rvt_temp_filename("COUVERTURE", TILE, {}) == f"{TILE}_couverture.tif"

    def test_nom_destination(self):
        _src, dest = get_rvt_source_and_dest_filenames("COUVERTURE", TILE, "0624", "6864", {})
        assert dest == "LHD_FXX_0624_6864_couverture_A_LAMB93.tif"

    def test_nom_destination_avec_suffixe_unicite(self):
        _src, dest = get_rvt_source_and_dest_filenames(
            "COUVERTURE", TILE, "0624", "6864", {}, name_suffix="_abc123"
        )
        assert dest == "LHD_FXX_0624_6864_couverture_A_LAMB93_abc123.tif"
