"""Noms des MNT et indices publiés selon le CRS du run (run réel 2026-10-31 : MNT allemands EPSG:25832 publiés
sous ``LHD_FXX_…_LAMB93_IGN69``). EPSG:2154 : noms d'avant à l'octet près (LiDAR HD IGN, reprise, caches) ;
autre CRS : ``EPSG_<code>_{x}_{y}_…``, sans Lambert-93 ni IGN69 — deux jetons de préfixe, de sorte que
x et y restent ``parts[2]``/``parts[3]`` pour tous les lecteurs (``extract_xy_*``, halo, conversion)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.indices_metres import anneau_px_du_nom
from pipeline.coords import extract_xy_from_filename, extract_xy_from_tile_name
from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames, prefixe_dalle

LD = {"ldo": {"angular_res": 15, "min_radius": 5, "max_radius": 10, "observer_h": 1.7, "ve_factor": 1}}


def _dest(produit, crs=None, **k):
    kw = {} if crs is None else {"crs": crs}
    return get_rvt_source_and_dest_filenames(produit, "LHD_FXX_0455_6057", "0455", "6057", LD, **k, **kw)[1]


@pytest.mark.parametrize("crs", [None, "EPSG:2154", "epsg:2154"])
@pytest.mark.parametrize("produit,attendu", [
    ("MNT", "LHD_FXX_0455_6057_MNT_A_0M50_LAMB93_IGN69.tif"),
    ("LD", "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93.tif"),
    ("DENSITE", "LHD_FXX_0455_6057_densite_A_LAMB93.tif"),
    ("COUVERTURE", "LHD_FXX_0455_6057_couverture_A_LAMB93.tif"),
    ("CVAT", "LHD_FXX_0455_6057_CVAT_A_LAMB93.tif"),
])
def test_lambert93_noms_d_avant_a_l_octet(crs, produit, attendu):
    assert _dest(produit, crs) == attendu


def test_lambert93_suffixe_et_resolution_inchanges():
    assert _dest("MNT", "EPSG:2154", name_suffix="_mnt_7", mnt_resolution=1.0) == \
        "LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69_mnt_7.tif"
    assert get_rvt_source_and_dest_filenames("M_HS", "T", "0455", "6057", {}, name_suffix="_u")[1].startswith(
        "LHD_FXX_0455_6057_M-HS") and _dest("CVAT", name_suffix="_u").endswith("_A_LAMB93_u.tif")


@pytest.mark.parametrize("produit,attendu", [
    ("MNT", "EPSG_25832_0455_6057_MNT_A_1M00_mnt_1066.tif"),
    ("LD", "EPSG_25832_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_mnt_1066.tif"),
    ("DENSITE", "EPSG_25832_0455_6057_densite_A_mnt_1066.tif"),
    ("CVAT", "EPSG_25832_0455_6057_CVAT_A_mnt_1066.tif"),
])
def test_autre_crs_nom_honnete(produit, attendu):
    nom = _dest(produit, "EPSG:25832", name_suffix="_mnt_1066", mnt_resolution=1.0)
    assert nom == attendu
    assert not any(j in nom for j in ("LHD", "FXX", "LAMB93", "IGN69"))
    xy = extract_xy_from_filename(nom)                       # lecteurs aval : x, y = parts[2], parts[3]
    assert (xy.x_km, xy.y_km) == (455, 6057)


def test_autre_crs_anneau_toujours_lu_dans_le_nom():
    assert anneau_px_du_nom(_dest("LD", "EPSG:25832")) == (5, 10)


def test_prefixe_sans_code_epsg():
    assert prefixe_dalle("0455", "6057", 'PROJCS["local",...]') == "CRS_PROJETE_0455_6057"
    assert extract_xy_from_tile_name(prefixe_dalle("0455", "6057", "EPSG:27700") + "_EXT_x") == ("0455", "6057")


def test_nom_de_dalle_interne_existing_mnt():
    from pipeline.modes.existing_mnt import _large_tile_name_for
    b = (455300.0, 6056000.0, 455651.0, 6056351.0)
    assert _large_tile_name_for(Path("mnt_1066.tif"), b) == "LHD_FXX_0455_6056_EXT_mnt_1066"
    assert _large_tile_name_for(Path("mnt_1066.tif"), b, crs="EPSG:2154") == "LHD_FXX_0455_6056_EXT_mnt_1066"
    assert _large_tile_name_for(Path("mnt_1066.tif"), b, crs="EPSG:25832") == "EPSG_25832_0455_6056_EXT_mnt_1066"


def _tif(path, origin, taille_m, crs):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin
    with rasterio.open(str(path), "w", driver="GTiff", width=10, height=10, count=1, dtype="uint8", crs=crs,
                       transform=from_origin(origin[0], origin[1], taille_m / 10, taille_m / 10)) as ds:
        ds.write(np.zeros((1, 10, 10), dtype="uint8"))


def test_indice_fourni_1_km_renomme_selon_le_crs(tmp_path):
    from pipeline.modes.existing_rvt import _normalized_rvt_name
    _tif(tmp_path / "ld.tif", (455000.0, 6057000.0), 1000.0, "EPSG:25832")
    assert _normalized_rvt_name(tif_path=tmp_path / "ld.tif", target_rvt="LD") == "LHD_FXX_0455_6057_LD_A_LAMB93.tif"
    assert _normalized_rvt_name(tif_path=tmp_path / "ld.tif", target_rvt="LD", crs="EPSG:2154") == \
        "LHD_FXX_0455_6057_LD_A_LAMB93.tif"
    assert _normalized_rvt_name(tif_path=tmp_path / "ld.tif", target_rvt="LD", crs="EPSG:25832") == \
        "EPSG_25832_0455_6057_LD_A.tif"


def test_republication_retire_le_nom_lambert93_d_un_run_precedent(tmp_path, monkeypatch):
    """Relance dans le même dossier après la correction : l'ancien ``LHD_FXX_…_LAMB93`` (TIF, pyramides, PNG)
    doit partir, sinon la mosaïque et la détection (qui lit tout le dossier) compteraient chaque dalle deux fois."""
    import pipeline.ign.products.results as results_mod
    from pipeline.output_paths import indice_png_dir, indice_tif_dir
    from pipeline.ign.products.rvt_naming import get_rvt_folder_name
    monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **k: True)

    temp, out = tmp_path / "temp", tmp_path / "out"
    temp.mkdir()
    tuile = "EPSG_25832_0455_6057_EXT_mnt_1066"
    _src, dest = get_rvt_source_and_dest_filenames("LD", tuile, "0455", "6057", LD, name_suffix="_u", crs="EPSG:25832")
    _tif(temp / dest, (455000.0, 6057000.0), 300.0, "EPSG:25832")
    tif_dir = indice_tif_dir(out, get_rvt_folder_name("LD", LD))
    png_dir = indice_png_dir(out, get_rvt_folder_name("LD", LD))
    tif_dir.mkdir(parents=True)
    png_dir.mkdir(parents=True)
    ancien = "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_u"
    for f in (tif_dir / f"{ancien}.tif", tif_dir / f"{ancien}.tif.ovr", png_dir / f"{ancien}.png",
              png_dir / f"{ancien}.pgw", tif_dir / "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_v.tif"):
        f.write_bytes(b"x")

    results_mod.copy_final_products_to_results(
        temp_dir=temp, output_dir=out, current_tile_name=tuile, products={"LD": True}, output_structure={},
        output_formats={"tif": True}, rvt_params=LD, name_suffix="_u", crs="EPSG:25832")

    assert sorted(p.name for p in tif_dir.iterdir()) == [
        "EPSG_25832_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_u.tif",
        "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_v.tif",   # autre dalle : jamais touchée
    ]
    assert list(png_dir.iterdir()) == []


def test_mnt_autre_resolution_et_ancien_nom_retires(tmp_path, monkeypatch):
    import pipeline.ign.products.results as results_mod
    from pipeline.output_paths import indice_tif_dir
    monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **k: True)
    temp, out = tmp_path / "temp", tmp_path / "out"
    temp.mkdir()
    tuile = "EPSG_25832_0455_6057_EXT_mnt_1066"
    _src, dest = get_rvt_source_and_dest_filenames("MNT", tuile, "0455", "6057", {}, name_suffix="_u",
                                                   mnt_resolution=1.0, crs="EPSG:25832")
    _tif(temp / dest, (455000.0, 6057000.0), 300.0, "EPSG:25832")
    d = indice_tif_dir(out, "MNT")
    d.mkdir(parents=True)
    for n in ("LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69_u.tif", "EPSG_25832_0455_6057_MNT_A_0M50_u.tif"):
        (d / n).write_bytes(b"x")
    results_mod.copy_final_products_to_results(
        temp_dir=temp, output_dir=out, current_tile_name=tuile, products={"MNT": True}, output_structure={},
        output_formats={"tif": True}, rvt_params={}, name_suffix="_u", mnt_resolution=1.0, crs="EPSG:25832")
    assert [p.name for p in d.iterdir()] == ["EPSG_25832_0455_6057_MNT_A_1M00_u.tif"]


def test_rapport_anonymise_les_noms_hors_lambert93():
    """Le code EPSG (5 chiffres) ne doit pas absorber le x : « EPSG_2…_6057 » laissait le y au km dans le rapport."""
    from app.services.rapport_run import anonymiser
    assert anonymiser("EPSG_25832_0455_6057_LD_A_mnt_1066.png : boom") == "EPSG_25832_…_LD_A_mnt_1066.png : boom"
    assert anonymiser("LHD_FXX_0455_6057_LD_A_LAMB93.tif") == "LHD_FXX_…_LD_A_LAMB93.tif"
