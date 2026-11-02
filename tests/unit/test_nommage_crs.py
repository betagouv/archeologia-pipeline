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


ANCIEN_LD = "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_u"
NOUVEAU_LD = "EPSG_25832_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_u"


def _publier(tmp_path, monkeypatch, produit, rvt, *, sources=(), prepare=lambda tif_dir, png_dir: None,
             log=lambda _: None, **k):
    """Publie une dalle EPSG:25832 (uid ``_u``) après ``prepare`` ; rend (dossier TIF, dossier PNG, dossier sortie)."""
    import pipeline.ign.products.results as results_mod
    from pipeline.ign.products.rvt_naming import get_rvt_folder_name
    from pipeline.output_paths import indice_png_dir, indice_tif_dir
    monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **kw: True)
    temp, out = tmp_path / "temp", tmp_path / "out"
    temp.mkdir()
    tuile = "EPSG_25832_0455_6057_EXT_mnt_1066"
    _src, dest = get_rvt_source_and_dest_filenames(produit, tuile, "0455", "6057", rvt, name_suffix="_u",
                                                   crs="EPSG:25832", **k)
    _tif(temp / dest, (455000.0, 6057000.0), 300.0, "EPSG:25832")
    tif_dir = indice_tif_dir(out, get_rvt_folder_name(produit, rvt))
    png_dir = indice_png_dir(out, get_rvt_folder_name(produit, rvt))
    tif_dir.mkdir(parents=True)
    png_dir.mkdir(parents=True)
    prepare(tif_dir, png_dir)
    results_mod.copy_final_products_to_results(
        temp_dir=temp, output_dir=out, current_tile_name=tuile, products={produit: True}, output_structure={},
        output_formats={"tif": True}, rvt_params=rvt, name_suffix="_u", crs="EPSG:25832",
        sources=tuple(s(tif_dir) for s in sources), log=log, **k)
    return tif_dir, png_dir, out


def test_republication_retire_le_nom_lambert93_d_un_run_precedent(tmp_path, monkeypatch):
    """Relance dans le même dossier après la correction : l'ancien ``LHD_FXX_…_LAMB93`` (TIF, pyramides, PNG, world
    file, détections en cache) doit partir, sinon mosaïque, détection et conversion compteraient la dalle deux fois."""
    def prepare(tif_dir, png_dir):
        _tif(tif_dir / f"{ANCIEN_LD}.tif", (455000.0, 6057000.0), 300.0, "EPSG:25832")
        for f in (tif_dir / f"{ANCIEN_LD}.tif.ovr", png_dir / f"{ANCIEN_LD}.png", png_dir / f"{ANCIEN_LD}.pgw"):
            f.write_bytes(b"x")
        raw = tif_dir.parents[3] / "technique" / "detection" / "enclos_fr_seg_ld_v1" / "raw_detections"
        raw.mkdir(parents=True)
        for f in (f"{ANCIEN_LD}.txt", f"{ANCIEN_LD}.json", "autre_dalle.txt"):
            (raw / f).write_bytes(b"x")
        _tif(tif_dir / "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_v.tif",
             (455000.0, 6057000.0), 300.0, "EPSG:25832")

    tif_dir, png_dir, out = _publier(tmp_path, monkeypatch, "LD", LD, prepare=prepare)
    assert sorted(p.name for p in tif_dir.iterdir()) == [
        f"{NOUVEAU_LD}.tif", "LHD_FXX_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_LAMB93_v.tif"]   # autre dalle gardée
    assert list(png_dir.iterdir()) == []
    raw = out / "technique" / "detection" / "enclos_fr_seg_ld_v1" / "raw_detections"
    assert [p.name for p in raw.iterdir()] == ["autre_dalle.txt"]


def test_jamais_un_fichier_du_dossier_source(tmp_path, monkeypatch):
    """Audit 2026-11-01 (D2) : MNT source = ``livrable/indices/MNT/tif`` d'un run antérieur, publié dans ce même
    dossier. Le préflight promet que la source n'est jamais supprimée : ni l'ancien nom, ni une autre résolution."""
    anciens = ("LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69_u.tif", "EPSG_25832_0455_6057_MNT_A_0M50_u.tif")

    def prepare(tif_dir, png_dir):
        for n in anciens:
            _tif(tif_dir / n, (455000.0, 6057000.0), 300.0, "EPSG:25832")

    tif_dir, _png, _out = _publier(tmp_path, monkeypatch, "MNT", {}, mnt_resolution=1.0, prepare=prepare,
                                   sources=(lambda d: d,))
    assert sorted(p.name for p in tif_dir.iterdir()) == sorted([*anciens, "EPSG_25832_0455_6057_MNT_A_1M00_u.tif"])


def test_jamais_la_dalle_d_un_vrai_run_lambert93(tmp_path, monkeypatch):
    """Audit 2026-11-01 (D2) : dossier de sortie partagé avec un vrai run français aux mêmes km : sa dalle est en
    EPSG:2154, pas dans le CRS du run, elle reste (comme son PNG)."""
    def prepare(tif_dir, png_dir):
        _tif(tif_dir / f"{ANCIEN_LD}.tif", (455000.0, 6057000.0), 1000.0, "EPSG:2154")
        (png_dir / f"{ANCIEN_LD}.png").write_bytes(b"x")

    tif_dir, png_dir, _out = _publier(tmp_path, monkeypatch, "LD", LD, prepare=prepare)
    assert sorted(p.name for p in tif_dir.iterdir()) == sorted([f"{ANCIEN_LD}.tif", f"{NOUVEAU_LD}.tif"])
    assert [p.name for p in png_dir.iterdir()] == [f"{ANCIEN_LD}.png"]


def test_mnt_autre_resolution_et_ancien_nom_retires(tmp_path, monkeypatch):
    def prepare(tif_dir, png_dir):
        for n in ("LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69_u.tif", "EPSG_25832_0455_6057_MNT_A_0M50_u.tif"):
            _tif(tif_dir / n, (455000.0, 6057000.0), 300.0, "EPSG:25832")
        _tif(tif_dir / "EPSG_25832_0455_6057_MNT_A_0M50_x_u.tif", (455000.0, 6057000.0), 300.0, "EPSG:25832")

    tif_dir, _png, _out = _publier(tmp_path, monkeypatch, "MNT", {}, mnt_resolution=1.0, prepare=prepare)
    assert sorted(p.name for p in tif_dir.iterdir()) == [
        "EPSG_25832_0455_6057_MNT_A_0M50_x_u.tif",          # autre dalle (uid « x_u ») : jamais touchée
        "EPSG_25832_0455_6057_MNT_A_1M00_u.tif"]


def test_lambert93_autre_resolution_toujours_retiree(tmp_path, monkeypatch):
    """Non-régression : en EPSG:2154, le MNT de la même dalle à une autre résolution part comme avant."""
    import pipeline.ign.products.results as results_mod
    from pipeline.output_paths import indice_tif_dir
    monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **kw: True)
    temp, out = tmp_path / "temp", tmp_path / "out"
    temp.mkdir()
    _src, dest = get_rvt_source_and_dest_filenames("MNT", "LHD_FXX_0455_6057", "0455", "6057", {}, mnt_resolution=1.0)
    _tif(temp / dest, (455000.0, 6057000.0), 1000.0, "EPSG:2154")
    d = indice_tif_dir(out, "MNT")
    d.mkdir(parents=True)
    _tif(d / "LHD_FXX_0455_6057_MNT_A_0M50_LAMB93_IGN69.tif", (455000.0, 6057000.0), 1000.0, "EPSG:2154")
    results_mod.copy_final_products_to_results(
        temp_dir=temp, output_dir=out, current_tile_name="LHD_FXX_0455_6057", products={"MNT": True},
        output_structure={}, output_formats={"tif": True}, rvt_params={}, mnt_resolution=1.0)
    assert [p.name for p in d.iterdir()] == ["LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69.tif"]


def test_existing_rvt_retire_l_ancien_nom_et_garde_la_source(tmp_path):
    """Audit 2026-11-01 (D4) : indice fourni de 1 km, renommé ``EPSG_25832_…`` ; l'ancien ``LHD_FXX_…_LAMB93`` du
    même dossier de sortie part (sinon la détection, qui lit tout le dossier, le traiterait deux fois)."""
    from pipeline.modes.existing_rvt import run_existing_rvt
    from pipeline.output_paths import indice_tif_dir
    src = tmp_path / "src"
    src.mkdir()
    _tif(src / "ld.tif", (455000.0, 6057000.0), 1000.0, "EPSG:25832")
    out = tmp_path / "out"
    d = indice_tif_dir(out, "RVT")
    d.mkdir(parents=True)
    _tif(d / "LHD_FXX_0455_6057_LD_A_LAMB93.tif", (455000.0, 6057000.0), 1000.0, "EPSG:25832")
    run_existing_rvt(existing_rvt_dir=src, output_dir=out, cv_config={"enabled": False, "target_rvt": "LD"},
                     output_structure={}, indices_folder_name="RVT", crs="EPSG:25832")
    assert [p.name for p in d.iterdir()] == ["EPSG_25832_0455_6057_LD_A.tif"]
    assert [p.name for p in src.iterdir()] == ["ld.tif"]


def test_rapport_anonymise_les_noms_hors_lambert93():
    """Le code EPSG (5 chiffres) ne doit pas absorber le x : « EPSG_2…_6057 » laissait le y au km dans le rapport."""
    from app.services.rapport_run import anonymiser
    assert anonymiser("EPSG_25832_0455_6057_LD_A_mnt_1066.png : boom") == "EPSG_25832_…_LD_A_mnt_1066.png : boom"
    assert anonymiser("LHD_FXX_0455_6057_LD_A_LAMB93.tif") == "LHD_FXX_…_LD_A_LAMB93.tif"


@pytest.mark.parametrize("message,attendu", [
    ("EPSG_3006_0455_6057_LD_A_u.png : x", "EPSG_3006_…_LD_A_u.png : x"),       # code EPSG à 4 chiffres
    ("EPSG_2157_0455_6057_LD_A.tif", "EPSG_2157_…_LD_A.tif"),
    ("EPSG_4647_32500_5800_MNT_A_1M00.tif", "EPSG_4647_…_MNT_A_1M00.tif"),       # coordonnées à 5 chiffres
    ("CRS_PROJETE_0455_6057_LD_A.tif", "CRS_PROJETE_…_LD_A.tif"),
    ("EPSG_32631_-012_5400_LD_A.tif", "EPSG_32631_…_LD_A.tif"),
    ("Dalle 0986-6859 abandonnée", "Dalle … abandonnée"),
])
def test_rapport_anonymise_tout_prefixe_de_dalle(message, attendu):
    """Audit 2026-11-01 : le motif générique effaçait un code EPSG à 4 chiffres et laissait fuir le y."""
    from app.services.rapport_run import anonymiser
    assert anonymiser(message) == attendu


ANCIENNE_TUILE = "LHD_FXX_0455_6057_EXT_mnt_1066"


def _orphelins(tif_dir, crs="EPSG:25832"):
    """État d'un dossier relancé après la 0.19.0 : publication Lambert-93 déjà retirée, mais intermédiaires et
    images annotées sous l'ancien nom encore là (run de_sh du 2026-11-01 : 3 370 fichiers, ≈ 615 Mo)."""
    temp, out = tif_dir.parents[4] / "temp", tif_dir.parents[3]
    for n in (f"{ANCIENNE_TUILE}_MNT.tif", f"{ANCIENNE_TUILE}_LD_A15_Rmin5_Rmax10_H1p7_V1.tif", f"{ANCIEN_LD}.tif",
              "LHD_FXX_0455_6057_MNT_A_1M00_LAMB93_IGN69_u.tif"):
        _tif(temp / n, (455000.0, 6057000.0), 300.0, crs)
    (temp / f"{ANCIENNE_TUILE}_VAT_T0_B0.tif.calcul").write_bytes(b"x")
    _tif(temp / "LHD_FXX_0455_6057_EXT_mnt_10660_MNT.tif", (455000.0, 6057000.0), 300.0, crs)   # autre dalle
    for modele in ("enclos_fr_seg_ld_v1", "lineaires_seg_v3_1"):
        ann = out / "technique" / "detection" / modele / "annotated_images"
        ann.mkdir(parents=True)
        for f in (f"{ANCIEN_LD}_detections.png", f"{ANCIEN_LD}_detections.pgw", "legend.png"):
            (ann / f).write_bytes(b"x")
    return temp, out


def test_relance_retire_intermediaires_et_images_annotees_lambert93(tmp_path, monkeypatch):
    journal: list = []
    etat: dict = {}

    def prepare(tif_dir, png_dir):
        etat["temp"], etat["out"] = _orphelins(tif_dir)

    _publier(tmp_path, monkeypatch, "LD", LD, prepare=prepare, log=journal.append)
    restants = sorted(p.name for p in etat["temp"].iterdir())
    assert restants == ["EPSG_25832_0455_6057_LD_A15_Rmin5_Rmax10_H1p7_V1_A_u.tif",     # l'entrée du run
                        "LHD_FXX_0455_6057_EXT_mnt_10660_MNT.tif"]                       # autre dalle : gardée
    for modele in ("enclos_fr_seg_ld_v1", "lineaires_seg_v3_1"):
        ann = etat["out"] / "technique" / "detection" / modele / "annotated_images"
        assert [p.name for p in ann.iterdir()] == ["legend.png"]
    # Une ligne par publication et une par dalle d'intermédiaires, jamais une par fichier (2 870 lignes pour
    # 1 640 noms : le même nom de cache existe dans chaque dossier de modèle).
    retraits = [m for m in journal if "retiré" in m]
    assert len(retraits) == 2 and "(4 fichiers)" in retraits[0] and retraits[1].endswith(": 5 fichiers")


def test_intermediaires_d_un_vrai_run_lambert93_gardes(tmp_path, monkeypatch):
    etat: dict = {}

    def prepare(tif_dir, png_dir):
        etat["temp"], etat["out"] = _orphelins(tif_dir, crs="EPSG:2154")

    _publier(tmp_path, monkeypatch, "LD", LD, prepare=prepare)
    assert len(list(etat["temp"].iterdir())) == 7       # rien de Lambert-93 n'a bougé (+ l'entrée du run)
    ann = etat["out"] / "technique" / "detection" / "enclos_fr_seg_ld_v1" / "annotated_images"
    assert len(list(ann.iterdir())) == 3
