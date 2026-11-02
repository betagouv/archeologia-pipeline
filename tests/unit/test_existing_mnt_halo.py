"""Halo inter-dalles des INDICES en mode « MNT fournis » (audit Fable du 2026-11-02, P2).

Une dalle STANDARD (1 km alignée) qui a des voisines dans le dossier est calculée sur « dalle + 50 m » découpés
dans un VRT des MNT fournis (``existing_mnt._materialiser_mnt_avec_marge``), puis rognée à sa cellule : plus de
couture des indices aux joints. Côté sans voisine, pas de marge (une marge sans donnée abaisserait le LD) ; une
dalle isolée garde la copie d'avant, à l'octet. Les TIF non rognés d'``intermediaires/`` servent ensuite au halo
de détection (``halo_source_dir``, ``resolve_uncropped_tif``).

GDAL et gdalwarp exigés : sautés dans le venv de développement, joués sous le Python de QGIS
(``python-qgis.bat run_tests.py tests/unit/test_existing_mnt_halo.py``, pytest sur PYTHONPATH).
"""
from __future__ import annotations

import filecmp
import json
import os
import shutil
from pathlib import Path

import pytest

gdal = pytest.importorskip("osgeo.gdal")
if shutil.which("gdalwarp") is None:
    pytest.skip("gdalwarp absent du PATH", allow_module_level=True)

import numpy as np  # noqa: E402

import pipeline.modes.existing_mnt as em  # noqa: E402
from pipeline.coords import extract_xy_from_tile_name, get_raster_bounds  # noqa: E402
from pipeline.ign.products.results import needs_refresh  # noqa: E402
from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames  # noqa: E402
from pipeline.output_paths import intermediaires_dir  # noqa: E402

PAS = 10.0
N = int(1000 / PAS)
NW = (700000.0, 6601000.0)   # coin NW du bloc 2 × 2 km


def _valeur(x, y):
    """Rampe continue sur tout le bloc : une couture se verrait comme un saut."""
    return (x - 699000.0) / 10.0 + (y - 6598000.0) / 100.0


def _dalle(dossier: Path, x0: float, ymax: float, *, nodata=-9999.0) -> Path:
    from osgeo import osr

    p = dossier / f"dalle_{int(x0)}_{int(ymax)}.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(p), N, N, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0, PAS, 0.0, ymax, 0.0, -PAS))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    xs = x0 + PAS * (np.arange(N) + 0.5)
    ys = ymax - PAS * (np.arange(N) + 0.5)
    b = ds.GetRasterBand(1)
    b.WriteArray(_valeur(xs[None, :], ys[:, None]).astype("float32"))
    if nodata is not None:
        b.SetNoDataValue(nodata)
    ds = None
    return p


def _bloc(dossier: Path, *, nodata=-9999.0) -> dict:
    dossier.mkdir(parents=True, exist_ok=True)
    return {(i, j): _dalle(dossier, NW[0] + 1000 * i, NW[1] - 1000 * j, nodata=nodata)
            for i in (0, 1) for j in (0, 1)}


def _faux_indices(temp_dir, current_tile_name, products, rvt_params, log, feedback=None):
    """create_visualization_products remplacé : chaque indice = copie du MNT de travail (fraîcheur respectée)."""
    x, y = extract_xy_from_tile_name(current_tile_name)
    mnt = temp_dir / f"{current_tile_name}_MNT.tif"
    for p, actif in products.items():
        if not actif or p == "MNT":
            continue
        src, _ = get_rvt_source_and_dest_filenames(p, current_tile_name, x, y, rvt_params)
        if needs_refresh(mnt, temp_dir / src):
            shutil.copy(mnt, temp_dir / src)


def _lancer(mnt_dir: Path, out: Path, monkeypatch, journal=None):
    monkeypatch.setattr(em, "create_visualization_products", _faux_indices)
    lignes = journal if journal is not None else []
    return em.run_existing_mnt(
        existing_mnt_dir=mnt_dir, output_dir=out, products={"MNT": True, "LD": True},
        output_structure={}, output_formats={}, rvt_params={}, log=lignes.append, crs="EPSG:2154",
    ), lignes


def _inter(out: Path, nom: str) -> Path:
    return intermediaires_dir(out) / nom


def _publie(out: Path, produit: str, cellule: str) -> Path:
    (p,) = [f for f in (out / "livrable" / "indices").glob(f"{produit}*/tif/*{cellule}*.tif")]
    return p


def _lire(p: Path):
    ds = gdal.Open(str(p))
    b = ds.GetRasterBand(1)
    return b.ReadAsArray(), b.GetNoDataValue()


def test_marge_cote_voisin_seulement_et_publie_a_la_cellule(tmp_path, monkeypatch):
    dalles = _bloc(tmp_path / "mnt")
    out = tmp_path / "sortie"
    res, journal = _lancer(tmp_path / "mnt", out, monkeypatch)
    assert res.total == 4
    # Dalle NW : voisines à l'est et au sud seulement → cellule + 50 m de ces deux côtés, cellule ailleurs.
    ext = _inter(out, "MNT_EXT_0700_6601_MNT.tif")
    assert get_raster_bounds(ext) == pytest.approx((700000.0, 6599950.0, 701050.0, 6601000.0))
    side = json.loads(_inter(out, "MNT_EXT_0700_6601_MNT.inputs.json").read_text(encoding="utf-8"))
    assert side == {"marge_m": 50.0, "voisins": sorted(p.name for p in dalles.values())}
    # La marge est la vraie donnée voisine : la rampe continue, sans saut au joint.
    arr, nodata = _lire(ext)
    attendu_x = 701000.0 + PAS / 2   # première colonne de la marge est
    assert arr[0, N] == pytest.approx(_valeur(attendu_x, 6601000.0 - PAS / 2))
    assert nodata == -9999.0 and not np.any(arr == -9999.0)
    # Publiés : LD et MNT rognés à la cellule exacte, MNT publié = MNT fourni.
    for produit in ("LD", "MNT"):
        assert get_raster_bounds(_publie(out, produit, "0700_6601")) == pytest.approx(
            (700000.0, 6600000.0, 701000.0, 6601000.0))
    # MNT publié = celui de l'ancien chemin (dalle copiée seule) : même rognage, même LERC (1 cm) — à l'identique.
    seule = tmp_path / "seule"
    seule.mkdir()
    shutil.copy2(dalles[(0, 0)], seule / dalles[(0, 0)].name)
    _lancer(seule, tmp_path / "sortie_seule", monkeypatch)
    assert np.array_equal(_lire(_publie(out, "MNT", "0700_6601"))[0],
                          _lire(_publie(tmp_path / "sortie_seule", "MNT", "0700_6601"))[0])
    assert any(m == "Marge inter-dalles : 4 dalles étendues de 50 m depuis leurs voisines"
               for m in journal)


def test_halo_de_detection_lit_l_indice_non_rogne(tmp_path, monkeypatch):
    from app.services.cv_post_service import resolve_uncropped_tif

    _bloc(tmp_path / "mnt")
    out = tmp_path / "sortie"
    _lancer(tmp_path / "mnt", out, monkeypatch)
    ld = resolve_uncropped_tif(_publie(out, "LD", "0701_6600"), intermediaires_dir(out), "LD", {})
    assert ld is not None and ld.name.startswith("MNT_EXT_0701_6600_")
    assert get_raster_bounds(ld) == pytest.approx((700950.0, 6599000.0, 702000.0, 6600050.0))


def test_relance_a_l_identique_ne_recalcule_rien_et_voisin_touche_re_extrait(tmp_path, monkeypatch):
    dalles = _bloc(tmp_path / "mnt")
    out = tmp_path / "sortie"
    _lancer(tmp_path / "mnt", out, monkeypatch)
    ext = _inter(out, "MNT_EXT_0700_6601_MNT.tif")
    ld = next(intermediaires_dir(out).glob("MNT_EXT_0700_6601_LD*.tif"))
    avant = (ext.stat().st_mtime_ns, ld.stat().st_mtime_ns)
    _lancer(tmp_path / "mnt", out, monkeypatch)
    assert (ext.stat().st_mtime_ns, ld.stat().st_mtime_ns) == avant
    # Une voisine remplacée (plus récente) : le MNT étendu est ré-extrait, l'indice suit.
    futur = ext.stat().st_mtime + 60
    os.utime(dalles[(1, 0)], (futur, futur))
    _lancer(tmp_path / "mnt", out, monkeypatch)
    assert ext.stat().st_mtime_ns != avant[0] and ld.stat().st_mtime_ns != avant[1]


def test_lot_en_l_chaque_dalle_avec_voisine_est_etendue(tmp_path, monkeypatch):
    """Lot en L (vu sur 3 dalles LHD réelles, 0223_6844 / 0224_6844 / 0224_6843) : la voisine DIAGONALE retenue par
    ``halo_inputs`` ne faisait que toucher la fenêtre étendue d'un seul côté → BuildVRT la sautait → « 1 source(s)
    sautée(s) » → repli sur la dalle seule (couture). Seules les dalles qui intersectent la fenêtre servent."""
    dalles = _bloc(tmp_path / "mnt")
    dalles.pop((0, 1)).unlink()   # reste (0,0) NW, (1,0) NE, (1,1) SE
    out = tmp_path / "sortie"
    res, journal = _lancer(tmp_path / "mnt", out, monkeypatch)
    assert res.total == 3
    assert not any("impossible" in m for m in journal)
    assert "Marge inter-dalles : 3 dalles étendues de 50 m depuis leurs voisines" in journal
    # NW : voisine à l'est seulement ; la diagonale SE n'est pas une source.
    assert get_raster_bounds(_inter(out, "MNT_EXT_0700_6601_MNT.tif")) == pytest.approx(
        (700000.0, 6600000.0, 701050.0, 6601000.0))
    side = json.loads(_inter(out, "MNT_EXT_0700_6601_MNT.inputs.json").read_text(encoding="utf-8"))
    assert side["voisins"] == sorted(p.name for k, p in dalles.items() if k in ((0, 0), (1, 0)))
    # SE : voisine au nord seulement.
    assert get_raster_bounds(_inter(out, "MNT_EXT_0701_6600_MNT.tif")) == pytest.approx(
        (701000.0, 6599000.0, 702000.0, 6600050.0))


def test_coin_diagonal_seul_pas_de_voisine_ni_d_erreur(tmp_path, monkeypatch):
    """Deux dalles qui ne se touchent que par un coin : aucune n'a de voisine (pas de marge), et rien d'« impossible »
    au journal — copie seule, comme une dalle isolée."""
    dalles = _bloc(tmp_path / "mnt")
    dalles.pop((1, 0)).unlink()
    dalles.pop((0, 1)).unlink()
    out = tmp_path / "sortie"
    _, journal = _lancer(tmp_path / "mnt", out, monkeypatch)
    assert not any("impossible" in m for m in journal)
    assert "Marge inter-dalles : 2 dalles de 1 km sans voisine" in journal
    assert filecmp.cmp(dalles[(0, 0)], _inter(out, "MNT_EXT_0700_6601_MNT.tif"), shallow=False)


def test_dalle_isolee_copie_a_l_octet(tmp_path, monkeypatch):
    mnt = tmp_path / "mnt"
    mnt.mkdir()
    seule = _dalle(mnt, *NW)
    out = tmp_path / "sortie"
    _, journal = _lancer(mnt, out, monkeypatch)
    temp = _inter(out, "MNT_EXT_0700_6601_MNT.tif")
    assert filecmp.cmp(seule, temp, shallow=False)
    assert not _inter(out, "MNT_EXT_0700_6601_MNT.inputs.json").exists()
    assert "Marge inter-dalles : 1 dalle de 1 km sans voisine" in journal


def test_voisine_retiree_retour_a_la_copie(tmp_path, monkeypatch):
    """Un MNT étendu d'un run précédent ne survit pas au retrait de ses voisines (sinon la copie, plus vieille
    que lui, le garderait : indices calculés sur des voisines disparues)."""
    dalles = _bloc(tmp_path / "mnt")
    out = tmp_path / "sortie"
    _lancer(tmp_path / "mnt", out, monkeypatch)
    for k in ((1, 0), (0, 1), (1, 1)):
        dalles[k].unlink()
    _lancer(tmp_path / "mnt", out, monkeypatch)
    assert filecmp.cmp(dalles[(0, 0)], _inter(out, "MNT_EXT_0700_6601_MNT.tif"), shallow=False)
    assert not _inter(out, "MNT_EXT_0700_6601_MNT.inputs.json").exists()


def test_source_sans_nodata_marge_hors_donnees_a_moins_9999(tmp_path, monkeypatch):
    _bloc(tmp_path / "mnt", nodata=None)
    out = tmp_path / "sortie"
    _lancer(tmp_path / "mnt", out, monkeypatch)
    arr, nodata = _lire(_inter(out, "MNT_EXT_0700_6601_MNT.tif"))
    assert nodata == -9999.0 and not np.any(arr == 0)


def test_source_sautee_repli_sur_la_copie_et_journal(tmp_path, monkeypatch):
    dalles = _bloc(tmp_path / "mnt")

    def saute(*a, **k):
        raise RuntimeError("1 source(s) sautée(s) par gdal.BuildVRT")

    monkeypatch.setattr(em, "_gdal_extract", saute)
    out = tmp_path / "sortie"
    _, journal = _lancer(tmp_path / "mnt", out, monkeypatch)
    assert filecmp.cmp(dalles[(0, 0)], _inter(out, "MNT_EXT_0700_6601_MNT.tif"), shallow=False)
    assert any("dalle_700000_6601000.tif" in m and "sautée" in m for m in journal)
