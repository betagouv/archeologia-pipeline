"""MNT et indices étrangers (règle 2026-10-31) : tout CRS projeté en mètres est accepté,
les sorties sont étiquetées dans le CRS d'entrée, jamais reprojetées.

Ce que le test tient, sur des rasters synthétiques écrits par rasterio :
- préflight : EPSG:25832 et EPSG:27700 acceptés et affichés (« EPSG:25832 · 2 dalle(s) ») ;
  refusés (bloquant) : géographique (4326), projeté en pieds (2263), mélange de CRS ;
  sans CRS ni déclaration : « EPSG:2154 supposé » ⚠ non bloquant (inchangé) ;
- ``crs_des_entrees`` (source unique du CRS d'un run) : celui des rasters, le déclaré, sinon 2154 ;
- étiquetage : MNT copié et TIF publiés sans CRS → CRS du run ; GeoPackage de détections dans
  ``cv_config["crs"]`` ; mosaïque, trace ``traitement.json`` et rapport ;
- non-régression : sans CRS fourni, tout reste en EPSG:2154.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")

from pipeline.ingest_plan import (  # noqa: E402
    IngestValidationError,
    crs_des_entrees,
    normaliser_crs,
    plan_raster_inputs,
)
from pipeline.preflight import _check_raster_crs  # noqa: E402
from pipeline.tilespec import crs_unit_is_metre  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"

# Origines plausibles dans chaque CRS (coin nord-ouest), 0,5 m/px.
_ORIGINES = {
    "EPSG:25832": (560000.0, 6020000.0),    # Schleswig-Holstein (UTM 32N)
    "EPSG:27700": (420000.0, 280000.0),     # British National Grid
    "EPSG:2154": (700000.0, 6600000.0),
    "EPSG:4326": (9.5, 54.3),
    "EPSG:2263": (980000.0, 200000.0),      # New York Long Island, pieds US
}


def _tif(path: Path, crs, *, origin=None, px=0.5, n=40) -> Path:
    import numpy as np
    from rasterio.transform import from_origin

    ox, oy = origin or _ORIGINES.get(crs, (700000.0, 6600000.0))
    profile = dict(driver="GTiff", width=n, height=n, count=1, dtype="float32",
                   transform=from_origin(ox, oy, px, px))
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(str(path), "w", **profile) as ds:
        ds.write(np.full((1, n, n), 12.0, dtype="float32"))
    return path


def _crs_lu(path: Path):
    with rasterio.open(str(path)) as ds:
        return ds.crs.to_string() if ds.crs else None


def _dossier(tmp_path, crs, n=2, nom="mnt"):
    d = tmp_path / nom
    d.mkdir()
    ox, oy = _ORIGINES[crs]
    for i in range(n):
        _tif(d / f"dalle_{i}.tif", crs, origin=(ox + i * 20.0, oy))
    return d


# --- unité du CRS ------------------------------------------------------------------------

@pytest.mark.parametrize("crs,attendu", [
    ("EPSG:2154", True), ("EPSG:25832", True), ("EPSG:27700", True), ("EPSG:2157", True),
    ("EPSG:28992", True), ("EPSG:25833", True), ("EPSG:2263", False), ("EPSG:4326", None),
])
def test_unite_du_crs(crs, attendu):
    assert crs_unit_is_metre(crs) is attendu


# --- préflight -----------------------------------------------------------------------------

@pytest.mark.parametrize("crs", ["EPSG:25832", "EPSG:27700"])
def test_preflight_accepte_un_crs_etranger_et_l_affiche(tmp_path, crs):
    d = _dossier(tmp_path, crs)
    results = []
    _check_raster_crs(str(d), ("tif",), results)
    assert [(r.name, r.ok, r.critical) for r in results] == [("CRS des rasters", True, True)]
    assert results[0].details == f"{crs} · 2 dalle(s)"


def test_preflight_lambert93_inchange(tmp_path):
    results = []
    _check_raster_crs(str(_dossier(tmp_path, "EPSG:2154")), ("tif",), results)
    assert results[0].ok is True and results[0].details == "EPSG:2154 · 2 dalle(s)"


@pytest.mark.parametrize("crs,motif", [("EPSG:4326", "géographique"), ("EPSG:2263", "pas en mètres")])
def test_preflight_refuse_geographique_et_non_metrique(tmp_path, crs, motif):
    results = []
    _check_raster_crs(str(_dossier(tmp_path, crs)), ("tif",), results)
    assert len(results) == 1 and results[0].ok is False and results[0].critical is True
    assert motif in results[0].details and crs in results[0].details


def test_preflight_refuse_le_melange(tmp_path):
    d = tmp_path / "mnt"
    d.mkdir()
    _tif(d / "a.tif", "EPSG:25832")
    _tif(d / "b.tif", "EPSG:27700")
    results = []
    _check_raster_crs(str(d), ("tif",), results)
    assert results[0].ok is False and results[0].critical is True
    assert "Mélange" in results[0].details and "EPSG:25832" in results[0].details


def test_preflight_sans_crs_suppose_2154(tmp_path):
    d = tmp_path / "mnt"
    d.mkdir()
    _tif(d / "a.tif", None)
    results = []
    _check_raster_crs(str(d), ("tif",), results)
    assert results[0].ok is False and results[0].critical is False
    assert results[0].details.startswith("EPSG:2154 supposé")


def test_preflight_sans_crs_avec_crs_declare_etranger(tmp_path):
    d = tmp_path / "mnt"
    d.mkdir()
    _tif(d / "a.tif", None, origin=_ORIGINES["EPSG:27700"])
    results = []
    _check_raster_crs(str(d), ("tif",), results, declared_crs="EPSG:27700")
    assert results[0].ok is True and results[0].details == "EPSG:27700 · 1 dalle(s)"


def test_plan_refuse_sans_crs_ni_defaut(tmp_path):
    with pytest.raises(IngestValidationError, match="CRS introuvable"):
        plan_raster_inputs([_tif(tmp_path / "a.tif", None)])


# --- CRS du run (source unique) ------------------------------------------------------------

def test_crs_des_entrees(tmp_path):
    assert crs_des_entrees(_dossier(tmp_path, "EPSG:25832", nom="de"), ("tif",)) == "EPSG:25832"
    assert crs_des_entrees(_dossier(tmp_path, "EPSG:2154", nom="fr"), ("tif",)) == "EPSG:2154"
    vide = tmp_path / "sans"
    vide.mkdir()
    _tif(vide / "a.tif", None)
    assert crs_des_entrees(vide, ("tif",)) == "EPSG:2154"
    assert crs_des_entrees(vide, ("tif",), declared_crs="EPSG:27700") == "EPSG:27700"
    assert crs_des_entrees(tmp_path / "absent", ("tif",)) == "EPSG:2154"   # jamais d'exception


def test_normaliser_crs():
    assert normaliser_crs(None) == "EPSG:2154"
    assert normaliser_crs("epsg:25832") == "EPSG:25832"


# --- étiquetage des sorties ------------------------------------------------------------------

def test_mnt_copie_sans_crs_prend_le_crs_du_run(tmp_path, monkeypatch):
    """existing_mnt : le MNT copié dans les intermédiaires reçoit le CRS du run avant le calcul
    des indices (qui en héritent) ; un MNT qui a son CRS n'est pas touché."""
    import pipeline.modes.existing_mnt as em

    for fn in ("create_visualization_products", "copy_products_without_crop", "copy_final_products_to_results"):
        monkeypatch.setattr(em, fn, lambda **k: None)
    _tif(tmp_path / "T_MNT.tif", None, origin=_ORIGINES["EPSG:27700"])
    _tif(tmp_path / "U_MNT.tif", "EPSG:25832")
    for nom in ("T", "U"):
        em._process_single_mnt_tile(
            current_tile_name=nom, temp_dir=tmp_path, output_dir=tmp_path, products={}, output_structure={},
            output_formats={}, rvt_params={}, skip_crop=True, log=lambda m: None, cancel_check=None,
            crs="EPSG:27700")
    assert _crs_lu(tmp_path / "T_MNT.tif") == "EPSG:27700"
    assert _crs_lu(tmp_path / "U_MNT.tif") == "EPSG:25832"


def _run_rvt_sans_cv(tmp_path, monkeypatch, crs):
    pytest.importorskip("shapely")  # pipeline.cv.__init__
    import pipeline.modes.existing_rvt as er

    src = tmp_path / "entree"
    src.mkdir()
    _tif(src / "ld.tif", None, origin=_ORIGINES["EPSG:27700"])
    out = tmp_path / "sortie"
    er.run_existing_rvt(existing_rvt_dir=src, output_dir=out, cv_config={"enabled": False},
                        output_structure={}, indices_folder_name="RVT", crs=crs)
    return next((out / "livrable").rglob("ld.tif")), src / "ld.tif"


def test_existing_rvt_tif_publie_etiquete_et_entree_intacte(tmp_path, monkeypatch):
    publie, source = _run_rvt_sans_cv(tmp_path, monkeypatch, "EPSG:27700")
    assert _crs_lu(publie) == "EPSG:27700"
    assert _crs_lu(source) is None              # le dossier de l'utilisateur n'est jamais modifié


def test_existing_rvt_sans_crs_du_run_comportement_historique(tmp_path, monkeypatch):
    publie, _ = _run_rvt_sans_cv(tmp_path, monkeypatch, None)
    assert _crs_lu(publie) is None              # étiqueté plus tard (mosaïque) en EPSG:2154, comme avant


def test_existing_rvt_transmet_le_crs_aux_detections(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    import pipeline.cv.runner as cv_runner
    import pipeline.modes.existing_rvt as er

    vus = []
    monkeypatch.setattr(cv_runner, "run_cv_on_folder", lambda **k: vus.append(dict(k["cv_config"])))
    monkeypatch.setattr(er, "_convert_tif_to_png_with_world",
                        lambda s, d: (Path(d).parent.mkdir(parents=True, exist_ok=True), Path(d).write_bytes(b"png")))
    src = tmp_path / "entree"
    src.mkdir()
    _tif(src / "ld.tif", "EPSG:25832")
    er.run_existing_rvt(existing_rvt_dir=src, output_dir=tmp_path / "sortie",
                        cv_config={"enabled": True, "target_rvt": "LD"}, output_structure={},
                        indices_folder_name="RVT", crs="EPSG:25832")
    assert vus and vus[0]["crs"] == "EPSG:25832"


def test_le_moteur_de_detection_lit_le_crs_du_run():
    """Les trois appels à la conversion des détections prennent ``cv_config["crs"]`` ; plus aucun
    « EPSG:2154 » écrit en dur à l'appel (lu en source : ``runner.py`` importe le moteur)."""
    for nom in ("runner.py", "runner_inference.py"):
        texte = (SRC / "pipeline" / "cv" / nom).read_text(encoding="utf-8")
        assert 'crs="EPSG:2154"' not in texte, nom
        assert len(re.findall(r'crs=\(cv_config or \{\}\)\.get\("crs"\) or "EPSG:2154"', texte)) >= 1, nom


def test_geopackage_des_detections_dans_le_crs_du_run(tmp_path):
    pytest.importorskip("geopandas")
    PIL_Image = pytest.importorskip("PIL.Image")
    import geopandas as gpd

    from pipeline.cv import cv_output
    from pipeline.cv.runner_shapefiles import deduplicate_cv_shapefiles_final
    from pipeline.cv.types import Detection

    modele = tmp_path / "models" / "m"
    (modele / "weights").mkdir(parents=True)
    (modele / "weights" / "best.onnx").write_bytes(b"")
    (modele / "weights" / "best.json").write_text(json.dumps({"task": "detection"}), encoding="utf-8")
    (modele / "classes.txt").write_text("four\n", encoding="utf-8")
    labels, pngs = tmp_path / "raw", tmp_path / "png"
    labels.mkdir()
    pngs.mkdir()
    PIL_Image.new("L", (200, 200), 128).save(pngs / "A.png")
    cv_output.save_detections_to_files(image_path=str(pngs / "A.png"), output_path=str(labels / "A.png"),
                                       detections=[Detection(0, 0.9, (20.0, 20.0, 60.0, 60.0))],
                                       img_width=200, img_height=200, jpg_folder_path=str(labels))
    ox, oy = _ORIGINES["EPSG:25832"]
    deduplicate_cv_shapefiles_final(
        labels_dir=labels, png_dir=pngs, shp_dir=tmp_path / "out" / "shp", target_rvt="LD",
        output_dir=tmp_path / "out", tif_transform_data={"A": (0.5, -0.5, ox, oy)},
        cv_config={"selected_model": "m", "models_dir": str(modele.parent), "confidence_threshold": 0.0},
        crs="EPSG:25832", log=lambda m: None)
    gdf = gpd.read_file(tmp_path / "out" / "livrable" / "detections" / "m" / "m.gpkg", layer="four")
    assert gdf.crs.to_epsg() == 25832
    assert gdf.geometry.iloc[0].bounds == pytest.approx((ox + 10, oy - 30, ox + 30, oy - 10))


def test_mosaique_reetiquette_dans_le_crs_du_run(tmp_path, monkeypatch):
    import pipeline.ign.products.results as res

    vus = []
    monkeypatch.setattr(res, "assign_crs_if_missing", lambda p, crs="EPSG:2154": vus.append(crs))
    _tif(tmp_path / "a.tif", None)
    # Construction simulée (gdalbuildvrt en échec) : seul compte le CRS de repli passé aux sources.
    monkeypatch.setattr(res.shutil, "which", lambda _n: "gdalbuildvrt")
    monkeypatch.setattr(res.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 1, "stderr": "", "stdout": ""})())
    res.build_vrt_index(tmp_path, crs="EPSG:25832")
    assert vus == ["EPSG:25832"]
    vus.clear()
    res.build_vrt_index(tmp_path)
    assert vus == ["EPSG:2154"]


def test_trace_et_rapport_portent_le_crs(tmp_path, monkeypatch):
    import time as _time

    from app.services import finalize_service

    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)

    class _R:
        def __getattr__(self, _n):
            return lambda *a, **k: None

    finalize_service.finalize_pipeline(output_dir=tmp_path, cv_cfg={}, rvt_params={}, reporter=_R(),
                                       start_time=_time.time(), tiles_processed=1, active_products=["MNT"],
                                       ui_config={}, crs="EPSG:25832")
    trace = json.loads((tmp_path / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert trace["crs"] == "EPSG:25832"
    assert finalize_service.crs_du_traitement(tmp_path) == "EPSG:25832"
    assert "Système de coordonnées des couches : EPSG:25832" in (
        tmp_path / "livrable" / "rapport.html").read_text(encoding="utf-8")


def test_trace_par_defaut_2154_et_absente_avant(tmp_path, monkeypatch):
    import time as _time

    from app.services import finalize_service

    assert finalize_service.crs_du_traitement(tmp_path) is None     # trace absente / antérieure
    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)

    class _R:
        def __getattr__(self, _n):
            return lambda *a, **k: None

    finalize_service.finalize_pipeline(output_dir=tmp_path, cv_cfg={}, rvt_params={}, reporter=_R(),
                                       start_time=_time.time(), tiles_processed=1, active_products=["MNT"],
                                       ui_config={})
    assert finalize_service.crs_du_traitement(tmp_path) == "EPSG:2154"


def test_runners_lidar_restent_en_2154():
    """ign_laz / local_laz ne calculent pas de CRS d'entrée : finalisation et détections au défaut."""
    texte = (SRC / "app" / "runners" / "ign_local_runner.py").read_text(encoding="utf-8")
    assert "crs_des_entrees" not in texte and "crs=" not in texte
    for nom in ("existing_mnt_runner.py", "existing_rvt_runner.py"):
        t = (SRC / "app" / "runners" / nom).read_text(encoding="utf-8")
        assert "crs_des_entrees(" in t and t.count("crs=crs") >= 2, nom


# --- R1 (audit 2026-10-31) : même CRS écrit en code EPSG et en WKT sans AUTHORITY ---------

def _wkt_sans_authority(code: int) -> str:
    from rasterio.crs import CRS

    wkt = CRS.from_epsg(code).to_wkt(version="WKT1_GDAL")
    sans = re.sub(r',AUTHORITY\["[A-Z]+","\d+"\]', "", wkt)
    assert "AUTHORITY" not in sans
    return sans


def _spec(nom, crs):
    from pipeline.tilespec import TileSpec

    ox, oy = _ORIGINES["EPSG:25832"]
    return TileSpec.from_values(source_path=Path(nom), bounds=(ox, oy - 100.0, ox + 100.0, oy),
                                pixel_size_x=0.5, pixel_size_y=-0.5, width_px=200, height_px=200, crs=crs)


def test_authid_et_wkt_equivalent_sans_authority_ne_sont_pas_un_melange():
    """GDAL 3.11 (QGIS 3.44) laisse un WKT de 25832 sans AUTHORITY tel quel : avec une dalle
    « EPSG:25832 » à côté, le préflight refusait un « mélange » qui n'en est pas un."""
    from pipeline.ingest_plan import plan_from_specs

    plan = plan_from_specs([_spec("a.tif", "EPSG:25832"), _spec("b.tif", _wkt_sans_authority(25832))])
    assert plan.crs == "EPSG:25832"


def test_wkt_seul_sans_authority_etiquete_en_code_epsg():
    """Un raster seul en WKT sans AUTHORITY : l'étiquette du run (trace, rapport, cv_config)
    est le code EPSG, jamais le WKT entier."""
    from pipeline.ingest_plan import plan_from_specs

    wkt = _wkt_sans_authority(25832)
    assert plan_from_specs([_spec("b.tif", wkt)]).crs == "EPSG:25832"
    assert normaliser_crs(wkt) == "EPSG:25832"


def test_vrai_melange_toujours_refuse_avec_wkt():
    from pipeline.ingest_plan import plan_from_specs

    with pytest.raises(IngestValidationError, match="Mélange"):
        plan_from_specs([_spec("a.tif", "EPSG:25832"), _spec("b.tif", _wkt_sans_authority(27700))])


def test_lecture_gdal_identifie_le_code_epsg(tmp_path):
    """Lecture par osgeo (le backend de QGIS) d'un GeoTIFF dont le WKT n'a pas d'AUTHORITY."""
    gdal = pytest.importorskip("osgeo.gdal")
    from pipeline.tilespec import _crs_from_gdal

    p = tmp_path / "a.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(p), 4, 4, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((560000, 1, 0, 6020000, 0, -1))
    ds.SetProjection(_wkt_sans_authority(25832))
    ds = None
    assert _crs_from_gdal(gdal.Open(str(p))) == "EPSG:25832"


# --- R2 (audit 2026-10-31) : CRS du projet, relance dans un dossier d'un autre CRS ----------

def test_crs_du_projet_couches_d_abord():
    from app.services.finalize_service import crs_du_projet

    assert crs_du_projet(["EPSG:25832", "EPSG:25832"], "EPSG:2154") == "EPSG:25832"   # trace périmée
    assert crs_du_projet(["", ""], "EPSG:27700") == "EPSG:27700"
    assert crs_du_projet([], None) == "EPSG:2154"


def test_alerte_relance_autre_crs():
    from pipeline.ingest_plan import alerte_relance_autre_crs

    assert alerte_relance_autre_crs(None, "EPSG:25832") is None                       # dossier neuf
    assert alerte_relance_autre_crs({"crs": "EPSG:25832"}, "EPSG:25832") is None
    assert alerte_relance_autre_crs({}, "EPSG:2154") is None                           # trace d'avant = 2154
    msg = alerte_relance_autre_crs({}, "EPSG:25832")
    assert msg and "EPSG:2154" in msg and "nouveau dossier" in msg
    assert "EPSG:27700" in alerte_relance_autre_crs({"crs": "EPSG:27700"}, "EPSG:25832")


def test_preflight_avertit_d_une_relance_dans_un_dossier_d_un_autre_crs(tmp_path):
    d = _dossier(tmp_path, "EPSG:25832")
    sortie = tmp_path / "sortie"
    (sortie / "livrable").mkdir(parents=True)
    (sortie / "livrable" / "traitement.json").write_text(json.dumps({"crs": "EPSG:2154"}), encoding="utf-8")
    results = []
    _check_raster_crs(str(d), ("tif",), results, output_dir=sortie)
    alerte = [r for r in results if r.name == "CRS du dossier de sortie"]
    assert results[0].ok is True                                 # le CRS des rasters reste accepté
    assert len(alerte) == 1 and alerte[0].ok is False and alerte[0].critical is False
    assert "EPSG:2154" in alerte[0].details
    (sortie / "livrable" / "traitement.json").write_text(json.dumps({"crs": "EPSG:25832"}), encoding="utf-8")
    results = []
    _check_raster_crs(str(d), ("tif",), results, output_dir=sortie)
    assert not [r for r in results if r.name == "CRS du dossier de sortie"]


def test_libelle_francais_du_controle():
    from app.services.vocabulaire import ENVIRONNEMENT

    assert ENVIRONNEMENT["CRS du dossier de sortie"] == "Système de coordonnées du dossier de sortie"


def test_trace_ecrite_avant_le_chargement_des_couches(tmp_path, monkeypatch):
    """Le .qgs (écrit au signal load_layers, sur le fil principal) relit la trace en repli : elle
    doit déjà porter le CRS du run quand le signal part."""
    import time as _time

    from app.services import finalize_service

    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: ["x.vrt"])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)
    vus = []

    class _R:
        def load_layers(self, *a, **k):
            vus.append(finalize_service.crs_du_traitement(tmp_path))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    finalize_service.finalize_pipeline(output_dir=tmp_path, cv_cfg={}, rvt_params={}, reporter=_R(),
                                       start_time=_time.time(), tiles_processed=1, active_products=["MNT"],
                                       ui_config={}, crs="EPSG:25832")
    assert vus == ["EPSG:25832"]


# --- R3 (audit 2026-10-31) : aire minimale par entité et repli des couches sans CRS ---------

def test_filtre_d_aire_par_entite_garde_le_crs_du_run(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    import pipeline.cv.conversion_shp as cs
    import pipeline.modes.existing_rvt as er
    import pipeline.output_paths as op

    gp = tmp_path / "a.gpkg"
    gp.write_bytes(b"")
    vus = []
    monkeypatch.setattr(op, "aires_par_entite", lambda out, run: [(gp, 10.0, frozenset())])
    monkeypatch.setattr(cs, "_filter_gpkg_by_min_area", lambda **k: vus.append(k.get("crs")))
    er._filtrer_aires_par_entite(tmp_path, {"crs": "EPSG:25832"}, lambda m: None)
    er._filtrer_aires_par_entite(tmp_path, {}, lambda m: None)
    assert vus == ["EPSG:25832", "EPSG:2154"]


def test_repli_d_une_couche_sans_crs(tmp_path):
    from app.services.finalize_service import crs_de_repli_pour

    tif = tmp_path / "livrable" / "indices" / "LD" / "tif"
    tif.mkdir(parents=True)
    gpkg = tmp_path / "livrable" / "detections" / "enclos" / "enclos.gpkg"
    assert crs_de_repli_pour(str(tif / "index_LD.vrt")) == "EPSG:2154"                 # pas de trace
    (tmp_path / "livrable" / "traitement.json").write_text(json.dumps({"crs": "EPSG:25832"}), encoding="utf-8")
    assert crs_de_repli_pour(str(tif / "index_LD.vrt")) == "EPSG:25832"
    assert crs_de_repli_pour(f"{gpkg}|layername=Enclos") == "EPSG:25832"
    assert crs_de_repli_pour("") == "EPSG:2154"


def test_runner_rvt_journalise_le_crs_du_run():
    t = (SRC / "app" / "runners" / "existing_rvt_runner.py").read_text(encoding="utf-8")
    assert "Système de coordonnées du run" in t
