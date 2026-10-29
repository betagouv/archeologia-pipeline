"""``pipeline.cv.runner_shapefiles.deduplicate_cv_shapefiles_final`` (lot 6 du chantier d'outillage,
2026-10-20) : l'étape « après inférence », de bout en bout sur disque — détections écrites par ``cv_output``
(le vrai format du moteur), profil d'un modèle minimal dans ``tmp_path``, GeoPackage relu par geopandas.

Ce que le test tient : le dédoublonnage des boîtes superposées d'une même image (stratégie « relation »,
meilleure confiance gardée), le routage par entité (un GeoPackage par entité, couches renommées), le
repli modèle-centré, le filtrage par aire minimale et l'exemption des couches de synthèse."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")
PIL_Image = pytest.importorskip("PIL.Image")

import geopandas as gpd  # noqa: E402

from pipeline.cv import cv_output  # noqa: E402
from pipeline.cv.runner_shapefiles import _run_model_slug, deduplicate_cv_shapefiles_final  # noqa: E402
from pipeline.cv.types import Detection  # noqa: E402

# Une image de 200 px à 0,5 m/px (100 m), origine Lambert-93 au coin nord-ouest.
_PX, _N = 0.5, 200
_TRANSFORMS = {"A": (_PX, -_PX, 800000.0, 6300000.0)}


@pytest.fixture
def modele(tmp_path) -> Path:
    """Modèle bbox minimal : poids (vides), classes, sidecar ``task`` → post-traitement « relation »."""
    d = tmp_path / "models" / "mon_modele"
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.onnx").write_bytes(b"")
    (d / "weights" / "best.json").write_text(json.dumps({"task": "detection"}), encoding="utf-8")
    (d / "classes.txt").write_text("four\ntumulus\n", encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path):
    labels, pngs = tmp_path / "raw", tmp_path / "png"
    labels.mkdir()
    pngs.mkdir()
    PIL_Image.new("L", (_N, _N), 128).save(pngs / "A.png")
    return {"labels": labels, "pngs": pngs, "out": tmp_path / "sortie", "shp": tmp_path / "sortie" / "shp"}


def _ecrire(env, dets):
    cv_output.save_detections_to_files(image_path=str(env["pngs"] / "A.png"), output_path=str(env["labels"] / "A.png"),
                                       detections=dets, img_width=_N, img_height=_N, jpg_folder_path=str(env["labels"]))


def _lancer(env, modele, **cv):
    journal = []
    cfg = {"selected_model": "mon_modele", "models_dir": str(modele.parent), "confidence_threshold": 0.0, **cv}
    deduplicate_cv_shapefiles_final(labels_dir=env["labels"], png_dir=env["pngs"], shp_dir=env["shp"], target_rvt="LD",
                                    output_dir=env["out"], cv_config=cfg, tif_transform_data=_TRANSFORMS, log=journal.append)
    return journal


def test_repli_modele_centre_et_doublons_fusionnes(env, modele):
    """Deux boîtes identiques (0,9 et 0,6) et une autre à part → 2 entités, la meilleure confiance gardée."""
    _ecrire(env, [Detection(0, 0.9, (20.0, 20.0, 60.0, 60.0)), Detection(0, 0.6, (20.0, 20.0, 60.0, 60.0)),
                  Detection(1, 0.8, (120.0, 120.0, 180.0, 180.0))])
    journal = _lancer(env, modele)
    gpkg = env["out"] / "livrable" / "detections" / "mon_modele" / "mon_modele.gpkg"
    assert gpkg.exists(), journal
    assert any("repli modèle-centré detections/mon_modele/" in m for m in journal)
    fours = gpd.read_file(gpkg, layer="four")
    assert len(fours) == 1 and fours["confidence"].iloc[0] == pytest.approx(0.9)
    assert fours.geometry.iloc[0].bounds == pytest.approx((800010.0, 6299970.0, 800030.0, 6299990.0))   # 0,5 m/px, y vers le bas
    assert len(gpd.read_file(gpkg, layer="tumulus")) == 1
    assert fours["model_name"].iloc[0] == "mon_modele"


def test_routage_par_entite_et_couches_renommees(env, modele):
    _ecrire(env, [Detection(0, 0.9, (20.0, 20.0, 60.0, 60.0)), Detection(1, 0.8, (120.0, 120.0, 180.0, 180.0))])
    entites = [{"id": "fours", "slug": "fours", "classes": ["four"], "layer_names": {"four": "Fours"}, "is_derived": False},
               {"id": "tertres", "slug": "tertres", "classes": ["tumulus"], "is_derived": False}]
    journal = _lancer(env, modele, entities=entites)
    assert any("routage par entité actif (2 GeoPackage(s)" in m for m in journal)
    detections = env["out"] / "livrable" / "detections"
    assert len(gpd.read_file(detections / "fours" / "fours.gpkg", layer="Fours")) == 1
    assert len(gpd.read_file(detections / "tertres" / "tertres.gpkg", layer="tumulus")) == 1
    assert not (detections / "mon_modele").exists()


def test_filtrage_par_aire_minimale_epargne_les_couches_de_synthese(env, modele, monkeypatch):
    """min_area_m2 : la petite boîte (25 m²) disparaît, la grande reste ; une couche nommée comme la sortie
    d'une règle de clustering n'est pas rabotée."""
    _ecrire(env, [Detection(0, 0.9, (20.0, 20.0, 30.0, 30.0)), Detection(1, 0.8, (100.0, 100.0, 180.0, 180.0))])
    entites = [{"id": "fours", "slug": "fours", "classes": ["four"], "is_derived": False},
               {"id": "tertres", "slug": "tertres", "classes": ["tumulus"], "is_derived": False}]
    _lancer(env, modele, entities=entites, min_area_m2=100.0)
    detections = env["out"] / "livrable" / "detections"
    assert not (detections / "fours" / "fours.gpkg").exists()           # vidé par le filtre → supprimé
    assert len(gpd.read_file(detections / "tertres" / "tertres.gpkg", layer="tumulus")) == 1


def test_sans_detection_le_bilan_le_dit(env, modele):
    _ecrire(env, [])
    journal = _lancer(env, modele)
    assert not (env["out"] / "livrable" / "detections" / "mon_modele" / "mon_modele.gpkg").exists()
    assert any("Computer Vision:" in m and ("0" in m or "aucun" in m.lower()) for m in journal[-2:]), journal[-3:]


def test_slug_du_modele_du_run():
    assert _run_model_slug({"selected_model": "C:/m/crateres_seg_ld_v1/weights/best.onnx"}) == "crateres_seg_ld_v1"
    assert _run_model_slug({"selected_model": "lineaires_seg_v3_1"}) == "lineaires_seg_v3_1"
    assert _run_model_slug({}) is None and _run_model_slug(None) is None
