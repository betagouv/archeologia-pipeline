"""Fiabilité affichée à la conversion (2026-09-09) : champs ``fiabilite`` /
``fiabilite_pct`` sur les détections individuelles et sidecar ``fiabilite.json``
à côté du GeoPackage — le bloc de run vient de l'orchestrateur."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("PIL")

import geopandas as gpd  # noqa: E402

from app.services.fiabilite import read_sidecar  # noqa: E402
from pipeline.cv.conversion_shp import create_shapefile_from_detections  # noqa: E402

_PX = 5.0
_W = 220
_TRANSFORMS = {"A": (_PX, -_PX, -50.0, 1050.0)}

BLOC = {
    "modele": "Modèle test",
    "provenance": "banc synthétique",
    "par_classe": {"obj": [
        {"categorie": "douteux", "seuil": 0.3, "garanti": 0.0, "mesure": 0.2, "n": 100},
        {"categorie": "probable", "seuil": 0.5, "garanti": 0.6, "mesure": 0.7, "n": 80},
        {"categorie": "quasi_certain", "seuil": 0.85, "garanti": 0.85, "mesure": None, "n": 12},
    ]},
}


def _yolo_line(cx: float, cy: float, w: float, h: float, conf: float) -> str:
    _pw, _ph, xo, yo = _TRANSFORMS["A"]
    return (f"0 {((cx - xo) / _PX) / _W:.6f} {((yo - cy) / _PX) / _W:.6f} "
            f"{w / _PX / _W:.6f} {h / _PX / _W:.6f} {conf:.2f}")


@pytest.fixture
def env(tmp_path):
    from PIL import Image

    labels, pngs = tmp_path / "raw", tmp_path / "png"
    labels.mkdir()
    pngs.mkdir()
    Image.new("L", (_W, _W), 128).save(pngs / "A.png")
    # quatre objets espacés, scores 0,35 / 0,6 / 0,9 / 0,25 (le dernier sous le seuil 0,3)
    (labels / "A.txt").write_text("\n".join([
        _yolo_line(200, 800, 40, 40, 0.35), _yolo_line(500, 800, 40, 40, 0.60),
        _yolo_line(800, 800, 40, 40, 0.90), _yolo_line(500, 300, 40, 40, 0.25),
    ]) + "\n", encoding="utf-8")
    return {"labels": labels, "pngs": pngs, "out": tmp_path / "out" / "obj.gpkg"}


def _convert(env, **kwargs):
    ok = create_shapefile_from_detections(
        labels_dir=str(env["labels"]), output_shapefile=str(env["out"]), png_dir=str(env["pngs"]),
        tif_transform_data=_TRANSFORMS, class_names={0: "obj"}, model_task="object_detection",
        postprocess_config={"merge_adjacent": False, "remove_overlaps": False},
        min_confidence=0.3, model_name="modele_test", **kwargs,
    )
    assert ok is True
    return gpd.read_file(env["out"], layer="obj")


def test_champs_et_sidecar(env):
    gdf = _convert(env, fiabilite=BLOC).sort_values("confidence")
    assert list(gdf["fiabilite"]) == ["Douteux", "Probable", "Très probable"]  # 0,25 filtré
    pcts = [None if p != p else p for p in gdf["fiabilite_pct"]]  # NaN -> None
    assert pcts == [20.0, 70.0, None]  # très probable : effectif insuffisant au banc
    entree = read_sidecar(env["out"], "obj")
    assert entree["classe"] == "obj" and entree["modele"] == "Modèle test"
    assert entree["provenance"] == "banc synthétique"
    assert [c["categorie"] for c in entree["categories"]] == ["douteux", "probable", "quasi_certain"]
    # effectifs par niveau des détections ÉCRITES (bilan de fin de run) : 0,25 filtré n'y est pas
    assert entree["effectifs"] == {"Douteux": 1, "Probable": 1, "Très probable": 1}
    assert json.loads((env["out"].parent / "fiabilite.json").read_text(encoding="utf-8"))["obj"]["classe"] == "obj"


def test_sans_bloc_rien_ne_change(env):
    gdf = _convert(env)
    assert "fiabilite" not in gdf.columns and "fiabilite_pct" not in gdf.columns
    assert not (env["out"].parent / "fiabilite.json").exists()
    assert set(gdf["conf_bin"]) == {"[0.3:0.4[", "[0.6:0.8[", "[0.8:1]"}  # tranches historiques intactes


def test_bloc_pour_une_autre_classe_ignore(env):
    gdf = _convert(env, fiabilite={"modele": "m", "par_classe": {"autre": BLOC["par_classe"]["obj"]}})
    assert "fiabilite" not in gdf.columns
    assert not (env["out"].parent / "fiabilite.json").exists()


@pytest.fixture
def env_accoles(tmp_path):
    from PIL import Image

    labels, pngs = tmp_path / "raw", tmp_path / "png"
    labels.mkdir()
    pngs.mkdir()
    Image.new("L", (_W, _W), 128).save(pngs / "A.png")
    # deux boîtes accolées (0,9 puis 0,2) et une isolée (0,2) ; seuil du run 0,16, seuil du modèle 0,3
    (labels / "A.txt").write_text("\n".join([
        _yolo_line(200, 800, 40, 40, 0.90), _yolo_line(240, 800, 40, 40, 0.20),
        _yolo_line(800, 300, 40, 40, 0.20),
    ]) + "\n", encoding="utf-8")
    return {"labels": labels, "pngs": pngs, "out": tmp_path / "out" / "obj.gpkg"}


def _convert_fusion(env, **kwargs):
    ok = create_shapefile_from_detections(
        labels_dir=str(env["labels"]), output_shapefile=str(env["out"]), png_dir=str(env["pngs"]),
        tif_transform_data=_TRANSFORMS, class_names={0: "obj"}, model_task="instance_segmentation",
        postprocess_config={"merge_adjacent": True, "remove_overlaps": False, "merge_buffer_m": 0.5},
        min_confidence=0.16, model_name="modele_test", **kwargs,
    )
    assert ok is True
    return gpd.read_file(env["out"], layer="obj")


def test_seuil_de_fusion_fige_la_fusion_au_seuil_du_modele(env_accoles):
    """Seuil abaissé à 0,16 (2026-10-22) : sans seuil de fusion, la boîte à 0,2 accolée
    rétrograde celle à 0,9 (moyenne 0,55) ; avec ``seuil_fusion_par_classe`` = seuil du
    modèle, les deux restent distinctes et les champs de fusion sont écrits."""
    gdf = _convert_fusion(env_accoles)
    assert len(gdf) == 2 and gdf["confidence"].max() == pytest.approx(0.55)
    gdf = _convert_fusion(env_accoles, seuil_fusion_par_classe={"obj": 0.3}).sort_values("confidence")
    assert len(gdf) == 3 and list(gdf["confidence"].round(2)) == [0.2, 0.2, 0.9]
    assert list(gdf["n_membres"]) == [1, 1, 1]
    assert list(gdf["conf_max"].round(2)) == [0.2, 0.2, 0.9]
    assert "longueur_m" in gdf.columns
