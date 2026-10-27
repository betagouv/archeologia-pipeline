"""``pipeline.cv.cv_output`` (lot 6 du chantier d'outillage, 2026-10-20) : aller-retour disque réel des
sorties d'inférence — ``.txt`` YOLO normalisé, ``.json`` relu par ``Detection.from_disk_dict`` (le format que
``conversion_shp`` consomme), sorties vides, image annotée, légende."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PIL_Image = pytest.importorskip("PIL.Image")

from pipeline.cv import cv_output  # noqa: E402
from pipeline.cv.class_utils import get_class_color  # noqa: E402
from pipeline.cv.types import Detection  # noqa: E402


def test_chemin_de_l_image_annotee(tmp_path):
    assert cv_output.get_detection_output_path("/x/img.jpg", "LD", str(tmp_path)) == str(tmp_path / "img_detections.png")
    assert cv_output.get_detection_output_path(str(tmp_path / "img.jpg"), "LD") == str(tmp_path / "img_detections.png")


def test_boites_en_yolo_normalise_et_json_relu(tmp_path):
    dets = [Detection(0, 0.9, (10.0, 20.0, 110.0, 70.0)), Detection(2, 0.4, (0.0, 0.0, 200.0, 100.0))]
    txt, js = cv_output.save_detections_to_files(
        image_path=str(tmp_path / "img.png"), output_path=str(tmp_path / "ann" / "img.png"), detections=dets,
        img_width=200, img_height=100, jpg_folder_path=str(tmp_path / "raw"))
    assert (txt, js) == (tmp_path / "raw" / "img.txt", tmp_path / "raw" / "img.json")
    assert txt.read_text(encoding="utf-8").splitlines() == ["0 0.300000 0.450000 0.500000 0.500000",
                                                            "2 0.500000 0.500000 1.000000 1.000000"]
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["image_dimensions"] == {"width": 200, "height": 100} and data["task"] == "detect"
    assert "model_type" not in data                                   # yolo : implicite
    assert [Detection.from_disk_dict(d) for d in data["detections"]] == dets


def test_polygones_de_segmentation_et_type_de_modele(tmp_path):
    trous = (((0.2, 0.2), (0.3, 0.2), (0.3, 0.3)),)
    det = Detection(1, 0.7, (10.0, 10.0, 50.0, 60.0), polygon=((0.1, 0.1), (0.5, 0.1), (0.5, 0.6)), polygon_holes=trous)
    txt, js = cv_output.save_detections_to_files(
        image_path="img.png", output_path=str(tmp_path / "ann" / "img.png"), detections=[det],
        img_width=100, img_height=100, task="segment", model_type="rfdetr")
    assert txt == tmp_path / "ann" / "img.txt"                        # sans jpg_folder_path : à côté de output_path
    assert txt.read_text(encoding="utf-8").splitlines() == ["1 0.100000 0.100000 0.500000 0.100000 0.500000 0.600000"]
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["task"] == "segment" and data["model_type"] == "rfdetr"
    relu = Detection.from_disk_dict(data["detections"][0])
    assert (relu.polygon, relu.polygon_holes, relu.confidence) == (det.polygon, trous, 0.7)
    assert relu.bbox == (0.1, 0.1, 0.5, 0.6)                          # dérivée du polygone (pas de bbox sur disque)


def test_sorties_vides_avec_les_dimensions_de_l_image(tmp_path):
    img = tmp_path / "img.png"
    PIL_Image.new("L", (64, 32)).save(img)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "img.txt").write_text("deja", encoding="utf-8")
    cv_output.save_empty_outputs(str(img), str(tmp_path / "ann" / "img.png"), str(raw))
    assert (raw / "img.txt").read_text(encoding="utf-8") == "deja"    # un .txt existant n'est pas écrasé
    assert json.loads((raw / "img.json").read_text(encoding="utf-8")) == {
        "image_path": str(img), "image_dimensions": {"width": 64, "height": 32}, "detections": []}


def test_sorties_vides_sans_dossier_ni_image_lisible(tmp_path):
    out = tmp_path / "ann" / "img.png"
    out.parent.mkdir()
    cv_output.save_empty_outputs(str(tmp_path / "absente.png"), str(out))
    assert out.with_suffix(".txt").read_text(encoding="utf-8") == ""
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["image_dimensions"] == {"width": None, "height": None}


def test_image_annotee_png_a_la_taille_de_la_source(tmp_path):
    source = PIL_Image.new("RGB", (120, 80), (0, 0, 0))
    dets = [Detection(0, 0.9, (10.0, 10.0, 50.0, 40.0)),
            Detection(1, 0.5, (0.0, 0.0, 0.0, 0.0), polygon=((0.5, 0.5), (0.9, 0.5), (0.9, 0.9)))]
    out = tmp_path / "ann" / "img_detections.png"
    cv_output.save_annotated_image(source, dets, str(out), class_names=["a", "b"])
    with PIL_Image.open(out) as res:
        assert (res.format, res.size) == ("PNG", (120, 80))
        assert res.convert("L").getextrema()[1] > 0                   # quelque chose a été dessiné
    assert source.getextrema() == ((0, 0), (0, 0), (0, 0))            # la source est intacte (copie)


def test_legende_texte_et_image(tmp_path):
    cv_output.save_legend_file(str(tmp_path / "ann"), ["four", "tumulus"], class_colors=[3, 0])
    lignes = (tmp_path / "ann" / "legend.txt").read_text(encoding="utf-8").splitlines()
    assert lignes[0] == "LÉGENDE DES CLASSES"
    assert lignes[3:] == [f"0: four - RGB{get_class_color(0, [3, 0])}", f"1: tumulus - RGB{get_class_color(1, [3, 0])}"]
    with PIL_Image.open(tmp_path / "ann" / "legend.png") as im:
        assert im.size == (400, 2 * 10 + 30 * 2 + 30)
    cv_output.save_legend_file(str(tmp_path / "vide"), [])
    assert not (tmp_path / "vide").exists()                           # rien sans classe


def test_un_dossier_illisible_ne_fait_pas_echouer(tmp_path):
    """Les sorties vides avalent l'erreur d'écriture (journal) : le run continue."""
    bloquant = tmp_path / "f.txt"
    bloquant.write_text("un fichier, pas un dossier", encoding="utf-8")
    cv_output.save_empty_outputs("img.png", str(Path(bloquant) / "img.png"), str(bloquant))
