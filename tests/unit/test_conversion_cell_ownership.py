"""Règle du centroïde dans ``create_shapefile_from_detections`` (halo inter-dalles).

Deux images à halo (dalle + 50 m) se recouvrent : un objet à cheval sur leur
frontière est rapporté par les deux (doublon), et un objet coupé au bord du
halo de l'une n'est qu'un fragment. Avec ``cell_bounds_by_stem``, chaque
image ne conserve que les détections centrées dans SA cellule rognée.
"""
from __future__ import annotations

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("PIL")

import geopandas as gpd  # noqa: E402

from pipeline.cv.conversion_shp import create_shapefile_from_detections  # noqa: E402

# Deux cellules 1 km (x 0..1000 et 1000..2000, y 0..1000), images à halo de
# 50 m : 220 px à 5 m/px, origine au coin nord-ouest du halo.
_PX = 5.0
_W = 220
_CELLS = {"A": (0.0, 0.0, 1000.0, 1000.0), "B": (1000.0, 0.0, 2000.0, 1000.0)}
_TRANSFORMS = {"A": (_PX, -_PX, -50.0, 1050.0), "B": (_PX, -_PX, 950.0, 1050.0)}


def _yolo_line(stem: str, cx: float, cy: float, w: float, h: float, conf: float = 0.8) -> str:
    """Boîte géo (centre, taille en m) → ligne YOLO normalisée pour l'image ``stem``."""
    _pw, _ph, xo, yo = _TRANSFORMS[stem]
    x_rel = ((cx - xo) / _PX) / _W
    y_rel = ((yo - cy) / _PX) / _W
    return f"0 {x_rel:.6f} {y_rel:.6f} {w / _PX / _W:.6f} {h / _PX / _W:.6f} {conf:.2f}"


@pytest.fixture
def env(tmp_path):
    from PIL import Image

    labels = tmp_path / "raw"
    pngs = tmp_path / "png"
    labels.mkdir()
    pngs.mkdir()
    for stem in ("A", "B"):
        Image.new("L", (_W, _W), 128).save(pngs / f"{stem}.png")
    # O1 : à cheval sur x=1000, centre en A → vu entier par A ET par B (doublon).
    # O2 : au milieu de B, vu par B seulement.
    # F  : fragment dans le halo de A (x 1010..1050), centre en B, absent de B.
    (labels / "A.txt").write_text(
        _yolo_line("A", 990, 500, 60, 60) + "\n" + _yolo_line("A", 1030, 300, 40, 40, 0.9) + "\n",
        encoding="utf-8",
    )
    (labels / "B.txt").write_text(
        _yolo_line("B", 990, 500, 60, 60, 0.7) + "\n" + _yolo_line("B", 1500, 500, 40, 40) + "\n",
        encoding="utf-8",
    )
    return {"labels": labels, "pngs": pngs, "out": tmp_path / "out" / "obj.gpkg"}


def _convert(env, **kwargs):
    ok = create_shapefile_from_detections(
        labels_dir=str(env["labels"]),
        output_shapefile=str(env["out"]),
        png_dir=str(env["pngs"]),
        tif_transform_data=_TRANSFORMS,
        class_names={0: "obj"},
        model_task="object_detection",
        postprocess_config={"merge_adjacent": False, "remove_overlaps": False},
        **kwargs,
    )
    assert ok is True
    return gpd.read_file(env["out"], layer="obj")


def test_sans_carte_des_cellules_tout_est_conserve(env):
    gdf = _convert(env)

    assert len(gdf) == 4  # doublon + fragment inclus (comportement sans halo)


def test_chaque_dalle_ne_rapporte_que_ses_objets(env):
    gdf = _convert(env, cell_bounds_by_stem=_CELLS)

    assert len(gdf) == 2
    centres = sorted((round(g.centroid.x), round(g.centroid.y)) for g in gdf.geometry)
    assert centres == [(990, 500), (1500, 500)]  # O1 une seule fois, O2 ; F écarté
    o1 = next(g for g in gdf.geometry if round(g.centroid.x) == 990)
    assert o1.bounds == pytest.approx((960.0, 470.0, 1020.0, 530.0))  # entier, pas coupé à x=1000
