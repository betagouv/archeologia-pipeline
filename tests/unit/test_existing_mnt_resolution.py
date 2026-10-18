"""Mode MNT existant : le MNT publié porte la résolution du raster fourni, pas « 0M50 » d'office."""
import pytest

rasterio = pytest.importorskip("rasterio")


def test_mnt_fourni_publie_a_sa_resolution(tmp_path, monkeypatch):
    import numpy as np
    from rasterio.transform import from_origin

    import pipeline.modes.existing_mnt as em

    tuile = "LHD_FXX_0623_6864"
    with rasterio.open(
        str(tmp_path / f"{tuile}_MNT.tif"), "w", driver="GTiff", width=4, height=4, count=1,
        dtype="float32", crs="EPSG:2154", transform=from_origin(623000.0, 6864000.0, 1.0, 1.0),
    ) as ds:
        ds.write(np.zeros((1, 4, 4), dtype="float32"))
    vu: dict = {}
    monkeypatch.setattr(em, "create_visualization_products", lambda **k: None)
    monkeypatch.setattr(em, "crop_final_products", lambda **k: vu.setdefault("crop", k.get("mnt_resolution")))
    monkeypatch.setattr(em, "copy_final_products_to_results", lambda **k: vu.setdefault("pub", k.get("mnt_resolution")))
    assert em._process_single_mnt_tile(
        current_tile_name=tuile, temp_dir=tmp_path, output_dir=tmp_path / "out", products={"MNT": True},
        output_structure={}, output_formats={"tif": True}, rvt_params={}, skip_crop=False,
        log=lambda _m: None, cancel_check=None,
    )
    assert vu == {"crop": 1.0, "pub": 1.0}
