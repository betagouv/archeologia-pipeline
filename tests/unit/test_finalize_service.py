from __future__ import annotations

from pathlib import Path

from app.services import finalize_service


class TestCollectVrtPathsAndBuild:
    def test_returns_existing_tif_and_png_vrts(self, tmp_path: Path):
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_vrt = idx_dir / "LD" / "tif" / "index.vrt"
        png_vrt = idx_dir / "LD" / "png" / "index.vrt"
        for path in (tif_vrt, png_vrt):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("vrt", encoding="utf-8")

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert set(paths) == {str(tif_vrt), str(png_vrt)}

    def test_does_not_build_or_return_vrt_for_annotated_images(self, tmp_path: Path, monkeypatch):
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        png_dir = idx_dir / "LD" / "png"
        annotated_dir = det_dir / "model" / "annotated_images"
        png_dir.mkdir(parents=True)
        annotated_dir.mkdir(parents=True)
        (png_dir / "tile.png").write_bytes(b"png")
        (annotated_dir / "tile.png").write_bytes(b"png")

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            (folder / output_name).write_text(pattern, encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert str(png_dir / "index.vrt") in paths
        assert not (annotated_dir / "index.vrt").exists()
        assert str(annotated_dir / "index.vrt") not in paths


class TestCollectDetectionLayers:
    def test_collects_geopackage_layers_from_shapefiles_directory(self, tmp_path: Path, monkeypatch):
        gpkg = tmp_path / "detections" / "model" / "shapefiles" / "detections.gpkg"
        gpkg.parent.mkdir(parents=True)
        gpkg.write_bytes(b"gpkg")
        monkeypatch.setattr(finalize_service, "_list_gpkg_layers", lambda _p: ["cratere", "zone"])

        paths = finalize_service._collect_shapefiles(tmp_path / "detections")

        assert paths == [
            f"{gpkg}|layername=cratere",
            f"{gpkg}|layername=zone",
        ]
