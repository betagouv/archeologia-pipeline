from __future__ import annotations

from pathlib import Path

from app.services import finalize_service


class TestCollectVrtPathsAndBuild:
    def test_builds_and_returns_vrt_for_tif_dirs_only(self, tmp_path: Path, monkeypatch):
        """Seuls les dossiers tif/ sont indexés ; png/ et annotated_images/ sont ignorés."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        png_dir = idx_dir / "LD" / "png"
        annotated_dir = det_dir / "model" / "annotated_images"
        for d in (tif_dir, png_dir, annotated_dir):
            d.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (png_dir / "tile.png").write_bytes(b"png")
        (annotated_dir / "tile.png").write_bytes(b"png")

        built: list[Path] = []

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            built.append(folder)
            (folder / output_name).write_text(pattern, encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert built == [tif_dir]
        assert paths == [str(tif_dir / "index.vrt")]
        assert not (png_dir / "index.vrt").exists()
        assert not (annotated_dir / "index.vrt").exists()

    def test_always_rebuilds_existing_vrt(self, tmp_path: Path, monkeypatch):
        """Un index.vrt déjà présent est systématiquement régénéré (pas de skip-if-exists)."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (tif_dir / "index.vrt").write_text("stale", encoding="utf-8")  # VRT obsolète déjà présent

        built: list[Path] = []

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            built.append(folder)
            (folder / output_name).write_text("fresh", encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert built == [tif_dir]  # rebuild déclenché malgré le VRT existant
        assert paths == [str(tif_dir / "index.vrt")]
        assert (tif_dir / "index.vrt").read_text(encoding="utf-8") == "fresh"

    def test_skips_tif_dir_without_tif_files(self, tmp_path: Path, monkeypatch):
        """Un dossier tif/ sans .tif ne génère ni ne retourne de VRT, même avec un index.vrt résiduel."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "index.vrt").write_text("stale", encoding="utf-8")  # résiduel, sans .tif

        def fail_build(*_a, **_k):
            raise AssertionError("build_vrt_index ne doit pas être appelé sans .tif")

        monkeypatch.setattr("pipeline.ign.products.results.build_vrt_index", fail_build)

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert paths == []


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
