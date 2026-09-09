"""Publication par fraîcheur (revue adversariale 2026-09-02).

Régression visée : ``copy_final_products_to_results`` refusait d'écraser un TIF
final existant (``not tif_path.exists()``) — un intermédiaire recalculé après
invalidation du cache (nouvelle résolution MNT, produit recoché plus tard,
``intermediaires/`` nettoyé à la main…) n'était jamais re-publié : ``indices/``
et le VRT resservaient silencieusement l'ancienne résolution (bug SRA HDF
2026-08-31, classe « paramètre ignoré »).

Nouveau contrat : publier quand la source est PLUS RÉCENTE que la destination
(``needs_refresh``). ``shutil.copy2`` préservant les mtimes, un re-run à cache
identique ne re-copie rien (reprise §22/§28.4 intacte).
"""
from __future__ import annotations

import os
import shutil

import pytest

from pipeline.ign.products.results import (
    copy_final_products_to_results,
    needs_refresh,
)
from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames

TILE = "LHD_FXX_0623_6864"


class TestNeedsRefresh:
    def test_missing_dest_needs_refresh(self, tmp_path):
        src = tmp_path / "src.tif"
        src.write_bytes(b"x")
        assert needs_refresh(src, tmp_path / "absent.tif") is True

    def test_newer_source_needs_refresh(self, tmp_path):
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        dst.write_bytes(b"old")
        src.write_bytes(b"new")
        t = dst.stat().st_mtime
        os.utime(src, (t + 10, t + 10))
        assert needs_refresh(src, dst) is True

    def test_copy2_result_is_up_to_date(self, tmp_path):
        # copy2 préserve le mtime → la destination fraîchement publiée est à jour.
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        src.write_bytes(b"x")
        shutil.copy2(src, dst)
        assert needs_refresh(src, dst) is False

    def test_newer_dest_is_up_to_date(self, tmp_path):
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        src.write_bytes(b"x")
        dst.write_bytes(b"x")
        t = src.stat().st_mtime
        os.utime(dst, (t + 10, t + 10))
        assert needs_refresh(src, dst) is False


class TestFreshPublication:
    """Comportement de bout en bout sur le produit MNT (fixture rasterio)."""

    def _publish(self, tmp_path, monkeypatch):
        import pipeline.ign.products.results as results_mod

        monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **k: True)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        messages: list = []
        copy_final_products_to_results(
            temp_dir=temp_dir,
            output_dir=tmp_path / "out",
            current_tile_name=TILE,
            products={"MNT": True},
            output_structure={},
            output_formats={"tif": True},
            rvt_params={},
            log=messages.append,
        )
        return messages

    def _write_source(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import numpy as np
        from rasterio.transform import from_origin

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        _src, dest_name = get_rvt_source_and_dest_filenames("MNT", TILE, "0623", "6864", {})
        path = temp_dir / dest_name
        with rasterio.open(
            str(path), "w", driver="GTiff", width=20, height=10, count=1,
            dtype="float32", crs="EPSG:2154",
            transform=from_origin(623000.0, 6864000.0, 0.5, 0.5),
        ) as ds:
            ds.write(np.zeros((1, 10, 20), dtype="float32"))
        return path

    def test_stale_final_is_republished(self, tmp_path, monkeypatch):
        src = self._write_source(tmp_path)
        self._publish(tmp_path, monkeypatch)
        # Cache invalidé → intermédiaire recalculé (plus récent que le publié)
        src.unlink()
        src = self._write_source(tmp_path)
        published = next((tmp_path / "out" / "indices" / "MNT" / "tif").glob("*.tif"))
        t = published.stat().st_mtime
        os.utime(src, (t + 10, t + 10))

        messages = self._publish(tmp_path, monkeypatch)

        assert any("copié" in m for m in messages)
        assert published.stat().st_mtime == pytest.approx(t + 10)

    def test_up_to_date_final_is_not_recopied(self, tmp_path, monkeypatch):
        # Reprise à cache identique (§22/§28.4) : pas de re-copie inutile.
        self._write_source(tmp_path)
        self._publish(tmp_path, monkeypatch)
        messages = self._publish(tmp_path, monkeypatch)
        assert not any("copié" in m for m in messages)

    def test_published_tif_keeps_source_mtime_despite_pyramids(self, tmp_path, monkeypatch):
        # gdaladdo (overviews internes) « rajeunit » le TIF publié → sur les
        # chaînes copy2 (existing_mnt, mtimes hérités du fichier source), un
        # intermédiaire re-matérialisé perdrait la comparaison de fraîcheur.
        # La publication re-tamponne le mtime depuis la source (copystat).
        import time

        import pipeline.ign.products.results as results_mod

        def _fake_pyramids(path, **_k):
            t = time.time() + 3600
            os.utime(path, (t, t))
            return True

        monkeypatch.setattr(results_mod, "build_raster_pyramids", _fake_pyramids)
        src = self._write_source(tmp_path)
        copy_final_products_to_results(
            temp_dir=tmp_path / "temp",
            output_dir=tmp_path / "out",
            current_tile_name=TILE,
            products={"MNT": True},
            output_structure={},
            output_formats={"tif": True},
            rvt_params={},
        )
        published = next((tmp_path / "out" / "indices" / "MNT" / "tif").glob("*.tif"))
        assert published.stat().st_mtime == pytest.approx(src.stat().st_mtime)


class TestCopyWithoutCropFreshness:
    """copy_products_without_crop (layouts small/large) : même contrat de fraîcheur."""

    def _paths(self, tmp_path):
        from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(exist_ok=True)
        src_name, dst_name = get_rvt_source_and_dest_filenames("MNT", TILE, "0623", "6864", {})
        return temp_dir, temp_dir / src_name, temp_dir / dst_name

    def _run(self, temp_dir):
        from pipeline.ign.products.crop import copy_products_without_crop

        return copy_products_without_crop(
            temp_dir=temp_dir,
            current_tile_name=TILE,
            products={"MNT": True},
            rvt_params={},
        )

    def test_stale_dst_is_recopied(self, tmp_path):
        temp_dir, src, dst = self._paths(tmp_path)
        dst.write_bytes(b"ancien")
        src.write_bytes(b"recalcule")
        t = dst.stat().st_mtime
        os.utime(src, (t + 10, t + 10))
        self._run(temp_dir)
        assert dst.read_bytes() == b"recalcule"

    def test_fresh_dst_is_kept(self, tmp_path):
        temp_dir, src, dst = self._paths(tmp_path)
        src.write_bytes(b"v1")
        shutil.copy2(src, dst)
        src.write_bytes(b"v2")  # même mtime ± : rétrograde src pour simuler cache intact
        t = dst.stat().st_mtime
        os.utime(src, (t - 10, t - 10))
        self._run(temp_dir)
        assert dst.read_bytes() == b"v1"
