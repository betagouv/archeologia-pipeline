"""Fraîcheur des produits RVT vis-à-vis de leur MNT source (convergence v3).

Le cache des 9 blocs RVT était à existence pure : un MNT temp re-matérialisé
(source remplacée en mode existing_mnt, suppression manuelle) ne déclenchait
jamais le recalcul des indices dérivés — TIF publiés, PNG et détections
restaient silencieusement calculés sur l'ancien terrain (voire réécrits en
« chimère » pixels-anciens/masque-nouveau par la boucle reclass).
Contrat : produit recalculé si le MNT source est PLUS RÉCENT que lui.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from pipeline.ign.products.indices import create_visualization_products
from pipeline.ign.products.crop import crop_final_products
from pipeline.ign.products.rvt_naming import get_rvt_temp_filename, get_rvt_source_and_dest_filenames

TILE = "LHD_FXX_0623_6864"


class TestVisualizationProductsFreshness:
    def _setup(self, tmp_path):
        mnt = tmp_path / f"{TILE}_MNT.tif"
        mnt.write_bytes(b"mnt")
        out = tmp_path / get_rvt_temp_filename("HS", TILE, {})
        out.write_bytes(b"hs")
        return mnt, out

    def _run(self, tmp_path, monkeypatch):
        import pipeline.ign.products.indices as indices_mod

        calls: list = []
        monkeypatch.setattr(
            indices_mod, "run_qgis_algorithm", lambda *a, **k: calls.append(a)
        )
        create_visualization_products(
            temp_dir=tmp_path,
            current_tile_name=TILE,
            products={"HS": True},
            rvt_params={},
        )
        return calls

    def test_cached_product_with_older_mnt_is_reused(self, tmp_path, monkeypatch):
        mnt, out = self._setup(tmp_path)
        t = out.stat().st_mtime
        os.utime(mnt, (t - 10, t - 10))  # produit généré APRÈS son MNT : à jour
        assert self._run(tmp_path, monkeypatch) == []

    def test_fresher_mnt_forces_recompute(self, tmp_path, monkeypatch):
        mnt, out = self._setup(tmp_path)
        t = out.stat().st_mtime
        os.utime(mnt, (t + 10, t + 10))  # MNT re-matérialisé plus récent
        assert len(self._run(tmp_path, monkeypatch)) == 1


class TestCropFinalProductsFreshness:
    def _paths(self, tmp_path):
        src_name, dst_name = get_rvt_source_and_dest_filenames("MNT", TILE, "0623", "6864", {})
        return tmp_path / src_name, tmp_path / dst_name

    def _run(self, tmp_path, monkeypatch):
        import pipeline.ign.products.crop as crop_mod

        calls: list = []

        def _fake_warp(cmd, **kwargs):
            calls.append(cmd)
            out = kwargs.get("output_path")
            if out is not None:
                Path(out).write_bytes(b"cropped")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(crop_mod, "run_subprocess_cancellable", _fake_warp)
        crop_final_products(
            temp_dir=tmp_path,
            current_tile_name=TILE,
            products={"MNT": True},
            rvt_params={},
            gdalwarp_path="gdalwarp",
        )
        return calls

    def test_fresh_crop_is_kept(self, tmp_path, monkeypatch):
        src, dst = self._paths(tmp_path)
        src.write_bytes(b"produit")
        dst.write_bytes(b"rogne")  # rogné APRÈS le produit (gdalwarp) : à jour
        t = dst.stat().st_mtime
        os.utime(src, (t - 10, t - 10))
        assert self._run(tmp_path, monkeypatch) == []

    def test_stale_crop_is_redone(self, tmp_path, monkeypatch):
        src, dst = self._paths(tmp_path)
        dst.write_bytes(b"rogne")
        src.write_bytes(b"recalcule")
        t = dst.stat().st_mtime
        os.utime(src, (t + 10, t + 10))  # produit recalculé plus récent
        calls = self._run(tmp_path, monkeypatch)
        assert len(calls) == 1
        # gdalwarp refuse d'écraser un dataset existant sans -overwrite
        # (« Output dataset exists… Please delete existing dataset ») — jamais
        # atteint avant la fraîcheur, puisqu'un rogné existant sautait toujours.
        assert "-overwrite" in calls[0]

    def test_missing_source_keeps_existing_crop(self, tmp_path, monkeypatch):
        # Comportement historique : source absente mais rogné présent → gardé.
        _src, dst = self._paths(tmp_path)
        dst.write_bytes(b"rogne")
        assert self._run(tmp_path, monkeypatch) == []
        assert dst.exists()
