"""Halo inter-dalles : re-fusion quand le jeu de voisins change (revue v4).

Scénario « extension de zone » : run 1 sélection {A} (marge est de A =
fabriquée), run 2 {A, B} mêmes paramètres → le crop de B était créé puis JETÉ
(merge_tiles réutilisait A_merged.laz sur existence pure) ; la marge restait
fabriquée et la conversion re-publiait les détections de marge du run 1 dans
la cellule de B (le clip s'élargit à l'union des TIF accumulés).

Contrat : le LAZ fusionné porte un sidecar ``<tile>_merged.inputs.json`` (jeu
de voisins candidats) ; jeu différent → re-fusion, dont le mtime frais ré-arme
toute la chaîne de fraîcheur (MNT → RVT → PNG → cache CV → ré-inférence) —
d'où la fraîcheur MNT/densité vis-à-vis du LAZ fusionné testée ici aussi.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from pipeline.ign.preprocess import (
    merge_tiles,
    merged_inputs_match,
    merged_inputs_sidecar,
    write_merged_inputs_sidecar,
)

TILE = "LHD_FXX_0623_6864"


class TestMergedInputsSidecar:
    def test_same_set_matches_regardless_of_order(self, tmp_path):
        out = tmp_path / "A_merged.laz"
        out.write_bytes(b"laz")
        write_merged_inputs_sidecar(out, [tmp_path / "b.laz", tmp_path / "a.laz"])
        assert merged_inputs_match(out, [tmp_path / "a.laz", tmp_path / "b.laz"]) is True

    def test_different_set_mismatches(self, tmp_path):
        out = tmp_path / "A_merged.laz"
        out.write_bytes(b"laz")
        write_merged_inputs_sidecar(out, [])
        assert merged_inputs_match(out, [tmp_path / "b.laz"]) is False

    def test_missing_sidecar_is_unknown(self, tmp_path):
        out = tmp_path / "A_merged.laz"
        out.write_bytes(b"laz")
        assert merged_inputs_match(out, []) is None

    def test_corrupt_sidecar_mismatches(self, tmp_path):
        out = tmp_path / "A_merged.laz"
        out.write_bytes(b"laz")
        merged_inputs_sidecar(out).write_text("{pas du json", encoding="utf-8")
        assert merged_inputs_match(out, []) is False


class TestMergeTilesSetFreshness:
    def _drive(self, tmp_path, monkeypatch, neighbors):
        import pipeline.ign.preprocess as pre_mod

        merges: list = []

        def _fake_merge(cmd, cancel=None):
            merges.append(cmd)
            Path(cmd[-1]).write_bytes(b"merged")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(pre_mod, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
        monkeypatch.setattr(pre_mod, "run_pdal_command_cancellable", _fake_merge)
        monkeypatch.setattr(pre_mod, "_pdal_exe", lambda: "pdal")

        central = tmp_path / f"{TILE}.laz"
        central.write_bytes(b"central")
        out = tmp_path / f"{TILE}_merged.laz"
        ok = merge_tiles(
            central_path=central, neighbor_paths=neighbors, output_path=out
        )
        return ok, merges, out

    def test_new_neighbor_forces_remerge(self, tmp_path, monkeypatch):
        # Run 1 : A seule (jeu vide, sidecar écrit). Run 2 : voisin B apparu.
        _ok, _m, out = self._drive(tmp_path, monkeypatch, [])
        assert merged_inputs_sidecar(out).exists()
        b = tmp_path / "B_neighbor_est.laz"
        b.write_bytes(b"crop")
        ok, merges, out = self._drive(tmp_path, monkeypatch, [b])
        assert ok is True
        assert len(merges) == 1  # re-fusion effective
        assert merged_inputs_match(out, [b]) is True  # sidecar mis à jour

    def test_same_set_reuses_cache(self, tmp_path, monkeypatch):
        b = tmp_path / "B_neighbor_est.laz"
        b.write_bytes(b"crop")
        self._drive(tmp_path, monkeypatch, [b])
        ok, merges, _out = self._drive(tmp_path, monkeypatch, [b])
        assert ok is True
        assert merges == []  # cache réutilisé

    def test_legacy_without_sidecar_adopts_current_set(self, tmp_path, monkeypatch):
        # Fusionné d'avant le correctif : pas de re-fusion forcée, le jeu
        # courant devient la référence (cohérent avec l'adoption cache_guard).
        central = tmp_path / f"{TILE}.laz"
        central.write_bytes(b"central")
        out = tmp_path / f"{TILE}_merged.laz"
        out.write_bytes(b"legacy")
        ok, merges, out2 = self._drive(tmp_path, monkeypatch, [])
        assert ok is True
        assert merges == []
        assert merged_inputs_sidecar(out).exists()


class TestTerrainAndDensityFreshness:
    """MNT/densité vs LAZ fusionné : un re-merge (mtime frais) force le recalcul."""

    def _mnt(self, tmp_path, monkeypatch, laz_delta):
        import pipeline.ign.products.mnt as mnt_mod
        from pipeline.ign.products.mnt import create_terrain_model

        calls: list = []

        def _fake_algo(algo_id, params, **_k):
            calls.append(algo_id)
            Path(params["OUTPUT"]).write_bytes(b"mnt")

        monkeypatch.setattr(mnt_mod, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
        monkeypatch.setattr(mnt_mod, "run_qgis_algorithm", _fake_algo)
        laz = tmp_path / f"{TILE}.laz"
        laz.write_bytes(b"laz")
        out = tmp_path / f"{TILE}_MNT.tif"
        out.write_bytes(b"mnt")
        t = out.stat().st_mtime
        os.utime(laz, (t + laz_delta, t + laz_delta))
        create_terrain_model(
            input_laz_path=laz,
            temp_dir=tmp_path,
            current_tile_name=TILE,
            mnt_resolution=0.5,
            tile_overlap_percent=5.0,
            filter_expression="",
        )
        return calls

    def test_mnt_cached_when_laz_older(self, tmp_path, monkeypatch):
        assert self._mnt(tmp_path, monkeypatch, laz_delta=-10) == []

    def test_mnt_recomputed_when_laz_fresher(self, tmp_path, monkeypatch):
        assert len(self._mnt(tmp_path, monkeypatch, laz_delta=+10)) == 1

    def _density(self, tmp_path, monkeypatch, laz_delta):
        import pipeline.ign.products.density as dens_mod
        from pipeline.ign.products.density import create_density_map

        calls: list = []

        def _fake_algo(algo_id, params, **_k):
            calls.append(algo_id)
            Path(params["OUTPUT"]).write_bytes(b"dens")

        monkeypatch.setattr(dens_mod, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
        monkeypatch.setattr(dens_mod, "run_qgis_algorithm", _fake_algo)
        laz = tmp_path / f"{TILE}.laz"
        laz.write_bytes(b"laz")
        out = tmp_path / f"{TILE}_densite.tif"
        out.write_bytes(b"dens")
        t = out.stat().st_mtime
        os.utime(laz, (t + laz_delta, t + laz_delta))
        create_density_map(
            input_laz_path=laz,
            temp_dir=tmp_path,
            current_tile_name=TILE,
            density_resolution=1.0,
            tile_overlap_percent=5.0,
            filter_expression="",
        )
        return calls

    def test_density_cached_when_laz_older(self, tmp_path, monkeypatch):
        assert self._density(tmp_path, monkeypatch, laz_delta=-10) == []

    def test_density_recomputed_when_laz_fresher(self, tmp_path, monkeypatch):
        assert len(self._density(tmp_path, monkeypatch, laz_delta=+10)) == 1
