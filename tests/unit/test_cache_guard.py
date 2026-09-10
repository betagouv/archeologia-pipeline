"""Garde de cache inter-runs (bug SRA HDF 2026-08-31 : densité/résolution MNT
ignorée au re-run dans le même output_dir).

Sémantique v2 (revue adversariale 2026-09-02) : la garde ne touche QUE le cache
``intermediaires/`` — jamais ``indices/`` (livrable accumulé multi-zones),
``sources/`` ni ``detections/``. La re-publication des finaux périmés est
assurée par la fraîcheur des mtimes côté ``results.py`` (needs_refresh).

- sidecar présent + signature différente (ou illisible) → purge d'intermediaires/ ;
- sidecar absent (dossier vierge OU antérieur au correctif OU nettoyé à la
  main) → adoption de la signature courante comme référence, sans purge ;
- rmtree bloqué par un verrou (couche QGIS ouverte) → RuntimeError actionnable,
  sidecar non réécrit (le run suivant re-purge).
"""
from __future__ import annotations

import pytest

from app.services.cache_guard import (
    SIDECAR_NAME,
    build_signature,
    ensure_cache_matches,
)


def _sig(mnt: float = 0.5, dens: float = 1.0, overlap: float = 5.0, filt: str = ""):
    return build_signature(
        mnt_resolution=mnt,
        density_resolution=dens,
        tile_overlap=overlap,
        filter_expression=filt,
    )


class TestBuildSignature:
    def test_equal_inputs_give_equal_signatures(self):
        assert _sig() == _sig()

    def test_changed_resolution_changes_signature(self):
        assert _sig(mnt=0.5) != _sig(mnt=10.0)

    def test_values_are_normalized(self):
        # int vs float et non-str : mêmes paramètres → même signature
        a = build_signature(
            mnt_resolution=1, density_resolution=1, tile_overlap=5, filter_expression=""
        )
        b = build_signature(
            mnt_resolution=1.0, density_resolution=1.0, tile_overlap=5.0, filter_expression=""
        )
        assert a == b


class TestEnsureCacheMatches:
    def test_fresh_output_dir_writes_sidecar_without_purge(self, tmp_path):
        inter = tmp_path / "intermediaires"
        purged = ensure_cache_matches(signature=_sig(), intermediaires=inter)
        assert purged is False
        assert (inter / SIDECAR_NAME).exists()

    def test_same_signature_keeps_cache(self, tmp_path):
        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(), intermediaires=inter)
        marker = inter / "dalle_MNT.tif"
        marker.write_bytes(b"x")
        purged = ensure_cache_matches(signature=_sig(), intermediaires=inter)
        assert purged is False
        assert marker.exists()

    def test_changed_signature_purges_intermediaires(self, tmp_path):
        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(mnt=0.5), intermediaires=inter)
        (inter / "dalle_MNT.tif").write_bytes(b"x")

        purged = ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter)

        assert purged is True
        assert not (inter / "dalle_MNT.tif").exists()
        # sidecar réécrit avec la nouvelle signature → un 3e run identique ne purge pas
        assert ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter) is False

    def test_purge_never_touches_indices_sources_nor_detections(self, tmp_path):
        # indices/ est le LIVRABLE accumulé (multi-zones, §22) — jamais purgé :
        # la re-publication des finaux périmés passe par needs_refresh (results.py).
        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(mnt=0.5), intermediaires=inter)
        (inter / "dalle_MNT.tif").write_bytes(b"x")
        mnt_tif = tmp_path / "indices" / "MNT" / "tif" / "zone_A.tif"
        mnt_tif.parent.mkdir(parents=True)
        mnt_tif.write_bytes(b"x")
        laz = tmp_path / "sources" / "dalles" / "dalle.laz"
        laz.parent.mkdir(parents=True)
        laz.write_bytes(b"x")
        gpkg = tmp_path / "detections" / "parcellaire" / "parcellaire.gpkg"
        gpkg.parent.mkdir(parents=True)
        gpkg.write_bytes(b"x")

        assert ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter) is True

        assert mnt_tif.exists()
        assert laz.exists()
        assert gpkg.exists()

    def test_missing_sidecar_adopts_baseline_without_purge(self, tmp_path):
        # Dossier d'avant le correctif (ou intermediaires/ nettoyé à la main) :
        # provenance inconnue → on adopte la signature courante SANS détruire le
        # cache (re-run §22 « ajout de dalle » à paramètres identiques = reprise
        # rapide, pas des heures de refusion). L'invalidation joue dès le run
        # suivant ; les finaux périmés sont couverts par needs_refresh.
        inter = tmp_path / "intermediaires"
        inter.mkdir()
        (inter / "dalle_MNT.tif").write_bytes(b"x")
        purged = ensure_cache_matches(signature=_sig(), intermediaires=inter)
        assert purged is False
        assert (inter / "dalle_MNT.tif").exists()
        assert (inter / SIDECAR_NAME).exists()

    def test_corrupted_sidecar_purges(self, tmp_path):
        # Sidecar présent mais illisible = provenance prouvée douteuse → purge.
        inter = tmp_path / "intermediaires"
        inter.mkdir()
        (inter / SIDECAR_NAME).write_text("{pas du json", encoding="utf-8")
        (inter / "dalle_MNT.tif").write_bytes(b"x")
        assert ensure_cache_matches(signature=_sig(), intermediaires=inter) is True
        assert not (inter / "dalle_MNT.tif").exists()

    def test_purge_logs_explicit_message(self, tmp_path):
        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(mnt=0.5), intermediaires=inter)
        (inter / "dalle_MNT.tif").write_bytes(b"x")
        messages: list = []
        ensure_cache_matches(
            signature=_sig(mnt=10.0), intermediaires=inter, log=messages.append
        )
        assert any("cache" in m.lower() for m in messages)

    def test_purge_removes_nested_directories(self, tmp_path):
        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(mnt=0.5), intermediaires=inter)
        sub = inter / "sous_dossier"
        sub.mkdir()
        (sub / "x.laz").write_bytes(b"x")
        assert ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter) is True
        assert not sub.exists()

    def test_partial_purge_failure_keeps_stale_sidecar(self, tmp_path, monkeypatch):
        # Un rmtree naïf du dossier entier supprimait run_params.json ('r')
        # AVANT un fichier verrouillé qui trie après ('z…', cas local_laz) : le
        # run suivant adoptait la NOUVELLE signature sur un cache périmé,
        # définitivement. La purge doit garder le sidecar pour la fin : en cas
        # d'échec, l'ANCIENNE signature survit → re-purge garantie.
        import pathlib

        inter = tmp_path / "intermediaires"
        ensure_cache_matches(signature=_sig(mnt=0.5), intermediaires=inter)
        (inter / "zone_sud_MNT.tif").write_bytes(b"x")

        real_unlink = pathlib.Path.unlink

        def _locked(self, *a, **k):
            if self.name == "zone_sud_MNT.tif":
                raise PermissionError(32, "Le processus ne peut pas accéder au fichier")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "unlink", _locked)
        with pytest.raises(RuntimeError, match="verrouillé"):
            ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter)
        monkeypatch.undo()

        import json

        previous = json.loads((inter / SIDECAR_NAME).read_text(encoding="utf-8"))
        assert previous == _sig(mnt=0.5)  # l'ancienne signature survit…
        # …et le run suivant re-purge bien une fois le verrou levé.
        assert ensure_cache_matches(signature=_sig(mnt=10.0), intermediaires=inter) is True
        assert not (inter / "zone_sud_MNT.tif").exists()
