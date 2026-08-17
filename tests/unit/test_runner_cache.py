"""Cache d'inférence : invalidation quand le PNG est plus récent que la détection.

Option B (halo inter-dalles) : le passage aux PNG à marge régénère les images
sans changer leurs stems. Un ``raw_detections/<stem>.txt`` calculé sur l'ancien
PNG rogné serait réutilisé tel quel (coordonnées normalisées → décalage de la
marge). La purge des caches plus anciens que leur PNG force la ré-inférence —
côté plugin comme côté binaire externe (qui saute par simple existence des
fichiers, sans regarder les dates).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv.runner_cache import purge_stale_cached_detections


_OLD = 1_000_000_000
_NEW = 1_000_000_100


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


class TestPurgeStaleCachedDetections:
    def test_cache_plus_ancien_que_le_png_purge(self, tmp_path):
        png = _touch(tmp_path / "png" / "dalle.png", _NEW)
        raw = tmp_path / "raw"
        txt = _touch(raw / "dalle.txt", _OLD)
        js = _touch(raw / "dalle.json", _OLD)

        purged = purge_stale_cached_detections(raw, [png])

        assert purged == 1
        assert not txt.exists() and not js.exists()

    def test_cache_plus_recent_que_le_png_conserve(self, tmp_path):
        png = _touch(tmp_path / "png" / "dalle.png", _OLD)
        raw = tmp_path / "raw"
        txt = _touch(raw / "dalle.txt", _NEW)

        purged = purge_stale_cached_detections(raw, [png])

        assert purged == 0
        assert txt.exists()

    def test_json_recent_protege_le_couple(self, tmp_path):
        # Le binaire peut réécrire le .json sans toucher le .txt : le couple
        # est valide si le PLUS RÉCENT des deux date d'après le PNG.
        png = _touch(tmp_path / "png" / "dalle.png", _OLD + 50)
        raw = tmp_path / "raw"
        txt = _touch(raw / "dalle.txt", _OLD)
        _touch(raw / "dalle.json", _NEW)

        purged = purge_stale_cached_detections(raw, [png])

        assert purged == 0
        assert txt.exists()

    def test_sans_cache_aucun_effet(self, tmp_path):
        png = _touch(tmp_path / "png" / "dalle.png", _NEW)
        raw = tmp_path / "raw"
        raw.mkdir()

        assert purge_stale_cached_detections(raw, [png]) == 0

    def test_png_absent_conserve_le_cache(self, tmp_path):
        raw = tmp_path / "raw"
        txt = _touch(raw / "dalle.txt", _OLD)

        purged = purge_stale_cached_detections(raw, [tmp_path / "png" / "dalle.png"])

        assert purged == 0
        assert txt.exists()

    def test_raw_dir_inexistant(self, tmp_path):
        png = _touch(tmp_path / "png" / "dalle.png", _NEW)

        assert purge_stale_cached_detections(tmp_path / "absent", [png]) == 0
