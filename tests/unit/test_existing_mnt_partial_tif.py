"""ROB-16 (audit v2) : un TIF temporaire PARTIEL laissé par un échec
gdal_translate (returncode ≠ 0 — disque plein, kill) était silencieusement
réutilisé au re-run (« déjà converti » via l'early-return) → indices RVT
calculés sur des données tronquées, sans aucun message.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.modes.existing_mnt import _copy_source_mnt_to_temp


def test_echec_gdal_translate_supprime_le_tif_partiel(tmp_path, monkeypatch):
    src = tmp_path / "mnt.asc"
    src.write_text("ncols 2\nnrows 2\n")
    out = tmp_path / "temp_mnt.tif"

    def fake_run(cmd, cancel=None, output_path=None):
        output_path.write_bytes(b"PARTIEL")  # sortie tronquée laissée par GDAL
        return SimpleNamespace(returncode=1, stderr="ERROR 3: disk full", stdout="")

    monkeypatch.setattr(
        "pipeline.modes.existing_mnt.run_subprocess_cancellable", fake_run
    )

    with pytest.raises(RuntimeError):
        _copy_source_mnt_to_temp(
            source_path=src, temp_mnt_path=out, gdal_translate="gdal_translate"
        )
    assert not out.exists(), "le TIF partiel doit être supprimé sur échec"


def test_residu_vide_n_est_pas_reutilise(tmp_path):
    src = tmp_path / "mnt.tif"
    src.write_bytes(b"CONTENU_TIF_VALIDE")
    out = tmp_path / "temp_mnt.tif"
    out.write_bytes(b"")  # résidu 0 octet d'un crash antérieur

    _copy_source_mnt_to_temp(source_path=src, temp_mnt_path=out, gdal_translate=None)

    assert out.read_bytes() == b"CONTENU_TIF_VALIDE"


def test_tif_existant_non_vide_est_reutilise(tmp_path):
    src = tmp_path / "mnt.tif"
    src.write_bytes(b"NOUVEAU")
    out = tmp_path / "temp_mnt.tif"
    out.write_bytes(b"DEJA_CONVERTI")

    _copy_source_mnt_to_temp(source_path=src, temp_mnt_path=out, gdal_translate=None)

    # Comportement existant conservé : pas de reconversion inutile.
    assert out.read_bytes() == b"DEJA_CONVERTI"
