"""ROB-12 (audit v2) : la phase fusion était tout-ou-rien.

Une dalle dont le merge PDAL échoue — ou dont le fichier central manque
(download échoué mais nom parsable dans fichier_tri.txt, ROB-08 v1) —
faisait avorter TOUT le run via un RuntimeError / FileNotFoundError fatal,
levé AVANT le try/finally du runner (donc sans finalisation).

Attendu : la fusion continue avec les dalles valides et rapporte les échecs
dans ``result.failed`` ; seul l'échec TOTAL (aucune dalle fusionnée) reste
fatal.
"""
from __future__ import annotations

import pytest

import pipeline.ign.preprocess as pp

TILE_A = "LHD_FXX_0500_6500_PTS_C.laz"
TILE_B = "LHD_FXX_0501_6500_PTS_C.laz"


def _write_sorted_list(tmp_path, names):
    f = tmp_path / "fichier_tri.txt"
    f.write_text("".join(f"{n},https://x/{n}\n" for n in names), encoding="utf-8")
    return f


def _fake_process_ok(task, file_index, log, cancel):
    merged = task.merged_dir / f"{task.tile_name}_merged.laz"
    merged.write_bytes(b"LAZ")
    return pp._PreprocessResult(
        index=task.index, tile_name=task.tile_name, merged_path=merged, success=True
    )


def _fake_process_fail(task, file_index, log, cancel):
    return pp._PreprocessResult(
        index=task.index, tile_name=task.tile_name, merged_path=None,
        success=False, error="PDAL merge failed (code=1)",
    )


@pytest.fixture
def patched_coords(monkeypatch):
    monkeypatch.setattr(
        pp, "_extract_coordinates", lambda filename, **kw: ("0500", "6500")
    )


def _run(tmp_path, dalles, sorted_list):
    return pp.prepare_merged_tiles(
        sorted_list_file=sorted_list,
        dalles_dir=dalles,
        output_dir=tmp_path / "out",
        tile_overlap_percent=20,
    )


def test_central_manquant_n_avorte_pas_le_lot(tmp_path, monkeypatch, patched_coords):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    (dalles / TILE_A).write_bytes(b"x")  # TILE_B ABSENTE (download échoué)
    sorted_list = _write_sorted_list(tmp_path, [TILE_A, TILE_B])
    monkeypatch.setattr(pp, "_process_single_tile_preprocess", _fake_process_ok)

    res = _run(tmp_path, dalles, sorted_list)

    assert len(res.merged_files) == 1
    assert len(res.failed) == 1
    assert TILE_B in res.failed[0]


def test_echec_merge_n_avorte_pas_le_lot(tmp_path, monkeypatch, patched_coords):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    (dalles / TILE_A).write_bytes(b"x")
    (dalles / TILE_B).write_bytes(b"x")
    sorted_list = _write_sorted_list(tmp_path, [TILE_A, TILE_B])

    def _fake(task, file_index, log, cancel):
        if task.filename == TILE_A:
            return _fake_process_fail(task, file_index, log, cancel)
        return _fake_process_ok(task, file_index, log, cancel)

    monkeypatch.setattr(pp, "_process_single_tile_preprocess", _fake)

    res = _run(tmp_path, dalles, sorted_list)

    assert len(res.merged_files) == 1
    assert len(res.failed) == 1
    assert "PDAL merge failed" in res.failed[0]


def test_echec_total_reste_fatal(tmp_path, monkeypatch, patched_coords):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    (dalles / TILE_A).write_bytes(b"x")
    (dalles / TILE_B).write_bytes(b"x")
    sorted_list = _write_sorted_list(tmp_path, [TILE_A, TILE_B])
    monkeypatch.setattr(pp, "_process_single_tile_preprocess", _fake_process_fail)

    with pytest.raises(RuntimeError):
        _run(tmp_path, dalles, sorted_list)


def test_nominal_sans_echec(tmp_path, monkeypatch, patched_coords):
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    (dalles / TILE_A).write_bytes(b"x")
    sorted_list = _write_sorted_list(tmp_path, [TILE_A])
    monkeypatch.setattr(pp, "_process_single_tile_preprocess", _fake_process_ok)

    res = _run(tmp_path, dalles, sorted_list)

    assert len(res.merged_files) == 1
    assert res.failed == []
