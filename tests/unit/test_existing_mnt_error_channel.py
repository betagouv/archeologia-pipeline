"""ROB-15 (audit v2) : les échecs par MNT doivent être routés sur un canal
d'erreur VISIBLE dans l'UI (reporter.error), pas seulement sur ``log`` (INFO,
filtré par la fenêtre qui n'affiche qu'à partir de USER_INFO=25).

La boucle d'isolation est stubée (process_items_isolated) : on teste le
routage des messages, pas le traitement GDAL.
"""
from __future__ import annotations

from pipeline.modes.existing_mnt import run_existing_mnt


def _fake_isolated_one_failure(items, process, *, cancel=None, on_failure=None):
    assert on_failure is not None
    on_failure(1, items[0], RuntimeError("asc corrompu"))
    return 0, [(1, items[0])]


def test_echec_mnt_et_recap_routes_vers_error_log(tmp_path, monkeypatch):
    mnt_dir = tmp_path / "mnt"
    mnt_dir.mkdir()
    (mnt_dir / "a.tif").write_bytes(b"x")  # jamais lu : la boucle est stubée

    monkeypatch.setattr(
        "pipeline.batch.process_items_isolated", _fake_isolated_one_failure
    )

    logs: list[str] = []
    errors: list[str] = []
    res = run_existing_mnt(
        existing_mnt_dir=mnt_dir,
        output_dir=tmp_path / "out",
        products={},
        output_structure={},
        output_formats={},
        rvt_params={},
        log=logs.append,
        error_log=errors.append,
    )

    assert res.total == 0
    # Échec par dalle visible côté UI.
    assert any("a.tif" in e and "échec" in e for e in errors)
    # Récapitulatif visible côté UI.
    assert any("1" in e and "échec" in e for e in errors if "a.tif" not in e)


def test_sans_error_log_les_messages_retombent_sur_log(tmp_path, monkeypatch):
    mnt_dir = tmp_path / "mnt"
    mnt_dir.mkdir()
    (mnt_dir / "a.tif").write_bytes(b"x")

    monkeypatch.setattr(
        "pipeline.batch.process_items_isolated", _fake_isolated_one_failure
    )

    logs: list[str] = []
    run_existing_mnt(
        existing_mnt_dir=mnt_dir,
        output_dir=tmp_path / "out",
        products={},
        output_structure={},
        output_formats={},
        rvt_params={},
        log=logs.append,
    )
    assert any("a.tif" in m and "échec" in m for m in logs)
