"""Un raster isolé trop petit pour le noyau RVT demandé doit être signalé.

Cas réel : commande LiDAR locale d'un archéologue (emprise de quelques centaines
de mètres) traitée avec les échelles MSTP par défaut (rayon large 2023 px). Il
n'y a qu'un raster, donc **aucune couture ne trahit le défaut** — RVT replie
l'emprise sur elle-même et le canal large est intégralement fabriqué. Sans cet
avertissement, l'image est plausible et sera lue comme un signal.

Même dispositif que ``test_existing_mnt_degenerate`` : boucle d'isolation stubée,
lecture raster monkeypatchée, traitement lourd neutralisé.
"""

from __future__ import annotations

from pathlib import Path

import pipeline.modes.existing_mnt as em
from pipeline.modes.existing_mnt import run_existing_mnt
from pipeline.tilespec import TileSpec

# 300 x 500 m à 0,5 m → 600 x 1000 px : bien plus petit que le noyau large.
_BOUNDS = (700000.0, 6599500.0, 700300.0, 6600000.0)


def _small_spec(path):
    return TileSpec.from_values(
        source_path=Path(path),
        bounds=_BOUNDS,
        pixel_size_x=0.5, pixel_size_y=-0.5,
        width_px=600, height_px=1000, crs="EPSG:2154",
    )


def _fake_isolated_calls_process(items, process, *, cancel=None, on_failure=None):
    for i, it in enumerate(items, start=1):
        process(i, it)
    return 0, []


def _run(tmp_path, monkeypatch, *, products, rvt_params):
    mnt_dir = tmp_path / "mnt"
    mnt_dir.mkdir()
    (mnt_dir / "commande_locale.tif").write_bytes(b"x")  # jamais lu réellement

    monkeypatch.setattr(
        "pipeline.batch.process_items_isolated", _fake_isolated_calls_process
    )
    monkeypatch.setattr(
        em.TileSpec, "from_raster", staticmethod(lambda p, **k: _small_spec(p))
    )
    monkeypatch.setattr(em, "get_raster_bounds", lambda _p: _BOUNDS)
    monkeypatch.setattr(em, "_process_single_mnt_tile", lambda **_kw: True)

    logs: list[str] = []
    errors: list[str] = []
    run_existing_mnt(
        existing_mnt_dir=mnt_dir,
        output_dir=tmp_path / "out",
        products=products, output_structure={}, output_formats={},
        rvt_params=rvt_params,
        log=logs.append, error_log=errors.append,
    )
    return logs, errors


def test_small_raster_with_default_mstp_is_reported(tmp_path, monkeypatch):
    _logs, errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={}
    )

    assert any("MSTP" in e and "2023" in e for e in errors)


def test_no_report_once_the_kernel_fits_the_raster(tmp_path, monkeypatch):
    _logs, errors = _run(
        tmp_path, monkeypatch,
        products={"MSTP": True},
        # rayon 100 px → (600-200)x(1000-200) = 53 % de l'emprise à voisinage
        # complet, au-dessus du seuil de dégénérescence.
        rvt_params={"mstp": {
            "broad_scale_min": 30, "broad_scale_max": 100, "broad_scale_step": 45
        }},
    )

    assert not any("MSTP" in e for e in errors)


def test_no_report_when_the_product_is_not_requested(tmp_path, monkeypatch):
    _logs, errors = _run(
        tmp_path, monkeypatch, products={"SVF": True}, rvt_params={}
    )

    assert not any("MSTP" in e for e in errors)
