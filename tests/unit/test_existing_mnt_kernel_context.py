"""Un raster isolé trop petit pour le noyau RVT demandé doit être signalé.

Cas réel : commande LiDAR locale d'un archéologue (emprise de quelques centaines
de mètres) traitée avec les échelles MSTP par défaut (rayon large 2023 px). Il
n'y a qu'un raster, donc **aucune couture ne trahit le défaut** — RVT replie
l'emprise sur elle-même et le canal large est intégralement fabriqué. Sans cet
avertissement, l'image est plausible et sera lue comme un signal.

Le signalement est un **avertissement** (⚠), pas une erreur (✗) : le raster est
calculé quand même. Routé sur le canal erreur, il gonflait le compteur ✗ et le
bandeau final annonçait « N erreurs » sur un run réussi (2026-09-22, 621 MNT
dont quatre lambeaux de bord de zone).

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


def _run(tmp_path, monkeypatch, *, products, rvt_params, n_files=1, warning_log=True):
    mnt_dir = tmp_path / "mnt"
    mnt_dir.mkdir()
    for i in range(n_files):
        (mnt_dir / f"commande_locale_{i}.tif").write_bytes(b"x")  # jamais lu réellement

    monkeypatch.setattr(
        "pipeline.batch.process_items_isolated", _fake_isolated_calls_process
    )
    monkeypatch.setattr(
        em.TileSpec, "from_raster", staticmethod(lambda p, **k: _small_spec(p))
    )
    monkeypatch.setattr(em, "get_raster_bounds", lambda _p: _BOUNDS)
    monkeypatch.setattr(em, "_process_single_mnt_tile", lambda **_kw: True)

    logs: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    channels = {"warning_log": warnings.append} if warning_log else {}
    run_existing_mnt(
        existing_mnt_dir=mnt_dir,
        output_dir=tmp_path / "out",
        products=products, output_structure={}, output_formats={},
        rvt_params=rvt_params,
        log=logs.append, error_log=errors.append, **channels,
    )
    return logs, warnings, errors


def test_small_raster_with_default_mstp_is_reported_as_a_warning(tmp_path, monkeypatch):
    _logs, warnings, errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={}
    )

    assert any("MSTP" in w and "2023" in w for w in warnings)
    assert not any("MSTP" in e for e in errors)


def test_warning_carries_no_glyph_of_its_own(tmp_path, monkeypatch):
    # Le journal préfixe déjà « ⚠ » : un « ⚠️ » dans le texte le doublerait.
    _logs, warnings, _errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={}
    )

    assert warnings and not warnings[0].startswith("⚠")


def test_a_single_raster_gets_the_direct_advice(tmp_path, monkeypatch):
    _logs, warnings, _errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={}
    )

    assert any("Réduisez" in w for w in warnings)


def test_in_a_batch_the_advice_is_scoped_to_the_raster(tmp_path, monkeypatch):
    _logs, warnings, _errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={}, n_files=2
    )

    assert any("tout le lot" in w for w in warnings)
    assert not any("Réduisez" in w for w in warnings)


def test_without_a_warning_channel_the_report_stays_visible(tmp_path, monkeypatch):
    # Un appelant qui ne fournit pas de canal avertissement ne doit pas perdre
    # le signalement : il retombe sur le canal erreur (visible), jamais sur
    # ``log`` (INFO, filtré par la fenêtre).
    _logs, _warnings, errors = _run(
        tmp_path, monkeypatch, products={"MSTP": True}, rvt_params={},
        warning_log=False,
    )

    assert any("MSTP" in e for e in errors)


def test_no_report_once_the_kernel_fits_the_raster(tmp_path, monkeypatch):
    _logs, warnings, errors = _run(
        tmp_path, monkeypatch,
        products={"MSTP": True},
        # rayon 100 px → (600-200)x(1000-200) = 53 % de l'emprise à voisinage
        # complet, au-dessus du seuil de dégénérescence.
        rvt_params={"mstp": {
            "broad_scale_min": 30, "broad_scale_max": 100, "broad_scale_step": 45
        }},
    )

    assert not any("MSTP" in m for m in warnings + errors)


def test_no_report_when_the_product_is_not_requested(tmp_path, monkeypatch):
    _logs, warnings, errors = _run(
        tmp_path, monkeypatch, products={"SVF": True}, rvt_params={}
    )

    assert not any("MSTP" in m for m in warnings + errors)
