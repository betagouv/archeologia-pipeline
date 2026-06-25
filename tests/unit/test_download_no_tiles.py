"""Quand AUCUNE dalle n'aboutit (réseau coupé / proxy invalide), le téléchargement
doit s'arrêter tout de suite avec un message actionnable (réseau/proxy) — pas
laisser l'échec remonter au stade fusion sous la forme cryptique « fichier
central introuvable », ni en « Impossible de déterminer les coordonnées des
dalles » (tri post-download sur une liste vide).
"""
from __future__ import annotations

import pytest

import pipeline.ign.downloader as dl


def test_zero_dalle_telechargee_leve_erreur_reseau_proxy(tmp_path, monkeypatch):
    input_file = tmp_path / "dalles.txt"
    input_file.write_text(
        "https://exemple.invalid/dalle_0500_6500.laz\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        dl,
        "parse_ign_input_file",
        lambda *a, **k: [
            ("dalle_0500_6500.laz", "https://exemple.invalid/dalle_0500_6500.laz")
        ],
    )
    monkeypatch.setattr(dl, "_get_proxy_config", lambda **k: {})
    # Tous les téléchargements échouent (échec définitif, pas d'annulation).
    monkeypatch.setattr(
        dl,
        "_download_task_worker",
        lambda task, *a, **k: dl._DownloadResult(
            index=task.index,
            filename=task.filename,
            success=False,
            skipped=False,
            error="proxy injoignable",
        ),
    )

    with pytest.raises(RuntimeError, match=r"(?i)aucune dalle"):
        dl.download_ign_dalles(
            input_file=input_file,
            output_dir=tmp_path / "out",
            cancel=lambda: False,
        )


def test_message_zero_dalle_mentionne_reseau_et_proxy(tmp_path, monkeypatch):
    input_file = tmp_path / "dalles.txt"
    input_file.write_text("https://exemple.invalid/a.laz\n", encoding="utf-8")

    monkeypatch.setattr(
        dl,
        "parse_ign_input_file",
        lambda *a, **k: [("a.laz", "https://exemple.invalid/a.laz")],
    )
    monkeypatch.setattr(dl, "_get_proxy_config", lambda **k: {})
    monkeypatch.setattr(
        dl,
        "_download_task_worker",
        lambda task, *a, **k: dl._DownloadResult(
            index=task.index, filename=task.filename,
            success=False, skipped=False, error="x",
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        dl.download_ign_dalles(
            input_file=input_file,
            output_dir=tmp_path / "out",
            cancel=lambda: False,
        )
    msg = str(exc.value).lower()
    assert "réseau" in msg
    assert "proxy" in msg
