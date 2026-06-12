"""ROB-17 (audit v2) : annuler pendant le téléchargement IGN doit se terminer
par l'exception canonique PipelineCancelled (→ « ⏹ Traitement annulé » en UI),
pas par un ValueError « Impossible de déterminer les coordonnées des dalles »
(le tri post-download court-circuité par cancel renvoie une liste vide).
"""
from __future__ import annotations

import pytest

from pipeline.cancellation import PipelineCancelled
import pipeline.ign.downloader as dl


def test_annulation_pendant_download_leve_pipeline_cancelled(tmp_path, monkeypatch):
    input_file = tmp_path / "dalles.txt"
    input_file.write_text(
        "https://exemple.invalid/dalle_0500_6500.laz\n", encoding="utf-8"
    )

    # Pas de réseau ni de QGIS dans ce test : parsing, proxy et worker stubés.
    monkeypatch.setattr(
        dl,
        "parse_ign_input_file",
        lambda *a, **k: [
            ("dalle_0500_6500.laz", "https://exemple.invalid/dalle_0500_6500.laz")
        ],
    )
    monkeypatch.setattr(dl, "_get_proxy_config", lambda **k: {})
    monkeypatch.setattr(
        dl,
        "_download_task_worker",
        lambda task, *a, **k: dl._DownloadResult(
            index=task.index,
            filename=task.filename,
            success=False,
            skipped=False,
            error="annulé",
        ),
    )

    with pytest.raises(PipelineCancelled):
        dl.download_ign_dalles(
            input_file=input_file,
            output_dir=tmp_path / "out",
            cancel=lambda: True,
        )
