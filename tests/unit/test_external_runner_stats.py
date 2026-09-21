"""Durée réelle d'un run CV : ``run_external_cv_runner(stats=…)`` dépose le
nombre d'images réellement inférées (status=done, cache exclu) et les secondes
écoulées — la matière de la ligne « ✓ … analysées en … » du journal. Vrai
sous-processus (script Python déguisé en binaire), comme le test CVPROC-01."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv.external_runner import run_external_cv_runner

FAKE_RUNNER = """
print("images=3")
print("progress=1/3 image=a.png status=processing")
print("[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 16/16 tuiles traitées")
print("progress=1/3 image=a.png status=done detections=2 mode=sahi")
print("progress=2/3 image=b.png status=skipped")
print("progress=3/3 image=c.png status=processing")
print("progress=3/3 image=c.png status=done detections=0 mode=sahi")
print("summary: success=2 processed=2 skipped=1 total_detections=2")
"""


def _make_fake_runner(tmp_path: Path) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER, encoding="utf-8")
    if os.name == "nt":
        ext = tmp_path / "fake_runner.bat"
        ext.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="ascii")
    else:
        ext = tmp_path / "fake_runner.sh"
        ext.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        ext.chmod(0o755)
    return ext


def test_stats_compte_les_images_inferees_et_les_secondes(tmp_path):
    jpg_dir = tmp_path / "jpg"
    jpg_dir.mkdir()
    stats: dict = {}
    total = run_external_cv_runner(
        ext=_make_fake_runner(tmp_path),
        jpg_dir=jpg_dir,
        target_rvt="LD",
        rvt_base_dir=None,
        cv_config={},
        single_jpg=None,
        run_shapefile_dedup=False,
        tif_transform_data=None,
        log=lambda _m: None,
        stats=stats,
    )
    assert total == 2
    assert stats["images_inferees"] == 2  # b.png servie par le cache : exclue
    assert stats["secondes"] > 0


def test_sans_stats_le_contrat_ne_change_pas(tmp_path):
    jpg_dir = tmp_path / "jpg"
    jpg_dir.mkdir()
    assert run_external_cv_runner(
        ext=_make_fake_runner(tmp_path),
        jpg_dir=jpg_dir,
        target_rvt="LD",
        rvt_base_dir=None,
        cv_config={},
        single_jpg=None,
        run_shapefile_dedup=False,
        tif_transform_data=None,
        log=lambda _m: None,
    ) == 2
