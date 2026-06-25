"""CVPROC-01 (audit v2) : le runner CV externe ne doit JAMAIS se bloquer parce
que le tube stderr se remplit. Avant le correctif : stderr=PIPE n'était drainé
qu'au communicate() final, APRÈS l'EOF de stdout → dès ~64 Ko de tracebacks
côté binaire (ex. modèle incompatible → traceback PAR image), le binaire
bloquait sur write(stderr), stdout se taisait, et le parent restait bloqué
dans readline — gel définitif de QGIS, Annuler inopérant.

Le test lance un vrai sous-processus qui émet ~600 Ko sur stderr AVANT toute
sortie stdout, avec un garde-fou de 30 s : sans correctif il échoue en
« DEADLOCK », avec correctif il passe en ~1 s.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv.external_runner import run_external_cv_runner

FAKE_RUNNER = """
import sys
chunk = "traceback-ligne-stderr-" + "x" * 2000
for _ in range(300):
    print(chunk, file=sys.stderr)
print("summary: success=1 total_detections=0")
"""


def _make_fake_runner(tmp_path: Path) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER, encoding="utf-8")
    if os.name == "nt":
        ext = tmp_path / "fake_runner.bat"
        ext.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="ascii"
        )
    else:
        ext = tmp_path / "fake_runner.sh"
        ext.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        ext.chmod(0o755)
    return ext


def test_stderr_volumineux_ne_bloque_pas_le_runner(tmp_path):
    ext = _make_fake_runner(tmp_path)
    jpg_dir = tmp_path / "jpg"
    jpg_dir.mkdir()
    logs: list[str] = []
    done = threading.Event()
    errors: list[BaseException] = []

    def _call():
        try:
            run_external_cv_runner(
                ext=ext,
                jpg_dir=jpg_dir,
                target_rvt="LD",
                rvt_base_dir=None,
                cv_config={},
                single_jpg=None,
                run_shapefile_dedup=False,
                tif_transform_data=None,
                log=logs.append,
            )
        except BaseException as e:  # noqa: BLE001 — relayé pour assertion
            errors.append(e)
        finally:
            done.set()

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    assert done.wait(timeout=30), (
        "DEADLOCK : run_external_cv_runner bloqué — le stderr du binaire "
        "n'est pas drainé pendant l'exécution (CVPROC-01)"
    )
    assert not errors, f"runner en échec inattendu : {errors}"
    # Les lignes stderr doivent être relayées EN DIRECT via le parseur stdout.
    assert any("traceback-ligne-stderr-" in line for line in logs)
