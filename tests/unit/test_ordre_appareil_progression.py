"""Ordre du journal d'un modèle (2026-11-02, run gb_eng_dalles_3x3) : la ligne de l'appareil
(⚡ carte graphique / ⚠ processeur) vient juste après « Modèle N/T », AVANT toute ligne
« Image i/N ». Le moteur ne décide qu'après la sonde de la première image : quand le
plugin a demandé la carte (``cv_config["carte_graphique"]["demande"]``),
``run_external_cv_runner`` diffère la progression jusqu'à la décision (ou jusqu'à la fin
d'une image sans décision, ou la fin du flux). Avant : « Image 1/9 », ⚡, puis une seconde
ligne « Image 9/9 » (la ligne ⚡ cassait la réécriture en place du panneau).
Vrai sous-processus, comme ``test_external_runner_stats``."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv.external_runner import run_external_cv_runner

DECISION = 'appareil={"appareil": "carte_graphique", "message": "détection sur la carte graphique"}'

SONDE = f"""
print("images=2")
print("progress=1/2 image=a.png status=processing")
print("[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 10/16 tuiles traitées")
print("[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 16/16 tuiles traitées")
print({DECISION!r})
print("progress=1/2 image=a.png status=done detections=2 mode=sahi")
print("progress=2/2 image=b.png status=processing")
print("[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 16/16 tuiles traitées")
print("progress=2/2 image=b.png status=done detections=0 mode=sahi")
print("summary: success=1 processed=2 skipped=0 total_detections=2")
"""

SANS_DECISION = """
print("progress=1/2 image=a.png status=skipped")
print("progress=2/2 image=b.png status=processing")
print("[cv_runner_onnx][INFO] RF-DETR Seg SAHI: 16/16 tuiles traitées")
print("progress=2/2 image=b.png status=done detections=0 mode=sahi")
print("summary: success=0 processed=1 skipped=1 total_detections=0")
"""

TOUT_EN_CACHE = """
print("progress=1/2 image=a.png status=skipped")
print("progress=2/2 image=b.png status=skipped")
print("summary: success=0 processed=0 skipped=2 total_detections=0")
"""


def _runner(tmp_path: Path, code: str) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(code, encoding="utf-8")
    if os.name == "nt":
        ext = tmp_path / "fake_runner.bat"
        ext.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="ascii")
    else:
        ext = tmp_path / "fake_runner.sh"
        ext.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        ext.chmod(0o755)
    return ext


def _evenements(tmp_path: Path, code: str, demande: bool):
    ev = []
    (tmp_path / "jpg").mkdir()
    run_external_cv_runner(
        ext=_runner(tmp_path, code), jpg_dir=tmp_path / "jpg", target_rvt="LD", rvt_base_dir=None,
        cv_config={"carte_graphique": {"demande": True}} if demande else {},
        single_jpg=None, run_shapefile_dedup=False, tif_transform_data=None, log=lambda _m: None,
        image_progress=lambda i, t, n: ev.append(("image", i, t)),
        tile_progress=lambda c, t: ev.append(("fenetre", c, t)),
        on_appareil=lambda d: ev.append(("appareil",)),
    )
    return ev


def test_carte_demandee_la_decision_precede_la_progression(tmp_path):
    ev = _evenements(tmp_path, SONDE, demande=True)
    assert ev[0] == ("appareil",)
    # Les fenêtres de l'image 1 analysée pendant la sonde ne sont pas relayées (elle est
    # rejouée sur la carte) ; la progression reprend normalement ensuite.
    assert ev[1:] == [("image", 1, 2), ("image", 2, 2), ("fenetre", 16, 16)]


def test_sans_decision_la_progression_est_liberee_a_la_fin_de_l_image(tmp_path):
    ev = _evenements(tmp_path, SANS_DECISION, demande=True)
    assert ev == [("image", 2, 2)]   # 1/2 (cache) remplacée par 2/2, émise à la fin de l'image


def test_tout_en_cache_la_derniere_progression_sort_a_la_fin_du_flux(tmp_path):
    assert _evenements(tmp_path, TOUT_EN_CACHE, demande=True) == [("image", 2, 2)]


def test_carte_non_demandee_rien_n_est_differe(tmp_path):
    ev = _evenements(tmp_path, SONDE, demande=False)
    assert ev[:3] == [("image", 1, 2), ("fenetre", 10, 16), ("fenetre", 16, 16)]
    assert ev[3] == ("appareil",)
