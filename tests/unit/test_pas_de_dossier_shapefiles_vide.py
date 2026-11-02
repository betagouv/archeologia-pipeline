"""Plus de dossier ``technique/detection/<modèle>/shapefiles/`` vide (relance du crash test de_sh, 2026-11-02) :
les sorties sont des GeoPackage d'entité ; le binaire créait le dossier d'office dès ``generate_shapefiles``,
alors que le plugin lui passe ``run_shapefile_dedup=False`` (l'écriture est faite côté plugin)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]


def test_binaire_ne_cree_pas_shapefiles_quand_le_plugin_ecrit(tmp_path):
    pytest.importorskip("onnxruntime")
    det = tmp_path / "detection" / "modele"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "jpg_dir": str(tmp_path / "png"), "detection_dir": str(det), "run_shapefile_dedup": False,
        "cv_config": {"enabled": True, "generate_shapefiles": True, "model_path": str(tmp_path / "absent.onnx")},
    }), encoding="utf-8")
    # Le CLI (source du binaire) s'arrête sur le modèle absent, APRÈS la préparation des dossiers.
    r = subprocess.run([sys.executable, str(RACINE / "dev/runner_onnx/cv_runner_onnx_cli.py"), "--config", str(cfg)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert not (det / "shapefiles").exists()


def test_plus_de_chemin_shapefiles_dans_output_paths():
    from pipeline import output_paths
    assert not hasattr(output_paths, "detection_shapefiles_dir")
