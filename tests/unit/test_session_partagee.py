"""Lot 2 (accélération 2026-10) : le binaire crée la session ONNX une fois par
modèle — run_onnx_inference la recréait à chaque image. Session impossible →
une session par image, comme avant (détections identiques au banc IA)."""
from __future__ import annotations

from pathlib import Path

import pipeline.cv.computer_vision_onnx as cvo

RACINE = Path(__file__).resolve().parents[2]


def test_session_creee_une_fois(monkeypatch):
    sentinelle = ("session", "images", [1, 3, 648, 648], {"model_type": "yolo"})
    appels: list = []

    def charger(p):
        appels.append(p)
        return sentinelle

    monkeypatch.setattr(cvo, "_load_onnx_model", charger)
    assert cvo.charger_session_partagee("m.onnx", lambda _m: None) is sentinelle
    assert appels == ["m.onnx"]


def test_echec_retombe_sur_une_session_par_image(monkeypatch):
    def boom(_p):
        raise RuntimeError("pilote cassé")

    monkeypatch.setattr(cvo, "_load_onnx_model", boom)
    logs: list = []
    assert cvo.charger_session_partagee("m.onnx", logs.append) is None
    assert any("une session par image" in m and "pilote cassé" in m for m in logs)


def test_le_cli_partage_la_session():
    cli = (RACINE / "dev/runner_onnx/cv_runner_onnx_cli.py").read_text(encoding="utf-8")
    boucle = cli[cli.index("for idx, jpg_file in enumerate(jpg_files, 1):"):cli.index("summary: success=")]
    assert cli.count("charger_session_partagee(") == 1
    assert "if not session_tentee:" in boucle and "onnx_session=onnx_session," in boucle
    # Créée à la première image À INFÉRER, pas avant le saut des images au cache.
    assert boucle.index("image_deja_traitee(") < boucle.index("charger_session_partagee(")
