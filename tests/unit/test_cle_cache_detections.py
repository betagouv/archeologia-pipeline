"""Clé du cache des détections (lot 0b, accélération 2026-10).

``raw_detections/`` était indexé par le seul nom d'image : relancer dans le même
dossier avec un autre seuil, un autre modèle ou un autre moteur resservait les
détections du lancement précédent (le binaire saute toute image au cache, sans
lire ``force_reprocess``). La clé ``cle_cache.json`` dit avec quoi le cache a
été calculé ; différente ou absente → purge et ré-inférence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

import pipeline.cv.runner as runner_mod  # noqa: E402
from pipeline.cv.runner_cache import (  # noqa: E402
    FICHIER_CLE_CACHE,
    aligner_cache_detections,
    cle_cache_detections,
    image_deja_traitee,
    version_moteur,
)
from pipeline.output_paths import detection_technique_raw_dir  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
STEM = "LHD_FXX_0500_6500_LD"


def _modele(tmp_path: Path, poids: bytes = b"onnx-v1") -> Path:
    models = tmp_path / "models"
    (models / "m" / "weights").mkdir(parents=True, exist_ok=True)
    (models / "m" / "weights" / "best.onnx").write_bytes(poids)
    (models / "m" / "args.yaml").write_text(
        "sahi:\n  slice_height: 648\n  slice_width: 648\n  overlap_ratio: 0.2\n", encoding="utf-8"
    )
    return models


def _cfg(models: Path, **surcharges) -> dict:
    return {"enabled": True, "selected_model": "m", "models_dir": str(models), "scan_all": True,
            "confidence_threshold": 0.4, "iou_threshold": 0.5, **surcharges}


def _cache(raw: Path, *stems: str) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (raw / f"{s}.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")
        (raw / f"{s}.json").write_text("{}", encoding="utf-8")


class TestCle:
    def test_meme_reglage_meme_cle(self, tmp_path):
        models = _modele(tmp_path)
        poids = models / "m" / "weights" / "best.onnx"
        assert cle_cache_detections(_cfg(models), poids, "binaire:a") == cle_cache_detections(
            _cfg(models), poids, "binaire:a")

    @pytest.mark.parametrize("surcharge", [
        {"confidence_threshold": 0.3},
        {"confidence_per_class": {"cratere": 0.25}},
        {"iou_threshold": 0.6},
        {"sahi": {"slice_height": 672, "slice_width": 672, "overlap_ratio": 0.2}},
    ])
    def test_un_reglage_d_analyse_change_la_cle(self, tmp_path, surcharge):
        models = _modele(tmp_path)
        poids = models / "m" / "weights" / "best.onnx"
        assert cle_cache_detections(_cfg(models, **surcharge), poids, "b") != cle_cache_detections(
            _cfg(models), poids, "b")

    def test_un_autre_onnx_change_la_cle(self, tmp_path):
        models = _modele(tmp_path)
        poids = models / "m" / "weights" / "best.onnx"
        avant = cle_cache_detections(_cfg(models), poids, "b")
        poids.write_bytes(b"onnx-v2")
        assert cle_cache_detections(_cfg(models), poids, "b") != avant

    def test_un_autre_args_yaml_change_la_cle(self, tmp_path):
        models = _modele(tmp_path)
        poids = models / "m" / "weights" / "best.onnx"
        avant = cle_cache_detections(_cfg(models), poids, "b")
        (models / "m" / "args.yaml").write_text("imgsz: 672\n", encoding="utf-8")
        assert cle_cache_detections(_cfg(models), poids, "b") != avant

    def test_un_autre_moteur_change_la_cle(self, tmp_path):
        models = _modele(tmp_path)
        poids = models / "m" / "weights" / "best.onnx"
        assert cle_cache_detections(_cfg(models), poids, "binaire:a") != cle_cache_detections(
            _cfg(models), poids, "binaire:b")

    def test_version_du_binaire_lue_dans_build_info(self, tmp_path):
        exe = tmp_path / "cv_runner_onnx.exe"
        exe.write_bytes(b"exe")
        assert version_moteur(exe) == "binaire:inconnu"
        (tmp_path / "build_info.json").write_text(
            json.dumps({"commit": "ff008e1", "sources_sha256": "abc"}), encoding="utf-8")
        assert version_moteur(exe) == "binaire:abc"
        assert version_moteur(None) == "python"


class TestAligner:
    def test_cache_sans_cle_purge(self, tmp_path):
        # Dossier écrit par la 0.14.1 : détections sans clé → ré-inférence.
        raw = tmp_path / "raw"
        _cache(raw, STEM)
        (raw / "classes.txt").write_text("cratere", encoding="utf-8")
        assert aligner_cache_detections(raw, {"k": 1}) == 1
        assert sorted(p.name for p in raw.iterdir()) == [FICHIER_CLE_CACHE]

    def test_meme_cle_garde_le_cache(self, tmp_path):
        raw = tmp_path / "raw"
        aligner_cache_detections(raw, {"k": 1})
        _cache(raw, STEM)
        assert aligner_cache_detections(raw, {"k": 1}) == 0
        assert (raw / f"{STEM}.json").exists()

    def test_autre_cle_purge_et_s_enregistre(self, tmp_path):
        raw = tmp_path / "raw"
        aligner_cache_detections(raw, {"k": 1})
        _cache(raw, STEM, "autre")
        assert aligner_cache_detections(raw, {"k": 2}) == 2
        assert json.loads((raw / FICHIER_CLE_CACHE).read_text(encoding="utf-8")) == {"k": 2}

    def test_dossier_absent(self, tmp_path):
        raw = tmp_path / "raw"
        assert aligner_cache_detections(raw, {"k": 1}) == 0
        assert (raw / FICHIER_CLE_CACHE).exists()


class TestImageDejaTraitee:
    def test_cache_present(self, tmp_path):
        _cache(tmp_path, STEM)
        assert image_deja_traitee(tmp_path, STEM, force_reprocess=False) is True

    def test_force_reprocess_ignore_le_cache(self, tmp_path):
        _cache(tmp_path, STEM)
        assert image_deja_traitee(tmp_path, STEM, force_reprocess=True) is False

    def test_sans_cache(self, tmp_path):
        assert image_deja_traitee(tmp_path, STEM, force_reprocess=False) is False

    def test_le_binaire_et_le_repli_appliquent_la_meme_regle(self):
        cli = (RACINE / "dev/runner_onnx/cv_runner_onnx_cli.py").read_text(encoding="utf-8")
        assert "image_deja_traitee(raw_dir, image_name, force_reprocess)" in cli
        assert 'force_reprocess = bool(cv_config.get("force_reprocess", False))' in cli
        repli = (RACINE / "src/pipeline/cv/runner_inference.py").read_text(encoding="utf-8")
        assert "image_deja_traitee(_raw_dir, image_name, force_reprocess)" in repli


@pytest.fixture
def moteur(tmp_path, monkeypatch):
    """Binaire factice : chaque appel écrit les détections de chaque PNG."""
    exe = tmp_path / "bin" / "cv_runner_onnx.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"exe")
    (exe.parent / "build_info.json").write_text(json.dumps({"sources_sha256": "s1"}), encoding="utf-8")
    appels: list = []

    def inferer(**kw):
        appels.append(kw)
        for png in sorted(Path(kw["jpg_dir"]).glob("*.png")):
            _cache(Path(kw["raw_dir"]), png.stem)
        return 0

    monkeypatch.setattr(runner_mod, "find_external_cv_runner", lambda log=None: exe)
    monkeypatch.setattr(runner_mod, "run_external_cv_runner", inferer)
    monkeypatch.setattr(runner_mod, "deduplicate_cv_shapefiles_final", lambda **kw: None)
    png = tmp_path / "png"
    png.mkdir()
    (png / f"{STEM}.png").write_bytes(b"png")
    return {"appels": appels, "png": png, "models": _modele(tmp_path), "out": tmp_path / "out", "exe": exe}


def _lancer(m, logs=None, **surcharges):
    runner_mod.run_cv_on_folder(
        jpg_dir=m["png"], cv_config=_cfg(m["models"], **surcharges), target_rvt="LD",
        output_dir=m["out"], log=(logs.append if logs is not None else (lambda _m: None)),
    )


class TestRunCvOnFolder:
    def test_relance_identique_reprend_le_cache(self, moteur):
        _lancer(moteur)
        logs: list = []
        _lancer(moteur, logs)
        assert len(moteur["appels"]) == 1
        assert any("inférence sautée" in m for m in logs)

    def test_seuil_abaisse_reinfere(self, moteur):
        _lancer(moteur)
        logs: list = []
        _lancer(moteur, logs, confidence_threshold=0.3)
        assert len(moteur["appels"]) == 2
        assert any("purgée(s), ré-inférence" in m for m in logs)

    def test_nouveau_binaire_reinfere(self, moteur):
        _lancer(moteur)
        (moteur["exe"].parent / "build_info.json").write_text(
            json.dumps({"sources_sha256": "s2"}), encoding="utf-8")
        _lancer(moteur)
        assert len(moteur["appels"]) == 2

    def test_cache_d_une_version_precedente_reinfere(self, moteur):
        raw = detection_technique_raw_dir(moteur["out"], "m")
        _cache(raw, STEM)   # 0.14.1 : détections sans clé
        _lancer(moteur)
        assert len(moteur["appels"]) == 1
        assert (raw / FICHIER_CLE_CACHE).exists()
