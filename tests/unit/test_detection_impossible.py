"""Lot 0d (accélération 2026-10) : un échec de détection se voit.

Le « repli » dans le processus de QGIS exige onnxruntime, absent des Pythons de
QGIS 3.44 (OSGeo4W) et 4.0 : quand le binaire s'arrête, il n'y a aucune
détection. Avant : fin « ✓ Terminé », erreur noyée dans le journal. Après : fin
non ✓, « Détection impossible » et son motif dans la vue, traitement.json
(cv_engine = "none", cv_motif) et rapport.html.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__, resolve_cv_runs

import pipeline.cv.runner as runner_mod  # noqa: E402
from app.progress_reporter import NullProgressReporter  # noqa: E402
from app.services.aide import rubrique_depannage  # noqa: E402
from app.services.cv_post_service import noter_echec_detection  # noqa: E402
from app.services.finalize_service import echec_detection, finalize_pipeline, moteur_detection  # noqa: E402
from pipeline.cancellation import DetectionImpossible  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
MOTIF = "moteur de détection absent de l'extension (cv_runner_onnx) — réinstallez l'extension"
ECHEC = {"modele": "Cratères d'obus (LD)", "model": "crateres_seg_ld_v1", "target_rvt": "LD",
         "entites": ["Cratère"], "images": 0, "secondes": 0.0, "moteur": "none", "motif": MOTIF}


def _png_et_modele(tmp_path):
    models = tmp_path / "models"
    (models / "m" / "weights").mkdir(parents=True)
    (models / "m" / "weights" / "best.onnx").write_bytes(b"onnx")
    png = tmp_path / "png"
    png.mkdir()
    (png / "LHD_FXX_0500_6500_LD.png").write_bytes(b"png")
    cfg = {"enabled": True, "selected_model": "m", "models_dir": str(models), "scan_all": True,
           "confidence_threshold": 0.4}
    return png, cfg


class TestRunner:
    def test_sans_binaire_ni_onnxruntime(self, tmp_path, monkeypatch):
        png, cfg = _png_et_modele(tmp_path)
        monkeypatch.setattr(runner_mod, "find_external_cv_runner", lambda log=None: None)
        monkeypatch.setattr(runner_mod, "_onnxruntime_disponible", lambda: False)
        with pytest.raises(DetectionImpossible, match="absent"):
            runner_mod.run_cv_on_folder(jpg_dir=png, cv_config=cfg, target_rvt="LD", output_dir=tmp_path / "out")

    def test_binaire_en_echec_sans_repli(self, tmp_path, monkeypatch):
        png, cfg = _png_et_modele(tmp_path)
        exe = tmp_path / "cv_runner_onnx.exe"
        exe.write_bytes(b"exe")

        def plante(**_kw):
            raise RuntimeError("cv_runner failed (code=2)")

        monkeypatch.setattr(runner_mod, "find_external_cv_runner", lambda log=None: exe)
        monkeypatch.setattr(runner_mod, "run_external_cv_runner", plante)
        monkeypatch.setattr(runner_mod, "_onnxruntime_disponible", lambda: False)
        with pytest.raises(DetectionImpossible, match="en échec"):
            runner_mod.run_cv_on_folder(jpg_dir=png, cv_config=cfg, target_rvt="LD", output_dir=tmp_path / "out")

    def test_echec_d_ecriture_apres_le_moteur_n_est_pas_une_detection_impossible(self, tmp_path, monkeypatch):
        png, cfg = _png_et_modele(tmp_path)
        exe = tmp_path / "cv_runner_onnx.exe"
        exe.write_bytes(b"exe")

        def verrouille(**_kw):
            raise RuntimeError("GeoPackage verrouillé")

        monkeypatch.setattr(runner_mod, "find_external_cv_runner", lambda log=None: exe)
        monkeypatch.setattr(runner_mod, "run_external_cv_runner", lambda **kw: 0)
        monkeypatch.setattr(runner_mod, "deduplicate_cv_shapefiles_final", verrouille)
        monkeypatch.setattr(runner_mod, "_onnxruntime_disponible", lambda: False)
        with pytest.raises(RuntimeError, match="GeoPackage verrouillé") as err:
            runner_mod.run_cv_on_folder(jpg_dir=png, cv_config=cfg, target_rvt="LD", output_dir=tmp_path / "out")
        assert not isinstance(err.value, DetectionImpossible)

    def test_le_moteur_employe_est_note(self, tmp_path, monkeypatch):
        png, cfg = _png_et_modele(tmp_path)
        exe = tmp_path / "cv_runner_onnx.exe"
        exe.write_bytes(b"exe")
        monkeypatch.setattr(runner_mod, "find_external_cv_runner", lambda log=None: exe)
        monkeypatch.setattr(runner_mod, "run_external_cv_runner", lambda **kw: 0)
        monkeypatch.setattr(runner_mod, "deduplicate_cv_shapefiles_final", lambda **kw: None)
        stats: dict = {}
        runner_mod.run_cv_on_folder(jpg_dir=png, cv_config=cfg, target_rvt="LD",
                                    output_dir=tmp_path / "out", stats=stats)
        assert stats["moteur"] == "binaire"


class TestStats:
    def test_un_echec_de_detection_est_trace(self):
        stats: list = []
        noter_echec_detection(
            stats, {"selected_model": "crateres_seg_ld_v1", "target_rvt": "LD", "entities": [{"label": "Cratère"}]},
            "Cratères d'obus (LD)", DetectionImpossible(MOTIF))
        assert stats == [ECHEC]

    def test_une_autre_erreur_ne_l_est_pas(self):
        stats: list = []
        noter_echec_detection(stats, {}, "M", RuntimeError("GeoPackage verrouillé"))
        assert stats == []

    def test_moteur_detection(self):
        assert moteur_detection([]) == (None, "")
        assert moteur_detection([{"moteur": "binaire"}, {"moteur": "cache"}]) == ("binaire+cache", "")
        assert moteur_detection([{"moteur": "binaire"}, ECHEC]) == ("none", MOTIF)


class _Rec(NullProgressReporter):
    def __init__(self):
        self.errors: list = []
        self.successes: list = []
        self.stages: list = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def user_success(self, msg: str) -> None:
        self.successes.append(msg)

    def stage(self, msg: str) -> None:
        self.stages.append(msg)


class TestFinalisation:
    def _fin(self, tmp_path, stats, outcome="success"):
        rec = _Rec()
        ok = finalize_pipeline(output_dir=tmp_path, cv_cfg={}, rvt_params={}, reporter=rec, slog=None,
                               start_time=time.time(), cv_stats=stats, outcome=outcome)
        return ok, rec

    def test_detection_impossible_n_est_pas_un_succes(self, tmp_path):
        debut = time.time()
        ok, rec = self._fin(tmp_path, [ECHEC])
        assert ok is False
        assert any(m.startswith("✗ Détection impossible : moteur de détection absent") for m in rec.errors)
        assert rec.successes == [] and "Terminé sans détection" in rec.stages
        trace = json.loads((tmp_path / "livrable" / "traitement.json").read_text(encoding="utf-8"))
        assert trace["cv_engine"] == "none" and trace["cv_motif"] == MOTIF
        rapport = (tmp_path / "livrable" / "rapport.html").read_text(encoding="utf-8")
        assert "terminé sans détection" in rapport and "Détection impossible" in rapport
        assert "<td>détection impossible</td>" in rapport
        assert echec_detection(tmp_path, debut) == MOTIF

    def test_erreur_fatale_reste_une_erreur(self, tmp_path):
        ok, rec = self._fin(tmp_path, [ECHEC], outcome="failed")
        assert ok is False
        assert "Interrompu par une erreur" in rec.stages and "Terminé sans détection" not in rec.stages

    def test_trace_d_un_lancement_precedent_ignoree(self, tmp_path):
        # Trace « détection impossible » laissée par un lancement antérieur : un
        # lancement plus récent qui échoue avant sa finalisation ne la reprend pas.
        self._fin(tmp_path, [ECHEC])
        assert echec_detection(tmp_path, time.time() + 5) == ""

    def test_binaire_employe_trace_sans_motif(self, tmp_path):
        debut = time.time()
        ok, _rec = self._fin(tmp_path, [dict(ECHEC, moteur="binaire", motif="", images=3, secondes=9.0)])
        assert ok is True
        trace = json.loads((tmp_path / "livrable" / "traitement.json").read_text(encoding="utf-8"))
        assert trace["cv_engine"] == "binaire" and "cv_motif" not in trace
        assert echec_detection(tmp_path, debut) == ""

    def test_sans_traitement_json(self, tmp_path):
        assert echec_detection(tmp_path, time.time()) == ""


class TestVueEtManuel:
    def test_la_ligne_de_fin_dit_detection_impossible(self):
        vue = (RACINE / "src/ui/run_view.py").read_text(encoding="utf-8")
        fin = vue[vue.index("def _show_end_banner("):vue.index("def _afficher_bilan(")]
        assert "echec_detection(" in fin and "✗ Détection impossible après" in fin
        # La trace n'est reprise que si elle date de ce lancement-ci.
        assert "self._run_lance_le" in fin and "self._run_lance_le = time.time()" in vue

    def test_renvoi_au_depannage(self):
        assert rubrique_depannage("✗ Détection impossible : moteur de détection absent") == \
            "La détection est impossible"
        assert rubrique_depannage(
            "Computer Vision: run 1 (modèle=X) en échec, ignoré : moteur de détection en échec (code=2)"
        ) == "La détection est impossible"

    def test_manuel_et_recette(self):
        assert "Détection impossible" in (RACINE / "aide/07-etape-5-traitement.md").read_text(encoding="utf-8")
        depannage = (RACINE / "aide/09-depannage.md").read_text(encoding="utf-8")
        assert "## La détection est impossible" in depannage and "moteur de secours" not in depannage
        recette = (RACINE / "tests/TESTS_MANUELS_QGIS.md").read_text(encoding="utf-8")
        assert "## 48." in recette and "Détection impossible" in recette
