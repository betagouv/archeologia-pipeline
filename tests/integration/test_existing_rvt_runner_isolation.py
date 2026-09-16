"""ROB-13 (audit v2) : ExistingRvtRunner doit isoler les erreurs par run CV
et TOUJOURS appeler finalize_pipeline (finally), comme les autres runners
après le correctif AUDIT ROB-02/03/04 (commit bc6eb33) — il avait été oublié.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

# resolve_cv_runs (importé par le runner) vit sous pipeline.cv → shapely requis.
pytest.importorskip("shapely")

from app.cancel_token import CancelToken
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context
from app.runners.existing_rvt_runner import ExistingRvtRunner
from pipeline.cancellation import PipelineCancelled


class RecordingReporter(NullProgressReporter):
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _ctx(config_with_output_dir: dict, tmp_path):
    rvt_dir = tmp_path / "rvt_in"
    rvt_dir.mkdir()
    cfg = config_with_output_dir
    cfg["app"]["files"]["data_mode"] = "existing_rvt"
    cfg["app"]["files"]["existing_rvt_dir"] = str(rvt_dir)
    cfg["computer_vision"] = {
        "enabled": True,
        "target_rvt": "LD",
        "runs": [
            {"model": "modele_a", "target_rvt": "LD"},
            {"model": "modele_b", "target_rvt": "SVF"},
        ],
    }
    return build_run_context(cfg)


@pytest.fixture
def finalized(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "app.runners.existing_rvt_runner.finalize_pipeline",
        lambda **kw: calls.append(kw),
    )
    return calls


class TestExistingRvtRunnerIsolation:
    def test_un_run_en_echec_n_avorte_ni_les_suivants_ni_la_finalisation(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)
        executed: list[str] = []

        def fake_run_existing_rvt(**kwargs):
            model = kwargs["cv_config"].get("selected_model")
            executed.append(model)
            if model == "modele_a":
                raise RuntimeError("TIF illisible")
            return SimpleNamespace(total_images=7)

        monkeypatch.setattr(
            "pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt
        )

        reporter = RecordingReporter()
        ExistingRvtRunner().run(
            ctx=ctx, reporter=reporter, cancel=CancelToken(threading.Event())
        )

        # Le run 2 a tourné malgré l'échec du run 1.
        assert executed == ["modele_a", "modele_b"]
        # La finalisation a toujours lieu — échec partiel absorbé = succès global.
        assert len(finalized) == 1
        assert finalized[0]["outcome"] == "success"
        # L'échec est visible côté UI (canal error, pas info).
        assert any("échec" in e for e in reporter.errors)

    def test_exception_fatale_finalise_quand_meme(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)

        def fake_run_existing_rvt(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt
        )

        ExistingRvtRunner().run(
            ctx=ctx, reporter=RecordingReporter(), cancel=CancelToken(threading.Event())
        )
        assert len(finalized) == 1

    def test_annulation_arrete_les_runs_mais_finalise(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)
        executed: list[str] = []

        def fake_run_existing_rvt(**kwargs):
            executed.append(kwargs["cv_config"].get("selected_model"))
            raise PipelineCancelled()

        monkeypatch.setattr(
            "pipeline.modes.existing_rvt.run_existing_rvt", fake_run_existing_rvt
        )

        # L'annulation ne doit PAS être isolée : arrêt immédiat + finalisation.
        ExistingRvtRunner().run(
            ctx=ctx, reporter=RecordingReporter(), cancel=CancelToken(threading.Event())
        )
        assert executed == ["modele_a"]
        assert len(finalized) == 1
        # ROB-14 : la finalisation sait que le run a été annulé (pas de « ✅ »).
        assert finalized[0]["outcome"] == "cancelled"


class TestExistingRvtVerdictHonnete:
    """Tous les runs CV en échec ne doit pas donner un succès.

    ``finalize_pipeline`` retombe sur ``tiles_processed`` quand aucun
    ``tiles_total`` ne lui est transmis. Or ce runner comptait des IMAGES : zéro
    image traitée donnait 0/0, ``bool(total)`` était faux, et la garde « 0 sur N »
    se désarmait (audit 2026-09-16 — même famille que le correctif ROB-02/03/04,
    qui avait déjà oublié ce runner une fois).
    """

    def test_tous_les_runs_en_echec_transmettent_un_total_honnete(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)

        def tout_echoue(**kwargs):
            raise RuntimeError("TIF illisible")

        monkeypatch.setattr(
            "pipeline.modes.existing_rvt.run_existing_rvt", tout_echoue
        )

        ExistingRvtRunner().run(
            ctx=ctx, reporter=RecordingReporter(), cancel=CancelToken(threading.Event())
        )

        assert len(finalized) == 1
        assert finalized[0]["tiles_processed"] == 0
        assert finalized[0].get("tiles_total") == 2, (
            "aucun total transmis : 0/0 désarme la garde « 0 sur N » et le run "
            "repart en succès alors qu'aucune image n'a été traitée"
        )

    def test_le_total_reste_dans_l_unite_des_images(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        """Traité et total comptent la MÊME chose : des images.

        ``structured_logger.end_pipeline`` imprime « Dalles traitées :
        traité/total ». Un total exprimé en runs CV donnait « 4/2 » dans le
        journal technique — deux unités dans une même fraction (relecture des
        commits, 2026-09-16). Le repli sur le nombre de runs ne concerne que le
        cas où AUCUN run n'aboutit, où le numérateur vaut 0.
        """
        ctx = _ctx(config_with_output_dir, tmp_path)

        def un_seul_marche(**kwargs):
            if kwargs["cv_config"].get("selected_model") == "modele_a":
                raise RuntimeError("TIF illisible")
            return SimpleNamespace(total_images=4, total_detections=None)

        monkeypatch.setattr(
            "pipeline.modes.existing_rvt.run_existing_rvt", un_seul_marche
        )

        ExistingRvtRunner().run(
            ctx=ctx, reporter=RecordingReporter(), cancel=CancelToken(threading.Event())
        )

        assert finalized[0]["tiles_processed"] == 4
        assert finalized[0].get("tiles_total") == 4, (
            "total en runs CV : le journal technique afficherait « 4/2 »"
        )

