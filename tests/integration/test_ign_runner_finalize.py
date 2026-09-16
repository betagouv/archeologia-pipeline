"""ROB-12 / TEST-11 (audit v2) : la phase fusion de IgnOrLocalRunner doit être
DANS le try/finally — un échec total de fusion doit quand même finaliser
(VRT + chargement de ce qui existe), et les échecs partiels doivent être
visibles côté UI (reporter.error).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("shapely")

from app.cancel_token import CancelToken
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context
from app.runners.ign_local_runner import IgnOrLocalRunner


class RecordingReporter(NullProgressReporter):
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)


class _StubStrategy:
    def __init__(self, tmp_path: Path):
        self._tmp = tmp_path

    def acquire(self, **kwargs):
        return SimpleNamespace(
            sorted_list_file=self._tmp / "fichier_tri.txt",
            dalles_dir=self._tmp / "dalles",
        )

    def merge_progress_start(self):
        return 5

    def merge_progress_end(self):
        return None

    def products_progress_start(self):
        return 10

    def products_progress_for_tile(self, i, n):
        return 20


def _ctx(config_with_output_dir: dict, tmp_path):
    cfg = config_with_output_dir
    cfg["app"]["files"]["data_mode"] = "ign_laz"
    cfg["app"]["files"]["input_file"] = str(tmp_path / "dalles.txt")
    # Pas de produit actif ni de CV : on teste le câblage fusion→finalisation.
    cfg["processing"]["products"] = {k: False for k in cfg["processing"]["products"]}
    cfg["computer_vision"] = {"enabled": False}
    return build_run_context(cfg)


@pytest.fixture
def finalized(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "app.runners.ign_local_runner.finalize_pipeline",
        lambda **kw: calls.append(kw),
    )
    return calls


@pytest.fixture
def stub_strategy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.runners.ign_local_runner.select_input_strategy",
        lambda mode, plan: _StubStrategy(tmp_path),
    )


class TestIgnRunnerFinalize:
    def test_echec_total_de_fusion_finalise_quand_meme(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized, stub_strategy
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)

        def boom(**kwargs):
            raise RuntimeError("Fusion impossible : aucune dalle fusionnée")

        monkeypatch.setattr("pipeline.ign.preprocess.prepare_merged_tiles", boom)

        with pytest.raises(RuntimeError):
            IgnOrLocalRunner().run(
                ctx=ctx,
                reporter=RecordingReporter(),
                cancel=CancelToken(threading.Event()),
            )
        # La finalisation a TOUJOURS lieu (produits déjà sur disque indexés)…
        assert len(finalized) == 1
        # …et elle sait que le run a ÉCHOUÉ (ROB-14 : pas de faux « ✅ »).
        assert finalized[0]["outcome"] == "failed"

    def test_echecs_partiels_de_fusion_visibles_en_ui(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized, stub_strategy
    ):
        ctx = _ctx(config_with_output_dir, tmp_path)
        merged = tmp_path / "LHD_FXX_0500_6500_merged.laz"
        merged.write_bytes(b"LAZ")

        def partial(**kwargs):
            return SimpleNamespace(
                merged_dir=tmp_path,
                temp_dir=tmp_path,
                merged_files=[merged],
                failed=["LHD_FXX_0501_6500: PDAL merge failed"],
            )

        monkeypatch.setattr("pipeline.ign.preprocess.prepare_merged_tiles", partial)

        reporter = RecordingReporter()
        IgnOrLocalRunner().run(
            ctx=ctx, reporter=reporter, cancel=CancelToken(threading.Event())
        )

        assert any("non fusionnée" in e for e in reporter.errors)
        assert len(finalized) == 1
        # Échec PARTIEL absorbé → l'issue globale reste un succès.
        assert finalized[0]["outcome"] == "success"


class TestIgnRunnerVerditHonnete:
    """Un run qui n'a rien produit ne doit jamais s'annoncer réussi.

    L'interface lit le retour du contrôleur comme « ok sauf si False »
    (ui/run_view.py) : renvoyer ``None`` sur un échec affiche donc
    « ✓ Pipeline terminé » sur un dossier vide. Ces deux tests verrouillent les
    deux portes par lesquelles ce mensonge passait (audit 2026-09-16).
    """

    def test_acquisition_sans_dalle_renvoie_un_echec(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized
    ):
        """Polygone n'intersectant aucune dalle, téléchargements tous en échec,
        dossier LAZ vide : la stratégie rend ``None``."""

        class _StrategieVide(_StubStrategy):
            def acquire(self, **kwargs):
                return None

        monkeypatch.setattr(
            "app.runners.ign_local_runner.select_input_strategy",
            lambda mode, plan: _StrategieVide(tmp_path),
        )
        ctx = _ctx(config_with_output_dir, tmp_path)

        verdict = IgnOrLocalRunner().run(
            ctx=ctx,
            reporter=RecordingReporter(),
            cancel=CancelToken(threading.Event()),
        )
        assert verdict is False, (
            "acquisition vide rendue comme un succès — l'interface afficherait "
            "« ✓ Pipeline terminé » sur un dossier sans aucune donnée"
        )

    def test_aucune_dalle_fusionnee_compte_les_echecs_dans_le_total(
        self, config_with_output_dir, tmp_path, monkeypatch, finalized, stub_strategy
    ):
        """0 fusionnée sur 3 tentées doit arriver à la finalisation comme 0/3.

        Transmettre 0/0 désarmerait la garde « 0 sur N » de finalize_service,
        qui teste ``bool(total)`` : le run repartirait en succès.
        """
        ctx = _ctx(config_with_output_dir, tmp_path)

        def tout_echoue(**kwargs):
            return SimpleNamespace(
                merged_dir=tmp_path,
                temp_dir=tmp_path,
                merged_files=[],
                failed=[
                    "LHD_FXX_0500_6500: PDAL merge failed",
                    "LHD_FXX_0501_6500: PDAL merge failed",
                    "LHD_FXX_0502_6500: PDAL merge failed",
                ],
            )

        monkeypatch.setattr("pipeline.ign.preprocess.prepare_merged_tiles", tout_echoue)

        IgnOrLocalRunner().run(
            ctx=ctx,
            reporter=RecordingReporter(),
            cancel=CancelToken(threading.Event()),
        )
        assert len(finalized) == 1
        assert finalized[0]["tiles_processed"] == 0
        assert finalized[0]["tiles_total"] == 3, (
            "le total ignore les dalles en échec : 0/0 désarme la garde « 0 sur N »"
        )

