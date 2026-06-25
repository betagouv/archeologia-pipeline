"""ROB-14 (audit v2) : finalize_pipeline annonçait INCONDITIONNELLEMENT le
succès (« ✅ PIPELINE TERMINÉ AVEC SUCCÈS », narrateur, barre à 100 %) — y
compris depuis les `finally` des runners alors qu'une exception fatale était
en vol ou que le run venait d'être annulé. Le statut doit suivre l'issue
réelle (success / cancelled / failed).
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("shapely")  # resolve_cv_runs importé par finalize_pipeline

from app.progress_reporter import NullProgressReporter
from app.services.finalize_service import finalize_pipeline


class RecordingReporter(NullProgressReporter):
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.progresses: list[int] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def user_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def user_success(self, msg: str) -> None:
        self.successes.append(msg)

    def progress(self, pct: int) -> None:
        self.progresses.append(pct)


def _finalize(tmp_path, reporter, **kw):
    finalize_pipeline(
        output_dir=tmp_path,
        cv_cfg={},
        rvt_params={},
        reporter=reporter,
        slog=None,
        start_time=time.time(),
        **kw,
    )


def test_outcome_par_defaut_reste_le_succes(tmp_path):
    rec = RecordingReporter()
    _finalize(tmp_path, rec)
    assert any("AVEC SUCCÈS" in m for m in rec.infos)
    assert any("Traitement terminé" in m for m in rec.successes)
    assert 100 in rec.progresses


def test_outcome_failed_n_annonce_pas_le_succes(tmp_path):
    rec = RecordingReporter()
    _finalize(tmp_path, rec, outcome="failed")
    # Bannière d'erreur, pas de succès.
    assert any("AVEC ERREURS" in m for m in rec.infos)
    assert not any("AVEC SUCCÈS" in m for m in rec.infos)
    # Message narratif d'échec visible (user_warning), pas de ✅.
    assert any("interrompu" in w for w in rec.warnings)
    assert not any("Traitement terminé" in m for m in rec.successes)
    # La barre ne saute pas à 100 % sur un échec.
    assert 100 not in rec.progresses


def test_outcome_cancelled_n_annonce_ni_succes_ni_echec(tmp_path):
    rec = RecordingReporter()
    _finalize(tmp_path, rec, outcome="cancelled")
    assert not any("AVEC SUCCÈS" in m for m in rec.infos)
    assert not any("Traitement terminé" in m for m in rec.successes)
    # Le « ⏹ Traitement annulé » est émis par le runner, pas par finalize.
    assert rec.warnings == []
