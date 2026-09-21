"""Chien de garde sur l'étape MNT (incident 2026-09-19).

Un LAZ corrompu fait boucler PDAL sans fin : sur la dalle LHD_FXX_0821_6327
le pipeline est resté 10 h muet dans ``pdal:exportrastertin``, sans erreur ni
progression, jusqu'à une annulation manuelle. L'écriture atomique
(``test_preprocess_ecriture_atomique``) empêche de fabriquer un tel LAZ, mais
pas d'en recevoir un (téléchargement véreux, erreur disque, LAZ fourni en
mode local). Au-delà d'une limite de temps, la dalle est abandonnée et le lot
continue — ``process_items_isolated`` fait le reste.

Deux propriétés tenues ici :
- le dépassement n'est PAS collant (l'annulation Qt l'est : elle tuerait tout
  le reste du run au lieu de la seule dalle en cours) ;
- un dépassement n'enchaîne pas sur le fallback ``pdal:exportraster``, qui
  relirait le même LAZ pour reboucler autant de temps.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from app.cancellable_feedback import create_cancellable_feedback
from pipeline.ign.products import mnt


class _FauxFeedbackQgis:
    """Comportement de QgsProcessingFeedback utile ici : cancel() est collant."""

    def __init__(self) -> None:
        self._canceled = False

    def isCanceled(self) -> bool:
        return self._canceled

    def cancel(self) -> None:
        self._canceled = True

    def setProgress(self, progress: float) -> None:
        pass


@pytest.fixture
def feedback(monkeypatch):
    faux = types.ModuleType("qgis.core")
    faux.QgsProcessingFeedback = _FauxFeedbackQgis
    monkeypatch.setitem(sys.modules, "qgis", types.ModuleType("qgis"))
    monkeypatch.setitem(sys.modules, "qgis.core", faux)
    return create_cancellable_feedback(lambda: False)


def test_sans_chien_de_garde_arme_rien_ne_sannule(feedback):
    assert feedback.isCanceled() is False
    feedback.start_watchdog(0)  # 0 = désarmé
    time.sleep(0.02)
    assert feedback.isCanceled() is False
    assert feedback.stop_watchdog() is False


def test_depassement_annule_lalgorithme_puis_se_rearme(feedback):
    feedback.start_watchdog(0.01)
    assert feedback.isCanceled() is False
    time.sleep(0.02)
    assert feedback.isCanceled() is True, "le dépassement doit rendre la main"

    assert feedback.stop_watchdog() is True, "le dépassement doit être rapporté"
    # Le point critique : la dalle suivante doit repartir sur un feedback sain.
    assert feedback.isCanceled() is False, "le dépassement a tué tout le run"
    assert feedback.stop_watchdog() is False


def test_annulation_utilisateur_reste_collante(monkeypatch):
    faux = types.ModuleType("qgis.core")
    faux.QgsProcessingFeedback = _FauxFeedbackQgis
    monkeypatch.setitem(sys.modules, "qgis", types.ModuleType("qgis"))
    monkeypatch.setitem(sys.modules, "qgis.core", faux)
    annule = {"oui": True}
    fb = create_cancellable_feedback(lambda: annule["oui"])

    assert fb.isCanceled() is True
    annule["oui"] = False
    assert fb.isCanceled() is True, "une annulation utilisateur ne se rétracte pas"


# --------------------------------------------------------------------------
# Câblage dans create_terrain_model
# --------------------------------------------------------------------------


@pytest.fixture
def mnt_pret(tmp_path, monkeypatch):
    """Neutralise tout sauf l'appel aux algorithmes QGIS."""
    monkeypatch.setattr(mnt, "validate_las_or_laz_with_pdal", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(mnt, "needs_refresh", lambda *a, **k: True)
    monkeypatch.setattr(mnt, "assign_crs_if_missing", lambda *a, **k: None)
    laz = tmp_path / "LHD_FXX_0821_6327_PTS_merged.laz"
    laz.write_bytes(b"LASF")
    return laz, tmp_path


def _creer_mnt(laz, temp_dir, **kw):
    return mnt.create_terrain_model(
        input_laz_path=laz,
        temp_dir=temp_dir,
        current_tile_name="LHD_FXX_0821_6327_PTS_merged",
        mnt_resolution=0.5,
        tile_overlap_percent=20,
        filter_expression="Classification = 2",
        **kw,
    )


class _FeedbackArmable:
    """Ce que ``_run_with_watchdog`` attend d'un feedback : armer / désarmer."""

    def __init__(self) -> None:
        self.expire = False

    def start_watchdog(self, timeout_s) -> None:
        self.expire = False

    def stop_watchdog(self) -> bool:
        expire, self.expire = self.expire, False
        return expire


def test_dalle_qui_depasse_le_delai_est_abandonnee_sans_fallback(mnt_pret, monkeypatch):
    laz, temp_dir = mnt_pret
    appels = []
    fb = _FeedbackArmable()

    def algo_qui_boucle(algorithm_id, parameters, **kwargs):
        appels.append(algorithm_id)
        fb.expire = True  # le chien de garde a mordu pendant l'exécution
        raise RuntimeError("Le processus s'est arrêté de façon inattendue")

    monkeypatch.setattr(mnt, "run_qgis_algorithm", algo_qui_boucle)

    with pytest.raises(TimeoutError):
        _creer_mnt(laz, temp_dir, feedback=fb)

    assert appels == ["pdal:exportrastertin"], (
        "après un dépassement, le fallback relirait le même LAZ pour reboucler"
    )


def test_echec_ordinaire_declenche_toujours_le_fallback(mnt_pret, monkeypatch):
    laz, temp_dir = mnt_pret
    appels = []

    def algo(algorithm_id, parameters, **kwargs):
        appels.append(algorithm_id)
        if algorithm_id == "pdal:exportrastertin":
            raise RuntimeError("algorithme indisponible")
        Path(parameters["OUTPUT"]).write_bytes(b"TIF")

    monkeypatch.setattr(mnt, "run_qgis_algorithm", algo)
    _creer_mnt(laz, temp_dir, feedback=_FeedbackArmable())

    assert appels == ["pdal:exportrastertin", "pdal:exportraster"]
