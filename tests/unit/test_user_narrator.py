"""Tests du système de logs duaux (USER_INFO + UserNarrator).

Couvre :

- L'enregistrement du niveau ``USER_INFO`` dans :mod:`logging`.
- Le routage technique vs narratif côté ``QtProgressReporter``.
- Le wording des événements émis par :class:`UserNarrator`.

Aucun de ces tests ne nécessite Qt : les signaux Qt sont mockés et le
logger est utilisé en standard Python.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.progress_reporter import USER_INFO, NullProgressReporter
from app.qt_progress_reporter import QtProgressReporter
from app.user_narrator import (
    PRODUCT_LABELS,
    UserNarrator,
    _format_duration,
    _human_count,
    create_user_narrator,
)


# ----------------------------------------------------------------------
# Niveau de log USER_INFO
# ----------------------------------------------------------------------
class TestUserInfoLevel:
    def test_value_between_info_and_warning(self):
        assert logging.INFO < USER_INFO < logging.WARNING

    def test_level_name_registered(self):
        assert logging.getLevelName(USER_INFO) == "USER_INFO"


# ----------------------------------------------------------------------
# QtProgressReporter : routage par niveau
# ----------------------------------------------------------------------
class TestQtProgressReporterRouting:
    def _setup(self):
        logger = logging.getLogger("test_qt_reporter")
        logger.setLevel(logging.DEBUG)
        # Reset handlers to isolate
        for h in list(logger.handlers):
            logger.removeHandler(h)
        emitter = MagicMock()
        reporter = QtProgressReporter(logger, emitter)
        return logger, reporter, emitter

    def test_info_emits_at_info_level(self):
        logger, reporter, _ = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.levelno, record.getMessage()))

        logger.addHandler(_Capture())
        reporter.info("technical detail")
        assert (logging.INFO, "technical detail") in captured

    def test_user_info_emits_at_user_info_level(self):
        logger, reporter, _ = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.levelno, record.getMessage()))

        logger.addHandler(_Capture())
        reporter.user_info("clear narrative")
        assert (USER_INFO, "clear narrative") in captured

    def test_user_info_passes_filter_at_user_info(self):
        """Un handler avec setLevel(USER_INFO) doit voir user_info mais pas info."""
        logger, reporter, _ = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        h = _Capture()
        h.setLevel(USER_INFO)
        logger.addHandler(h)

        reporter.info("dev only")
        reporter.user_info("user message")
        reporter.error("oops")

        assert "dev only" not in captured
        assert "user message" in captured
        assert "oops" in captured

    def test_error_emits_at_error_level(self):
        logger, reporter, _ = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.levelno, record.getMessage()))

        logger.addHandler(_Capture())
        reporter.error("fatal")
        assert (logging.ERROR, "fatal") in captured

    def test_user_warning_passes_filter(self):
        logger, reporter, _ = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.levelno, record.getMessage()))

        h = _Capture()
        h.setLevel(USER_INFO)
        logger.addHandler(h)

        reporter.user_warning("careful")
        # WARNING (30) > USER_INFO (25), donc visible.
        assert (logging.WARNING, "careful") in captured

    def test_stage_does_not_emit_log(self):
        logger, reporter, emitter = self._setup()
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        logger.addHandler(_Capture())
        reporter.stage("phase X")
        # stage() ne doit PAS aller dans le logger (uniquement signal Qt)
        assert "phase X" not in captured
        emitter.stage.emit.assert_called_once_with("phase X")


# ----------------------------------------------------------------------
# NullProgressReporter : tous les canaux sont silencieux
# ----------------------------------------------------------------------
class TestNullProgressReporter:
    def test_user_info_does_not_raise(self):
        r = NullProgressReporter()
        # Ne lève pas, ne retourne rien.
        assert r.user_info("anything") is None

    def test_all_methods_present(self):
        r = NullProgressReporter()
        r.info("a")
        r.error("b")
        r.user_info("c")
        r.user_warning("d")
        r.user_success("e")
        r.stage("f")
        r.progress(50)
        r.load_layers([], [], [])


# ----------------------------------------------------------------------
# Helpers de formatage
# ----------------------------------------------------------------------
class TestHelpers:
    def test_human_count_singular(self):
        assert _human_count(1, "dalle") == "1 dalle"

    def test_human_count_plural(self):
        assert _human_count(5, "dalle") == "5 dalles"

    def test_human_count_explicit_plural(self):
        assert _human_count(3, "dalle traitée", "dalles traitées") == "3 dalles traitées"

    def test_format_duration_seconds(self):
        assert _format_duration(15) == "15s"

    def test_format_duration_minutes(self):
        assert _format_duration(125) == "2min 05s"

    def test_format_duration_minutes_round(self):
        assert _format_duration(120) == "2min"

    def test_format_duration_hours(self):
        assert _format_duration(3725) == "1h 02min"


# ----------------------------------------------------------------------
# UserNarrator : wording et ordre des appels
# ----------------------------------------------------------------------
class TestUserNarratorEvents:
    def _make(self):
        reporter = MagicMock()
        return UserNarrator(reporter), reporter

    def test_pipeline_starting_records_start_time(self):
        narrator, reporter = self._make()
        narrator.pipeline_starting("téléchargement IGN")
        reporter.user_info.assert_called_once()
        msg = reporter.user_info.call_args[0][0]
        assert "Démarrage" in msg
        assert "téléchargement IGN" in msg

    def test_pipeline_complete_uses_recorded_start(self):
        narrator, reporter = self._make()
        narrator.pipeline_starting("X")
        narrator.pipeline_complete(tiles_processed=3, products=["MNT", "SVF"])
        success_msg = reporter.user_success.call_args[0][0]
        assert "✅" in success_msg
        assert "3 dalles traitées" in success_msg
        assert "modèle de terrain" in success_msg
        assert "facteur de vue du ciel" in success_msg

    def test_pipeline_complete_explicit_start_time(self):
        """``start_time`` explicite court-circuite l'horodatage interne."""
        import time

        narrator, reporter = self._make()
        # Pas de pipeline_starting() préalable.
        narrator.pipeline_complete(start_time=time.time() - 65)
        msg = reporter.user_success.call_args[0][0]
        assert "min" in msg

    def test_pipeline_failed(self):
        narrator, reporter = self._make()
        narrator.pipeline_starting("X")
        narrator.pipeline_failed("erreur réseau")
        msg = reporter.user_warning.call_args[0][0]
        assert "interrompu" in msg
        assert "erreur réseau" in msg

    def test_tiles_resolution_done_singular(self):
        narrator, reporter = self._make()
        narrator.tiles_resolution_done(1)
        msg = reporter.user_info.call_args[0][0]
        assert "1 dalle identifiée" in msg

    def test_tiles_resolution_done_plural(self):
        narrator, reporter = self._make()
        narrator.tiles_resolution_done(7)
        msg = reporter.user_info.call_args[0][0]
        assert "7 dalles identifiées" in msg

    def test_download_start_uses_count(self):
        narrator, reporter = self._make()
        narrator.download_start(12)
        msg = reporter.user_info.call_args[0][0]
        assert "12 dalles" in msg
        assert "Téléchargement" in msg

    def test_products_phase_start_lists_products(self):
        narrator, reporter = self._make()
        narrator.products_phase_start(5, ["MNT", "SVF", "M_HS"])
        msg = reporter.user_info.call_args[0][0]
        assert "5 dalles" in msg
        assert "facteur de vue du ciel" in msg
        assert "ombrage multi-directionnel" in msg

    def test_products_phase_start_mnt_only(self):
        narrator, reporter = self._make()
        narrator.products_phase_start(2, ["MNT"])
        msg = reporter.user_info.call_args[0][0]
        assert "MNT seul" in msg

    def test_tile_progress_truncates_long_names(self):
        narrator, reporter = self._make()
        narrator.tile_progress(1, 3, "LHD_FXX_0775_6300_PTS_C_LAMB93_IGN69_extra_long")
        msg = reporter.user_info.call_args[0][0]
        assert "1/3" in msg
        assert "…" in msg

    def test_cv_start_singular(self):
        narrator, reporter = self._make()
        narrator.cv_start(1)
        msg = reporter.user_info.call_args[0][0]
        assert "Détection" in msg
        assert "(" not in msg  # pas de "(N modèles)"

    def test_cv_start_multi(self):
        narrator, reporter = self._make()
        narrator.cv_start(3)
        msg = reporter.user_info.call_args[0][0]
        assert "3 modèles" in msg

    def test_cv_complete_zero(self):
        narrator, reporter = self._make()
        narrator.cv_complete(0)
        msg = reporter.user_info.call_args[0][0]
        assert "aucune" in msg

    def test_cv_complete_some(self):
        narrator, reporter = self._make()
        narrator.cv_complete(7)
        msg = reporter.user_info.call_args[0][0]
        assert "7 zones détectées" in msg


class TestUserNarratorFactory:
    def test_create_returns_instance(self):
        r = NullProgressReporter()
        n = create_user_narrator(r)
        assert isinstance(n, UserNarrator)

    def test_product_labels_cover_all_products(self):
        # Tous les codes produits utilisés dans products_cfg doivent
        # avoir un libellé humain.
        expected_codes = {"MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"}
        assert expected_codes.issubset(set(PRODUCT_LABELS.keys()))
