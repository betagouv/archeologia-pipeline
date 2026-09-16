from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from app.run_context import build_run_context
from app.runners.registry import get_runner
from app.runners.ign_local_runner import IgnOrLocalRunner
from app.runners.existing_mnt_runner import ExistingMntRunner
from app.runners.existing_rvt_runner import ExistingRvtRunner


class TestRunnersIntegration:
    """⚠ Les trois premiers tests de ce fichier étaient
    ``assert runner is not None`` et ``hasattr(runner, "run")`` : un
    constructeur Python ne rend jamais None et la méthode est écrite dans le
    même dépôt. Ils ne pouvaient pas échouer (audit 2026-09-16). Ce qui compte
    vraiment est que les quatre runners honorent le MÊME contrat d'appel — c'est
    ce que le contrôleur suppose quand il délègue sans savoir lequel il tient.
    """

    @pytest.mark.parametrize(
        "classe", [IgnOrLocalRunner, ExistingMntRunner, ExistingRvtRunner]
    )
    def test_chaque_runner_honore_le_contrat_d_appel(self, classe):
        """``run(ctx=…, reporter=…, cancel=…, slog=…)``, en mots-clés.

        Le contrôleur appelle TOUJOURS par mot-clé (pipeline_controller.py) : un
        runner qui renommerait un paramètre planterait à l'exécution, dans QGIS,
        sur le seul mode concerné — donc potentiellement longtemps après.
        """
        params = inspect.signature(classe().run).parameters
        for requis in ("ctx", "reporter", "cancel"):
            assert requis in params, f"{classe.__name__}.run sans paramètre {requis}"
        # ``slog`` est passé par le contrôleur : optionnel, mais accepté.
        assert "slog" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), f"{classe.__name__}.run refuserait le slog du contrôleur"

    def test_registry_returns_correct_runner_types(self):
        assert isinstance(get_runner("ign_laz"), IgnOrLocalRunner)
        assert isinstance(get_runner("local_laz"), IgnOrLocalRunner)
        assert isinstance(get_runner("existing_mnt"), ExistingMntRunner)
        assert isinstance(get_runner("existing_rvt"), ExistingRvtRunner)

    def test_registry_raises_for_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_runner("unknown_mode")

    def test_run_context_extracts_mode_correctly(self, config_with_output_dir: dict):
        config_with_output_dir["app"]["files"]["data_mode"] = "existing_mnt"
        ctx = build_run_context(config_with_output_dir)
        
        assert ctx.mode == "existing_mnt"

    def test_run_context_extracts_files_config(self, config_with_output_dir: dict):
        config_with_output_dir["app"]["files"]["existing_mnt_dir"] = "/tmp/mnt"
        ctx = build_run_context(config_with_output_dir)

        assert ctx.files.existing_mnt_dir == Path("/tmp/mnt")
