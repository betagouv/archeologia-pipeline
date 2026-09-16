from __future__ import annotations

import inspect

import pytest

from pipeline.preflight import run_preflight, CheckResult


class TestPreflightChecks:
    """⚠ ``assert callable(run_preflight)`` après l'avoir importé en tête de
    module, et la relecture des champs d'une dataclass qu'on vient d'écrire,
    ne testaient rien du dépôt (audit 2026-09-16). Remplacés par ce que le
    préflight PROMET : refuser quand une entrée requise manque, accepter
    quand tout est là.
    """

    def test_preflight_has_expected_signature(self):
        sig = inspect.signature(run_preflight)
        param_names = list(sig.parameters.keys())

        assert "log" in param_names
        assert "mode" in param_names
        assert "cv_config" in param_names
        assert "products" in param_names
        assert "files_config" in param_names
        assert "output_dir" in param_names

    def test_dossier_rvt_inexistant_est_refuse(self, tmp_path):
        """Le cas qui motive le préflight : lancer sur un dossier absent.

        ``isinstance(result, bool)`` était vrai quoi qu'il arrive ; ce qui
        compte est que le VERDICT soit négatif et qu'il soit EXPLIQUÉ dans le
        journal — l'utilisateur doit savoir quoi corriger.
        """
        logs: list[str] = []
        result = run_preflight(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={},
            log=logs.append,
            files_config={"existing_rvt_dir": str(tmp_path / "absent")},
            output_dir=None,
        )
        assert result is False
        assert logs, "un refus sans explication au journal est inexploitable"

    @pytest.mark.parametrize(
        "critique, verdict_attendu",
        [(True, False), (False, True)],
        ids=["critique -> refus", "optionnel -> passe"],
    )
    def test_seul_un_echec_critique_arrete_le_lancement(
        self, monkeypatch, critique, verdict_attendu
    ):
        """``critical`` est ce qui sépare l'avertissement du refus.

        ⚠ La version précédente de ce test construisait deux ``CheckResult``
        puis relisait les champs qu'elle venait d'écrire, sans jamais appeler
        ``run_preflight`` : exactement l'anti-pattern que ce fichier prétend
        supprimer (relecture des commits, 2026-09-16). On force donc la liste
        des contrôles et on vérifie le VERDICT que le contrôleur lira.
        """
        echec = CheckResult(
            name="gdaladdo", ok=False, details="absent", critical=critique
        )
        succes = CheckResult(name="pdal", ok=True, details="/bin/pdal", critical=True)
        monkeypatch.setattr(
            "pipeline.preflight.collect_preflight_results",
            lambda **kw: [succes, echec],
        )

        logs: list[str] = []
        verdict = run_preflight(
            mode="existing_rvt",
            cv_config={"enabled": False},
            products={},
            log=logs.append,
            files_config={},
            output_dir=None,
        )

        assert verdict is verdict_attendu
        # Dans les DEUX cas le contrôle en échec est nommé au journal : un
        # optionnel absent reste visible même quand il ne bloque pas.
        assert any("gdaladdo" in ligne for ligne in logs)
