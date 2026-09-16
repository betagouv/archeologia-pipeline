"""Le contrôleur de pipeline : ce qu'il fait AVANT de lancer un runner.

Il ne fait que trois choses (cf. CLAUDE.md) : valider le contexte, passer le
préflight, puis déléguer au runner du mode. Ce sont ces trois portes qu'on
verrouille ici — chacune doit pouvoir REFUSER de lancer, et le refus doit
remonter comme un échec, pas comme un silence.

Les deux premiers tests de ce fichier étaient ``assert controller is not None``
et ``hasattr(controller, "run")`` : un constructeur Python ne rend jamais None
et la méthode est écrite dans le même dépôt. Ils ne pouvaient pas échouer
(audit 2026-09-16) ; ils sont remplacés par les portes réelles.
"""
from __future__ import annotations

import threading
from pathlib import Path

from app.cancel_token import CancelToken
from app.pipeline_controller import PipelineController
from app.progress_reporter import NullProgressReporter
from app.run_context import build_run_context, validate_run_context


class _ReporterEnregistreur(NullProgressReporter):
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _ctx_valide(cfg: dict, tmp_path: Path):
    """Contexte qui PASSE la validation, pour atteindre les portes suivantes.

    La configuration par défaut en mode IGN est invalide (ni zone ni produit) :
    sans ces deux réglages, le contrôleur refusait dès la première porte et les
    tests du préflight passaient pour la mauvaise raison.
    """
    liste = tmp_path / "dalles.txt"
    liste.write_text("LHD_FXX_0500_6500,http://exemple/x.laz\n", encoding="utf-8")
    cfg["app"]["files"]["data_mode"] = "ign_laz"
    cfg["app"]["files"]["input_file"] = str(liste)
    cfg["processing"]["products"] = dict(cfg["processing"]["products"], MNT=True)
    ctx = build_run_context(cfg)
    erreurs, _ = validate_run_context(ctx)
    assert erreurs == [], f"le contexte de test est invalide : {erreurs}"
    return ctx


class TestPipelineControllerPortes:
    def test_preflight_en_echec_empeche_le_lancement(
        self, config_with_output_dir: dict, tmp_path: Path, monkeypatch
    ):
        """Un préflight négatif doit rendre False ET ne lancer aucun runner.

        C'est la promesse du contrôleur : pas d'outil externe, pas de run. Si le
        refus n'empêchait pas la délégation, le pipeline partirait sans PDAL et
        échouerait bien plus loin, avec un message incompréhensible.
        """
        lances: list[str] = []
        monkeypatch.setattr("pipeline.preflight.run_preflight", lambda **kw: False)
        monkeypatch.setattr(
            "app.runners.registry.get_runner",
            lambda mode: lances.append(mode),
        )

        ctx = _ctx_valide(config_with_output_dir, tmp_path)
        verdict = PipelineController().run(
            ctx=ctx,
            reporter=_ReporterEnregistreur(),
            cancel=CancelToken(threading.Event()),
        )

        assert verdict is False
        assert lances == [], "un runner a été lancé malgré un préflight en échec"

    def test_contexte_invalide_refuse_avant_meme_le_preflight(
        self, config_with_output_dir: dict, monkeypatch
    ):
        """Une erreur de configuration se voit sans lancer d'outil externe.

        Le préflight coûte cher (il sonde pdal, gdal, les algorithmes QGIS) :
        un mode sans dossier d'entrée doit être refusé avant.
        """
        appels: list[int] = []
        monkeypatch.setattr(
            "pipeline.preflight.run_preflight",
            lambda **kw: appels.append(1) or True,
        )
        cfg = config_with_output_dir
        cfg["app"]["files"]["data_mode"] = "existing_mnt"
        cfg["app"]["files"]["existing_mnt_dir"] = ""   # requis pour ce mode

        reporter = _ReporterEnregistreur()
        verdict = PipelineController().run(
            ctx=build_run_context(cfg),
            reporter=reporter,
            cancel=CancelToken(threading.Event()),
        )

        assert verdict is False
        assert appels == [], "le préflight a été lancé malgré un contexte invalide"
        assert reporter.errors, "l'erreur de configuration n'est pas remontée à l'UI"

    def test_annulation_avant_le_lancement_ne_lance_rien(
        self, config_with_output_dir: dict, tmp_path: Path, monkeypatch
    ):
        """Annuler entre le préflight et le runner rend None, pas False.

        L'interface distingue les deux : False affiche un échec, None une
        annulation demandée par l'utilisateur.
        """
        lances: list[str] = []
        monkeypatch.setattr("pipeline.preflight.run_preflight", lambda **kw: True)
        monkeypatch.setattr(
            "app.runners.registry.get_runner", lambda mode: lances.append(mode)
        )

        evenement = threading.Event()
        evenement.set()
        verdict = PipelineController().run(
            ctx=_ctx_valide(config_with_output_dir, tmp_path),
            reporter=_ReporterEnregistreur(),
            cancel=CancelToken(evenement),
        )

        assert verdict is None
        assert lances == []


class TestPipelineControllerIntegration:
    def test_run_context_builds_correctly(
        self, config_with_output_dir: dict, temp_output_dir: Path
    ):
        ctx = build_run_context(config_with_output_dir)
        assert ctx.output_dir == temp_output_dir
        assert ctx.mode == "ign_laz"

    def test_cancel_token_is_set(self):
        cancel_event = threading.Event()
        cancel = CancelToken(cancel_event)

        assert cancel.is_cancelled() is False
        cancel_event.set()
        assert cancel.is_cancelled() is True
