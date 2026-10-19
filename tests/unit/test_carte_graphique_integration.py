"""Carte graphique (0.15.0) : la décision du binaire traverse la chaîne — ligne ``appareil=``
relue par ``external_runner``, déposée dans ``stats``, narrée, puis dans le rapport et
``traitement.json`` ; le CLI sonde après la première image et enveloppe la session."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.services.finalize_service import trace_appareil
from app.services.rapport_run import _cv_runs
from app.user_narrator import UserNarrator
from pipeline.cv import appareil as ap
from pipeline.cv.external_runner import _parse_runner_stdout

RACINE = Path(__file__).resolve().parents[2]


class TestLigneAppareil:
    def test_relayee_au_journal_et_au_callback(self):
        decision = ap.decision_memorisee(1, "RTX")
        logs, recus = [], []
        assert _parse_runner_stdout(ap.ligne(decision), logs.append, on_appareil=recus.append) is None
        assert recus == [decision]
        assert any("carte graphique « RTX »" in m for m in logs)

    def test_ligne_illisible_relayee_telle_quelle(self):
        logs = []
        _parse_runner_stdout("appareil={pas du json", logs.append)
        assert logs == ["[cv_runner] appareil={pas du json"]

    def test_callback_en_erreur_n_arrete_pas_la_lecture(self):
        def boom(_d):
            raise RuntimeError("ui")
        logs = []
        _parse_runner_stdout(ap.ligne(ap.decision_memorisee(0, "X")), logs.append, on_appareil=boom)
        assert logs


class TestCli:
    def test_le_cli_sonde_apres_la_premiere_image_et_enveloppe_la_session(self):
        cli = (RACINE / "dev/runner_onnx/cv_runner_onnx_cli.py").read_text(encoding="utf-8")
        boucle = cli[cli.index("for idx, jpg_file in enumerate(jpg_files, 1):"):cli.index("summary: success=")]
        # Sonde après l'inférence processeur de la première image, avant la ligne de progression.
        assert boucle.index("run_onnx_inference(") < boucle.index("_sonder(jpg_file, image_name, secondes_image")
        assert boucle.index("_sonder(jpg_file") < boucle.index("status=done")
        # Carte déjà retenue : appliquée à la création de la session, sans sonde.
        assert boucle.index("charger_session_partagee(") < boucle.index("_carte_retenue(onnx_session)")
        assert cli.count("avec_repli(") == 2 and "decision_repli(" in cli
        # Windows seulement, et jamais sans bloc posé par le plugin.
        assert 'sys.platform == "win32"' in cli and 'cv_config.get("carte_graphique")' in cli


class TestRapportEtTrace:
    def test_rapport_dit_la_carte(self):
        html = _cv_runs([{"modele": "M", "model": "m", "target_rvt": "LD", "seuil": 0.3, "images": 4,
                          "secondes": 8.0, "appareil": "carte_graphique", "carte": "RTX <4060>"}])
        assert "sur la carte graphique « RTX &lt;4060&gt; »" in html

    def test_rapport_dit_le_repli(self):
        html = _cv_runs([{"modele": "M", "model": "m", "target_rvt": "LD", "images": 4, "secondes": 8.0,
                          "appareil": "processeur", "repli": True}])
        assert "repli sur le processeur" in html

    def test_rapport_sans_appareil_inchange(self):
        html = _cv_runs([{"modele": "M", "model": "m", "target_rvt": "LD", "images": 4, "secondes": 8.0}])
        assert "carte graphique" not in html and "processeur" not in html

    def test_trace_traitement_json(self):
        assert trace_appareil({"appareil": "carte_graphique", "carte": "RTX"}) == {"appareil": "carte_graphique", "carte": "RTX"}
        assert trace_appareil({"appareil": "processeur", "repli": True}) == {"appareil": "processeur", "repli_processeur": True}
        assert trace_appareil({"moteur": "binaire"}) == {} and trace_appareil(None) == {}

    def test_trace_apparie_modele_et_indice(self):
        # Deux runs du même modèle sur deux indices : chacun son appareil dans traitement.json.
        src = (RACINE / "src/app/services/finalize_service.py").read_text(encoding="utf-8")
        bloc = src[src.index('"cv_runs": ['):src.index('"detections_entities": [')]
        assert 's.get("target_rvt") == r.get("target_rvt", "")' in bloc


def test_le_message_du_moteur_sans_directml_renvoie_a_la_bonne_rubrique():
    from app.services.aide import rubrique_depannage
    assert rubrique_depannage("⚠️ « M » : carte graphique demandée, mais le moteur de détection livré ne la "
                              "prend pas en charge ; détection sur le processeur") == "La carte graphique n'est pas utilisée"
    assert rubrique_depannage("✗ Détection impossible après 2min : moteur de détection introuvable") == \
        "La détection est impossible"


def test_le_cli_leve_sans_json_et_sonde_sans_detection():
    cli = (RACINE / "dev/runner_onnx/cv_runner_onnx_cli.py").read_text(encoding="utf-8")
    assert "inférence sans résultat sur la carte" in cli
    boucle = cli[cli.index("for idx, jpg_file in enumerate(jpg_files, 1):"):cli.index("summary: success=")]
    assert "if sonde_a_faire and ok" not in boucle            # « ok » est faux sans détection
    assert "sortie_cpu.exists()" in boucle and "decision_non_sondee(" in cli
    assert "seuil_sonde = float(effective_conf) if is_segmentation" in cli
    # 0.15.1 : le binaire énumère lui-même ses cartes et vérifie l'index d'une carte mémorisée.
    assert "cartes_poste = app_mod.adaptateurs_dxgi()" in cli and "carte_au_bon_index(cartes_poste" in cli
    assert 'cg.get("adaptateurs")' not in cli


def test_le_registre_du_poste_n_est_ni_versionne_ni_livre():
    assert "carte_graphique.json" in (RACINE / ".gitignore").read_text(encoding="utf-8")
    assert '"carte_graphique.json"' in (RACINE / "dev/package_plugin.py").read_text(encoding="utf-8")


class TestNarrateur:
    def _narr(self):
        r = MagicMock()
        return UserNarrator(r), r

    def test_carte_retenue_en_info(self):
        n, r = self._narr()
        n.cv_appareil("Cratères", ap.decision_memorisee(1, "RTX"))
        r.user_info.assert_called_once()
        assert "« Cratères »" in r.user_info.call_args[0][0] and "RTX" in r.user_info.call_args[0][0]
        r.user_warning.assert_not_called()

    def test_carte_ecartee_en_avertissement(self):
        n, r = self._narr()
        n.cv_appareil("Cratères", {"appareil": "processeur", "message": "carte graphique écartée : lente"})
        r.user_warning.assert_called_once()
        assert r.user_warning.call_args[0][0].startswith("⚠️ « Cratères » : carte graphique écartée")

    def test_fin_de_run_dit_la_carte(self):
        n, r = self._narr()
        n.cv_run_done("M", 3, 9.0, appareil="carte_graphique")
        assert r.user_info.call_args[0][0].endswith("sur la carte graphique")
        n.cv_run_done("M", 3, 9.0)
        assert "carte graphique" not in r.user_info.call_args[0][0]
