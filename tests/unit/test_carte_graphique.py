"""Carte graphique (0.15.0, vague 3), côté plugin : adaptateurs du poste, décision mémorisée dans
le profil (clé = cartes + pilotes + binaire + modèle), bloc transmis au moteur, trace du journal.
Décochée par défaut ; un refus est réessayé après sept jours ; une carte retenue l'est jusqu'à ce
que la clé change (pilote, binaire, modèle)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from app.services import carte_graphique as cg

ADS = [{"id": 0, "nom": "Radeon", "pilote": "31.0.1"}, {"id": 1, "nom": "RTX", "pilote": "32.0.15"}]
BIN = {"sources_sha256": "abc", "onnxruntime": {"paquet": "onnxruntime-directml", "version": "1.24.4"}}
T0 = dt.datetime(2026, 10, 18, 12, 0)
RETENUE = {"appareil": "carte_graphique", "device_id": 1, "carte": "RTX"}
REFUSEE = {"appareil": "processeur", "motif": "Radeon : pas plus rapide que le processeur"}


class TestCle:
    def test_stable_et_sensible_a_chaque_partie(self):
        a = cg.cle(ADS, BIN, "m1-123")
        assert a == cg.cle(list(reversed(ADS)), BIN, "m1-123")   # ordre des cartes sans effet
        assert a != cg.cle(ADS, {**BIN, "sources_sha256": "def"}, "m1-123")
        assert a != cg.cle(ADS, BIN, "m1-124")
        assert a != cg.cle([{**ADS[1], "pilote": "33"}], BIN, "m1-123")


class TestDecisionMemorisee:
    def test_retenue_toujours_valide(self):
        reg = {"k": {**RETENUE, "date": T0.isoformat()}}
        assert cg.decision_valide(reg, "k", T0 + dt.timedelta(days=400)) == reg["k"]

    def test_refus_reessaye_apres_sept_jours(self):
        reg = {"k": {**REFUSEE, "date": T0.isoformat()}}
        assert cg.decision_valide(reg, "k", T0 + dt.timedelta(days=6)) == reg["k"]
        assert cg.decision_valide(reg, "k", T0 + dt.timedelta(days=8)) is None

    def test_cle_inconnue_ou_entree_illisible(self):
        assert cg.decision_valide({}, "k", T0) is None
        assert cg.decision_valide({"k": "n'importe quoi"}, "k", T0) is None
        assert cg.decision_valide({"k": {**REFUSEE, "date": "hier"}}, "k", T0) is None

    def test_memoriser_et_relire(self, tmp_path):
        p = tmp_path / "profil" / cg.NOM_REGISTRE
        cg.memoriser(p, "k", RETENUE, T0)
        reg = cg.lire(p)
        assert reg["k"]["carte"] == "RTX" and reg["k"]["date"] == T0.isoformat()
        cg.memoriser(p, "k2", REFUSEE, T0)
        assert set(cg.lire(p)) == {"k", "k2"}
        assert cg.lire(tmp_path / "absent.json") == {}
        p.write_text("{corrompu", encoding="utf-8")
        assert cg.lire(p) == {}


class TestBlocRun:
    def test_option_decochee_rien(self):
        assert cg.bloc_run(False, ADS, None, BIN) is None

    def test_sonde_a_faire(self):
        bloc = cg.bloc_run(True, ADS, None, BIN)
        assert bloc["demande"] is True and bloc["adaptateurs"] == ADS and "device_id" not in bloc

    def test_retenue_memorisee(self):
        bloc = cg.bloc_run(True, ADS, {**RETENUE, "date": T0.isoformat()}, BIN)
        assert bloc["demande"] is True and bloc["device_id"] == 1

    def test_refus_memorise(self):
        bloc = cg.bloc_run(True, ADS, {**REFUSEE, "date": T0.isoformat()}, BIN)
        assert bloc["demande"] is False and "écartée" in bloc["message"] and "18/10/2026" in bloc["message"]

    def test_sans_carte_ou_hors_windows(self):
        bloc = cg.bloc_run(True, [], None, BIN)
        assert bloc["demande"] is False and "aucune carte" in bloc["message"]

    def test_binaire_sans_directml(self):
        bloc = cg.bloc_run(True, ADS, None, {"sources_sha256": "abc"})
        assert bloc["demande"] is False and "moteur de détection" in bloc["message"]


class TestEnregistrer:
    def test_decision_du_moteur_memorisee(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        stats = {"appareil": {"appareil": "carte_graphique", "device_id": 1, "carte": "RTX",
                              "sonde": {"secondes": 1.0}, "message": "…"}}
        d = cg.enregistrer(stats, "k", p, T0)
        assert d["carte"] == "RTX" and "message" not in d and "sonde" in d
        assert cg.lire(p)["k"]["date"] == T0.isoformat()

    def test_repli_en_cours_de_run_n_est_pas_memorise(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        stats = {"appareil": {"appareil": "processeur", "repli": True, "motif": "erreur"}}
        assert cg.enregistrer(stats, "k", p, T0) is None and not p.exists()

    def test_decision_deja_memorisee_non_reecrite(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        stats = {"appareil": {"appareil": "carte_graphique", "device_id": 1, "memorise": True}}
        assert cg.enregistrer(stats, "k", p, T0) is None and not p.exists()

    def test_sans_appareil(self, tmp_path):
        assert cg.enregistrer({}, "k", tmp_path / "x.json", T0) is None


class TestPreparer:
    def test_bloc_pose_dans_la_config_du_run(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        run_cfg = {"selected_model": "m"}
        cfg2, ctx = cg.preparer(run_cfg, True, adaptateurs=ADS, binaire=BIN, empreinte_modele="m-1",
                                registre=p, maintenant=T0)
        assert run_cfg == {"selected_model": "m"}               # jamais modifiée en place
        assert cfg2["carte_graphique"]["demande"] is True and ctx["cle"] == cg.cle(ADS, BIN, "m-1")
        assert ctx["message"] is None

    def test_refus_memorise_narre_sans_sonder(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        cle = cg.cle(ADS, BIN, "m-1")
        cg.memoriser(p, cle, REFUSEE, T0)
        cfg2, ctx = cg.preparer({"selected_model": "m"}, True, adaptateurs=ADS, binaire=BIN,
                                empreinte_modele="m-1", registre=p, maintenant=T0 + dt.timedelta(days=1))
        assert cfg2["carte_graphique"]["demande"] is False and "écartée" in ctx["message"]

    def test_option_decochee(self, tmp_path):
        cfg2, ctx = cg.preparer({"selected_model": "m"}, False, adaptateurs=ADS, binaire=BIN,
                                empreinte_modele="m-1", registre=tmp_path / "r.json", maintenant=T0)
        assert "carte_graphique" not in cfg2 and ctx["cle"] is None


def test_empreinte_modele(tmp_path):
    poids = tmp_path / "best.onnx"
    poids.write_bytes(b"x" * 10)
    e = cg.empreinte_modele(poids)
    assert e.startswith("best.onnx:10:")
    assert cg.empreinte_modele(tmp_path / "absent.onnx") == "absent.onnx"
    assert cg.empreinte_modele(None) == "?"


def test_adaptateurs_hors_windows_vide(monkeypatch):
    monkeypatch.setattr(cg.sys, "platform", "linux")
    cg.adaptateurs.cache_clear()
    assert cg.adaptateurs() == []
    cg.adaptateurs.cache_clear()


def test_defaut_decoche_et_persistance_ui():
    racine = Path(__file__).resolve().parents[2]
    from config.config_manager import ConfigManager
    assert ConfigManager(racine).default_config()["computer_vision"]["carte_graphique"] is False
    etape4 = (racine / "src/ui/steps/step_4_launch.py").read_text(encoding="utf-8")
    assert "carte_graphique_changed" in etape4 and "setChecked(False)" in etape4
    wizard = (racine / "src/ui/wizard_dialog.py").read_text(encoding="utf-8")
    assert '["carte_graphique"] = ' in wizard and "set_carte_graphique(" in wizard


def test_le_journal_renvoie_au_depannage():
    from app.services.aide import rubrique_depannage
    assert rubrique_depannage("⚠️ « Cratères » : carte graphique écartée : Radeon : pas plus rapide") == \
        "La carte graphique n'est pas utilisée"
    assert rubrique_depannage("⚠️ « Cratères » : carte graphique : erreur à l'image 3 ; suite sur le processeur") == \
        "La carte graphique n'est pas utilisée"
