"""Carte graphique (0.15.0, vague 3), côté plugin : adaptateurs du poste, décision mémorisée dans
le profil (clé = cartes + pilotes + version d'onnxruntime, depuis le 2026-10-26 : ni le binaire ni le
modèle, « à partir du moment où elle est détectée et cochée, on doit l'utiliser »), bloc transmis au
moteur, trace du journal. Cochée par défaut (2026-10-26) ; un refus est réessayé après sept jours ;
une carte retenue l'est pour tous les modèles et toutes les livraisons, jusqu'à ce que la clé change
(carte, pilote, version du moteur ONNX)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from app.services import carte_graphique as cg

ADS = [{"id": 0, "nom": "Radeon", "pilote": "31.0.1"}, {"id": 1, "nom": "RTX", "pilote": "32.0.15"}]
BIN = {"sources_sha256": "abc", "onnxruntime": {"paquet": "onnxruntime-directml", "version": "1.24.4"}}
T0 = dt.datetime(2026, 10, 18, 12, 0)
RETENUE = {"appareil": "carte_graphique", "device_id": 1, "carte": "RTX"}
REFUSEE = {"appareil": "processeur", "motif": "Radeon : pas plus rapide que le processeur"}


class TestCle:
    def test_stable_et_sensible_aux_cartes_pilotes_et_moteur_seulement(self):
        """2026-10-26 : la clé ne porte plus ni le sha du binaire ni l'empreinte du modèle — chaque
        modèle et chaque livraison re-sondaient la carte (sept sondes sur un traitement à sept
        modèles, une de plus après chaque recompilation), ce que l'utilisateur a refusé."""
        a = cg.cle(ADS, BIN)
        assert a == cg.cle(list(reversed(ADS)), BIN)                                   # ordre des cartes sans effet
        assert a == cg.cle(ADS, {**BIN, "sources_sha256": "def"})                      # binaire recompilé : même clé
        assert a != cg.cle([{**ADS[1], "pilote": "33"}], BIN)                          # pilote changé
        assert a != cg.cle(ADS, {**BIN, "onnxruntime": {"paquet": "onnxruntime-directml", "version": "1.25"}})
        assert "abc" not in a and "onnx" not in a                                      # ni sha, ni modèle

    def test_une_carte_retenue_sert_a_tous_les_modeles_et_apres_recompilation(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        cg.memoriser(p, cg.cle(ADS, BIN), RETENUE, T0)
        autre_binaire = {**BIN, "sources_sha256": "recompile"}
        for run_cfg in ({"selected_model": "crateres"}, {"selected_model": "lineaires"}):
            cfg2, ctx = cg.preparer(run_cfg, True, adaptateurs=ADS, binaire=autre_binaire, registre=p,
                                    maintenant=T0 + dt.timedelta(days=30))
            assert cfg2["carte_graphique"] == {"demande": True, "device_id": 1, "carte": "RTX"}
            assert ctx["decision"] is not None and ctx["message"] is None


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
        # Le moteur énumère lui-même les cartes : le bloc ne transmet aucune liste (l'ordre vu
        # de QGIS n'est pas le sien — vécu le 18/10, noms inversés sur un portable à deux cartes).
        bloc = cg.bloc_run(True, ADS, None, BIN)
        assert bloc == {"demande": True}

    def test_retenue_memorisee(self):
        bloc = cg.bloc_run(True, ADS, {**RETENUE, "date": T0.isoformat()}, BIN)
        assert bloc == {"demande": True, "device_id": 1, "carte": "RTX"}   # index vérifié par le moteur

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

    def test_repli_en_cours_de_run_devient_un_refus(self, tmp_path):
        # Une carte qui plante à l'image k ne doit pas être re-sondée, retenue et repliée à chaque run.
        p = tmp_path / cg.NOM_REGISTRE
        stats = {"appareil": {"appareil": "processeur", "repli": True, "motif": "DML_ERROR", "image": 7,
                              "message": "…"}}
        d = cg.enregistrer(stats, "k", p, T0)
        assert d["appareil"] == "processeur" and "image 7" in d["motif"] and "DML_ERROR" in d["motif"]
        assert cg.decision_valide(cg.lire(p), "k", T0 + dt.timedelta(days=1)) is not None
        assert cg.decision_valide(cg.lire(p), "k", T0 + dt.timedelta(days=8)) is None

    def test_ce_que_le_moteur_marque_non_memorisable_ne_l_est_pas(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        for d in ({"appareil": "processeur", "memorisable": False, "motif": "indisponible aujourd'hui"},
                  {"appareil": "carte_graphique", "device_id": 1, "carte": "RTX", "memorisable": False},
                  {"appareil": "processeur", "non_sondee": True, "memorisable": False, "motif": "x"}):
            assert cg.enregistrer({"appareil": d}, "k", p, T0) is None
        assert not p.exists()

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
        cfg2, ctx = cg.preparer(run_cfg, True, adaptateurs=ADS, binaire=BIN, registre=p, maintenant=T0)
        assert run_cfg == {"selected_model": "m"}               # jamais modifiée en place
        assert cfg2["carte_graphique"]["demande"] is True and ctx["cle"] == cg.cle(ADS, BIN)
        assert ctx["message"] is None

    def test_refus_memorise_narre_sans_sonder(self, tmp_path):
        p = tmp_path / cg.NOM_REGISTRE
        cle = cg.cle(ADS, BIN)
        cg.memoriser(p, cle, REFUSEE, T0)
        cfg2, ctx = cg.preparer({"selected_model": "m"}, True, adaptateurs=ADS, binaire=BIN,
                                registre=p, maintenant=T0 + dt.timedelta(days=1))
        assert cfg2["carte_graphique"]["demande"] is False and "écartée" in ctx["message"]

    def test_option_decochee(self, tmp_path):
        cfg2, ctx = cg.preparer({"selected_model": "m"}, False, adaptateurs=ADS, binaire=BIN,
                                registre=tmp_path / "r.json", maintenant=T0)
        assert "carte_graphique" not in cfg2 and ctx["cle"] is None


def test_adaptateurs_hors_windows_vide(monkeypatch):
    monkeypatch.setattr(cg.sys, "platform", "linux")
    cg.adaptateurs.cache_clear()
    assert cg.adaptateurs() == []
    cg.adaptateurs.cache_clear()


def test_adaptateurs_viennent_du_module_du_moteur(monkeypatch):
    from pipeline.cv import appareil as ap
    monkeypatch.setattr(cg.sys, "platform", "win32")
    monkeypatch.setattr(ap, "adaptateurs_dxgi", lambda: [{"id": 0, "nom": "X", "pilote": "1"}])
    cg.adaptateurs.cache_clear()
    assert cg.adaptateurs() == [{"id": 0, "nom": "X", "pilote": "1"}]
    cg.adaptateurs.cache_clear()
    assert "ctypes" not in (Path(cg.__file__).read_text(encoding="utf-8"))


def test_defaut_coche_et_persistance_ui():
    """Cochée par défaut depuis le 2026-10-26 (« et par défaut on la coche ») : la configuration par
    défaut, la case de l'étape 4 et le repli du chargement d'une ancienne configuration sans la clé."""
    racine = Path(__file__).resolve().parents[2]
    from config.config_manager import ConfigManager
    assert ConfigManager(racine).default_config()["computer_vision"]["carte_graphique"] is True
    etape4 = (racine / "src/ui/steps/step_4_launch.py").read_text(encoding="utf-8")
    assert "carte_graphique_changed" in etape4 and "self._carte_check.setChecked(True)" in etape4
    assert "self._carte = True" in etape4
    wizard = (racine / "src/ui/wizard_dialog.py").read_text(encoding="utf-8")
    assert '["carte_graphique"] = ' in wizard and 'set_carte_graphique(bool(cv.get("carte_graphique", True)))' in wizard
    import json
    assert json.loads((racine / "config.example.json").read_text(encoding="utf-8"))["computer_vision"]["carte_graphique"] is True


def test_le_journal_renvoie_au_depannage():
    from app.services.aide import rubrique_depannage
    assert rubrique_depannage("⚠️ « Cratères » : carte graphique écartée : Radeon : pas plus rapide") == \
        "La carte graphique n'est pas utilisée"
    assert rubrique_depannage("⚠️ « Cratères » : carte graphique : erreur à l'image 3 ; suite sur le processeur") == \
        "La carte graphique n'est pas utilisée"


def test_decision_memorisee_la_plus_recente():
    reg = {
        "k1": {**REFUSEE, "date": T0.isoformat()},
        "k2": {**RETENUE, "date": (T0 + dt.timedelta(days=2)).isoformat()},
        "k3": "n'importe quoi",
        "k4": {**RETENUE, "date": "hier"},
    }
    d = cg.decision_memorisee(reg)
    assert d is not None and d["carte"] == "RTX" and d["date_fr"] == "20/10/2026"
    assert cg.decision_memorisee({}) is None
    assert cg.decision_memorisee({"k": "x"}) is None
    assert "RTX" in cg.phrase_decision(d) and "20/10/2026" in cg.phrase_decision(d)
    refus = cg.decision_memorisee({"k1": reg["k1"]})
    assert "processeur" in cg.phrase_decision(refus) and "18/10/2026" in cg.phrase_decision(refus)


def test_oublier_supprime_le_registre(tmp_path):
    p = tmp_path / "profil" / cg.NOM_REGISTRE
    cg.memoriser(p, "k", RETENUE, T0)
    assert p.is_file()
    assert cg.oublier(p) is True
    assert not p.exists()
    assert cg.oublier(p) is False          # rien à oublier : rien, sans erreur
