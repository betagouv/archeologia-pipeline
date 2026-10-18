"""Carte graphique (0.15.0, vague 3) : la sonde du binaire, pure et sans ORT.

La première image est inférée sur le processeur (résultat gardé) puis rejouée sur chaque carte ;
une carte est retenue si elle ouvre vraiment DirectML, si ses détections recoupent celles du
processeur hors bascules au seuil, et si elle est plus rapide. Repli processeur annoncé à la
première erreur en cours de traitement."""
from __future__ import annotations

import json

import pytest

from pipeline.cv import appareil as ap


def _det(conf, x=0.0, cls=0, dx=0.0):
    return {"class_id": cls, "confidence": conf,
            "bbox_absolute": {"minx": x + dx, "miny": 0.0, "maxx": x + 10.0 + dx, "maxy": 10.0}}


SEUIL = 0.30


def seuil_de(_d):
    return SEUIL


class TestComparerDetections:
    def test_memes_objets_memes_scores(self):
        cpu = [_det(0.9, 0), _det(0.5, 100)]
        res = ap.comparer_detections(cpu, [dict(d) for d in cpu], seuil_de)
        assert res["conforme"] and res["verifiable"]
        assert res["apparies"] == 2 and res["non_retrouvees"] == 0 and res["en_trop"] == 0
        assert res["ecart_conf_max"] == 0.0

    def test_objet_franc_non_retrouve(self):
        res = ap.comparer_detections([_det(0.9, 0), _det(0.8, 100)], [_det(0.9, 0)], seuil_de)
        assert not res["conforme"] and res["non_retrouvees"] == 1

    def test_objet_en_trop_sur_la_carte(self):
        res = ap.comparer_detections([_det(0.9, 0)], [_det(0.9, 0), _det(0.7, 100)], seuil_de)
        assert not res["conforme"] and res["en_trop"] == 1

    def test_bascule_au_seuil_toleree(self):
        # Un objet à 0,33 (seuil 0,30 + moins de 0,05) absent d'un côté = bascule, pas un écart.
        res = ap.comparer_detections([_det(0.9, 0), _det(0.33, 100)], [_det(0.9, 0)], seuil_de)
        assert res["conforme"] and res["bascules"] == 1 and res["non_retrouvees"] == 0

    def test_ecart_de_confiance_trop_grand(self):
        res = ap.comparer_detections([_det(0.9, 0)], [_det(0.9 - ap.ECART_CONF_MAX - 0.01, 0)], seuil_de)
        assert not res["conforme"] and res["ecart_conf_max"] > ap.ECART_CONF_MAX

    def test_petit_ecart_de_confiance_accepte(self):
        res = ap.comparer_detections([_det(0.9, 0)], [_det(0.89, 0)], seuil_de)
        assert res["conforme"] and res["ecart_conf_max"] == pytest.approx(0.01)

    def test_classe_differente_ne_s_apparie_pas(self):
        res = ap.comparer_detections([_det(0.9, 0, cls=0)], [_det(0.9, 0, cls=1)], seuil_de)
        assert not res["conforme"] and res["non_retrouvees"] == 1 and res["en_trop"] == 1

    def test_decalage_leger_apparie(self):
        res = ap.comparer_detections([_det(0.9, 0)], [_det(0.9, 0, dx=2.0)], seuil_de)
        assert res["apparies"] == 1 and res["conforme"]

    def test_sans_detection_parite_non_verifiable(self):
        res = ap.comparer_detections([], [], seuil_de)
        assert res["conforme"] and not res["verifiable"]

    def test_polygone_sans_bbox(self):
        poly = {"class_id": 0, "confidence": 0.8, "polygon": [0, 0, 10, 0, 10, 10, 0, 10]}
        paires = {"class_id": 0, "confidence": 0.8, "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        assert ap.comparer_detections([poly], [paires], seuil_de)["apparies"] == 1


class _Fausse:
    def __init__(self, nom, erreurs=0):
        self.nom, self.erreurs, self.appels = nom, erreurs, 0

    def run(self, *a, **k):
        self.appels += 1
        if self.erreurs:
            self.erreurs -= 1
            raise RuntimeError(f"{self.nom} cassée")
        return [self.nom]

    def get_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"] if self.nom != "cpu" else ["CPUExecutionProvider"]

    def get_inputs(self):
        return ["x"]


def _sonde(adaptateurs, ouvrir, inferer, t_cpu=10.0, dets_cpu=None, journal=None):
    return ap.sonder(adaptateurs, t_cpu, dets_cpu if dets_cpu is not None else [_det(0.9, 0)],
                     seuil_de, ouvrir, inferer, (journal or []).append)


class TestSonder:
    ADS = [{"id": 0, "nom": "Radeon"}, {"id": 1, "nom": "RTX"}]

    def test_retient_la_carte_conforme_la_plus_rapide(self):
        sessions = {0: (_Fausse("radeon"), "x", [1, 3, 8, 8], {}), 1: (_Fausse("rtx"), "x", [1, 3, 8, 8], {})}
        durees = {"radeon": 6.0, "rtx": 1.0}
        decision, session = _sonde(self.ADS, lambda i: sessions[i],
                                   lambda s: (durees[s[0].nom], [_det(0.9, 0)]))
        assert decision["appareil"] == ap.CARTE and decision["device_id"] == 1 and decision["carte"] == "RTX"
        assert session is sessions[1]
        assert decision["sonde"]["secondes"] == 1.0 and decision["sonde"]["secondes_processeur"] == 10.0
        assert "RTX" in decision["message"] and "processeur" in decision["message"]

    def test_carte_plus_lente_ecartee(self):
        s = (_Fausse("igpu"), "x", [1, 3, 8, 8], {})
        decision, session = _sonde(self.ADS[:1], lambda i: s, lambda _s: (9.0, [_det(0.9, 0)]))
        assert decision["appareil"] == ap.PROCESSEUR and session is None
        assert "plus rapide" in decision["motif"] and "Radeon" in decision["message"]

    def test_resultats_differents_ecartes(self):
        s = (_Fausse("gpu"), "x", [1, 3, 8, 8], {})
        decision, _ = _sonde(self.ADS[:1], lambda i: s, lambda _s: (1.0, [_det(0.9, 0), _det(0.8, 200)]))
        assert decision["appareil"] == ap.PROCESSEUR and "différents" in decision["motif"]

    def test_session_impossible_ou_retombee_sur_le_processeur(self):
        def ouvrir(i):
            if i == 0:
                raise RuntimeError("pilote absent")
            return None   # ORT a retenté en CPU : pas une carte
        decision, _ = _sonde(self.ADS, ouvrir, lambda _s: (1.0, []))
        assert decision["appareil"] == ap.PROCESSEUR
        assert "pilote absent" in decision["motif"] and "DirectML" in decision["motif"]

    def test_erreur_d_inference_ecartee(self):
        s = (_Fausse("gpu"), "x", [1, 3, 8, 8], {})

        def inferer(_s):
            raise RuntimeError("DML_ERROR")
        decision, _ = _sonde(self.ADS[:1], lambda i: s, inferer)
        assert decision["appareil"] == ap.PROCESSEUR and "DML_ERROR" in decision["motif"]

    def test_sans_carte(self):
        decision, session = _sonde([], lambda i: None, lambda _s: (1.0, []))
        assert decision["appareil"] == ap.PROCESSEUR and session is None and "aucune carte" in decision["motif"]

    def test_parite_non_verifiable_acceptee_sur_la_vitesse_mais_pas_memorisee(self):
        s = (_Fausse("gpu"), "x", [1, 3, 8, 8], {})
        decision, _ = _sonde(self.ADS[:1], lambda i: s, lambda _s: (1.0, []), dets_cpu=[])
        assert decision["appareil"] == ap.CARTE and "aucune détection" in decision["message"]
        assert decision["memorisable"] is False

    def test_parite_verifiee_memorisable(self):
        s = (_Fausse("gpu"), "x", [1, 3, 8, 8], {})
        decision, _ = _sonde(self.ADS[:1], lambda i: s, lambda _s: (1.0, [_det(0.9, 0)]))
        assert decision["memorisable"] is True
        assert "écart de confiance max 0,000" in decision["message"]   # virgule décimale

    def test_la_decision_est_serialisable(self):
        s = (_Fausse("gpu"), "x", [1, 3, 8, 8], {})
        decision, _ = _sonde(self.ADS[:1], lambda i: s, lambda _s: (1.0, [_det(0.9, 0)]))
        ligne = ap.ligne(decision)
        assert ligne.startswith("appareil=")
        assert ap.lire_ligne(ligne) == json.loads(ligne[len("appareil="):])
        assert ap.lire_ligne("progress=1/2") is None


class TestSessionAvecRepli:
    def test_bascule_une_fois_et_le_dit(self):
        dml, cpu = _Fausse("dml", erreurs=1), _Fausse("cpu")
        motifs = []
        tup = ap.avec_repli((dml, "x", [1], {}), lambda: (cpu, "x", [1], {}), motifs.append)
        session = tup[0]
        assert tup[1:] == ("x", [1], {})
        assert session.run(None, {}) == ["cpu"]
        assert session.replie and motifs == ["RuntimeError: dml cassée"]
        assert session.run(None, {}) == ["cpu"] and cpu.appels == 2
        assert session.get_providers() == ["CPUExecutionProvider"]

    def test_sans_erreur_reste_sur_la_carte(self):
        dml = _Fausse("dml")
        tup = ap.avec_repli((dml, "x", [1], {}), lambda: (_Fausse("cpu"), "x", [1], {}), lambda m: None)
        assert tup[0].run(None, {}) == ["dml"] and not tup[0].replie

    def test_une_seconde_erreur_remonte(self):
        dml, cpu = _Fausse("dml", erreurs=1), _Fausse("cpu", erreurs=1)
        tup = ap.avec_repli((dml, "x", [1], {}), lambda: (cpu, "x", [1], {}), lambda m: None)
        with pytest.raises(RuntimeError, match="cpu cassée"):
            tup[0].run(None, {})


class TestMessages:
    def test_repli_en_cours_de_traitement(self):
        d = ap.decision_repli("RuntimeError: DML", image=7)
        assert d["appareil"] == ap.PROCESSEUR and d["repli"] is True
        assert "image 7" in d["message"] and "processeur" in d["message"]

    def test_decision_memorisee(self):
        d = ap.decision_memorisee(1, "RTX")
        assert d["appareil"] == ap.CARTE and d["device_id"] == 1 and d["memorise"] is True
        assert "RTX" in d["message"]

    def test_indisponible_et_non_sondee_jamais_memorisees(self):
        d = ap.decision_indisponible("RTX", "occupée")
        assert d["appareil"] == ap.PROCESSEUR and d["memorisable"] is False and "indisponible" in d["message"]
        n = ap.decision_non_sondee("aucune inférence aboutie")
        assert n["appareil"] == ap.PROCESSEUR and n["non_sondee"] and n["memorisable"] is False
        assert "non vérifiée" in n["message"]

    def test_court(self):
        assert ap.court(RuntimeError("a\n  b   " + "x" * 300)) == ("a b " + "x" * 300)[:160]

    def test_session_sur_carte(self):
        assert ap.session_sur_carte(_Fausse("dml")) and not ap.session_sur_carte(_Fausse("cpu"))
        assert not ap.session_sur_carte(object())

    def test_providers_dml(self):
        assert ap.providers_dml(1) == [("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]
