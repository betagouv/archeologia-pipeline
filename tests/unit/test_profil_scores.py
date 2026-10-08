"""Profil des scores d'une classe : lecture des bandes de l'évaluation et agrégation (pur)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.app.services.fiabilite import Categorie
from src.app.services.profil_scores import (
    Bande,
    Profil,
    agreger,
    bilan_au_seuil,
    charger_bandes,
    charger_bandes_par_zone,
    disposer_etiquettes,
    f1max_depuis_bandes,
    libelle_zone,
    phrase_precision_rappel,
    placer_libelles_lignes,
    precision_rappel,
    profil_pour_classe,
    profils_par_zone,
    seuil_f1max,
    zones_sans_objet,
)

CATS = (
    Categorie("douteux", 0.29, 0.0, 0.26, 100),
    Categorie("possible", 0.35, 0.35, 0.45, 100),
    Categorie("probable", 0.50, 0.60, 0.72, 100),
    Categorie("quasi_certain", 0.65, 0.85, 0.92, 100),
)


def _bandes(spec):
    return [{"lo": lo, "hi": round(lo + 0.01, 2), "tp": tp, "fp": fp} for lo, tp, fp in spec]


def _modele(tmp_path: Path, nom="m1", cle_modeles=None, fiabilite=None, bloc=None):
    d = tmp_path / nom
    (d / "entrainement" / "evaluation").mkdir(parents=True)
    card = {"thresholds": {"fiabilite": fiabilite if fiabilite is not None else {"provenance": "x"}}}
    (d / "model_card.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    ev = {"modeles": {cle_modeles or nom: bloc or {}}}
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps(ev), encoding="utf-8")
    return d


def test_bandes_globales_et_agregation(tmp_path):
    bloc = {"global": {"etude_seuil": {"bandes": _bandes([(0.05, 1, 10), (0.06, 2, 10), (0.10, 3, 5), (0.29, 4, 4), (0.30, 5, 1)])}}}
    d = _modele(tmp_path, bloc=bloc)
    p = profil_pour_classe(d, "cratere", CATS)
    assert p is not None and p.n_sous_seuil == 31      # bandes fines dont hi <= 0,29 : 11 + 12 + 8
    bandes = charger_bandes(d, "cratere")
    assert len(bandes) == 5
    agg = agreger(bandes)
    assert [(b.lo, b.hi, b.tp, b.fp) for b in agg] == [
        (0.05, 0.1, 3, 20), (0.1, 0.15, 3, 5), (0.25, 0.3, 4, 4), (0.3, 0.35, 5, 1),
    ]
    assert agg[0].part_vrais == 3 / 23 and agg[0].total == 23


def test_par_classe_prime_sur_global_et_cle_modele_differente(tmp_path):
    bloc = {
        "global": {"etude_seuil": {"bandes": _bandes([(0.5, 1, 1)])}},
        "par_classe": {"four": {"etude_seuil": {"bandes": _bandes([(0.5, 9, 1)])}}},
    }
    # L'id du dossier n'est pas la clé de ``modeles`` (poids renommés) : le seul modèle évalué est pris.
    d = _modele(tmp_path, nom="ponctuelles", cle_modeles="ponctuelles_ep34", bloc=bloc)
    assert charger_bandes(d, "four")[0].tp == 9
    assert charger_bandes(d, "charbonniere")[0].tp == 1       # repli global


def test_zones_sommees_comme_le_validateur(tmp_path):
    bloc = {
        "par_classe": {"parcellaire": {"etude_seuil": {"bandes": _bandes([(0.3, 100, 100)])}}},
        "par_zone_classe": {
            "A": {"parcellaire": {"bandes": _bandes([(0.3, 10, 2)])}},
            "B": {"parcellaire": {"bandes": _bandes([(0.3, 5, 1), (0.31, 1, 0)])}},
            "C": {"parcellaire": {"bandes": _bandes([(0.3, 1000, 1000)])}},   # hors zones déclarées
        },
    }
    fiab = {"provenance": "x", "zones": {"parcellaire": ["A", "B"]}}
    d = _modele(tmp_path, nom="lin", fiabilite=fiab, bloc=bloc)
    bandes = charger_bandes(d, "parcellaire", None, ["A", "B"])
    assert [(b.lo, b.tp, b.fp) for b in bandes] == [(0.3, 15, 3), (0.31, 1, 0)]
    p = profil_pour_classe(d, "parcellaire", CATS)
    assert p is not None and p.bandes[0].tp == 16        # zones lues dans le model_card
    assert p.n_sous_seuil == 0


def test_source_declaree_dans_le_model_card(tmp_path):
    d = _modele(tmp_path, nom="tr", fiabilite={"provenance": "x", "source": "entrainement/evaluation_couverture/metriques_eval.json"})
    autre = d / "entrainement" / "evaluation_couverture"
    autre.mkdir()
    (autre / "metriques_eval.json").write_text(json.dumps({"modeles": {"tr": {"global": {"etude_seuil": {"bandes": _bandes([(0.4, 7, 3)])}}}}}), encoding="utf-8")
    p = profil_pour_classe(d, "tranchees", CATS)
    assert p is not None and p.bandes == (Bande(0.4, 0.45, 7, 3),)


def test_profil_proprietes_et_cas_degrades(tmp_path):
    p = Profil("c", (Bande(0.25, 0.3, 1, 9), Bande(0.65, 0.7, 9, 1)), CATS)
    assert p.seuil == 0.29 and p.coupures == (0.29, 0.35, 0.5, 0.65) and p.total == 20
    assert p.categorie_de(0.2) is None
    assert p.categorie_de(0.29).categorie == "douteux"
    assert p.categorie_de(0.64).categorie == "probable"
    assert p.categorie_de(0.9).categorie == "quasi_certain"
    assert profil_pour_classe(tmp_path / "absent", "c", CATS) is None
    assert profil_pour_classe(tmp_path, "c", ()) is None
    d = _modele(tmp_path, nom="vide", bloc={"global": {}})
    assert profil_pour_classe(d, "c", CATS) is None
    assert agreger([]) == []


def test_les_deux_fiches_posent_la_figure():
    """Garde-fou sans QGIS : fiche de classe et fiche ⓘ du modèle appellent ``figure_profil``."""
    racine = Path(__file__).resolve().parents[2]
    for rel in ("src/ui/dialogs/class_info_dialog.py", "src/ui/dialogs/model_info_dialog.py"):
        assert "figure_profil(" in (racine / rel).read_text(encoding="utf-8"), rel


def test_agregation_coupee_aux_seuils():
    """La barre [0,25 ; 0,30[ ne chevauche plus le seuil 0,26 : elle est scindée."""
    fines = [Bande(round(0.25 + i * 0.01, 2), round(0.26 + i * 0.01, 2), i + 1, 10) for i in range(5)]
    agg = agreger(fines, 0.05, coupures=[0.26, 0.30])
    assert [(b.lo, b.hi) for b in agg] == [(0.25, 0.26), (0.26, 0.3)]
    assert (agg[0].tp, agg[0].fp) == (1, 10) and (agg[1].tp, agg[1].fp) == (2 + 3 + 4 + 5, 40)
    # sans coupure : une seule bande de 0,05
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05)] == [(0.25, 0.3)]


def test_eclats_fusionnes_du_meme_cote_de_la_coupure():
    """0,20–0,26 d'un seul tenant sous le seuil 0,26 ; 0,29–0,35 d'un seul tenant au niveau douteux."""
    fines = [Bande(round(0.20 + i * 0.01, 2), round(0.21 + i * 0.01, 2), 1, 1) for i in range(15)]  # 0,20 → 0,35
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05, coupures=[0.26, 0.30])] == [(0.2, 0.26), (0.26, 0.3), (0.3, 0.35)]
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05, coupures=[0.29, 0.35])] == [(0.2, 0.25), (0.25, 0.29), (0.29, 0.35)]
    # les effectifs sont conservés
    assert sum(b.tp for b in agreger(fines, 0.05, coupures=[0.26, 0.30])) == 15


def test_la_figure_decline_la_couleur_comme_la_legende():
    """Garde-fou sans QGIS : même registre de couleur, même table STYLE_SPEC et même
    ``apply_confidence`` que la légende de QGIS (ui/layer_loader) — la teinte reste la
    même par classe, seules saturation et clarté changent d'un niveau à l'autre."""
    racine = Path(__file__).resolve().parents[2]
    widget = (racine / "src/ui/widgets/profil_scores.py").read_text(encoding="utf-8")
    legende = (racine / "src/ui/layer_loader.py").read_text(encoding="utf-8")
    for motif in ("color_for_class", "apply_confidence", "STYLE_SPEC"):
        assert motif in widget and motif in legende, motif
    import colorsys
    from src.pipeline.cv.color_palette import apply_confidence
    base = (148, 249, 6)
    teintes = {round(colorsys.rgb_to_hsv(*[c / 255 for c in apply_confidence(base, r)])[0] * 360)
               for r in (0.3, 0.5, 0.7, 0.9)}
    assert len(teintes) == 1, teintes      # une seule teinte pour les quatre niveaux


# ------------------------------------------------------------------ lot 1 (2026-10-08)
def test_bilan_au_seuil_et_phrase():
    """Ce qu'un seuil garde des vrais objets et écarte des fausses, sur les bandes fines."""
    fines = [Bande(0.20, 0.21, 1, 50), Bande(0.29, 0.30, 4, 9), Bande(0.30, 0.31, 20, 5), Bande(0.60, 0.61, 75, 1)]
    ref = bilan_au_seuil(fines, 0.29)          # seuil du modèle
    assert (ref.vrais_gardes, ref.fausses_gardees, ref.vrais_total, ref.fausses_total, ref.ecartees) == (99, 15, 100, 65, 51)
    assert ref.phrase() == "Au seuil 0,29, le banc garde 99 détections correctes et 15 fausses ; 51 sont écartées."
    assert ref.phrase(ref) == ref.phrase()
    # relatif au seuil du modèle, jamais en part d'un total (critère de couverture des linéaires)
    assert bilan_au_seuil(fines, 0.30).phrase(ref) == (
        "En montant le seuil à 0,3 : −4 détections correctes (−4 %) et −9 fausses (−60 %) "
        "par rapport au seuil du modèle (0,29), sur le banc."
    )
    assert bilan_au_seuil(fines, 0.20).phrase(ref) == (
        "En baissant le seuil à 0,2 : +1 détections correctes (+1 %) et +50 fausses (+333 %) "
        "par rapport au seuil du modèle (0,29), sur le banc."
    )
    # une bande à cheval compte avec son lo ; sans donnée : phrase vide
    assert bilan_au_seuil(fines, 0.295).vrais_gardes == 95
    assert bilan_au_seuil([], 0.3).phrase() == ""


def test_seuil_f1max_par_classe_puis_global_et_porte_par_le_profil(tmp_path):
    bloc = {
        "global": {"seuil_f1max": 0.37, "etude_seuil": {"bandes": _bandes([(0.3, 5, 1)])}},
        "par_classe": {"four": {"seuil_f1max": 0.41, "etude_seuil": {"bandes": _bandes([(0.3, 9, 1)])}}},
    }
    d = _modele(tmp_path, bloc=bloc)
    assert seuil_f1max(d, "four") == 0.41
    assert seuil_f1max(d, "charbonniere") == 0.37      # repli global
    assert seuil_f1max(tmp_path / "absent", "x") is None
    p = profil_pour_classe(d, "four", CATS)
    assert p is not None and p.seuil_f1max == 0.41 and p.fines == (Bande(0.3, 0.31, 9, 1),)
    assert p.zone == ""
    # l'équilibre vient de l'évaluation de RÉFÉRENCE, même quand la fiabilité est
    # mesurée ailleurs (couverture des linéaires : F1-max 0,185 < seuil déployé 0,26)
    autre = d / "entrainement" / "evaluation_couverture"
    autre.mkdir()
    (autre / "metriques_eval.json").write_text(json.dumps({"modeles": {"m1": {
        "global": {"seuil_f1max": 0.185, "etude_seuil": {"bandes": _bandes([(0.1, 7, 3)])}}}}}), encoding="utf-8")
    card = {"thresholds": {"fiabilite": {"provenance": "x", "source": "entrainement/evaluation_couverture/metriques_eval.json"}}}
    (d / "model_card.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    p2 = profil_pour_classe(d, "four", CATS)
    assert p2 is not None and p2.fines == (Bande(0.1, 0.11, 7, 3),) and p2.seuil_f1max == 0.41


def test_profils_par_zone_restreints_et_libelles(tmp_path):
    bloc = {
        "par_classe": {"parcellaire": {"etude_seuil": {"bandes": _bandes([(0.3, 100, 100)])}}},
        "par_zone_classe": {
            "grand_est/54_foret_de_haye": {"parcellaire": {"bandes": _bandes([(0.3, 10, 2)])}},
            "centre_val_de_loire/41_blois": {"parcellaire": {"bandes": _bandes([(0.5, 5, 1)])}},
            "irlande/ie_galway_01": {"parcellaire": {"bandes": _bandes([(0.3, 1000, 1000)])}},
        },
    }
    # zones déclarées : seules celles-là (même périmètre que la mesure de fiabilité)
    fiab = {"provenance": "x", "zones": {"parcellaire": ["grand_est/54_foret_de_haye", "centre_val_de_loire/41_blois"]}}
    d = _modele(tmp_path, nom="lin", fiabilite=fiab, bloc=bloc)
    assert [z for z, _b in charger_bandes_par_zone(d, "parcellaire", None, fiab["zones"]["parcellaire"])] == [
        "grand_est/54_foret_de_haye", "centre_val_de_loire/41_blois",
    ]
    profils = profils_par_zone(d, "parcellaire", CATS)
    assert [(p.zone, p.total, p.seuil_f1max) for p in profils] == [
        ("grand_est/54_foret_de_haye", 12, None), ("centre_val_de_loire/41_blois", 6, None),
    ]
    assert [p.n_gt for p in profils] == [None, None] and profils[0].critere == ""   # pas de n_gt ni de critère ici
    assert profils[0].coupures == (0.29, 0.35, 0.5, 0.65)   # les coupures de la classe, pas de la zone
    # sans restriction : toutes les zones du fichier ; une seule zone → rien
    d2 = _modele(tmp_path, nom="tout", bloc=bloc)
    assert len(profils_par_zone(d2, "parcellaire", CATS)) == 3
    une = {"par_zone_classe": {"a/1_x": {"c": {"bandes": _bandes([(0.3, 1, 1)])}}}}
    assert profils_par_zone(_modele(tmp_path, nom="une", bloc=une), "c", CATS) == []
    assert libelle_zone("grand_est/54_foret_de_haye") == "Foret de haye"
    assert libelle_zone("irlande/ie_galway_01") == "Galway 01"
    assert libelle_zone("verdun") == "Verdun"


def test_disposer_etiquettes_sans_chevauchement():
    """Première rangée où l'étiquette tient ; rangée de plus sinon ; jamais hors cadre."""
    # écartées (large), possible (étroite), probable, très probable : possible descend
    elements = [(60.0, 50.0), (110.0, 46.0), (150.0, 50.0), (260.0, 70.0)]
    assert disposer_etiquettes(elements, 300.0) == [(0, 35.0), (1, 87.0), (0, 125.0), (0, 225.0)]
    # deux étroites côte à côte → trois rangées
    assert [r for r, _g in disposer_etiquettes([(100.0, 40.0), (110.0, 40.0), (120.0, 40.0)], 300.0)] == [0, 1, 2]
    # ramenée dans le cadre : à gauche comme à droite
    assert disposer_etiquettes([(5.0, 40.0), (298.0, 40.0)], 300.0) == [(0, 0.0), (0, 260.0)]
    # rien ne se chevauche sur une même rangée, quelles que soient les largeurs
    import random
    rnd = random.Random(7)
    for _ in range(200):
        els = sorted((rnd.uniform(0, 500), rnd.uniform(20, 90)) for _ in range(6))
        dispo = disposer_etiquettes(els, 500.0)
        par_rangee: dict = {}
        for (_cx, larg), (r, g) in zip(els, dispo):
            assert 0.0 <= g and g + larg <= 500.0 + 1e-6
            for g2, l2 in par_rangee.get(r, []):
                assert g >= g2 + l2 + 4.0 - 1e-9 or g2 >= g + larg + 4.0 - 1e-9
            par_rangee.setdefault(r, []).append((g, larg))
    assert disposer_etiquettes([], 100.0) == []


def test_precision_et_rappel_au_seuil(tmp_path):
    """Précision comptée sur les bandes, rappel interpolé dans la table de l'évaluation."""
    bloc = {
        "global": {"etude_seuil": {
            "bandes": _bandes([(0.2, 10, 90), (0.3, 30, 20), (0.4, 50, 5)]),
            "tableau": [{"seuil": 0.2, "P": 0.4, "R": 0.9}, {"seuil": 0.3, "P": 0.7, "R": 0.8},
                        {"seuil": 0.4, "P": 0.9, "R": 0.5}],
        }},
    }
    d = _modele(tmp_path, bloc=bloc)
    p = profil_pour_classe(d, "c", CATS)
    assert p is not None and p.tableau == ((0.2, 0.4, 0.9), (0.3, 0.7, 0.8), (0.4, 0.9, 0.5))
    assert precision_rappel(p, 0.3) == (80 / 105, 0.8)               # bandes ≥ 0,30 : 80 vraies, 25 fausses
    prec, rap = precision_rappel(p, 0.35)
    assert prec == 50 / 55 and abs(rap - 0.65) < 1e-9                 # rappel interpolé entre 0,30 et 0,40
    assert precision_rappel(p, 0.1) == (90 / 205, 0.9) and precision_rappel(p, 0.9)[1] == 0.5
    assert precision_rappel(p, 0.99) == (None, 0.5)                   # plus rien de gardé
    assert phrase_precision_rappel(0.761, 0.8) == "précision 76 % · rappel 80 % · F1 78 %"
    assert phrase_precision_rappel(0.5, None) == "précision 50 %" and phrase_precision_rappel(None, None) == ""
    # un profil par zone n'a pas de table : rappel = vraies gardées / objets annotés au
    # critère objet, rien au critère de couverture (fragments), rien sans effectif
    zone = Profil("c", p.bandes, CATS, fines=p.fines, zone="z")
    assert precision_rappel(zone, 0.3)[1] is None
    assert precision_rappel(Profil("c", p.bandes, CATS, fines=p.fines, zone="z", n_gt=100, critere="iou"), 0.3)[1] == 0.8
    assert precision_rappel(Profil("c", p.bandes, CATS, fines=p.fines, zone="z", n_gt=100, critere="couverture"), 0.3)[1] is None
    assert precision_rappel(Profil("c", p.bandes, CATS, fines=p.fines, zone="z", n_gt=10, critere="iou"), 0.3)[1] == 1.0   # plafonné


def test_rappel_par_zone_au_critere_objet(tmp_path):
    """n_gt et critère lus dans l'évaluation ; le profil de la classe (zones sommées) les porte aussi."""
    bloc = {
        "par_classe": {"cratere": {"etude_seuil": {"bandes": _bandes([(0.3, 100, 100)])}}},
        "par_zone_classe": {
            "a/1_x": {"cratere": {"n_gt": 50, "bandes": _bandes([(0.3, 40, 10), (0.6, 5, 1)])}},
            "b/2_y": {"cratere": {"n_gt": 20, "bandes": _bandes([(0.3, 5, 5)])}},
        },
    }
    d = _modele(tmp_path, nom="cr", bloc=bloc)
    ev = json.loads((d / "entrainement" / "evaluation" / "metriques_eval.json").read_text(encoding="utf-8"))
    ev["critere"] = "iou"
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps(ev), encoding="utf-8")
    zones = profils_par_zone(d, "cratere", CATS)
    assert [(z.zone, z.n_gt, z.critere) for z in zones] == [("a/1_x", 50, "iou"), ("b/2_y", 20, "iou")]
    # une zone sans objet annoté (n_gt = 0) n'est pas un petit multiple, la fiche la nomme
    ev2 = json.loads((d / "entrainement" / "evaluation" / "metriques_eval.json").read_text(encoding="utf-8"))
    ev2["modeles"]["cr"]["par_zone_classe"]["c/3_vide"] = {"cratere": {"n_gt": 0, "bandes": _bandes([(0.1, 0, 30)])}}
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps(ev2), encoding="utf-8")
    assert [z.zone for z in profils_par_zone(d, "cratere", CATS)] == ["a/1_x", "b/2_y"]
    assert zones_sans_objet(d, "cratere") == ["c/3_vide"]
    assert precision_rappel(zones[0], 0.5) == (5 / 6, 5 / 50)      # au-dessus de 0,5 : 5 vraies, 1 fausse
    # équilibre F1 de la zone : à 0,30 F1 = 2·45/(45+11+50) = 0,85, à 0,60 F1 = 2·5/(5+1+50) = 0,18
    assert zones[0].seuil_f1max == 0.3 and zones[1].seuil_f1max == 0.3
    assert f1max_depuis_bandes([], 10) is None and f1max_depuis_bandes(zones[0].fines, None) is None
    fines = [Bande(0.2, 0.21, 1, 50), Bande(0.4, 0.41, 20, 2), Bande(0.6, 0.61, 5, 0)]
    assert f1max_depuis_bandes(fines, 30) == 0.4           # 0,2 : 52/108 ; 0,4 : 50/57 ; 0,6 : 10/35
    assert precision_rappel(zones[0], 0.3) == (45 / 56, 45 / 50)
    # profil de la classe restreint à des zones déclarées : n_gt = somme des zones
    card = {"thresholds": {"fiabilite": {"provenance": "x", "zones": {"cratere": ["a/1_x", "b/2_y"]}}}}
    (d / "model_card.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    p = profil_pour_classe(d, "cratere", CATS)
    assert p is not None and p.n_gt == 70 and p.critere == "iou" and p.tableau == ()
    assert precision_rappel(p, 0.3)[1] == 50 / 70
    # critère déduit quand le champ manque (évaluations anciennes) : « iou » si appariement IoU en tête,
    # « couverture » si c'est l'évaluation de couverture, sinon inconnu
    from src.app.services.profil_scores import critere_evaluation
    del ev["critere"]
    ev["iou"] = 0.5
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps(ev), encoding="utf-8")
    assert critere_evaluation(d) == "iou"
    autre = d / "entrainement" / "evaluation_couverture"
    autre.mkdir()
    (autre / "metriques_eval.json").write_text(json.dumps({"modeles": {}}), encoding="utf-8")
    assert critere_evaluation(d, "entrainement/evaluation_couverture/metriques_eval.json") == "couverture"
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps({"modeles": {}}), encoding="utf-8")
    assert critere_evaluation(d) == ""


def test_placer_libelles_lignes_option_a():
    """Seuil d'abord ; l'équilibre change de côté, s'abrège, puis se tait ; jamais de recouvrement."""
    seuil = (200.0, [("seuil 0,59", 50.0)], [0])
    def eq(x):
        return (x, [("équilibre (F1) 0,37", 90.0), ("F1 0,37", 35.0)], [0])
    # loin du seuil : texte entier, à droite de sa ligne
    assert placer_libelles_lignes([seuil, eq(100.0)], 1, 320.0) == [(0, 203.0, "seuil 0,59"), (0, 103.0, "équilibre (F1) 0,37")]
    # juste avant le seuil : à droite il toucherait « seuil », il passe à gauche
    assert placer_libelles_lignes([seuil, eq(180.0)], 1, 320.0)[1] == (0, 87.0, "équilibre (F1) 0,37")
    # coincé entre le bord gauche et le seuil (posé à 123) : forme courte, à droite de sa ligne
    proche = (120.0, [("seuil 0,59", 50.0)], [0])
    assert placer_libelles_lignes([proche, eq(60.0)], 1, 320.0)[1] == (0, 63.0, "F1 0,37")
    # lignes collées dans une figure étroite : il se tait, le seuil reste
    res = placer_libelles_lignes([(20.0, [("seuil 0,59", 50.0)], [0]), eq(24.0)], 1, 90.0)
    assert res[0] == (0, 23.0, "seuil 0,59") and res[1] is None
    # deux rangées : une coupure voisine descend d'une rangée
    res = placer_libelles_lignes([seuil, (210.0, [("0,35", 20.0)], [0, 1])], 2, 320.0)
    assert res[1] == (1, 213.0, "0,35")
    assert placer_libelles_lignes([], 1, 100.0) == []
    # seuil juste avant l'équilibre (0,29 / 0,37) : le seuil passe à gauche de sa ligne
    # pour laisser l'équilibre entier à droite de la sienne
    res = placer_libelles_lignes([(105.0, [("seuil 0,29", 50.0)], [0]), eq(130.0)], 1, 320.0)
    assert res == [(0, 52.0, "seuil 0,29"), (0, 133.0, "équilibre (F1) 0,37")]
    # la ligne du seuil ne se traverse jamais : l'équilibre (0,28, juste après le seuil
    # 0,26) ne se pose pas à gauche de sa ligne en travers du seuil, même sur une autre rangée
    seuil26 = (100.0, [("seuil 0,26", 50.0)], [0, 1])
    eq28 = (110.0, [("équilibre (F1) 0,28", 90.0)], [0, 1])
    res = placer_libelles_lignes([seuil26, eq28], 2, 400.0, lignes=[100.0, 110.0])
    assert res[1] is not None and not (res[1][1] - 2 < 100.0 < res[1][1] + 90.0 + 2)
    # à variante égale, la place qui ne traverse pas de ligne est préférée
    res = placer_libelles_lignes([(300.0, [("seuil 0,8", 40.0)], [0]), (100.0, [("0,35", 20.0)], [0, 1])],
                                 2, 400.0, lignes=[300.0, 100.0, 112.0])
    assert res[1] == (0, 77.0, "0,35")            # à gauche : à droite elle traversait la ligne en 112


def test_couleur_de_la_couche_qualifiee_en_comparaison():
    """A9 : la figure prend la clé du registre de la COUCHE — « classe — Modèle » en A/B."""
    from src.app.services.model_orchestrator import InstalledModel, layer_name_for_class

    m = InstalledModel(
        name="lin_v3", display_name="Modèle linéaires", weights_path=None, target_rvt="LD",
        status="production", coverage={"parcellaire": ("parcellaire",)}, class_names=("parcellaire",),
    )
    assert layer_name_for_class(m, "parcellaire", "parcellaire") == "parcellaire"
    assert layer_name_for_class(m, "parcellaire", "parcellaire", compared=True) == "parcellaire — Modèle linéaires"
    racine = Path(__file__).resolve().parents[2]
    etape3 = (racine / "src/ui/steps/step_3_detection.py").read_text(encoding="utf-8")
    assert etape3.count("layer_name_for_class(") >= 2      # carte (mini-profil) ET fiche
    assert "card.set_profils(" in etape3


def test_le_widget_suit_le_seuil_et_les_fiches_posent_les_zones():
    """Garde-fou sans QGIS : seuil mobile + bilan dans la carte, zones dans la fiche."""
    racine = Path(__file__).resolve().parents[2]
    widget = (racine / "src/ui/widgets/profil_scores.py").read_text(encoding="utf-8")
    for motif in ("def set_seuil", "categories_effectives", "def bilan", "seuil_f1max", "DashLine",
                  "QToolTip.showText", "def contextMenuEvent", "def figures_par_zone",
                  "disposer_etiquettes(", "def resizeEvent", "placer_libelles_lignes("):
        assert motif in widget, motif
    carte = (racine / "src/ui/widgets/entity_card.py").read_text(encoding="utf-8")
    assert "fig.set_seuil(seuil)" in carte and "phrase_bilan" not in carte      # bilan dans l'infobulle seulement
    assert "self.phrase_bilan()" in widget
    fiche = (racine / "src/ui/dialogs/class_info_dialog.py").read_text(encoding="utf-8")
    assert "figures_par_zone(" in fiche and "figure.set_seuil(seuil)" in fiche
    # « Tester un seuil » dans les deux fiches, précision/rappel dans le widget
    assert "ligne_essai_seuil(" in fiche
    assert "ligne_essai_seuil(" in (racine / "src/ui/dialogs/model_info_dialog.py").read_text(encoding="utf-8")
    assert "def phrase_precision_rappel" in widget and "def ligne_essai_seuil" in widget


def test_carte_entite_selection_par_le_haut_et_vignette_inerte():
    """Garde-fou sans QGIS : la vignette n'ouvre plus la fiche et le clic ne coche que
    dans le haut de la carte (vignette, titre, description, ligne du modèle)."""
    carte = (Path(__file__).resolve().parents[2] / "src/ui/widgets/entity_card.py").read_text(encoding="utf-8")
    assert "self._thumb.clicked.connect" not in carte
    assert "WA_TransparentForMouseEvents" in carte
    assert "_bas_zone_selection()" in carte


def test_fiches_distinguees_option_a():
    """Garde-fou sans QGIS : liseré + étiquette de nature dans les deux fiches ; la couleur
    d'une classe n'apparaît que dans sa fiche, la fiche du modèle reste ardoise."""
    racine = Path(__file__).resolve().parents[2]
    classe = (racine / "src/ui/dialogs/class_info_dialog.py").read_text(encoding="utf-8")
    modele = (racine / "src/ui/dialogs/model_info_dialog.py").read_text(encoding="utf-8")
    assert "STRUCTURE DÉTECTABLE" in classe and "FicheLisere" in classe and 'f"Structure · {titre}"' in classe
    assert "MODÈLE DE DÉTECTION · ONNX" in modele and "ModelInfoLisere" in modele and "Détecte :" in modele
    assert (racine / "src/ui/theme/icons/modele.svg").is_file()
