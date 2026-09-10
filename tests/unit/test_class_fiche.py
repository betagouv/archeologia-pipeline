"""Fiche de classe — construction depuis un ``model_card.yaml`` parsé.

La fiche est ce que voit l'archéologue à l'étape 3 : à quoi ressemble la
structure, où et en quelle quantité le modèle l'a apprise, ce qu'elle n'est
pas, et dans quelle optique s'en servir. Tout est optionnel : un modèle sans
bloc ``fiche`` doit produire une fiche dégradée, jamais une exception.
"""
import pytest

from app.services.class_fiche import (
    ClassFiche,
    build_all_fiches,
    build_class_fiche,
)


CARD_COMPLET = {
    "id": "depressions_grandes_seg_ld_v1",
    "display_name": "Grandes dépressions circulaires (LD)",
    "status": "beta",
    "task": "instance_segmentation",
    "preferred_rvt": {"type": "LD", "params": {"rmin_px": 10}},
    "mnt": {"resolution": 0.5},
    "classes": [
        {
            "id": 0,
            "name": "depression_circulaire_grande",
            "label_fr": "Dépression circulaire grande",
            "description": "Grande dépression circulaire en cuvette.",
            "fiche": {
                "resume": "Mardelles, dolines et cuvettes de 14 m et plus.",
                "reconnaitre": "Cuvette sombre à contour net, souvent en grappe.",
                "usage": "Plateaux et massifs forestiers, LD 0,5 m.",
                "hors_cible": [
                    "Dépressions de moins de 14 m",
                    "Fosses d'extraction allongées",
                ],
                "vignettes": [
                    {
                        "brut": "vignettes/depression_00_brut.jpg",
                        "annote": "vignettes/depression_00_annote.jpg",
                        "zone": "Chailluz (25)",
                        "legende": "15 cuvettes sur une dalle de test",
                    },
                    {"brut": "vignettes/depression_01_brut.jpg"},
                ],
                "entrainement": {
                    "corpus": "depressions_grandes_648_v1",
                    "annotation": "masques SAM 2.1 sur boîtes revues à la main",
                    "zones": [
                        {"nom": "Fénétrange (57)", "tuiles": 620, "objets": 1834},
                        {"nom": "Chailluz (25)", "tuiles": 310, "objets": 902},
                    ],
                    "splits": {
                        "train": {"tuiles": 1285, "objets": 3577},
                        "valid": {"tuiles": 330, "objets": 948},
                        "test": {"tuiles": 219, "objets": 817},
                    },
                },
            },
        }
    ],
    "thresholds": {
        "confidence_default": 0.29,
        "confidence_per_class": {"depression_circulaire_grande": 0.29},
        "fiabilite": {
            "par_classe": {
                "depression_circulaire_grande": [
                    {"categorie": "douteux", "seuil": 0.29, "garanti": 0.0,
                     "mesure": 0.222, "n": 252},
                    {"categorie": "probable", "seuil": 0.45, "garanti": 0.6,
                     "mesure": 0.714, "n": 384},
                ]
            }
        },
    },
    "known_limitations": ["4 zones seulement", "contours issus de SAM"],
}

CARD_NU = {
    "id": "lineaires_seg_v3_1",
    "display_name": "Structures linéaires 3 classes (LD)",
    "status": "beta",
    "preferred_rvt": {"type": "LD"},
    "classes": [
        {"id": 0, "name": "parcellaire", "label_fr": "parcellaire", "description": ""},
        {"id": 1, "name": "talus_fosse", "label_fr": "", "description": "Talus ou fossé."},
    ],
    "thresholds": {"confidence_default": 0.26},
}


class TestFicheComplete:
    def test_identite_et_libelle(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert f is not None
        assert f.nom == "depression_circulaire_grande"
        assert f.label == "Dépression circulaire grande"
        assert f.modele_id == "depressions_grandes_seg_ld_v1"
        assert f.modele == "Grandes dépressions circulaires (LD)"

    def test_textes_de_presentation(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert f.resume.startswith("Mardelles")
        assert "grappe" in f.reconnaitre
        assert f.usage.startswith("Plateaux")
        assert f.hors_cible == (
            "Dépressions de moins de 14 m",
            "Fosses d'extraction allongées",
        )

    def test_vignettes_ordre_et_champs_optionnels(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert len(f.vignettes) == 2
        v0, v1 = f.vignettes
        assert v0.brut == "vignettes/depression_00_brut.jpg"
        assert v0.annote == "vignettes/depression_00_annote.jpg"
        assert v0.zone == "Chailluz (25)"
        assert v1.brut == "vignettes/depression_01_brut.jpg"
        assert v1.annote == ""  # absent → chaîne vide, jamais None

    def test_entrainement_zones_et_splits(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        e = f.entrainement
        assert e is not None
        assert e.corpus == "depressions_grandes_648_v1"
        assert "SAM" in e.annotation
        assert [z.nom for z in e.zones] == ["Fénétrange (57)", "Chailluz (25)"]
        assert e.zones[0].objets == 1834
        assert dict((s.nom, s.objets) for s in e.splits) == {
            "train": 3577, "valid": 948, "test": 817,
        }

    def test_entrainement_totaux(self):
        e = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande").entrainement
        assert e.total_tuiles == 1285 + 330 + 219
        assert e.total_objets == 3577 + 948 + 817

    def test_contexte_technique(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert f.rvt == "LD"
        assert "Local Dominance" in f.rvt_label
        assert f.resolution_m == 0.5
        assert f.seuil == 0.29
        assert f.statut == "beta"
        assert "egmentation" in f.task_label

    def test_fiabilite_reprise_du_bloc_thresholds(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert [c.categorie for c in f.fiabilite] == ["douteux", "probable"]
        assert f.fiabilite[1].mesure == pytest.approx(0.714)

    def test_limites_du_modele(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert f.limites == ("4 zones seulement", "contours issus de SAM")

    def test_est_complete(self):
        f = build_class_fiche(CARD_COMPLET, "depression_circulaire_grande")
        assert f.est_complete is True
        assert f.manques == ()


class TestFicheDegradee:
    """Un modèle sans bloc ``fiche`` doit rester présentable."""

    def test_repli_sur_description(self):
        f = build_class_fiche(CARD_NU, "talus_fosse")
        assert f.resume == "Talus ou fossé."
        assert f.usage == ""
        assert f.vignettes == ()
        assert f.entrainement is None

    def test_repli_du_libelle_sur_le_nom_technique(self):
        f = build_class_fiche(CARD_NU, "talus_fosse")
        assert f.label == "talus_fosse"

    def test_seuil_repli_sur_confidence_default(self):
        f = build_class_fiche(CARD_NU, "parcellaire")
        assert f.seuil == 0.26

    def test_incomplete_et_manques_nommes(self):
        f = build_class_fiche(CARD_NU, "parcellaire")
        assert f.est_complete is False
        assert set(f.manques) == {"resume", "usage", "vignettes", "entrainement"}

    def test_classe_inconnue_renvoie_none(self):
        assert build_class_fiche(CARD_NU, "cratere") is None

    def test_card_vide_ne_leve_pas(self):
        assert build_class_fiche({}, "parcellaire") is None
        assert build_class_fiche(None, "parcellaire") is None


class TestTolerance:
    """Le builder ne lève jamais : une donnée mal formée est ignorée."""

    def _card(self, fiche):
        return {
            "id": "m", "display_name": "M", "classes": [
                {"name": "c", "label_fr": "C", "fiche": fiche},
            ],
        }

    def test_vignette_sans_brut_ignoree(self):
        f = build_class_fiche(self._card({"vignettes": [{"annote": "x.jpg"}, {"brut": "ok.jpg"}]}), "c")
        assert [v.brut for v in f.vignettes] == ["ok.jpg"]

    def test_vignettes_non_liste_ignorees(self):
        f = build_class_fiche(self._card({"vignettes": "vignettes/x.jpg"}), "c")
        assert f.vignettes == ()

    def test_hors_cible_chaine_devient_element_unique(self):
        f = build_class_fiche(self._card({"hors_cible": "Les petites cuvettes"}), "c")
        assert f.hors_cible == ("Les petites cuvettes",)

    def test_zone_sans_nom_ignoree(self):
        card = self._card({"entrainement": {"zones": [{"tuiles": 10}, {"nom": "Blois"}]}})
        f = build_class_fiche(card, "c")
        assert [z.nom for z in f.entrainement.zones] == ["Blois"]

    def test_effectifs_non_numeriques_valent_zero(self):
        card = self._card({"entrainement": {"zones": [{"nom": "Blois", "objets": "beaucoup"}]}})
        f = build_class_fiche(card, "c")
        assert f.entrainement.zones[0].objets == 0

    def test_entrainement_vide_reste_none(self):
        f = build_class_fiche(self._card({"entrainement": {}}), "c")
        assert f.entrainement is None

    def test_fiche_non_dict_ignoree(self):
        f = build_class_fiche(self._card("du texte"), "c")
        assert isinstance(f, ClassFiche)
        assert f.resume == ""


class TestBuildAll:
    def test_une_fiche_par_classe_dans_l_ordre_du_card(self):
        fiches = build_all_fiches(CARD_NU)
        assert [f.nom for f in fiches] == ["parcellaire", "talus_fosse"]

    def test_doublon_de_classe_dedoublonne(self):
        card = dict(CARD_NU)
        card["classes"] = list(CARD_NU["classes"]) + [{"name": "parcellaire"}]
        assert [f.nom for f in build_all_fiches(card)] == ["parcellaire", "talus_fosse"]

    def test_card_vide(self):
        assert build_all_fiches({}) == ()
