"""Réinitialisation aux valeurs par défaut, PAR produit et PAR entité.

Un bouton global remettait tout à zéro d'un coup, aux deux étapes. C'est trop
brutal : on règle finement un indice, on veut annuler ce réglage-là, et on perd
les onze autres au passage (demande utilisateur 2026-09-16). La portée d'une
réinitialisation est donc désormais un produit, ou une entité.

Ce module est la logique pure de cette portée ; le câblage Qt vit dans
``ui/steps/`` et ``ui/widgets/``, hors pytest, et
``tests/unit/test_reset_defauts_ui.py`` vérifie par AST qu'il l'utilise bien.
"""
from __future__ import annotations

import pytest

from src.app.services.indices_model import rvt_keys
from src.app.services.reglages_defaut import (
    SECTION_PRODUIT,
    TUILAGE,
    a_des_surcharges,
    champs_du_produit,
    effacer_surcharges,
    phrase_entite_reinitialisee,
    phrase_produit_reinitialise,
    produit_de_section,
    produits_declares,
)

# (produit, section, clé, défaut) — extrait réaliste de ce que l'étape 2 enregistre.
CHAMPS = [
    ("MNT", ("processing",), "filter_expression", "Classification = 2"),
    ("MNT", ("processing",), "mnt_resolution", 0.5),
    ("DENSITE", ("processing",), "density_resolution", 1.0),
    ("COUVERTURE", ("processing",), "coverage_threshold_percent", 30),
    ("SVF", ("rvt_params", "svf"), "radius", 10),
    ("SVF", ("rvt_params", "svf"), "num_directions", 16),
    ("LD", ("rvt_params", "ldo"), "max_radius", 20),
    ("_TUILAGE", ("processing",), "tile_overlap", 20),
]


# ------------------------------------------------------- portée par produit

def test_champs_du_produit_ne_rend_que_les_siens():
    svf = champs_du_produit(CHAMPS, "SVF")
    assert [c[2] for c in svf] == ["radius", "num_directions"]


def test_champs_du_produit_distingue_des_produits_de_meme_section():
    """MNT, Densité et Couverture écrivent tous dans ``processing`` : c'est le
    produit déclaré qui les sépare, jamais la section."""
    assert [c[2] for c in champs_du_produit(CHAMPS, "MNT")] == [
        "filter_expression", "mnt_resolution",
    ]
    assert [c[2] for c in champs_du_produit(CHAMPS, "DENSITE")] == ["density_resolution"]
    assert [c[2] for c in champs_du_produit(CHAMPS, "COUVERTURE")] == [
        "coverage_threshold_percent",
    ]


def test_champs_du_produit_inconnu_est_vide():
    assert champs_du_produit(CHAMPS, "ZZZ") == ()
    assert champs_du_produit(CHAMPS, "") == ()


def test_champs_du_produit_preserve_l_ordre_de_declaration():
    """L'ordre des champs est celui de l'onglet : le message de confirmation et
    les valeurs réécrites suivent l'ordre que l'utilisateur voit."""
    assert champs_du_produit(CHAMPS, "SVF") == tuple(
        c for c in CHAMPS if c[0] == "SVF"
    )


def test_produits_declares_dedoublonne_en_gardant_l_ordre():
    assert produits_declares(CHAMPS) == (
        "MNT", "DENSITE", "COUVERTURE", "SVF", "LD", "_TUILAGE",
    )


# ------------------------------------------------------- portée par entité

def test_effacer_surcharges_ne_touche_que_l_entite_visee():
    seuils = {"cratere": {"confidence_threshold": 0.4}, "four": {"confidence_threshold": 0.3}}
    cluster = {"cratere": {"eps_m": 60}}
    n = effacer_surcharges("cratere", seuils, cluster)
    assert n == 2
    assert seuils == {"four": {"confidence_threshold": 0.3}}
    assert cluster == {}


def test_effacer_surcharges_compte_les_dictionnaires_reellement_touches():
    """Le compte sert au message de confirmation : il doit dire ce qui a bougé,
    pas le nombre de dictionnaires examinés."""
    seuils = {"four": {"confidence_threshold": 0.3}}
    cluster = {}
    assert effacer_surcharges("four", seuils, cluster) == 1
    assert seuils == {} and cluster == {}


def test_effacer_surcharges_sans_rien_a_effacer_est_inerte():
    seuils = {"four": {}}
    cluster = {"four": {}}
    avant = (dict(seuils), dict(cluster))
    assert effacer_surcharges("cratere", seuils, cluster) == 0
    assert (seuils, cluster) == avant


def test_effacer_surcharges_refuse_une_cle_vide():
    """Une entité sans identifiant viderait silencieusement une entrée ``''``."""
    seuils = {"": {"confidence_threshold": 0.4}}
    assert effacer_surcharges("", seuils) == 0
    assert seuils == {"": {"confidence_threshold": 0.4}}


# ------------------------------------------------------- confirmation

@pytest.mark.parametrize("n,attendu", [
    (0, ""),
    (1, "1 réglage"),
    (3, "3 réglages"),
])
def test_phrase_produit_compte_les_reglages(n, attendu):
    phrase = phrase_produit_reinitialise("Sky-View Factor", n)
    if not attendu:
        assert phrase == ""          # rien à dire quand rien n'a bougé
    else:
        assert attendu in phrase and "Sky-View Factor" in phrase


def test_phrase_produit_nomme_le_produit_pas_le_lot():
    """Le message doit dire CE QUI a été remis à zéro : c'est ce qui distingue
    une réinitialisation ciblée d'un retour global."""
    assert "Local Dominance" in phrase_produit_reinitialise("Local Dominance", 2)


# ------------------------------------------- section → produit

def test_section_rvt_donne_le_produit():
    assert produit_de_section(("rvt_params", "svf")) == "SVF"
    assert produit_de_section(("rvt_params", "ldo")) == "LD"     # « ldo », pas « ld »
    assert produit_de_section(("rvt_params", "mdh")) == "M_HS"   # « mdh », pas « m_hs »
    assert produit_de_section(("rvt_params", "slope")) == "SLO"


def test_section_processing_ne_determine_rien():
    """MNT, Densité, Couverture et le tuilage partagent ``processing`` : s'y fier
    les réinitialiserait ensemble, ce qui est exactement le défaut corrigé."""
    assert produit_de_section(("processing",)) == ""


def test_section_inconnue_ou_vide_rend_une_chaine_vide():
    for section in ((), ("rvt_params",), ("rvt_params", "inconnue"), None):
        assert produit_de_section(section) == ""


def test_chaque_indice_rvt_du_pipeline_a_sa_section():
    """Un indice sans section verrait son bouton « ↺ Défauts » ne rien faire."""
    manquants = [k for k in rvt_keys() if k not in set(SECTION_PRODUIT.values())]
    assert manquants == []


def test_aucune_section_ne_pointe_vers_un_produit_fantome():
    connus = set(rvt_keys())
    fantomes = {s: p for s, p in SECTION_PRODUIT.items() if p not in connus}
    assert fantomes == {}


def test_le_tuilage_n_est_pas_un_produit_du_pipeline():
    """C'est un pseudo-produit : il vaut pour tous les indices et n'a pas d'onglet."""
    assert TUILAGE not in rvt_keys()
    assert TUILAGE.startswith("_")


# ------------------------------------------- bouton actif ou non

def test_a_des_surcharges_dit_si_le_bouton_sert():
    seuils = {"cratere": {"confidence_threshold": 0.4}}
    cluster = {"four": {"eps_m": 60}}
    assert a_des_surcharges("cratere", seuils, cluster)
    assert a_des_surcharges("four", seuils, cluster)
    assert not a_des_surcharges("enclos", seuils, cluster)
    assert not a_des_surcharges("", seuils, cluster)


def test_a_des_surcharges_ignore_un_argument_qui_n_est_pas_un_dict():
    assert not a_des_surcharges("cratere", None, "bancal", 42)


def test_phrase_entite_vide_quand_rien_n_a_bouge():
    assert phrase_entite_reinitialisee("Cratères", 0) == ""
    assert "Cratères" in phrase_entite_reinitialisee("Cratères", 1)

