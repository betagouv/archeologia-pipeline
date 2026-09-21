"""Fiche d'un produit de l'étape 2 — module pur, testable hors QGIS.

La fiche est ce que l'archéologue lit avant de cocher un produit : ce que
l'image montre, à quoi elle sert, ce qu'elle ne montre pas, comment elle est
calculée, avec quels réglages et d'après quelles sources. Même exigence que
pour les fiches de classes : tout est optionnel, un bloc absent se replie sur
la description du catalogue, jamais une exception.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.app.services.indice_fiche import (
    VERDICTS,
    IndiceFiche,
    build_all_fiches,
    build_comparaison,
    build_indice_fiche,
    default_fiches_path,
    load_indices_fiches,
)
from src.app.services.indices_model import all_products

DONNEES = {
    "SVF": {
        "resume": "Part du ciel visible depuis chaque point.",
        "lire": "Creux sombres, crêtes claires.",
        "usage": ["Fossés et chemins creux", "Cratères"],
        "limites": ["Peu d'information en terrain plat"],
        "methode": ["Horizon mesuré dans 16 directions"],
        "parametres": [
            {"cle": "svf.radius", "label": "Rayon (px)", "defaut": 10,
             "sens": "Distance de recherche de l'horizon."},
            {"cle": "svf.num_directions", "label": "Directions", "defaut": "16",
             "sens": "16 est optimal."},
        ],
        "references": [
            {"citation": "Zakšek et al. 2011", "url": "https://doi.org/10.3390/rs3020398"},
        ],
        "vignettes": [
            {"image": "indices_vignettes/SVF.jpg", "legende": "Mardelles",
             "source": "Fénétrange (57)", "cadrage": {"x": 0.25, "y": 0.5, "cote": 0.25}},
        ],
    },
}


# ------------------------------------------------------------------ complet

def test_fiche_complete_remonte_tous_les_blocs():
    f = build_indice_fiche(DONNEES, "SVF")
    assert f.resume == "Part du ciel visible depuis chaque point."
    assert f.lire == "Creux sombres, crêtes claires."
    assert f.usage == ("Fossés et chemins creux", "Cratères")
    assert f.limites == ("Peu d'information en terrain plat",)
    assert f.methode == ("Horizon mesuré dans 16 directions",)
    assert f.est_complete


def test_identite_vient_du_catalogue_pas_du_json():
    """Pas de second catalogue de produits : sigle, nom technique et titre
    métier restent ceux d'``indices_model`` / ``visu_catalogue``."""
    f = build_indice_fiche(DONNEES, "SVF")
    assert f.cle == "SVF"
    assert f.tag == "SVF"
    assert f.nom == "Sky-View Factor"
    assert f.titre == "Creux & dépressions"
    assert f.famille == "RVT"


def test_parametres_typage_tolerant():
    f = build_indice_fiche(DONNEES, "SVF")
    assert [p.label for p in f.parametres] == ["Rayon (px)", "Directions"]
    # Un défaut numérique est rendu en texte : la fiche affiche, elle ne calcule pas.
    assert [p.defaut for p in f.parametres] == ["10", "16"]
    assert f.parametres[0].cle == "svf.radius"


def test_references_et_vignettes():
    f = build_indice_fiche(DONNEES, "SVF")
    assert f.references[0].citation == "Zakšek et al. 2011"
    assert f.references[0].url.endswith("rs3020398")
    v = f.vignettes[0]
    assert v.image == "indices_vignettes/SVF.jpg"
    assert v.legende == "Mardelles"
    assert v.cadrage == (0.25, 0.5, 0.25)


# ------------------------------------------------------------------ dégradé

def test_produit_sans_entree_se_replie_sur_le_catalogue():
    """Un produit pas encore documenté reste affichable : le résumé se replie
    sur la description du catalogue, et ``manques`` dit ce qui reste à écrire."""
    f = build_indice_fiche({}, "MNT")
    assert f.resume == "Altitude du sol"        # indices_model.description
    assert f.usage == ()
    assert not f.est_complete
    assert "lire" in f.manques and "usage" in f.manques


def test_donnees_mal_formees_sont_ignorees_sans_exception():
    bancal = {"SVF": {
        "resume": {"pas": "une chaîne"},
        "usage": 42,
        "parametres": [{"label": "sans clé"}, "pas un dict", {"cle": "svf.radius"}],
        "vignettes": [{"legende": "sans image"}, {"image": "ok.jpg", "cadrage": "bancal"}],
        "references": ["pas un dict", {"url": "sans citation"}],
    }}
    f = build_indice_fiche(bancal, "SVF")
    assert f.resume == "Révèle creux et dépressions"   # repli catalogue
    assert f.usage == ()
    assert [p.cle for p in f.parametres] == ["svf.radius"]   # l'entrée sans clé est écartée
    assert [v.image for v in f.vignettes] == ["ok.jpg"]      # celle sans image est écartée
    assert f.vignettes[0].cadrage is None
    assert f.references == ()                                # citation obligatoire


def test_cle_inconnue_reste_affichable():
    """Un catalogue publié plus tard ne doit pas casser une version installée."""
    f = build_indice_fiche({}, "ZZZ")
    assert isinstance(f, IndiceFiche)
    assert f.cle == "ZZZ" and f.tag == "ZZZ"


# ------------------------------------------------------------------ ensemble

def test_build_all_suit_l_ordre_du_selecteur_de_l_etape_2():
    """La liste de gauche de la fiche se lit comme les cartes qu'on vient de
    quitter : produits de base, puis indices RVT. Pas l'ordre du mur de
    l'onglet Visualisation, qui range par intérêt de consultation."""
    from src.app.services.indices_model import base_keys, rvt_keys

    cles = [f.cle for f in build_all_fiches(DONNEES)]
    assert cles == base_keys() + rvt_keys()
    assert cles[:3] == ["MNT", "DENSITE", "COUVERTURE"]
    assert cles[3] == "HS"      # premier indice RVT, comme sur la grille


def test_chargement_fichier_absent_ne_leve_pas(tmp_path):
    assert load_indices_fiches(tmp_path / "nexiste_pas.json") == {}


def test_chargement_json_invalide_ne_leve_pas(tmp_path):
    p = tmp_path / "casse.json"
    p.write_text("{ceci n'est pas du JSON", encoding="utf-8")
    assert load_indices_fiches(p) == {}


# ------------------------------------------- contrat du fichier livré

@pytest.fixture(scope="module")
def livrees():
    return load_indices_fiches()


def test_chaque_produit_du_pipeline_a_une_fiche_complete(livrees):
    """Ajouter un produit sans sa fiche laisserait une carte muette à l'étape 2."""
    incomplets = {
        f.cle: f.manques for f in build_all_fiches(livrees) if not f.est_complete
    }
    assert incomplets == {}


def test_chaque_fiche_livree_cite_ses_sources(livrees):
    sans = [f.cle for f in build_all_fiches(livrees) if not f.references]
    assert sans == []


def test_chaque_parametre_cite_un_reglage_qui_existe(livrees):
    """Un paramètre de fiche doit exister dans la config écrite par l'étape 2 —
    sinon la fiche documente un réglage fantôme."""
    from src.app.services.indices_model import rvt_keys  # noqa: F401

    sections = {
        "processing", "hs", "mdh", "svf", "opns", "slope", "ldo", "slrm", "vat",
        "mstp", "cvat", "prism", "crim",
    }
    fautifs = [
        (f.cle, p.cle)
        for f in build_all_fiches(livrees)
        for p in f.parametres
        if p.cle.split(".")[0] not in sections or "." not in p.cle
    ]
    assert fautifs == []


def test_chaque_vignette_declaree_existe_sur_disque(livrees):
    """Une vignette annoncée et absente afficherait un cadre vide en production."""
    racine = default_fiches_path().parent
    manquantes = [
        v.image
        for f in build_all_fiches(livrees)
        for v in f.vignettes
        if not (racine / v.image).is_file()
    ]
    assert manquantes == []


def test_le_fichier_livre_est_du_json_objet():
    brut = json.loads(Path(default_fiches_path()).read_text(encoding="utf-8"))
    assert isinstance(brut, dict) and brut


# ------------------------------------------- tableaux comparatifs

COMPARAISON = {
    "_comparaison": {
        "note": "Lequel prendre.",
        "legende": {"-": "inadapté", "o": "indistinct", "+": "adapté",
                    "++": "très adapté", "bidon": "ignoré"},
        "tableaux": [
            {
                "titre": "Ce que chaque produit sait faire",
                "colonnes": [
                    {"cle": "plat", "libelle": "Terrain plat", "aide": "…"},
                    {"cle": "sans_cle_ignoree"},
                    {"libelle": "colonne sans clé"},
                ],
                "cases": {
                    "SVF": {"plat": "o", "inconnue": "+"},
                    "LD": {"plat": "++"},
                    "MNT": {"plat": "verdict bidon"},
                },
                "source": "Synthèse des sources des fiches.",
            },
            {"titre": "sans colonne, écarté"},
            {"colonnes": [{"cle": "x"}]},
        ],
    },
}


def test_comparaison_lit_les_tableaux():
    c = build_comparaison(COMPARAISON)
    assert not c.est_vide
    assert [t.titre for t in c.tableaux] == ["Ce que chaque produit sait faire"]
    t = c.tableaux[0]
    assert [col.cle for col in t.colonnes] == ["plat", "sans_cle_ignoree"]
    assert t.colonnes[0].libelle == "Terrain plat"
    assert t.colonnes[1].libelle == "sans_cle_ignoree"   # repli sur la clé
    assert t.source.startswith("Synthèse")


def test_comparaison_nettoie_les_cases():
    """Une colonne non déclarée ou un verdict inventé laissent la case vide :
    mieux vaut un blanc qu'un symbole arbitraire."""
    t = build_comparaison(COMPARAISON).tableaux[0]
    assert t.verdict("SVF", "plat") == "o"
    assert t.verdict("SVF", "inconnue") == ""      # colonne non déclarée
    assert t.verdict("MNT", "plat") == ""          # verdict hors vocabulaire
    assert t.verdict("CVAT", "plat") == ""         # produit absent du tableau


def test_comparaison_legende_limitee_au_vocabulaire():
    c = build_comparaison(COMPARAISON)
    assert dict(c.legende).keys() == {"-", "o", "+", "++"}


def test_comparaison_absente_ne_leve_pas():
    for donnees in ({}, {"_comparaison": None}, {"_comparaison": {"tableaux": "bancal"}}):
        c = build_comparaison(donnees)
        assert c.est_vide and c.tableaux == ()


# ------------------------------------------- contrat du fichier livré

#: Produits sans ligne dans les tableaux transcrits, et pourquoi. Les nommer
#: ici plutôt que relâcher le test fait échouer l'ajout d'un produit qu'on
#: aurait oublié de documenter.
NON_EVALUES = {
    # La source (Kokalj 2025) évalue des visualisations de relief : ni le
    # modèle d'altitude brut, ni les deux produits de qualité de la donnée.
    "MNT", "DENSITE", "COUVERTURE",
    # L'openness seule n'a pas de ligne dans la transcription livrée (la page
    # RVT « Choosing a visualization » n'en porte pas). À remplir si une
    # édition ultérieure de la source en donne une — jamais de verdict maison.
    "OPNS",
}


def test_comparaison_livree_couvre_tous_les_produits(livrees):
    """Un produit absent d'un tableau y laisserait une ligne muette.

    Sauf les trois que la source n'évalue pas : les nommer ici plutôt que de
    relacher le test fait échouer l'ajout d'un produit qu'on aurait oublié de
    documenter, au lieu de le laisser passer en silence.
    """
    comp = build_comparaison(livrees)
    assert comp.tableaux, "aucun tableau comparatif livré"
    cles = [p.key for p in all_products()]
    assert NON_EVALUES <= set(cles), "produit disparu du pipeline"
    trous = {
        t.titre: sorted(k for k in cles if k not in (t.cases or {}))
        for t in comp.tableaux
    }
    assert trous == {t.titre: sorted(NON_EVALUES) for t in comp.tableaux}


def test_comparaison_livree_n_a_que_des_verdicts_connus():
    """Lu sur le JSON BRUT, pas à travers le modèle.

    ``_cases`` écarte déjà tout verdict inconnu : le vérifier sur son résultat
    ne pouvait donc jamais échouer. Ce qu'on veut savoir, c'est si le fichier
    LIVRÉ contient une faute de frappe — une case silencieusement vidée à
    l'affichage (tautologie signalée par l'audit 2026-09-16).
    """
    brut = json.loads(Path(default_fiches_path()).read_text(encoding="utf-8"))
    tableaux = brut.get("_comparaison", {}).get("tableaux", [])
    assert tableaux, "aucun tableau comparatif dans le fichier livré"
    fautifs = [
        (t.get("titre"), produit, col, v)
        for t in tableaux
        for produit, ligne in (t.get("cases") or {}).items()
        for col, v in ligne.items()
        if v not in VERDICTS
    ]
    assert fautifs == [], f"verdicts hors vocabulaire dans le JSON livré : {fautifs}"


def test_comparaison_livree_ne_cite_pas_de_colonne_fantome():
    """Une case rangée sous une colonne non déclarée est perdue en silence."""
    brut = json.loads(Path(default_fiches_path()).read_text(encoding="utf-8"))
    fautifs = []
    for t in brut.get("_comparaison", {}).get("tableaux", []):
        declarees = {c.get("cle") for c in t.get("colonnes", [])}
        for produit, ligne in (t.get("cases") or {}).items():
            fautifs += [(t.get("titre"), produit, col)
                        for col in ligne if col not in declarees]
    assert fautifs == [], f"colonnes non déclarées : {fautifs}"


def test_comparaison_livree_ne_laisse_pas_de_case_a_trou(livrees):
    """Une ligne présente doit être remplie sur TOUTES ses colonnes.

    Une case oubliée s'affiche comme un point, exactement comme un produit que
    la source n'évalue pas : la transcription perdrait un verdict sans que rien
    ne le signale. 154 cases ont été recopiées à la main, c'est le genre de
    faute qui se voit ici et nulle part ailleurs.
    """
    trous = [
        (t.titre, produit, col.cle)
        for t in build_comparaison(livrees).tableaux
        for produit in (t.cases or {})
        for col in t.colonnes
        if not t.verdict(produit, col.cle)
    ]
    assert trous == [], f"cases vides dans une ligne renseignée : {trous}"


def test_comparaison_livree_cite_sa_source(livrees):
    sans = [t.titre for t in build_comparaison(livrees).tableaux if not t.source]
    assert sans == []



# ------------------------------------------- citations traduites (règle utilisateur)

def test_aucune_citation_anglaise_dans_les_fiches_livrees():
    """Les fiches sont lues par des archéologues francophones : une citation
    laissée en anglais est un trou dans la lecture, pas une marque d'érudition.

    Le détecteur compte les mots-outils anglais DISTINCTS à l'intérieur d'un
    couple de guillemets : une phrase citée en atteint toujours deux, un nom
    propre laissé en VO — l'algorithme QGIS « Export to raster », la page RVT
    « Choosing a visualization », la colonne « Complexity » d'un tableau —
    n'en atteint jamais deux. Ce sont des noms, pas des citations : ils restent.
    """
    from src.app.services.indice_fiche import default_fiches_path

    brut = Path(default_fiches_path()).read_text(encoding="utf-8")
    mots_outils = re.compile(
        r"\b(the|and|is|are|of|it|that|with|for|to|in|does|not|can|be|which|by|as"
        r"|from|on|you|your|this|these|all|more|than|its|was|were|has|have|but"
        r"|because|while|such|they|their|an|a)\b",
        re.IGNORECASE,
    )
    anglaises = [
        m.group(1)
        for m in re.finditer(r"«\s*([^»]{3,}?)\s*»", brut)
        if len({w.lower() for w in mots_outils.findall(m.group(1))}) >= 2
    ]
    assert anglaises == [], (
        "citations à traduire dans data/indices_fiches.json : " + " | ".join(anglaises)
    )


# ------------------------------------------- recettes de fusion (contre rvt-qgis)

#: Nom RVT d'une visualisation -> sigle du plugin. Les deux openness ne sont pas
#: des produits de l'étape 2 : leur sigle est celui des fichiers de RVT
#: (``rvt/default.py:get_opns_file_name``), pas une invention maison.
_SIGLE_RVT = {
    "Sky-View Factor": "SVF",
    "Openness - Positive": "OPEN-POS",
    "Openness - Negative": "OPEN-NEG",
    "Slope gradient": "SLO",
    "Hillshade": "HS",
    "Multiple directions hillshade": "M_HS",
}


def _dossier_rvt_qgis():
    """Le plugin rvt-qgis voisin, ou None (poste sans QGIS, CI)."""
    plugins = Path(__file__).resolve().parents[2].parent
    for cand in sorted(plugins.glob("rvt*")):
        if (cand / "settings" / "blender_VAT.json").is_file():
            return cand
    return None


def _recette_rvt(couches):
    """Couches RVT -> [(sigle, mode, opacité)] du FOND vers la SURFACE.

    ``BlenderCombination.render_all_images`` itère ``range(len(layers)-1, -1, -1)``
    (blend.py) : la dernière couche déclarée initialise l'image, c'est le fond.
    """
    return [
        (_SIGLE_RVT[c["visualization_method"]], c["blend_mode"], int(c["opacity"]))
        for c in reversed(couches)
    ]


@pytest.mark.parametrize("produit", ["VAT", "PRISM"])
def test_resume_d_une_composition_donne_mode_et_opacite_de_chaque_couche(livrees, produit):
    """Le résumé d'un produit composé doit énoncer sa recette exacte.

    Ces deux recettes viennent de fichiers de rvt-qgis, pas de notre code : elles
    peuvent bouger à une mise à jour du plugin voisin sans que rien ne le signale.
    Et l'ordre est contre-intuitif (dernière couche déclarée = fond), ce qui a déjà
    produit une fiche qui annonçait l'empilement à l'envers.

    ``apply_terrain`` ne réécrit que les étirements min/max, jamais le mode de
    fusion ni l'opacité ni l'ordre : la recette du JSON vaut pour tous les terrains.
    """
    rvt_dir = _dossier_rvt_qgis()
    if rvt_dir is None:
        pytest.skip("plugin rvt-qgis absent de ce poste")

    if produit == "VAT":
        brut = json.loads((rvt_dir / "settings" / "blender_VAT.json").read_text("utf-8"))
        couches = brut["combination"]["layers"]
    else:
        brut = json.loads(
            (rvt_dir / "settings" / "default_blender_combinations.json").read_text("utf-8")
        )
        couches = next(
            c["combination"]["layers"] for c in brut["combinations"]
            if c["combination"]["name"] == "Prismatic openness"
        )

    resume = livrees[produit]["resume"]
    recette = _recette_rvt(couches)

    absents = [
        f"{sigle} ({mode}, {opacite} %)"
        for sigle, mode, opacite in recette
        if not re.search(
            rf"(?<![A-Z_]){re.escape(sigle)}[^(]{{0,30}}\(\s*{mode},\s*{opacite}\s*%\s*\)",
            resume,
        )
    ]
    assert absents == [], (
        f"résumé de {produit} : couches sans leur mode/opacité — {absents}\n{resume}"
    )

    positions = [
        re.search(rf"(?<![A-Z_]){re.escape(sigle)}", resume).start()
        for sigle, _, _ in recette
    ]
    assert positions == sorted(positions), (
        f"résumé de {produit} : couches citées hors de l'ordre fond -> surface "
        f"({[s for s, _, _ in recette]})"
    )
