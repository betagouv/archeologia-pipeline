"""Aide intégrée : module pur ``app.services.aide`` et contrats sur ``aide/*.md`` livrés.

Le manuel est versionné avec le plugin : ces contrats sont ce qui empêche qu'il
retarde à nouveau sur l'interface (un produit ou une entité ajoutés sans leur
mention, un lien ou une image cassés, une version écrite à la main).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.app.services.aide import (
    CHAPITRE_PAR_ETAPE,
    CLE_NOUVEAUTES,
    Chapitre,
    charger_chapitres,
    chapitre,
    cle_du_fichier,
    images,
    liens_internes,
    nouveautes_markdown,
    rechercher,
    resoudre_cible,
    sections,
    slug,
    texte_brut,
)
from src.app.services.citations import citations_anglaises
from src.app.services.indices_model import all_products

RACINE = Path(__file__).resolve().parents[2]
AIDE = RACINE / "aide"


# ------------------------------------------------------------------ unitaires

def test_slug_sans_accents_ni_ponctuation():
    assert slug("Réglages avancés… (étape 2)") == "reglages-avances-etape-2"
    assert slug("  Ce que ça ne montre pas  ") == "ce-que-ca-ne-montre-pas"


def test_sections_ignore_les_blocs_de_code():
    md = "# Titre\n\n```\n# pas un titre\n```\n\n## Sous-titre ##\n### Trois\n"
    assert sections(md) == [
        (1, "Titre", "titre"),
        (2, "Sous-titre", "sous-titre"),
        (3, "Trois", "trois"),
    ]


def test_cle_du_fichier_retire_le_prefixe_numerique():
    assert cle_du_fichier(Path("03-etape-1-source.md")) == "etape-1-source"
    assert cle_du_fichier(Path("glossaire.md")) == "glossaire"


def test_charger_chapitres_ordre_titre_et_dossier_absent(tmp_path):
    (tmp_path / "02-b.md").write_text("# Deuxième\n\ntexte", encoding="utf-8")
    (tmp_path / "01-a.md").write_text("sans titre de niveau 1", encoding="utf-8")
    ch = charger_chapitres(tmp_path)
    assert [c.cle for c in ch] == ["a", "b"]
    assert ch[1].titre == "Deuxième"
    assert ch[0].titre == "01-a"            # repli : nom du fichier
    assert charger_chapitres(tmp_path / "absent") == []


def test_nouveautes_depuis_le_changelog(tmp_path):
    meta = tmp_path / "metadata.txt"
    meta.write_text(
        "[general]\nname=X\nversion=0.14.0\nchangelog=0.14.0 (2026-10-08)\n"
        "    * Un manuel intégré.\n    * La grille à jour.\n"
        " 0.13.1 (2026-09-22)\n    * Correctif.\n",
        encoding="utf-8",
    )
    md = nouveautes_markdown(meta)
    assert md.startswith("# Nouveautés")
    assert "## 0.14.0 (2026-10-08)" in md and "## 0.13.1 (2026-09-22)" in md
    assert "- Un manuel intégré." in md and "- Correctif." in md
    assert md.index("0.14.0") < md.index("0.13.1")
    ch = charger_chapitres(tmp_path / "vide", metadata_path=meta)
    assert [c.cle for c in ch] == [CLE_NOUVEAUTES]
    assert nouveautes_markdown(tmp_path / "absent.txt") == ""


def test_nouveautes_echappe_le_texte_brut_du_changelog(tmp_path):
    """« archeologia.<version>.zip » faisait une balise HTML (puces vides ensuite),
    « max_workers » une italique : le changelog est du texte, pas du Markdown."""
    from src.app.services.aide import echapper_markdown

    meta = tmp_path / "metadata.txt"
    meta.write_text(
        "[general]\nchangelog=0.7.1 (2026-06-18)\n"
        "    * nom du ZIP conforme (archeologia.<version>.zip) ; max_workers=2 [x]\n",
        encoding="utf-8",
    )
    md = nouveautes_markdown(meta)
    assert r"archeologia.\<version\>.zip" in md
    assert r"max\_workers=2 \[x\]" in md
    assert echapper_markdown("a*b_c<d>e#f|g") == r"a\*b\_c\<d\>e\#f\|g"
    assert echapper_markdown("sans ponctuation, 0.5 m (ok)") == "sans ponctuation, 0.5 m (ok)"


def test_liens_internes_et_images():
    md = ("[a](etape-2-produits.md#reglages) [b](#ici) [c](glossaire.md) "
          "[ext](https://x.y/z) ![img](img/capture.png) [m](mailto:a@b.c) "
          "[![agrandir](img/grande.png)](img/grande.png)")
    assert liens_internes(md) == [("etape-2-produits", "reglages"), ("", "ici"), ("glossaire", "")]
    assert images(md) == ["img/capture.png", "img/grande.png"]
    assert resoudre_cible("x.md#y") == ("x", "y")


def test_chapitre_a_l_ancre():
    c = Chapitre("x", "X", "# X\n\n## Réglages avancés\n")
    assert c.a_l_ancre("reglages-avances")
    assert not c.a_l_ancre("absent")
    assert chapitre([c], "x") is c and chapitre([c], "y") is None


# ------------------------------------------------------------------ contrats sur aide/ livré

@pytest.fixture(scope="module")
def chapitres():
    ch = charger_chapitres(AIDE, metadata_path=RACINE / "metadata.txt")
    assert ch, "aide/ est vide : le manuel intégré n'a rien à montrer"
    return ch


def test_chaque_etape_a_son_chapitre(chapitres):
    cles = {c.cle for c in chapitres}
    manquants = [v for v in CHAPITRE_PAR_ETAPE.values() if v not in cles]
    assert manquants == [], f"chapitres absents d'aide/ : {manquants}"
    assert CLE_NOUVEAUTES in cles


def test_chaque_chapitre_commence_par_un_titre(chapitres):
    for c in chapitres:
        assert c.markdown.lstrip().startswith("# "), f"{c.cle} : pas de titre « # » en tête"


def test_liens_internes_resolvent(chapitres):
    casses = []
    for c in chapitres:
        for cle, ancre in liens_internes(c.markdown):
            cible = chapitre(chapitres, cle) if cle else c
            if cible is None or (ancre and not cible.a_l_ancre(ancre)):
                casses.append(f"{c.cle} → {cle or '(même chapitre)'}#{ancre}")
    assert casses == [], "liens internes cassés : " + " | ".join(casses)


def test_images_existent(chapitres):
    absentes = [
        f"{c.cle} : {img}" for c in chapitres for img in images(c.markdown)
        if not (AIDE / img).is_file()
    ]
    assert absentes == [], "images absentes d'aide/ : " + " | ".join(absentes)


def test_aucune_version_ecrite_a_la_main(chapitres):
    """La version du manuel est celle du plugin par construction : aucun chapitre
    n'écrit « v0.x » ; seules les Nouveautés, rendues depuis metadata.txt, en portent."""
    fautifs = [
        c.cle for c in chapitres
        if c.cle != CLE_NOUVEAUTES and re.search(r"\b0\.\d+\.\d+\b", c.markdown)
    ]
    assert fautifs == [], f"numéro de version écrit à la main dans : {fautifs}"


def test_aucune_citation_anglaise(chapitres):
    fautes = [f"{c.cle} : {q}" for c in chapitres for q in citations_anglaises(c.markdown)]
    assert fautes == [], "citations à traduire : " + " | ".join(fautes)


def test_etape_2_nomme_chaque_produit(chapitres):
    """Un produit ajouté au pipeline sans sa ligne dans le manuel échoue ici."""
    md = chapitre(chapitres, CHAPITRE_PAR_ETAPE[2]).markdown
    oublies = [p.tag for p in all_products() if not re.search(rf"\b{re.escape(p.tag)}\b", md)]
    assert oublies == [], f"produits absents du chapitre de l'étape 2 : {oublies}"


def test_etape_3_nomme_chaque_entite(chapitres):
    md = chapitre(chapitres, CHAPITRE_PAR_ETAPE[3]).markdown.lower()
    catalogue = json.loads((RACINE / "data" / "entities_catalog.json").read_text(encoding="utf-8"))
    entites = catalogue["entities"] if isinstance(catalogue, dict) else catalogue
    oubliees = [e["label"] for e in entites if e["label"].lower() not in md]
    assert oubliees == [], f"entités absentes du chapitre de l'étape 3 : {oubliees}"


def test_rubriques_de_depannage_existent(chapitres):
    """Chaque renvoi du journal (« voir Dépannage › … ») pointe un titre réel."""
    from src.app.services.aide import CLE_DEPANNAGE, _RUBRIQUES_DEPANNAGE, rubrique_depannage

    dep = chapitre(chapitres, CLE_DEPANNAGE)
    titres = {t for _n, t, _s in dep.sections}
    manquants = [titre for _m, titre in _RUBRIQUES_DEPANNAGE if titre not in titres]
    assert manquants == [], f"rubriques absentes du chapitre Dépannage : {manquants}"
    assert rubrique_depannage("LD : le noyau atteint 20 px mais la marge…") == "Le noyau atteint N pixels"
    assert rubrique_depannage("Connection to data.geopf.fr timed out") == "Le téléchargement échoue"
    assert rubrique_depannage("Dalle 3/12 : OK") == ""


def test_ui_branche_le_manuel():
    """Garde-fou sans QGIS : les deux points d'entrée appellent bien ``ouvrir_aide``,
    en demandant les Nouveautés non lues (A2) ; le journal est un navigateur de texte
    dont les lignes ⚠/✗ portent un lien « manuel:depannage#… » (A1) ; la fenêtre
    a l'historique, le zoom et la recherche dans tout le manuel (A3, A10)."""
    for rel in ("src/ui/wizard_dialog.py", "main.py"):
        src = (RACINE / rel).read_text(encoding="utf-8")
        assert "ouvrir_aide" in src and "nouveautes_si_non_lues=True" in src, rel
    run_view = (RACINE / "src/ui/run_view.py").read_text(encoding="utf-8")
    assert "QTextBrowser()" in run_view and "QPlainTextEdit()" not in run_view
    assert 'href="manuel:' in run_view and "anchorClicked.connect" in run_view
    dlg = (RACINE / "src/ui/dialogs/aide_dialog.py").read_text(encoding="utf-8")
    for motif in ("rechercher(", "_remplir_resultats", "_historique", "StandardKey.Back",
                  "zoom_demande", "StandardKey.ZoomIn", "nouveautes_non_lues", "marquer_nouveautes_lues"):
        assert motif in dlg, motif


def test_rechercher_dans_tout_le_manuel():
    """Casse et accents ignorés, forme exacte restituée, section et extrait, code ignoré."""
    a = Chapitre("a", "Chapitre A", "# Chapitre A\n\nIntro sur la **fiabilité**.\n\n## Réglages\n\n- la fiabilité ici\n\n```\nfiabilite dans du code\n```\n")
    b = Chapitre("b", "Chapitre B", "# Chapitre B\n\nRien.\n")
    res = rechercher([a, b], "fiabilite")
    assert [(r.cle, r.ancre, r.section, r.motif) for r in res] == [
        ("a", "", "", "fiabilité"), ("a", "reglages", "Réglages", "fiabilité"),
    ]
    assert res[0].extrait == "Intro sur la fiabilité." and res[1].extrait == "la fiabilité ici"
    assert rechercher([a, b], "RÉGLAGES")[0].motif == "Réglages"
    assert rechercher([a, b], "  ") == [] and rechercher([a, b], "absent") == []
    assert texte_brut("- **Seuil** : voir [la fiche](x.md#y) et `code` | a | b |") == "Seuil : voir la fiche et code · a · b"
    assert texte_brut("### Titre ![img](img/x.png)") == "Titre"
