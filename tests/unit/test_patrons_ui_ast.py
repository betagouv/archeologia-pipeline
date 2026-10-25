"""Patrons uniques de l'interface rationalisée (2026-10-24), vérifiés par AST :
un seul bouton de fiche « ⓘ », un seul geste pour cocher, les réglages dans la
carte, les « ↺ » libellés par leur infobulle, le verrou du MNT sans toast."""
import ast
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
UI = RACINE / "src" / "ui"
STEP2 = UI / "steps" / "step_2_indices.py"
STEP3 = UI / "steps" / "step_3_detection.py"
CARD = UI / "widgets" / "entity_row.py"
VIGNETTE = UI / "widgets" / "vignette.py"
DIALOG = UI / "dialogs" / "indice_info_dialog.py"
DIALOG_CLASSE = UI / "dialogs" / "class_info_dialog.py"
DIALOG_MODELE = UI / "dialogs" / "model_info_dialog.py"
WIZARD = UI / "wizard_dialog.py"
PANEL = UI / "widgets" / "fiche_panel.py"


def _fonctions(source: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _appels(node: ast.AST) -> set:
    noms = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            noms.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return noms


def test_un_seul_patron_de_fiche():
    for fichier in (STEP2, STEP3, CARD):
        src = fichier.read_text(encoding="utf-8")
        assert 'QPushButton("Fiche")' not in src, fichier.name
        assert "FicheButton(" not in src, fichier.name
    for fichier in (STEP2, CARD):
        assert "InfoButton" in fichier.read_text(encoding="utf-8"), fichier.name
    for fichier in list((UI / "steps").glob("*.py")) + list((UI / "widgets").glob("*.py")):
        assert "FicheButton" not in fichier.read_text(encoding="utf-8"), fichier


def test_mnt_verrou_sans_toast():
    fn = _fonctions(STEP2.read_text(encoding="utf-8"))["_on_product_clicked"]
    assert "show_toast" not in _appels(fn), "le verrou du MNT est une infobulle, plus un toast"


def test_etape_2_sans_vue_separee():
    src = STEP2.read_text(encoding="utf-8")
    for interdit in ("QStackedWidget", "QTabWidget", "QTabBar", "_build_advanced", "_make_param_tab"):
        assert interdit not in src, interdit
    assert "set_reglages(" in src, "les réglages se déplient dans la carte"


def test_reinit_tooltip_libellee():
    for fichier in (STEP2, CARD):
        src = fichier.read_text(encoding="utf-8")
        tooltips = re.findall(r'setToolTip\(\s*(?:f?"([^"]*)")', src)
        reinit = [t for t in tooltips if t.startswith("Revenir au")]
        assert reinit, f"{fichier.name} : aucun « ↺ » libellé « Revenir au … »"


def test_cartes_focusables_au_clavier():
    src = STEP2.read_text(encoding="utf-8")
    assert "Qt.FocusPolicy.StrongFocus" in src and "keyPressEvent" in _fonctions(src)


def test_lien_comparer_depuis_la_page():
    assert "Comparer les images ▸" in STEP2.read_text(encoding="utf-8")
    assert "__comparaison__" in DIALOG.read_text(encoding="utf-8")


# ---------------------------------------------------------------- lot 4 : panneau ⓘ, lignes

def _fonction(fichier: Path, nom: str) -> ast.AST:
    fns = _fonctions(fichier.read_text(encoding="utf-8"))
    assert nom in fns, f"{fichier.name} : {nom} introuvable"
    return fns[nom]


def test_fiches_sans_exec_depuis_les_pages():
    """Une fiche s'ouvre dans le panneau latéral, jamais en modal depuis une page."""
    for fichier in (STEP2, STEP3, CARD):
        src = fichier.read_text(encoding="utf-8")
        # Seul le menu « Changer ▾ » (QMenu) a le droit de s'ouvrir sur place.
        modaux = [
            ast.unparse(n) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "exec"
            and getattr(n.func.value, "id", "") != "menu"
        ]
        assert not modaux, f"{fichier.name} ouvre encore un dialogue modal : {modaux}"
        assert "fiche_requested" in src, fichier.name
    assert "class FichePanel" in PANEL.read_text(encoding="utf-8")
    for fichier, fabrique in ((DIALOG, "corps_fiche_indice"), (DIALOG_CLASSE, "corps_fiche_entite"),
                              (DIALOG_MODELE, "corps_fiche_modele")):
        assert f"def {fabrique}(" in fichier.read_text(encoding="utf-8"), fabrique
    assert "modele_demande" in DIALOG_CLASSE.read_text(encoding="utf-8"), "la fiche de structure demande celle du modèle"
    assert "modele_requested" in STEP3.read_text(encoding="utf-8")


def test_lecture_seule_conserve_fiches():
    """Pendant un traitement, les pages sont en lecture seule mais le panneau ⓘ reste ouvrable."""
    src = WIZARD.read_text(encoding="utf-8")
    assert "self._panel = FichePanel(" in src
    fn = _fonction(WIZARD, "_enter_review_mode")
    texte = ast.unparse(fn)
    assert "_panel" not in texte, "la lecture seule ne doit pas toucher au panneau des fiches"
    # Les boutons ⓘ des lignes de l'étape 3 ne sont pas désactivés par la lecture seule.
    ro = ast.unparse(_fonction(CARD, "set_readonly"))
    assert "_fiche_btn" not in ro and "_model_info" not in ro, "les ⓘ restent cliquables en lecture seule"


def test_etape_3_sans_filtres():
    src = STEP3.read_text(encoding="utf-8")
    for interdit in ("_add_filter_chip", "_apply_filter", "EntityChip", "QButtonGroup", "_filter"):
        assert interdit not in src, interdit
    assert "EntityRow" in src and "EntityCard" not in src


def test_seuil_deux_decimales():
    """D-07 : la case affiche deux décimales ; la valeur du modèle garde ses trois décimales
    (jamais réécrite tant qu'on ne la touche pas) et l'infobulle la montre."""
    src = CARD.read_text(encoding="utf-8")
    assert "self._conf_spin.setDecimals(2)" in src
    assert "setDecimals(3)" not in src
    assert "nombre(" in src and ", 3)" in src, "l'infobulle montre le seuil appliqué à trois décimales"


def test_trois_decimales_conservees():
    """C-01 : la page ne stocke que la clé modifiée, sans arrondir la valeur émise."""
    texte = ast.unparse(_fonction(STEP3, "_on_thresholds_changed"))
    assert "fusionner_surcharge(" in texte
    assert "round(" not in texte and ".2f" not in texte and ".3f" not in texte


def test_decocher_conserve_les_seuils():
    """C-02 : décocher une structure ne perd ni ses seuils ni ses paramètres de regroupement."""
    texte = ast.unparse(_fonction(STEP3, "collect_into"))
    assert "cv['entity_thresholds'] = dict(self._entity_thresholds)" in texte
    assert "cv['entity_cluster_params'] = dict(self._entity_cluster_params)" in texte


def test_lignes_sans_modele():
    src = CARD.read_text(encoding="utf-8")
    assert '"nomodel"' in src
    assert "elidedText(" in src, "la description est élidée, texte complet en infobulle"
    assert "thresholds_changed = pyqtSignal(str, dict)" in src


# ---------------------------------------------------------------- lot 5 : étape 4

STEP4 = UI / "steps" / "step_4_launch.py"


def test_bandeau_quatre_etats():
    src = STEP4.read_text(encoding="utf-8")
    for etat in ("incomplete", "pending", "tool_missing", "warn", "ok"):
        assert f'setProperty("state", "{etat}")' in src, etat
    assert 'setProperty("state", "error")' not in src, "l'ancien état « error » est remplacé"
    assert "manuel:depannage#" in src, "le bandeau « outil manquant » renvoie au Dépannage"
    assert "classer_erreurs(" in src, "les points à corriger sont rangés par étape"


def test_details_en_infobulle():
    """Dans l'Environnement, le détail brut (chemin, erreur) ne va qu'en infobulle ;
    la colonne porte le résumé français (resume_detail) et le nom est traduit."""
    fn = _fonction(STEP4, "_render_preflight")
    usages = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            for a in n.args:
                if isinstance(a, ast.Attribute) and a.attr == "details":
                    usages.append(getattr(n.func, "attr", None) or getattr(n.func, "id", ""))
    assert usages and set(usages) <= {"setToolTip", "resume_detail"}, usages
    texte = ast.unparse(fn)
    assert "libelle_environnement(" in texte


def test_recap_modifier_par_section():
    src = STEP4.read_text(encoding="utf-8")
    assert "step: int = 0" in src and "modifier_demande = pyqtSignal(int)" in src
    assert 'QPushButton("modifier")' in src
    wizard = WIZARD.read_text(encoding="utf-8")
    recap = ast.unparse(_fonction(WIZARD, "_update_launch_recap"))
    for n in (1, 2, 3):
        assert f"step={n}" in recap, f"section sans lien « modifier » vers l'étape {n}"
    assert "modifier_demande.connect(self._goto_step)" in wizard
    assert "reverifier_demande.connect(" in wizard
    assert "sorties_prevues(" in recap


def test_bouton_gris_dit_pourquoi():
    src = WIZARD.read_text(encoding="utf-8")
    assert "_next_hint" in src and 'setObjectName("ActionHint")' in src
    texte = ast.unparse(_fonction(WIZARD, "_update_launch_button"))
    assert "Lancement impossible" in texte and "classer_erreurs(" in texte


# ---------------------------------------------------------------- revue de branche (2026-10-25)

def test_entree_ne_coche_pas_une_carte_depuis_ses_reglages():
    """Entrée dans un champ des réglages dépliés remonte à la carte (QAbstractSpinBox,
    QComboBox et QLineEdit l'ignorent) : seule Espace coche, à l'étape 2 comme à la 3."""
    for fichier, classe in ((STEP2, "_IndexCard"), (CARD, "EntityRow")):
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
        cls = next(n for n in ast.walk(arbre) if isinstance(n, ast.ClassDef) and n.name == classe)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "keyPressEvent")
        texte = ast.unparse(fn)
        assert "Key_Space" in texte and "Key_Return" not in texte and "Key_Enter" not in texte, classe


def test_lecture_seule_rend_le_reinit_et_garde_les_fiches_de_l_etape_2():
    """Fin de traitement : la page de l'étape 3 se rafraîchit (le « ↺ » d'une structure réglée
    redevient actif) ; à l'étape 2, la lecture seule ne grise plus toute la carte (son « ⓘ » et
    « ▸ Réglages » restent) : elle passe par IndexCard.set_readonly."""
    ro3 = ast.unparse(_fonction(STEP3, "set_readonly"))
    assert "self._refresh()" in ro3
    src2 = STEP2.read_text(encoding="utf-8")
    assert "card.setEnabled(not ro)" not in src2 and "card.set_readonly(ro)" in src2
    assert "def set_readonly" in src2


def test_le_profil_suit_le_seuil_applique():
    """D-07 : la ligne orange du mini-profil suit le seuil appliqué (trois décimales), pas la
    valeur arrondie de la case."""
    texte = ast.unparse(_fonction(CARD, "_maj_profils"))
    assert "_seuil_effectif" in texte


def test_tampon_avant_journal_referme_sur_chaque_sortie():
    """Les sorties anticipées de start_run (réorganisation refusée ou incomplète) referment le
    tampon, sinon tout ce que le journal émet hors traitement s'y accumule."""
    src = (UI / "run_view.py").read_text(encoding="utf-8")
    lancer = src[src.index("def start_run("):src.index("def request_cancel(")]
    apres = lancer[lancer.index("self._tampon.actif = True"):]
    assert apres.count("return") == apres.count("self._tampon.actif = False") - 1   # + la fermeture avant le thread


def test_largeur_avant_panneau_ignoree_si_maximisee():
    texte = ast.unparse(_fonction(WIZARD, "_poser_fiche"))
    assert "if not self.isMaximized()" in texte or "isMaximized() else 0" in texte


# ---------------------------------------------------------------- lot 8 (2026-10-25)

def test_etape_3_nomme_les_couches_par_la_structure():
    src = STEP3.read_text(encoding="utf-8")
    assert src.count("label=") >= 2 and "ancien=" in src, "layer_name_for_class(label=…) et couleur_de_classe(ancien=…)"
    profil = (UI / "widgets" / "profil_scores.py").read_text(encoding="utf-8")
    assert "def couleur_de_classe(classe: str, ancien: str = \"\")" in profil


# ---------------------------------------------------------------- nom du modèle hors de la ligne (2026-10-25)

def test_la_ligne_ne_porte_plus_le_nom_du_modele():
    """Demande de l'utilisateur : ni nom du modèle ni son « ⓘ » sur la ligne d'une structure ;
    un seul ⓘ par ligne (la structure). Le modèle se lit dans « Ce qui sera analysé » (phrase
    + ⓘ), dans la fiche de la structure (lien « Fiche du modèle ») et dans le menu « Modèle ▾ »
    quand plusieurs modèles couvrent la structure."""
    src = CARD.read_text(encoding="utf-8")
    assert "_model_name" not in src and "_model_info" not in src
    assert src.count("InfoButton(") == 1 and 'QPushButton("Modèle ▾")' in src
    page = STEP3.read_text(encoding="utf-8")
    assert "row.modele_requested" not in page and "self.modele_requested.emit(m)" in page


def test_reglages_sous_chaque_ligne_comme_a_l_etape_2():
    """Demande de l'utilisateur (2026-10-25) : plus de case globale « Seuils et réglages par
    structure » ; chaque ligne cochée porte son lien « ▸ Réglages » (même patron #CardToggle
    que les images de l'étape 2) qui déplie ses seuils, son profil et ses paramètres de
    regroupement ; la page recharge les profils d'une ligne quand elle se déplie."""
    page = STEP3.read_text(encoding="utf-8")
    assert "Seuils et réglages par structure" not in page and "_adv_check" not in page
    assert "row.reglages_toggled.connect(" in page and "row.est_deplie()" in page
    src = CARD.read_text(encoding="utf-8")
    assert 'setObjectName("CardToggle")' in src and "reglages_toggled = pyqtSignal(str, bool)" in src
    assert "def est_deplie(self)" in src and "def _basculer_reglages(self)" in src
