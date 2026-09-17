"""La fenêtre défile vers le bas, jamais sur le côté (2026-09-17).

Deux constats utilisateur, une même cause : le contenu imposait sa taille à la
fenêtre. Les cartes de l'étape 3 exigeaient 1148 px de large pour 1071 px
disponibles (barre horizontale en bas), et l'étape 2, qui ne défilait pas,
imposait au dialogue une hauteur minimale de 711 px — hors écran en bas sur un
portable.

``src/ui/`` n'est pas collecté par pytest (pas de QGIS ici) : la garde est un
contrôle AST du câblage réel, comme ``test_reset_defauts_ui.py``.
"""
import ast
import os

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _source(rel):
    with open(os.path.join(RACINE, rel), encoding="utf-8") as f:
        return f.read()


def _appels(arbre, attribut):
    """Tous les appels ``<quelque chose>.<attribut>(...)`` de l'arbre."""
    return [
        n for n in ast.walk(arbre)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attribut
    ]


def _coupe_barre_horizontale(arbre):
    for appel in _appels(arbre, "setHorizontalScrollBarPolicy"):
        arg = ast.unparse(appel.args[0]) if appel.args else ""
        if "ScrollBarAlwaysOff" in arg:
            return True
    return False


def test_etape_3_ne_defile_jamais_horizontalement():
    arbre = ast.parse(_source("src/ui/steps/step_3_detection.py"))
    assert _coupe_barre_horizontale(arbre), (
        "la zone défilante de l'étape 3 doit interdire la barre horizontale"
    )


def test_les_pages_sans_zone_defilante_sont_enveloppees():
    """Étapes 1 et 2 : sans enveloppe, leur hauteur devient celle de la fenêtre."""
    arbre = ast.parse(_source("src/ui/wizard_dialog.py"))
    enveloppees = {
        ast.unparse(appel.args[0])
        for appel in _appels(arbre, "addWidget")
        if appel.args and "_defilante(" in ast.unparse(appel.args[0])
    }
    assert any("_source_page" in e for e in enveloppees), "étape 1 non enveloppée"
    assert any("_indices_page" in e for e in enveloppees), "étape 2 non enveloppée"
    assert _coupe_barre_horizontale(arbre), (
        "l'enveloppe des pages doit interdire la barre horizontale"
    )


def test_la_taille_d_ouverture_est_bornee_a_l_ecran():
    arbre = ast.parse(_source("src/ui/wizard_dialog.py"))
    resize = _appels(arbre, "resize")
    assert resize, "le dialogue doit fixer sa taille d'ouverture"
    assert any(
        "min(" in ast.unparse(n)
        for appel in resize
        for n in ast.walk(appel)
        if isinstance(n, ast.Call)
    ) or any(
        "availableGeometry" in ast.unparse(n) for n in ast.walk(arbre)
    ), "la taille d'ouverture doit être bornée à l'écran disponible"


def test_les_libelles_de_carte_se_replient():
    """Un libellé non repliable fait de sa largeur de texte une largeur mini."""
    src = _source("src/ui/widgets/entity_card.py")
    for attribut in ("self._label", "self._model_name"):
        assert f"{attribut}.setWordWrap(True)" in src, (
            f"{attribut} doit se replier, sinon deux colonnes de cartes ne "
            "tiennent plus dans la page"
        )
