"""Réinitialisation aux valeurs par défaut — câblage Qt des étapes 2 et 3.

Depuis le 2026-09-16 la portée est **ciblée** : un bouton par produit à l'étape 2,
un bouton par entité à l'étape 3. Les deux boutons globaux qui remettaient tout à
zéro d'un coup sont retirés (demande utilisateur).

Le code UI exige QGIS (``qgis.PyQt``), indisponible ici : on ne peut pas cliquer.
On procède donc en deux étages, comme avant :

  1. **par AST / texte**, on vérifie que le vrai code câble bien ce contrat —
     c'est la seule garde contre un retour du bouton global, et contre un produit
     dont le bouton ne réinitialiserait rien ;
  2. **par doublures**, on rejoue la mécanique — effacer sans recréer, rester
     inerte quand il n'y a rien, et surtout **ne pas déborder** sur les voisins.

La logique de portée elle-même est pure et testée à part :
``tests/unit/test_reglages_defaut.py``.
"""
import ast
import os
import re

import pytest

from src.app.services.indices_model import all_products
from src.app.services.reglages_defaut import (
    TUILAGE,
    a_des_surcharges,
    effacer_surcharges,
    produit_de_section,
    produits_declares,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEP3 = os.path.join(ROOT, "src", "ui", "steps", "step_3_detection.py")
STEP2 = os.path.join(ROOT, "src", "ui", "steps", "step_2_indices.py")
CARD = os.path.join(ROOT, "src", "ui", "widgets", "entity_card.py")

DEFAUT_CONF, DEFAUT_AIRE = 0.25, 200.0


def _lire(p):
    return open(p, encoding="utf-8").read()


# ======================================================================
# 1. Le vrai code câble-t-il la portée ciblée ?
# ======================================================================

def _champs_declares():
    """``[(produit, clé)]`` lus dans les appels ``self._reg(...)`` de l'étape 2.

    On lit le code plutôt que de l'importer : ``step_2_indices`` tire ``qgis.PyQt``,
    absent ici. L'AST reste fidèle — c'est la déclaration réelle, pas une copie.
    """
    arbre = ast.parse(_lire(STEP2))
    out = []
    for n in ast.walk(arbre):
        if not (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_reg"):
            continue
        section = tuple(
            e.value for e in n.args[0].elts if isinstance(e, ast.Constant)
        ) if n.args and isinstance(n.args[0], ast.Tuple) else ()
        cle = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else "?"
        explicite = ""
        for kw in n.keywords:
            if kw.arg == "produit":
                if isinstance(kw.value, ast.Constant):
                    explicite = kw.value.value
                elif isinstance(kw.value, ast.Name):      # produit=TUILAGE
                    explicite = TUILAGE if kw.value.id == "TUILAGE" else kw.value.id
        out.append((explicite or produit_de_section(section), cle))
    return out


def test_chaque_parcours_du_registre_depaquette_six_elements():
    """Le descripteur a gagné la clé de produit : tout ``for`` resté à cinq noms
    lève ``ValueError`` — et seulement dans QGIS, à la première sauvegarde de
    configuration. Un dépaquetage avait effectivement été oublié le 2026-09-16.
    """
    arbre = ast.parse(_lire(STEP2))
    fautifs, vus = [], 0
    for n in ast.walk(arbre):
        if not (isinstance(n, ast.For)
                and getattr(n.iter, "attr", "") == "_adv_fields"):
            continue
        vus += 1
        noms = len(n.target.elts) if isinstance(n.target, ast.Tuple) else 1
        if noms != 6:
            fautifs.append((n.lineno, noms))
    # Anti-test-creux : si la détection ne trouve plus aucun parcours, c'est que
    # le registre a été renommé, et la garde ne garde plus rien.
    assert vus >= 3, f"seulement {vus} parcours de _adv_fields trouvés — garde inopérante"
    assert fautifs == [], (
        f"parcours de _adv_fields à {fautifs} noms au lieu de 6 "
        "(ligne, nombre) — ValueError garanti à l'usage"
    )


def test_chaque_champ_avance_est_rattache_a_un_produit():
    """Un champ sans produit serait raté par toutes les réinitialisations, et le
    garde-fou de ``_reg`` lèverait au démarrage dans QGIS."""
    orphelins = [cle for produit, cle in _champs_declares() if not produit]
    assert orphelins == [], f"champs sans produit : {orphelins}"


def test_chaque_produit_du_pipeline_a_au_moins_un_champ():
    """Sinon son onglet afficherait un bouton « ↺ Défauts » sans effet."""
    declares = set(produits_declares(_champs_declares()))
    manquants = [p.key for p in all_products() if p.key not in declares]
    assert manquants == [], f"produits sans champ avancé : {manquants}"


def test_le_tuilage_garde_une_reinitialisation():
    """Il n'a pas d'onglet : sans sa propre clé, il serait le seul réglage à
    n'avoir plus aucune réinitialisation depuis le retrait du bouton global."""
    assert TUILAGE in produits_declares(_champs_declares())


def test_etape2_pose_un_bouton_par_onglet_et_un_pour_le_tuilage():
    src = _lire(STEP2)
    assert "_mk_reset_btn(key, product(key).full_name)" in src, (
        "l'en-tête d'onglet ne pose plus de bouton de réinitialisation"
    )
    assert "_mk_reset_btn(TUILAGE" in src, "la carte Tuilage n'a pas son bouton"


def test_etape2_reinitialise_par_produit_et_non_en_bloc():
    src = _lire(STEP2)
    assert "champs_du_produit(" in src, (
        "l'étape 2 n'utilise pas la portée par produit du module pur"
    )
    assert "_reset_produit" in src
    for interdit in ("_reset_advanced", "Réinit. val. par défaut"):
        assert interdit not in src, f"étape 2 : le bouton global subsiste ({interdit})"


def test_etape3_reinitialise_par_entite_et_non_en_bloc():
    src = _lire(STEP3)
    assert "effacer_surcharges(" in src, (
        "l'étape 3 n'utilise pas l'effacement ciblé du module pur"
    )
    assert "a_des_surcharges(" in src, "l'état du bouton n'est plus calculé par entité"
    assert "card.reset_requested.connect(self._on_reset_entity)" in src, (
        "le signal de la carte n'est pas branché"
    )
    for interdit in (
        "_on_reset_defaults",
        "self._entity_thresholds.clear()",
        "self._entity_cluster_params.clear()",
        "Réinit. val. défaut du modèle",
    ):
        assert interdit not in src, f"étape 3 : le bouton global subsiste ({interdit})"


def test_la_carte_porte_son_propre_bouton():
    src = _lire(CARD)
    assert "reset_requested = pyqtSignal(str)" in src, "la carte n'a pas son signal"
    assert "_reinit_btn" in src, "la carte n'a pas son bouton"
    assert "self._reinit_btn.setEnabled(bool(reinit_possible))" in src, (
        "le bouton reste actif même sans rien à rétablir"
    )
    assert "self._adv_row.addWidget(self._reinit_btn)" in src, (
        "le bouton n'est pas sur la ligne des réglages avancés — il ne suivrait "
        "donc ni sa visibilité ni sa désactivation quand l'entité est incluse"
    )


def test_le_garde_loading_de_la_carte_survit():
    """Sans lui, chaque ``setValue`` du rafraîchissement recréerait la surcharge
    qu'on vient d'effacer."""
    assert re.search(r"if show_adv:\s*\n\s*self\._loading = True", _lire(CARD)), (
        "entity_card.py : le garde _loading autour des setValue a disparu"
    )


def test_les_deux_etapes_gardent_le_meme_langage():
    """Même préfixe « ↺ » et même confirmation par Toast, aux deux étapes."""
    for nom, src in (("step_3_detection", _lire(STEP3)), ("step_2_indices", _lire(STEP2))):
        assert "↺" in src, f"{nom} : le préfixe « ↺ » commun a disparu"
        assert "show_toast" in src, f"{nom} : pas de confirmation par Toast"


# ======================================================================
# 2. La mécanique, rejouée sur des doublures
# ======================================================================

class SpinDouble:
    """Doublure de NoWheelDoubleSpinBox : setValue déclenche le signal, comme Qt."""

    def __init__(self, on_change):
        self.value_ = 0.0
        self._on_change = on_change

    def setValue(self, v):
        self.value_ = float(v)
        self._on_change()


class CarteDouble:
    """Reproduit EntityCard : garde ``_loading`` autour des setValue, et un
    bouton dont l'état ne dépend que de SES propres surcharges."""

    def __init__(self, eid, page):
        self.eid, self.page, self._loading = eid, page, False
        self.conf = SpinDouble(self._emit)
        self.aire = SpinDouble(self._emit)
        self.reinit_actif = False

    def _emit(self):
        if not self._loading:
            self.page.on_thresholds_changed(self.eid, self.conf.value_, self.aire.value_)

    def update_state(self, conf_override, area_override, reinit_possible):
        self.reinit_actif = reinit_possible
        self._loading = True
        try:
            self.conf.setValue(conf_override if conf_override is not None else DEFAUT_CONF)
            self.aire.setValue(area_override if area_override is not None else DEFAUT_AIRE)
        finally:
            self._loading = False

    def cliquer_reinit(self):
        if self.reinit_actif:
            self.page.on_reset_entity(self.eid)


class PageDouble:
    """Reproduit Step3DetectionPage pour les seules parties concernées."""

    def __init__(self, eids):
        self.entity_thresholds = {}
        self.entity_cluster_params = {}
        self.cartes = {e: CarteDouble(e, self) for e in eids}
        self.n_changed = 0
        self.readonly = False

    def on_thresholds_changed(self, eid, conf, aire):
        self.entity_thresholds[eid] = {"confidence_threshold": conf, "min_area_m2": aire}

    def refresh(self):
        for eid, c in self.cartes.items():
            ov = self.entity_thresholds.get(eid, {})
            c.update_state(
                ov.get("confidence_threshold"), ov.get("min_area_m2"),
                reinit_possible=not self.readonly and a_des_surcharges(
                    eid, self.entity_thresholds, self.entity_cluster_params
                ),
            )

    def on_reset_entity(self, eid):
        if not effacer_surcharges(eid, self.entity_thresholds, self.entity_cluster_params):
            return
        self.refresh()
        self.n_changed += 1


@pytest.fixture
def page():
    p = PageDouble(["parcellaire", "talus", "fosse", "chemin_creux"])
    p.entity_thresholds = {
        "chemin_creux": {"confidence_threshold": 0.15, "min_area_m2": 0.0},
        "fosse": {"confidence_threshold": 0.20, "min_area_m2": 0.0},
        "parcellaire": {"confidence_threshold": 0.30, "min_area_m2": 50.0},
    }
    p.entity_cluster_params = {"fosse": {"eps_m": 60}}
    p.refresh()
    return p


def test_reinitialiser_une_entite_ne_touche_pas_les_voisines(page):
    """C'est l'invariant de la demande : on annule UN réglage, pas tous."""
    page.cartes["fosse"].cliquer_reinit()
    assert "fosse" not in page.entity_thresholds
    assert "fosse" not in page.entity_cluster_params
    assert page.entity_thresholds["chemin_creux"]["confidence_threshold"] == 0.15
    assert page.entity_thresholds["parcellaire"]["confidence_threshold"] == 0.30


def test_la_carte_reinitialisee_retombe_sur_les_defauts_du_modele(page):
    page.cartes["parcellaire"].cliquer_reinit()
    c = page.cartes["parcellaire"]
    assert (c.conf.value_, c.aire.value_) == (DEFAUT_CONF, DEFAUT_AIRE)


def test_le_rafraichissement_ne_recree_pas_la_surcharge_effacee(page):
    """Chaque ``setValue`` émet : sans le garde ``_loading``, la surcharge
    reviendrait aussitôt et le bouton n'aurait jamais l'air de marcher."""
    page.cartes["fosse"].cliquer_reinit()
    assert "fosse" not in page.entity_thresholds, (
        f"surcharge recréée par le rafraîchissement : {page.entity_thresholds}"
    )


def test_le_bouton_d_une_entite_sans_surcharge_est_inactif(page):
    assert not page.cartes["talus"].reinit_actif
    page.cartes["talus"].cliquer_reinit()
    assert page.n_changed == 0, "un clic sur un bouton inactif a agi"


def test_le_bouton_s_eteint_apres_usage(page):
    page.cartes["fosse"].cliquer_reinit()
    assert not page.cartes["fosse"].reinit_actif
    assert page.cartes["chemin_creux"].reinit_actif, (
        "les voisines ont perdu leur bouton alors qu'elles gardent leurs réglages"
    )


def test_aucun_bouton_n_est_actif_en_lecture_seule(page):
    page.readonly = True
    page.refresh()
    assert not any(c.reinit_actif for c in page.cartes.values())


def test_reinitialiser_deux_entites_de_suite_les_efface_toutes_deux(page):
    page.cartes["fosse"].cliquer_reinit()
    page.cartes["chemin_creux"].cliquer_reinit()
    assert set(page.entity_thresholds) == {"parcellaire"}
    assert page.n_changed == 2
