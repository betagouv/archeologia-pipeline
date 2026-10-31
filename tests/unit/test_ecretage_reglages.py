"""Réglage hors bornes dans une configuration chargée : l'étape 2 l'écrête (setValue), désormais en le disant ;
et une erreur de bornes_errors se range sur l'étape 2 (pastille du rail), pas sur l'étape 1."""
import os

import pytest

from app.services.indices_metres import en_px
from app.services.rvt_kernel_context import bornes_errors, message_choix_inconnu, message_ecretage

WIZARD = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ui", "wizard_dialog.py")


def test_ecretage_nomme_le_reglage_la_valeur_lue_et_la_valeur_retenue():
    m = message_ecretage("LD", "angular_res", 2, 4, 32, False)
    assert "LD" in m and "Résolution angulaire" in m and "2" in m and "4 retenu" in m
    assert "150" in message_ecretage("LD", "max_radius", 150, 1, 100, False)
    assert "100 retenu" in message_ecretage("LD", "max_radius", 150, 1, 100, False)
    assert "« ve_factor »" in message_ecretage("HS", "ve_factor", 0, 1, 100, False)   # sans libellé connu : la clé


@pytest.mark.parametrize("lue", [4, 32, "10", 12.9, None, "abc"])
def test_pas_de_message_dans_les_bornes_ou_sans_nombre(lue):
    assert message_ecretage("LD", "angular_res", lue, 4, 32, False) is None


def test_choix_inconnu_nomme_le_reglage_la_valeur_lue_et_la_valeur_retenue():
    # Un choix d'une configuration chargée que l'étape 2 ne propose pas (colormap CRIM « viridis »,
    # 5 choix dans la liste) retombait en silence sur le défaut : même défaut que l'écrêtage.
    m = message_choix_inconnu("CRIM", "colormap", "viridis", ["OrRd", "Reds_r"], "OrRd")
    assert m.startswith("CRIM / « colormap »") and "viridis" in m and "OrRd retenu" in m and "Reds_r" in m
    assert message_choix_inconnu("CRIM", "colormap", "OrRd", ["OrRd", "Reds_r"], "OrRd") is None
    assert message_choix_inconnu("CRIM", "colormap", None, ["OrRd"], "OrRd") is None   # clé absente : le défaut
    assert "7" in message_choix_inconnu("OPNS", "opns_type", 7, [0, 1], 0)


def _classify_error_step():
    """Le rangement vit dans le module pur ``validation_ui`` (lot 2, 2026-10-24)."""
    from app.services.validation_ui import classer_erreurs

    return lambda message: next(iter(classer_erreurs([message])))


@pytest.mark.parametrize("produit,section,valeurs", [
    ("LD", "ldo", {"min_radius": 30, "max_radius": 20}),
    ("LD", "ldo", {"angular_res": 2}),
    ("SLRM", "slrm", {"radius_m": 40}),   # mètres (2026-10-31) : hors des bornes du réglage, 5 à 25 m
    ("MSTP", "mstp", {"lightness": 6.0}),
])
def test_erreur_de_bornes_rangee_sur_l_etape_2(produit, section, valeurs):
    classer = _classify_error_step()
    erreurs = bornes_errors({produit: True}, en_px({section: valeurs}, 0.5))
    assert erreurs and all(classer(e) == 2 for e in erreurs)
