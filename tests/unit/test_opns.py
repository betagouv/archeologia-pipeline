"""Produit OPNS — openness, un seul réglage de type (positive / négative).

Le produit n'apporte pas de calcul : il appelle ``rvt:rvt_opns``, comme le SVF
appelle ``rvt:rvt_svf``. Ce qui doit être tenu ici, c'est ce que le plugin
décide *avant* l'appel :

1. les **bornes dures** de l'algorithme (rayon 10–50 px, directions 8–64,
   bruit 0–3) — hors de là, Processing refuse le paramètre et la dalle échoue ;
2. l'**invariant du nom** — le dossier de sortie et l'appel doivent lire les
   mêmes valeurs, sinon un rayon de 200 px donnerait un dossier ``_R200``
   contenant une image calculée à 50 ;
3. le **type dans le nom** — positive et négative sont deux images sans
   rapport, elles ne doivent jamais partager un dossier.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.ign.products.rvt_naming import (
    HORIZON_DIRECTIONS_RANGE,
    HORIZON_NOISE_RANGE,
    HORIZON_RADIUS_RANGE,
    get_rvt_folder_name,
    get_rvt_temp_filename,
    opns_settings,
    svf_settings,
)

TILE = "LHD_FXX_0624_6864"

#: Bornes lues dans ``rvt-qgis/processing_provider/rvt_opns.py`` (minValue /
#: maxValue des QgsProcessingParameterNumber). Les recopier ici fait échouer le
#: test si quelqu'un élargit les nôtres sans vérifier celles de RVT.
BORNES_RVT = {
    "radius": (10, 50),
    "num_directions": (8, 64),
    "noise_remove": (0, 3),
}


def test_nos_bornes_sont_celles_de_rvt():
    assert HORIZON_RADIUS_RANGE == BORNES_RVT["radius"]
    assert HORIZON_DIRECTIONS_RANGE == BORNES_RVT["num_directions"]
    assert HORIZON_NOISE_RANGE == BORNES_RVT["noise_remove"]


def test_defauts_sans_config():
    """Dict vide → les défauts de RVT, pas des zéros."""
    assert opns_settings({}) == {
        "opns_type": 0,
        "radius": 10,
        "num_directions": 16,
        "noise_remove": 0,
        "ve_factor": 1,
    }
    assert opns_settings({"opns": None}) == opns_settings({})
    assert opns_settings({"opns": "bancal"}) == opns_settings({})


@pytest.mark.parametrize("cle,trop_bas,trop_haut", [
    ("radius", 2, 200),
    ("num_directions", 3, 400),
    ("noise_remove", -5, 9),
])
def test_valeurs_hors_bornes_ramenees(cle, trop_bas, trop_haut):
    lo, hi = BORNES_RVT[cle]
    assert opns_settings({"opns": {cle: trop_bas}})[cle] == lo
    assert opns_settings({"opns": {cle: trop_haut}})[cle] == hi


def test_virgule_francaise_acceptee():
    """Même tolérance que les autres indices : « 12,9 » saisi à la main."""
    assert opns_settings({"opns": {"radius": "12,9"}})["radius"] == 12


@pytest.mark.parametrize("valeur,attendu", [
    (0, 0), (1, 1), ("1", 1), (2, 0), (-1, 0), (None, 0), ("bancal", 0),
])
def test_type_est_binaire(valeur, attendu):
    """Un seul réglage de type, comme rvt-qgis : tout ce qui n'est pas 1 = positive."""
    assert opns_settings({"opns": {"opns_type": valeur}})["opns_type"] == attendu


def test_le_type_separe_les_dossiers():
    """Positive et négative sont deux images sans rapport : jamais le même dossier."""
    pos = get_rvt_folder_name("OPNS", {"opns": {"opns_type": 0}})
    neg = get_rvt_folder_name("OPNS", {"opns": {"opns_type": 1}})
    assert pos == "OPNS_Pos_R10_D16_V1_N0"
    assert neg == "OPNS_Neg_R10_D16_V1_N0"
    assert pos != neg


def test_chaque_reglage_separe_les_dossiers():
    """Deux réglages = deux dossiers, sinon un run écrase le précédent."""
    base = {"opns": {"opns_type": 0, "radius": 10, "num_directions": 16,
                     "ve_factor": 1, "noise_remove": 0}}
    noms = {get_rvt_folder_name("OPNS", base)}
    for cle, autre in [("opns_type", 1), ("radius", 30), ("num_directions", 32),
                       ("ve_factor", 2), ("noise_remove", 3)]:
        variante = {"opns": {**base["opns"], cle: autre}}
        nom = get_rvt_folder_name("OPNS", variante)
        assert nom not in noms, f"{cle} n'apparaît pas dans le nom de dossier"
        noms.add(nom)


def test_le_nom_de_dossier_ne_ment_pas_sur_les_bornes():
    """Un rayon hors bornes est ramené AU CALCUL : le nom doit l'être aussi.

    C'est l'invariant de ``get_rvt_param_suffix`` (cf. CLAUDE.md, « même
    rvt_params des deux côtés ») appliqué aux bornes : sans lui, le dossier
    ``OPNS_Pos_R200_…`` contiendrait une image calculée à 50 px.
    """
    params = {"opns": {"radius": 200, "num_directions": 400}}
    assert opns_settings(params)["radius"] == 50
    assert get_rvt_folder_name("OPNS", params) == "OPNS_Pos_R50_D64_V1_N0"
    assert "R200" not in get_rvt_temp_filename("OPNS", TILE, params)


def test_nom_de_fichier_temporaire():
    assert get_rvt_temp_filename("OPNS", TILE, {}) == (
        f"{TILE}_OPNS_Pos_R10_D16_V1_N0.tif"
    )


# --------------------------------------------------------------- le branchement

_INDICES = Path(__file__).resolve().parents[2] / "src" / "pipeline" / "ign" / "products" / "indices.py"


def _branche_opns() -> str:
    """Le bloc ``if products.get("OPNS")`` d'indices.py, en source.

    ``indices.py`` importe QGIS au chargement : il n'est pas importable en
    standalone (cf. CLAUDE.md, « deux contextes d'exécution »). On le lit donc,
    comme ``test_reset_defauts_ui`` lit l'étape 2.
    """
    src = _INDICES.read_text(encoding="utf-8")
    debut = src.index('if products.get("OPNS"')
    fin = src.index('if products.get("SLO"', debut)
    return src[debut:fin]


def test_la_branche_appelle_le_bon_algorithme():
    assert 'run_qgis_algorithm("rvt:rvt_opns"' in _branche_opns()


def test_la_branche_lit_les_reglages_par_la_source_unique():
    """Relire ``rvt_params["opns"]`` à la main ici décorrélerait nom et calcul."""
    branche = _branche_opns()
    assert "opns_settings(rvt_params)" in branche
    for cle in ("radius", "num_directions", "noise_remove", "opns_type"):
        assert f'opns["{cle}"]' in branche, cle


def test_la_branche_passe_tous_les_parametres_de_l_algorithme():
    """Un paramètre oublié prendrait le défaut de RVT, pas celui de l'étape 2."""
    branche = _branche_opns()
    attendus = {"INPUT", "OUTPUT", "NOISE_REMOVE", "NUM_DIRECTIONS", "OPNS_TYPE",
                "RADIUS", "SAVE_AS_8BIT", "VE_FACTOR"}
    # [A-Z0-9_] et non [A-Z_] : SAVE_AS_8BIT porte un chiffre.
    passes = set(re.findall(r'"([A-Z0-9_]+)":', branche))
    assert attendus <= passes, f"paramètres absents : {sorted(attendus - passes)}"


# --------------------------------------------- le SVF partage ces mêmes bornes

def _branche(produit: str, suivant: str) -> str:
    src = _INDICES.read_text(encoding="utf-8")
    debut = src.index(f'if products.get("{produit}"')
    return src[debut:src.index(f'if products.get("{suivant}"', debut)]


def test_svf_a_les_memes_defauts():
    assert svf_settings({}) == {
        "radius": 10, "num_directions": 16, "noise_remove": 0, "ve_factor": 1,
    }


@pytest.mark.parametrize("cle,trop_bas,trop_haut", [
    ("radius", 2, 200),
    ("num_directions", 3, 400),
    ("noise_remove", -5, 9),
])
def test_svf_hors_bornes_ramene(cle, trop_bas, trop_haut):
    """``rvt:rvt_svf`` déclare exactement les mêmes bornes que ``rvt:rvt_opns``.

    Jusqu'au 2026-09-21, l'étape 2 laissait saisir un rayon SVF de 100 000 px et
    le pipeline le passait tel quel : Processing refusait le paramètre et la
    dalle échouait. Le SVF est désormais borné comme l'openness.
    """
    lo, hi = BORNES_RVT[cle]
    assert svf_settings({"svf": {cle: trop_bas}})[cle] == lo
    assert svf_settings({"svf": {cle: trop_haut}})[cle] == hi


def test_le_nom_de_dossier_svf_ne_ment_pas_sur_les_bornes():
    params = {"svf": {"radius": 200, "num_directions": 400}}
    assert get_rvt_folder_name("SVF", params) == "SVF_R50_D64_V1_N0"


def test_le_nom_de_dossier_svf_par_defaut_est_inchange():
    """Le cadrage des bornes ne doit RIEN changer aux runs déjà faits."""
    assert get_rvt_folder_name("SVF", {}) == "SVF_R10_D16_V1_N0"
    assert get_rvt_folder_name("SVF", {"svf": {"radius": 20}}) == "SVF_R20_D16_V1_N0"


def test_la_branche_svf_lit_les_reglages_par_la_source_unique():
    branche = _branche("SVF", "OPNS")
    assert "svf_settings(rvt_params)" in branche
    for cle in ("radius", "num_directions", "noise_remove"):
        assert f'svf["{cle}"]' in branche, cle


@pytest.mark.parametrize("section,produit,spins", [
    ("svf", "SVF", [("svf_noise", 0, 3), ("svf_dirs", 8, 64), ("svf_radius", 10, 50)]),
    ("opns", "OPNS", [("opns_noise", 0, 3), ("opns_dirs", 8, 64), ("opns_radius", 10, 50)]),
])
def test_les_champs_de_l_etape_2_sont_bornes_comme_rvt(section, produit, spins):
    """Un champ plus large que l'algorithme laisse saisir une valeur qui échouera.

    Lu en source : ``src/ui/`` n'est pas collecté par pytest (pas de QGIS en
    standalone), donc c'est le seul endroit où cette régression se verrait
    ailleurs qu'au runtime dans QGIS.
    """
    etape2 = (Path(__file__).resolve().parents[2] / "src" / "ui" / "steps"
              / "step_2_indices.py").read_text(encoding="utf-8")
    for nom, lo, hi in spins:
        attendu = f"{nom} = self._mk_spin({lo}, {hi}, "
        assert attendu in etape2, f"{produit} : {nom} n'est pas borné ({attendu!r})"
