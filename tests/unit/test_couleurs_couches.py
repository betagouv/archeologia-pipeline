"""Couleurs de couche pour le rapport (lot 6, 2026-10-24) : lecture pure du registre
``class_color_registry.json`` (rang = position) et déclinaison par niveau, comme la légende."""
from app.services.couleurs_couches import (
    couleur_hex_de_couche,
    couleurs_du_bilan,
    hex_rgb,
    rang_de_couche,
    teintes_par_niveau,
)

REGISTRE = {"classes": ["Parcellaire", "cratere", "talus_fosse"]}


def _palette(rang):
    return (rang, 2 * rang, 3 * rang)


def _declinaison(base, repr_):
    return tuple(int(round(v * repr_)) for v in base)


def test_rang_et_couleur_depuis_le_registre():
    assert rang_de_couche(REGISTRE, "parcellaire") == 0          # même normalisation que le registre
    assert rang_de_couche(REGISTRE, " Cratere ") == 1
    assert rang_de_couche(REGISTRE, "inconnue") is None
    assert rang_de_couche({}, "parcellaire") is None
    assert couleur_hex_de_couche(REGISTRE, "talus_fosse", palette=_palette) == "#020406"
    assert couleur_hex_de_couche(REGISTRE, "inconnue", palette=_palette) is None
    assert hex_rgb((255, 0, 16)) == "#ff0010"


def test_teintes_par_niveau_suivent_la_legende():
    t = teintes_par_niveau((100, 100, 100), declinaison=_declinaison)
    assert set(t) == {"quasi_certain", "probable", "possible", "douteux"}
    assert t["quasi_certain"] == "#5a5a5a" and t["possible"] == "#323232" and t["douteux"] == "#1e1e1e"
    # Sans fonction injectée : quatre teintes distinctes, la plus sûre la plus sombre.
    d = teintes_par_niveau((120, 160, 200))
    assert len(set(d.values())) == 4 and all(v.startswith("#") and len(v) == 7 for v in d.values())
    assert int(d["quasi_certain"][1:3], 16) < int(d["douteux"][1:3], 16)


def test_couleurs_du_bilan_ignore_les_couches_inconnues():
    c = couleurs_du_bilan(REGISTRE, ["parcellaire", "inconnue"], palette=_palette, declinaison=_declinaison)
    assert set(c) == {"parcellaire"} and c["parcellaire"]["possible"] == "#000000"
