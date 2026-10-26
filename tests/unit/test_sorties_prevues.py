"""« Ce qui sera produit » (étape 4) : les chemins du livrable que le traitement écrira,
calculés depuis la configuration, sans QGIS (le nom des dossiers d'images est injecté :
``rvt_naming`` vit sous ``pipeline.ign.products``, dont l'import tire QGIS)."""
from app.services.sorties_prevues import sorties_prevues


def _nom(key, _params):
    return {"SVF": "SVF_R10_D16_V1_N0", "LD": "LD_A15_Rmin10_Rmax20_H1p7_V1"}.get(key, key)


def _config(mode="ign_laz", enabled=True):
    return {
        "app": {"files": {"data_mode": mode, "output_dir": "C:/sortie"}},
        "processing": {"products": {"MNT": True, "SVF": True, "LD": True, "HS": False, "DENSITE": False}},
        "rvt_params": {},
        "computer_vision": {
            "enabled": enabled,
            "runs": [
                {"model": "crateres", "target_rvt": "LD",
                 "entities": [{"id": "cratere", "slug": "crateres"}, {"id": "regroupement_crateres", "slug": "regroupement-de-crateres"}]},
                {"model": "lineaires", "target_rvt": "LD", "entities": [{"id": "parcellaire", "slug": "parcellaire"}]},
            ],
        },
    }


def test_ign_laz_liste_indices_et_detections():
    s = sorties_prevues(_config(), nom_dossier_indice=_nom)
    assert s[:3] == ["livrable/projet.qgs", "livrable/rapport.html", "livrable/traitement.json"]
    assert "livrable/indices/MNT/tif/" in s
    assert "livrable/indices/SVF_R10_D16_V1_N0/tif/" in s
    assert "livrable/indices/LD_A15_Rmin10_Rmax20_H1p7_V1/tif/" in s
    assert "livrable/indices/HS/tif/" not in s                       # décochée
    assert s[-3:] == ["livrable/detections/crateres/", "livrable/detections/regroupement-de-crateres/",
                      "livrable/detections/parcellaire/"]


def test_existing_rvt_sans_indices():
    s = sorties_prevues(_config(mode="existing_rvt"), nom_dossier_indice=_nom)
    assert not [x for x in s if "/indices/" in x]
    assert "livrable/detections/parcellaire/" in s


def test_detection_desactivee_sans_detections():
    s = sorties_prevues(_config(enabled=False), nom_dossier_indice=_nom)
    assert not [x for x in s if "/detections/" in x]


def test_chemins_relatifs_v3():
    for chemin in sorties_prevues(_config(), nom_dossier_indice=_nom):
        assert chemin.startswith("livrable/") and "C:" not in chemin and "\\" not in chemin
