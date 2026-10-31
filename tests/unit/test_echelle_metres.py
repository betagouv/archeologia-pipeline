"""Échelle en mètres de la détection et des contrôles (règle 2026-10-31) : fenêtre par modèle, remplissage au lieu
d'étirement, avertissements d'échelle, bornes du secours Processing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.indices_metres import en_px
from app.services.model_orchestrator import InstalledModel, avertissements_echelle, discover_installed_models
from app.services.rvt_kernel_context import bornes_errors
from app.services.indices_metres import gsd_du_modele
from pipeline.cv.model_config import sahi_a_l_echelle
from pipeline.cv.sahi_lite import REMPLISSAGE, completer_fenetre, slice_image

_MODELES = Path(__file__).resolve().parents[2] / "data" / "models"
SAHI = {"slice_height": 648, "slice_width": 648, "overlap_ratio": 0.2}


# ── B. fenêtre en mètres ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("slice_px,gsd_modele,gsd_entree,attendu", [
    (648, 1.0, 1.0, 648),     # enclos à 1 m : 648 m de terrain, comme à l'entraînement
    (648, 0.5, 1.0, 324),     # modèles à 0,5 m sur un MNT à 1 m : 324 px = 324 m
    (672, 0.5, 1.0, 336),     # ponctuelles
    (252, 0.5, 1.0, 126),     # cratères (fenêtre agrandie à 648 par le modèle : 126 m)
    (648, 1.0, 0.5, 1296),    # enclos en France à 0,5 m : 1 296 px = 648 m (au lieu de 324 m)
    (648, 0.5, 0.25, 1296),   # MNT plus fin : réduction (INTER_AREA)
    (64, 0.5, 2.0, 32),       # plancher 32 px
])
def test_fenetre_par_modele(slice_px, gsd_modele, gsd_entree, attendu):
    s = sahi_a_l_echelle({"slice_height": slice_px, "slice_width": slice_px, "overlap_ratio": 0.2},
                         gsd_modele, gsd_entree)
    assert s["slice_height"] == s["slice_width"] == attendu
    assert s["overlap_ratio"] == 0.2


@pytest.mark.parametrize("gsd_modele,gsd_entree", [(0.5, 0.5), (None, 1.0), (0.5, None)])
def test_fenetre_inchangee_a_resolution_egale_ou_inconnue(gsd_modele, gsd_entree):
    """France à 0,5 m : le dict tel quel (même clé de cache des détections qu'avant)."""
    assert sahi_a_l_echelle(SAHI, gsd_modele, gsd_entree) == SAHI


def test_gsd_du_modele_lu_dans_training_params(tmp_path):
    """Audit 7 : UNE source du GSD d'entraînement (indices_metres.gsd_de_fiche), lue par la fenêtre de détection,
    l'orchestrateur, le validateur et le script d'harmonisation."""
    import pipeline.cv.model_config as mc
    assert not hasattr(mc, "gsd_entrainement")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.onnx").write_bytes(b"")
    (tmp_path / "training_params.json").write_text(json.dumps({"mnt": {"resolution": 1.0}}), encoding="utf-8")
    assert gsd_du_modele(tmp_path / "weights" / "best.onnx") == 1.0
    (tmp_path / "training_params.json").write_text(json.dumps(
        {"mnt": {"resolution": 1.0}, "rvt": {"params": {"gsd_m": 0.5}}}), encoding="utf-8")
    assert gsd_du_modele(tmp_path) == 0.5
    (tmp_path / "training_params.json").unlink()
    assert gsd_du_modele(tmp_path) is None


# ── C. remplir au lieu d'étirer ───────────────────────────────────────────────────────────────────────
def test_fenetre_plus_petite_completee_sans_deformation():
    """186 × 300 px (cas réel : fenêtres MNT allemandes plus petites que la fenêtre) → 324 × 324, le contenu à
    l'origine intact, le reste au sans-donnée 255 : plus d'étirement anisotrope."""
    img = np.random.default_rng(0).integers(0, 254, size=(186, 300, 3), dtype=np.uint8)
    (fen,), h, w = slice_image(img, 324, 324, 0.2, 0.2)
    assert (h, w) == (186, 300) and fen.starting_pixel == [0, 0]
    assert fen.image.shape == (324, 324, 3)
    assert np.array_equal(fen.image[:186, :300], img)
    assert (fen.image[186:] == REMPLISSAGE).all() and (fen.image[:, 300:] == REMPLISSAGE).all()


def test_fenetre_pleine_rendue_telle_quelle():
    img = np.zeros((700, 700, 3), dtype=np.uint8)
    fens, _, _ = slice_image(img, 648, 648, 0.2, 0.2)
    assert all(f.image.shape == (648, 648, 3) for f in fens)
    a = np.ones((10, 10), dtype=np.uint8)
    assert completer_fenetre(a, 10, 10) is a


def test_pretraitement_inter_area_en_reduction():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from PIL import Image
    from pipeline.cv.computer_vision_onnx import _preprocess_image
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(1296, 1296, 3), dtype=np.uint8)
    t = _preprocess_image(Image.fromarray(img), (648, 648), "rfdetr")
    attendu = cv2.resize(img, (648, 648), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    assert np.allclose(t[0], ((attendu - mean) / std).transpose(2, 0, 1), atol=1e-6)
    # Agrandissement ou taille égale : INTER_LINEAR, comme avant (France à 0,5 m identique)
    petit = img[:324, :324]
    t2 = _preprocess_image(Image.fromarray(petit), (648, 648), "rfdetr")
    lin = cv2.resize(petit, (648, 648), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    assert np.allclose(t2[0], ((lin - mean) / std).transpose(2, 0, 1), atol=1e-6)


# ── D. avertissements ─────────────────────────────────────────────────────────────────────────────────
def _modele(nom, gsd, anneau, slice_px=648):
    return InstalledModel(name=nom, display_name=nom, weights_path=None, target_rvt="LD", status="", coverage={},
                          class_names=(), gsd_m=gsd, anneau_m=anneau, sahi_slice_px=slice_px)


def test_pas_d_avertissement_en_france_aux_defauts():
    rvt = en_px({}, 0.5)
    assert avertissements_echelle([_modele("crateres", 0.5, (5.0, 10.0))], 0.5, rvt) == []


def test_avertissement_de_resolution_a_1m():
    rvt = en_px({}, 1.0)
    msgs = avertissements_echelle([_modele("lineaires", 0.5, (5.0, 10.0)), _modele("enclos", 1.0, (5.0, 10.0))],
                                  1.0, rvt)
    assert len(msgs) == 1 and "lineaires" in msgs[0]
    assert "1 m/pixel" in msgs[0] and "0,5 m/pixel" in msgs[0] and "324 px au lieu de 648" in msgs[0]


def test_avertissement_anneau_different_et_indices_fournis():
    rvt = en_px({"ldo": {"min_radius_m": 10.0, "max_radius_m": 20.0}}, 0.5)
    (msg,) = avertissements_echelle([_modele("crateres", 0.5, (5.0, 10.0))], 0.5, rvt)
    assert "5 à 10 m" in msg and "10 à 20 m" in msg
    (msg,) = avertissements_echelle([_modele("crateres", 0.5, (5.0, 10.0))], 0.5, rvt, indices_calcules=False)
    assert "pas connus" in msg


@pytest.mark.skipif(not any(_MODELES.glob("*/model_card.yaml")), reason="aucun modèle installé")
def test_les_7_modeles_installes_ont_le_meme_anneau_en_metres():
    """Le faux « DIVERGENTS » : enclos (5/10 px à 1 m) et les autres (10/20 px à 0,5 m) écrivent le même anneau
    5–10 m sous des clés différentes ; l'ancienne garde comparait les clés brutes."""
    modeles = discover_installed_models(_MODELES)
    assert len(modeles) >= 7
    assert len({m.rvt_params for m in modeles}) > 1          # ce que comparait l'ancienne garde
    for m in modeles:
        assert m.anneau_m == (5.0, 10.0), (m.name, m.anneau_m)
        assert m.gsd_m in (0.5, 1.0), (m.name, m.gsd_m)
    assert {m.anneau_m for m in modeles} == {(5.0, 10.0)}     # ce que compare la nouvelle : rien à signaler
    a_1m = avertissements_echelle(modeles, 1.0, en_px({}, 1.0))
    assert len(a_1m) == sum(1 for m in modeles if m.gsd_m == 0.5)   # un message par modèle appris à 0,5 m
    assert avertissements_echelle([m for m in modeles if m.gsd_m == 0.5], 0.5, en_px({}, 0.5)) == []


# ── A. bornes du secours Processing ───────────────────────────────────────────────────────────────────
def test_secours_processing_refuse_un_rayon_converti_hors_bornes():
    """SVF 5 m à 1 m = 5 px : le moteur intégré le calcule, rvt-qgis (10-50 px) le refuserait — dit avant."""
    rvt = en_px({}, 1.0)
    assert bornes_errors({"SVF": True}, rvt) == []
    (msg,) = bornes_errors({"SVF": True}, rvt, secours_processing=True)
    assert "5 m = 5 px à 1 m/px" in msg and "10 à 50 px" in msg and "rvt-qgis" in msg
    assert bornes_errors({"SVF": True}, en_px({}, 0.5), secours_processing=True) == []   # France : rien


@pytest.mark.parametrize("gsd_modele,taille_modele", [(0.5, 648), (0.5, 672), (1.0, 648)])
def test_un_objet_garde_sa_taille_d_entrainement_dans_la_fenetre(gsd_modele, taille_modele):
    """Un disque de 10 m de rayon sur un MNT à 1 m : après fenêtre en mètres et remise à la taille du modèle, il
    fait 10 / gsd_modele pixels de rayon — sa taille dans les tuiles d'entraînement (20 px à 0,5 m, 10 px à 1 m)."""
    pytest.importorskip("cv2")
    from PIL import Image
    from pipeline.cv.computer_vision_onnx import _preprocess_image
    gsd_entree = 1.0
    yy, xx = np.mgrid[:1200, :1200]
    img = np.where((yy - 600) ** 2 + (xx - 600) ** 2 <= 10 ** 2, 0, 200).astype(np.uint8)
    img = np.repeat(img[:, :, None], 3, axis=2)
    s = sahi_a_l_echelle({"slice_height": taille_modele, "slice_width": taille_modele, "overlap_ratio": 0.2},
                         gsd_modele, gsd_entree)
    fen = next(f for f in slice_image(img, s["slice_height"], s["slice_width"], 0.2, 0.2)[0]
               if (f.image[:, :, 0] == 0).sum() == (img[:, :, 0] == 0).sum())   # fenêtre qui contient le disque
    t = _preprocess_image(Image.fromarray(fen.image), (taille_modele, taille_modele), "rfdetr")
    sombre = t[0, 0] < (100 / 255 - 0.485) / 0.229
    rayon_px = np.sqrt(sombre.sum() / np.pi)
    assert abs(rayon_px - 10 / gsd_modele) < 0.6, rayon_px
    assert s["slice_width"] * gsd_entree == pytest.approx(taille_modele * gsd_modele, abs=1)   # même terrain


def test_rayons_internes_des_melanges_tous_en_metres():
    """Audit 3 : tout attribut en PIXELS de rvt.default.DefaultValues lu par un mélange est remis à la même distance
    au sol — sim_shadow_dist (ombre de « Sky illumination », combinaison City), échelles MSTP comprises ;
    identité à 0,5 m."""
    import types
    from pipeline.ign.products.moteur_indices import rayons_en_metres
    d = types.SimpleNamespace(svf_r_max=10, slrm_rad_cell=20, ld_min_rad=10, ld_max_rad=20, ld_rad_inc=1,
                              sim_shadow_dist=100, mstp_local_scale=(1, 5, 1), mstp_meso_scale=(5, 50, 5),
                              mstp_broad_scale=(50, 500, 50))
    avant = dict(vars(d))
    rayons_en_metres(d, 0.5)
    assert vars(d) == avant
    rayons_en_metres(d, 1.0)
    assert (d.svf_r_max, d.slrm_rad_cell, d.ld_min_rad, d.ld_max_rad, d.sim_shadow_dist) == (5, 10, 5, 10, 50)
    assert d.ld_rad_inc == 1                                 # un pas d'un pixel reste un pixel
    assert d.mstp_meso_scale == (3, 25, 3) and d.mstp_broad_scale == (25, 250, 25)
