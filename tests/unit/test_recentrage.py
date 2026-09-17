"""Recentrage des enclos tranchés par une couture SAHI (2026-09-17).

Un enclos à cheval sur une couture n'est entier dans aucune des deux fenêtres :
chacune n'en voit qu'un morceau, coupé à plat sur le bord. On redemande alors
l'objet au modèle sur une fenêtre centrée sur lui.

Ces tests verrouillent les deux moitiés du contrat :
  - on rejoue, et on remplace par la seconde version, un enclos posé sur une
    couture (le symptôme) ;
  - on ne rejoue RIEN d'autre : ni loin d'une couture, ni sur un bord d'image,
    ni sur une autre classe, ni sur un objet trop grand pour la fenêtre
    (la non-régression cratères / linéaires).
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv import recentrage  # noqa: E402
from pipeline.cv.recentrage import (  # noqa: E402
    apparier,
    coutures_depuis_slices,
    fenetre_centree,
    recentrer_les_coupes,
    sur_une_couture,
    tient_dans_la_fenetre,
    vers_l_image,
)

FENETRE = 648
IMAGE = 2000
COUTURES = ([519, 648, 1038, 1167, 1352], [519, 648, 1038, 1167, 1352])
NOMS = ["enclos", "fosse"]


class _ImageBouchon:
    """Suffit au module : il n'appelle que ``crop``."""

    def __init__(self):
        self.crops = []

    def crop(self, boite):
        self.crops.append(tuple(boite))
        return tuple(boite)


def _inference(reponses, journal=None):
    def _inferer(_crop):
        if journal is not None:
            journal.append(_crop)
        return list(reponses)
    return _inferer


def _enclos(bbox, class_id=0, confidence=0.4):
    return {"class_id": class_id, "confidence": confidence, "bbox": list(bbox)}


# --- briques pures -----------------------------------------------------------

def test_coutures_exclut_les_bords_de_l_image():
    # Deux fenêtres 648 au pas de 519 sur une image de 1167 px.
    bboxes = [(0, 0, 648, 648), (519, 0, 1167, 648)]
    xs, ys = coutures_depuis_slices(bboxes, largeur=1167, hauteur=648)
    assert xs == [519, 648]   # 0 et 1167 sont des bords d'image, pas des coutures
    assert ys == []           # idem verticalement : 0 et 648


def test_sur_une_couture_ne_compte_que_les_cotes_poses_dessus():
    assert sur_une_couture([400, 300, 519, 420], *COUTURES)          # xmax sur 519
    assert sur_une_couture([400, 515, 460, 600], *COUTURES)          # ymin à 4 px
    assert not sur_une_couture([400, 300, 500, 420], *COUTURES)      # 19 px : vrai bord


def test_tient_dans_la_fenetre_ecarte_les_objets_plus_grands():
    assert tient_dans_la_fenetre([0, 0, 640, 640], FENETRE, FENETRE)
    assert not tient_dans_la_fenetre([0, 0, 700, 100], FENETRE, FENETRE)


def test_fenetre_centree_garde_sa_taille_et_reste_dans_l_image():
    x1, y1, x2, y2 = fenetre_centree([900, 900, 1000, 1000], FENETRE, FENETRE, IMAGE, IMAGE)
    assert (x2 - x1, y2 - y1) == (FENETRE, FENETRE)
    assert (x1 + x2) // 2 == 950 and (y1 + y2) // 2 == 950

    # Contre un bord : la fenêtre glisse au lieu de rétrécir (l'échelle du
    # modèle ne doit pas bouger d'une détection à l'autre).
    bord = fenetre_centree([0, 0, 40, 40], FENETRE, FENETRE, IMAGE, IMAGE)
    assert bord == (0, 0, FENETRE, FENETRE)


def test_vers_l_image_recale_bbox_et_polygone_normalise():
    det = {"class_id": 0, "bbox": [10, 20, 50, 60],
           "polygon": [0.0, 0.0, 1.0, 1.0], "area": 1234.0}
    sortie = vers_l_image(det, (100, 200, 748, 848), largeur=IMAGE, hauteur=IMAGE)
    assert sortie["bbox"] == [110.0, 220.0, 150.0, 260.0]
    # (0·648 + 100)/2000, (0·648 + 200)/2000, (1·648 + 100)/2000, (1·648 + 200)/2000
    assert sortie["polygon"] == pytest.approx([0.05, 0.10, 0.374, 0.424])
    assert sortie["area"] == 1234.0  # pixels², le crop n'est pas redimensionné


def test_apparier_prend_le_meilleur_recouvrement_de_la_meme_classe():
    origine = _enclos([400, 300, 519, 420])
    loin = _enclos([1000, 1000, 1100, 1100])
    autre_classe = _enclos([400, 300, 640, 420], class_id=1)
    entier = _enclos([400, 300, 640, 420])
    assert apparier(origine, [loin, autre_classe, entier]) is entier
    assert apparier(origine, [loin, autre_classe]) is None


# --- le contrat complet ------------------------------------------------------

def test_remplace_un_enclos_pose_sur_une_couture_par_sa_version_entiere():
    image = _ImageBouchon()
    coupe = _enclos([400, 300, 519, 420])
    # Dans le repère du crop (fenêtre (136, 36, 784, 684)), l'objet entier
    # déborde à droite de là où la couture l'avait tranché.
    entier = {"class_id": 0, "confidence": 0.81, "bbox": [264, 264, 500, 384]}
    journal = []

    sortie = recentrer_les_coupes(
        [coupe], image=image, largeur=IMAGE, hauteur=IMAGE, coutures=COUTURES,
        fenetre_px=(FENETRE, FENETRE), inferer=_inference([entier], journal),
        noms_de_classes=NOMS,
    )

    assert image.crops == [(136, 36, 784, 684)]
    assert len(journal) == 1
    assert len(sortie) == 1
    # La SECONDE version, et elle seule : coordonnées image, pas celles du crop.
    assert sortie[0]["confidence"] == 0.81
    assert sortie[0]["bbox"] == [400.0, 300.0, 636.0, 420.0]


def test_supprime_la_coupe_que_la_seconde_passe_ne_confirme_pas():
    sortie = recentrer_les_coupes(
        [_enclos([400, 300, 519, 420])], image=_ImageBouchon(), largeur=IMAGE,
        hauteur=IMAGE, coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
        inferer=_inference([]), noms_de_classes=NOMS,
    )
    assert sortie == []


def test_supprime_aussi_quand_la_seconde_passe_detecte_ailleurs():
    """Une détection au même endroit, pas n'importe où dans la fenêtre."""
    ailleurs = {"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 60, 60]}
    sortie = recentrer_les_coupes(
        [_enclos([400, 300, 519, 420])], image=_ImageBouchon(), largeur=IMAGE,
        hauteur=IMAGE, coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
        inferer=_inference([ailleurs]), noms_de_classes=NOMS,
    )
    assert sortie == []


@pytest.mark.parametrize("bbox, pourquoi", [
    ([700, 700, 800, 800], "loin de toute couture"),
    ([0, 700, 100, 800], "posé sur un bord d'image, pas sur une couture"),
    ([100, 100, 900, 460], "plus large que la fenêtre : rien à recentrer"),
])
def test_ne_rejoue_pas_ce_qui_n_est_pas_tranche_par_une_couture(bbox, pourquoi):
    image = _ImageBouchon()
    entree = [_enclos(bbox)]
    journal = []
    sortie = recentrer_les_coupes(
        entree, image=image, largeur=IMAGE, hauteur=IMAGE, coutures=COUTURES,
        fenetre_px=(FENETRE, FENETRE), inferer=_inference([], journal),
        noms_de_classes=NOMS,
    )
    assert sortie == entree, pourquoi
    assert journal == [], pourquoi
    assert image.crops == []


def test_ne_touche_pas_aux_autres_classes():
    """« Seulement pour les enclos » : une fosse sur une couture passe intacte."""
    fosse = _enclos([400, 300, 519, 420], class_id=1)
    journal = []
    sortie = recentrer_les_coupes(
        [fosse], image=_ImageBouchon(), largeur=IMAGE, hauteur=IMAGE,
        coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
        inferer=_inference([], journal), noms_de_classes=NOMS,
    )
    assert sortie == [fosse]
    assert journal == []


def test_sans_vocabulaire_du_modele_on_ne_touche_a_rien():
    entree = [_enclos([400, 300, 519, 420])]
    sortie = recentrer_les_coupes(
        entree, image=_ImageBouchon(), largeur=IMAGE, hauteur=IMAGE,
        coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
        inferer=_inference([]), noms_de_classes=None,
    )
    assert sortie is entree


def test_une_seconde_passe_en_echec_ne_supprime_rien():
    def _explose(_crop):
        raise RuntimeError("onnxruntime a lâché")

    entree = [_enclos([400, 300, 519, 420])]
    sortie = recentrer_les_coupes(
        entree, image=_ImageBouchon(), largeur=IMAGE, hauteur=IMAGE,
        coutures=COUTURES, fenetre_px=(FENETRE, FENETRE), inferer=_explose,
        noms_de_classes=NOMS,
    )
    assert sortie == entree


def test_l_annulation_utilisateur_traverse_le_recentrage():
    """Une annulation n'est pas une panne d'inférence : elle ne doit pas être mangée."""
    from pipeline.cancellation import PipelineCancelled

    def _annule(_crop):
        raise PipelineCancelled()

    with pytest.raises(PipelineCancelled):
        recentrer_les_coupes(
            [_enclos([400, 300, 519, 420])], image=_ImageBouchon(), largeur=IMAGE,
            hauteur=IMAGE, coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
            inferer=_annule, noms_de_classes=NOMS,
        )


def test_l_interrupteur_rend_l_etape_inerte(monkeypatch):
    monkeypatch.setattr(recentrage, "RECENTRAGE_ACTIF", False)
    entree = [_enclos([400, 300, 519, 420])]
    journal = []
    sortie = recentrer_les_coupes(
        entree, image=_ImageBouchon(), largeur=IMAGE, hauteur=IMAGE,
        coutures=COUTURES, fenetre_px=(FENETRE, FENETRE),
        inferer=_inference([], journal), noms_de_classes=NOMS,
    )
    assert sortie is entree
    assert journal == []
