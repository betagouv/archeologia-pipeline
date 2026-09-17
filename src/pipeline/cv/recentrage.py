"""Recentrage des enclos tranchés par une fenêtre SAHI (2026-09-17).

Un enclos plus large que le recouvrement SAHI (fenêtre 648 px, recouvrement
20 % = 129 px, soit 65 m à 0,5 m/px) peut tomber à cheval sur une couture sans
tenir entièrement dans aucune des deux fenêtres. Chacune n'en voit qu'un
morceau : le masque sort coupé à plat sur le bord, et le score — calculé sur
une structure tronquée — s'effondre.

On ne recoud pas le morceau manquant : on **redemande** l'objet au modèle sur
une fenêtre CENTRÉE sur lui, où il est entier et loin de tout bord. La réponse
de cette seconde passe remplace la première. Si le modèle ne redit rien à cet
endroit, le morceau initial n'était pas un objet : il est supprimé.

Restreint par ``CLASSES_RECENTRAGE``, aujourd'hui aux **enclos** : objets
compacts et peu nombreux, donc seconde passe utile et bon marché. La seule
condition technique est de tenir dans une fenêtre (``tient_dans_la_fenetre``).
Mesuré le 2026-09-17 sur ``lineaires_seg_v3_1`` : 97 % des détections de
parcellaire tranchées par une couture tiennent dans les 324 m d'une fenêtre
(plus grande dimension médiane 100 m), et les ouvrir à la règle divise les
coupures par 4 à 45 — au prix de ×1,7 à ×3,6 en temps et de 11 à 18 % de
surface détectée en moins. Arbitrage non tranché, d'où la liste fermée.

Module **pur** : ni ONNX, ni numpy, ni Qt. La seconde passe est injectée par
l'appelant (``inferer``), ce qui rend tout le module testable avec une
inférence bouchonnée.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..cancellation import PipelineCancelled
from .sahi_lite import compute_ios

logger = logging.getLogger(__name__)

BBox = Sequence[float]

# --- Interrupteur ------------------------------------------------------------
# À False, ``recentrer_les_coupes`` rend sa liste d'entrée inchangée : aucune
# seconde passe, aucune suppression. Sert d'A/B — l'étape SUPPRIME des
# détections, il faut pouvoir comparer les deux sorties sur la même dalle.
# Recompiler le runner après changement : python dev/runner_onnx/build.py
RECENTRAGE_ACTIF = True

# Classes concernées, par NOM (``classes.txt`` du modèle). Volontairement une
# constante et non un réglage d'``args.yaml`` : la règle ne vaut aujourd'hui
# que pour les enclos et n'a été mesurée que sur eux. Le jour où une deuxième
# famille compacte la demande, c'est ici que ça s'ouvre — et à ce moment-là
# seulement ça vaudra un réglage par modèle.
CLASSES_RECENTRAGE = ("enclos", "enclos_circulaire")

# Distance (px) sous laquelle un côté de boîte est considéré posé SUR la
# couture. 4 px : une coupe franche s'aligne au pixel près sur la ligne de
# découpe ; au-delà, le bord de l'objet est un vrai bord.
MARGE_COUTURE_PX = 4

# Appariement première passe ↔ seconde passe : IoS (intersection sur la PLUS
# PETITE des deux boîtes). C'est la bonne métrique ici parce que la première
# est par construction un FRAGMENT de la seconde — une IoU serait basse alors
# que l'appariement est évident. Même seuil et même vocabulaire que
# ``overlap_strategy: relation`` en aval.
IOS_APPARIEMENT = 0.5


def coutures_depuis_slices(
    slice_bboxes: Iterable[Sequence[int]], largeur: int, hauteur: int,
) -> Tuple[List[int], List[int]]:
    """Lignes de découpe INTÉRIEURES, depuis des boîtes ``[x1, y1, x2, y2]``.

    Les bords de l'image n'en sont pas : un objet qui s'arrête sur un bord
    d'image est tronqué par la DONNÉE, pas par le découpage — il n'y a rien à
    recentrer, et le halo inter-dalles s'en occupe.
    """
    xs, ys = set(), set()
    for x1, y1, x2, y2 in slice_bboxes:
        xs.update((x1, x2))
        ys.update((y1, y2))
    return (sorted(v for v in xs if 0 < v < largeur),
            sorted(v for v in ys if 0 < v < hauteur))


def sur_une_couture(
    bbox: BBox, coutures_x: Sequence[int], coutures_y: Sequence[int],
    marge: int = MARGE_COUTURE_PX,
) -> bool:
    """Vrai si un côté de la boîte est posé sur une couture."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (any(abs(x1 - c) <= marge or abs(x2 - c) <= marge for c in coutures_x)
            or any(abs(y1 - c) <= marge or abs(y2 - c) <= marge for c in coutures_y))


def tient_dans_la_fenetre(bbox: BBox, largeur_f: int, hauteur_f: int) -> bool:
    """Vrai si l'objet peut tenir ENTIER dans une fenêtre d'inférence.

    Sinon recentrer ne sert à rien : l'objet débordera de la seconde fenêtre
    comme il débordait de la première, et on le supprimerait pour rien.
    """
    return ((float(bbox[2]) - float(bbox[0])) <= largeur_f
            and (float(bbox[3]) - float(bbox[1])) <= hauteur_f)


def fenetre_centree(
    bbox: BBox, largeur_f: int, hauteur_f: int, largeur: int, hauteur: int,
) -> Tuple[int, int, int, int]:
    """Fenêtre ``(x1, y1, x2, y2)`` de taille d'une tuile SAHI, centrée sur ``bbox``.

    Taille conservée (jamais un cadrage « au plus juste ») : le modèle a été
    entraîné à une échelle donnée, un crop plus large ou plus serré serait
    redimensionné vers l'entrée du réseau et changerait cette échelle. La
    fenêtre est simplement ramenée à l'intérieur de l'image.
    """
    largeur_f = min(int(largeur_f), int(largeur))
    hauteur_f = min(int(hauteur_f), int(hauteur))
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    x1 = max(0, min(int(round(cx - largeur_f / 2.0)), int(largeur) - largeur_f))
    y1 = max(0, min(int(round(cy - hauteur_f / 2.0)), int(hauteur) - hauteur_f))
    return (x1, y1, x1 + largeur_f, y1 + hauteur_f)


def vers_l_image(
    det: Dict, fenetre: Tuple[int, int, int, int], largeur: int, hauteur: int,
) -> Dict:
    """Détection exprimée dans le crop → coordonnées de l'image entière.

    Conventions de ``computer_vision_onnx`` : ``bbox`` en pixels du raster
    d'inférence, ``polygon``/``polygon_holes`` à plat et NORMALISÉS par les
    dimensions de ce raster. ``area`` est en pixels² et ne bouge pas — le crop
    n'est pas redimensionné.
    """
    x1, y1, x2, y2 = fenetre
    f_largeur, f_hauteur = x2 - x1, y2 - y1

    def _recaler(plat: Sequence[float]) -> List[float]:
        return [((v * f_largeur + x1) / largeur) if i % 2 == 0
                else ((v * f_hauteur + y1) / hauteur)
                for i, v in enumerate(plat)]

    sortie = dict(det)
    b = det["bbox"]
    sortie["bbox"] = [float(b[0]) + x1, float(b[1]) + y1,
                      float(b[2]) + x1, float(b[3]) + y1]
    if det.get("polygon") is not None:
        sortie["polygon"] = _recaler(det["polygon"])
    if det.get("polygon_holes"):
        sortie["polygon_holes"] = [_recaler(t) for t in det["polygon_holes"]]
    return sortie


def apparier(
    origine: Dict, candidats: Sequence[Dict], seuil: float = IOS_APPARIEMENT,
) -> Optional[Dict]:
    """Candidat de même classe qui recouvre le mieux ``origine`` (IoS ≥ seuil)."""
    meilleur, meilleur_score = None, float(seuil)
    for cand in candidats:
        if cand.get("class_id") != origine.get("class_id"):
            continue
        score = compute_ios(list(origine["bbox"]), list(cand["bbox"]))
        if score >= meilleur_score:
            meilleur, meilleur_score = cand, score
    return meilleur


def recentrer_les_coupes(
    detections: List[Dict],
    *,
    image,
    largeur: int,
    hauteur: int,
    coutures: Tuple[Sequence[int], Sequence[int]],
    fenetre_px: Tuple[int, int],
    inferer: Callable[[object], List[Dict]],
    noms_de_classes: Optional[Sequence[str]] = None,
    classes: Sequence[str] = CLASSES_RECENTRAGE,
    marge_px: int = MARGE_COUTURE_PX,
    ios: float = IOS_APPARIEMENT,
) -> List[Dict]:
    """Rejoue les détections tranchées par une couture, sur une fenêtre centrée.

    ``inferer(crop)`` rend les détections du crop dans les conventions de
    ``_run_rfdetr_seg_with_sahi`` (bbox en pixels du crop, polygone normalisé
    au crop). Une détection retenue est REMPLACÉE par son appariement ; sans
    appariement elle est SUPPRIMÉE. Les autres traversent inchangées, dans
    l'ordre d'entrée.
    """
    if not RECENTRAGE_ACTIF or not detections:
        return detections
    # Sans le vocabulaire du modèle, impossible de savoir si on a affaire à des
    # enclos : on ne touche à rien plutôt que de recentrer au hasard.
    vises = set(classes)
    ids_vises = {i for i, nom in enumerate(noms_de_classes or ())
                 if str(nom).strip() in vises}
    if not ids_vises:
        return detections

    coutures_x, coutures_y = coutures
    if not coutures_x and not coutures_y:
        return detections

    largeur_f, hauteur_f = int(fenetre_px[0]), int(fenetre_px[1])
    retenues: List[Dict] = []
    n_remplacees = n_supprimees = 0

    for det in detections:
        bbox = det.get("bbox")
        if (det.get("class_id") not in ids_vises
                or not bbox
                or not tient_dans_la_fenetre(bbox, largeur_f, hauteur_f)
                or not sur_une_couture(bbox, coutures_x, coutures_y, marge_px)):
            retenues.append(det)
            continue

        fenetre = fenetre_centree(bbox, largeur_f, hauteur_f, largeur, hauteur)
        try:
            brutes = inferer(image.crop(fenetre)) or []
        except PipelineCancelled:
            # L'annulation utilisateur traverse : elle n'est pas une panne
            # d'inférence, et la manger figerait le pipeline sur « en cours ».
            raise
        except Exception as exc:  # une seconde passe ratée ne supprime rien
            logger.warning("Recentrage: seconde passe échouée sur %s (%s) — "
                           "détection conservée telle quelle", fenetre, exc)
            retenues.append(det)
            continue

        candidats = [vers_l_image(d, fenetre, largeur, hauteur) for d in brutes]
        remplacante = apparier(det, candidats, ios)
        if remplacante is None:
            n_supprimees += 1
            continue
        retenues.append(remplacante)
        n_remplacees += 1

    if n_remplacees or n_supprimees:
        logger.info("Recentrage sur couture: %d détection(s) rejouée(s) et "
                    "remplacée(s), %d supprimée(s) faute de confirmation",
                    n_remplacees, n_supprimees)
    return retenues
