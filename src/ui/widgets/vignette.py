"""Vignette d'accès à une fiche — le carré cliquable de 44 px, partagé.

Deux endroits montrent la même chose au même format : la carte d'une entité à
détecter (étape 3, :mod:`entity_card`) et la carte d'un produit à calculer
(étape 2, :mod:`steps.step_2_indices`). Un seul widget, donc un seul visuel —
c'est la demande : la fiche d'un indice s'ouvre comme celle d'un modèle.

Le découpage est le point délicat. Une vignette couvre plusieurs centaines de
mètres de terrain ; réduite telle quelle à 44 px elle devient une bouillie
grise. La fiche montre toujours le cadre entier, l'icône n'en montre que la
fenêtre carrée déclarée par ``cadrage`` (fractions de l'image, bornée en amont
par ``app.services.class_fiche.cadrage_fractions``).

Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from typing import Optional, Tuple

from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import QPushButton

#: Côté de la vignette de carte, en px logiques.
TAILLE = 44

#: Marge intérieure : laisse respirer le liseré du cadre.
_MARGE = 2


def pixmap_ajuste(
    source,
    cote: int,
    *,
    dpr: float = 1.0,
    cadrage: Optional[Tuple[float, float, float]] = None,
) -> QPixmap:
    """Pixmap carré de ``cote`` px **logiques**, rendu à la densité de l'écran.

    Même discipline que :func:`ui.icons.colored_pixmap` : on rastérise à
    ``cote × dpr`` pixels **physiques** et on pose le ratio correspondant sur le
    pixmap, si bien que Qt le dessine pixel pour pixel à la taille logique
    demandée.

    Sans cela, un pixmap chargé depuis un fichier porte un ratio de 1 : sur un
    écran à 125 % ou 150 %, il n'occupe que ``cote / dpr`` px logiques dans un
    cadre de ``cote`` px, et le fond clair du cadre apparaît en liseré sur les
    côtés (constat utilisateur 2026-09-16, sur les vignettes de classes comme
    de produits) — en plus d'être rééchantillonné, donc flou.

    ``source`` est un chemin ou un ``QPixmap``. Un fichier illisible rend un
    pixmap nul, que l'appelant traite comme une absence d'image.
    """
    pix = QPixmap(source) if isinstance(source, str) else source
    if pix is None or pix.isNull():
        return QPixmap()
    if cadrage:
        pix = decouper_cadrage(pix, cadrage)
    cote = max(1, int(cote))
    phys = max(1, round(cote * (dpr if dpr and dpr > 0 else 1.0)))
    out = pix.scaled(
        phys, phys,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    # Ratio RÉEL après arrondi (et non ``dpr``) : c'est ce qui garantit que le
    # plus grand côté retombe exactement sur ``cote`` px logiques.
    out.setDevicePixelRatio(phys / cote)
    return out


def decouper_cadrage(pix: QPixmap, cadrage: Tuple[float, float, float]) -> QPixmap:
    """Découpe la fenêtre ``(x, y, côté)`` fractionnaire du pixmap.

    Le cadrage est déjà borné à l'image par ``app.services.class_fiche`` ; on
    reborne quand même ici (un pixmap non carré donnerait un rectangle hors
    limites) et on rend l'image entière si le découpage est vide.
    """
    x, y, cote = cadrage
    c = max(1, int(round(cote * pix.width())))
    left = min(max(0, int(round(x * pix.width()))), max(0, pix.width() - c))
    top = min(max(0, int(round(y * pix.height()))), max(0, pix.height() - c))
    decoupe = pix.copy(left, top, c, c)
    return pix if decoupe.isNull() else decoupe


class FicheThumb(QPushButton):
    """Carré de 44 px qui ouvre une fiche. Sans image : cadre d'attente.

    Émet :attr:`clicked` (hérité de ``QPushButton``) ; c'est l'appelant qui
    décide quelle fiche ouvrir. En tant que bouton, il consomme son propre clic
    et ne déclenche donc pas la bascule de la carte qui le porte.
    """

    def __init__(self, tooltip: str = "Voir la fiche", parent=None):
        super().__init__("", parent)
        self.setObjectName("FicheThumb")
        self.setFixedSize(TAILLE, TAILLE)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)

    def set_vignette(
        self,
        chemin: Optional[str],
        *,
        disponible: bool = True,
        cadrage: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Pose l'image, ou le cadre d'attente si elle manque.

        ``chemin`` absent ou illisible → cadre d'attente (la vignette n'a pas
        encore été produite). ``disponible=False`` → carré éteint : il n'y a
        rien à montrer et la fiche ne s'ouvrira pas.
        """
        self.setEnabled(disponible)
        cote = TAILLE - _MARGE * 2
        pix = pixmap_ajuste(
            chemin, cote, dpr=self.devicePixelRatioF(), cadrage=cadrage
        ) if chemin else QPixmap()
        if pix.isNull():
            self.setIcon(QIcon())
            self.setText("◌" if disponible else "")
            self.setProperty("state", "vide")
        else:
            self.setText("")
            self.setIcon(QIcon(pix))
            self.setIconSize(QSize(cote, cote))   # px logiques, comme le pixmap
            self.setProperty("state", "plein")
        # Repolish : la propriété dynamique pilote le style du cadre.
        self.style().unpolish(self)
        self.style().polish(self)


class FicheButton(QPushButton):
    """Le lien texte « Fiche » de l'en-tête d'une carte."""

    def __init__(self, tooltip: str = "", parent=None):
        super().__init__("Fiche", parent)
        self.setObjectName("FicheBtn")
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)


__all__ = ["FicheButton", "FicheThumb", "TAILLE", "decouper_cadrage", "pixmap_ajuste"]
