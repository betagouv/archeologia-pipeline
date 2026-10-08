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

#: Ardoise des fiches de MODÈLE (liseré, étiquette, lien) — jamais une couleur de classe :
#: la couleur d'une classe est réservée à sa fiche, qui reprend celle de sa couche
#: (option A « liseré et étiquette de nature », validée par l'utilisateur le 2026-10-08).
ARDOISE = "#3d4b5c"


def couleur_texte(rgb) -> str:
    """La couleur ``rgb`` assombrie jusqu'à rester lisible en texte sur fond blanc
    (luminance < 0,33) : une classe cyan ou vert vif garde sa teinte en étiquette."""
    r, g, b = (int(v) for v in rgb)
    for t in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        m = [round(v * (1 - t)) for v in (r, g, b)]
        if (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255 < 0.33:
            return "#%02x%02x%02x" % tuple(m)
    return "#%02x%02x%02x" % tuple(round(v * 0.3) for v in (r, g, b))


def pastille(rgb, cote: int = 10, dpr: float = 1.0):
    """Un disque plein de la couleur ``rgb`` (liseré assombri), net à la densité d'écran."""
    from qgis.PyQt.QtCore import QRectF, Qt
    from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPixmap

    pm = QPixmap(round(cote * dpr), round(cote * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(couleur_texte(rgb)), 1))
    p.setBrush(QColor(*[int(v) for v in rgb]))
    p.drawEllipse(QRectF(0.5, 0.5, cote - 1, cote - 1))
    p.end()
    return pm


#: Côté de la vignette de carte, en px logiques.
TAILLE = 44

#: Marge intérieure : laisse respirer le liseré du cadre.
_MARGE = 2


def vignette_annotee_recoloree(chemin_brut: str, chemin_annote: str, rgb) -> QPixmap:
    """L'image « Vérité terrain » avec ses contours dans la couleur ``rgb`` de la classe
    (``app.services.recolorer_annotation``). Image brute absente, tailles différentes ou
    numpy indisponible → l'image annotée telle quelle, contours jaunes."""
    from qgis.PyQt.QtGui import QImage

    annote = QImage(chemin_annote)
    brut = QImage(chemin_brut)
    if annote.isNull():
        return QPixmap()
    if brut.isNull() or brut.size() != annote.size() or rgb is None:
        return QPixmap.fromImage(annote)
    try:
        import numpy as np

        from ...app.services.recolorer_annotation import recolorer

        def tableau(img):
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            w, h, ligne = img.width(), img.height(), img.bytesPerLine()
            ptr = img.bits()
            ptr.setsize(h * ligne)
            return np.frombuffer(ptr, np.uint8).reshape(h, ligne)[:, : w * 3].reshape(h, w, 3).copy()

        out = np.ascontiguousarray(recolorer(tableau(brut), tableau(annote), rgb))
        h, w = out.shape[:2]
        img = QImage(out.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()   # copie : out est local
        return QPixmap.fromImage(img)
    except Exception:  # noqa: BLE001 — confort visuel, jamais bloquant
        return QPixmap.fromImage(annote)


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
