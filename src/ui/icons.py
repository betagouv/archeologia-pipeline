"""Chargement + recoloration des icônes SVG du wizard V2.

Les SVG (``theme/icons/*.svg``) sont monochromes, couleur en dur ``#2c2c2c``
(``currentColor`` ne fonctionne pas via QIcon dans Qt — cf. README des icônes).
On remplace la couleur par celle voulue puis on rend en :class:`QPixmap`
rasterisé **directement à la résolution physique de l'écran** (DPR réel,
pixels entiers) : aucun rééchantillonnage au dessin, donc pas de flou, quel
que soit le facteur d'échelle Windows (100/125/150 %).

L'import de ``QtSvg`` est **défensif** : si le module manque dans l'install
QGIS, les icônes sont simplement vides — le dialogue continue de fonctionner.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from qgis.PyQt.QtCore import QByteArray, Qt
from qgis.PyQt.QtGui import QIcon, QPainter, QPixmap
from qgis.PyQt.QtWidgets import QApplication

try:
    from qgis.PyQt.QtSvg import QSvgRenderer
    SVG_AVAILABLE = True
except Exception:  # PyQt5.QtSvg absent de certaines installs QGIS
    QSvgRenderer = None  # type: ignore[assignment]
    SVG_AVAILABLE = False

_ICON_DIR = Path(__file__).parent / "theme" / "icons"
_DEFAULT_HARDCODED = "#2c2c2c"


@lru_cache(maxsize=64)
def _svg_text(name: str) -> str:
    path = _ICON_DIR / f"{name}.svg"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except Exception:
        return ""


def _recolor(svg: str, color: str) -> str:
    """Remplace la couleur en dur (#2c2c2c) — et currentColor par sécurité."""
    return (
        svg.replace(f'"{_DEFAULT_HARDCODED}"', f'"{color}"')
        .replace('"#2C2C2C"', f'"{color}"')
        .replace("currentColor", color)
    )


def _screen_dpr() -> float:
    """DPR réel de l'écran (1.0 hors application, p. ex. en import standalone)."""
    app = QApplication.instance()
    try:
        return float(app.devicePixelRatio()) if app is not None else 1.0
    except Exception:
        return 1.0


def colored_pixmap(name: str, color: str, size: int = 24, dpr: float | None = None) -> QPixmap:
    """Rend l'icône ``name`` teintée en ``color`` (hex) à ``size`` px logiques.

    Le SVG est rasterisé à ``size × dpr`` pixels physiques **entiers**, et le
    pixmap porte exactement ce ratio : QLabel le dessine alors pixel pour
    pixel, sans rééchantillonnage (la source historique du flou — un rendu 2×
    fixe rescalé à l'affichage). Passer ``dpr=widget.devicePixelRatioF()``
    pour être exact en multi-écrans ; à défaut le DPR de l'application sert.

    Retourne un pixmap vide si QtSvg est absent ou l'icône introuvable.
    """
    svg = _svg_text(name)
    if not svg or not SVG_AVAILABLE:
        return QPixmap()
    if dpr is None or dpr <= 0:
        dpr = _screen_dpr()
    size = max(1, size)
    renderer = QSvgRenderer(QByteArray(_recolor(svg, color).encode("utf-8")))
    px = max(1, round(size * dpr))  # pixels physiques entiers
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(px / size)
    return pixmap


def colored_icon(name: str, color: str, size: int = 24, dpr: float | None = None) -> QIcon:
    return QIcon(colored_pixmap(name, color, size, dpr))
