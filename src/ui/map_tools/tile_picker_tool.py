"""Outil-carte de sélection des dalles IGN sur le canevas (``QgsMapTool``).

L'utilisateur **clique une dalle pour la (dé)sélectionner** ou **trace une boîte
pour en sélectionner plusieurs**. La sélection vit dans la couche elle-même
(``QgsVectorLayer.select`` / ``selectByIds``) → surlignage natif QGIS « gratuit »
et source de vérité unique (``selectedFeatureIds``). ``Échap`` annule.

Premier ``QgsMapTool`` du projet. Module importé uniquement au runtime (jamais
collecté par pytest), donc l'import de ``qgis.gui``/``qgis.core`` au niveau module
est volontaire (la classe de base est nécessaire dès la définition de classe).
Enums systématiquement scopés (compat Qt5/Qt6).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.core import (
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsMapTool, QgsRubberBand

from ...app.services.grid_view import grid_is_hidden

# Déplacement écran (px) en deçà duquel un relâché est traité comme un clic
# (sélection d'une dalle) plutôt qu'un glisser-boîte.
_CLICK_THRESHOLD_PX = 4


class TilePickerMapTool(QgsMapTool):
    """Sélection de dalles par clic (toggle) ou glisser-boîte (ajout)."""

    selection_changed = pyqtSignal(int)  # nb de dalles sélectionnées
    cancelled = pyqtSignal()             # Échap
    too_zoomed_out = pyqtSignal()        # action tentée alors que la grille est masquée

    def __init__(self, canvas, layer: "QgsVectorLayer"):
        super().__init__(canvas)
        self._canvas = canvas
        self._layer = layer
        self._drag_start = None  # QPoint du début de glisser, ou None
        self._rubber: "QgsRubberBand | None" = None

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def activate(self):  # noqa: N802 (signature Qt)
        super().activate()
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def deactivate(self):  # noqa: N802 (signature Qt)
        self._reset_rubber()
        super().deactivate()

    def cleanup(self) -> None:
        """À appeler par l'appelant à la fin (validation/annulation). Idempotent."""
        self._reset_rubber()
        self._drag_start = None

    # ------------------------------------------------------------------
    # Événements souris
    # ------------------------------------------------------------------
    def canvasPressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def canvasMoveEvent(self, event):  # noqa: N802
        if self._drag_start is not None:
            remove = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._update_rubber(event.pos(), remove=remove)

    def canvasReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        start = self._drag_start
        self._drag_start = None
        self._reset_rubber()
        if start is None:
            return
        # Vue trop dézoomée : la grille n'est pas dessinée → une sélection serait
        # invisible. On refuse l'action et on prévient (U1).
        if self._grid_hidden():
            self.too_zoomed_out.emit()
            return
        if (event.pos() - start).manhattanLength() < _CLICK_THRESHOLD_PX:
            self._toggle_at(event.pos())
        else:
            # Ctrl + glisser-boîte = retirer de la sélection (convention QGIS).
            remove = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._select_box(start, event.pos(), remove=remove)

    def _grid_hidden(self) -> bool:
        """La grille est-elle masquée à l'échelle courante (vue trop dézoomée) ?"""
        return grid_is_hidden(
            self._layer.hasScaleBasedVisibility(),
            self._canvas.scale(),
            self._layer.minimumScale(),
        )

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()

    # ------------------------------------------------------------------
    # Sélection
    # ------------------------------------------------------------------
    def _toggle_at(self, screen_pt) -> None:
        pt = self._to_layer_point(self.toMapCoordinates(screen_pt))
        rect = QgsRectangle(pt.x() - 1.0, pt.y() - 1.0, pt.x() + 1.0, pt.y() + 1.0)
        point_geom = QgsGeometry.fromPointXY(pt)
        req = QgsFeatureRequest().setFilterRect(rect)
        fid = None
        for feat in self._layer.getFeatures(req):  # fenêtre via index .qix
            geom = feat.geometry()
            if geom is not None and geom.intersects(point_geom):
                fid = feat.id()
                break
        if fid is None:
            return
        if fid in set(self._layer.selectedFeatureIds()):
            self._layer.deselect(fid)
        else:
            self._layer.select(fid)
        self.selection_changed.emit(len(self._layer.selectedFeatureIds()))

    def _select_box(self, start_pt, end_pt, *, remove: bool = False) -> None:
        rect_map = QgsRectangle(self.toMapCoordinates(start_pt), self.toMapCoordinates(end_pt))
        rect = self._to_layer_rect(rect_map)
        ids = [f.id() for f in self._layer.getFeatures(QgsFeatureRequest().setFilterRect(rect))]
        if not ids:
            return
        behavior = (
            QgsVectorLayer.SelectBehavior.RemoveFromSelection
            if remove
            else QgsVectorLayer.SelectBehavior.AddToSelection
        )
        self._layer.selectByIds(ids, behavior)
        self.selection_changed.emit(len(self._layer.selectedFeatureIds()))

    # ------------------------------------------------------------------
    # Transformations CRS canevas → couche (quadrillage = EPSG:2154)
    # ------------------------------------------------------------------
    def _transform(self) -> "QgsCoordinateTransform | None":
        canvas_crs = self._canvas.mapSettings().destinationCrs()
        layer_crs = self._layer.crs()
        if canvas_crs == layer_crs:
            return None
        return QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())

    def _to_layer_point(self, map_pt: "QgsPointXY") -> "QgsPointXY":
        xform = self._transform()
        return xform.transform(map_pt) if xform is not None else map_pt

    def _to_layer_rect(self, map_rect: "QgsRectangle") -> "QgsRectangle":
        xform = self._transform()
        return xform.transformBoundingBox(map_rect) if xform is not None else map_rect

    # ------------------------------------------------------------------
    # Rubber band (rectangle de glisser)
    # ------------------------------------------------------------------
    def _ensure_rubber(self) -> "QgsRubberBand":
        if self._rubber is None:
            rb = QgsRubberBand(self._canvas, QgsWkbTypes.GeometryType.PolygonGeometry)
            rb.setWidth(2)
            self._rubber = rb
        return self._rubber

    def _update_rubber(self, screen_pt, remove: bool = False) -> None:
        rb = self._ensure_rubber()
        # Orange = ajout ; gris = retrait (Ctrl) → signale le mode « gomme ».
        rb.setColor(QColor(110, 110, 110, 70) if remove else QColor(230, 80, 40, 60))
        rect = QgsRectangle(self.toMapCoordinates(self._drag_start), self.toMapCoordinates(screen_pt))
        rb.setToGeometry(QgsGeometry.fromRect(rect), None)

    def _reset_rubber(self) -> None:
        if self._rubber is not None:
            self._rubber.reset(QgsWkbTypes.GeometryType.PolygonGeometry)
