"""F1 (audit « rassemblement de polygones ») — distinction empty / failure.

``create_shapefile_from_detections`` renvoyait ``False`` AUSSI pour le cas
légitime « aucune détection trouvée », rendant ce cas indistinguable d'une
vraie panne (et provoquant, côté narrateur, un faux « ❌ Échec » sur tout run
sans détection). Le contrat corrigé :

- répertoire de labels présent mais sans détection → ``True`` (succès, rien à
  écrire) ;
- vraie panne (répertoire de labels absent, exception) → ``False``.

Test au niveau de la VRAIE fonction (nécessite geopandas → ``skip`` hors
environnement complet QGIS/dev).
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__
pytest.importorskip("geopandas")  # create_shapefile_from_detections (top-level import)

from pipeline.cv.conversion_shp import create_shapefile_from_detections  # noqa: E402


def test_empty_labels_dir_is_success_not_failure(tmp_path):
    # Répertoire présent mais sans aucun fichier de détection → 0 détection
    # légitime : ce n'est PAS une panne, on doit renvoyer True.
    labels = tmp_path / "labels"
    labels.mkdir()
    out = tmp_path / "detections.gpkg"
    ok = create_shapefile_from_detections(
        labels_dir=str(labels), output_shapefile=str(out)
    )
    assert ok is True


def test_missing_labels_dir_is_failure(tmp_path):
    # Répertoire de labels ABSENT → vraie panne, doit rester False.
    out = tmp_path / "detections.gpkg"
    ok = create_shapefile_from_detections(
        labels_dir=str(tmp_path / "does_not_exist"), output_shapefile=str(out)
    )
    assert ok is False
