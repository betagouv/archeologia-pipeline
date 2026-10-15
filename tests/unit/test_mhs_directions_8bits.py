"""Étape 2 › M-HS (0.16.0) : « Nombre de directions » masqué quand la sortie est en 8 bits, où il est sans effet
(l'image 8 bits compose trois éclairages). src/ui ne s'importe pas hors QGIS : lecture du source ; rendu vérifié
hors écran (dev/rendu_hors_ecran.py captures) et en recette § 52.4."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[2] / "src/ui/steps/step_2_indices.py").read_text(encoding="utf-8")


def test_directions_masquees_en_8_bits():
    assert "cellule_directions = mdh_dirs.parentWidget()" in SRC
    assert "mdh_8.toggled.connect(lambda huit: cellule_directions.setVisible(not huit))" in SRC
    assert "cellule_directions.setVisible(not mdh_8.isChecked())" in SRC
