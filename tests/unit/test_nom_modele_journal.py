"""Le journal de l'assistant nomme un modèle par son ``display_name`` (recette 0.14.0)."""
from app.services.cv_post_service import _model_display_name


def test_display_name_du_model_card(tmp_path):
    (tmp_path / "m1" / "weights").mkdir(parents=True)
    (tmp_path / "m1" / "model_card.yaml").write_text("display_name: Modèle cratères\n", encoding="utf-8")
    assert _model_display_name(str(tmp_path / "m1" / "weights" / "best.onnx")) == "Modèle cratères"
    assert _model_display_name(str(tmp_path / "m1")) == "Modèle cratères"


def test_repli_sur_le_nom_du_dossier(tmp_path):
    (tmp_path / "m2").mkdir()
    assert _model_display_name(str(tmp_path / "m2")) == "m2"
    assert _model_display_name("modele_inconnu") == "modele_inconnu"
