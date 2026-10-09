"""Vérité terrain dans la couleur de la classe (2026-10-08) : recoloration pure (numpy)."""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from app.services.recolorer_annotation import part_de_trace, recolorer  # noqa: E402


def test_le_relief_gris_reste_le_trace_jaune_prend_la_couleur():
    brut = np.full((4, 4, 3), 120, np.uint8)
    brut[0, 0] = 30                                   # relief sombre
    annote = brut.copy()
    annote[1, 1] = (230, 220, 40)                     # tracé jaune plein
    annote[2, 2] = (140, 136, 110)                    # bord anticrénelé, en partie
    annote[3, 3] = (122, 121, 118)                    # bruit de chrominance JPEG
    out = recolorer(brut, annote, (6, 239, 249))
    assert tuple(out[0, 0]) == (30, 30, 30) and tuple(out[0, 1]) == (120, 120, 120)
    assert tuple(out[1, 1]) == (6, 239, 249)
    assert tuple(out[3, 3]) == (120, 120, 120)        # sous le seuil : relief inchangé
    t = part_de_trace(annote)
    assert 0.2 < t[2, 2] < 0.5 and t[3, 3] == 0.0 and t[1, 1] == 1.0
    mi = out[2, 2].astype(int)
    assert 120 > mi[0] > 6 and 239 > mi[1] > 120      # mélange relief / couleur au bord


def test_tailles_differentes_rend_l_annotee():
    a = np.zeros((2, 2, 3), np.uint8)
    b = np.zeros((3, 3, 3), np.uint8)
    assert recolorer(a, b, (1, 2, 3)) is b


def test_vraies_vignettes_tous_les_traces_sont_recolores():
    """Sur les vignettes livrées : après recoloration, plus aucun pixel jaune franc."""
    Image = pytest.importorskip("PIL.Image")
    racine = Path(__file__).resolve().parents[2]
    paires = [(b, b.with_name(b.name.replace("_brut.", "_annote."))) for b in sorted((racine / "data" / "models").glob("*/vignettes/*_brut.*"))]
    paires = [(b, a) for b, a in paires if a.is_file()][:6]
    if not paires:
        pytest.skip("aucune vignette installée")
    for b, a in paires:
        brut = np.asarray(Image.open(b).convert("RGB"))
        annote = np.asarray(Image.open(a).convert("RGB"))
        out = recolorer(brut, annote, (40, 90, 220)).astype(int)
        jaune = (out[..., 0] > 160) & (out[..., 1] > 150) & (out[..., 2] < 90)
        assert jaune.mean() < 0.001, f"{a.name} : {jaune.mean():.3%} de pixels jaunes restants"


def test_l_apercu_de_la_fiche_recolore():
    racine = Path(__file__).resolve().parents[2]
    src = (racine / "src/ui/dialogs/class_info_dialog.py").read_text(encoding="utf-8")
    assert "vignette_annotee_recoloree(" in src and "_Apercu(fiche, model_dir, couleur=base)" in src
