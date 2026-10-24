"""Système de conception (lot 7 de l'interface rationalisée, 2026-10-24) : contrastes
AA, cibles de 24 px, anneau de focus, pas de texte sous 10 px hors badges.

Lu dans ``v2.qss`` sans Qt : chaque règle ``sélecteur { corps }`` donne ses couleurs
de texte, de fond et de bordure. Un couple texte/fond se mesure dans la même règle,
ou pour une règle « parent enfant » dont le parent (la partie avant l'espace) a un
fond explicite. Les états ``:disabled`` sont hors contrat (WCAG les exempte)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

QSS = Path(__file__).resolve().parents[2] / "src" / "ui" / "theme" / "v2.qss"
COMMENTAIRES = re.compile(r"/\*.*?\*/", re.DOTALL)
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

#: Cases dont le ✓ est blanc sur blanc tant que la carte n'est pas cochée : voulu.
SANS_CONTRASTE = {"#IndexCheck", "#EntityCheck"}
#: Badges en MAJUSCULES, seuls textes autorisés sous 10 px.
BADGES_9PX = {"#RailOpt", "#StageBadge", "#IndexTag", "#RecapBadge", "#FacultatifTag",
              "#FicheNature", "#FicheColonne", "#ModelInfoKicker"}
#: Cartes dont la bordure doit se détacher du fond (3:1).
CARTES = ("#IndexCard", "#EntityLine", "#RunRow", "#Collapsible", "#RunBilanBox", "#StageButton", "#VisuCard")
#: Boutons et liens : hauteur de contenu déclarée ≥ 20 px (24 px avec le padding).
CIBLES = ("QPushButton", "#GhostButton", "#EntityResetBtn", "#SeeLink", "#CardToggle", "#EntityChangeBtn",
          "#FicheTab", "#EntityActivateBtn", "#FicheNav", "#FicheToggle", "#FichePanelFermer",
          "#JournalChipWarn", "#JournalChipErr", "#VisuFamilyBtn")
#: Anneaux de focus attendus (clavier).
FOCUS = ("#IndexCard:focus", "#EntityLine:focus", "#StageButton:focus", "#RailItem:focus", "#InfoButton:focus",
         "QPushButton:focus", "QCheckBox::indicator:focus", "#SeeLink:focus", "#CardToggle:focus",
         "#EntityChangeBtn:focus", "#EntityResetBtn:focus", "#FicheTab:focus")


def regles() -> dict:
    """``{sélecteur: {propriété: valeur}}`` ; une règle à plusieurs sélecteurs est dupliquée."""
    texte = COMMENTAIRES.sub("", QSS.read_text(encoding="utf-8"))
    out: dict = {}
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", texte):
        decls = {}
        for d in m.group(2).split(";"):
            if ":" in d:
                prop, _, val = d.partition(":")
                decls[prop.strip().lower()] = val.strip()
        for sel in m.group(1).split(","):
            sel = " ".join(sel.split())
            if sel:
                out.setdefault(sel, {}).update(decls)
    return out


def luminance(hexa: str) -> float:
    def lin(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hexa[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contraste(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    clair, sombre = max(la, lb), min(la, lb)
    return (clair + 0.05) / (sombre + 0.05)


def _hex(valeur: str):
    m = HEX.search(valeur or "")
    return m.group(0).lower() if m else None


def _fond(decls: dict):
    return _hex(decls.get("background") or decls.get("background-color") or "")


def _bordure(decls: dict):
    return _hex(decls.get("border") or decls.get("border-color") or "")


def couples_texte_fond():
    """``(sélecteur, texte, fond)`` mesurables, hors ``:disabled``."""
    r = regles()
    for sel, decls in r.items():
        if ":disabled" in sel or sel in SANS_CONTRASTE:
            continue
        texte = _hex(decls.get("color", ""))
        if not texte:
            continue
        fond = _fond(decls)
        if fond is None and " " in sel:
            parent = sel.rsplit(" ", 1)[0]
            fond = _fond(r.get(parent, {}))
        if fond:
            yield sel, texte, fond


def test_contraste_du_texte_aa():
    fautes = [f"{sel}: {texte} sur {fond} = {contraste(texte, fond):.2f}"
              for sel, texte, fond in couples_texte_fond() if contraste(texte, fond) < 4.5]
    assert not fautes, "texte sous 4,5:1 :\n  - " + "\n  - ".join(fautes)


def test_contraste_des_bordures_de_carte():
    r = regles()
    fautes = []
    for sel in CARTES:
        decls = r[sel]
        fond, bord = _fond(decls) or "#ffffff", _bordure(decls)
        assert bord, f"{sel} : pas de bordure déclarée"
        if contraste(bord, fond) < 3.0:
            fautes.append(f"{sel}: bordure {bord} sur {fond} = {contraste(bord, fond):.2f}")
    assert not fautes, "bordure sous 3:1 :\n  - " + "\n  - ".join(fautes)


def test_texte_attenue_unique():
    """Un seul gris de texte atténué, à 7:1 sur blanc ; l'ancien #5a5a5a (6,3:1) ne revient pas."""
    texte = COMMENTAIRES.sub("", QSS.read_text(encoding="utf-8"))
    assert "#5a5a5a" not in texte.lower()
    assert contraste("#4f5a66", "#ffffff") >= 7.0


def test_tailles_minimales():
    r = regles()
    fautes = []
    for sel in CIBLES:
        decls = r.get(sel) or {}
        m = re.search(r"(\d+)px", decls.get("min-height", ""))
        if not m or int(m.group(1)) < 20:
            fautes.append(f"{sel}: min-height = {decls.get('min-height', 'absent')}")
    assert not fautes, "cibles sous 24 px :\n  - " + "\n  - ".join(fautes)


def test_pas_de_texte_sous_10px_hors_badges():
    fautes = []
    for sel, decls in regles().items():
        m = re.search(r"(\d+)px", decls.get("font-size", ""))
        if m and int(m.group(1)) < 10 and sel.split(":")[0].split("[")[0] not in BADGES_9PX:
            fautes.append(f"{sel}: {decls['font-size']}")
    assert not fautes, "texte sous 10 px :\n  - " + "\n  - ".join(fautes)


@pytest.mark.parametrize("sel", FOCUS)
def test_focus_present(sel):
    assert sel in regles(), f"pas d'anneau de focus pour {sel}"


def test_le_calcul_est_celui_du_wcag():
    assert abs(contraste("#000000", "#ffffff") - 21.0) < 0.01
    assert 4.5 < contraste("#2b79c2", "#ffffff") < 4.6     # le bleu du thème passe AA de justesse sur blanc
    assert contraste("#2b79c2", "#d6e6f4") < 4.5             # … et pas sur une carte cochée
