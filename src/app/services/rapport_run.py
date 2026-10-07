"""Rapport de traitement — module PUR : un fichier HTML dans le dossier de sortie.

À la fin d'un traitement, ``finalize_service`` assemble ce que le journal et les
métadonnées savent déjà — mode et source, dalles, produits et leurs réglages,
modèles lancés avec leur durée mesurée, bilan de fiabilité (``bilan_fiabilite``),
avertissements du journal avec leur renvoi au manuel — et une vignette de la zone
rendue depuis la mosaïque du premier indice, puis écrit ``rapport.html`` (et
``rapport_vignette.png``) à la racine du dossier de sortie. La vue d'exécution
l'ouvre d'un clic (bouton « Rapport »). C'est le document à joindre au dossier de
prospection : aucune ressource externe, il se lit hors ligne et s'imprime.

Ce module ne dépend ni de Qt ni de GDAL : :func:`vignette_depuis_vrt` importe
``osgeo.gdal`` à l'appel et rend ``None`` sans lui. Les chiffres sont ceux du
run (durées chronométrées, effectifs) — jamais une estimation.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .aide import rubrique_depannage

NOM_RAPPORT = "rapport.html"
NOM_VIGNETTE = "rapport_vignette.png"

#: Produits préférés pour la vignette, du plus lisible au moins parlant.
_ORDRE_VIGNETTE = ("VAT", "CVAT", "PRISM", "CRIM", "M_HS", "HS", "SVF", "LD", "OPNS",
                   "SLRM", "MSTP", "SLO", "MNT", "COUVERTURE", "DENSITE")

_LIGNE_LOG = re.compile(r"^.*? - (WARNING|ERROR) - (.*)$")
_DALLE = re.compile(r"(\d{4})_(\d{4})")

_ISSUES = {
    "success": "terminé",
    "cancelled": "annulé",
    "failed": "en échec",
}


@dataclass(frozen=True)
class Avertissement:
    niveau: str          # "warn" | "err"
    message: str
    rubrique: str = ""   # titre de la rubrique de Dépannage du manuel, "" sinon


@dataclass(frozen=True)
class DonneesRapport:
    version: str
    date: str
    mode: str
    output_dir: str
    issue: str = "success"
    duree_s: float = 0.0
    tiles_processed: int = 0
    tiles_total: Optional[int] = None
    dalles: Tuple[str, ...] = ()
    produits: Tuple[Tuple[str, str], ...] = ()          # (code, libellé)
    rvt_params: Mapping[str, Any] = field(default_factory=dict)
    cv_runs: Tuple[Mapping[str, Any], ...] = ()         # modele, target_rvt, entites, images, secondes
    bilan: Tuple[Mapping[str, Any], ...] = ()           # LigneBilan.to_dict()
    avertissements: Tuple[Avertissement, ...] = ()
    vignette: Optional[str] = None                      # chemin relatif au rapport


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
def extraire_avertissements(texte_log: str, max_lignes: int = 60) -> List[Avertissement]:
    """Les lignes WARNING / ERROR du journal de fichier (format
    ``asctime - LEVEL - message``), dédoublonnées, dans l'ordre, avec la rubrique
    de Dépannage qui en parle."""
    out: List[Avertissement] = []
    vus = set()
    for ligne in texte_log.splitlines():
        m = _LIGNE_LOG.match(ligne.strip())
        if not m:
            continue
        message = m.group(2).strip()
        if not message or message in vus:
            continue
        vus.add(message)
        out.append(Avertissement("err" if m.group(1) == "ERROR" else "warn", message, rubrique_depannage(message)))
        if len(out) >= max_lignes:
            break
    return out


def nom_dalle(stem: str) -> str:
    """``LHD_FXX_0873_6506_MNT_A_0M50_LAMB93_IGN69`` → ``0873-6506`` ; sinon le nom tel quel."""
    m = _DALLE.search(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem


def dalles_depuis_vrt(vrt_paths: Sequence[str]) -> List[str]:
    """Les dalles du run = les TIF du dossier de la première mosaïque, un nom par dalle."""
    for vrt in vrt_paths:
        if not vrt:
            continue
        dossier = Path(vrt).parent
        stems = sorted(p.stem for p in dossier.glob("*.tif"))
        if stems:
            return [nom_dalle(s) for s in stems]
    return []


def choisir_vrt_vignette(vrt_paths: Sequence[str]) -> Optional[str]:
    """La mosaïque la plus parlante pour la vignette (VAT, CVAT, … puis MNT)."""
    par_produit: Dict[str, str] = {}
    for vrt in vrt_paths:
        if not vrt:
            continue
        p = Path(vrt)
        produit = (p.parent.parent.name if p.parent.name == "tif" else p.parent.name).split("_")[0]
        par_produit.setdefault(produit, str(vrt))
    for code in _ORDRE_VIGNETTE:
        if code in par_produit:
            return par_produit[code]
    return next(iter(par_produit.values()), None)


def vignette_depuis_vrt(vrt: str, png: Path, largeur: int = 720) -> Optional[str]:
    """Rend la mosaïque en PNG 8 bits de ``largeur`` px (étirement min-max approché),
    sans GDAL ou en cas d'échec → ``None``. Thread worker, aucun Qt."""
    try:
        from osgeo import gdal

        gdal.UseExceptions()
        src = gdal.Open(str(vrt))
        if src is None:
            return None
        bande = src.GetRasterBand(1)
        mn, mx = bande.ComputeRasterMinMax(True)      # approché : overviews / échantillon
        if mx <= mn:
            return None
        nodata = bande.GetNoDataValue()
        options = gdal.TranslateOptions(
            format="PNG", width=largeur, height=0, outputType=gdal.GDT_Byte,
            scaleParams=[[mn, mx, 1, 255]],
            **({"noData": nodata} if nodata is not None else {}),
        )
        png.parent.mkdir(parents=True, exist_ok=True)
        pam = gdal.GetConfigOption("GDAL_PAM_ENABLED")
        gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")     # pas de .aux.xml à côté du PNG
        try:
            if gdal.Translate(str(png), src, options=options) is None:
                return None
        finally:
            gdal.SetConfigOption("GDAL_PAM_ENABLED", pam)
        src = None
        return png.name if png.is_file() else None
    except Exception:  # noqa: BLE001 — la vignette est un confort
        return None


# ----------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------
_CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; color: #2c2c2c; margin: 32px auto; max-width: 900px; padding: 0 16px; }
h1 { font-size: 22px; margin-bottom: 4px; } h2 { font-size: 15px; margin-top: 28px; border-bottom: 1px solid #e3e3e3; padding-bottom: 4px; }
p, li, td, th { font-size: 13px; } .sous { color: #5a5a5a; } .ok { color: #2a7a2a; } .ko { color: #a83838; } .annule { color: #8a5e18; }
table { border-collapse: collapse; margin: 8px 0; } th, td { border: 1px solid #dddddd; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; color: #5a5a5a; font-weight: normal; }
.barre { display: inline-block; height: 12px; vertical-align: middle; }
.n4 { background: #1d5a96; } .n3 { background: #2b79c2; } .n2 { background: #7fb0dc; } .n1 { background: #c9dcee; }
.leg span { display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 10px; vertical-align: middle; }
.warn { color: #8a5e18; } .err { color: #a83838; } .renvoi { color: #7d786c; }
img.vignette { max-width: 100%; border: 1px solid #c4c4c4; }
details summary { cursor: pointer; color: #1d5a96; } code { background: #f0f0f0; padding: 0 3px; }
@media print { body { margin: 0; } details { display: block; } }
"""

_NIVEAUX = (("quasi_certain", "très probables", "n4"), ("probable", "probables", "n3"),
            ("possible", "possibles", "n2"), ("douteux", "douteuses", "n1"))


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _nb(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def format_duree(secondes: float) -> str:
    s = int(round(secondes))
    h, reste = divmod(s, 3600)
    m, s = divmod(reste, 60)
    if h:
        return f"{h}h {m:02d}min"
    if m:
        return f"{m}min {s:02d}s"
    return f"{s}s"


def _parametres(rvt_params: Mapping[str, Any]) -> str:
    lignes = []
    for section, valeurs in sorted((rvt_params or {}).items()):
        if isinstance(valeurs, Mapping):
            detail = ", ".join(f"{_e(k)} = {_e(v)}" for k, v in valeurs.items())
        else:
            detail = _e(valeurs)
        lignes.append(f"<tr><th>{_e(section)}</th><td>{detail}</td></tr>")
    return f"<table>{''.join(lignes)}</table>" if lignes else "<p class=sous>Réglages par défaut.</p>"


def _bilan(bilan: Sequence[Mapping[str, Any]]) -> str:
    if not bilan:
        return "<p class=sous>Aucune détection avec fiabilité mesurée.</p>"
    maximum = max(int(ligne.get("total") or 0) for ligne in bilan) or 1
    lignes = []
    for ligne in bilan:
        eff = ligne.get("effectifs") or {}
        total = int(ligne.get("total") or 0)
        longueur = 260 * total / maximum
        segments = "".join(
            f'<span class="barre {cls}" style="width:{longueur * int(eff.get(cat, 0)) / total:.0f}px"></span>'
            for cat, _lab, cls in _NIVEAUX if total and int(eff.get(cat, 0)) > 0
        )
        detail = ", ".join(f"{_nb(eff[cat])} {lab}" for cat, lab, _cls in _NIVEAUX if eff.get(cat))
        lignes.append(
            f"<tr><td>{_e(ligne.get('label') or ligne.get('slug') or '')}</td>"
            f"<td>{_nb(total)}</td><td>{segments}</td><td>{detail}</td></tr>"
        )
    legende = "".join(f'<span class="{cls}"></span>{lab}' for _cat, lab, cls in _NIVEAUX)
    return (
        "<table><tr><th>Entité</th><th>Détections</th><th>Du plus sûr au plus douteux</th><th>Par niveau</th></tr>"
        f"{''.join(lignes)}</table><p class=\"sous leg\">{legende}</p>"
    )


def _cv_runs(runs: Sequence[Mapping[str, Any]]) -> str:
    if not runs:
        return "<p class=sous>Pas de détection automatique dans ce traitement.</p>"
    lignes = []
    for r in runs:
        images, secondes = r.get("images"), r.get("secondes")
        if images is None:
            duree = "—"
        elif not images:
            duree = "aucune image analysée (résultats déjà en cache)"
        else:
            duree = f"{_nb(images)} image{'s' if images > 1 else ''} en {format_duree(float(secondes or 0))}"
            if images > 1 and secondes:
                duree += f" (≈ {format_duree(float(secondes) / images)} par image)"
        entites = ", ".join(_e(x) for x in (r.get("entites") or [])) or "—"
        lignes.append(f"<tr><td>{_e(r.get('modele') or r.get('model') or '')}</td><td>{_e(r.get('target_rvt') or '')}</td>"
                      f"<td>{entites}</td><td>{duree}</td></tr>")
    return ("<table><tr><th>Modèle</th><th>Indice</th><th>Entités</th><th>Images analysées, durée mesurée</th></tr>"
            f"{''.join(lignes)}</table>")


def _avertissements(liste: Sequence[Avertissement]) -> str:
    if not liste:
        return "<p class=sous>Aucun avertissement.</p>"
    items = []
    for a in liste:
        glyphe = "✗" if a.niveau == "err" else "⚠"
        renvoi = f' <span class=renvoi>· voir le manuel › Dépannage › « {_e(a.rubrique)} »</span>' if a.rubrique else ""
        items.append(f'<li class="{a.niveau}">{glyphe} {_e(a.message)}{renvoi}</li>')
    return f"<ul>{''.join(items)}</ul>"


def _dalles(dalles: Sequence[str], tiles_processed: int, tiles_total: Optional[int]) -> str:
    compte = f"{_nb(tiles_processed)}"
    if tiles_total is not None and tiles_total != tiles_processed:
        compte += f" sur {_nb(tiles_total)}"
    texte = f"<p>{compte} dalle{'s' if tiles_processed > 1 else ''} traitée{'s' if tiles_processed > 1 else ''}.</p>"
    if not dalles:
        return texte
    noms = ", ".join(_e(d) for d in dalles)
    if len(dalles) > 24:
        return texte + f"<details><summary>Liste des {len(dalles)} dalles</summary><p class=sous>{noms}</p></details>"
    return texte + f"<p class=sous>{noms}</p>"


def construire_html(d: DonneesRapport) -> str:
    classe_issue = {"success": "ok", "cancelled": "annule"}.get(d.issue, "ko")
    issue = _ISSUES.get(d.issue, d.issue)
    produits = ", ".join(f"{_e(lab)} (<code>{_e(code)}</code>)" for code, lab in d.produits) or "aucun"
    vignette = (f'<img class=vignette src="{_e(d.vignette)}" alt="Vignette de la zone traitée">'
                if d.vignette else "<p class=sous>Pas de vignette (mosaïque indisponible).</p>")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Rapport de traitement — {_e(d.date)}</title>
<style>{_CSS}</style></head><body>
<h1>Rapport de traitement</h1>
<p class=sous>Archéolog'IA v{_e(d.version)} — {_e(d.date)} — {_e(d.mode)}</p>
<p>Traitement <b class="{classe_issue}">{_e(issue)}</b> en {_e(format_duree(d.duree_s))}. Dossier : <code>{_e(d.output_dir)}</code></p>
<h2>Zone traitée</h2>
{vignette}
{_dalles(d.dalles, d.tiles_processed, d.tiles_total)}
<h2>Produits et réglages</h2>
<p>{produits}</p>
{_parametres(d.rvt_params)}
<h2>Détection automatique</h2>
{_cv_runs(d.cv_runs)}
<h2>Bilan de fiabilité — par où commencer</h2>
{_bilan(d.bilan)}
<h2>Avertissements du journal</h2>
{_avertissements(d.avertissements)}
<p class=sous>La configuration complète est dans <code>metadata.json</code>, le journal détaillé dans <code>pipeline_log_*.txt</code>. Les durées sont chronométrées, les effectifs comptés : aucun chiffre de ce rapport n'est une estimation.</p>
</body></html>
"""


def ecrire_rapport(output_dir: Path, d: DonneesRapport) -> Path:
    chemin = Path(output_dir) / NOM_RAPPORT
    chemin.write_text(construire_html(d), encoding="utf-8")
    return chemin
