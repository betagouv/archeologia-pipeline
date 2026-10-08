"""Rapport de traitement — module PUR : un fichier HTML dans le dossier de sortie.

À la fin d'un traitement, ``finalize_service`` assemble ce que le journal et les
métadonnées savent déjà — mode, surface couverte, produits et leurs réglages en
mots, modèles sous leur nom avec leur seuil et leur durée mesurée, bilan de
fiabilité (``bilan_fiabilite``), durées par étape, avertissements du journal avec
leur renvoi au manuel, sources — et une vignette de la zone rendue depuis la
mosaïque du premier indice, puis écrit ``rapport.html`` (et ``rapport_vignette.png``)
à la racine du dossier de sortie. La vue d'exécution l'ouvre d'un clic (bouton
« Rapport »). C'est le document à joindre au dossier de prospection : aucune
ressource externe, il se lit hors ligne et s'imprime.

Le rapport ne situe pas la zone (demande utilisateur 2026-10-08) : ni nom de dalle
(une coordonnée Lambert-93 au kilomètre), ni chemin de dossier ; les avertissements
sont anonymisés (:func:`anonymiser`) et la vignette, sans coordonnée, est ramenée à
720 px de plus grand côté. Les chiffres sont ceux du run — durées chronométrées,
effectifs comptés, surface mesurée sur les rasters — jamais une estimation.

Ce module ne dépend ni de Qt ni de GDAL : :func:`vignette_depuis_vrt` et
:func:`surface_km2_depuis_vrt` importent ``osgeo.gdal`` à l'appel et rendent
``None`` sans lui.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .aide import rubrique_depannage
from .reglages_defaut import SECTION_PRODUIT

NOM_RAPPORT = "rapport.html"
NOM_VIGNETTE = "rapport_vignette.png"

#: Produits préférés pour la vignette, du plus lisible au moins parlant.
_ORDRE_VIGNETTE = ("VAT", "CVAT", "PRISM", "CRIM", "M_HS", "HS", "SVF", "LD", "OPNS",
                   "SLRM", "MSTP", "SLO", "MNT", "COUVERTURE", "DENSITE")

_LIGNE_LOG = re.compile(r"^.*? - (WARNING|ERROR) - (.*)$")
#: Coordonnées d'une dalle IGN (``0988_6872`` ou ``0988-6872``) : situent au kilomètre.
_DALLE = re.compile(r"\d{4}[_-]\d{4}")
#: Chemin absolu Windows (``D:\…``, ``\\serveur\…``) ou POSIX (``/…/…``), jusqu'au
#: prochain blanc ou guillemet ; il faut au moins un second séparateur (« 1/4 » n'est
#: pas un chemin).
_CHEMIN = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/)[^\s\"'<>|]*[\\/][^\s\"'<>|]*")

_ISSUES = {
    "success": "terminé",
    "cancelled": "annulé",
    "failed": "en échec",
}

#: Section de ``rvt_params`` de chaque produit (l'inverse de ``SECTION_PRODUIT``).
_SECTION_PAR_PRODUIT = {produit: section for section, produit in SECTION_PRODUIT.items()}
_TERRAIN = {0: "terrain général", 1: "terrain plat", 2: "terrain pentu"}
_PALETTES = {"OrRd": "orangé vers rouge", "Reds_r": "rouge inversé", "YlOrRd": "jaune, orangé, rouge",
             "Greys_r": "gris inversé", "gist_earth": "terrain"}

#: Ce que garantit chaque niveau de fiabilité (part de vrais objets au banc) : le
#: même contrat pour tous les modèles, cf. chapitre « Vos résultats » du manuel.
_NIVEAUX = (("quasi_certain", "très probables", "très probable", "n4", "au moins 85 % de vrais objets"),
            ("probable", "probables", "probable", "n3", "au moins 60 %"),
            ("possible", "possibles", "possible", "n2", "au moins 35 %"),
            ("douteux", "douteuses", "douteux", "n1", "moins de 35 %"))

_DONNEES_PAR_MODE = {
    "ign_laz": "LiDAR HD de l'IGN, Licence Ouverte 2.0.",
    "local_laz": "nuages de points LiDAR fournis par l'utilisateur.",
    "existing_mnt": "modèle numérique de terrain fourni par l'utilisateur.",
    "existing_rvt": "indices de visualisation fournis par l'utilisateur.",
}


@dataclass(frozen=True)
class Avertissement:
    niveau: str          # "warn" | "err"
    message: str         # anonymisé : sans chemin ni coordonnée de dalle
    rubrique: str = ""   # titre de la rubrique de Dépannage du manuel, "" sinon
    occurrences: int = 1


@dataclass(frozen=True)
class DonneesRapport:
    version: str
    date: str
    mode: str                                           # libellé du mode (bandeau de l'étape 1)
    data_mode: str = ""                                 # ign_laz, local_laz, existing_mnt, existing_rvt
    issue: str = "success"
    duree_s: float = 0.0
    tiles_processed: int = 0
    tiles_total: Optional[int] = None
    surface_km2: Optional[float] = None                 # mesurée sur les rasters du run
    produits: Tuple[Tuple[str, str], ...] = ()          # (code, libellé)
    rvt_params: Mapping[str, Any] = field(default_factory=dict)
    cv_runs: Tuple[Mapping[str, Any], ...] = ()         # modele, target_rvt, entites, seuil, images, secondes
    bilan: Tuple[Mapping[str, Any], ...] = ()           # LigneBilan.to_dict()
    etapes: Tuple[Tuple[str, str, float], ...] = ()     # (étape, détail, secondes mesurées)
    avertissements: Tuple[Avertissement, ...] = ()
    vignette: Optional[str] = None                      # chemin relatif au rapport


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
def anonymiser(message: str) -> str:
    """Un message du journal sans rien qui situe la zone : un chemin devient son
    nom de fichier, et toute coordonnée de dalle devient « … »."""
    def _nom(m: "re.Match[str]") -> str:
        return re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1] or "…"
    return _DALLE.sub("…", _CHEMIN.sub(_nom, message))


def extraire_avertissements(texte_log: str, max_lignes: int = 60) -> List[Avertissement]:
    """Les lignes WARNING / ERROR du journal de fichier (format
    ``asctime - LEVEL - message``), anonymisées, regroupées quand elles sont
    identiques (``occurrences``), dans l'ordre, avec la rubrique de Dépannage qui
    en parle."""
    out: List[Avertissement] = []
    index: Dict[Tuple[str, str], int] = {}
    for ligne in texte_log.splitlines():
        m = _LIGNE_LOG.match(ligne.strip())
        if not m:
            continue
        message = anonymiser(m.group(2).strip())
        if not message:
            continue
        niveau = "err" if m.group(1) == "ERROR" else "warn"
        cle = (niveau, message)
        if cle in index:
            i = index[cle]
            out[i] = replace(out[i], occurrences=out[i].occurrences + 1)
        elif len(out) < max_lignes:
            index[cle] = len(out)
            out.append(Avertissement(niveau, message, rubrique_depannage(message)))
    return out


def _tifs_de_la_mosaique(vrt_paths: Sequence[str]) -> List[Path]:
    for vrt in vrt_paths:
        if not vrt:
            continue
        tifs = sorted(Path(vrt).parent.glob("*.tif"))
        if tifs:
            return tifs
    return []


def surface_km2_depuis_vrt(vrt_paths: Sequence[str]) -> Optional[float]:
    """Surface couverte par les dalles de la première mosaïque, en km², mesurée
    sur chaque raster (emprise = taille × résolution ; les dalles rognées ne se
    recouvrent pas). Sans GDAL ou sans raster → ``None``."""
    tifs = _tifs_de_la_mosaique(vrt_paths)
    if not tifs:
        return None
    try:
        from osgeo import gdal

        gdal.UseExceptions()
    except ImportError:
        return None
    total = 0.0
    for tif in tifs:
        try:
            ds = gdal.Open(str(tif))
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            total += abs(gt[1] * ds.RasterXSize * gt[5] * ds.RasterYSize)
            ds = None
        except Exception:  # noqa: BLE001 — un raster illisible ne compte pas
            continue
    return total / 1e6 if total > 0 else None


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


def vignette_depuis_vrt(vrt: str, png: Path, cote: int = 720) -> Optional[str]:
    """Rend la mosaïque en PNG 8 bits de ``cote`` px de plus grand côté (étirement
    min-max approché), sans GDAL ou en cas d'échec → ``None``. Thread worker, aucun Qt."""
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
        portrait = src.RasterYSize > src.RasterXSize
        options = gdal.TranslateOptions(
            format="PNG", width=0 if portrait else cote, height=cote if portrait else 0,
            outputType=gdal.GDT_Byte, scaleParams=[[mn, mx, 1, 255]],
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
.garde { border-left: 3px solid #8a5e18; background: #fbf1df; padding: 8px 12px; margin: 14px 0; }
table { border-collapse: collapse; margin: 8px 0; } th, td { border: 1px solid #dddddd; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; color: #5a5a5a; font-weight: normal; } td.num { text-align: right; white-space: nowrap; }
.barre { display: inline-block; height: 12px; vertical-align: middle; }
.n4 { background: #1d5a96; } .n3 { background: #2b79c2; } .n2 { background: #7fb0dc; } .n1 { background: #c9dcee; }
.leg span { display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 10px; vertical-align: middle; }
.warn { color: #8a5e18; } .err { color: #a83838; } .renvoi { color: #7d786c; }
img.vignette { max-width: 100%; max-height: 720px; border: 1px solid #c4c4c4; }
code { background: #f0f0f0; padding: 0 3px; }
@media print { body { margin: 0; } }
"""


def _e(x: Any) -> str:
    """Texte d'un nœud HTML : < > & échappés, apostrophes et guillemets gardés lisibles."""
    return html.escape(str(x), quote=False)


def _a(x: Any) -> str:
    """Valeur d'attribut HTML : guillemets échappés aussi."""
    return html.escape(str(x), quote=True)


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _dec(x: Any, decimales: int = 1) -> str:
    """Nombre en écriture française : ``1.7`` → « 1,7 », ``1.0`` → « 1 », ``12.0`` → « 12 »."""
    try:
        s = f"{float(x):.{decimales}f}"
    except (TypeError, ValueError):
        return str(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def format_duree(secondes: float) -> str:
    s = int(round(secondes))
    h, m = s // 3600, (s % 3600) // 60
    if h:
        return f"{h}h {m:02d}min"
    if m:
        return f"{m}min {s % 60:02d}s"
    return f"{s}s"


def _bruit(n: Any) -> str:
    n = int(n or 0)
    return "sans suppression du bruit" if n == 0 else f"suppression du bruit {n}"


def decrire_reglages(section: str, valeurs: Optional[Mapping[str, Any]]) -> str:
    """Les réglages d'une section de ``rvt_params`` en mots et en unités, avec le
    vocabulaire de l'étape 2 : « rayon 10 px, 16 directions, sans suppression du
    bruit ». ``None`` → « réglages par défaut »."""
    if section == "cvat":
        return "composition fixe"
    if not valeurs:
        return "réglages par défaut"
    g = valeurs.get
    m: List[str] = []
    if section == "hs":
        m += [f"azimut solaire {_dec(g('sun_azimuth', 315), 0)}°", f"élévation solaire {_dec(g('sun_elevation', 35), 0)}°"]
    elif section == "mdh":
        m += [f"{_nb(int(g('num_directions', 16)))} directions", f"élévation solaire {_dec(g('sun_elevation', 35), 0)}°"]
    elif section in ("svf", "opns"):
        if section == "opns":
            m.append("ouverture négative, les creux" if int(g("opns_type", 0) or 0) == 1 else "ouverture positive, les saillies")
        m += [f"rayon {_dec(g('radius', 10), 0)} px", f"{_nb(int(g('num_directions', 16)))} directions", _bruit(g("noise_remove", 0))]
    elif section == "slope":
        m.append("en pourcentage" if int(g("unit", 0) or 0) == 1 else "en degrés")
    elif section == "ldo":
        m += [f"rayon de {_dec(g('min_radius', 10), 0)} à {_dec(g('max_radius', 20), 0)} px",
              f"résolution angulaire {_dec(g('angular_res', 15), 0)}°",
              f"hauteur d'observateur {_dec(g('observer_h', 1.7))} m"]
    elif section == "slrm":
        m.append(f"rayon {_dec(g('radius', 20), 0)} px")
    elif section in ("vat", "prism"):
        m.append(_TERRAIN.get(int(g("terrain_type", 0) or 0), "terrain général"))
    elif section == "crim":
        palette = str(g("colormap", "OrRd"))
        m += [f"palette {_PALETTES.get(palette, palette)}",
              f"coupes de la palette de {_dec(g('min_colormap_cut', 0.0), 2)} à {_dec(g('max_colormap_cut', 1.0), 2)}"]
    elif section == "mstp":
        for nom, cle in (("locale", "local"), ("méso", "meso"), ("large", "broad")):
            m.append(f"échelle {nom} de {_dec(g(f'{cle}_scale_min'), 0)} à {_dec(g(f'{cle}_scale_max'), 0)} px "
                     f"par pas de {_dec(g(f'{cle}_scale_step'), 0)}")
        m.append(f"luminosité {_dec(g('lightness', 1.2))}")
    ve = g("ve_factor")
    if ve is not None and float(ve) != 1.0:
        m.append(f"exagération verticale ×{_dec(ve)}")
    return ", ".join(m) if m else "réglages par défaut"


def _zone(d: DonneesRapport) -> str:
    n = d.tiles_processed
    pluriel = "s" if n > 1 else ""
    if d.surface_km2 is not None:
        s = d.surface_km2
        surface = _nb(int(round(s))) if s >= 10 else _dec(s, 1)
        texte = f"{surface} km² couverts, en {_nb(n)} dalle{pluriel}"
    else:
        texte = f"{_nb(n)} dalle{pluriel} traitée{pluriel}"
    if d.tiles_total is not None and d.tiles_total != n:
        texte += f", sur {_nb(d.tiles_total)} prévues"
    return f"<p>{texte}.</p>" + ("<p class=sous>Ce rapport ne porte ni nom de dalle ni coordonnée : "
                    "les couches du dossier de traitement les ont.</p>")


def _produits(d: DonneesRapport) -> str:
    if not d.produits:
        return "<p class=sous>Aucun produit.</p>"
    libelles = ", ".join(f"{_e(lab)} (<code>{_e(code)}</code>)" for code, lab in d.produits)
    if d.data_mode == "existing_rvt":
        return (f"<p>Indices fournis avec les données, calculés hors du plugin ; leurs réglages "
                f"ne sont pas connus : {libelles}.</p>")
    lignes = []
    for code, lab in d.produits:
        if code == "MNT" and d.data_mode == "existing_mnt":
            desc = "fourni avec les données"
        elif code in ("MNT", "DENSITE", "COUVERTURE"):
            desc = "sans réglage"
        else:
            section = _SECTION_PAR_PRODUIT.get(code, "")
            desc = decrire_reglages(section, (d.rvt_params or {}).get(section)) if section else "réglages par défaut"
        lignes.append(f"<tr><th>{_e(lab)} (<code>{_e(code)}</code>)</th><td>{_e(desc)}</td></tr>")
    tete = ("<p>Modèle de terrain fourni avec les données ; les indices ont été calculés par le plugin.</p>"
            if d.data_mode == "existing_mnt" else "")
    return tete + f"<table>{''.join(lignes)}</table>"


def _bilan(bilan: Sequence[Mapping[str, Any]], surface_km2: Optional[float]) -> str:
    if not bilan:
        return "<p class=sous>Aucune détection avec fiabilité mesurée.</p>"
    maximum = max(int(ligne.get("total") or 0) for ligne in bilan) or 1
    densite = surface_km2 is not None and surface_km2 > 0
    lignes = []
    for ligne in bilan:
        eff = ligne.get("effectifs") or {}
        total = int(ligne.get("total") or 0)
        longueur = 260 * total / maximum
        segments = "".join(
            f'<span class="barre {cls}" style="width:{longueur * int(eff.get(cat, 0)) / total:.0f}px"></span>'
            for cat, _lab, _sing, cls, _g in _NIVEAUX if total and int(eff.get(cat, 0)) > 0
        )
        detail = ", ".join(f"{_nb(eff[cat])} {lab}" for cat, lab, _sing, _cls, _g in _NIVEAUX if eff.get(cat))
        col_densite = f'<td class=num>{_dec(total / surface_km2, 1)}</td>' if densite else ""
        lignes.append(
            f"<tr><td>{_e(ligne.get('label') or ligne.get('slug') or '')}</td>"
            f"<td class=num>{_nb(total)}</td>{col_densite}<td>{segments}</td><td>{detail}</td></tr>"
        )
    th_densite = "<th>Par km²</th>" if densite else ""
    legende = " · ".join(f'<span class="{cls}"></span>{sing} : {garantie}'
                        for _cat, _lab, sing, cls, garantie in _NIVEAUX)
    return (
        f"<table><tr><th>Entité</th><th>Détections</th>{th_densite}<th>Du plus sûr au plus douteux</th><th>Par niveau</th></tr>"
        f"{''.join(lignes)}</table><p class=\"sous leg\">{legende}</p>"
    )


def _cv_runs(runs: Sequence[Mapping[str, Any]]) -> str:
    if not runs:
        return "<p class=sous>Pas de détection automatique dans ce traitement.</p>"
    lignes = []
    for r in runs:
        images, secondes = r.get("images"), r.get("secondes")
        if images is None:
            analyse = "—"
        elif not images:
            analyse = "reprise du run précédent"
        else:
            analyse = f"{_nb(images)} image{'s' if images > 1 else ''} en {format_duree(float(secondes or 0))}"
            if images > 1 and secondes:
                par_image = float(secondes) / images
                analyse += (f" (≈ {_dec(par_image, 1)} s par image)" if par_image < 10
                            else f" (≈ {format_duree(par_image)} par image)")
        entites = ", ".join(_e(x) for x in (r.get("entites") or [])) or "—"
        modele = _e(r.get("modele") or r.get("model") or "")
        if r.get("target_rvt"):
            modele += f", sur {_e(r['target_rvt'])}"
        seuil = r.get("seuil")
        lignes.append(f"<tr><td>{entites}</td><td>{modele}</td>"
                      f"<td class=num>{f'{float(seuil):.2f}'.replace('.', ',') if seuil is not None else '—'}</td><td>{analyse}</td></tr>")
    return ("<table><tr><th>Entités</th><th>Modèle</th><th>Seuil</th><th>Analyse, durée mesurée</th></tr>"
            f"{''.join(lignes)}</table>")


def _etapes(etapes: Sequence[Tuple[str, str, float]]) -> str:
    lignes = "".join(f"<tr><td>{_e(nom)}</td><td>{_e(detail)}</td><td class=num>{_e(format_duree(s))}</td></tr>"
                     for nom, detail, s in etapes)
    return f"<table><tr><th>Étape</th><th>Détail</th><th>Durée mesurée</th></tr>{lignes}</table>"


def _avertissements(liste: Sequence[Avertissement]) -> str:
    if not liste:
        return "<p class=sous>Aucun avertissement.</p>"
    items = []
    for a in liste:
        glyphe = "✗" if a.niveau == "err" else "⚠"
        fois = f" <span class=sous>(×{_nb(a.occurrences)})</span>" if a.occurrences > 1 else ""
        renvoi = f' <span class=renvoi>· voir le manuel › Dépannage › « {_e(a.rubrique)} »</span>' if a.rubrique else ""
        items.append(f'<li class="{a.niveau}">{glyphe} {_e(a.message)}{fois}{renvoi}</li>')
    return (f"<ul>{''.join(items)}</ul><p class=sous>Les chemins et les noms de dalles sont masqués ; "
            "le journal du traitement garde le détail.</p>")


def _sources(d: DonneesRapport) -> str:
    lignes = []
    donnees = _DONNEES_PAR_MODE.get(d.data_mode)
    if donnees:
        lignes.append(f"<p>Données : {_e(donnees)}</p>")
    if d.data_mode and d.data_mode != "existing_rvt":
        lignes.append("<p>Indices de visualisation : Relief Visualization Toolbox (ZRC SAZU), "
                      "par l'extension QGIS rvt-qgis.</p>")
    modeles = []
    for r in d.cv_runs:
        nom = str(r.get("modele") or r.get("model") or "")
        if nom and nom not in modeles:
            modeles.append(nom)
    if modeles:
        lignes.append(f"<p>Modèles de détection : {', '.join(_e(m) for m in modeles)}. La fiche de chaque "
                      "structure, à l'étape 3 du plugin, dit où et sur quoi le modèle a appris.</p>")
    return "".join(lignes) or "<p class=sous>—</p>"


def construire_html(d: DonneesRapport) -> str:
    classe_issue = {"success": "ok", "cancelled": "annule"}.get(d.issue, "ko")
    issue = _ISSUES.get(d.issue, d.issue)
    vignette = (f'<img class=vignette src="{_a(d.vignette)}" alt="Vignette de la zone traitée">'
                if d.vignette else "<p class=sous>Pas de vignette (mosaïque indisponible).</p>")
    if d.cv_runs or d.bilan:
        garde = ("Les détections sont des hypothèses produites par des modèles d'apprentissage : chacune se "
                 "vérifie sur le relief avant toute interprétation. Ce rapport ne localise aucune détection ; "
                 "les couches géolocalisées restent dans le dossier du traitement, à côté de ce rapport.")
    else:
        garde = ("Ce rapport ne localise pas la zone traitée ; les produits restent dans le dossier du "
                 "traitement, à côté de ce rapport.")
    etapes = f"<h2>Durées par étape</h2>\n{_etapes(d.etapes)}\n" if d.etapes else ""
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Rapport de traitement — {_e(d.date)}</title>
<style>{_CSS}</style></head><body>
<h1>Rapport de traitement</h1>
<p class=sous>Archéolog'IA v{_e(d.version)} — {_e(d.date)} — {_e(d.mode)}</p>
<p class=garde>{garde}</p>
<p>Traitement <b class="{classe_issue}">{_e(issue)}</b> en {_e(format_duree(d.duree_s))}.</p>
<h2>Zone traitée</h2>
{vignette}
{_zone(d)}
<h2>Produits</h2>
{_produits(d)}
<h2>Détection automatique</h2>
{_cv_runs(d.cv_runs)}
<h2>Bilan de fiabilité — par où commencer</h2>
{_bilan(d.bilan, d.surface_km2)}
{etapes}<h2>Avertissements du journal</h2>
{_avertissements(d.avertissements)}
<h2>Sources</h2>
{_sources(d)}
<p class=sous>La configuration complète est dans <code>metadata.json</code>, le journal détaillé dans <code>pipeline_log_*.txt</code>, tous deux dans le dossier du traitement. Les durées sont chronométrées, les effectifs comptés, la surface mesurée : aucun chiffre de ce rapport n'est une estimation.</p>
</body></html>
"""


def ecrire_rapport(output_dir: Path, d: DonneesRapport) -> Path:
    chemin = Path(output_dir) / NOM_RAPPORT
    chemin.write_text(construire_html(d), encoding="utf-8")
    return chemin
