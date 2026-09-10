"""Fiabilité des détections — catégories « douteux / possible / probable / très probable ».

Décision utilisateur du 2026-09-09 (propositions A + D) : les catégories affichées
dans QGIS sont définies par la **part de vrais objets mesurée sur le banc** dans la
tranche de score (précision locale), pas par le score brut. « Probable » veut donc
dire la même chose pour tous les modèles (≥ 60 % de vrais objets) ; ce sont les
**coupures de score qui bougent, par classe**. Elles sont choisies à l'évaluation
(training-models, cellule 11bis du notebook) et figées dans le ``model_card.yaml`` :

.. code-block:: yaml

    thresholds:
      fiabilite:
        par_classe:
          depression_circulaire_grande:
          - {categorie: douteux,       seuil: 0.29, garanti: 0.0,  mesure: 0.14, n: 447}
          - {categorie: possible,      seuil: 0.35, garanti: 0.35, mesure: 0.47, n: 272}
          - {categorie: probable,      seuil: 0.45, garanti: 0.60, mesure: 0.72, n: 335}
          - {categorie: quasi_certain, seuil: 0.65, garanti: 0.85, mesure: 0.95, n: 1013}
        provenance: "..."

``seuil`` = score à partir duquel la catégorie s'applique (la première = le seuil de
la classe), ``garanti`` = niveau de part de vrais objets qui définit la catégorie,
``mesure`` = part réellement mesurée sur le banc (``null`` sous ``N_MIN_MESURE``
détections), ``n`` = effectif du banc dans la catégorie.

Ce module est **pur** (pas de QGIS, pas de shapely) : il porte le contrat, la
catégorisation d'une détection, les textes (légende, résumé de couche, infobulle
de détection, aide de l'étape 3) et le sidecar ``fiabilite.json`` écrit à côté du
GeoPackage — lu identiquement par le chargement live et l'écriture du ``.qgs``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CATEGORIES: Tuple[str, ...] = ("douteux", "possible", "probable", "quasi_certain")
LABELS_FR: Dict[str, str] = {
    "douteux": "Douteux",
    "possible": "Possible",
    "probable": "Probable",
    "quasi_certain": "Très probable",
}
N_MIN_MESURE = 30
SIDECAR_NAME = "fiabilite.json"
# Champs écrits dans le GeoPackage à la conversion.
CHAMP_LABEL = "fiabilite"        # libellé FR de la catégorie (clé de la symbologie)
CHAMP_PCT = "fiabilite_pct"      # part de vrais objets mesurée sur le banc, en % (NULL si non mesurée)

# Rendu retenu (décision utilisateur 2026-09-09, après essai de deux alternatives — D à
# remplissage, qui cachait la structure détectée, et B à motif de trait) : le rendu
# D'ORIGINE des tranches de confiance, appliqué aux catégories. Contour seul, sans
# remplissage, dans la couleur de la classe (registre : TOUJOURS la même couleur pour
# une classe) déclinée en luminosité par catégorie via ``color_palette.apply_confidence``
# (plus sombre = plus sûr, plus clair = plus douteux). `repr` = valeur représentative
# donnée au dégradé, fixe par catégorie donc identique pour tous les modèles.
STYLE_SPEC: Dict[str, Dict[str, Any]] = {
    "quasi_certain": {"repr": 0.9, "outline_width": 0.6},   # base assombrie de 30 %
    "probable":      {"repr": 0.7, "outline_width": 0.6},   # base assombrie de 15 %
    "possible":      {"repr": 0.5, "outline_width": 0.6},   # couleur de base
    "douteux":       {"repr": 0.3, "outline_width": 0.6},   # base éclaircie de 35 %
}


@dataclass(frozen=True)
class Categorie:
    categorie: str
    seuil: float
    garanti: float
    mesure: Optional[float]
    n: int

    @property
    def label(self) -> str:
        return LABELS_FR.get(self.categorie, self.categorie)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Contrat model_card → catégories
# ----------------------------------------------------------------------
def _parse_categorie(raw: Any) -> Optional[Categorie]:
    if not isinstance(raw, Mapping):
        return None
    cat = str(raw.get("categorie") or "").strip()
    if cat not in CATEGORIES:
        return None
    try:
        seuil = float(raw.get("seuil"))
        garanti = float(raw.get("garanti", 0.0) or 0.0)
        n = int(raw.get("n", 0) or 0)
        mesure = raw.get("mesure")
        mesure = None if mesure is None else float(mesure)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= seuil <= 1.0) or not (0.0 <= garanti <= 1.0):
        return None
    if mesure is not None and not (0.0 <= mesure <= 1.0):
        return None
    return Categorie(cat, seuil, garanti, mesure, n)


def parse_fiabilite(thresholds: Any) -> Tuple[Dict[str, Tuple[Categorie, ...]], str]:
    """``(par_classe, provenance)`` depuis ``model_card:thresholds``.

    Tolérant : bloc absent → ``({}, "")`` ; une classe dont la liste est invalide
    (catégorie inconnue, seuils non croissants, doublon) est ignorée — jamais fatal
    (model_card édité à la main), le plugin retombe alors sur les tranches
    ``conf_bin`` historiques pour cette classe.
    """
    if not isinstance(thresholds, Mapping):
        return {}, ""
    bloc = thresholds.get("fiabilite")
    if not isinstance(bloc, Mapping):
        return {}, ""
    out: Dict[str, Tuple[Categorie, ...]] = {}
    for classe, liste in (bloc.get("par_classe") or {}).items():
        if not isinstance(liste, list):
            continue
        cats = [c for c in (_parse_categorie(x) for x in liste) if c is not None]
        if len(cats) != len(liste) or not cats:
            continue
        ordre = [CATEGORIES.index(c.categorie) for c in cats]
        seuils = [c.seuil for c in cats]
        if ordre != sorted(ordre) or len(set(ordre)) != len(ordre):
            continue
        if any(b <= a for a, b in zip(seuils, seuils[1:])):
            continue
        out[str(classe)] = tuple(cats)
    return out, str(bloc.get("provenance") or "")


def categories_effectives(
    cats: Sequence[Categorie], seuil_effectif: float
) -> Tuple[Categorie, ...]:
    """Catégories d'une classe pour le seuil EFFECTIF du run (surcharge UI comprise).

    Invariant seuil = symbologie = filtrage : la catégorie la plus basse commence
    exactement au seuil effectif. Seuil relevé par l'utilisateur → les catégories
    entièrement sous le seuil disparaissent, la première restante démarre au seuil ;
    seuil abaissé → la catégorie basse s'étend jusqu'au seuil (sa ``mesure`` reste
    celle du banc au seuil du modèle, la partie ajoutée n'étant pas mesurée).
    """
    ordonnees = sorted(cats, key=lambda c: c.seuil)
    if not ordonnees:
        return ()
    s = float(seuil_effectif)
    gardees: List[Categorie] = []
    for i, c in enumerate(ordonnees):
        fin = ordonnees[i + 1].seuil if i + 1 < len(ordonnees) else 1.01
        if fin <= s + 1e-9:
            continue  # entièrement sous le seuil effectif
        gardees.append(c)
    if not gardees:
        return ()
    premiere = gardees[0]
    if abs(premiere.seuil - s) > 1e-9:
        gardees[0] = Categorie(premiere.categorie, s, premiere.garanti, premiere.mesure, premiere.n)
    return tuple(gardees)


def categoriser(confidence: Any, cats: Sequence[Categorie]) -> Optional[Categorie]:
    """Catégorie d'une détection (score ``confidence``), ``None`` si sous la première."""
    if confidence is None or not cats:
        return None
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return None
    if 1.0 < c <= 10.0:  # même tolérance que conf_bin (confiance sur [0,10])
        c /= 10.0
    trouvee: Optional[Categorie] = None
    for cat in sorted(cats, key=lambda x: x.seuil):
        if c >= cat.seuil - 1e-9:
            trouvee = cat
    return trouvee


def pct(x: Optional[float]) -> Optional[int]:
    """Part → pourcentage entier (``None`` conservé)."""
    return None if x is None else int(round(float(x) * 100))


# ----------------------------------------------------------------------
# Textes (légende, résumé de couche, infobulle, aide étape 3)
# ----------------------------------------------------------------------
def labels_legende(cats: Sequence[Categorie]) -> Dict[str, str]:
    """Libellé court de légende par catégorie : le niveau GARANTI seulement
    (« Probable · ≥ 60 % de vrais ») ; la catégorie basse dit la borne du dessus."""
    ordonnees = sorted(cats, key=lambda c: c.seuil)
    out: Dict[str, str] = {}
    for i, c in enumerate(ordonnees):
        if c.garanti > 0:
            out[c.categorie] = f"{c.label} · ≥ {pct(c.garanti)} % de vrais"
        elif i + 1 < len(ordonnees):
            out[c.categorie] = f"{c.label} · < {pct(ordonnees[i + 1].garanti)} % de vrais"
        else:
            out[c.categorie] = f"{c.label} · part de vrais non garantie"
    return out


def phrase_mesure(cat: Categorie) -> str:
    """« ≥ 85 % de vrais objets sur le banc (mesuré : 95 % sur 1 013 détections) »."""
    garantie = (f"≥ {pct(cat.garanti)} % de vrais objets sur le banc" if cat.garanti > 0
                else "part de vrais objets non garantie")
    if cat.mesure is None:
        return f"{garantie} (effectif insuffisant : {cat.n} détection{'s' if cat.n > 1 else ''})"
    return f"{garantie} (mesuré : {pct(cat.mesure)} % sur {cat.n:,} détections)".replace(",", chr(32))  # séparateur de milliers = espace simple


def texte_resume(
    modele: str, classe: str, cats: Sequence[Categorie], provenance: str = ""
) -> str:
    """Résumé de couche (abstract QGIS / métadonnées) : la table complète de la classe."""
    lignes = [f"Fiabilité des détections « {classe} » — {modele}"]
    for c in sorted(cats, key=lambda x: x.seuil, reverse=True):
        lignes.append(f"• {c.label} (score ≥ {c.seuil:.2f}) : {phrase_mesure(c)}")
    lignes.append("Les catégories sont définies par la part de vrais objets mesurée sur "
                  "l'évaluation du modèle, pas par le score brut ; elles se comparent "
                  "donc d'un modèle à l'autre.")
    if provenance:
        lignes.append(f"Source : {provenance}")
    return "\n".join(lignes)


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def maptip_html(cats: Sequence[Categorie], modele: str) -> str:
    """Gabarit d'infobulle QGIS (``setMapTipTemplate``) : phrase de la catégorie de
    LA détection survolée, score brut et modèle. Expressions QGIS entre ``[% %]``."""
    branches = []
    for c in cats:
        if c.mesure is None:
            texte = (f"Part de vrais objets non mesurée pour cette tranche (effectif insuffisant "
                     f"sur le banc : {c.n})" + (f" ; garantie ≥ {pct(c.garanti)} %." if c.garanti > 0 else "."))
        else:
            texte = (f"Sur le banc, {pct(c.mesure)} % des détections de cette tranche étaient de "
                     f"vrais objets" + (f" (garantie ≥ {pct(c.garanti)} %)." if c.garanti > 0
                                        else " (tranche la plus basse, non garantie)."))
        branches.append(f'WHEN "{CHAMP_LABEL}" = {_sql_str(c.label)} THEN {_sql_str(texte)}')
    cas = "CASE " + " ".join(branches) + " ELSE 'Fiabilité non renseignée.' END"
    return (
        "<div style=\"font-family:sans-serif;font-size:12px;max-width:340px\">"
        f"<b>[% \"model_pred\" %]</b> — fiabilité <b>[% \"{CHAMP_LABEL}\" %]</b><br>"
        f"[% {cas} %]<br>"
        f"<span style=\"color:#666\">Score brut [% round(\"confidence\", 2) %] · {modele}</span>"
        "</div>"
    )


def hint_etape3(par_classe: Mapping[str, Sequence[Categorie]]) -> str:
    """Aide sous la case « Confiance » (étape 3) : les coupures effectives, par classe."""
    parties = []
    for classe, cats in par_classe.items():
        if not cats:
            continue
        coupures = " · ".join(f"{c.label.lower()} dès {c.seuil:.2f}" for c in sorted(cats, key=lambda x: x.seuil))
        parties.append(f"{classe} : {coupures}" if len(par_classe) > 1 else coupures)
    if not parties:
        return ""
    return "Fiabilité affichée — " + " ; ".join(parties)


# ----------------------------------------------------------------------
# Bloc de run (computer_vision.runs[].fiabilite) et sidecar fiabilite.json
# ----------------------------------------------------------------------
def run_block(
    par_classe_modele: Mapping[str, Sequence[Categorie]],
    seuils_par_classe: Mapping[str, float],
    classes: Iterable[str],
    *,
    modele: str,
    provenance: str = "",
) -> Optional[Dict[str, Any]]:
    """Bloc JSON-sérialisable du run : catégories EFFECTIVES (au seuil de la classe
    dans ce run) pour les classes du run qui en ont. ``None`` si aucune."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for classe in classes:
        cats = par_classe_modele.get(classe)
        if not cats or classe not in seuils_par_classe:
            continue
        eff = categories_effectives(cats, float(seuils_par_classe[classe]))
        if eff:
            out[classe] = [c.to_dict() for c in eff]
    if not out:
        return None
    return {"modele": str(modele), "provenance": str(provenance or ""), "par_classe": out}


def categories_du_run(bloc: Any, classe: str) -> Tuple[Categorie, ...]:
    """Catégories d'une classe depuis le bloc de run (dicts JSON) ; ``()`` si absentes."""
    if not isinstance(bloc, Mapping):
        return ()
    liste = (bloc.get("par_classe") or {}).get(classe)
    if not isinstance(liste, list):
        return ()
    cats = [c for c in (_parse_categorie(x) for x in liste) if c is not None]
    return tuple(sorted(cats, key=lambda c: c.seuil))


def sidecar_path(gpkg_path: Any) -> Path:
    return Path(gpkg_path).parent / SIDECAR_NAME


def write_sidecar(gpkg_path: Any, layer_name: str, entree: Mapping[str, Any]) -> Path:
    """Fusionne ``entree`` sous la clé ``layer_name`` dans ``fiabilite.json`` (à côté du
    GeoPackage). ``entree`` = {"classe", "modele", "provenance", "categories": [dicts]}."""
    p = sidecar_path(gpkg_path)
    data: Dict[str, Any] = {}
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            data = {}
    data[str(layer_name)] = dict(entree)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def read_sidecar(gpkg_path: Any, layer_name: str) -> Optional[Dict[str, Any]]:
    """Entrée du sidecar pour une couche, ou ``None`` (absent, illisible, couche inconnue)."""
    p = sidecar_path(gpkg_path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entree = data.get(str(layer_name)) if isinstance(data, Mapping) else None
    return dict(entree) if isinstance(entree, Mapping) else None


def categories_sidecar(entree: Optional[Mapping[str, Any]]) -> Tuple[Categorie, ...]:
    if not isinstance(entree, Mapping):
        return ()
    cats = [c for c in (_parse_categorie(x) for x in (entree.get("categories") or [])) if c is not None]
    return tuple(sorted(cats, key=lambda c: c.seuil))
