"""Fiabilité observée sur vos runs — module PUR (sqlite3 et json de la bibliothèque standard).

Le banc d'évaluation donne, par niveau, une part de vrais objets *mesurée sur le
corpus du modèle*. Le terrain dit ce qu'elle vaut ici : les verdicts saisis dans
QGIS dans le champ ``validation`` des GeoPackages de détections — vocabulaire
**fixe**, celui du formulaire du projet de validation : ``oui`` (vrai objet),
``non`` (fausse détection), ``peut-être`` (à revoir, ne compte pas comme vérifié)
— sont agrégés par modèle, classe et niveau sur les **runs connus** du poste.
La fiche de classe affiche alors, à côté de la mesure du banc, la mesure chez
vous : « probable : 72 % au banc, 64 % sur vos 85 vérifications ».

Les runs connus sont un registre local du profil QGIS (``runs_connus.json``,
dossiers de sortie, les 50 derniers), alimenté au lancement de chaque run. Un
GeoPackage est une base SQLite : on y lit ``gpkg_contents`` et les colonnes
``model_name``, ``model_pred``, ``fiabilite``, ``validation``, ``corr_pred`` sans
OGR — lisible hors QGIS, donc testable. ``corr_pred`` renseigné avec une autre
classe fait d'un ``oui`` une fausse détection *pour la classe prédite*.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .fiabilite import LABELS_FR

REGISTRE = "runs_connus.json"
MAX_RUNS = 50
VERDICTS: Tuple[str, ...] = ("oui", "non", "peut-être")
N_MIN_POURCENT = 20        # en dessous, on dit « 3 vraies sur 4 vérifiées », pas un pourcentage


# ----------------------------------------------------------------------
# Registre des runs connus
# ----------------------------------------------------------------------
def registre_path(base_dir: Path) -> Path:
    return Path(base_dir) / REGISTRE


def _lire_registre(base_dir: Path) -> List[dict]:
    p = registre_path(base_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("output_dir")] if isinstance(data, list) else []


def enregistrer_run(base_dir: Path, output_dir: Path) -> Path:
    """Ajoute ``output_dir`` au registre (déplacé en fin s'il y est déjà, les
    ``MAX_RUNS`` derniers conservés). Jamais d'exception : un registre illisible
    repart de zéro."""
    entrees = [d for d in _lire_registre(base_dir) if Path(d["output_dir"]) != Path(output_dir)]
    entrees.append({"output_dir": str(output_dir), "date": _dt.datetime.now().isoformat(timespec="seconds")})
    entrees = entrees[-MAX_RUNS:]
    p = registre_path(base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entrees, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def runs_connus(base_dir: Path) -> List[Path]:
    """Les dossiers de sortie du registre qui existent encore, du plus ancien au plus récent."""
    return [Path(d["output_dir"]) for d in _lire_registre(base_dir) if Path(d["output_dir"]).is_dir()]


# ----------------------------------------------------------------------
# Lecture des verdicts
# ----------------------------------------------------------------------
_COLONNES = ("model_name", "model_pred", "fiabilite", "validation", "corr_pred")


def _couches(con: sqlite3.Connection) -> List[str]:
    try:
        rows = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'").fetchall()
    except sqlite3.Error:
        return []
    return [str(r[0]) for r in rows]


def lire_verdicts_gpkg(gpkg: Path) -> Iterator[Tuple[str, str, str, str, str]]:
    """``(model_name, model_pred, fiabilite, validation, corr_pred)`` de chaque
    détection qui porte un verdict, toutes couches du GeoPackage."""
    try:
        con = sqlite3.connect(f"file:{Path(gpkg).as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        for couche in _couches(con):
            try:
                colonnes = {r[1] for r in con.execute(f'PRAGMA table_info("{couche}")').fetchall()}
                if not {"model_pred", "fiabilite", "validation"} <= colonnes:
                    continue
                champs = ", ".join(f'"{c}"' if c in colonnes else "''" for c in _COLONNES)
                rows = con.execute(
                    f'SELECT {champs} FROM "{couche}" WHERE "validation" IS NOT NULL AND TRIM("validation") != \'\''
                ).fetchall()
            except sqlite3.Error:
                continue
            for r in rows:
                yield tuple(str(x or "").strip() for x in r)  # type: ignore[misc]
    finally:
        con.close()


@dataclass
class Compte:
    verifies: int = 0      # oui + non
    vrais: int = 0         # oui (sans correction de classe)
    a_revoir: int = 0      # peut-être

    @property
    def part_vrais(self) -> Optional[float]:
        return self.vrais / self.verifies if self.verifies else None

    def phrase(self) -> str:
        """« 64 % sur vos 85 vérifications » ou, en dessous de 20, « 3 vraies sur 4 vérifiées »."""
        if not self.verifies:
            return f"{self.a_revoir} à revoir, aucune vérifiée" if self.a_revoir else ""
        if self.verifies >= N_MIN_POURCENT:
            texte = f"{round(self.part_vrais * 100)} % sur vos {self.verifies} vérifications"
        else:
            texte = f"{self.vrais} vraie{'s' if self.vrais > 1 else ''} sur {self.verifies} vérifiée{'s' if self.verifies > 1 else ''}"
        if self.a_revoir:
            texte += f", {self.a_revoir} à revoir"
        return texte


@dataclass
class Observation:
    par_categorie: Dict[str, Compte] = field(default_factory=dict)   # catégorie → compte
    n_runs: int = 0                                                   # runs lus (avec verdicts ou non)

    @property
    def total_verifies(self) -> int:
        return sum(c.verifies for c in self.par_categorie.values())

    @property
    def total_a_revoir(self) -> int:
        return sum(c.a_revoir for c in self.par_categorie.values())


def _ajouter(compte: Compte, validation: str, model_pred: str, corr_pred: str) -> None:
    v = validation.lower()
    if v == "oui":
        compte.verifies += 1
        if not corr_pred or corr_pred == model_pred:
            compte.vrais += 1          # corrigé vers une autre classe : faux pour la classe prédite
    elif v == "non":
        compte.verifies += 1
    elif v in ("peut-être", "peut-etre", "peut être"):
        compte.a_revoir += 1


def agreger(dossiers: Sequence[Path]) -> Dict[Tuple[str, str], Dict[str, Compte]]:
    """``{(modèle, classe): {catégorie: Compte}}`` sur les GeoPackages de
    ``detections/<slug>/<slug>.gpkg`` de chaque dossier."""
    par_label = {lab.lower(): cat for cat, lab in LABELS_FR.items()}
    out: Dict[Tuple[str, str], Dict[str, Compte]] = {}
    for dossier in dossiers:
        for gpkg in sorted(Path(dossier).glob("detections/*/*.gpkg")):
            for modele, classe, fiab, validation, corr in lire_verdicts_gpkg(gpkg):
                cat = par_label.get(fiab.lower())
                if cat is None or not classe:
                    continue
                compte = out.setdefault((modele, classe), {}).setdefault(cat, Compte())
                _ajouter(compte, validation, classe, corr)
    return out


def observer(base_dir: Path, modele: str, classe: str) -> Observation:
    """Ce que vos runs connus disent de ``classe`` du ``modele`` : un compte par
    niveau. Vide (``n_runs`` = 0) sans registre."""
    dossiers = runs_connus(base_dir)
    if not dossiers:
        return Observation()
    comptes = agreger(dossiers).get((modele, classe), {})
    return Observation(par_categorie=comptes, n_runs=len(dossiers))
