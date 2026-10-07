"""Aide intégrée — module PUR (sans Qt) : chapitres Markdown, sommaire, ancres, nouveautés.

Le manuel vit dans ``aide/<nn>-<cle>.md`` à la racine du plugin, **livré dans le ZIP**
(à la différence de ``docs/``, doc développeur exclue du paquet). Un chapitre = un
fichier : son titre est la première ligne ``# ``, sa clé le nom de fichier sans
préfixe numérique ni extension (``03-etape-1-source.md`` → ``etape-1-source``). Le
chapitre « Nouveautés » n'est pas un fichier : il est rendu depuis le ``changelog=``
de ``metadata.txt``, déjà écrit à chaque version — une seule source.

Qt ne pose pas d'ancre sur les titres Markdown : la fenêtre navigue par **bloc de
titre** (``ui/dialogs/aide_dialog``) et ce module fournit l'identifiant stable d'un
titre, :func:`slug`, que les liens internes emploient :
``[texte](etape-2-produits.md#reglages-avances)`` ou ``[texte](#reglages-avances)``.

``tests/unit/test_aide.py`` tient les contrats : chaque étape a son chapitre, chaque
lien interne et chaque image résolvent, chaque produit et chaque entité sont nommés,
aucune citation anglaise, aucune version écrite à la main.
"""
from __future__ import annotations

import configparser
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

AIDE_DIRNAME = "aide"
CLE_NOUVEAUTES = "nouveautes"

#: Étape de l'assistant → clé du chapitre ouvert par le « ? » (et F1).
CHAPITRE_PAR_ETAPE: Dict[int, str] = {
    1: "etape-1-source",
    2: "etape-2-produits",
    3: "etape-3-detection",
    4: "etape-4-lancer",
}

_TITRE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_LIEN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_VERSION_CHANGELOG = re.compile(r"^(\d+\.\d+\.\d+)\s*(\(.*\))?\s*$")
# Le changelog est du texte brut : « archeologia.<version>.zip » y est une balise
# HTML pour le rendu Markdown (tout ce qui suit disparaît), « max_workers » une
# italique. On échappe la ponctuation que CommonMark interprète.
_PONCTUATION_MD = re.compile(r"([\\*_<>\[\]{}#|~])")


def echapper_markdown(texte: str) -> str:
    """Texte brut → texte Markdown qui s'affiche tel quel."""
    return _PONCTUATION_MD.sub(r"\\\1", texte)


def slug(titre: str) -> str:
    """Identifiant stable d'un titre : sans accents, minuscules, tirets.

    ``« Réglages avancés… »`` → ``reglages-avances``. C'est ce que les liens
    internes écrivent après ``#`` et ce que la fenêtre compare au texte du bloc.
    """
    sans_accents = unicodedata.normalize("NFKD", titre).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sans_accents.lower()).strip("-")


def sections(markdown: str) -> List[Tuple[int, str, str]]:
    """``(niveau, titre, slug)`` de chaque titre ``#``/``##``/``###``, hors blocs de code."""
    out: List[Tuple[int, str, str]] = []
    dans_code = False
    for ligne in markdown.splitlines():
        if ligne.lstrip().startswith("```"):
            dans_code = not dans_code
            continue
        if dans_code:
            continue
        m = _TITRE.match(ligne)
        if m:
            titre = m.group(2).strip()
            out.append((len(m.group(1)), titre, slug(titre)))
    return out


def cle_du_fichier(chemin: Path) -> str:
    return re.sub(r"^\d+-", "", Path(chemin).stem)


@dataclass(frozen=True)
class Chapitre:
    cle: str
    titre: str
    markdown: str

    @property
    def sections(self) -> List[Tuple[int, str, str]]:
        return sections(self.markdown)

    def a_l_ancre(self, ancre: str) -> bool:
        return any(s == ancre for _n, _t, s in self.sections)


def charger_chapitres(dossier: Path, metadata_path: Optional[Path] = None) -> List[Chapitre]:
    """Les chapitres d'``aide/`` dans l'ordre des fichiers, puis « Nouveautés ».

    Dossier absent → liste vide (le plugin reste utilisable, le manuel dit
    simplement qu'il n'a rien à montrer). ``metadata_path`` absent ou illisible →
    pas de chapitre Nouveautés, jamais d'exception.
    """
    dossier = Path(dossier)
    chapitres: List[Chapitre] = []
    if dossier.is_dir():
        for f in sorted(dossier.glob("*.md")):
            texte = f.read_text(encoding="utf-8")
            titre = next((t for n, t, _s in sections(texte) if n == 1), f.stem)
            chapitres.append(Chapitre(cle_du_fichier(f), titre, texte))
    if metadata_path is not None:
        nouveautes = nouveautes_markdown(metadata_path)
        if nouveautes:
            chapitres.append(Chapitre(CLE_NOUVEAUTES, "Nouveautés", nouveautes))
    return chapitres


def nouveautes_markdown(metadata_path: Path) -> str:
    """Le ``changelog=`` de ``metadata.txt`` rendu en Markdown (``## version`` + puces).

    Le champ est écrit à chaque livraison pour le gestionnaire d'extensions de
    QGIS : le manuel le relit tel quel plutôt que d'en tenir une seconde copie.
    """
    try:
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(metadata_path, encoding="utf-8"):
            return ""
        brut = parser.get("general", "changelog", fallback="")
    except (configparser.Error, OSError):
        return ""
    lignes = ["# Nouveautés", ""]
    for ligne in brut.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        m = _VERSION_CHANGELOG.match(ligne)
        if m:
            lignes.append("")
            lignes.append(f"## {m.group(1)} {m.group(2) or ''}".rstrip())
            lignes.append("")
        elif ligne.startswith(("*", "-")):
            lignes.append("- " + echapper_markdown(ligne[1:].strip()))
        else:
            lignes.append(echapper_markdown(ligne))
    return "\n".join(lignes).rstrip() + "\n" if len(lignes) > 2 else ""


def est_image(cible: str) -> bool:
    return cible.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))


def liens_internes(markdown: str) -> List[Tuple[str, str]]:
    """``(cle_chapitre, ancre)`` de chaque lien interne ; ``cle`` vide = même chapitre.

    Une image liée à elle-même (``[![…](img/x.png)](img/x.png)``, ouverture en
    taille réelle) n'est pas un lien de chapitre : elle est contrôlée par
    :func:`images`.
    """
    out: List[Tuple[str, str]] = []
    for cible in _LIEN.findall(markdown):
        if re.match(r"^[a-z]+:", cible) or est_image(cible):   # externe, ou image
            continue
        out.append(resoudre_cible(cible))
    return out


def resoudre_cible(cible: str) -> Tuple[str, str]:
    """``'x.md#y'`` → ``('x', 'y')`` ; ``'#y'`` → ``('', 'y')`` ; ``'x.md'`` → ``('x', '')``."""
    fichier, _, ancre = cible.partition("#")
    cle = cle_du_fichier(Path(fichier)) if fichier else ""
    return cle, ancre


def images(markdown: str) -> List[str]:
    return list(_IMAGE.findall(markdown))


def chapitre(chapitres: Sequence[Chapitre], cle: str) -> Optional[Chapitre]:
    return next((c for c in chapitres if c.cle == cle), None)


CLE_DEPANNAGE = "depannage"

#: Message du journal (⚠ / ✗) → titre de la rubrique de Dépannage qui en parle.
#: Motif cherché sans tenir compte de la casse ; le premier qui matche gagne.
#: ``test_aide`` vérifie que chaque titre existe bien dans le chapitre.
_RUBRIQUES_DEPANNAGE: Tuple[Tuple[str, str], ...] = (
    (r"noyau atteint", "Le noyau atteint N pixels"),
    (r"v[ée]rifications pr[ée]alables ont [ée]chou", "Les vérifications préalables ont échoué"),
    (r"timed out|data\.geopf\.fr|proxy|t[ée]l[ée]chargement.*[ée]chou", "Le téléchargement échoue"),
    (r"abandonn", "Une dalle est abandonnée"),
    (r"PDAL a crash|moins de workers|max_workers", "Erreur PDAL ou mémoire"),
    (r"aucune zone d'int[ée]r[êe]t|aucun run", "Aucun run, ou aucune détection"),
    (r"interromp", "Le traitement s'est interrompu"),
)


def rubrique_depannage(message: str) -> str:
    """Titre de la rubrique de Dépannage qui répond à ``message`` ; ``""`` sinon."""
    for motif, titre in _RUBRIQUES_DEPANNAGE:
        if re.search(motif, message, re.IGNORECASE):
            return titre
    return ""
