"""Modes de données + stades de la frise + validation de chemin (pur, testable).

La frise de l'étape 1 du wizard V2 EST le sélecteur de mode : chaque stade
cliquable fixe le point d'entrée du pipeline. Les stades AVANT le point d'entrée
sont « sautés », celui d'entrée est mis en avant, ceux d'après sont « exécutés ».

Ce module centralise (pour que l'UI ne réécrive pas la logique) :
- ``DATA_MODES`` : par ``data_mode``, la clé de config de la source, son type
  (fichier/dossier), les libellés du bandeau et du champ, le stade d'entrée.
- ``PIPELINE_STAGES`` : les 5 stades de la frise (4 cliquables + Détection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModeInfo:
    mode: str
    icon: str             # nom d'icône SVG (theme/icons/<icon>.svg)
    entry_stage: int      # id du stade d'entrée dans la frise (1..4)
    config_key: str       # clé dans app.files.* portant la source
    is_file: bool         # True = fichier attendu, False = dossier
    source_label: str     # libellé du champ source (étape 1)
    placeholder: str
    banner_label: str     # titre (gras) du bandeau
    banner_sub: str       # complément (atténué) du bandeau
    description: str       # phrase explicative (bandeau)
    valid_exts: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StageInfo:
    id: int               # 1..5, ordre de la frise
    icon: str
    label: str
    sub: str
    mode: Optional[str]   # None = stade non cliquable (Détection IA)
    optional: bool = False


DATA_MODES: Dict[str, ModeInfo] = {
    "ign_laz": ModeInfo(
        mode="ign_laz",
        icon="download",
        entry_stage=1,
        config_key="input_file",
        is_file=True,
        source_label="Zone d'étude, liste de dalles ou dalles choisies sur la carte",
        placeholder="fichier.shp/.gpkg (polygone ou points) ou liste de dalles .txt",
        banner_label="Téléchargement IGN",
        banner_sub="à partir d'un polygone ou de points",
        description="Le plugin télécharge les dalles LiDAR HD de l'IGN, puis calcule tout : "
        "le MNT, les images du relief et, si vous le demandez, les structures.",
        valid_exts=(".shp", ".dbf", ".geojson", ".json", ".gpkg", ".txt"),
    ),
    "local_laz": ModeInfo(
        mode="local_laz",
        icon="pointcloud",
        entry_stage=2,
        config_key="local_laz_dir",
        is_file=False,
        source_label="Dossier des fichiers LAZ/LAS",
        placeholder="C:/data/laz/",
        banner_label="Nuages locaux",
        banner_sub="LAZ / LAS sur disque",
        description="Vous avez déjà les nuages de points : le traitement calcule le MNT, "
        "les images du relief et cherche les structures.",
    ),
    "existing_mnt": ModeInfo(
        mode="existing_mnt",
        icon="raster",
        entry_stage=3,
        config_key="existing_mnt_dir",
        is_file=False,
        source_label="Dossier des MNT",
        placeholder="C:/data/mnt/",
        banner_label="MNT existant",
        banner_sub="TIF / ASC",
        description="Vous fournissez un MNT déjà calculé : le traitement calcule les "
        "images du relief puis cherche les structures.",
    ),
    "existing_rvt": ModeInfo(
        mode="existing_rvt",
        icon="indices",
        entry_stage=4,
        config_key="existing_rvt_dir",
        is_file=False,
        source_label="Dossier des images du relief",
        placeholder="C:/data/indices/",
        banner_label="Images du relief existantes",
        banner_sub="TIF (SVF, LD, M-HS…)",
        description="Vous avez déjà les images du relief : ce point de départ n'a de sens "
        "que pour y chercher des structures.",
    ),
}

PIPELINE_STAGES: List[StageInfo] = [
    StageInfo(1, "download", "Dalles IGN à télécharger", "connexion Internet", "ign_laz"),
    StageInfo(2, "pointcloud", "Nuages LiDAR", "fichiers LAZ ou LAS", "local_laz"),
    StageInfo(3, "raster", "MNT existant", "TIF en Lambert-93", "existing_mnt"),
    StageInfo(4, "indices", "Images du relief existantes", "TIF en Lambert-93", "existing_rvt"),
    StageInfo(5, "detection", "Structures à chercher", "facultatif", None, optional=True),
]

_ORDER = ["ign_laz", "local_laz", "existing_mnt", "existing_rvt"]


def ordered_modes() -> List[str]:
    """Modes dans l'ordre de la frise (gauche → droite)."""
    return list(_ORDER)


def mode_info(mode: str) -> ModeInfo:
    """Métadonnées d'un mode ; retombe sur ``ign_laz`` si inconnu."""
    return DATA_MODES.get(mode, DATA_MODES["ign_laz"])


def pipeline_stages() -> List[StageInfo]:
    """Les 5 stades de la frise, dans l'ordre."""
    return list(PIPELINE_STAGES)


def path_state(
    text: str,
    *,
    expect_dir: bool,
    allow_create: bool = False,
    valid_exts: Tuple[str, ...] = (),
) -> str:
    """État d'un chemin : ``"ok"`` | ``"warn"`` | ``"error"`` (vide → ``"ok"``).

    - **dossier** : existe → ok ; inexistant + ``allow_create`` → ok ; sinon error.
    - **fichier** : inexistant → error ; existe mais extension hors ``valid_exts``
      → warn (extension inattendue) ; sinon ok.
    """
    text = (text or "").strip()
    if not text:
        return "ok"
    p = Path(text)
    if expect_dir:
        if p.is_dir():
            return "ok"
        if allow_create and not p.exists():
            return "ok"
        return "error"
    if not p.is_file():
        return "error"
    if valid_exts and p.suffix.lower() not in valid_exts:
        return "warn"
    return "ok"


def normalize_vector_input(path: Path) -> Path:
    """Un ``.dbf`` désigné à la place du shapefile → le ``.shp`` voisin.

    Le ``.dbf`` seul ne porte pas la géométrie ; sans ``.shp`` voisin le
    chemin est rendu tel quel (``validate_run_context`` signale l'erreur).
    """
    if path.suffix.lower() == ".dbf":
        shp = path.with_suffix(".shp")
        if shp.exists():
            return shp
    return path


def path_is_valid(text: str, *, expect_dir: bool, allow_create: bool = False) -> bool:
    """Compat : ``True`` si le chemin n'est pas en erreur (ok ou warn)."""
    return path_state(text, expect_dir=expect_dir, allow_create=allow_create) != "error"
