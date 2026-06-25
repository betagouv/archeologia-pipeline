"""Stratégie d'acquisition des LAZ pour ``IgnOrLocalRunner``.

Le runner partage 80 % de sa logique entre les modes ``ign_laz`` et
``local_laz`` : seule la **phase d'acquisition** des LAZ et le **plan
de progression** affiché à l'utilisateur diffèrent. Encapsuler ces
différences dans une stratégie permet au runner de devenir inconscient
du mode après l'appel à :meth:`InputStrategy.acquire` — et facilite
l'ajout d'une 3e source LiDAR (drone, photogrammétrie…) en
implémentant une nouvelle stratégie.

Le plan de progression n'est plus codé en dur ici : il est porté par un
:class:`~app.runners.progress_plan.ProgressPlan` (construit par le runner à
partir du mode + activation CV) et lu via ses bandes ``download`` / ``merge``
/ ``products``. Cela garantit des bandes contiguës avec la phase CV et la
finalisation (plus de recul de la barre au démarrage de la CV).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

from ..progress_reporter import report_stage_id
from ..progress_stages import Stage
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator

if TYPE_CHECKING:
    from ..cancel_token import CancelToken
    from ..progress_reporter import ProgressReporter
    from ..run_context import ProcessingConfig, RunContext
    from ..structured_logger import StructuredLogger
    from .progress_plan import ProgressPlan


def persist_resolved_dalles_list(input_file: Path, output_dir: Path) -> Path:
    """Recopie une liste de dalles déjà résolue vers ``<output_dir>/dalles_urls.txt``.

    L'entrée d'un run ``ign_laz`` peut être une liste ``.txt`` (``nom,url``) issue de
    la **sélection des dalles sur la carte** : elle vit dans un dossier temporaire
    (``data/temp_zones/``) écrasé à chaque sélection. En la persistant dans le dossier
    de sortie **dès le début du run** (avant tout téléchargement), un run interrompu
    laisse une liste réutilisable pour reprendre — au **même emplacement** que la
    branche polygone (``resolve_tiles_from_polygon`` écrit aussi ``dalles_urls.txt``).

    No-op si la source EST déjà la destination (re-run pointant directement sur ce
    fichier) — ``shutil`` lèverait sinon ``SameFileError``. Retourne le chemin
    persistant.
    """
    dest = output_dir / "dalles_urls.txt"
    try:
        same = input_file.resolve() == dest.resolve()
    except OSError:  # chemin illisible : on tente la copie quand même
        same = False
    if not same:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_file, dest)
    return dest


class AcquireResult(Protocol):
    """Résultat d'une acquisition de LAZ.

    Les champs ``sorted_list_file`` et ``dalles_dir`` sont consommés par
    :func:`pipeline.ign.preprocess.prepare_merged_tiles` indépendamment
    du mode.
    """

    sorted_list_file: Path
    dalles_dir: Path


class InputStrategy(Protocol):
    """Stratégie d'acquisition + plan de progression associé."""

    def acquire(
        self,
        *,
        ctx: "RunContext",
        reporter: "ProgressReporter",
        cancel: "CancelToken",
        slog: Optional["StructuredLogger"],
        processing: "ProcessingConfig",
    ) -> Optional[AcquireResult]:
        """Récupère les LAZ et retourne ``(sorted_list_file, dalles_dir)``.

        Retourne ``None`` si l'acquisition échoue (un message d'erreur
        a déjà été émis via ``reporter.error``).
        """

    def merge_progress_start(self) -> int: ...

    def merge_progress_end(self) -> Optional[int]:
        """Borne de fin pour la phase fusion. ``None`` = aucune mise à
        jour de progression à émettre (mode local : la fusion est
        rapide et la barre reste à 0 jusqu'au début des produits)."""

    def products_progress_start(self) -> int: ...

    def products_progress_for_tile(self, i: int, total: int) -> int:
        """Pourcentage à afficher après la dalle ``i`` (1-indexed) sur
        ``total`` dalles."""


# ----------------------------------------------------------------------
# Mode IGN : téléchargement des dalles via les URLs/zone du config
# ----------------------------------------------------------------------
class IgnDownloadStrategy:
    """Mode ``ign_laz`` : résolution des dalles + téléchargement IGN."""

    def __init__(self, plan: "ProgressPlan"):
        self._plan = plan

    def acquire(
        self,
        *,
        ctx: "RunContext",
        reporter: "ProgressReporter",
        cancel: "CancelToken",
        slog: Optional["StructuredLogger"],
        processing: "ProcessingConfig",
    ) -> Optional[AcquireResult]:
        from ...pipeline.ign.downloader import download_ign_dalles

        narrator = create_user_narrator(reporter)

        # Pré-conditions garanties par validate_run_context (V3.3) :
        # input_file non-null, existant, et ctx.output_dir non-null.
        input_path = ctx.files.input_file
        assert input_path is not None
        assert ctx.output_dir is not None

        # Détection du type d'entrée : shapefile/geojson → résolution des dalles
        is_vector = input_path.suffix.lower() in (".shp", ".geojson", ".json", ".gpkg")
        if is_vector:
            from ...pipeline.ign.tile_resolver import resolve_tiles_from_polygon

            log_section("RÉSOLUTION DES DALLES IGN", "download", slog=slog, reporter=reporter)
            report_stage_id(reporter, Stage.DOWNLOAD)
            reporter.stage("Identification des dalles à télécharger")
            reporter.progress(self._plan.download[0])
            narrator.tiles_resolution_start()

            urls_file = ctx.output_dir / "dalles_urls.txt"
            n_tiles = resolve_tiles_from_polygon(
                polygon_path=input_path,
                output_file=urls_file,
                log=lambda m: reporter.info(m),
                cancel=lambda: cancel.is_cancelled(),
            )
            if n_tiles == 0:
                reporter.error("Aucune dalle IGN trouvée pour la zone sélectionnée")
                return None
            narrator.tiles_resolution_done(n_tiles)
            input_path = urls_file
        else:
            # Entrée déjà résolue (.txt : sélection sur carte ou liste pré-établie).
            # On la persiste dans le dossier de sortie DÈS LE DÉBUT (avant tout
            # téléchargement) : un run interrompu laisse alors une liste réutilisable
            # pour reprendre (la source vit dans un dossier temporaire écrasable).
            try:
                input_path = persist_resolved_dalles_list(input_path, ctx.output_dir)
                reporter.info(f"Liste des dalles enregistrée: {input_path.name}")
            except Exception as e:  # noqa: BLE001 — best-effort, ne bloque pas le run
                reporter.info(f"Liste des dalles non persistée ({e}) — téléchargement direct")

        log_section("TÉLÉCHARGEMENT DES DALLES IGN", "download", slog=slog, reporter=reporter)
        report_stage_id(reporter, Stage.DOWNLOAD)
        reporter.stage("Téléchargement des dalles")
        # Compte les URLs à télécharger pour informer l'utilisateur.
        try:
            n_to_download = sum(
                1 for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        except Exception:
            n_to_download = 0
        if n_to_download:
            narrator.download_start(n_to_download)

        def _on_dalle_downloaded(i: int, n: int, filename: str, success: bool) -> None:
            # Nom court : on ne veut pas du suffixe ``.copc.laz`` dans le
            # journal — la troncature est gérée côté narrator.
            base = filename.replace(".copc.laz", "").replace(".laz", "")
            narrator.download_tile_progress(i, n, base, success)

        return download_ign_dalles(
            input_file=input_path,
            output_dir=ctx.output_dir,
            log=lambda m: reporter.info(m),
            progress=lambda p: reporter.progress(
                self._plan.at(self._plan.download, int(p) / 100.0)
            ),
            stage=lambda s: reporter.stage(str(s)),
            cancel=lambda: cancel.is_cancelled(),
            max_workers=processing.max_workers,
            on_tile_done=_on_dalle_downloaded,
        )

    def merge_progress_start(self) -> int:
        return self._plan.merge[0]

    def merge_progress_end(self) -> Optional[int]:
        return self._plan.merge[1]

    def products_progress_start(self) -> int:
        return self._plan.products[0]

    def products_progress_for_tile(self, i: int, total: int) -> int:
        return self._plan.at(self._plan.products, i / max(1, total))


# ----------------------------------------------------------------------
# Mode local : indexation d'un dossier de LAZ déjà présents sur disque
# ----------------------------------------------------------------------
class LocalLazStrategy:
    """Mode ``local_laz`` : indexation d'un dossier LAZ local."""

    def __init__(self, plan: "ProgressPlan"):
        self._plan = plan

    def acquire(
        self,
        *,
        ctx: "RunContext",
        reporter: "ProgressReporter",
        cancel: "CancelToken",
        slog: Optional["StructuredLogger"],
        processing: "ProcessingConfig",
    ) -> Optional[AcquireResult]:
        from ...pipeline.modes.local_laz import run_local_laz

        narrator = create_user_narrator(reporter)
        # Pré-conditions garanties par validate_run_context (V3.3).
        local_dir = ctx.files.local_laz_dir
        assert local_dir is not None
        assert ctx.output_dir is not None

        log_section("INDEXATION DES NUAGES LOCAUX", "download", slog=slog, reporter=reporter)
        reporter.stage("Indexation des nuages locaux")
        reporter.progress(self._plan.products[0])
        result = run_local_laz(
            local_laz_dir=local_dir,
            output_dir=ctx.output_dir,
            log=lambda m: reporter.info(m),
        )
        if result is not None:
            try:
                with result.sorted_list_file.open("r", encoding="utf-8") as f:
                    n_tiles = sum(1 for line in f if line.strip())
            except Exception:
                n_tiles = 0
            if n_tiles:
                narrator.local_laz_indexed(n_tiles)
        return result

    def merge_progress_start(self) -> int:
        return self._plan.merge[0]

    def merge_progress_end(self) -> Optional[int]:
        # Mode local : la fusion est rapide et sa bande est dégénérée
        # (``merge == (0, 0)``) ; on ne pousse aucune mise à jour pour ne pas
        # ré-émettre 0. La barre reste à 0 jusqu'au début des produits.
        return None

    def products_progress_start(self) -> int:
        return self._plan.products[0]

    def products_progress_for_tile(self, i: int, total: int) -> int:
        return self._plan.at(self._plan.products, i / max(1, total))


def select_input_strategy(mode: str, plan: "ProgressPlan") -> InputStrategy:
    """Retourne la stratégie correspondant au mode du run.

    ``plan`` fournit les bandes de progression (contiguës avec la phase CV /
    finalisation). Lève :class:`ValueError` pour un mode inconnu.
    """
    if mode == "ign_laz":
        return IgnDownloadStrategy(plan)
    if mode == "local_laz":
        return LocalLazStrategy(plan)
    raise ValueError(f"Mode d'acquisition LAZ inconnu : {mode!r}")
