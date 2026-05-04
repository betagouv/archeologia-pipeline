"""Stratégie d'acquisition des LAZ pour ``IgnOrLocalRunner``.

Le runner partage 80 % de sa logique entre les modes ``ign_laz`` et
``local_laz`` : seule la **phase d'acquisition** des LAZ et le **plan
de progression** affiché à l'utilisateur diffèrent. Encapsuler ces
différences dans une stratégie permet au runner de devenir inconscient
du mode après l'appel à :meth:`InputStrategy.acquire` — et facilite
l'ajout d'une 3e source LiDAR (drone, photogrammétrie…) en
implémentant une nouvelle stratégie.

Le plan de progression est intentionnellement préservé bit-pour-bit :

- Mode IGN : 0–25 % téléchargement, 25–35 % fusion, 35–95 % produits.
- Mode local : 0 % au démarrage, échelle absolue 0–100 % pour la
  boucle des produits (pas de phase téléchargement à représenter).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

from ..structured_logger import log_section
from ..user_narrator import create_user_narrator

if TYPE_CHECKING:
    from ..cancel_token import CancelToken
    from ..progress_reporter import ProgressReporter
    from ..run_context import ProcessingConfig, RunContext
    from ..structured_logger import StructuredLogger


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

    DOWNLOAD_RANGE = (0, 25)
    MERGE_RANGE = (25, 35)
    PRODUCTS_RANGE = (35, 95)

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
            reporter.stage("Identification des dalles à télécharger")
            reporter.progress(self.DOWNLOAD_RANGE[0])
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

        log_section("TÉLÉCHARGEMENT DES DALLES IGN", "download", slog=slog, reporter=reporter)
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
        return download_ign_dalles(
            input_file=input_path,
            output_dir=ctx.output_dir,
            log=lambda m: reporter.info(m),
            progress=lambda p: reporter.progress(
                int(self.DOWNLOAD_RANGE[0] + (self.DOWNLOAD_RANGE[1] - self.DOWNLOAD_RANGE[0]) * (int(p) / 100.0))
            ),
            stage=lambda s: reporter.stage(str(s)),
            cancel=lambda: cancel.is_cancelled(),
            max_workers=processing.max_workers,
        )

    def merge_progress_start(self) -> int:
        return self.MERGE_RANGE[0]

    def merge_progress_end(self) -> Optional[int]:
        return self.MERGE_RANGE[1]

    def products_progress_start(self) -> int:
        return self.PRODUCTS_RANGE[0]

    def products_progress_for_tile(self, i: int, total: int) -> int:
        frac = i / max(1, total)
        lo, hi = self.PRODUCTS_RANGE
        return int(round(lo + (hi - lo) * frac))


# ----------------------------------------------------------------------
# Mode local : indexation d'un dossier de LAZ déjà présents sur disque
# ----------------------------------------------------------------------
class LocalLazStrategy:
    """Mode ``local_laz`` : indexation d'un dossier LAZ local."""

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
        reporter.progress(0)
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
        return 0

    def merge_progress_end(self) -> Optional[int]:
        # Mode local : pas de marqueur de fin de fusion (la barre reste
        # à 0 jusqu'au début de la phase produits, qui démarre aussi
        # à 0 mais en échelle absolue 0–100).
        return None

    def products_progress_start(self) -> int:
        return 0

    def products_progress_for_tile(self, i: int, total: int) -> int:
        return int(round(100.0 * i / max(1, total)))


def select_input_strategy(mode: str) -> InputStrategy:
    """Retourne la stratégie correspondant au mode du run.

    Lève :class:`ValueError` pour un mode inconnu.
    """
    if mode == "ign_laz":
        return IgnDownloadStrategy()
    if mode == "local_laz":
        return LocalLazStrategy()
    raise ValueError(f"Mode d'acquisition LAZ inconnu : {mode!r}")
