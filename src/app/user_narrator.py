"""Narrateur haut-niveau pour l'utilisateur final non expert.

Le pipeline émet beaucoup de logs techniques (commandes PDAL, paramètres
RVT, lambdas internes) qui sont précieux pour le développeur mais
indéchiffrables pour un archéologue qui veut juste savoir où on en est.

Ce module centralise les **événements narratifs** : un nombre fini de
phrases courtes et claires, traduites de l'état métier du pipeline. Le
contrat est simple : chaque méthode publique correspond à un moment
identifiable du pipeline ("téléchargement démarré", "12 dalles
trouvées", "détection terminée"). Le wording évite le jargon technique
(pas de "RVT", "PDAL", "ONNX" — on parle d'"indices visuels", de
"préparation", de "détection automatique").

Le narrateur écrit via ``reporter.user_info()`` / ``user_success()`` /
``user_warning()`` : ces messages sont visibles dans la fenêtre QGIS
**et** dans le fichier de log, alors que les ``reporter.info()``
techniques restent fichier-only après filtrage.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .progress_reporter import ProgressReporter


# Libellés métier des produits RVT — évite "M_HS", "SVF" pour
# l'utilisateur non technique. Le code interne garde les codes courts.
PRODUCT_LABELS = {
    "MNT": "modèle de terrain (MNT)",
    "DENSITE": "carte de densité",
    "M_HS": "ombrage multi-directionnel",
    "SVF": "facteur de vue du ciel (SVF)",
    "SLO": "carte des pentes",
    "LD": "détection des dépressions locales",
    "SLRM": "résidu local (SLRM)",
    "VAT": "visualisation pour l'archéologie (VAT)",
}


def _human_count(n: int, singular: str, plural: Optional[str] = None) -> str:
    """Renvoie ``"1 dalle"`` ou ``"5 dalles"`` selon ``n``."""
    if n == 1:
        return f"1 {singular}"
    return f"{n} {plural or (singular + 's')}"


def _format_duration(seconds: float) -> str:
    """Format humain d'une durée (en secondes)."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}min {secs:02d}s" if secs else f"{mins}min"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins:02d}min"


class UserNarrator:
    """Émet des messages narratifs aux moments-clés du pipeline.

    Garder cette classe **petite et concrète** : pas d'API générique,
    une méthode par événement métier. Si un nouveau message s'avère
    utile, ajouter une méthode dédiée plutôt qu'un ``log(template,
    **kwargs)`` qui ré-éparpillerait le wording.
    """

    def __init__(self, reporter: "ProgressReporter"):
        self._r = reporter
        self._pipeline_started_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Cycle de vie global
    # ------------------------------------------------------------------
    def pipeline_starting(self, mode_label: str) -> None:
        self._pipeline_started_at = time.time()
        self._r.user_info(f"▶ Démarrage du traitement ({mode_label})")

    def pipeline_complete(
        self,
        *,
        tiles_processed: int = 0,
        products: Optional[list] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Annonce la fin du pipeline.

        ``start_time`` peut être passé explicitement quand le narrateur
        utilisé n'est pas celui qui a vu ``pipeline_starting`` (cas
        typique : narrateur instancié dans :mod:`finalize_service`).
        """
        if start_time is not None:
            duration = _format_duration(time.time() - start_time)
        else:
            duration = self._elapsed_str()
        parts = [f"✅ Traitement terminé en {duration}"]
        if tiles_processed > 0:
            parts.append(_human_count(tiles_processed, "dalle traitée", "dalles traitées"))
        if products:
            labels = [PRODUCT_LABELS.get(p, p) for p in products]
            parts.append("produits : " + ", ".join(labels))
        self._r.user_success(" — ".join(parts))

    def pipeline_failed(self, message: str) -> None:
        duration = self._elapsed_str()
        self._r.user_warning(
            f"⚠ Le traitement s'est interrompu après {duration} : {message}. "
            f"Voir le journal détaillé dans le dossier de sortie pour plus d'informations."
        )

    def pipeline_cancelled(self) -> None:
        self._r.user_info("⏹ Traitement annulé par l'utilisateur")

    # ------------------------------------------------------------------
    # Préflight
    # ------------------------------------------------------------------
    def preflight_ok(self) -> None:
        self._r.user_info("✓ Vérifications préalables OK")

    def preflight_failed(self) -> None:
        self._r.user_warning(
            "✗ Les vérifications préalables ont échoué. "
            "Vérifiez les outils installés et les fichiers d'entrée."
        )

    # ------------------------------------------------------------------
    # Acquisition LAZ (mode IGN ou local)
    # ------------------------------------------------------------------
    def tiles_resolution_start(self) -> None:
        self._r.user_info("🔎 Identification des dalles à télécharger…")

    def tiles_resolution_done(self, n_tiles: int) -> None:
        self._r.user_info(
            f"📍 {_human_count(n_tiles, 'dalle identifiée', 'dalles identifiées')} pour la zone"
        )

    def download_start(self, n_tiles: int) -> None:
        self._r.user_info(
            f"📥 Téléchargement de {_human_count(n_tiles, 'dalle', 'dalles')} IGN…"
        )

    def download_done(self, n_downloaded: int) -> None:
        self._r.user_info(
            f"📥 {_human_count(n_downloaded, 'dalle téléchargée', 'dalles téléchargées')}"
        )

    def local_laz_indexed(self, n_tiles: int) -> None:
        self._r.user_info(
            f"📂 {_human_count(n_tiles, 'dalle locale trouvée', 'dalles locales trouvées')}"
        )

    # ------------------------------------------------------------------
    # Préparation et calcul des produits
    # ------------------------------------------------------------------
    def merging_start(self) -> None:
        self._r.user_info("🧩 Fusion des dalles avec leurs voisines…")

    def products_phase_start(self, total_tiles: int, active_products: list) -> None:
        labels = [PRODUCT_LABELS.get(p, p) for p in active_products if p != "MNT"]
        if labels:
            details = " (MNT + " + ", ".join(labels) + ")"
        else:
            details = " (MNT seul)"
        self._r.user_info(
            f"🛠 Calcul des produits sur "
            f"{_human_count(total_tiles, 'dalle', 'dalles')}{details}…"
        )

    def tile_progress(self, index: int, total: int, tile_name: str) -> None:
        # Tronqué pour rester lisible (les noms IGN sont longs).
        short = tile_name if len(tile_name) <= 30 else tile_name[:27] + "…"
        self._r.user_info(f"   • Dalle {index}/{total} : {short}")

    # ------------------------------------------------------------------
    # Computer Vision
    # ------------------------------------------------------------------
    def cv_start(self, n_runs: int) -> None:
        if n_runs == 1:
            self._r.user_info("🤖 Détection automatique en cours…")
        else:
            self._r.user_info(
                f"🤖 Détection automatique en cours ({n_runs} modèles configurés)…"
            )

    def cv_run_start(self, run_idx: int, total: int, model_name: str, target_rvt: str) -> None:
        rvt_label = PRODUCT_LABELS.get(target_rvt, target_rvt)
        self._r.user_info(
            f"   • Modèle {run_idx}/{total} : « {model_name} » sur {rvt_label}"
        )

    def cv_complete(self, n_detections: int) -> None:
        if n_detections == 0:
            self._r.user_info("🤖 Détection terminée : aucune zone d'intérêt détectée")
        else:
            self._r.user_info(
                f"🤖 Détection terminée : "
                f"{_human_count(n_detections, 'zone détectée', 'zones détectées')}"
            )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------
    def finalize_start(self) -> None:
        self._r.user_info("📦 Assemblage des résultats finaux (mosaïque, projet QGIS)…")

    def layers_loaded(self, n_rasters: int, n_vectors: int) -> None:
        parts = []
        if n_rasters:
            parts.append(_human_count(n_rasters, "couche raster", "couches raster"))
        if n_vectors:
            parts.append(_human_count(n_vectors, "couche vecteur", "couches vecteur"))
        if parts:
            self._r.user_info("🗺 " + " et ".join(parts) + " ajoutées au projet QGIS")

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _elapsed_str(self) -> str:
        if self._pipeline_started_at is None:
            return "0s"
        return _format_duration(time.time() - self._pipeline_started_at)


def create_user_narrator(reporter: "ProgressReporter") -> UserNarrator:
    return UserNarrator(reporter)
