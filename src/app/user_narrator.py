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

    def _user_info_transient(self, msg: str, group: str) -> None:
        """Émet une sous-progression dans le canal transient si dispo.

        Le canal "transient" (cf. :meth:`ProgressReporter.user_info_transient`)
        permet à la zone log Qt de réécrire une seule ligne par ``group``
        plutôt que d'ajouter une ligne par appel. Le fichier ``.txt``
        reçoit toujours toutes les lignes (trace complète).

        Si le reporter ne connaît pas le canal transient (test, mock,
        implémentation legacy), on retombe sur ``user_info`` — la sortie
        est alors empilée comme une ligne normale.
        """
        fn = getattr(self._r, "user_info_transient", None)
        if fn is None:
            self._r.user_info(msg)
            return
        try:
            fn(msg, group)
        except Exception:
            self._r.user_info(msg)

    def _metric(self, current: int, total: int, label: str) -> None:
        """Émet un compteur structuré (i/n) si le reporter le supporte.

        Optionnel : un reporter sans ``metric`` (legacy/test) est simplement
        ignoré — le compteur n'est qu'un complément du canal narratif texte.
        """
        fn = getattr(self._r, "metric", None)
        if fn is None:
            return
        try:
            fn(int(current), int(total), str(label))
        except Exception:
            pass

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

    def download_tile_progress(self, index: int, total: int, tile_name: str) -> None:
        """Sous-progression du téléchargement (1 dalle terminée).

        Ligne unique réécrite à chaque appel dans la zone log Qt — voir
        :meth:`_user_info_transient`.
        """
        short = tile_name if len(tile_name) <= 30 else tile_name[:27] + "…"
        self._user_info_transient(
            f"   • Dalle {index}/{total} téléchargée : {short}",
            group="download_tile_progress",
        )
        self._metric(index, total, "dalles")

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

    def merging_tile_progress(self, index: int, total: int, tile_name: str) -> None:
        """Sous-progression de la fusion : 1 dalle fusionnée.

        Ligne unique réécrite à chaque dalle (canal transient).
        """
        short = tile_name if len(tile_name) <= 30 else tile_name[:27] + "…"
        suffix = f" : {short}" if short else ""
        self._user_info_transient(
            f"   • Dalle {index}/{total} fusionnée{suffix}",
            group="merging_tile_progress",
        )
        self._metric(index, total, "dalles")

    def products_phase_start(self, total_tiles: int, active_products: list) -> None:
        codes = [p for p in active_products if p != "MNT"]
        if codes:
            details = " (MNT + " + ", ".join(codes) + ")"
        else:
            details = " (MNT seul)"
        self._r.user_info(
            f"🛠 Calcul des produits sur "
            f"{_human_count(total_tiles, 'dalle', 'dalles')}{details}…"
        )
        self._metric(0, total_tiles, "dalles")  # initialise le total de la phase

    def tile_progress(self, index: int, total: int, tile_name: str) -> None:
        # Tronqué pour rester lisible (les noms IGN sont longs).
        # Ligne unique réécrite à chaque dalle (canal transient).
        short = tile_name if len(tile_name) <= 30 else tile_name[:27] + "…"
        self._user_info_transient(
            f"   • Dalle {index}/{total} : {short}",
            group="tile_progress",
        )
        self._metric(index, total, "dalles")

    def mnt_progress(self, index: int, total: int, mnt_name: str) -> None:
        """Sous-progression du traitement des MNT existants (mode existing_mnt).

        Ligne unique réécrite à chaque MNT (canal transient) + compteur
        structuré « i/n MNT » pour la timeline.
        """
        short = mnt_name if len(mnt_name) <= 30 else mnt_name[:27] + "…"
        self._user_info_transient(
            f"   • MNT {index}/{total} : {short}",
            group="mnt_progress",
        )
        self._metric(index, total, "MNT")

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
        self._r.user_info(
            f"   • Modèle {run_idx}/{total} : « {model_name} » sur {target_rvt}"
        )

    def cv_run_image_progress(
        self, model_name: str, index: int, total: int, image_name: str
    ) -> None:
        """Sous-progression au sein d'un run CV : ``index/total`` images traitées.

        Ligne unique réécrite à chaque image (canal transient). Le
        compteur seul change visuellement — comble le long silence entre
        ``cv_run_start`` et ``cv_complete`` quand le binaire ONNX traite
        plusieurs images d'affilée.
        """
        short = image_name if len(image_name) <= 30 else image_name[:27] + "…"
        self._user_info_transient(
            f"      ↳ Image {index}/{total} : {short}",
            group="cv_image_progress",
        )
        self._metric(index, total, "images")

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

    def finalize_layers_count(self, n_total: int) -> None:
        """Annonce le total de couches ajoutées au projet QGIS.

        Remplace le défilement des lignes ``"Couche raster chargée: X"``
        par un compteur unique en fin de phase. Les détails par couche
        (nom, classe, RGB) descendent au niveau ``INFO`` (fichier-only).
        """
        if n_total > 0:
            self._r.user_info(
                f"📂 {_human_count(n_total, 'couche ajoutée', 'couches ajoutées')} au projet QGIS"
            )

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _elapsed_str(self) -> str:
        if self._pipeline_started_at is None:
            return "0s"
        return _format_duration(time.time() - self._pipeline_started_at)


def create_user_narrator(reporter: "ProgressReporter") -> UserNarrator:
    return UserNarrator(reporter)
