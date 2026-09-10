"""Lance le pipeline SANS QGIS ouvert, pour fabriquer un jeu de démo dense.

Le plugin tourne normalement dans QGIS (worker Qt piloté par le wizard). Ici on
appelle le même ``PipelineController`` depuis le Python de QGIS, en console :
même code, même préflight, même sortie — seul le reporter change (console au
lieu de Qt).

    & "C:\\Program Files\\QGIS 4.0.3\\bin\\python-qgis.bat" dev\\demo_comite\\run_pipeline.py \\
        --x0 388 --x1 393 --y0 6816 --y1 6820 --out D:/pipeline_results/demo_comite

    --dry-run   n'écrit que la liste de dalles, ne lance rien
    --tiles N   ne garde que les N premières dalles (répétition à blanc)

Pourquoi ce script existe : les jeux de D:\\pipeline_results sont soit denses
avec 1 indice, soit riches en indices mais dispersés. L'onglet Visualisation a
besoin des deux à la fois — un bloc jointif avec les 12 produits.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
PLUGINS = ROOT.parent

_pkg = types.ModuleType("archeo")
_pkg.__path__ = [str(ROOT)]
sys.modules["archeo"] = _pkg

QUADRILLAGE = ROOT / "data" / "quadrillage_france" / "TA_diff_pkk_lidarhd_classe.shp"

from archeo.src.app.progress_reporter import USER_INFO  # noqa: E402

TOUS_LES_PRODUITS = (
    "MNT", "DENSITE", "COUVERTURE", "HS", "M_HS", "SVF",
    "SLO", "LD", "SLRM", "VAT", "MSTP", "CVAT",
)


class ConsoleReporter:
    """Le contrat ProgressReporter, en console.

    Tout passe par un ``logging.Logger`` — c'est lui que ``file_logging``
    branche sur le fichier de log du run. Un reporter qui se contenterait
    d'imprimer perdrait le canal technique : le détail du préflight, les
    commandes PDAL/GDAL, les paramètres RVT. (Première version faite comme ça :
    le préflight a échoué et le log était vide.)

    Le canal narratif est en plus imprimé, pour suivre l'avancement en direct.
    """

    def __init__(self):
        self._logger = logging.getLogger("archeologia_pipeline")
        self._dernier_pct = -1

    # canal technique : fichier seulement
    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)
        print(f"  ERREUR  {msg}", flush=True)

    # canal narratif : fichier + console
    def user_info(self, msg: str) -> None:
        self._logger.log(USER_INFO, msg)
        print(f"  {msg}", flush=True)

    def user_warning(self, msg: str) -> None:
        self._logger.warning(msg)
        print(f"  ATTENTION  {msg}", flush=True)

    def user_success(self, msg: str) -> None:
        self._logger.log(USER_INFO, msg)
        print(f"  OK  {msg}", flush=True)

    def user_info_transient(self, msg: str, group: str) -> None:
        self._logger.log(USER_INFO, msg, extra={"transient_group": str(group)})
        print(f"    {msg}", flush=True)

    def stage(self, msg: str) -> None:
        self._logger.log(USER_INFO, f"--- {msg}")
        print(f"=== {msg}", flush=True)

    def progress(self, pct: int) -> None:
        if pct // 10 != self._dernier_pct // 10:
            self._dernier_pct = pct
            print(f"    [{pct} %]", flush=True)

    def load_layers(self, *a, **k) -> None:
        pass                                   # pas de canevas ici


def selectionne_dalles(x0: int, x1: int, y0: int, y1: int) -> list:
    """(nom_pkk, url) des dalles IGN du rectangle, lues dans le quadrillage livré."""
    import pyogrio

    gdf = pyogrio.read_dataframe(
        QUADRILLAGE, bbox=(x0 * 1000, (y0 - 1) * 1000, (x1 + 1) * 1000, y1 * 1000))
    tiles = []
    for nom, url in zip(gdf["nom_pkk"], gdf["url_telech"]):
        m = re.search(r"(\d{4})_(\d{4})", str(nom))
        if not m:
            continue
        x, y = int(m.group(1)), int(m.group(2))
        if x0 <= x <= x1 and y0 <= y <= y1:
            tiles.append(((x, y), str(nom), str(url)))
    tiles.sort()
    return [(nom, url) for _, nom, url in tiles]


#: Références gardées vivantes pour toute la durée du run. Sans elles, le
#: fournisseur RVT est ramassé par le GC et se retire du registre Processing :
#: le préflight le déclarait alors « absent » alors qu'on venait d'y compter
#: 13 algorithmes.
_QGIS = {}


def init_qgis():
    """QgsApplication + Processing + le fournisseur d'algorithmes RVT."""
    from qgis.core import QgsApplication

    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], False)
    app.initQgis()

    prefix = Path(QgsApplication.prefixPath())
    for extra in (prefix / "python" / "plugins", prefix.parent / "qgis" / "python" / "plugins"):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.append(str(extra))

    from processing.core.Processing import Processing
    Processing.initialize()

    # rvt-qgis porte un tiret : on l'enregistre sous un nom importable. Son
    # dossier doit aussi être dans sys.path pour que ses algorithmes trouvent
    # le paquet « rvt » qu'il embarque.
    rvt_dir = PLUGINS / "rvt-qgis"
    if str(rvt_dir) not in sys.path:
        sys.path.append(str(rvt_dir))
    mod = types.ModuleType("rvt_qgis")
    mod.__path__ = [str(rvt_dir)]
    sys.modules["rvt_qgis"] = mod
    from rvt_qgis.processing_provider.provider import Provider

    provider = Provider()
    QgsApplication.processingRegistry().addProvider(provider)
    _QGIS["app"] = app
    _QGIS["provider"] = provider

    registry = QgsApplication.processingRegistry()
    algos = [a.id() for a in registry.algorithms() if a.id().startswith("rvt")]
    print(f"algorithmes RVT enregistrés : {len(algos)}")
    # On vérifie exactement ce que vérifiera le préflight, pas juste un compte.
    if registry.providerById("rvt") is None:
        raise SystemExit("le fournisseur « rvt » n'est pas dans le registre")
    for attendu in ("rvt:rvt_hillshade", "rvt:rvt_svf", "rvt:rvt_ld", "rvt:rvt_mstp"):
        if registry.algorithmById(attendu) is None:
            raise SystemExit(f"algorithme introuvable : {attendu}")
    return app, provider


def construis_config(out: Path, liste: Path) -> dict:
    return {
        "app": {"files": {
            "output_dir": str(out),
            "data_mode": "ign_laz",
            "input_file": str(liste),
            "local_laz_dir": "", "existing_mnt_dir": "", "existing_rvt_dir": "",
        }},
        "processing": {
            "mnt_resolution": 0.5,
            "density_resolution": 1.0,
            "tile_overlap": 20,
            "max_workers": 3,
            "products": {k: True for k in TOUS_LES_PRODUITS},
        },
        "computer_vision": {"enabled": False, "runs": []},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", type=int, default=388)
    ap.add_argument("--x1", type=int, default=393)
    ap.add_argument("--y0", type=int, default=6816)
    ap.add_argument("--y1", type=int, default=6820)
    ap.add_argument("--out", default="D:/pipeline_results/demo_comite")
    ap.add_argument("--tiles", type=int, default=0, help="limiter le nombre de dalles")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiles = selectionne_dalles(args.x0, args.x1, args.y0, args.y1)
    attendu = (args.x1 - args.x0 + 1) * (args.y1 - args.y0 + 1)
    print(f"{len(tiles)} dalles trouvées sur {attendu} attendues "
          f"({args.x0}-{args.x1} / {args.y0}-{args.y1})")
    if len(tiles) < attendu:
        print("  (le rectangle a des trous dans le quadrillage IGN)")
    if args.tiles:
        tiles = tiles[:args.tiles]
        print(f"  limité à {len(tiles)} dalles")
    if not tiles:
        raise SystemExit("aucune dalle : rectangle hors couverture LiDAR HD")

    from archeo.src.app.services.tile_selection import estimate_download_size, format_dalles_urls

    liste = out / "dalles_urls.txt"
    liste.write_text(format_dalles_urls(tiles), encoding="utf-8")
    print(f"liste écrite : {liste}  ({estimate_download_size(len(tiles))})")
    if args.dry_run:
        return 0

    init_qgis()          # garde ses références dans _QGIS

    from archeo.src.app.cancel_token import CancelToken
    from archeo.src.app.pipeline_controller import PipelineController, file_logging
    from archeo.src.app.run_context import build_run_context

    ctx = build_run_context(construis_config(out, liste))
    reporter = ConsoleReporter()
    print(f"\nproduits : {', '.join(TOUS_LES_PRODUITS)}")
    print(f"sortie   : {out}\n")

    with file_logging(ctx.output_dir, reporter):
        ok = PipelineController().run(
            ctx=ctx, reporter=reporter, cancel=CancelToken(threading.Event()))

    print("\nRESULTAT :", "OK" if ok is not False else "ECHEC")
    return 0 if ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
