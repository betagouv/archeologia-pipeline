import sys
from pathlib import Path

collect_ignore = ["__init__.py", "main.py"]
collect_ignore_glob = [
    "src/ui/*",
    # Modules pipeline qui dépendent de QGIS au top-level (ou le font transitivement).
    # Les autres modules de src/pipeline (coords, geo_utils, output_paths,
    # subprocess_utils, types, ign/coords_fallback, ign/pdal_validation,
    # ign/tile_resolver) sont importables hors QGIS et donc collectables.
    "src/pipeline/preflight.py",
    "src/pipeline/ign/downloader.py",
    "src/pipeline/ign/preprocess.py",
    "src/pipeline/ign/products/*",
    "src/pipeline/modes/*",
    "src/pipeline/cv/*",
]

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
