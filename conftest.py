import sys
from pathlib import Path

collect_ignore = ["__init__.py", "main.py"]
collect_ignore_glob = ["src/ui/*", "src/pipeline/*"]

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
