# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
import os

# Chemin vers la racine du plugin
# SPECPATH peut être un dossier ou un fichier selon comment PyInstaller est appelé
if os.path.isfile(SPECPATH):
    SPEC_DIR = os.path.dirname(SPECPATH)
else:
    SPEC_DIR = SPECPATH
PLUGIN_ROOT = os.path.abspath(os.path.join(SPEC_DIR, '..', '..'))

# Hidden imports - sahi_lite est inclus dans src/pipeline/cv/
# On n'a plus besoin de sahi (remplacé par sahi_lite numpy-only)
hiddenimports = [
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi._pybind_state',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'numpy',
    'cv2',
    'shapely',
    'shapely.geometry',
    'geopandas',
    'fiona',
    'pyproj',
    'pyogrio',
    'pyogrio._geometry',
    'pyogrio._io',
    'pyogrio._ogr',
    'pyogrio.raw',
    'pyogrio.geopandas',
]

a = Analysis(
    [os.path.join(SPEC_DIR, 'cv_runner_onnx_cli.py')],
    pathex=[PLUGIN_ROOT],
    binaries=[],
    datas=[(os.path.join(PLUGIN_ROOT, 'src'), 'src')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'ultralytics',
        'rfdetr',
        'transformers',
        'timm',
        'tensorflow',
        'keras',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cv_runner_onnx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
