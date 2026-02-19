"""
Script de compilation du runner ONNX unifié.

Ce runner est beaucoup plus léger que les runners YOLO/RF-DETR car il n'inclut
que onnxruntime au lieu de PyTorch + ultralytics/rfdetr.

Usage:
    python build.py              # Compile le runner (~200-300 MB)
    python build.py --clean      # Nettoie les builds
    python build.py --gpu        # Compile avec support GPU (onnxruntime-gpu)
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


RUNNER_DIR = Path(__file__).resolve().parent
DEV_DIR = RUNNER_DIR.parent
PLUGIN_ROOT = DEV_DIR.parent
REQUIREMENTS_BUILD = DEV_DIR / "requirements" / "build.txt"


def _read_deps(use_gpu: bool = False) -> list:
    """Lit les dépendances depuis requirements/build.txt.

    Si *use_gpu* est True, remplace ``onnxruntime`` par ``onnxruntime-gpu``.
    """
    if not REQUIREMENTS_BUILD.exists():
        print(f"[WARN] {REQUIREMENTS_BUILD} introuvable, utilisation des dépendances par défaut")
        deps = [
            "pyinstaller", "onnxruntime", "pillow", "numpy",
            "pyyaml", "opencv-python-headless", "shapely", "geopandas", "fiona",
        ]
    else:
        deps = []
        for line in REQUIREMENTS_BUILD.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            deps.append(line)
    if use_gpu:
        deps = [d.replace("onnxruntime", "onnxruntime-gpu") if d.startswith("onnxruntime") else d for d in deps]
    return deps


def run_cmd(cmd: list, cwd: Path = None) -> int:
    """Exécute une commande et affiche la sortie."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def create_venv(venv_path: Path) -> tuple:
    """Crée un environnement virtuel."""
    if venv_path.exists():
        print(f"[INFO] Environnement existant: {venv_path}")
    else:
        print(f"[INFO] Création de l'environnement: {venv_path}")
        run_cmd([sys.executable, "-m", "venv", str(venv_path)])
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return python_exe, pip_exe


def install_deps(pip_exe: Path, deps: list) -> int:
    """Installe les dépendances."""
    print("[INFO] Installation des dépendances...")
    run_cmd([str(pip_exe), "install", "--upgrade", "pip"])
    return run_cmd([str(pip_exe), "install"] + deps)


def build_runner(python_exe: Path, spec_file: Path, dist_dir: Path) -> int:
    """Compile le runner avec PyInstaller."""
    print(f"[INFO] Compilation de {spec_file.name}...")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    return run_cmd([
        str(python_exe), "-m", "PyInstaller",
        "--clean",
        "--distpath", str(dist_dir),
        "--workpath", str(RUNNER_DIR / "build"),
        str(spec_file),
    ], cwd=spec_file.parent)


def copy_to_third_party(src: Path, dest_dir: Path) -> bool:
    """Copie le binaire vers third_party."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    
    if dest.exists():
        print(f"[INFO] Suppression de l'ancien binaire: {dest}")
        dest.unlink()
    
    print(f"[INFO] Copie vers {dest}")
    shutil.copy2(src, dest)
    
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[INFO] Taille du binaire: {size_mb:.1f} MB")
    
    return True


def build_onnx_runner(use_gpu: bool = False) -> bool:
    """Compile le runner ONNX."""
    print("\n" + "=" * 60)
    print("COMPILATION DU RUNNER ONNX" + (" (GPU)" if use_gpu else " (CPU)"))
    print("=" * 60 + "\n")
    
    venv_path = RUNNER_DIR / ".venv_onnx"
    spec_file = RUNNER_DIR / "cv_runner_onnx.spec"
    dist_dir = RUNNER_DIR / "dist"
    
    if not spec_file.exists():
        print(f"[ERROR] Fichier spec non trouvé: {spec_file}")
        return False
    
    # Créer l'environnement
    python_exe, pip_exe = create_venv(venv_path)
    
    # Installer les dépendances
    deps = _read_deps(use_gpu=use_gpu)
    if install_deps(pip_exe, deps) != 0:
        print("[ERROR] Échec de l'installation des dépendances")
        return False
    
    # Compiler
    if build_runner(python_exe, spec_file, dist_dir) != 0:
        print("[ERROR] Échec de la compilation")
        return False
    
    # Copier vers third_party
    if platform.system() == "Windows":
        binary = dist_dir / "cv_runner_onnx.exe"
        dest_dir = PLUGIN_ROOT / "third_party" / "cv_runner_onnx" / "windows"
    else:
        binary = dist_dir / "cv_runner_onnx"
        dest_dir = PLUGIN_ROOT / "third_party" / "cv_runner_onnx" / "linux"
    
    if not binary.exists():
        print(f"[ERROR] Binaire non trouvé: {binary}")
        return False
    
    copy_to_third_party(binary, dest_dir)
    
    print("\n[SUCCESS] Runner ONNX compilé avec succès!")
    return True


def clean_builds():
    """Nettoie les builds."""
    print("\n" + "=" * 60)
    print("NETTOYAGE DES BUILDS")
    print("=" * 60 + "\n")
    
    dirs_to_clean = [
        RUNNER_DIR / ".venv_onnx",
        RUNNER_DIR / "dist",
        RUNNER_DIR / "build",
    ]
    
    for d in dirs_to_clean:
        if d.exists():
            print(f"[INFO] Suppression de {d}")
            shutil.rmtree(d)
    
    print("\n[SUCCESS] Nettoyage terminé!")


def main():
    parser = argparse.ArgumentParser(
        description="Compile le runner ONNX unifié",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python build.py              # Compile le runner CPU (~100-150 MB)
    python build.py --gpu        # Compile le runner GPU (~300 MB)
    python build.py --clean      # Nettoie les builds
        """
    )
    
    parser.add_argument("--gpu", action="store_true", help="Utiliser onnxruntime-gpu")
    parser.add_argument("--clean", action="store_true", help="Nettoie les builds")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_builds()
        if not args.gpu:
            return 0
    
    if not build_onnx_runner(use_gpu=args.gpu):
        return 1
    
    print("\n" + "=" * 60)
    print("COMPILATION TERMINÉE")
    print("=" * 60)
    print(f"\nRunner ONNX: {PLUGIN_ROOT / 'third_party' / 'cv_runner_onnx'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
