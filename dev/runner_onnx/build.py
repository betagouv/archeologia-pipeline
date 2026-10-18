"""
Script de compilation du runner ONNX unifié.

Ce runner est beaucoup plus léger que les runners YOLO/RF-DETR car il n'inclut
que onnxruntime au lieu de PyTorch + ultralytics/rfdetr.

Usage:
    python build.py              # Compile le runner : DirectML sous Windows (processeur + carte graphique), CPU ailleurs
    python build.py --cpu        # Compile le runner processeur seul (onnxruntime)
    python build.py --gpu        # Compile avec support CUDA (onnxruntime-gpu)
    python build.py --clean      # Nettoie les builds

Sous Windows, le binaire livré est compilé avec ``onnxruntime-directml`` (version figée
ORT_DIRECTML) : sa partie processeur donne les mêmes détections que ``onnxruntime`` (banc
2026-10-15, 30 127 détections brutes identiques) et sa partie DirectML sert l'option
« Détection sur la carte graphique » de l'étape 4. ``build_info.json`` note le paquet et la
version ; le plugin refuse l'option si le binaire n'a pas DirectML.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RUNNER_DIR = Path(__file__).resolve().parent
DEV_DIR = RUNNER_DIR.parent
PLUGIN_ROOT = DEV_DIR.parent
REQUIREMENTS_BUILD = DEV_DIR / "requirements" / "build.txt"
#: Paquet ORT du binaire Windows : figé (DirectML est en maintenance chez ORT, une autre
#: version peut changer les noyaux et donc les détections sur la carte).
ORT_DIRECTML = "onnxruntime-directml==1.24.4"
#: Les trois paquets ORT s'installent dans le même dossier ``onnxruntime/`` : un seul à la fois.
PAQUETS_ORT = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")


def hash_cv_sources(plugin_root: Path = PLUGIN_ROOT) -> str:
    """SHA256 des sources opérantes embarquées dans l'exe (src/pipeline/cv/**.py + CLI).

    L'exe PyInstaller fige un instantané de src/ : tout commit touchant
    src/pipeline/cv/ ou le CLI rend le binaire livré PÉRIMÉ (T2 de l'audit
    2026-08-31 : binaire du 16/06, 4 commits CV morts pendant 2 mois et demi).
    Ce hash, écrit dans build_info.json à côté de l'exe, est recomparé par
    tests/unit/test_runner_binary_fresh.py — rouge tant qu'on n'a pas relancé
    ``python dev/runner_onnx/build.py``.
    """
    h = hashlib.sha256()
    fichiers = sorted((plugin_root / "src" / "pipeline" / "cv").rglob("*.py"))
    fichiers.append(plugin_root / "dev" / "runner_onnx" / "cv_runner_onnx_cli.py")
    for f in fichiers:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def infos_onnxruntime(python_exe: Path) -> dict:
    """Paquet, version et fournisseurs ORT du venv de build, lus par le plugin
    (``app/services/carte_graphique.binaire_directml``) pour savoir si l'option carte
    graphique peut servir."""
    code = ("import json, onnxruntime as o; print(json.dumps({'version': o.__version__, "
            "'fournisseurs': o.get_available_providers()}))")
    try:
        out = subprocess.run([str(python_exe), "-c", code], capture_output=True, text=True, check=True).stdout
        info = json.loads(out.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"paquet": "?", "version": "?", "erreur": str(e)[:120]}
    directml = "DmlExecutionProvider" in info.get("fournisseurs", [])
    cuda = "CUDAExecutionProvider" in info.get("fournisseurs", [])
    paquet = "onnxruntime-directml" if directml else ("onnxruntime-gpu" if cuda else "onnxruntime")
    return {"paquet": paquet, "version": info.get("version", "?"), "directml": directml}


def write_build_info(dest_dir: Path, python_exe: Path = None) -> None:
    """build_info.json (commit, date, sha256 des sources, paquet ORT) à côté de l'exe."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PLUGIN_ROOT,
            capture_output=True, text=True,
        ).stdout.strip() or "?"
    except Exception:
        commit = "?"
    info = {
        "commit": commit,
        "date": datetime.now().isoformat(timespec="seconds"),
        "sources_sha256": hash_cv_sources(),
        "onnxruntime": infos_onnxruntime(python_exe) if python_exe else {"paquet": "?", "version": "?"},
    }
    (dest_dir / "build_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] build_info.json écrit (commit {commit}, sources {info['sources_sha256'][:12]}…)")


def _read_deps(use_gpu: bool = False, use_directml: bool = False) -> list:
    """Lit les dépendances depuis requirements/build.txt.

    Si *use_gpu* est True, remplace ``onnxruntime`` par ``onnxruntime-gpu`` ; si
    *use_directml* est True, par :data:`ORT_DIRECTML` (version figée).
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
    if use_directml:
        deps = [ORT_DIRECTML if d.startswith("onnxruntime") else d for d in deps]
    elif use_gpu:
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
    """Installe les dépendances ; retire d'abord les autres paquets ORT (même dossier)."""
    print("[INFO] Installation des dépendances...")
    run_cmd([str(pip_exe), "install", "--upgrade", "pip"])
    voulu = next((d for d in deps if d.startswith("onnxruntime")), "onnxruntime")
    nom_voulu = voulu.split("==")[0].split(">=")[0].strip()
    autres = [p for p in PAQUETS_ORT if p != nom_voulu]
    run_cmd([str(pip_exe), "uninstall", "-y", *autres])
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


def build_onnx_runner(use_gpu: bool = False, use_directml: bool = False) -> bool:
    """Compile le runner ONNX."""
    print("\n" + "=" * 60)
    print("COMPILATION DU RUNNER ONNX" + (" (DirectML)" if use_directml else " (GPU)" if use_gpu else " (CPU)"))
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
    deps = _read_deps(use_gpu=use_gpu, use_directml=use_directml)
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
        dest_dir = PLUGIN_ROOT / "data" / "third_party" / "cv_runner_onnx" / "windows"
    else:
        binary = dist_dir / "cv_runner_onnx"
        dest_dir = PLUGIN_ROOT / "data" / "third_party" / "cv_runner_onnx" / "linux"
    
    if not binary.exists():
        print(f"[ERROR] Binaire non trouvé: {binary}")
        return False
    
    copy_to_third_party(binary, dest_dir)
    write_build_info(dest_dir, python_exe)

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
    python build.py              # Windows : DirectML (processeur + carte graphique) ; ailleurs : CPU
    python build.py --cpu        # Compile le runner processeur seul
    python build.py --gpu        # Compile le runner CUDA (~300 MB)
    python build.py --clean      # Nettoie les builds
        """
    )

    parser.add_argument("--gpu", action="store_true", help="Utiliser onnxruntime-gpu (CUDA)")
    parser.add_argument("--cpu", action="store_true", help="Processeur seul (onnxruntime), sans DirectML")
    parser.add_argument("--clean", action="store_true", help="Nettoie les builds")

    args = parser.parse_args()

    if args.clean:
        clean_builds()
        if not (args.gpu or args.cpu or platform.system() == "Windows"):
            return 0

    use_directml = platform.system() == "Windows" and not args.gpu and not args.cpu
    if not build_onnx_runner(use_gpu=args.gpu, use_directml=use_directml):
        return 1
    
    print("\n" + "=" * 60)
    print("COMPILATION TERMINÉE")
    print("=" * 60)
    print(f"\nRunner ONNX: {PLUGIN_ROOT / 'data' / 'third_party' / 'cv_runner_onnx'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
