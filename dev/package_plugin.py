#!/usr/bin/env python3
"""
Script pour créer un ZIP du plugin prêt à distribuer.
Exclut tous les fichiers de développement (venv, pycache, git, tests, etc.)
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime

# Dossier racine du plugin (ce script est dans dev/)
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# Nom du plugin (sera le nom du dossier dans le ZIP une fois dézippé)
PLUGIN_NAME = "archeologia"

# Nom du fichier ZIP
ZIP_FILENAME = "main.zip"

# Patterns à exclure
EXCLUDE_DIRS = {
    ".git",
    ".githooks",
    ".pytest_cache",
    "__pycache__",
    "dev",       # Tout l'outillage développeur (requirements, runner_onnx, package_plugin)
    "tests",
    ".venv",
    "node_modules",
}

EXCLUDE_FILES = {
    ".gitignore",
    ".gitkeep",
    ".talismanrc",
    "conftest.py",
    "config.json",
    "pytest.ini",
    "run_tests.py",
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".egg-info",
    ".egg",
    ".whl",
    ".tar.gz",
    ".log",
}

def should_exclude(path: Path, relative_path: str) -> bool:
    """Vérifie si un fichier/dossier doit être exclu."""
    name = path.name
    
    # Exclure les dossiers spécifiques
    if path.is_dir() and name in EXCLUDE_DIRS:
        return True
    
    # Exclure les fichiers spécifiques
    if path.is_file() and name in EXCLUDE_FILES:
        return True
    
    # Exclure par extension
    if path.is_file() and path.suffix in EXCLUDE_EXTENSIONS:
        return True
    
    # Exclure les fichiers .pt et .pth (modèles PyTorch) - on garde seulement .onnx
    if path.is_file() and path.suffix in (".pt", ".pth"):
        return True
    
    return False


def create_plugin_zip(output_dir: Path = None) -> Path:
    """Crée le ZIP du plugin."""
    if output_dir is None:
        output_dir = PLUGIN_ROOT.parent
    
    # Nom du fichier ZIP
    zip_path = output_dir / ZIP_FILENAME
    
    print(f"Création du ZIP: {zip_path}")
    print(f"Source: {PLUGIN_ROOT}")
    print("-" * 60)
    
    files_added = 0
    files_excluded = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PLUGIN_ROOT):
            root_path = Path(root)
            relative_root = root_path.relative_to(PLUGIN_ROOT)
            
            # Filtrer les dossiers à exclure (modifie dirs in-place pour éviter la descente)
            dirs[:] = [d for d in dirs if not should_exclude(root_path / d, str(relative_root / d))]
            
            for file in files:
                file_path = root_path / file
                relative_path = relative_root / file
                
                if should_exclude(file_path, str(relative_path)):
                    files_excluded += 1
                    continue
                
                # Chemin dans le ZIP (avec le nom du plugin comme dossier racine)
                arcname = f"{PLUGIN_NAME}/{relative_path}"
                
                zf.write(file_path, arcname)
                files_added += 1
                print(f"  + {relative_path}")
    
    print("-" * 60)
    print(f"Fichiers ajoutés: {files_added}")
    print(f"Fichiers exclus: {files_excluded}")
    print(f"ZIP créé: {zip_path}")
    print(f"Taille: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    return zip_path


if __name__ == "__main__":
    create_plugin_zip()
