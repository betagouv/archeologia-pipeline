#!/usr/bin/env python3
"""
Pipeline de traitement LIDAR archéologique
Conversion Python du script.bat original

Ce script traite les données LIDAR pour créer des modèles de terrain
et des produits de visualisation pour l'archéologie.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import urllib.request
import urllib.parse
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time
import datetime
import math
import requests
import psutil

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

def load_config():
    """Charge la configuration depuis le fichier JSON"""
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class LidarPipeline:
    """Pipeline de traitement des données LIDAR archéologiques"""
    
    def __init__(self, work_dir: str = None, ui_callback=None, input_file: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        # Charger la configuration
        self.config = load_config()
        
        # Initialiser ui_config
        self.ui_config = {}
        # Configuration effective figée pendant un run (par défaut, la config de base)
        self.effective_config = dict(self.config)
        
        # Utiliser le fichier d'entrée fourni ou celui de la configuration
        self.input_file = input_file if input_file else self.config['app']['files'].get('input_file', 'entrainement_liste.txt')
        # Fichiers de tri dans le dossier output
        self.temp_file = self.work_dir / 'temp_sortie.txt'
        self.sorted_file = self.work_dir / 'fichier_tri.txt'
        self.output_dir = self.config['app']['files'].get('output_dir', 'Output')
        self.num_dalle = 0
        
        # Callback pour l'interface utilisateur
        self.ui_callback = ui_callback
        
        # Flag pour l'arrêt du pipeline
        self.stop_requested = False
        self.current_output_file = None  # Fichier en cours de création
        
        # Dictionnaire pour stocker les données de géoréférencement des TIF
        self.tif_transform_data = {}

        # Métriques de performance par étape
        self._metrics = {}
        self._process = psutil.Process(os.getpid())
        self._max_rss_mb = 0.0
        
        # Configuration du logging (AVANT setup_directories)
        # Fichier de log écrasé à chaque démarrage
        log_file = self.work_dir / "pipeline_logs.txt"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # mode='w' pour écraser, UTF-8 pour les emojis
                logging.StreamHandler()  # Affichage console aussi
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_to_ui(self, message: str, level: str = "info"):
        """Envoie un message vers l'interface utilisateur si disponible"""
        if self.ui_callback:
            self.ui_callback(message)
        
        # Log aussi dans le système de logging standard
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
            
        # Appeler setup_directories seulement si pas encore fait
        if not hasattr(self, '_directories_setup'):
            self.setup_directories()
            self._directories_setup = True
            
        # Initialiser les chemins des outils après setup_directories
        if not hasattr(self, 'qgis_process'):
            # Construire les chemins des outils depuis le dossier bin OSGeo4W
            osgeo4w_bin, osgeo4w_root = self._get_osgeo_paths()
            self.qgis_process = f"{osgeo4w_bin}/qgis_process-qgis.bat"
            self.gdalwarp = f"{osgeo4w_bin}/gdalwarp.exe"
    
    def get_config_value(self, section: str, key=None, default=None):
        """Récupère une valeur depuis la configuration EFFECTIVE (figée au démarrage du run)."""
        cfg = getattr(self, 'effective_config', self.config)
        if key is None:
            return cfg.get(section, default if default is not None else {})
        return cfg.get(section, {}).get(key, default)

    def _get_osgeo_paths(self) -> Tuple[str, str]:
        """Retourne (osgeo4w_bin, osgeo4w_root) à partir de la configuration tools.

        - Si tools.osgeo4w_bin est défini, on l'utilise directement et on déduit
          la racine comme son parent.
        - Sinon, on tombe en arrière sur tools.osgeo4w_root (racine) et on
          considère <root>/bin comme dossier bin.
        """
        tools_cfg = self.get_config_value('tools')
        bin_dir = tools_cfg.get('osgeo4w_bin')
        if bin_dir:
            bin_path = Path(bin_dir)
            root_path = bin_path.parent
        else:
            root_str = tools_cfg.get('osgeo4w_root', r"C:\OSGeo4W")
            root_path = Path(root_str)
            bin_path = root_path / "bin"
        return str(bin_path), str(root_path)

    def _get_pdal_path(self) -> str:
        tools_cfg = self.get_config_value('tools')
        pdal_path = tools_cfg.get('pdal')
        if pdal_path and Path(pdal_path).exists():
            return str(Path(pdal_path))

        osgeo4w_bin, _ = self._get_osgeo_paths()
        candidate = Path(osgeo4w_bin) / "pdal.exe"
        if candidate.exists():
            return str(candidate)
        return "pdal"

    def _ensure_pdal_available(self) -> bool:
        pdal_cmd = self._get_pdal_path()
        if pdal_cmd.lower() == "pdal":
            if shutil.which("pdal"):
                return True
            self.logger.error("PDAL introuvable (commande 'pdal' non disponible). Vérifiez l'installation OSGeo4W/QGIS ou renseignez tools.pdal dans config.json")
            return False
        if not Path(pdal_cmd).exists():
            self.logger.error(f"PDAL introuvable au chemin configuré: {pdal_cmd}")
            return False
        return True

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Fusion profonde de deux dictionnaires (override prend le dessus)."""
        if not isinstance(base, dict):
            return override
        result = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = LidarPipeline._deep_merge_dicts(result[k], v)
            else:
                result[k] = v
        return result
        
    def setup_directories(self):
        """Crée les dossiers nécessaires pour l'organisation des fichiers"""
        # Définir les chemins des dossiers de base
        self.input_dir = self.work_dir / "dalles"  # Fichiers LIDAR téléchargés (dans output/dalles)
        self.temp_dir = self.work_dir / "Temp"    # Fichiers intermédiaires
        self.results_dir = self.work_dir / "results"  # Dossier results pour les produits finaux

        # Créer les dossiers de base
        # En mode local_laz, les dalles sont lues directement depuis local_laz_dir,
        # donc on ne crée pas le dossier dalles dans output.
        data_mode = self.get_config_value('processing', 'data_mode', 'ign_laz')
        base_directories = []
        if data_mode not in ('local_laz', 'local_ign'):
            base_directories.append(self.input_dir)
        base_directories.extend([self.temp_dir, self.results_dir])

        for directory in base_directories:
            directory.mkdir(exist_ok=True)
            self.logger.info(f"Dossier créé/vérifié: {directory}")
        
        # Créer la structure de dossiers organisée dans results
        output_structure = self.get_config_value('processing', 'output_structure', {})
        
        # Récupérer les produits sélectionnés par l'utilisateur
        products_config = self.get_config_value('processing', 'products', {})
        
        # Dossiers pour MNT et DENSITE (TIF seulement) dans results - seulement si sélectionnés
        if 'MNT' in output_structure and products_config.get('MNT', False):
            mnt_dir = self.results_dir / output_structure['MNT']
            mnt_dir.mkdir(exist_ok=True)
            # Sous-dossier TIF seulement pour MNT
            (mnt_dir / 'tif').mkdir(exist_ok=True)
            self.logger.info(f"Structure MNT créée: {mnt_dir}")
            
        if 'DENSITE' in output_structure and products_config.get('DENSITE', False):
            densite_dir = self.results_dir / output_structure['DENSITE']
            densite_dir.mkdir(exist_ok=True)
            # Sous-dossier TIF seulement pour DENSITE
            (densite_dir / 'tif').mkdir(exist_ok=True)
            self.logger.info(f"Structure DENSITE créée: {densite_dir}")
        
        # Dossiers pour RVT dans results - seulement pour les produits RVT sélectionnés
        if 'RVT' in output_structure:
            rvt_config = output_structure['RVT']
            
            # Vérifier si au moins un produit RVT est sélectionné
            rvt_products_selected = any(products_config.get(rvt_type, False) for rvt_type in ['M_HS', 'SVF', 'SLO', 'LD', 'VAT'])
            
            if rvt_products_selected:
                rvt_base_dir = self.results_dir / rvt_config['base_dir']
                rvt_base_dir.mkdir(exist_ok=True)
                
                # Créer chaque sous-dossier RVT avec ses formats - seulement si sélectionné
                for rvt_type in ['M_HS', 'SVF', 'SLO', 'LD', 'VAT']:
                    if rvt_type in rvt_config and products_config.get(rvt_type, False):
                        rvt_type_dir = rvt_base_dir / rvt_config[rvt_type]
                        rvt_type_dir.mkdir(exist_ok=True)
                        # Sous-dossiers TIF et JPG pour chaque type RVT
                        (rvt_type_dir / 'tif').mkdir(exist_ok=True)
                        (rvt_type_dir / 'jpg').mkdir(exist_ok=True)
                        self.logger.info(f"Structure RVT {rvt_type} créée: {rvt_type_dir}")

    def save_config(self):
        """Saves the current configuration to the config.json file"""
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.logger.info("Configuration saved successfully.")

    def run_qgis_algorithm(self, algorithm_id: str, parameters: Dict[str, str]) -> bool:
        # Utiliser le chemin OSGeo4W bin configuré en priorité, puis fallback
        osgeo4w_bin, osgeo4w_root = self._get_osgeo_paths()
        possible_paths = [
            self.qgis_process,
            f"{osgeo4w_bin}/qgis_process-qgis.bat",
            r"C:\Program Files\QGIS 3.34.0\bin\qgis_process-qgis.bat",
            r"C:\Program Files\QGIS 3.32.0\bin\qgis_process-qgis.bat", 
            r"C:\Program Files\QGIS 3.30.0\bin\qgis_process-qgis.bat",
            r"C:\OSGeo4W\bin\qgis_process-qgis.bat",
            r"C:\OSGeo4W64\bin\qgis_process-qgis.bat"
        ]
        
        qgis_found = None
        for path in possible_paths:
            if Path(path).exists():
                qgis_found = path
                self.logger.info(f"QGIS trouvé à: {path}")
                break
        
        if not qgis_found:
            self.logger.error(f"QGIS introuvable dans les chemins suivants:")
            for path in possible_paths:
                self.logger.error(f"  - {path}")
            self.logger.error("Veuillez installer QGIS ou corriger le chemin dans config.json")
            return False
        
        # Utiliser le chemin trouvé
        self.qgis_process = qgis_found

        # Normaliser certains paramètres de chemin (INPUT/OUTPUT) et vérifier l'existence
        normalized_parameters = dict(parameters or {})
        try:
            input_val = normalized_parameters.get("INPUT")
            if input_val:
                input_path = Path(str(input_val))
                if not input_path.exists():
                    self.logger.error(f"QGIS: fichier INPUT introuvable: {input_path}")
                    self.logger.error("Action requise: supprimer la dalle/nuage associé(e) et la retélécharger (fichier manquant ou non accessible).")
                    try:
                        parent_dir = input_path.parent
                        if parent_dir.exists():
                            candidates = sorted([p.name for p in parent_dir.glob(f"{input_path.stem}*")])
                            if candidates:
                                self.logger.error(f"QGIS: fichiers similaires dans {parent_dir}: {candidates[:10]}")
                    except Exception:
                        pass
                    return False
                normalized_parameters["INPUT"] = str(input_path.resolve()).replace('\\', '/')

            output_val = normalized_parameters.get("OUTPUT")
            if output_val:
                output_path = Path(str(output_val))
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                normalized_parameters["OUTPUT"] = str(output_path.resolve()).replace('\\', '/')

            # BOUNDS et autres paramètres restent inchangés
        except Exception as e:
            self.logger.warning(f"QGIS: échec de normalisation/validation des chemins paramètres: {e}")

        # Construire la commande
        cmd = [self.qgis_process, "run", algorithm_id, "--"]
        for key, value in normalized_parameters.items():
            cmd.append(f"{key}={value}")

        self.logger.info("Exécution QGIS : " + " ".join(cmd))

        # Préparer l'environnement avec PATH étendu et variables QGIS
        env = os.environ.copy()
        osgeo4w_bin, osgeo4w_root = self._get_osgeo_paths()
        
        # Détecter si on est dans un exécutable PyInstaller
        is_frozen = getattr(sys, 'frozen', False)
        
        # Configuration complète de l'environnement QGIS
        env["PATH"] = f"{osgeo4w_bin};" + env.get("PATH", "")
        env["QGIS_PREFIX_PATH"] = osgeo4w_root
        env["QGIS_DEBUG"] = "1"
        
        # Variables spécifiques pour PyInstaller
        if is_frozen:
            env["PYTHONPATH"] = f"{osgeo4w_root}/apps/qgis/python;" + env.get("PYTHONPATH", "")
            env["QT_PLUGIN_PATH"] = f"{osgeo4w_root}/apps/Qt5/plugins;" + env.get("QT_PLUGIN_PATH", "")
            env["GDAL_DATA"] = f"{osgeo4w_root}/share/gdal"
            env["PROJ_LIB"] = f"{osgeo4w_root}/share/proj"
            env["GDAL_DRIVER_PATH"] = f"{osgeo4w_bin}/gdalplugins"
            # Forcer l'utilisation de l'environnement OSGeo4W
            env["OSGEO4W_ROOT"] = osgeo4w_root
            self.logger.info("🔧 Configuration environnement QGIS pour exécutable PyInstaller")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            if result.stdout.strip():
                self.logger.info(f"QGIS stdout:\n{result.stdout}")
            if result.stderr.strip():
                self.logger.warning(f"QGIS stderr:\n{result.stderr}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error("Erreur QGIS :")
            if e.stdout:
                self.logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR:\n{e.stderr}")

            try:
                stderr_lower = (e.stderr or "").lower()
                if "could not load source layer" in stderr_lower and "input" in stderr_lower and "not found" in stderr_lower:
                    input_hint = normalized_parameters.get("INPUT")
                    if input_hint:
                        self.logger.error(f"Action requise: supprimer la dalle/nuage associé(e) et la retélécharger (INPUT introuvable côté QGIS): {input_hint}")
                    else:
                        self.logger.error("Action requise: supprimer la dalle/nuage associé(e) et la retélécharger (INPUT introuvable côté QGIS).")
            except Exception:
                pass

            # En cas d’échec silencieux, tenter d’afficher la liste des algos disponibles
            self.logger.info("Diagnostic : récupération de la liste des algorithmes QGIS disponibles...")
            try:
                list_cmd = [self.qgis_process, "list"]
                list_result = subprocess.run(list_cmd, capture_output=True, text=True, env=env)
                self.logger.info("Liste des algorithmes disponibles :\n" + list_result.stdout)
            except Exception as diag_error:
                self.logger.error(f"Impossible d’obtenir la liste des algorithmes : {diag_error}")
            return False


    def _snapshot_resources(self):
        mem = self._process.memory_info().rss
        mem_mb = mem / (1024 * 1024)
        if mem_mb > getattr(self, "_max_rss_mb", 0.0):
            self._max_rss_mb = mem_mb
        gpu_mem = None
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated()
            except Exception:
                gpu_mem = None
        return time.time(), mem, gpu_mem

    def _record_metric(self, tile_id: str, step: str, start_snapshot, end_snapshot):
        t0, mem0, gpu0 = start_snapshot
        t1, mem1, gpu1 = end_snapshot
        duration = t1 - t0
        mem_delta_mb = (mem1 - mem0) / (1024 * 1024)
        mem_end_mb = mem1 / (1024 * 1024)
        gpu_delta_mb = None
        if gpu0 is not None and gpu1 is not None:
            gpu_delta_mb = (gpu1 - gpu0) / (1024 * 1024)
        entry = {
            "tile": tile_id,
            "duration_s": duration,
            "mem_delta_mb": mem_delta_mb,
            "mem_end_mb": mem_end_mb,
            "gpu_delta_mb": gpu_delta_mb,
        }
        self._metrics.setdefault(step, []).append(entry)

    def _measure_step(self, tile_id: str, step: str, func, *args, **kwargs):
        start = self._snapshot_resources()
        success = func(*args, **kwargs)
        end = self._snapshot_resources()
        self._record_metric(tile_id, step, start, end)
        return success

    def _print_metrics_summary(self):
        if not self._metrics:
            self.logger.info("Aucune métrique de performance collectée.")
            return
        self.logger.info("=== RÉCAPITULATIF DES MÉTRIQUES PAR ÉTAPE ===")
        for step, records in self._metrics.items():
            n = len(records)
            total_time = sum(r["duration_s"] for r in records)
            avg_time = total_time / n if n else 0.0
            avg_mem = sum(r["mem_delta_mb"] for r in records) / n if n else 0.0
            avg_mem_end = sum(r["mem_end_mb"] for r in records) / n if n else 0.0
            gpu_values = [r["gpu_delta_mb"] for r in records if r["gpu_delta_mb"] is not None]
            avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else None
            self.logger.info(f"Étape: {step} - appels: {n}")
            self.logger.info(f"  Temps total: {total_time:.2f}s, temps moyen: {avg_time:.2f}s")
            self.logger.info(f"  ΔRAM moyen: {avg_mem:.2f} MB")
            self.logger.info(f"  RAM moyenne en fin d'étape: {avg_mem_end:.2f} MB")
            if avg_gpu is not None:
                self.logger.info(f"  ΔGPU moyen: {avg_gpu:.2f} MB")
        if getattr(self, "_max_rss_mb", 0.0) > 0.0:
            self.logger.info(f"RAM maximale observée pour le processus pendant le pipeline: {self._max_rss_mb:.2f} MB")
        self.logger.info("=== FIN RÉCAPITULATIF MÉTRIQUES ===")

    def clean_temp_files(self):
        """Supprime les fichiers temporaires"""
        temp_files = [self.temp_file, self.sorted_file]
        for file in temp_files:
            if Path(file).exists():
                Path(file).unlink()
    
    def cleanup_on_stop(self):
        """Nettoie uniquement le fichier en cours de création lors d'un arrêt"""
        try:
            # Supprimer uniquement le fichier le plus récent dans le dossier Temp
            latest_file = None
            latest_mtime = -1
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                for p in self.temp_dir.iterdir():
                    if p.is_file():
                        mtime = p.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_file = p

            if latest_file and latest_file.exists():
                latest_file.unlink()
                self.logger.info(f"Fichier temporaire le plus récent supprimé: {latest_file.name}")
            else:
                self.logger.info("Aucun fichier en cours de création à supprimer")

            self.logger.info("Arrêt propre - fichiers précédents conservés")

        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage d'arrêt: {e}")
    
    def parse_input_file(self) -> List[Tuple[str, str]]:
        """Parse le fichier d'entrée et retourne la liste des fichiers triés"""
        self.logger.info("Début du tri des fichiers")
        
        if not Path(self.input_file).exists():
            raise FileNotFoundError(f"Fichier d'entrée non trouvé: {self.input_file}")
        
        temp_data = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Deux formats acceptés :
                # 1) "nom_fichier,URL" (format historique)
                # 2) "URL" seule (on déduit alors nom_fichier depuis l'URL)
                if ',' in line:
                    nom_fichier, lien = line.split(',', 1)
                else:
                    # Ligne contenant uniquement une URL
                    lien = line
                    try:
                        parsed = urllib.parse.urlparse(lien)
                        nom_fichier = Path(parsed.path).name
                    except Exception:
                        nom_fichier = ""

                if not nom_fichier:
                    self.logger.warning(f"Ligne ignorée (nom de fichier introuvable): {line}")
                    continue

                # Extraire les coordonnées à partir du nom SANS extension(s)
                # Exemple: LHD_FXX_0948_6755.copc.laz -> LHD_FXX_0948_6755
                p = Path(nom_fichier)
                # Gérer les doubles extensions (.copc.laz) en appliquant stem deux fois
                nom_sans_ext = Path(p.stem).stem
                parts = nom_sans_ext.split('_')
                if len(parts) >= 4:
                    try:
                        x_coord = int(parts[2])
                        y_coord = int(parts[3])
                        temp_data.append((x_coord, y_coord, nom_fichier, lien))
                    except (ValueError, IndexError):
                        self.logger.warning(f"Impossible d'extraire les coordonnées de: {nom_fichier}")
        
        # Tri par coordonnées
        temp_data.sort(key=lambda x: (x[0], x[1]))
        
        # Sauvegarde du fichier trié
        with open(self.sorted_file, 'w', encoding='utf-8') as f:
            for _, _, nom_fichier, lien in temp_data:
                f.write(f"{nom_fichier},{lien}\n")
        
        return [(nom_fichier, lien) for _, _, nom_fichier, lien in temp_data]
    
    def download_file(self, url: str, filename: str) -> bool:
        """Télécharge un fichier depuis une URL dans le dossier Input"""
        # Chemin complet dans le dossier Input
        file_path = self.input_dir / filename
        
        if file_path.exists():
            self.log_to_ui(f"✅ {filename} déjà téléchargé")
            return True
        
        # Si l'URL ressemble à un chemin local existant, copier le fichier au lieu de le télécharger
        try:
            local_path_str = url
            if url.startswith("file://"):
                local_path_str = url[7:]
            local_path = Path(local_path_str)
            if local_path.exists() and local_path.is_file():
                # En mode local_laz, si le dossier d'entrée est déjà le dossier source,
                # on évite de recopier les fichiers dans output/dalles.
                try:
                    data_mode = self.get_config_value('processing', 'data_mode', 'ign_laz')
                except Exception:
                    data_mode = 'ign_laz'

                try:
                    if data_mode in ('local_laz', 'local_ign') and hasattr(self, 'input_dir') \
                            and Path(self.input_dir).resolve() == local_path.parent.resolve():
                        self.log_to_ui(f"📥 Utilisation directe du fichier local {local_path} (pas de copie en mode local)")
                        return True
                except Exception:
                    # En cas de problème avec resolve(), on retombe sur le comportement standard de copie
                    pass

                self.log_to_ui(f"📥 Copie du fichier local {local_path} vers {file_path}...")
                shutil.copy2(str(local_path), str(file_path))
                self.logger.info(f"Copie locale terminée: {file_path}")
                return True
        except Exception as e:
            self.logger.warning(f"Impossible de traiter {url} comme chemin local: {e}")

        self.log_to_ui(f"📥 Téléchargement de {filename}...")
        
        # Utiliser les paramètres de téléchargement par défaut
        download_config = {}
        timeout = download_config.get('timeout', 300)
        chunk_size = download_config.get('chunk_size', 8192)
        max_retries = download_config.get('max_retries', 3)
        retry_delay = download_config.get('retry_delay', 5)
        
        for attempt in range(max_retries):
            try:
                # Désactiver le proxy pour l'instant
                use_proxy = False
                proxies = getattr(self, 'proxy_config', None) if use_proxy else None
                response = requests.get(url, proxies=proxies, stream=True, timeout=timeout)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                
                self.logger.info(f"Téléchargement terminé: {file_path}")
                return True
                
            except requests.RequestException as e:
                self.logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée pour {filename}: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Nouvelle tentative dans {retry_delay} secondes...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Échec définitif du téléchargement pour {filename}")
                    return False
        
        return False
    
    def extract_coordinates(self, filename: str) -> Tuple[str, str]:
        """Extrait les coordonnées X et Y du nom de fichier"""
        # Enlever proprement les extensions multiples (ex: .copc.laz)
        p = Path(filename)
        nom_sans_ext = Path(p.stem).stem
        parts = nom_sans_ext.split('_')
        if len(parts) >= 4:
            return parts[2], parts[3]
        raise ValueError(f"Impossible d'extraire les coordonnées de: {filename}")
    
    def calculate_neighbor_coordinates(self, x: str, y: str) -> List[Tuple[int, int, int]]:
        """Calcule les coordonnées des dalles voisines"""
        min_x = int(f"1{x}") - 10000
        min_y = int(f"1{y}") - 10000
        
        neighbors = []
        place_dalle = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip central tile
                    continue
                
                place_dalle += 1
                voisin_x = min_x + dx
                voisin_y = min_y + dy
                neighbors.append((voisin_x, voisin_y, place_dalle))
        
        return neighbors
    
    def format_coordinate(self, coord: int) -> str:
        """Formate une coordonnée avec des zéros en préfixe"""
        return f"{coord:04d}"
    
    def find_neighbor_file(self, voisin_x: int, voisin_y: int) -> Optional[Tuple[str, str]]:
        """Trouve le fichier correspondant aux coordonnées voisines"""
        coord_x = self.format_coordinate(voisin_x)
        coord_y = self.format_coordinate(voisin_y)
        search_pattern = f"{coord_x}_{coord_y}"
        
        try:
            with open(self.sorted_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if search_pattern in line:
                        parts = line.strip().split(',', 1)
                        if len(parts) == 2:
                            return parts[0], parts[1]
        except FileNotFoundError:
            self.logger.error(f"Fichier trié non trouvé: {self.sorted_file}")
        
        return None
    
    def get_raster_bounds(self, raster_file: str) -> Tuple[float, float, float, float]:
        """Extrait les coordonnées réelles d'un fichier raster avec gdalinfo"""
        osgeo4w_bin, _ = self._get_osgeo_paths()
        gdalinfo_path = f"{osgeo4w_bin}/gdalinfo.exe"
        
        try:
            result = subprocess.run([gdalinfo_path, raster_file], 
                                   capture_output=True, text=True, check=True)
            
            # Parser la sortie pour extraire les coordonnées
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Upper Left' in line:
                    # Format: Upper Left  (  809999.750, 6335000.250)
                    coords = line.split('(')[1].split(')')[0]
                    x_min, y_max = map(float, coords.split(','))
                elif 'Lower Right' in line:
                    # Format: Lower Right (  811000.250, 6333999.750)
                    coords = line.split('(')[1].split(')')[0]
                    x_max, y_min = map(float, coords.split(','))
            
            return x_min, y_min, x_max, y_max
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des coordonnées de {raster_file}: {e}")
            return None

    def build_raster_pyramids(self, raster_files, levels: Optional[List[int]] = None) -> bool:
        """Construit des pyramides (overviews) pour un ou plusieurs rasters avec gdaladdo.

        Args:
            raster_files: Chemin unique (str) ou liste de chemins vers les rasters TIF.
            levels: Liste des niveaux de sur-échantillonnage (ex: [2, 4, 8, 16]).
                    Si None, des niveaux par défaut adaptés aux MNT seront utilisés.

        Retourne True si toutes les générations réussissent, False sinon.
        """
        try:
            osgeo4w_bin, _ = self._get_osgeo_paths()
            gdaladdo_path = f"{osgeo4w_bin}/gdaladdo.exe"

            if not Path(gdaladdo_path).exists():
                self.logger.error(f"gdaladdo introuvable à l'emplacement attendu: {gdaladdo_path}")
                return False

            # Normaliser en liste
            if isinstance(raster_files, (str, Path)):
                raster_list = [str(raster_files)]
            else:
                raster_list = [str(r) for r in raster_files]

            # Niveaux par défaut adaptés à une diffusion type cartes.gouv
            if levels is None or len(levels) == 0:
                levels = [2, 4, 8, 16]

            levels_str = [str(l) for l in levels]

            # Utiliser un environnement cohérent avec OSGeo4W
            env = os.environ.copy()
            env["PATH"] = f"{osgeo4w_bin};" + env.get("PATH", "")

            all_success = True

            for raster_file in raster_list:
                if not Path(raster_file).exists():
                    self.logger.error(f"Raster introuvable pour génération des pyramides: {raster_file}")
                    all_success = False
                    continue

                cmd = [gdaladdo_path, "-r", "average", raster_file] + levels_str

                self.logger.info("=== GÉNÉRATION DES PYRAMIDES RASTER ===")
                self.logger.info(f"Raster: {raster_file}")
                self.logger.info(f"Niveaux: {', '.join(levels_str)}")
                self.logger.info(f"Commande: {' '.join(cmd)}")

                result = subprocess.run(cmd, capture_output=True, text=True, env=env)

                if result.returncode == 0:
                    self.logger.info("Pyramides générées avec succès.")
                    if result.stdout.strip():
                        self.logger.info(f"gdaladdo stdout:\n{result.stdout}")
                    if result.stderr.strip():
                        self.logger.warning(f"gdaladdo stderr:\n{result.stderr}")
                else:
                    self.logger.error(f"Échec de gdaladdo (code {result.returncode}) pour {raster_file}.")
                    if result.stdout.strip():
                        self.logger.error(f"gdaladdo stdout:\n{result.stdout}")
                    if result.stderr.strip():
                        self.logger.error(f"gdaladdo stderr:\n{result.stderr}")
                    all_success = False

            return all_success

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des pyramides raster: {e}")
            return False
    
    def calculate_crop_bounds(self, voisin_x: int, voisin_y: int, place_dalle: int, margin_m: int) -> Dict[str, str]:
        """Calcule les limites de rognage d'une dalle voisine.
        margin_m correspond à la marge en mètres (ex: 50 pour reproduire 050/950)."""
        bounds = {}
        
        if place_dalle == 1:
            # Place dalle 1 : coin supérieur gauche
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}{(1000 - margin_m):03d}"
            xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
            bounds['xmax'] = f"{xnum2:04d}000"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}{(1000 - margin_m):03d}"
            bounds['ymax'] = f"{voisin_y:04d}000"
            
        elif place_dalle == 2:
            # Place dalle 2 : coin supérieur centre
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}{(1000 - margin_m):03d}"
            xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
            bounds['xmax'] = f"{xnum2:04d}000"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}000"
            bounds['ymax'] = f"{voisin_y:04d}000"
            
        elif place_dalle == 3:
            # Place dalle 3 : coin supérieur droit
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}{(1000 - margin_m):03d}"
            xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
            bounds['xmax'] = f"{xnum2:04d}000"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}000"
            ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymax'] = f"{ynum2:04d}{margin_m:03d}"
            
        elif place_dalle == 4:
            # Place dalle 4 : centre gauche
            bounds['xmin'] = f"{voisin_x:04d}000"
            xnum = int(f"1{voisin_x:04d}") - 10000 + 1
            bounds['xmax'] = f"{xnum:04d}000"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}{(1000 - margin_m):03d}"
            bounds['ymax'] = f"{voisin_y:04d}000"
            
        elif place_dalle == 5:
            # Place dalle 5 : centre droit
            bounds['xmin'] = f"{voisin_x:04d}000"
            xnum = int(f"1{voisin_x:04d}") - 10000 + 1
            bounds['xmax'] = f"{xnum:04d}000"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}000"
            ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymax'] = f"{ynum2:04d}{margin_m:03d}"
            
        elif place_dalle == 6:
            # Place dalle 6 : coin inférieur gauche
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}000"
            bounds['xmax'] = f"{xnum:04d}{margin_m:03d}"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}{(1000 - margin_m):03d}"
            bounds['ymax'] = f"{voisin_y:04d}000"
            
        elif place_dalle == 7:
            # Place dalle 7 : coin inférieur centre
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}000"
            bounds['xmax'] = f"{xnum:04d}{margin_m:03d}"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}000"
            bounds['ymax'] = f"{voisin_y:04d}000"
            
        else:  # place_dalle == 8
            # Place dalle 8 : coin inférieur droit
            xnum = int(f"1{voisin_x:04d}") - 10000
            bounds['xmin'] = f"{xnum:04d}000"
            bounds['xmax'] = f"{xnum:04d}{margin_m:03d}"
            ynum = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymin'] = f"{ynum:04d}000"
            ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
            bounds['ymax'] = f"{ynum2:04d}{margin_m:03d}"
        
        return bounds
    
    def crop_neighbor_tile(self, input_file: str, output_file: str, bounds: Dict[str, str]) -> bool:
        """Rogne une dalle voisine avec PDAL"""
        # Chemins complets : input dans Input, output dans Temp
        input_path = str(self.input_dir / input_file)
        output_path = str(self.temp_dir / output_file)
        
        self.logger.info(f"=== ROGNAGE DALLE VOISINE ===")
        self.logger.info(f"Fichier source: {input_path}")
        self.logger.info(f"Fichier sortie: {output_path}")
        self.logger.info(f"Coordonnées: xmin={bounds['xmin']}, xmax={bounds['xmax']}, ymin={bounds['ymin']}, ymax={bounds['ymax']}")
        
        if Path(output_path).exists():
            self.logger.info(f"Dalle voisine déjà rognée: {output_file}")
            # Valider le fichier existant
            if not self.validate_laz_file(output_path):
                self.logger.warning(f"Fichier voisin existant invalide: {output_file}. Il sera régénéré.")
                try:
                    Path(output_path).unlink()
                except Exception as e:
                    self.logger.warning(f"Impossible de supprimer {output_file}: {e}")
            else:
                return True
        # Continuer pour régénérer si invalide
        
        # Créer le pipeline PDAL pour le rognage
        pipeline_config = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": input_path
                },
                {
                    "type": "filters.crop",
                    "bounds": f"([{bounds['xmin']},{bounds['xmax']}],[{bounds['ymin']},{bounds['ymax']}])"
                },
                {
                    "type": "writers.las",
                    "filename": output_path,
                    "compression": "laszip"
                }
            ]
        }
        
        # Créer un fichier pipeline temporaire
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(pipeline_config, temp_file, indent=2)
            pipeline_file = temp_file.name
        
        try:
            pdal_cmd = self._get_pdal_path()
            cmd = [pdal_cmd, "pipeline", pipeline_file]
            self.logger.info(f"Exécution commande PDAL: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Vérifier que le fichier de sortie a été créé
            if Path(output_path).exists():
                # Validation du fichier généré
                if self.validate_laz_file(output_path):
                    self.logger.info(f"Dalle voisine rognée: {output_file}")
                    return True
                else:
                    self.logger.warning(f"Dalle voisine générée invalide (VLR corrompu?): {output_file}")
                    return False
            else:
                self.logger.error(f"Fichier de sortie non créé: {output_path}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur PDAL lors du rognage de {input_file}:")
            self.logger.error(f"Code de retour: {e.returncode}")
            self.logger.error(f"Commande: {' '.join(cmd)}")
            
            if e.stdout:
                self.logger.error(f"STDOUT PDAL:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR PDAL:\n{e.stderr}")
                
            # Analyser l'erreur pour donner des conseils
            if "No points to write" in str(e.stderr):
                self.logger.warning("Zone de rognage vide - aucun point LIDAR dans cette zone")
            elif "bounds" in str(e.stderr).lower():
                self.logger.warning("Problème de coordonnées - vérifier les limites de rognage")
                
            return False
        finally:
            # Nettoyer le fichier pipeline temporaire
            try:
                os.unlink(pipeline_file)
            except:
                pass

    def validate_laz_file(self, laz_path: str) -> bool:
        """Valide un fichier LAZ/LAS avec PDAL. Retourne True si le fichier est lisible.
        Utilise: pdal info --metadata <file>
        """
        try:
            pdal_cmd = self._get_pdal_path()
            result = subprocess.run([pdal_cmd, "info", "--metadata", laz_path], capture_output=True, text=True)
            if result.returncode == 0:
                return True
            # Diagnostic commun pour VLR corrompu
            err = (result.stderr or "") + "\n" + (result.stdout or "")
            if "VLR" in err and "size too large" in err:
                self.logger.warning(f"Validation PDAL a détecté un VLR corrompu: {Path(laz_path).name}")
            else:
                self.logger.warning(f"Validation PDAL échouée pour {Path(laz_path).name} (code {result.returncode})")
            if result.stderr:
                self.logger.debug(f"PDAL info STDERR ({Path(laz_path).name}):\n{result.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("PDAL introuvable: impossible de valider les fichiers LAZ/LAS. Vérifiez l'installation OSGeo4W/QGIS ou renseignez tools.pdal dans config.json")
            raise
        except Exception as e:
            self.logger.warning(f"Impossible de valider {laz_path} avec PDAL: {e}")
            return False

    def merge_tiles(self, central_path: str, neighbor_files: List[str], output_file: str = "merged.laz") -> bool:
        """Fusionne la dalle centrale avec ses voisines"""
        
        # Vérifier que le fichier source existe
        if not Path(central_path).exists():
            self.logger.error(f"Fichier source introuvable: {central_path}")
            return False
            
        output_path = str(self.temp_dir / output_file)
        
        self.logger.info(f"=== FUSION DES DALLES ===")
        self.logger.info(f"Fichier central: {central_path}")
        self.logger.info(f"Fichier de sortie: {output_path}")
        
        if Path(output_path).exists():
            output_size = Path(output_path).stat().st_size
            self.logger.info(f"Fusion déjà effectuée: {output_path} ({output_size:,} octets, {output_size/1024/1024:.1f} MB)")
            return True
        
        # Vérifier que le fichier central existe
        if not Path(central_path).exists():
            self.logger.error(f"Fichier central introuvable: {central_path}")
            return False
        
        central_size = Path(central_path).stat().st_size
        self.logger.info(f"Taille fichier central: {central_size:,} octets ({central_size/1024/1024:.1f} MB)")
        
        # Construire la liste des fichiers avec chemins complets
        all_files = [central_path]
        total_neighbor_size = 0
        
        self.logger.info(f"Dalles voisines demandées: {len(neighbor_files)}")
        for i, neighbor_file in enumerate(neighbor_files, 1):
            neighbor_path = str(self.temp_dir / neighbor_file)
            if Path(neighbor_path).exists():
                neighbor_size = Path(neighbor_path).stat().st_size
                total_neighbor_size += neighbor_size
                all_files.append(neighbor_path)
                self.logger.info(f"  Voisin {i}: {neighbor_file} ({neighbor_size:,} octets, {neighbor_size/1024/1024:.1f} MB)")
            else:
                self.logger.warning(f"  Voisin {i}: {neighbor_file} - FICHIER MANQUANT")
        
        self.logger.info(f"Fichiers à fusionner: {len(all_files)} (1 central + {len(all_files)-1} voisins)")
        self.logger.info(f"Taille totale estimée: {(central_size + total_neighbor_size):,} octets ({(central_size + total_neighbor_size)/1024/1024:.1f} MB)")
        
        # Valider tous les fichiers avant fusion et ignorer les voisins corrompus
        valid_files = []
        try:
            for f in all_files:
                if self.validate_laz_file(f):
                    valid_files.append(f)
                else:
                    self.logger.warning(f"Fichier invalide ignoré pour fusion: {f}")
            if not self.validate_laz_file(central_path):
                self.logger.error("Le fichier central est invalide (probablement corrompu). Supprimez-le et relancez pour re-télécharger.")
                return False
        except FileNotFoundError:
            if len(all_files) <= 1:
                self.logger.warning("PDAL introuvable: fusion remplacée par une simple copie du fichier central")
                try:
                    shutil.copy2(central_path, output_path)
                    self.logger.info(f"Copie effectuée: {output_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Échec de la copie du central vers la sortie: {e}")
                    return False
            return False
        # Si seulement le central est valide, copier comme sortie et continuer
        if len(valid_files) <= 1:
            self.logger.warning("Aucun voisin valide disponible. La fusion sera remplacée par une simple copie du central.")
            try:
                shutil.copy2(central_path, output_path)
                self.logger.info(f"Copie effectuée: {output_path}")
                return True
            except Exception as e:
                self.logger.error(f"Échec de la copie du central vers la sortie: {e}")
                return False
        
        pdal_cmd = self._get_pdal_path()
        cmd = [pdal_cmd, "merge"] + valid_files + [output_path]
        self.logger.info(f"Commande PDAL: {' '.join(cmd[:3])} [fichiers...] {output_path}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Vérifier le résultat
            if Path(output_path).exists():
                final_size = Path(output_path).stat().st_size
                self.logger.info(f"Fusion réussie: {output_path} ({final_size:,} octets, {final_size/1024/1024:.1f} MB)")
                
                # Analyser le résultat
                expected_min_size = central_size * 0.8  # Au moins 80% de la taille centrale
                if final_size < expected_min_size:
                    self.logger.warning(f"Fichier fusionné plus petit qu'attendu (< {expected_min_size:,} octets)")
                elif final_size > central_size * 1.2:  # Plus de 20% de plus que le central
                    self.logger.info("Fusion semble avoir ajouté des données voisines")
                else:
                    self.logger.warning("Fusion n'a peut-être pas ajouté beaucoup de données voisines")
                    
                return True
            else:
                self.logger.error(f"Fichier de sortie non créé: {output_path}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur PDAL lors de la fusion:")
            self.logger.error(f"Code de retour: {e.returncode}")
            self.logger.error(f"Commande: {' '.join(cmd[:5])}...")
            if e.stdout:
                self.logger.error(f"STDOUT PDAL:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR PDAL:\n{e.stderr}")
            # Lister les fichiers effectivement passés à PDAL et marquer ceux manquants
            try:
                self.logger.info("Diagnostic des entrées de fusion:")
                for idx, f in enumerate(all_files, 1):
                    exists = Path(f).exists()
                    size = Path(f).stat().st_size if exists else 0
                    self.logger.info(f"  [{idx}] {'OK' if exists else 'MISSING'} - {f}{'' if not exists else f' ({size/1024/1024:.1f} MB)'}")
            except Exception as diage:
                self.logger.warning(f"Impossible d'énumérer les entrées: {diage}")
            return False
    
    def crop_mnt_to_tile_bounds(self, x: str, y: str, mnt_file: str) -> bool:
        """Rogne le MNT aux dimensions exactes de la dalle selon la configuration"""
        ymax_r = str(central_y * tile_height + tile_height)
        
        self.logger.info(f"=== ROGNAGE MNT AUX DIMENSIONS EXACTES ===")
        self.logger.info(f"Dalle {x}_{y}: {xmin_r},{ymin_r} -> {xmax_r},{ymax_r} ({tile_width}x{tile_height}m)")
        
        cmd = [
            self.gdalwarp,
            "-te", xmin_r, ymin_r, xmax_r, ymax_r,
            "-overwrite",
            "-dstnodata", "0",
            "-co", "COMPRESS=LZW",
            input_path,
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if Path(output_path).exists():
                # Remplacer le MNT original par le MNT rogné
                Path(input_path).unlink()  # Supprimer l'original
                Path(output_path).rename(input_path)  # Renommer le rogné
                
                self.logger.info(f"MNT rogné aux dimensions exactes: {mnt_file}")
                return True
            else:
                self.logger.error(f"Échec du rognage MNT: fichier de sortie non créé")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur lors du rognage MNT: {e}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            return False
    
    def create_terrain_model(self, input_file: str, output_file: str = None) -> bool:
        """Crée le modèle numérique de terrain (MNT) avec marge artificielle"""
        if output_file is None:
            output_file = f"{self.current_tile_name}_MNT.tif"
        
        # Utiliser des chemins absolus - input dans Temp, output dans Temp
        input_path = str(self.temp_dir / input_file)
        output_path = str(self.temp_dir / output_file)

        # Vérifier l'existence du fichier d'entrée (sinon QGIS échoue avec "source layer not found")
        if not Path(input_path).exists():
            self.logger.error(f"Fichier d'entrée MNT introuvable: {input_path}")
            return False

        # Vérifier que le LAZ d'entrée est lisible (sinon QGIS peut remonter un faux "not found")
        try:
            if not self.validate_laz_file(input_path):
                self.logger.error(f"Fichier d'entrée MNT illisible/corrompu: {input_path}")
                self.logger.error("Action requise: supprimer la dalle/nuage associé(e) et la retélécharger (fichier corrompu ou non ouvrable par PDAL/QGIS).")
                return False
        except FileNotFoundError:
            # PDAL introuvable: validate_laz_file() l'a déjà loggé, on laisse l'échec remonter
            return False
        
        if Path(output_path).exists():
            self.logger.info(f"MNT déjà créé: {output_file}")
            return True
        
        # Récupérer le paramètre de marge
        margin_percent = self.get_config_value('processing', 'tile_overlap', 5) / 100.0
        margin_meters = 1000 * margin_percent  # 1000m = taille de dalle
        
        # Calculer les coordonnées étendues avec marge
        x_km = int(self.current_tile_name.split('_')[2])  # Ex: 928 de "LHD_FXX_0928_6613"
        y_km = int(self.current_tile_name.split('_')[3])  # Ex: 6613
        
        # Coordonnées de base de la dalle 1km x 1km
        base_xmin = x_km * 1000
        base_xmax = (x_km + 1) * 1000
        base_ymin = (y_km - 1) * 1000  # y-1 car y est le coin supérieur
        base_ymax = y_km * 1000
        
        # Étendre avec la marge
        extended_xmin = base_xmin - margin_meters
        extended_xmax = base_xmax + margin_meters
        extended_ymin = base_ymin - margin_meters
        extended_ymax = base_ymax + margin_meters
        
        self.logger.info(f"Création MNT avec marge {margin_percent*100}%")
        self.logger.info(f"Zone étendue: {extended_xmin},{extended_ymin},{extended_xmax},{extended_ymax}")
        
        parameters = {
            "INPUT": input_path,
            "OUTPUT": output_path,
            "RESOLUTION": str(self.get_config_value('processing', 'mnt_resolution', 0.5)),
            "FILTER_EXPRESSION": self.get_config_value('processing', 'filter_expression', "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9"),
            "BOUNDS": f"{extended_xmin},{extended_ymin},{extended_xmax},{extended_ymax}"
        }
        
        # Enregistrer le fichier en cours de création
        self.current_output_file = output_path
        
        success = self.run_qgis_algorithm("pdal:exportrastertin", parameters)
        if success:
            self.logger.info(f"MNT créé: {output_file}")
        else:
            self.logger.error(f"Échec création MNT avec pdal:exportrastertin")
            self.logger.error(f"Paramètres utilisés: {parameters}")
            
            # Fallback: essayer avec pdal:exportraster
            self.logger.info("🔄 Tentative de fallback avec pdal:exportraster...")
            success = self.run_qgis_algorithm("pdal:exportraster", parameters)
            if success:
                self.logger.info(f"✅ MNT créé avec fallback: {output_file}")
            else:
                self.logger.error("❌ Échec création MNT avec fallback pdal:exportraster")
        
        self.current_output_file = None
        return success
    
    def create_density_map(self, input_file: str, output_file: str = None) -> bool:
        """Crée la carte de densité avec marge artificielle"""
        if output_file is None:
            # Toujours générer en .tif
            output_file = f"{self.current_tile_name}_densite.tif"
        
        # Utiliser des chemins absolus - input dans Temp, output dans Temp
        input_path = str(self.temp_dir / input_file)
        output_path = str(self.temp_dir / output_file)
        
        if Path(output_path).exists():
            self.logger.info(f"Carte de densité déjà créée: {output_file}")
            return True
        
        # Récupérer le paramètre de marge
        margin_percent = self.get_config_value('processing', 'tile_overlap', 5) / 100.0
        margin_meters = 1000 * margin_percent
        
        # Calculer les coordonnées étendues avec marge
        x_km = int(self.current_tile_name.split('_')[2])
        y_km = int(self.current_tile_name.split('_')[3])
        
        base_xmin = x_km * 1000
        base_xmax = (x_km + 1) * 1000
        base_ymin = (y_km - 1) * 1000
        base_ymax = y_km * 1000
        
        extended_xmin = base_xmin - margin_meters
        extended_xmax = base_xmax + margin_meters
        extended_ymin = base_ymin - margin_meters
        extended_ymax = base_ymax + margin_meters
        
        self.logger.info(f"Création densité avec marge {margin_percent*100}% ({margin_meters}m)")
        
        parameters = {
            "INPUT": input_path,
            "OUTPUT": output_path,
            "BOUNDS": f"{extended_xmin},{extended_ymin},{extended_xmax},{extended_ymax}",
            "RESOLUTION": str(self.get_config_value('processing', 'density_resolution', 1.0)),
            "FILTER_EXPRESSION": self.get_config_value('processing', 'filter_expression', "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9")
        }
        
        # Note: pdal:density n'a pas de paramètre TILE_SIZE - la taille est déterminée par l'algorithme PDAL
        
        # Enregistrer le fichier en cours de création
        self.current_output_file = output_path
        
        success = self.run_qgis_algorithm("pdal:density", parameters)
        if success:
            self.logger.info(f"Carte de densité créée: {output_file}")
        
        self.current_output_file = None
        return success
    
    def create_visualization_products(self, input_file: str = "MNT.tif") -> Dict[str, bool]:
        """Crée les produits de visualisation (M_HS, SVF, SLO, LD) selon la configuration"""
        results = {}
        
        # Récupérer la configuration des produits depuis l'UI ou le fichier config
        if self.ui_config and 'processing' in self.ui_config and 'products' in self.ui_config['processing']:
            products = self.ui_config['processing']['products']
        else:
            products = self.config['processing'].get('products', {})
        
        if self.ui_config and 'rvt_params' in self.ui_config:
            rvt_params = self.ui_config['rvt_params']
        else:
            rvt_params = self.config.get('rvt_params', {})
        
        # Log de débogage pour vérifier la configuration
        self.logger.info(f"Configuration RVT chargée: {rvt_params}")

        # Utiliser les chemins complets dans le dossier Temp
        input_path = str(self.temp_dir / input_file)
        
        # -------------- Visualisation du relief --------------
        # ------ Creation des M_HS (Multiple Directions Hillshades) -------
        if products.get('M_HS', False):
            self.logger.info(f"Creation du M-HS de la dalle {self.num_dalle}")
            output_file = f"{self.current_tile_name}_hillshade.tif"
            output_path = str(self.temp_dir / output_file)
            
            if not Path(output_path).exists():
                # Utiliser les paramètres MDH de la configuration
                mdh_params = rvt_params.get('mdh', {})
                parameters = {
                    "INPUT": input_path,
                    "OUTPUT": output_path,
                    "NUM_DIRECTIONS": str(mdh_params.get('num_directions', 16)),
                    "SAVE_AS_8BIT": str(mdh_params.get('save_as_8bit', True)),
                    "SUN_ELEVATION": str(mdh_params.get('sun_elevation', 35)),
                    "VE_FACTOR": str(mdh_params.get('ve_factor', 1))
                }
                
                success = self._measure_step(self.current_tile_name, "RVT_M_HS", self.run_qgis_algorithm, "rvt:rvt_multi_hillshade", parameters)
                if success:
                    self.logger.info("fin M-HS")
                    results['M_HS'] = True
                else:
                    results['M_HS'] = False
            else:
                self.logger.info(f"M-HS déjà créé: {output_file}")
                results['M_HS'] = True
        else:
            self.logger.info("M-HS désactivé dans la configuration")
            results['M_HS'] = False
        
        # ------ Creation des SVF (RVT Sky-view factor) -------
        if products.get('SVF', False):
            self.logger.info(f"Creation du SVF de la dalle {self.num_dalle}")
            output_file = f"{self.current_tile_name}_SVF.tif"
            output_path = str(self.temp_dir / output_file)
            
            if not Path(output_path).exists():
                # Utiliser les paramètres SVF de la configuration
                svf_params = rvt_params.get('svf', {})
                parameters = {
                    "INPUT": input_path,
                    "OUTPUT": output_path,
                    "NOISE_REMOVE": str(svf_params.get('noise_remove', 0)),
                    "NUM_DIRECTIONS": str(svf_params.get('num_directions', 16)),
                    "RADIUS": str(svf_params.get('radius', 10)),
                    "SAVE_AS_8BIT": str(svf_params.get('save_as_8bit', True)),
                    "VE_FACTOR": str(svf_params.get('ve_factor', 1))
                }
                
                success = self._measure_step(self.current_tile_name, "RVT_SVF", self.run_qgis_algorithm, "rvt:rvt_svf", parameters)
                if success:
                    self.logger.info("fin SVF")
                    results['SVF'] = True
                else:
                    results['SVF'] = False
            else:
                self.logger.info(f"SVF déjà créé: {output_file}")
                results['SVF'] = True
        else:
            self.logger.info("SVF désactivé dans la configuration")
            results['SVF'] = False
        
        # ------ Creation des SLO (RVT Slope) -------
        if products.get('SLO', False):
            self.logger.info(f"Creation du SLO de la dalle {self.num_dalle}")
            output_file = f"{self.current_tile_name}_Slope.tif"
            output_path = str(self.temp_dir / output_file)
            
            if not Path(output_path).exists():
                # Utiliser les paramètres Slope de la configuration
                slope_params = rvt_params.get('slope', {})
                parameters = {
                    "INPUT": input_path,
                    "OUTPUT": output_path,
                    "UNIT": str(slope_params.get('unit', 0)),
                    "VE_FACTOR": str(slope_params.get('ve_factor', 1)),
                    "SAVE_AS_8BIT": str(slope_params.get('save_as_8bit', True))
                }
                
                success = self._measure_step(self.current_tile_name, "RVT_SLO", self.run_qgis_algorithm, "rvt:rvt_slope", parameters)
                if success:
                    self.logger.info("fin SLO")
                    results['SLO'] = True
                else:
                    results['SLO'] = False
            else:
                self.logger.info(f"SLO déjà créé: {output_file}")
                results['SLO'] = True
        else:
            self.logger.info("SLO désactivé dans la configuration")
            results['SLO'] = False
        
        # ------ Creation des LD (RVT Local Dominance) -------
        if products.get('LD', False):
            # Utiliser un identifiant clair (nom de la dalle) plutôt qu'un compteur
            self.logger.info(f"Creation du LD de la dalle {self.current_tile_name}")
            output_file = f"{self.current_tile_name}_LD.tif"
            output_path = str(self.temp_dir / output_file)
            
            if not Path(output_path).exists():
                # Utiliser les paramètres LDO de la configuration
                ldo_params = rvt_params.get('ldo', {})
                parameters = {
                    "INPUT": input_path,
                    "OUTPUT": output_path,
                    "ANGULAR_RES": str(ldo_params.get('angular_res', 15)),
                    "MIN_RADIUS": str(ldo_params.get('min_radius', 10)),
                    "MAX_RADIUS": str(ldo_params.get('max_radius', 20)),
                    "OBSERVER_H": str(ldo_params.get('observer_h', 1.7)),
                    "VE_FACTOR": str(ldo_params.get('ve_factor', 1)),
                    "SAVE_AS_8BIT": str(ldo_params.get('save_as_8bit', True))
                }
                
                success = self._measure_step(self.current_tile_name, "RVT_LD", self.run_qgis_algorithm, "rvt:rvt_ld", parameters)
                if success:
                    self.logger.info("fin LD")
                    results['LD'] = True
                else:
                    results['LD'] = False
            else:
                self.logger.info(f"LD déjà créé: {output_file}")
                results['LD'] = True
        else:
            self.logger.info("LD désactivé dans la configuration")
            results['LD'] = False
        
        # ------ VAT (Visualization for Archaeological Topography) -------
        if products.get('VAT', False):
            self.logger.info(f"Creation du VAT de la dalle {self.current_tile_name}")

            # Paramètres VAT spécifiques (avec valeurs par défaut raisonnables)
            vat_params = rvt_params.get('vat', {})

            # Type de terrain (0: general, 1: flat, 2: steep)
            terrain_type = str(vat_params.get('terrain_type', 0))

            # Combinaison de couches RVT utilisée par Blender (0 = preset par défaut)
            blend_combination = str(vat_params.get('blend_combination', 0))

            # Sauvegarde en 8bit contrôlée par la config
            save_as_8bit = str(vat_params.get('save_as_8bit', True))

            # On encode le preset dans les noms intermédiaires pour distinguer les caches
            preset_suffix = f"_T{terrain_type}_B{blend_combination}"

            # Chemin de base pour la sortie VAT (le plugin ajoute suffixes/extension)
            vat_output_base = self.temp_dir / f"{self.current_tile_name}_VAT{preset_suffix}_outputs"

            # Si un TIF VAT standard pour ce preset existe déjà, ne pas tout recalculer
            standard_vat_tif = self.temp_dir / f"{self.current_tile_name}_VAT{preset_suffix}.tif"
            if not standard_vat_tif.exists():

                parameters = {
                    # Chemin MNT étendu en entrée
                    "INPUT": input_path,
                    # Paramètres globaux de l'algorithme RVT Blender
                    "distance_units": "meters",
                    "area_units": "m2",
                    "ellipsoid": "EPSG:7019",
                    # Configuration VAT
                    "BLEND_COMBINATION": blend_combination,
                    "TERRAIN_TYPE": terrain_type,
                    # Production en 8 bits selon la config
                    "SAVE_AS_8BIT": save_as_8bit,
                    # On désactive systématiquement la sortie float
                    "SAVE_AS_FLOAT": "false",
                    # Chemin de base de sortie; le plugin ajoute les suffixes/extension
                    "OUTPUT": str(vat_output_base),
                }

                success = self._measure_step(self.current_tile_name, "RVT_VAT", self.run_qgis_algorithm, "rvt:rvt_blender", parameters)
                if success:
                    self.logger.info("fin VAT")

                    # Le plugin RVT-QGIS crée typiquement un fichier *_VAT*_outputs_8bit.tif dans Temp
                    expected_tif = Path(str(vat_output_base) + "_8bit.tif")
                    if expected_tif.exists():
                        try:
                            shutil.copy2(str(expected_tif), str(standard_vat_tif))
                            self.logger.info(f"VAT TIF standardisé pour le pipeline: {standard_vat_tif} (source: {expected_tif.name})")
                            results['VAT'] = True
                        except Exception as e:
                            self.logger.error(f"Erreur lors de la copie du VAT TIF: {e}")
                            results['VAT'] = False
                    else:
                        # Fallback: chercher d'autres TIF associés au préfixe VAT_outputs pour ce preset
                        candidates = sorted(self.temp_dir.glob(f"{self.current_tile_name}_VAT{preset_suffix}_outputs*.tif"))
                        if candidates:
                            try:
                                shutil.copy2(str(candidates[0]), str(standard_vat_tif))
                                self.logger.info(f"VAT TIF standardisé (fallback) pour le pipeline: {standard_vat_tif} (source: {candidates[0].name})")
                                results['VAT'] = True
                            except Exception as e:
                                self.logger.error(f"Erreur lors de la copie du VAT TIF (fallback): {e}")
                                results['VAT'] = False
                        else:
                            self.logger.warning(f"Aucun TIF VAT*_outputs* trouvé dans Temp après exécution: préfixe={vat_output_base}")
                            results['VAT'] = False

                else:
                    results['VAT'] = False
            else:
                self.logger.info(f"VAT déjà créé pour le preset terrain_type={terrain_type}, blend_combination={blend_combination}: {standard_vat_tif}")
                results['VAT'] = True
        else:
            results['VAT'] = False
        
        self.logger.info("debut rognage final")
        return results
    
    
    def convert_mdh_to_rgb(self, input_file: str, output_file: str) -> bool:
        """Convertit un MDH en niveaux de gris vers RGB avec palette de couleurs"""
        try:
            
            # Vérifier si GDAL est disponible pour la conversion
            gdal_translate = shutil.which('gdal_translate')
            gdaldem = shutil.which('gdaldem')
            
            if gdal_translate and gdaldem:
                # Étape 1: Normaliser les valeurs avec gdal_translate
                temp_normalized = str(Path(output_file).with_suffix('.temp.tif'))
                
                cmd_normalize = [
                    gdal_translate,
                    '-of', 'GTiff',
                    '-ot', 'Byte',
                    '-scale',  # Normaliser automatiquement vers 0-255
                    input_file,
                    temp_normalized
                ]
                
                result1 = subprocess.run(cmd_normalize, capture_output=True, text=True)
                
                if result1.returncode == 0:
                    # Étape 2: Appliquer une palette de couleurs terrain avec gdaldem
                    cmd_color = [
                        gdaldem,
                        'color-relief',
                        temp_normalized,
                        self.create_color_palette(),
                        output_file,
                        '-of', 'GTiff'
                    ]
                    
                    result2 = subprocess.run(cmd_color, capture_output=True, text=True)
                    
                    # Nettoyer le fichier temporaire
                    try:
                        Path(temp_normalized).unlink()
                    except:
                        pass
                    
                    if result2.returncode == 0:
                        self.logger.info(f"MDH converti en RGB avec palette terrain: {Path(output_file).name}")
                        return True
                    else:
                        self.logger.warning(f"Échec application palette MDH: {result2.stderr}")
                else:
                    self.logger.warning(f"Échec normalisation MDH: {result1.stderr}")
            
            # Fallback: copier le fichier tel quel
            shutil.copy2(input_file, output_file)
            self.logger.info(f"MDH copié sans conversion RGB (GDAL indisponible): {Path(output_file).name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion MDH RGB: {e}")
            return False
    
    def create_color_palette(self) -> str:
        """Crée un fichier de palette de couleurs pour le terrain"""
        palette_content = """0 0 0 0
50 139 69 19
100 205 133 63
150 222 184 135
200 245 245 220
255 255 255 255"""
        
        palette_file = self.temp_dir / "terrain_palette.txt"
        palette_file.write_text(palette_content)
        return str(palette_file)
    
    def extract_tif_transform_data(self, input_tif: str) -> tuple:
        """Extrait les données de géoréférencement d'un fichier TIF"""
        try:
            # Essayer d'utiliser rasterio en priorité
            try:
                import rasterio
                with rasterio.open(input_tif) as dataset:
                    transform = dataset.transform
                    pixel_width = transform.a
                    pixel_height = transform.e  # négatif
                    x_origin = transform.c
                    y_origin = transform.f
                    return pixel_width, pixel_height, x_origin, y_origin
                    
            except ImportError:
                # Fallback vers GDAL
                try:
                    from osgeo import gdal
                    dataset = gdal.Open(input_tif)
                    if dataset:
                        geotransform = dataset.GetGeoTransform()
                        if geotransform and geotransform != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                            x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height = geotransform
                            dataset = None
                            return pixel_width, pixel_height, x_origin, y_origin
                        dataset = None
                except ImportError:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Impossible d'extraire les données de géoréférencement de {input_tif}: {e}")
            
        return None, None, None, None

    def convert_tif_to_jpg(self, input_tif: str, output_jpg: str, product_type: str = "", quality: int = 95, reference_tif_path: str = None) -> bool:
        """Convertit un fichier TIF en JPG en utilisant la fonction du module convert_tif_to_jpg"""
        try:
            # S'assurer que le répertoire de sortie existe
            output_path = Path(output_jpg)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extraire et stocker les données de géoréférencement pour conversion_shp.py
            # Utiliser le TIF de référence si fourni, sinon le TIF source
            ref_tif = reference_tif_path if reference_tif_path else input_tif
            base_name = Path(output_jpg).stem
            pixel_width, pixel_height, x_origin, y_origin = self.extract_tif_transform_data(ref_tif)
            if all(v is not None for v in [pixel_width, pixel_height, x_origin, y_origin]):
                # Stocker dans le dictionnaire global pour conversion_shp.py
                self.tif_transform_data[base_name] = (pixel_width, pixel_height, x_origin, y_origin)
                self.logger.debug(f"Données de géoréférencement stockées pour {base_name} (référence: {Path(ref_tif).name})")
            
            # Import dynamique de la fonction convert_tif_to_jpg
            import importlib.util
            convert_module_path = Path(__file__).parent / "convert_tif_to_jpg.py"
            spec = importlib.util.spec_from_file_location("convert_tif_to_jpg", convert_module_path)
            convert_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(convert_module)
            
            # Utiliser la fonction existante du module convert_tif_to_jpg avec création du fichier world
            # Passer le TIF de référence pour le géoréférencement
            success = convert_module.convert_tif_to_jpg(input_tif, output_jpg, quality, create_world_file=True, reference_tif_path=reference_tif_path)
            
            if success:
                self.logger.info(f"Conversion réussie: {output_jpg} (qualité: {quality})")
            else:
                self.logger.error(f"Échec de la conversion: {input_tif} -> {output_jpg}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion TIF->JPG: {e}")
            return False
    
    def apply_computer_vision_if_enabled(self, jpg_path: str, product_name: str) -> bool:
        """
{{ ... }}
        Applique la détection par computer vision si elle est activée et sur le bon RVT
        
        Args:
            jpg_path (str): Chemin vers l'image JPG créée
            product_name (str): Nom du produit (MDH, SVF, SLO, LDO, etc.)
        
        Returns:
            bool: True si la détection s'est bien déroulée ou n'était pas nécessaire, False en cas d'erreur
        """
        try:
            # Récupérer la configuration CV depuis l'UI
            cv_config = self.get_config_value('computer_vision')
            cv_enabled = cv_config.get('enabled', False)
            
            if not cv_enabled:
                return True  # Pas d'erreur, juste pas activé
            
            # Vérifier si c'est le bon RVT cible
            target_rvt = cv_config.get('target_rvt', 'MDH')
            if product_name != target_rvt:
                return True  # Pas le bon RVT, pas d'erreur
            
            # Récupérer le modèle sélectionné
            selected_model = cv_config.get('selected_model', '')
            if not selected_model:
                self.logger.warning("Computer Vision activée mais aucun modèle sélectionné")
                return False
            
            # Récupérer le seuil de confiance
            confidence_threshold = cv_config.get('confidence_threshold', 0.3)
            
            # Récupérer le seuil IoU
            iou_threshold = cv_config.get('iou_threshold', 0.5)
            
            # Récupérer l'option pour générer des images annotées
            generate_annotated_images = cv_config.get('generate_annotated_images', False)
            
            # Récupérer l'option pour générer des shapefiles
            generate_shapefiles = cv_config.get('generate_shapefiles', False)
            
            # Récupérer les paramètres SAHI et les figer pour éviter les modifications pendant l'exécution
            sahi_config = cv_config.get('sahi', {})
            slice_height = int(sahi_config.get('slice_height', 750))  # Conversion explicite en int
            slice_width = int(sahi_config.get('slice_width', 750))   # Conversion explicite en int
            overlap_ratio = float(sahi_config.get('overlap_ratio', 0.2))  # Conversion explicite en float
            
            # Log des paramètres figés pour debug
            self.log_to_ui(f"📏 Paramètres SAHI figés: {slice_height}x{slice_width}, overlap={overlap_ratio}")
            
            # Construire les chemins vers le modèle
            models_dir = cv_config.get('models_dir', 'models')
            model_dir = Path(models_dir) / selected_model
            model_path = model_dir / "weights" / "best.pt"
            args_path = model_dir / "args.yaml"
            
            # Vérifier que les fichiers du modèle existent
            if not model_path.exists():
                self.logger.error(f"Fichier de poids du modèle non trouvé: {model_path}")
                return False
            
            if not args_path.exists():
                self.logger.error(f"Fichier de configuration du modèle non trouvé: {args_path}")
                return False
            
            # Importer le module de computer vision
            from . import computer_vision
            
            # Générer le chemin de sortie pour les détections
            detection_path = computer_vision.get_detection_output_path(jpg_path, target_rvt)
            
            # Périmètre de traitement CV: par défaut seulement l'image courante; option scan_all pour tout le dossier
            jpg_dir = Path(jpg_path).parent
            scan_all = bool(cv_config.get('scan_all', False))
            if scan_all:
                jpg_files = sorted(jpg_dir.glob("*.jpg"))
                self.logger.info(f"🔎 CV scan_all activé: {len(jpg_files)} images à analyser dans {jpg_dir}")
            else:
                jpg_files = [Path(jpg_path)]
                self.logger.info(f"🔎 CV limité à l'image courante: {Path(jpg_path).name}")
            
            # Créer les répertoires de sortie si nécessaire
            annotated_output_dir = None
            shapefile_output_dir = None
            # Répertoire de base pour les sorties RVT (annotations, shapefiles)
            # Par défaut: parent du dossier JPG, mais peut être surchargé (ex: mode existing_rvt)
            rvt_base_dir = getattr(self, 'override_rvt_base_dir', jpg_dir.parent)
            
            if generate_annotated_images:
                # Créer le répertoire annotated_images dans RVT/TYPE (ex: RVT/LDO/annotated_images)
                annotated_output_dir = rvt_base_dir / "annotated_images"
                annotated_output_dir.mkdir(exist_ok=True)
                self.logger.info(f"📁 Répertoire créé pour les images annotées: {annotated_output_dir}")
            
            if generate_shapefiles:
                # Créer le répertoire shapefiles dans RVT/TYPE (ex: RVT/LDO/shapefiles)
                shapefile_output_dir = rvt_base_dir / "shapefiles"
                shapefile_output_dir.mkdir(exist_ok=True)
                self.logger.info(f"📁 Répertoire créé pour les shapefiles: {shapefile_output_dir}")

                # Mémoriser pour déduplication finale (après toutes les dalles)
                self._cv_labels_dir_for_dedup = str(jpg_dir)
                self._cv_shapefile_output_dir_for_dedup = str(shapefile_output_dir)
                self._cv_crs_for_dedup = "EPSG:2154"
            
            success_count = 0
            skipped_already_processed = 0
            for jpg_file in jpg_files:
                jpg_path_current = str(jpg_file)
                detection_output_path = computer_vision.get_detection_output_path(jpg_path_current, target_rvt, str(annotated_output_dir) if annotated_output_dir else None)
                
                # Skip si déjà traité: présence d'une image annotée OU d'un .json/.txt correspondant
                image_name = jpg_file.stem
                labels_txt = (jpg_dir / f"{image_name}.txt")
                labels_json = (jpg_dir / f"{image_name}.json")
                annotated_img = Path(detection_output_path)
                if annotated_img.exists() or labels_txt.exists() or labels_json.exists():
                    self.logger.info(f"⏭️ Détection déjà présente, on saute: {jpg_file.name}")
                    skipped_already_processed += 1
                    continue
                
                self.logger.info(f"🔍 Inférence CV sur: {jpg_file.name}")
                
                success = computer_vision.run_inference(
                    image_path=jpg_path_current,
                    model_path=str(model_path),
                    args_path=str(args_path),
                    output_path=detection_output_path,
                    confidence_threshold=confidence_threshold,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_ratio=overlap_ratio,
                    generate_annotated_images=generate_annotated_images,
                    annotated_output_dir=str(annotated_output_dir) if annotated_output_dir else None,
                    iou_threshold=iou_threshold,
                    jpg_folder_path=str(jpg_dir)
                )
                
                if success:
                    success_count += 1
                    self.logger.info(f"✅ Détection CV terminée: {jpg_file.name}")
                else:
                    # Pas de détection n'est pas une erreur bloquante, on loggue en info pour réduire le bruit
                    self.logger.info(f"ℹ️ Aucune détection (ou inférence sans sortie) pour: {jpg_file.name}")
            
            self.logger.info(f"🎯 Computer Vision terminée: {success_count}/{len(jpg_files)} images traitées avec succès")

            # Cas particulier: en mode scan_all, si toutes les images ont déjà des résultats
            # (annotated image ou fichiers de labels présents), on considère que c'est un succès
            # logique même si success_count == 0.
            if success_count == 0 and scan_all and skipped_already_processed == len(jpg_files):
                self.logger.info("ℹ️ Toutes les images avaient déjà des détections, aucune nouvelle inférence nécessaire")
                return True

            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'application de la computer vision: {e}")
            return False

    def _deduplicate_cv_shapefiles_final(self) -> None:
        """Déduplication des shapefiles CV à exécuter une seule fois en fin de pipeline."""
        try:
            cv_config = self.get_config_value('computer_vision')
            if not cv_config or not cv_config.get('enabled', False):
                return
            if not cv_config.get('generate_shapefiles', False):
                return

            labels_dir = getattr(self, '_cv_labels_dir_for_dedup', None)
            shp_dir = getattr(self, '_cv_shapefile_output_dir_for_dedup', None)
            crs = getattr(self, '_cv_crs_for_dedup', 'EPSG:2154')

            # Fallback: même si aucune nouvelle inférence n'a eu lieu (shapefiles déjà présents),
            # reconstruire les chemins attendus dans results à partir de la configuration.
            if not labels_dir or not shp_dir:
                try:
                    target_rvt = cv_config.get('target_rvt', 'LD')
                    output_structure = self.get_config_value('processing', 'output_structure', {})
                    rvt_cfg = output_structure.get('RVT', {}) if isinstance(output_structure, dict) else {}
                    base_dir_name = rvt_cfg.get('base_dir', 'RVT')
                    type_dir_name = rvt_cfg.get(target_rvt, target_rvt)
                    rvt_base = self.results_dir / base_dir_name / type_dir_name
                    fallback_labels_dir = rvt_base / 'jpg'
                    fallback_shp_dir = rvt_base / 'shapefiles'
                    if not labels_dir:
                        labels_dir = str(fallback_labels_dir)
                    if not shp_dir:
                        shp_dir = str(fallback_shp_dir)
                except Exception:
                    pass

            if not labels_dir or not shp_dir:
                return

            from pathlib import Path
            shp_dir_path = Path(shp_dir)
            if not shp_dir_path.exists():
                return

            shapefile_paths = [str(p) for p in shp_dir_path.glob('*.shp')]
            
            from . import conversion_shp

            # Générer les shapefiles une seule fois en fin de pipeline (même si tous les outputs existaient déjà)
            try:
                target_rvt = cv_config.get('target_rvt', 'LD')
                shapefile_path = shp_dir_path / f"detections_{target_rvt}.shp"
                
                class_names = None
                try:
                    from . import computer_vision
                    models_dir = cv_config.get('models_dir', 'models')
                    selected_model = cv_config.get('selected_model', '')
                    if selected_model:
                        from pathlib import Path as _Path
                        model_dir = _Path(models_dir) / selected_model
                        model_path = model_dir / "weights" / "best.pt"
                        if model_path.exists():
                            class_names = computer_vision.get_class_names_from_model(str(model_path))
                except Exception as e:
                    self.logger.warning(f"Impossible de récupérer les noms des classes depuis le modèle: {e}")
                
                if not class_names:
                    class_names = {0: "four_charbonnier"}

                conversion_shp.create_shapefile_from_detections(
                    labels_dir=str(labels_dir),
                    output_shapefile=str(shapefile_path),
                    tif_transform_data=self.tif_transform_data,
                    crs=str(crs),
                    temp_dir=str(self.temp_dir),
                    class_names=class_names,
                )
            except Exception as e:
                self.logger.warning(f"Génération finale des shapefiles ignorée (erreur): {e}")

            shapefile_paths = [str(p) for p in shp_dir_path.glob('*.shp')]
            if not shapefile_paths:
                return

            # Nécessaire pour construire les emprises: il faut des JPG/JGW dans labels_dir.
            labels_dir_path = Path(labels_dir)
            if not labels_dir_path.exists():
                return

            self.logger.info("🧹 Déduplication finale des shapefiles (IoU>0.1) en fin de pipeline...")
            conversion_shp.deduplicate_shapefiles_final(
                labels_dir=str(labels_dir),
                shapefile_paths=shapefile_paths,
                iou_threshold=0.1,
                crs=crs,
            )
            self.logger.info("✅ Déduplication finale shapefiles terminée")
        except Exception as e:
            self.logger.warning(f"Déduplication finale shapefiles ignorée (erreur): {e}")
    
    def crop_final_products(self, x: str, y: str, neighbors: List[Tuple[int, int, int]]) -> bool:
        """Rogne les produits finaux en enlevant les marges selon le pourcentage configuré"""
        
        self.logger.info(f"=== DÉBUT ROGNAGE FINAL DALLE {x}_{y} ===")
        self.logger.info(f"Nombre de positions voisines considérées: {len(neighbors)}")
        
        # Rognage fixe 1000x1000m comme dans l'ancien pipeline (pas de marges)
        self.logger.info("Rognage fixe 1000x1000m (pas de marges)")
        
        # DEBUG: Log des coordonnées extraites
        self.logger.info(f"DEBUG: Coordonnées extraites du nom - x='{x}', y='{y}'")
        self.logger.info(f"DEBUG: Type x={type(x)}, Type y={type(y)}")
        
        # Référence : utiliser le MNT pour déterminer les coordonnées réelles du fichier source
        reference_file = f"{self.current_tile_name}_MNT.tif"
        reference_path = str(self.temp_dir / reference_file)
        
        # Extraire les coordonnées réelles du fichier source étendu
        bounds = self.get_raster_bounds(reference_path)
        if not bounds:
            self.logger.error(f"Impossible d'extraire les coordonnées de {reference_file}")
            return False
        
        xmin_source, ymin_source, xmax_source, ymax_source = bounds
        self.logger.info(f"Coordonnées source étendues: {xmin_source}, {ymin_source}, {xmax_source}, {ymax_source}")
        
        # Calculer les coordonnées de rognage fixes 1000x1000m (comme ancien pipeline)
        # x et y sont au format "0804" et "6341" représentant les kilomètres
        # Pour "0804" -> dalle va de 804000 à 805000
        # Pour "6341" -> dalle va de 6340000 à 6341000 (y représente le coin supérieur)
        
        try:
            # Convertir les coordonnées du nom de fichier en entiers
            x_km = int(x)  # Ex: 804 pour "0804"
            y_km = int(y)  # Ex: 6341 pour "6341"
            
            # Calculer les coordonnées exactes de la dalle de 1km x 1km (SANS marges)
            # IMPORTANT: y représente le coin SUPÉRIEUR, donc pour y=6341:
            # - ymin = 6340000 (y-1) * 1000
            # - ymax = 6341000 (y) * 1000
            target_xmin = x_km * 1000
            target_xmax = (x_km + 1) * 1000
            target_ymin = (y_km - 1) * 1000  # y-1 car y est le coin supérieur
            target_ymax = y_km * 1000
            
            self.logger.info(f"Coordonnées dalle centrale calculées depuis nom de fichier:")
            self.logger.info(f"  X: {x_km} -> {target_xmin} à {target_xmax}")
            self.logger.info(f"  Y: {y_km} -> {target_ymin} à {target_ymax}")
            
        except ValueError as e:
            self.logger.error(f"Erreur conversion coordonnées: x='{x}', y='{y}' - {e}")
            return False
        
        # Vérifier que les coordonnées calculées sont cohérentes avec le fichier source
        source_center_x = (xmin_source + xmax_source) / 2
        source_center_y = (ymin_source + ymax_source) / 2
        target_center_x = (target_xmin + target_xmax) / 2
        target_center_y = (target_ymin + target_ymax) / 2
        
        # Calculer la distance entre les centres
        distance_x = abs(source_center_x - target_center_x)
        distance_y = abs(source_center_y - target_center_y)
        
        self.logger.info(f"Vérification cohérence:")
        self.logger.info(f"  Centre source: ({source_center_x}, {source_center_y})")
        self.logger.info(f"  Centre cible: ({target_center_x}, {target_center_y})")
        self.logger.info(f"  Distance: X={distance_x}m, Y={distance_y}m")
        
        # Si la distance est trop importante (>600m), il y a probablement un problème
        if distance_x > 600 or distance_y > 600:
            self.logger.warning(f"Distance importante entre centres (X={distance_x}m, Y={distance_y}m)")
            self.logger.warning("Cela peut indiquer un problème de coordonnées")
            
            # Utiliser une approche de fallback basée sur le centre du fichier source
            self.logger.info("Utilisation du fallback basé sur le centre du fichier source")
            
            # Arrondir le centre source aux kilomètres les plus proches
            estimated_x_km = round(source_center_x / 1000)
            estimated_y_km = round(source_center_y / 1000)
            
            target_xmin = estimated_x_km * 1000 - 500
            target_xmax = estimated_x_km * 1000 + 500
            target_ymin = estimated_y_km * 1000 - 500
            target_ymax = estimated_y_km * 1000 + 500
            
            # Ajuster pour avoir des coordonnées rondes
            target_xmin = (target_xmin // 1000) * 1000
            target_xmax = target_xmin + 1000
            target_ymin = (target_ymin // 1000) * 1000
            target_ymax = target_ymin + 1000
            
            self.logger.info(f"Coordonnées fallback: {target_xmin}, {target_ymin}, {target_xmax}, {target_ymax}")
        
        # Vérifier que les coordonnées de rognage sont valides
        if target_xmin >= target_xmax or target_ymin >= target_ymax:
            self.logger.error(f"Coordonnées de rognage invalides: xmin={target_xmin} >= xmax={target_xmax} ou ymin={target_ymin} >= ymax={target_ymax}")
            self.logger.error(f"Source: {xmin_source}-{xmax_source}, {ymin_source}-{ymax_source}")
            self.logger.error(f"Cible: {target_xmin}-{target_xmax}, {target_ymin}-{target_ymax}")
            return False
        
        # Vérifier que les dimensions sont exactement 1000m x 1000m
        width = target_xmax - target_xmin
        height = target_ymax - target_ymin
        if width != 1000 or height != 1000:
            self.logger.error(f"Dimensions incorrectes: {width}x{height}m (attendu: 1000x1000m)")
            return False
        
        # Vérifier que la zone de rognage est contenue dans le fichier source
        if (target_xmin < xmin_source or target_xmax > xmax_source or 
            target_ymin < ymin_source or target_ymax > ymax_source):
            self.logger.error("Zone de rognage en dehors du fichier source:")
            self.logger.error(f"  Rognage: {target_xmin}-{target_xmax}, {target_ymin}-{target_ymax}")
            self.logger.error(f"  Source:  {xmin_source}-{xmax_source}, {ymin_source}-{ymax_source}")
            return False
        
        # Utiliser les coordonnées calculées
        xmin_r = str(target_xmin)
        xmax_r = str(target_xmax)
        ymin_r = str(target_ymin)
        ymax_r = str(target_ymax)
        
        self.logger.info(f"Rognage final pour dalle centrale {x}_{y}:")
        self.logger.info(f"Coordonnées de rognage finales: xmin={xmin_r}, ymin={ymin_r}, xmax={xmax_r}, ymax={ymax_r}")
        
        # Récupérer la configuration depuis l'UI ou le fichier config
        if self.ui_config and 'processing' in self.ui_config and 'products' in self.ui_config['processing']:
            products = self.ui_config['processing']['products']
        else:
            products = self.config['processing'].get('products', {})
        output_format = self.config['processing'].get('output_image_format', '.tif')

        # Récupérer aussi les paramètres RVT pour connaître le preset VAT courant
        # (terrain_type / blend_combination) afin d'utiliser le bon fichier source en Temp
        rvt_params = self.get_config_value('rvt_params')
        vat_params = (rvt_params or {}).get('vat', {})
        terrain_type = str(vat_params.get('terrain_type', 0))
        blend_combination = str(vat_params.get('blend_combination', 0))
        preset_suffix = f"_T{terrain_type}_B{blend_combination}"

        # Définir tous les fichiers possibles à rogner
        # Tuples (fichier_source_temp, fichier_sortie_rogné)
        all_files = {
            'MNT': (f"{self.current_tile_name}_MNT.tif", f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif"),
            'M_HS': (f"{self.current_tile_name}_hillshade.tif", f"LHD_FXX_{x}_{y}_M-HS_A_LAMB93.tif"),
            'SVF': (f"{self.current_tile_name}_SVF.tif", f"LHD_FXX_{x}_{y}_SVF_A_LAMB93.tif"),
            'SLO': (f"{self.current_tile_name}_Slope.tif", f"LHD_FXX_{x}_{y}_SLO_A_LAMB93.tif"),
            'LD': (f"{self.current_tile_name}_LD.tif", f"LHD_FXX_{x}_{y}_LD_A_LAMB93.tif"),
            'DENSITE': (f"{self.current_tile_name}_densite.tif", f"LHD_FXX_{x}_{y}_densite_A_LAMB93.tif"),
            # Pour VAT, utiliser le TIF standardisé par preset généré plus haut
            'VAT': (f"{self.current_tile_name}_VAT{preset_suffix}.tif", f"LHD_FXX_{x}_{y}_VAT_A_LAMB93.tif"),
        }
        
        # Créer la liste des fichiers à rogner selon la configuration
        files_to_crop = []
        
        # Ajouter tous les produits activés
        for product_name in ['MNT', 'DENSITE', 'M_HS', 'SVF', 'SLO', 'LD', 'VAT']:
            if products.get(product_name, False):
                files_to_crop.append(all_files[product_name])
                self.logger.info(f"Produit {product_name} activé, sera rogné")
            else:
                self.logger.info(f"Produit {product_name} désactivé, ignoré")
        
        success = True
        for input_file, output_name in files_to_crop:
            # Chemins complets : input dans Temp, output dans Temp (sera déplacé par copy_final_products)
            input_path = str(self.temp_dir / input_file)
            output_path = self.temp_dir / output_name
            
            self.logger.info(f"=== ROGNAGE PRODUIT: {input_file} ===")
            self.logger.info(f"Fichier source: {input_path}")
            self.logger.info(f"Fichier sortie: {output_path}")
            self.logger.info(f"Source existe: {Path(input_path).exists()}")
            self.logger.info(f"Sortie existe: {output_path.exists()}")
            
            if not output_path.exists() and Path(input_path).exists():
                self.logger.info(f"Rognage de {input_path} vers {output_path}")
                self.logger.info(f"Extent: {xmin_r} {ymin_r} {xmax_r} {ymax_r}")
                
                # Utiliser les mêmes paramètres que le script batch original
                # Format: gdalwarp -te xmin ymin xmax ymax input output -of GTiff
                cmd = [
                    self.gdalwarp,
                    "-te", xmin_r, ymin_r, xmax_r, ymax_r,
                    input_path,
                    str(output_path),
                    "-of", "GTiff"
                ]
                
                # Log spécifique selon le type de produit
                if "M-HS" in input_file or "hillshade" in input_file:
                    self.logger.info("Rognage M-HS (comme script batch)")
                elif "densite" in input_file:
                    self.logger.info("Rognage DENSITE (comme script batch)")
                elif any(rvt_product in input_file for rvt_product in ["SVF", "SLO", "LD"]):
                    self.logger.info("Rognage RVT (comme script batch)")
                elif "MNT" in input_file:
                    self.logger.info("Rognage MNT (comme script batch)")
                else:
                    self.logger.info("Rognage standard (comme script batch)")
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self.logger.info(f"Produit final créé: {output_path}")
                    if result.stderr:
                        self.logger.warning(f"GDALWARP stderr: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Erreur lors du rognage de {input_file}: {e}")
                    if e.stdout:
                        self.logger.error(f"STDOUT: {e.stdout}")
                    if e.stderr:
                        self.logger.error(f"STDERR: {e.stderr}")
                    success = False
            elif output_path.exists():
                self.logger.info(f"Produit final déjà existant: {output_path}")
            elif not Path(input_path).exists():
                self.logger.warning(f"Fichier source manquant pour rognage: {input_path}")
                # Cas particulier: en mode existing_mnt, il est possible que la densité
                # n'ait pas été générée; ne pas faire échouer tout le rognage pour ça.
                if "densite" not in input_file:
                    success = False
            else:
                self.logger.warning(f"Condition de rognage non remplie pour {input_file}")
        
        self.logger.info(f"=== FIN ROGNAGE FINAL DALLE {x}_{y} - RÉSULTAT: {'SUCCÈS' if success else 'ÉCHEC'} ===")
        return success
    
    def copy_final_products(self, x: str, y: str) -> None:
        """Copie les produits finaux dans la structure organisée selon la configuration"""
        # Récupérer la configuration des produits depuis l'UI ou le fichier config
        if self.ui_config and 'processing' in self.ui_config and 'products' in self.ui_config['processing']:
            products = self.ui_config['processing']['products']
        else:
            products = self.config['processing'].get('products', {})
        
        # Récupérer la configuration de sortie
        output_structure = self.config['processing'].get('output_structure', {})
        output_formats = self.config['processing'].get('output_formats', {})
        
        self.logger.info(f"TIF: toujours généré et rogné")
        self.logger.info(f"JPG: utilisation des versions NON rognées depuis Temp")
        
        success = True
        
        # Définir les noms de fichiers sources
        # TIF rognés (pour les fichiers TIF finaux) et non rognés (pour les JPG)
        source_files_cropped = {
            'MNT': f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif",
            'DENSITE': f"LHD_FXX_{x}_{y}_densite_A_LAMB93.tif",
            'M_HS': f"LHD_FXX_{x}_{y}_M-HS_A_LAMB93.tif",
            'SVF': f"LHD_FXX_{x}_{y}_SVF_A_LAMB93.tif",
            'SLO': f"LHD_FXX_{x}_{y}_SLO_A_LAMB93.tif",
            'LD': f"LHD_FXX_{x}_{y}_LD_A_LAMB93.tif",
            'VAT': f"LHD_FXX_{x}_{y}_VAT_A_LAMB93.tif",
        }

        # Pour les sources NON rognées, on utilise les fichiers intermédiaires de Temp
        # Pour VAT, il faut tenir compte du preset courant (terrain_type / blend_combination)
        rvt_params = self.get_config_value('rvt_params')
        vat_params = (rvt_params or {}).get('vat', {})
        terrain_type = str(vat_params.get('terrain_type', 0))
        blend_combination = str(vat_params.get('blend_combination', 0))
        preset_suffix = f"_T{terrain_type}_B{blend_combination}"

        source_files_uncropped = {
            'MNT': f"{self.current_tile_name}_MNT.tif",
            'DENSITE': f"{self.current_tile_name}_densite.tif",
            'M_HS': f"{self.current_tile_name}_hillshade.tif",  # M-HS est créé comme hillshade
            'SVF': f"{self.current_tile_name}_SVF.tif",
            'SLO': f"{self.current_tile_name}_Slope.tif",
            'LD': f"{self.current_tile_name}_LD.tif",
            # Pour VAT, utiliser le TIF standardisé par preset généré dans Temp
            'VAT': f"{self.current_tile_name}_VAT{preset_suffix}.tif",
        }
        
        # Traiter chaque produit activé
        for product_name in ['MNT', 'DENSITE', 'M_HS', 'SVF', 'SLO', 'LD', 'VAT']:
            if not products.get(product_name, False):
                self.logger.info(f"Produit {product_name} désactivé, ignoré")
                continue
                
            source_file_cropped = source_files_cropped[product_name]
            source_file_uncropped = source_files_uncropped[product_name]
            
            # Fichiers sources : rogné depuis Temp, non rogné depuis Temp
            input_path_cropped = self.temp_dir / source_file_cropped
            input_path_uncropped = self.temp_dir / source_file_uncropped
            
            if not input_path_cropped.exists():
                self.logger.warning(f"Produit source rogné manquant: {source_file_cropped}")
                continue
            
            if not input_path_uncropped.exists():
                self.logger.warning(f"Produit source non rogné manquant: {source_file_uncropped}")
                continue
            
            # Déterminer le dossier de destination selon le type de produit
            if product_name in ['MNT', 'DENSITE']:
                # Produits MNT et DENSITE : dossiers directs dans results
                base_dir = self.results_dir / output_structure.get(product_name, product_name)
            else:
                # Produits RVT : sous-dossiers dans RVT dans results
                rvt_config = output_structure.get('RVT', {})
                rvt_base = rvt_config.get('base_dir', 'RVT')
                rvt_subdir = rvt_config.get(product_name, product_name)
                base_dir = self.results_dir / rvt_base / rvt_subdir
            
            # Nom de fichier de sortie
            if product_name == 'MNT':
                output_name = f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69"
            elif product_name == 'DENSITE':
                output_name = f"LHD_FXX_{x}_{y}_densite_A_LAMB93"
            else:
                # Adapter les noms affichés pour M_HS -> M-HS
                display_name = 'M-HS' if product_name == 'M_HS' else product_name
                output_name = f"LHD_FXX_{x}_{y}_{display_name}_A_LAMB93"
            
            # Copier en TIF si activé (utiliser la version ROGNÉE)
            if output_formats.get('tif', True):
                tif_dir = base_dir / 'tif'
                tif_dir.mkdir(parents=True, exist_ok=True)
                tif_path = tif_dir / f"{output_name}.tif"
                if not tif_path.exists():
                    try:
                        shutil.copy2(str(input_path_cropped), str(tif_path))
                        self.logger.info(f"TIF rogné copié: {tif_path.relative_to(self.results_dir)}")
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la copie TIF rogné de {product_name}: {e}")
                        success = False
                        continue
                else:
                    self.logger.info(f"TIF déjà existant: {tif_path.relative_to(self.results_dir)}")

                # Générer les pyramides raster pour le TIF final
                self.build_raster_pyramids(str(tif_path))
            
            # Convertir en JPG selon la configuration RVT OU si Computer Vision est activé
            jpg_config = output_formats.get('jpg', {})
            cv_config = self.get_config_value('computer_vision')
            cv_enabled = cv_config.get('enabled', False)
            cv_target_rvt = cv_config.get('target_rvt')
            
            should_convert_jpg = False
            
            if product_name in ['MNT', 'DENSITE']:
                # Pour MNT et DENSITE: contrôle uniquement par la config JPG
                should_convert_jpg = jpg_config.get(product_name, False)
            else:  # Produits RVT
                # Pour RVT: selon config par produit
                should_convert_jpg = jpg_config.get(product_name, False)
                # Si la CV est activée et cible explicitement ce RVT, forcer le JPG
                if cv_enabled and cv_target_rvt == product_name:
                    should_convert_jpg = True
            
            if should_convert_jpg:
                jpg_dir = base_dir / 'jpg'
                jpg_dir.mkdir(parents=True, exist_ok=True)
                jpg_path = jpg_dir / f"{output_name}.jpg"
                if not jpg_path.exists():
                    # Utiliser le fichier étendu depuis Temp pour créer le JPG
                    # IMPORTANT: Utiliser la géoréférence du TIF ÉTENDU (même raster que l'image YOLO)
                    if self.convert_tif_to_jpg(str(input_path_uncropped), str(jpg_path), product_name):
                        self.logger.info(f"JPG créé: {jpg_path.relative_to(self.results_dir)}")
                        
                        # Appliquer la détection CV si activée et sur le bon RVT
                        self.apply_computer_vision_if_enabled(str(jpg_path), product_name)
                    else:
                        self.logger.error(f"Échec de la conversion JPG pour {product_name}")
                        success = False
                else:
                    self.logger.info(f"JPG déjà existant: {jpg_path.relative_to(self.results_dir)}")
            else:
                self.logger.info(f"Conversion JPG désactivée pour {product_name}")
        
        return success
    
    def process_tile(self, nom_fichier: str, url: str) -> bool:
        """Traite une dalle complète"""
        # Numéro de dalle géré dans run(); ne pas ré-incrémenter ici
        # Log UI détaillé déjà émis par run(); éviter les doublons ici
        
        # Stocker le nom de la dalle pour les fichiers temporaires
        self.current_tile_name = nom_fichier.replace('.copc.laz', '').replace('.laz', '')
        
        try:
            x, y = self.extract_coordinates(nom_fichier)
            
            # Téléchargement dalle centrale
            self.log_to_ui(f"📥 Téléchargement de la dalle centrale...")
            if not self.download_file(url, nom_fichier):
                self.log_to_ui(f"❌ Échec du téléchargement", "error")
                return False
            self.log_to_ui(f"✅ Dalle centrale téléchargée")
            
            # Traitement dalles voisines
            self.log_to_ui(f"🔍 Recherche des dalles voisines...")
            neighbors = self.calculate_neighbor_coordinates(x, y)
            # Calculer la marge en mètres à partir de la configuration UI (tile_overlap en % de 1000m)
            tile_overlap_percent = self.get_config_value('processing', 'tile_overlap', 5)
            try:
                tile_overlap_percent = float(tile_overlap_percent)
            except Exception:
                tile_overlap_percent = 5.0
            margin_m = max(0, min(999, int(round(1000.0 * tile_overlap_percent / 100.0))))
            neighbor_files = []
            
            for voisin_x, voisin_y, place_dalle in neighbors:
                neighbor_info = self.find_neighbor_file(voisin_x, voisin_y)
                if not neighbor_info:
                    continue
                
                neighbor_file, neighbor_url = neighbor_info
                
                if not self.download_file(neighbor_url, neighbor_file):
                    continue
                
                bounds = self.calculate_crop_bounds(voisin_x, voisin_y, place_dalle, margin_m)
                output_file = f"{self.current_tile_name}_neighbor_{place_dalle}.laz"
                
                if self.crop_neighbor_tile(neighbor_file, output_file, bounds):
                    neighbor_files.append(output_file)
            
            self.log_to_ui(f"✅ {len(neighbor_files)} dalles voisines traitées")
            
            # Fusion avec dalles voisines
            self.log_to_ui(f"🔗 Fusion des dalles...")
            merged_file = f"{self.current_tile_name}_merged.laz"
            # Construire le chemin complet vers le fichier central
            central_file_path = str(self.input_dir / nom_fichier)
            if not self.merge_tiles(central_file_path, neighbor_files, merged_file):
                self.log_to_ui(f"❌ Échec de la fusion", "error")
                return False
            self.log_to_ui(f"✅ Fusion terminée")
            
            # Récupérer la configuration des produits depuis la configuration EFFECTIVE
            products = self.get_config_value('processing', 'products', {})
            
            # Création MNT étendu (avec marges des dalles voisines) - TOUJOURS généré pour les RVT
            self.log_to_ui(f"🏔️ Création du MNT...")
            if not self._measure_step(self.current_tile_name, "MNT", self.create_terrain_model, merged_file):
                self.log_to_ui(f"❌ Échec création MNT", "error")
                return False
            self.log_to_ui(f"✅ MNT créé")
            
            # Création de la carte de densité si activée
            if products.get('DENSITE', True):  # Par défaut activée
                self.log_to_ui(f"📊 Création de la carte de densité...")
                self._measure_step(self.current_tile_name, "DENSITE", self.create_density_map, merged_file)
                self.log_to_ui(f"✅ Carte de densité créée")
            
            # Créer les produits RVT à partir du MNT étendu
            self.log_to_ui(f"🎨 Création des produits de visualisation...")
            mnt_file = f"{self.current_tile_name}_MNT.tif"
            self._measure_step(self.current_tile_name, "RVT", self.create_visualization_products, input_file=mnt_file)
            self.log_to_ui(f"✅ Produits de visualisation créés")
            
            # Rognage final de tous les produits pour enlever les marges
            self.log_to_ui(f"✂️ Rognage final...")
            self.crop_final_products(x, y, neighbors)
            
            # Copier les produits finaux dans la structure organisée
            self.log_to_ui(f"📁 Organisation des fichiers de sortie...")
            self.copy_final_products(x, y)
            
            self.log_to_ui(f"🎉 Dalle {self.num_dalle} terminée avec succès")
            return True
        except Exception as e:
            self.log_to_ui(f"❌ Erreur lors du traitement: {e}", "error")
            return False
    
    def run(self, ui_config: dict = None) -> bool:
        """Exécute le pipeline complet avec configuration UI"""
        try:
            # Mettre à jour la configuration UI si fournie et figer la configuration effective
            if ui_config:
                self.ui_config = ui_config
                self.effective_config = self._deep_merge_dicts(self.config, self.ui_config)
            else:
                self.effective_config = self.config

            self.logger.info("Début du pipeline LIDAR")
            # S'assurer que les dossiers de travail (input_dir, temp_dir, results_dir) sont prêts
            if not hasattr(self, '_directories_setup'):
                self.setup_directories()
                self._directories_setup = True

            # Initialiser les chemins des outils QGIS/GDAL si nécessaire
            if not hasattr(self, 'qgis_process'):
                osgeo4w_bin, osgeo4w_root = self._get_osgeo_paths()
                self.qgis_process = f"{osgeo4w_bin}/qgis_process-qgis.bat"
                self.gdalwarp = f"{osgeo4w_bin}/gdalwarp.exe"

            if not self._ensure_pdal_available():
                return False

            self.clean_temp_files()

            # Sélection du mode de données
            data_mode = self.get_config_value('processing', 'data_mode', 'ign_laz')
            self.logger.info(f"Mode de données sélectionné: {data_mode}")

            if data_mode == 'existing_mnt':
                return self.run_from_existing_mnt()
            if data_mode == 'existing_rvt':
                return self.run_from_existing_rvt()
            if data_mode in ('local_laz', 'local_ign'):
                return self.run_from_local_laz()

            # Mode par défaut: pipeline complet à partir d'une liste de dalles IGN (fichier texte)
            return self.run_from_ign_laz()

        except Exception as e:
            self.logger.error(f"Erreur fatale dans le pipeline: {e}")
            return False

    def run_from_ign_laz(self) -> bool:
        """Mode historique: exécution complète à partir des dalles IGN (LAZ via URL)."""
        file_list = self.parse_input_file()

        if not file_list:
            self.logger.error("Aucun fichier à traiter")
            return False

        self.total_dalles = len(file_list)
        self.logger.info(f"Traitement de {self.total_dalles} fichiers (mode IGN LAZ)")

        # Boucle principale de traitement des dalles
        for index, (nom_fichier, lien) in enumerate(file_list, start=1):
            if not self._process_single_tile(nom_fichier, lien, index, self.total_dalles):
                return False

        self._deduplicate_cv_shapefiles_final()
        self.logger.info("Pipeline terminé avec succès (mode IGN LAZ)")
        self._print_metrics_summary()
        return True

    def run_from_local_laz(self) -> bool:
        """Mode nuages locaux: lit un dossier de LAZ/LAS et applique le pipeline standard.

        On attend que processing.local_laz_dir pointe vers un dossier contenant des
        fichiers LAS/LAZ nommés selon le schéma habituel: ..._xxxx_yyyy[.copc].laz
        Les coordonnées xxxx, yyyy sont extraites du nom de fichier.
        """
        local_dir_str = self.get_config_value('processing', 'local_laz_dir', "")
        if not local_dir_str:
            self.logger.error("Mode local_laz sélectionné mais aucun dossier nuages locaux n'est configuré")
            return False

        local_dir = Path(local_dir_str)
        if not local_dir.exists() or not local_dir.is_dir():
            self.logger.error(f"Dossier nuages locaux inexistant ou invalide: {local_dir}")
            return False

        # En mode local_laz, on travaille directement dans le dossier source pour les dalles
        # afin d'éviter toute recopie inutile dans output/dalles.
        self.input_dir = local_dir

        # Lister les nuages (LAS/LAZ) dans le dossier
        laz_files = list(local_dir.glob("*.laz")) + list(local_dir.glob("*.las"))
        if not laz_files:
            self.logger.error(f"Aucun fichier LAZ/LAS trouvé dans {local_dir}")
            return False

        # Construire une liste triée par coordonnées, comme parse_input_file
        temp_data = []  # (x_coord, y_coord, nom_fichier_sortie, chemin_local)
        for path in laz_files:
            name = path.name
            # Extraire xxxx, yyyy depuis le nom sans extensions multiples
            p = Path(name)
            nom_sans_ext = Path(p.stem).stem
            parts = nom_sans_ext.split('_')
            if len(parts) >= 4:
                try:
                    x_coord = int(parts[2])
                    y_coord = int(parts[3])
                    temp_data.append((x_coord, y_coord, name, str(path)))
                except Exception:
                    self.logger.warning(f"Impossible d'extraire les coordonnées de: {name}")
            else:
                self.logger.warning(f"Nom de fichier inattendu (coords manquantes): {name}")

        if not temp_data:
            self.logger.error("Aucun nuage local avec coordonnées valides trouvé")
            return False

        # Tri par coordonnées
        temp_data.sort(key=lambda x: (x[0], x[1]))

        # En mode local, on génère aussi fichier_tri.txt pour permettre la recherche des dalles voisines.
        # Format identique au mode ign_laz: "nom_fichier,lien" mais ici lien=chemin local.
        try:
            with open(self.sorted_file, 'w', encoding='utf-8') as f:
                for _, _, nom_fichier, chemin_local in temp_data:
                    f.write(f"{nom_fichier},{chemin_local}\n")
            self.logger.info(f"Fichier trié généré (mode local): {self.sorted_file}")
        except Exception as e:
            self.logger.error(f"Impossible de générer le fichier trié {self.sorted_file}: {e}")

        # Construire la liste des dalles (nom logique, chemin local)
        file_list = [(nom_fichier, chemin_local) for _, _, nom_fichier, chemin_local in temp_data]

        self.total_dalles = len(file_list)
        self.logger.info(f"Mode local_laz: {self.total_dalles} nuages locaux trouvés dans {local_dir}")

        for index, (nom_fichier, chemin_local) in enumerate(file_list, start=1):
            if self.stop_requested:
                self.logger.info("Arrêt du pipeline demandé par l'utilisateur (mode local_laz)")
                self.cleanup_on_stop()
                return False

            # _process_single_tile appellera process_tile, qui utilisera download_file;
            # download_file a déjà été adapté pour copier un fichier local si le "lien"
            # pointe vers un chemin existant.
            if not self._process_single_tile(nom_fichier, chemin_local, index, self.total_dalles):
                return False

        self._deduplicate_cv_shapefiles_final()
        self.logger.info("Mode local_laz terminé avec succès")
        self._print_metrics_summary()
        return True

    def run_from_existing_mnt(self) -> bool:
        """Mode MNT existants: utilise un dossier de MNT (TIF/ASC) déjà géoréférencés.

        Pour chaque MNT, on déduit les coordonnées de dalle (xxxx, yyyy) à partir du
        nom de fichier ou, à défaut, du géoréférencement, puis on génère les produits
        RVT et on copie les résultats comme dans le pipeline standard.
        """
        # Récupérer le dossier de MNT existants depuis la configuration effective
        mnt_dir_str = self.get_config_value('processing', 'existing_mnt_dir', "")
        if not mnt_dir_str:
            self.logger.error("Mode existing_mnt sélectionné mais aucun dossier MNT n'est configuré")
            return False

        mnt_dir = Path(mnt_dir_str)
        if not mnt_dir.exists() or not mnt_dir.is_dir():
            self.logger.error(f"Dossier MNT inexistant ou invalide: {mnt_dir}")
            return False

        # Lister les fichiers MNT (GeoTIFF et ASC)
        mnt_files: List[Path] = list(mnt_dir.glob("*.tif")) + list(mnt_dir.glob("*.tiff")) + list(mnt_dir.glob("*.asc"))
        if not mnt_files:
            self.logger.error(f"Aucun fichier MNT (*.tif, *.tiff, *.asc) trouvé dans {mnt_dir}")
            return False

        self.total_dalles = len(mnt_files)
        self.logger.info(f"Mode existing_mnt: {self.total_dalles} MNT trouvés dans {mnt_dir}")

        for index, mnt_path in enumerate(mnt_files, start=1):
            if self.stop_requested:
                self.logger.info("Arrêt du pipeline demandé par l'utilisateur (mode existing_mnt)")
                self.cleanup_on_stop()
                return False

            # Mettre à jour l'index courant de dalle
            self.num_dalle = index

            # Déterminer x, y et current_tile_name à partir du nom de fichier ou du géoréférencement
            x_str, y_str, tile_name = self._infer_tile_coords_from_mnt(mnt_path)
            if x_str is None or y_str is None or tile_name is None:
                self.logger.error(f"Impossible de déduire les coordonnées pour le MNT: {mnt_path.name}")
                return False

            self.current_tile_name = tile_name

            progress_percent = (index / self.total_dalles) * 100
            if self.ui_callback:
                self.ui_callback(f"MNT {index}/{self.total_dalles}: {mnt_path.name}", progress_percent)

            self.logger.info(f"=== TRAITEMENT MNT EXISTANT {index}/{self.total_dalles}: {mnt_path.name} ===")
            self.logger.info(f"Dalle déduite: {self.current_tile_name} (x={x_str}, y={y_str})")

            # Préparer le MNT dans le dossier Temp sous le nom attendu
            mnt_temp_path = self.temp_dir / f"{self.current_tile_name}_MNT.tif"

            try:
                if not mnt_temp_path.exists():
                    if mnt_path.suffix.lower() in [".tif", ".tiff"]:
                        shutil.copy2(str(mnt_path), str(mnt_temp_path))
                        self.logger.info(f"MNT copié dans Temp: {mnt_temp_path}")
                    elif mnt_path.suffix.lower() == ".asc":
                        # Conversion ASC -> TIF avec gdal_translate
                        osgeo4w_bin, _ = self._get_osgeo_paths()
                        gdal_translate_path = f"{osgeo4w_bin}/gdal_translate.exe"
                        cmd = [gdal_translate_path, str(mnt_path), str(mnt_temp_path)]
                        self.logger.info(f"Conversion ASC->TIF: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        if result.stderr:
                            self.logger.warning(f"gdal_translate stderr: {result.stderr}")
                        self.logger.info(f"MNT converti en TIF dans Temp: {mnt_temp_path}")
                    else:
                        self.logger.error(f"Format MNT non supporté: {mnt_path.suffix}")
                        return False
                else:
                    self.logger.info(f"MNT déjà présent dans Temp: {mnt_temp_path}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Erreur lors de la conversion ASC->TIF pour {mnt_path.name}: {e}")
                if e.stdout:
                    self.logger.error(f"STDOUT: {e.stdout}")
                if e.stderr:
                    self.logger.error(f"STDERR: {e.stderr}")
                return False
            except Exception as e:
                self.logger.error(f"Erreur lors de la préparation du MNT {mnt_path.name}: {e}")
                return False

            # Générer les produits de visualisation à partir du MNT existant
            self.log_to_ui("🎨 Création des produits de visualisation à partir du MNT existant...")
            if not self.create_visualization_products(input_file=mnt_temp_path.name):
                self.log_to_ui("❌ Échec de la création des produits de visualisation", "error")
                return False
            self.log_to_ui("✅ Produits de visualisation créés à partir du MNT existant")

            # Rognage final des produits pour obtenir une dalle 1000x1000m
            self.log_to_ui("✂️ Rognage final des produits (MNT existant)...")
            if not self.crop_final_products(x_str, y_str, []):
                self.log_to_ui("❌ Échec du rognage final des produits", "error")
                return False

            # Copier les produits finaux dans la structure organisée
            self.log_to_ui("📁 Organisation des fichiers de sortie (MNT existant)...")
            self.copy_final_products(x_str, y_str)

            self.log_to_ui(f"🎉 MNT existant {index}/{self.total_dalles} traité avec succès")

        self._deduplicate_cv_shapefiles_final()
        self.logger.info("Mode existing_mnt terminé avec succès")
        return True

    def run_from_existing_rvt(self) -> bool:
        """Mode indices RVT existants: applique la détection CV à partir de TIF RVT (ex: *_LD.tif).

        Le dossier existing_rvt_dir doit contenir des rasters RVT en TIF/TIFF, par exemple
        LHD_FXX_0948_6753_PTS_C_LAMB93_IGN69_LD.tif. Le pipeline convertit ces TIF en JPG
        puis lance la détection CV en mode scan_all sur ces JPG.
        """
        rvt_dir_str = self.get_config_value('processing', 'existing_rvt_dir', "")
        if not rvt_dir_str:
            self.logger.error("Mode existing_rvt sélectionné mais aucun dossier RVT n'est configuré")
            return False

        rvt_dir = Path(rvt_dir_str)
        if not rvt_dir.exists() or not rvt_dir.is_dir():
            self.logger.error(f"Dossier RVT inexistant ou invalide: {rvt_dir}")
            return False

        # Rechercher les TIF RVT (LD, SVF, etc.)
        tif_files = sorted(list(rvt_dir.glob("*.tif")) + list(rvt_dir.glob("*.tiff")))
        if not tif_files:
            self.logger.error(f"Aucun fichier TIF/TIFF trouvé dans {rvt_dir} pour le mode existing_rvt")
            return False

        cv_config = self.get_config_value('computer_vision')
        if not cv_config.get('enabled', False):
            self.logger.error("Mode existing_rvt sélectionné mais la computer vision est désactivée")
            return False

        target_rvt = cv_config.get('target_rvt', 'LD')

        # Déterminer le répertoire de base dans results pour ce type RVT
        output_structure = self.get_config_value('processing', 'output_structure', {})
        rvt_output_dir = None
        try:
            if 'RVT' in output_structure:
                rvt_cfg = output_structure['RVT']
                base_dir_name = rvt_cfg.get('base_dir', 'RVT')
                type_dir_name = rvt_cfg.get(target_rvt, target_rvt)
                rvt_output_dir = self.results_dir / base_dir_name / type_dir_name
                rvt_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Impossible de déterminer le dossier RVT de sortie, utilisation du dossier d'entrée RVT: {e}")

        # Répertoire pour les JPG dérivés des TIF
        if rvt_output_dir is not None:
            jpg_output_dir = rvt_output_dir / 'jpg'
            jpg_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Fallback: utiliser le dossier d'entrée RVT
            jpg_output_dir = rvt_dir

        # Convertir les TIF en JPG dans jpg_output_dir (structure standard results/RVT/<TYPE>/jpg)
        jpg_files: List[Path] = []
        for tif_path in tif_files:
            # On convertit tous les TIF présents; la sélection par type se fait via target_rvt
            jpg_path = jpg_output_dir / (tif_path.stem + '.jpg')
            if not jpg_path.exists():
                self.logger.info(f"Conversion TIF->JPG (existing_rvt): {tif_path.name} -> {jpg_path.name}")
                if not self.convert_tif_to_jpg(str(tif_path), str(jpg_path), product_type=target_rvt, reference_tif_path=str(tif_path)):
                    self.logger.error(f"Échec de la conversion TIF->JPG pour {tif_path.name}")
                    return False
            else:
                self.logger.info(f"JPG déjà présent pour {tif_path.name}: {jpg_path.name}")
            jpg_files.append(jpg_path)

        if not jpg_files:
            self.logger.error(f"Aucun JPG disponible après conversion dans {jpg_output_dir} pour le mode existing_rvt")
            return False

        # Forcer scan_all pour analyser toutes les images du dossier
        try:
            if 'computer_vision' not in self.effective_config:
                self.effective_config['computer_vision'] = {}
            self.effective_config['computer_vision']['scan_all'] = True
        except Exception:
            pass

        self.logger.info(f"Mode existing_rvt: {len(jpg_files)} images (issues de TIF) dans {jpg_output_dir}")

        # Surcharger temporairement le répertoire de base pour annotated_images/shapefiles
        if rvt_output_dir is not None:
            self.override_rvt_base_dir = rvt_output_dir

        first_jpg = jpg_files[0]
        self.log_to_ui(f"🔎 Détection CV sur les indices RVT existants (TIF) dans {rvt_dir} (type {target_rvt})...")

        try:
            success = self.apply_computer_vision_if_enabled(str(first_jpg), target_rvt)
        finally:
            # Nettoyer l'override pour ne pas impacter les autres modes/appels
            if hasattr(self, 'override_rvt_base_dir'):
                delattr(self, 'override_rvt_base_dir')

        if success:
            self._deduplicate_cv_shapefiles_final()
            self.logger.info("Mode existing_rvt terminé avec succès")
            return True
        else:
            self.logger.error("Mode existing_rvt: échec de la détection CV")
            return False

    def _infer_tile_coords_from_mnt(self, mnt_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Tente de déduire (x, y, current_tile_name) pour un MNT donné.

        Priorité:
        1) Extraire xxxx, yyyy depuis le nom de fichier (blocs numériques de 4 chiffres)
        2) À défaut, utiliser le géoréférencement du raster (bounds) pour calculer les kilomètres
        """
        name = mnt_path.stem
        parts = name.split('_')

        numeric_parts: List[str] = []
        for p in parts:
            try:
                if len(p) == 4:
                    int(p)
                    numeric_parts.append(p)
            except ValueError:
                continue

        if len(numeric_parts) >= 2:
            x_str = numeric_parts[-2]
            y_str = numeric_parts[-1]
            tile_name = name  # On conserve le nom de base du fichier comme tile_name
            self.logger.info(f"Coordonnées MNT déduites du nom de fichier: x={x_str}, y={y_str}")
            return x_str, y_str, tile_name

        # Fallback: utiliser le géoréférencement du raster
        self.logger.info(f"Impossible d'extraire xxxx_yyyy du nom {name}, utilisation du géoréférencement")
        bounds = self.get_raster_bounds(str(mnt_path))
        if not bounds:
            self.logger.error(f"Impossible d'extraire les bounds du MNT: {mnt_path}")
            return None, None, None

        xmin, ymin, xmax, ymax = bounds
        try:
            x_km = int(xmin // 1000)
            y_km = int(ymax // 1000)
            x_str = f"{x_km:04d}"
            y_str = f"{y_km:04d}"
            tile_name = f"MNT_EXT_{x_str}_{y_str}"
            self.logger.info(f"Coordonnées MNT déduites des bounds: x={x_str}, y={y_str} (tile={tile_name})")
            return x_str, y_str, tile_name
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des coordonnées MNT depuis les bounds: {e}")
            return None, None, None

    def _process_single_tile(self, nom_fichier: str, lien: str, index: int, total: int) -> bool:
        """Traite une seule dalle avec logs et progression."""
        # Vérifier si l'arrêt a été demandé
        if self.stop_requested:
            self.logger.info("Arrêt du pipeline demandé par l'utilisateur")
            self.cleanup_on_stop()
            return False

        # Mettre à jour l'index courant de dalle
        self.num_dalle = index

        # Calcul et affichage de la progression
        progress_percent = (index / total) * 100
        self.logger.info(f"=== TRAITEMENT DALLE {index}/{total}: {nom_fichier} ===")
        if self.ui_callback:
            self.ui_callback(f"Dalle {index}/{total}: {nom_fichier}", progress_percent)

        # Traitement effectif de la dalle
        if not self.process_tile(nom_fichier, lien):
            self.logger.error(f"Échec du traitement de {nom_fichier}")
            return False

        return True


if __name__ == "__main__":
    pipeline = LidarPipeline()
    success = pipeline.run()
    sys.exit(0 if success else 1)
