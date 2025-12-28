#!/usr/bin/env python3
"""
Application de lancement du pipeline LIDAR avec interface graphique Tkinter
Permet de configurer et lancer le pipeline LIDAR avec vérifications préalables
"""
import sys
import os
from pathlib import Path
import logging
import json

# IMPORTANT: Initialiser l'environnement runtime AVANT tous les autres imports
# Ceci résout les conflits DLL GDAL dans l'exécutable PyInstaller
try:
    from src.build.runtime_config import initialize_exe_environment
    runtime_env = initialize_exe_environment()
    print(f"Environnement runtime initialise: GDAL fix = {runtime_env.get('gdal_fixed', False)}")
except ImportError:
    # En mode développement, la configuration runtime peut ne pas être nécessaire
    print("WARNING: Configuration runtime non disponible (mode developpement)")
    runtime_env = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import re
# Import du pipeline
from src.pipeline_lidar import LidarPipeline

def load_config():
    """Charge la configuration depuis le fichier JSON"""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    """Sauvegarde la configuration dans le fichier JSON"""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def setup_logging(log_dir: Path | None = None):
    """Configure le système de logging
    Args:
        log_dir (Path|None): Dossier où écrire pipeline_logs.txt. Par défaut: cwd/output
    """
    # Choisir le dossier de logs
    if log_dir is None:
        log_dir = Path.cwd() / "output"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reconfigurer les handlers de la racine proprement
    root_logger = logging.getLogger()
    # Nettoyer les anciens handlers pour éviter doublons
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / 'pipeline_logs.txt', encoding='utf-8')
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

class PipelineApp:
    """Application Tkinter pour lancer le pipeline LIDAR"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline LIDAR - Traitement MNT")
        
        # Chargement de la configuration
        self.config = load_config()
        
        # Configuration de la fenêtre
        window_size = self.config['app']['ui']['window_size']
        self.center_window(window_size['width'], window_size['height'])
        
        # Permettre le redimensionnement
        self.root.resizable(True, True)
        
        # Variables de configuration - Chargement depuis JSON
        files_config = self.config['app']['files']
        self.input_file = tk.StringVar(value=files_config.get('input_file', ""))  # Fichier dalles d'entrée
        self.output_dir = tk.StringVar(value=files_config.get('output_dir', str(Path.cwd() / "Output")))  # Dossier de sortie
        
        # Variables des outils - Chargement depuis JSON
        tools_config = self.config['tools']
        # Nouveau: on stocke directement le dossier bin OSGeo4W (osgeo4w_bin).
        # Fallback: si seule la racine osgeo4w_root existe encore, on ajoute "bin".
        default_root = tools_config.get('osgeo4w_root', "C:/OSGeo4W")
        default_bin = tools_config.get('osgeo4w_bin', f"{default_root}/bin")
        self.osgeo4w_bin = tk.StringVar(value=default_bin)  # Dossier bin OSGeo4W
        
        # Variables de configuration générale - Initialisation depuis JSON
        processing_config = self.config['processing']
        
        self.mnt_resolution = tk.DoubleVar(value=processing_config.get('mnt_resolution', 0.5))
        self.density_resolution = tk.DoubleVar(value=processing_config.get('density_resolution', 1.0))
        self.tile_overlap = tk.IntVar(value=processing_config.get('tile_overlap', 20))
        self.filter_expression = tk.StringVar(value=processing_config.get('filter_expression', "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9"))
        self.data_mode = tk.StringVar(value=processing_config.get('data_mode', 'ign_laz'))
        self.local_laz_dir = tk.StringVar(value=processing_config.get('local_laz_dir', ""))
        self.existing_mnt_dir = tk.StringVar(value=processing_config.get('existing_mnt_dir', ""))
        self.existing_rvt_dir = tk.StringVar(value=processing_config.get('existing_rvt_dir', ""))
        
        # Variables des produits
        products = processing_config.get('products', {})
        self.product_mnt = tk.BooleanVar(value=products.get('MNT', True))
        self.product_densite = tk.BooleanVar(value=products.get('DENSITE', True))
        self.product_m_hs = tk.BooleanVar(value=products.get('M_HS', True))
        self.product_svf = tk.BooleanVar(value=products.get('SVF', True))
        self.product_slo = tk.BooleanVar(value=products.get('SLO', True))
        self.product_ld = tk.BooleanVar(value=products.get('LD', True))
        self.product_vat = tk.BooleanVar(value=products.get('VAT', False))
        
        # Variables Computer Vision - Initialisation depuis JSON
        cv_config = self.config.get('computer_vision', {})
        self.cv_enabled = tk.BooleanVar(value=cv_config.get('enabled', False))
        self.cv_selected_model = tk.StringVar(value=cv_config.get('selected_model', ''))
        self.cv_target_rvt = tk.StringVar(value=cv_config.get('target_rvt', 'LD'))
        self.cv_confidence_threshold = tk.DoubleVar(value=cv_config.get('confidence_threshold', 0.3))
        self.cv_iou_threshold = tk.DoubleVar(value=cv_config.get('iou_threshold', 0.5))
        self.cv_generate_annotated_images = tk.BooleanVar(value=cv_config.get('generate_annotated_images', False))
        self.cv_generate_shapefiles = tk.BooleanVar(value=cv_config.get('generate_shapefiles', False))
        self.cv_models_dir = tk.StringVar(value=cv_config.get('models_dir', 'models'))
        
        # Variables SAHI - Initialisation depuis JSON
        sahi_config = cv_config.get('sahi', {})
        self.cv_slice_height = tk.IntVar(value=sahi_config.get('slice_height', 750))
        self.cv_slice_width = tk.IntVar(value=sahi_config.get('slice_width', 750))
        self.cv_overlap_ratio = tk.DoubleVar(value=sahi_config.get('overlap_ratio', 0.2))
        
        # Variables RVT - Initialisation depuis JSON
        rvt_params = self.config['rvt_params']
        jpg_config = self.config['processing'].get('output_formats', {}).get('jpg', {})
        # MDH
        mdh_params = rvt_params.get('mdh', {})
        self.mdh_num_directions = tk.IntVar(value=mdh_params.get('num_directions', 16))
        self.mdh_sun_elevation = tk.IntVar(value=mdh_params.get('sun_elevation', 35))
        self.mdh_ve_factor = tk.IntVar(value=mdh_params.get('ve_factor', 1))
        self.mdh_save_as_8bit = tk.BooleanVar(value=mdh_params.get('save_as_8bit', True))
        self.jpg_mhs = tk.BooleanVar(value=jpg_config.get('M_HS', True))
        # SVF
        svf_params = rvt_params.get('svf', {})
        self.svf_noise_remove = tk.IntVar(value=svf_params.get('noise_remove', 0))
        self.svf_num_directions = tk.IntVar(value=svf_params.get('num_directions', 16))
        self.svf_radius = tk.IntVar(value=svf_params.get('radius', 10))
        self.svf_ve_factor = tk.IntVar(value=svf_params.get('ve_factor', 1))
        self.svf_save_as_8bit = tk.BooleanVar(value=svf_params.get('save_as_8bit', True))
        self.jpg_svf = tk.BooleanVar(value=jpg_config.get('SVF', True))
        # Slope
        slope_params = rvt_params.get('slope', {})
        self.slope_unit = tk.IntVar(value=slope_params.get('unit', 0))
        self.slope_ve_factor = tk.IntVar(value=slope_params.get('ve_factor', 1))
        self.slope_save_as_8bit = tk.BooleanVar(value=slope_params.get('save_as_8bit', True))
        self.jpg_slo = tk.BooleanVar(value=jpg_config.get('SLO', True))
        # LDO
        ldo_params = rvt_params.get('ldo', {})
        self.ldo_angular_res = tk.IntVar(value=ldo_params.get('angular_res', 15))
        self.ldo_min_radius = tk.IntVar(value=ldo_params.get('min_radius', 10))
        self.ldo_max_radius = tk.IntVar(value=ldo_params.get('max_radius', 20))
        self.ldo_observer_h = tk.DoubleVar(value=ldo_params.get('observer_h', 1.7))
        self.ldo_ve_factor = tk.IntVar(value=ldo_params.get('ve_factor', 1))
        self.ldo_save_as_8bit = tk.BooleanVar(value=ldo_params.get('save_as_8bit', True))
        self.jpg_ld = tk.BooleanVar(value=jpg_config.get('LD', True))

        # VAT
        vat_params = rvt_params.get('vat', {})
        # Type de terrain: 0=general, 1=flat, 2=steep
        self.vat_terrain_type = tk.IntVar(value=vat_params.get('terrain_type', 0))
        # Sauvegarde en 8bit
        self.vat_save_as_8bit = tk.BooleanVar(value=vat_params.get('save_as_8bit', True))
        # Export JPG (+JGW)
        self.jpg_vat = tk.BooleanVar(value=jpg_config.get('VAT', True))
        
        # État de traitement
        self.processing = False
        self.pipeline = None
        
        # Créer l'interface
        self.create_widgets()
        
    def center_window(self, width, height):
        """Centre la fenêtre sur l'écran"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Crée les widgets de l'interface"""
        # Créer un canvas avec scrollbar pour permettre le scroll avec la molette
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        self.canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Lier les événements de scroll de la molette
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Forcer le canvas à utiliser toute la largeur disponible
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        
        # Pack scrollbar et canvas dans le bon ordre
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Frame principal pour le contenu avec gestion responsive
        main_frame = ttk.Frame(self.scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configuration responsive pour le contenu principal
        main_frame.grid_columnconfigure(0, weight=1)
        
        # === Configuration des chemins et des sources de données ===
        paths_frame = ttk.LabelFrame(main_frame, text="Sources de données et chemins", padding=10)
        paths_frame.pack(fill=tk.X, pady=(0, 10))
        paths_frame.grid_columnconfigure(1, weight=1)
        
        # Dossier de sortie
        ttk.Label(paths_frame, text="Dossier de sortie:").grid(row=0, column=0, sticky=tk.W, pady=2)
        output_dir_frame = ttk.Frame(paths_frame)
        output_dir_frame.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        output_dir_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Entry(output_dir_frame, textvariable=self.output_dir).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(output_dir_frame, text="Parcourir", command=self.browse_output_dir).grid(row=0, column=1, sticky=tk.E)
        
        # Mode de données
        ttk.Label(paths_frame, text="Source des données (mode):").grid(row=1, column=0, sticky=tk.W, pady=2)
        data_mode_frame = ttk.Frame(paths_frame)
        data_mode_frame.grid(row=1, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        data_mode_frame.grid_columnconfigure(0, weight=1)

        self.data_mode_combo = ttk.Combobox(data_mode_frame, textvariable=self.data_mode, state="readonly")
        self.data_mode_combo['values'] = (
            'ign_laz',      # Données IGN (LAZ via URL)
            'local_laz',    # Nuages locaux (LAZ/LAS)
            'existing_mnt', # MNT existants (ASC/TIF)
            'existing_rvt'  # Indices RVT existants (TIF RVT)
        )
        self.data_mode_combo.grid(row=0, column=0, sticky=tk.EW)
        self.data_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_data_mode_changed())
        
        # Sous-titre pour les sources spécifiques au mode
        ttk.Label(paths_frame, text="Sources spécifiques au mode:", font=('TkDefaultFont', 9, 'bold')).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 2)
        )

        # Fichier dalles d'entrée (mode ign_laz)
        ttk.Label(paths_frame, text="Fichier dalles IGN (liste URLs):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.input_file_frame = ttk.Frame(paths_frame)
        self.input_file_frame.grid(row=3, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        self.input_file_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(self.input_file_frame, textvariable=self.input_file).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(self.input_file_frame, text="Parcourir", command=self.browse_input_file).grid(row=0, column=1, sticky=tk.E)

        # Dossier nuages locaux (mode local_laz)
        ttk.Label(paths_frame, text="Dossier nuages locaux (LAZ/LAS):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.local_laz_frame = ttk.Frame(paths_frame)
        self.local_laz_frame.grid(row=4, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        self.local_laz_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(self.local_laz_frame, textvariable=self.local_laz_dir).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(self.local_laz_frame, text="Parcourir", command=self.browse_local_laz_dir).grid(row=0, column=1, sticky=tk.E)

        # Dossier MNT existants (mode existing_mnt)
        ttk.Label(paths_frame, text="Dossier MNT existants (TIF/ASC):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.existing_mnt_frame = ttk.Frame(paths_frame)
        self.existing_mnt_frame.grid(row=5, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        self.existing_mnt_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(self.existing_mnt_frame, textvariable=self.existing_mnt_dir).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(self.existing_mnt_frame, text="Parcourir", command=self.browse_existing_mnt_dir).grid(row=0, column=1, sticky=tk.E)

        # Dossier RVT existants (mode existing_rvt)
        ttk.Label(paths_frame, text="Dossier indices RVT existants (TIF RVT):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.existing_rvt_frame = ttk.Frame(paths_frame)
        self.existing_rvt_frame.grid(row=6, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        self.existing_rvt_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(self.existing_rvt_frame, textvariable=self.existing_rvt_dir).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(self.existing_rvt_frame, text="Parcourir", command=self.browse_existing_rvt_dir).grid(row=0, column=1, sticky=tk.E)

        # Texte d'aide sous le mode de données
        self.data_mode_help = tk.StringVar()
        ttk.Label(paths_frame, textvariable=self.data_mode_help, foreground="gray", font=('TkDefaultFont', 8)).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=(2, 0)
        )

        # === Chemins des outils ===
        tools_paths_frame = ttk.LabelFrame(main_frame, text="Chemins des outils", padding=10)
        tools_paths_frame.pack(fill=tk.X, pady=(0, 10))
        tools_paths_frame.grid_columnconfigure(1, weight=1)

        # Chemin OSGeo4W bin
        ttk.Label(tools_paths_frame, text="Dossier OSGeo4W bin:").grid(row=0, column=0, sticky=tk.W, pady=2)
        osgeo4w_frame = ttk.Frame(tools_paths_frame)
        osgeo4w_frame.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        osgeo4w_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(osgeo4w_frame, textvariable=self.osgeo4w_bin).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(osgeo4w_frame, text="Parcourir", command=self.browse_osgeo4w_bin).grid(row=0, column=1, sticky=tk.E)

        # Dossier des modèles CV
        ttk.Label(tools_paths_frame, text="Dossier modèles CV:").grid(row=1, column=0, sticky=tk.W, pady=2)
        models_frame = ttk.Frame(tools_paths_frame)
        models_frame.grid(row=1, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        models_frame.grid_columnconfigure(0, weight=1)

        ttk.Entry(models_frame, textvariable=self.cv_models_dir).grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(models_frame, text="Parcourir", command=self.browse_models_dir).grid(row=0, column=1, sticky=tk.E)
        
        # === Configuration générale ===
        config_frame = ttk.LabelFrame(main_frame, text="Configuration générale", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame principal pour l'organisation horizontale
        config_main_frame = ttk.Frame(config_frame)
        config_main_frame.pack(fill=tk.X, pady=5)
        
        # Première ligne : Résolutions (affichage horizontal compact)
        resolutions_frame = ttk.Frame(config_main_frame)
        resolutions_frame.pack(fill=tk.X, pady=2)
        
        # MNT Resolution
        ttk.Label(resolutions_frame, text="MNT:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(resolutions_frame, textvariable=self.mnt_resolution, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(resolutions_frame, text="m").pack(side=tk.LEFT, padx=(0, 15))
        
        # Density Resolution
        ttk.Label(resolutions_frame, text="Densité:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(resolutions_frame, textvariable=self.density_resolution, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(resolutions_frame, text="m").pack(side=tk.LEFT, padx=(0, 15))
        
        # Tile Overlap (Marge)
        ttk.Label(resolutions_frame, text="Marge:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(resolutions_frame, textvariable=self.tile_overlap, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(resolutions_frame, text="%").pack(side=tk.LEFT, padx=(0, 0))
        
        # Deuxième ligne : Expression filtre
        filter_frame = ttk.Frame(config_main_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(filter_frame, text="Expression filtre:").pack(side=tk.LEFT, padx=(0, 5))
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_expression)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))
        
        # Troisième ligne : Produits à générer (affichage horizontal compact)
        products_frame = ttk.Frame(config_main_frame)
        products_frame.pack(fill=tk.X, pady=5)
        ttk.Label(products_frame, text="Produits:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Checkbutton(products_frame, text="MNT", variable=self.product_mnt).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="DENSITE", variable=self.product_densite).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="M-HS", variable=self.product_m_hs, command=self.on_rvt_products_changed).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="SVF", variable=self.product_svf, command=self.on_rvt_products_changed).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="SLO", variable=self.product_slo, command=self.on_rvt_products_changed).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="LD", variable=self.product_ld, command=self.on_rvt_products_changed).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Checkbutton(products_frame, text="VAT", variable=self.product_vat, command=self.on_rvt_products_changed).pack(side=tk.LEFT, padx=(0, 0))
        
        # Bouton reset pour configuration générale
        reset_config_btn = ttk.Button(config_frame, text="Remettre par défaut", command=self.reset_general_config)
        reset_config_btn.pack(pady=5)
        
        # === Paramètres RVT ===
        rvt_frame = ttk.LabelFrame(main_frame, text="Paramètres RVT", padding=10)
        rvt_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Créer un notebook pour organiser les paramètres RVT
        rvt_notebook = ttk.Notebook(rvt_frame)
        rvt_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet M-HS
        mdh_frame = ttk.Frame(rvt_notebook)
        rvt_notebook.add(mdh_frame, text="M-HS")
        
        ttk.Label(mdh_frame, text="Nombre directions:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(mdh_frame, textvariable=self.mdh_num_directions, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mdh_frame, text="Élévation solaire:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(mdh_frame, textvariable=self.mdh_sun_elevation, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(mdh_frame, text="Facteur VE:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(mdh_frame, textvariable=self.mdh_ve_factor, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Checkbutton(mdh_frame, text="Sauver en 8bit", variable=self.mdh_save_as_8bit).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(mdh_frame, text="Exporter en JPG (+JGW)", variable=self.jpg_mhs).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Onglet SVF
        svf_frame = ttk.Frame(rvt_notebook)
        rvt_notebook.add(svf_frame, text="SVF")
        
        ttk.Label(svf_frame, text="Suppression bruit:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(svf_frame, textvariable=self.svf_noise_remove, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(svf_frame, text="Nombre directions:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(svf_frame, textvariable=self.svf_num_directions, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(svf_frame, text="Rayon:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(svf_frame, textvariable=self.svf_radius, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(svf_frame, text="Facteur VE:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(svf_frame, textvariable=self.svf_ve_factor, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Checkbutton(svf_frame, text="Sauver en 8bit", variable=self.svf_save_as_8bit).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(svf_frame, text="Exporter en JPG (+JGW)", variable=self.jpg_svf).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Onglet Slope
        slope_frame = ttk.Frame(rvt_notebook)
        rvt_notebook.add(slope_frame, text="Slope")
        
        ttk.Label(slope_frame, text="Unité:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(slope_frame, textvariable=self.slope_unit, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(slope_frame, text="Facteur VE:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(slope_frame, textvariable=self.slope_ve_factor, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Checkbutton(slope_frame, text="Sauver en 8bit", variable=self.slope_save_as_8bit).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(slope_frame, text="Exporter en JPG (+JGW)", variable=self.jpg_slo).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Onglet LD
        ldo_frame = ttk.Frame(rvt_notebook)
        rvt_notebook.add(ldo_frame, text="LD")
        
        ttk.Label(ldo_frame, text="Résolution angulaire:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ldo_frame, textvariable=self.ldo_angular_res, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(ldo_frame, text="Rayon min:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ldo_frame, textvariable=self.ldo_min_radius, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(ldo_frame, text="Rayon max:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ldo_frame, textvariable=self.ldo_max_radius, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(ldo_frame, text="Hauteur observateur:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ldo_frame, textvariable=self.ldo_observer_h, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(ldo_frame, text="Facteur VE:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ldo_frame, textvariable=self.ldo_ve_factor, width=10).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Checkbutton(ldo_frame, text="Sauver en 8bit", variable=self.ldo_save_as_8bit).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(ldo_frame, text="Exporter en JPG (+JGW)", variable=self.jpg_ld).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        # Onglet VAT
        vat_frame = ttk.Frame(rvt_notebook)
        rvt_notebook.add(vat_frame, text="VAT")

        ttk.Label(vat_frame, text="Type de terrain:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        terrain_frame = ttk.Frame(vat_frame)
        terrain_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Radiobutton(terrain_frame, text="Général", variable=self.vat_terrain_type, value=0).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(terrain_frame, text="Plat", variable=self.vat_terrain_type, value=1).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(terrain_frame, text="Pentu", variable=self.vat_terrain_type, value=2).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Checkbutton(vat_frame, text="Sauver en 8bit", variable=self.vat_save_as_8bit).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(vat_frame, text="Exporter en JPG (+JGW)", variable=self.jpg_vat).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Bouton reset pour paramètres RVT
        reset_rvt_btn = ttk.Button(rvt_frame, text="Remettre par défaut", command=self.reset_rvt_params)
        reset_rvt_btn.pack(pady=5)
        
        # === Détection par Computer Vision ===
        cv_frame = ttk.LabelFrame(main_frame, text="Détection par Computer Vision", padding=10)
        cv_frame.pack(fill=tk.X, pady=(0, 10))
        cv_frame.grid_columnconfigure(1, weight=1)
        
        # Checkbox pour activer/désactiver
        ttk.Checkbutton(cv_frame, text="Activer la détection par computer vision", 
                       variable=self.cv_enabled, command=self.on_cv_enabled_changed).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Sélecteur de modèle
        ttk.Label(cv_frame, text="Modèle:").grid(row=1, column=0, sticky=tk.W, pady=2)
        model_frame = ttk.Frame(cv_frame)
        model_frame.grid(row=1, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        model_frame.grid_columnconfigure(0, weight=1)
        
        self.cv_model_combo = ttk.Combobox(model_frame, textvariable=self.cv_selected_model, state="readonly")
        self.cv_model_combo.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        
        ttk.Button(model_frame, text="Actualiser", command=self.refresh_models).grid(row=0, column=1, sticky=tk.E)
        
        # Sélecteur RVT cible
        ttk.Label(cv_frame, text="RVT cible:").grid(row=2, column=0, sticky=tk.W, pady=2)
        rvt_frame_cv = ttk.Frame(cv_frame)
        rvt_frame_cv.grid(row=2, column=1, sticky=tk.EW, padx=(10, 0), pady=2)
        rvt_frame_cv.grid_columnconfigure(0, weight=1)
        
        self.cv_rvt_combo = ttk.Combobox(rvt_frame_cv, textvariable=self.cv_target_rvt, state="readonly")
        self.cv_rvt_combo.grid(row=0, column=0, sticky=tk.EW)
        
        # Seuils de confiance et IoU
        thresholds_frame = ttk.Frame(cv_frame)
        thresholds_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=2)
        thresholds_frame.grid_columnconfigure(1, weight=1)
        thresholds_frame.grid_columnconfigure(3, weight=1)
        
        # Seuil de confiance
        ttk.Label(thresholds_frame, text="Confiance:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(thresholds_frame, from_=0.1, to=1.0, increment=0.1, 
                   textvariable=self.cv_confidence_threshold, width=8, format="%.1f").grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # Seuil IoU
        ttk.Label(thresholds_frame, text="IoU:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        ttk.Spinbox(thresholds_frame, from_=0.1, to=1.0, increment=0.1, 
                   textvariable=self.cv_iou_threshold, width=8, format="%.1f").grid(row=0, column=3, sticky=tk.W)
        
        # Note explicative pour les seuils
        ttk.Label(cv_frame, text="Confiance: 0.1=sensible, 0.5=équilibré, 0.9=strict | IoU: 0.1=permissif, 0.5=équilibré, 0.9=strict", 
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(2, 5))
        
        # Options de sortie
        output_options_frame = ttk.LabelFrame(cv_frame, text="Options de sortie", padding="5")
        output_options_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Checkbutton(output_options_frame, text="Générer des images avec bounding boxes", 
                       variable=self.cv_generate_annotated_images).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(output_options_frame, text="Générer des shapefiles", variable=self.cv_generate_shapefiles).grid(row=1, column=0, sticky=tk.W, pady=2)

        # Appliquer l'état initial des champs selon le mode de données courant
        self.on_data_mode_changed()
        
        # Paramètres SAHI
        sahi_frame = ttk.LabelFrame(cv_frame, text="Paramètres SAHI", padding="5")
        sahi_frame.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=5)
        sahi_frame.grid_columnconfigure(1, weight=1)
        sahi_frame.grid_columnconfigure(3, weight=1)
        
        # Taille des slices
        ttk.Label(sahi_frame, text="Taille des slices:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        slice_size_frame = ttk.Frame(sahi_frame)
        slice_size_frame.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(slice_size_frame, text="H:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(slice_size_frame, textvariable=self.cv_slice_height, width=8).grid(row=0, column=1, sticky=tk.W, padx=(2, 5))
        
        ttk.Label(slice_size_frame, text="L:").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        ttk.Entry(slice_size_frame, textvariable=self.cv_slice_width, width=8).grid(row=0, column=3, sticky=tk.W, padx=(2, 0))
        
        # Overlap ratio
        ttk.Label(sahi_frame, text="Chevauchement:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        ttk.Entry(sahi_frame, textvariable=self.cv_overlap_ratio, width=8).grid(row=0, column=3, sticky=tk.W)
        
        # Note explicative
        note_label = ttk.Label(cv_frame, text="Note: La détection sera appliquée sur le RVT sélectionné. Seuls les RVT activés sont disponibles.", 
                              font=('TkDefaultFont', 8), foreground='gray')
        note_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Initialiser la liste des modèles et RVT
        self.refresh_models()
        self.update_available_rvt()
        self.on_cv_enabled_changed()
        
        # Lier le changement du dossier modèles à l'actualisation de la liste
        self.cv_models_dir.trace('w', lambda *args: self.refresh_models())
        
        # === Contrôles de traitement ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        control_frame.grid_columnconfigure(3, weight=1)
        
        self.process_button = ttk.Button(control_frame, text="Lancer le pipeline", 
                                        command=self.start_processing, style="Accent.TButton")
        self.process_button.grid(row=0, column=0, padx=(0, 10), sticky=tk.W)
        
        self.stop_button = ttk.Button(control_frame, text="Arrêter", 
                                     command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10), sticky=tk.W)
        
        ttk.Button(control_frame, text="Sauvegarder préférences", 
                  command=self.save_preferences).grid(row=0, column=2, padx=(0, 10), sticky=tk.W)
        
        # === Barre de progression ===
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))
        
        self.status_label = ttk.Label(progress_frame, text="Prêt")
        self.status_label.grid(row=0, column=1, sticky=tk.E)
        
        # === Journal des opérations ===
        log_frame = ttk.LabelFrame(main_frame, text="Journal des opérations", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        # Zone de texte responsive
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW, pady=(0, 5))
        
        # Bouton pour effacer le journal
        clear_btn_frame = ttk.Frame(log_frame)
        clear_btn_frame.grid(row=1, column=0, sticky=tk.EW)
        ttk.Button(clear_btn_frame, text="Effacer le journal", 
                  command=self.clear_log).pack()
    
    def browse_input_file(self):
        """Ouvre une boîte de dialogue pour sélectionner le fichier dalles d'entrée"""
        filename = filedialog.askopenfilename(
            title="Sélectionner le fichier dalles d'entrée",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
    
    def browse_output_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier de sortie"
        )
        if directory:
            self.output_dir.set(directory)
    
    def browse_osgeo4w_bin(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier bin d'OSGeo4W"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier OSGeo4W/bin",
            initialdir="C:/"
        )
        if directory:
            self.osgeo4w_bin.set(directory)
    
    def browse_models_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des modèles"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier contenant les modèles de computer vision",
            initialdir=self.cv_models_dir.get() if self.cv_models_dir.get() else "models"
        )
        if directory:
            self.cv_models_dir.set(directory)
            # Actualiser immédiatement la liste des modèles
            self.refresh_models()

    def browse_local_laz_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des nuages locaux"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier contenant les nuages de points locaux (LAZ/LAS)"
        )
        if directory:
            self.local_laz_dir.set(directory)

    def browse_existing_mnt_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des MNT existants"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier contenant les MNT existants (TIF/ASC)"
        )
        if directory:
            self.existing_mnt_dir.set(directory)

    def browse_existing_rvt_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier des indices RVT existants"""
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier contenant les images RVT (indices existants)"
        )
        if directory:
            self.existing_rvt_dir.set(directory)
    
    def log_message(self, message, progress=None):
        """Ajoute un message au journal et met à jour la progression"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
        # Mettre à jour la progression si fournie
        if progress is not None:
            self.update_progress(progress, f"Traitement en cours... {progress:.1f}%")
        
        self.root.update_idletasks()
    
    def clear_log(self):
        """Efface le journal"""
        self.log_text.delete(1.0, tk.END)
    
    def on_data_mode_changed(self):
        """Active/désactive les champs de sources selon le mode de données sélectionné."""
        mode = self.data_mode.get()

        # ign_laz: fichier liste obligatoire, autres sources désactivées
        if mode == 'ign_laz':
            self._enable_widget_recursive(self.input_file_frame)
            self._disable_widget_recursive(self.local_laz_frame)
            self._disable_widget_recursive(self.existing_mnt_frame)
            self._disable_widget_recursive(self.existing_rvt_frame)
            self.data_mode_help.set("ign_laz : utilise un fichier texte listant les dalles IGN (URLs LAZ) comme source.")

        # local_laz: dossier LAZ obligatoire
        elif mode == 'local_laz':
            self._disable_widget_recursive(self.input_file_frame)
            self._enable_widget_recursive(self.local_laz_frame)
            self._disable_widget_recursive(self.existing_mnt_frame)
            self._disable_widget_recursive(self.existing_rvt_frame)
            self.data_mode_help.set("local_laz : utilise un dossier de nuages locaux (LAZ/LAS) comme point de départ.")

        # existing_mnt: dossier MNT obligatoire
        elif mode == 'existing_mnt':
            self._disable_widget_recursive(self.input_file_frame)
            self._disable_widget_recursive(self.local_laz_frame)
            self._enable_widget_recursive(self.existing_mnt_frame)
            self._disable_widget_recursive(self.existing_rvt_frame)
            self.data_mode_help.set("existing_mnt : repart d'un dossier de MNT existants (TIF/ASC) pour générer RVT et CV.")

        # existing_rvt: dossier RVT obligatoire
        elif mode == 'existing_rvt':
            self._disable_widget_recursive(self.input_file_frame)
            self._disable_widget_recursive(self.local_laz_frame)
            self._disable_widget_recursive(self.existing_mnt_frame)
            self._enable_widget_recursive(self.existing_rvt_frame)
            self.data_mode_help.set("existing_rvt : applique la CV sur un dossier de TIF RVT existants (LD, etc.).")

        else:
            # Fallback: tout activer si mode inconnu
            self._enable_widget_recursive(self.input_file_frame)
            self._enable_widget_recursive(self.local_laz_frame)
            self._enable_widget_recursive(self.existing_mnt_frame)
            self._enable_widget_recursive(self.existing_rvt_frame)
            self.data_mode_help.set("")
    
    def update_progress(self, value, status_text):
        """Met à jour la barre de progression et le texte de statut"""
        self.progress_var.set(value)
        self.status_label.config(text=status_text)
        self.root.update_idletasks()
    
    def start_processing(self):
        """Démarre le traitement du pipeline dans un thread séparé"""
        if self.processing:
            return
        
        # Validation des paramètres selon le mode de données
        mode = self.data_mode.get()

        if mode == 'ign_laz':
            input_file_path = Path(self.input_file.get())
            if not input_file_path.exists():
                messagebox.showerror("Erreur", f"Le fichier dalles IGN n'existe pas:\n{input_file_path}")
                return
        elif mode == 'local_laz':
            local_dir = Path(self.local_laz_dir.get())
            if not local_dir.exists() or not local_dir.is_dir():
                messagebox.showerror("Erreur", f"Le dossier de nuages locaux n'est pas valide:\n{local_dir}")
                return
        elif mode == 'existing_mnt':
            mnt_dir = Path(self.existing_mnt_dir.get())
            if not mnt_dir.exists() or not mnt_dir.is_dir():
                messagebox.showerror("Erreur", f"Le dossier de MNT existants n'est pas valide:\n{mnt_dir}")
                return
        elif mode == 'existing_rvt':
            rvt_dir = Path(self.existing_rvt_dir.get())
            if not rvt_dir.exists() or not rvt_dir.is_dir():
                messagebox.showerror("Erreur", f"Le dossier d'indices RVT existants n'est pas valide:\n{rvt_dir}")
                return
        
        output_dir_path = Path(self.output_dir.get())
        if not output_dir_path.exists():
            try:
                output_dir_path.mkdir(parents=True, exist_ok=True)
                self.log_message(f"Dossier de sortie créé: {output_dir_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de créer le dossier de sortie:\n{output_dir_path}\n{e}")
                return
        
        # Capturer la configuration UI avant de désactiver les contrôles
        # Ceci évite que les paramètres changent pendant l'exécution
        self.frozen_ui_config = self.get_ui_config()
        self.log_message(f"🔒 Configuration UI capturée - SAHI: {self.frozen_ui_config['computer_vision']['sahi']['slice_height']}x{self.frozen_ui_config['computer_vision']['sahi']['slice_width']}")
        
        # Désactiver les contrôles pour éviter les modifications pendant l'exécution
        self.disable_ui_controls()
        self.process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.processing = True
        self.stop_requested = False
        
        # Effacer le journal
        self.clear_log()
        
        # Lancer le traitement dans un thread séparé
        threading.Thread(target=self.process_thread, daemon=True).start()
    
    def process_thread(self):
        """Thread de traitement du pipeline"""
        try:
            # Configuration du logging dans le dossier choisi par l'utilisateur
            setup_logging(Path(self.output_dir.get()))
            
            # Créer l'instance du pipeline avec callback pour les logs
            self.pipeline = LidarPipeline(
                work_dir=self.output_dir.get(),
                ui_callback=self.log_message,
                input_file=self.input_file.get()
            )
            self.pipeline.stop_requested = False
            
            # Lancer le pipeline avec la configuration figée
            success = self.pipeline.run(self.frozen_ui_config)
            
            if success:
                self.log_message("🎉 Pipeline terminé avec succès!")
                self.update_progress(100, "Terminé")
            else:
                self.update_progress(0, "Échec du pipeline")
                self.log_message("❌ Échec du pipeline")
                messagebox.showerror("Erreur", "Échec du pipeline.\nVoir le journal pour plus de détails.")
            
        except KeyboardInterrupt:
            self.log_message("⏹️ Pipeline interrompu par l'utilisateur")
            self.update_progress(0, "Pipeline interrompu")
            messagebox.showwarning("Interrompu", "Pipeline interrompu par l'utilisateur")
            
        except Exception as e:
            error_message = f"Erreur inattendue: {str(e)}"
            self.log_message(f"💥 {error_message}")
            self.update_progress(0, "Erreur")
            messagebox.showerror("Erreur", f"Une erreur inattendue s'est produite:\n{str(e)}")
            
        finally:
            # Réactiver les contrôles
            self.enable_ui_controls()
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.processing = False
    
    def stop_processing(self):
        """Arrête le traitement en cours"""
        if self.processing:
            self.log_message("⏹️ Demande d'arrêt du pipeline...")
            self.stop_requested = True
            if self.pipeline:
                self.pipeline.stop_requested = True
                # Nettoyer les fichiers en cours et revenir à l'état initial
                self.pipeline.cleanup_on_stop()
            self.processing = False
            self.stop_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.NORMAL)
            # Réactiver tous les contrôles UI pour permettre la modification des paramètres
            self.enable_ui_controls()
            self.update_progress(0, "Arrêté par l'utilisateur")
            self.log_message("🔄 Application remise à l'état initial")
    
    def save_preferences(self):
        """Sauvegarde les préférences utilisateur dans le fichier JSON"""
        try:
            # Sauvegarder les fichiers d'entrée et de sortie
            self.config['app']['files']['input_file'] = self.input_file.get()
            self.config['app']['files']['output_dir'] = self.output_dir.get()
            
            # Sauvegarder le dossier OSGeo4W bin
            self.config.setdefault('tools', {})
            self.config['tools']['osgeo4w_bin'] = self.osgeo4w_bin.get()
            
            # Sauvegarder le dossier des modèles
            self.config['computer_vision']['models_dir'] = self.cv_models_dir.get()

            # Sauvegarder le mode de données et les dossiers associés
            self.config['processing']['data_mode'] = self.data_mode.get()
            self.config['processing']['local_laz_dir'] = self.local_laz_dir.get()
            self.config['processing']['existing_mnt_dir'] = self.existing_mnt_dir.get()
            self.config['processing']['existing_rvt_dir'] = self.existing_rvt_dir.get()
            
            # Mettre à jour la configuration avec les valeurs de l'interface
            self.config['processing']['mnt_resolution'] = self.mnt_resolution.get()
            self.config['processing']['density_resolution'] = self.density_resolution.get()
            self.config['processing']['tile_overlap'] = self.tile_overlap.get()
            self.config['processing']['filter_expression'] = self.filter_expression.get()
            
            # Produits
            self.config['processing']['products']['MNT'] = self.product_mnt.get()
            self.config['processing']['products']['DENSITE'] = self.product_densite.get()
            self.config['processing']['products']['M_HS'] = self.product_m_hs.get()
            self.config['processing']['products']['SVF'] = self.product_svf.get()
            self.config['processing']['products']['SLO'] = self.product_slo.get()
            self.config['processing']['products']['LD'] = self.product_ld.get()
            self.config['processing']['products']['VAT'] = self.product_vat.get()
            
            # Paramètres RVT
            self.config['rvt_params']['mdh']['num_directions'] = self.mdh_num_directions.get()
            self.config['rvt_params']['mdh']['sun_elevation'] = self.mdh_sun_elevation.get()
            self.config['rvt_params']['mdh']['ve_factor'] = self.mdh_ve_factor.get()
            self.config['rvt_params']['mdh']['save_as_8bit'] = self.mdh_save_as_8bit.get()
            
            self.config['rvt_params']['svf']['noise_remove'] = self.svf_noise_remove.get()
            self.config['rvt_params']['svf']['num_directions'] = self.svf_num_directions.get()
            self.config['rvt_params']['svf']['radius'] = self.svf_radius.get()
            self.config['rvt_params']['svf']['ve_factor'] = self.svf_ve_factor.get()
            self.config['rvt_params']['svf']['save_as_8bit'] = self.svf_save_as_8bit.get()
            
            self.config['rvt_params']['slope']['unit'] = self.slope_unit.get()
            self.config['rvt_params']['slope']['ve_factor'] = self.slope_ve_factor.get()
            self.config['rvt_params']['slope']['save_as_8bit'] = self.slope_save_as_8bit.get()
            
            self.config['rvt_params']['ldo']['angular_res'] = self.ldo_angular_res.get()
            self.config['rvt_params']['ldo']['min_radius'] = self.ldo_min_radius.get()
            self.config['rvt_params']['ldo']['max_radius'] = self.ldo_max_radius.get()
            self.config['rvt_params']['ldo']['observer_h'] = self.ldo_observer_h.get()
            self.config['rvt_params']['ldo']['ve_factor'] = self.ldo_ve_factor.get()
            self.config['rvt_params']['ldo']['save_as_8bit'] = self.ldo_save_as_8bit.get()

            # VAT
            self.config['rvt_params'].setdefault('vat', {})['terrain_type'] = self.vat_terrain_type.get()
            self.config['rvt_params']['vat']['save_as_8bit'] = self.vat_save_as_8bit.get()
            # Sorties JPG RVT
            jpg_cfg = self.config['processing'].setdefault('output_formats', {}).setdefault('jpg', {})
            jpg_cfg['M_HS'] = self.jpg_mhs.get()
            jpg_cfg['SVF'] = self.jpg_svf.get()
            jpg_cfg['SLO'] = self.jpg_slo.get()
            jpg_cfg['LD'] = self.jpg_ld.get()
            jpg_cfg['VAT'] = self.jpg_vat.get()
            
            # Computer Vision
            self.config['computer_vision']['enabled'] = self.cv_enabled.get()
            self.config['computer_vision']['selected_model'] = self.cv_selected_model.get()
            self.config['computer_vision']['target_rvt'] = self.cv_target_rvt.get()
            self.config['computer_vision']['confidence_threshold'] = self.cv_confidence_threshold.get()
            self.config['computer_vision']['iou_threshold'] = self.cv_iou_threshold.get()
            self.config['computer_vision']['generate_annotated_images'] = self.cv_generate_annotated_images.get()
            self.config['computer_vision']['generate_shapefiles'] = self.cv_generate_shapefiles.get()
            
            # SAHI parameters
            self.config['computer_vision']['sahi']['slice_height'] = self.cv_slice_height.get()
            self.config['computer_vision']['sahi']['slice_width'] = self.cv_slice_width.get()
            self.config['computer_vision']['sahi']['overlap_ratio'] = self.cv_overlap_ratio.get()
            
            # Sauvegarder dans le fichier JSON
            save_config(self.config)
            
            messagebox.showinfo("Sauvegardé", "Préférences et configuration sauvegardées avec succès!")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde:\n{e}")
    
    def reset_general_config(self):
        """Remet les valeurs par défaut pour la configuration générale"""
        self.mnt_resolution.set(0.5)
        self.density_resolution.set(1.0)
        self.tile_overlap.set(20)
        self.filter_expression.set("Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9")
        
        # Produits par défaut
        self.product_mnt.set(True)
        self.product_densite.set(True)
        self.product_m_hs.set(True)
        self.product_svf.set(True)
        self.product_slo.set(True)
        self.product_ld.set(True)
    
    def reset_rvt_params(self):
        """Remet les valeurs par défaut pour les paramètres RVT"""
        # MDH
        self.mdh_num_directions.set(16)
        self.mdh_sun_elevation.set(35)
        self.mdh_ve_factor.set(1)
        self.mdh_save_as_8bit.set(True)
        
        # SVF
        self.svf_noise_remove.set(0)
        self.svf_num_directions.set(16)
        self.svf_radius.set(10)
        self.svf_ve_factor.set(1)
        self.svf_save_as_8bit.set(True)
        
        # Slope
        self.slope_unit.set(0)
        self.slope_ve_factor.set(1)
        self.slope_save_as_8bit.set(True)
        
        # LDO
        self.ldo_angular_res.set(15)
        self.ldo_min_radius.set(10)
        self.ldo_max_radius.set(20)
        self.ldo_observer_h.set(1.7)
        self.ldo_ve_factor.set(1)
        self.ldo_save_as_8bit.set(True)

        # VAT
        self.vat_terrain_type.set(0)
        self.vat_save_as_8bit.set(True)
        self.jpg_vat.set(True)
    
    def get_ui_config(self):
        """Récupère la configuration depuis l'interface utilisateur"""
        return {
            'processing': {
                'mnt_resolution': self.mnt_resolution.get(),
                'density_resolution': self.density_resolution.get(),
                'tile_overlap': self.tile_overlap.get(),
                'filter_expression': self.filter_expression.get(),
                'data_mode': self.data_mode.get(),
                'local_laz_dir': self.local_laz_dir.get(),
                'existing_mnt_dir': self.existing_mnt_dir.get(),
                'existing_rvt_dir': self.existing_rvt_dir.get(),
                'products': {
                    'MNT': self.product_mnt.get(),
                    'DENSITE': self.product_densite.get(),
                    'M_HS': self.product_m_hs.get(),
                    'SVF': self.product_svf.get(),
                    'SLO': self.product_slo.get(),
                    'LD': self.product_ld.get(),
                    'VAT': self.product_vat.get()
                }
            },
            'tools': {
                'osgeo4w_bin': self.osgeo4w_bin.get()
            },
            'rvt_params': {
                'mdh': {
                    'num_directions': self.mdh_num_directions.get(),
                    'sun_elevation': self.mdh_sun_elevation.get(),
                    've_factor': self.mdh_ve_factor.get(),
                    'save_as_8bit': self.mdh_save_as_8bit.get()
                },
                'svf': {
                    'noise_remove': self.svf_noise_remove.get(),
                    'num_directions': self.svf_num_directions.get(),
                    'radius': self.svf_radius.get(),
                    've_factor': self.svf_ve_factor.get(),
                    'save_as_8bit': self.svf_save_as_8bit.get()
                },
                'slope': {
                    'unit': self.slope_unit.get(),
                    've_factor': self.slope_ve_factor.get(),
                    'save_as_8bit': self.slope_save_as_8bit.get()
                },
                'ldo': {
                    'angular_res': self.ldo_angular_res.get(),
                    'min_radius': self.ldo_min_radius.get(),
                    'max_radius': self.ldo_max_radius.get(),
                    'observer_h': self.ldo_observer_h.get(),
                    've_factor': self.ldo_ve_factor.get(),
                    'save_as_8bit': self.ldo_save_as_8bit.get()
                },
                'vat': {
                    'terrain_type': self.vat_terrain_type.get(),
                    'blend_combination': 0,
                    'save_as_8bit': self.vat_save_as_8bit.get()
                }
            },
            'computer_vision': {
                'enabled': self.cv_enabled.get(),
                'selected_model': self.cv_selected_model.get(),
                'target_rvt': self.cv_target_rvt.get(),
                'models_dir': self.cv_models_dir.get(),
                'confidence_threshold': self.cv_confidence_threshold.get(),
                'iou_threshold': self.cv_iou_threshold.get(),
                'generate_annotated_images': self.cv_generate_annotated_images.get(),
                'generate_shapefiles': self.cv_generate_shapefiles.get(),
                'sahi': {
                    'slice_height': self.cv_slice_height.get(),
                    'slice_width': self.cv_slice_width.get(),
                    'overlap_ratio': self.cv_overlap_ratio.get()
                }
            }
        }
    
    def refresh_models(self):
        """Actualise la liste des modèles disponibles dans le répertoire models"""
        models_path = Path(self.cv_models_dir.get())
        models = []
        
        if models_path.exists():
            # Chercher tous les répertoires dans models/
            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    # Vérifier que le modèle a les fichiers requis
                    weights_file = model_dir / "weights" / "best.pt"
                    args_file = model_dir / "args.yaml"
                    
                    if weights_file.exists() and args_file.exists():
                        models.append(model_dir.name)
        # Tri pour un ordre déterministe (évite des changements aléatoires)
        models.sort()

        # Sauvegarder la sélection actuelle
        previous_selection = self.cv_selected_model.get()

        # Mettre à jour la combobox
        self.cv_model_combo['values'] = models

        # Politique de sélection:
        # - Si une sélection existe et qu'elle est encore disponible, la conserver
        # - Si aucune sélection n'existe, et qu'il y a des modèles, sélectionner le premier
        # - Si la sélection n'est plus disponible, NE PAS la remplacer automatiquement
        #   (évite les changements inattendus). Laisser l'utilisateur choisir.
        if previous_selection:
            if previous_selection in models:
                self.cv_selected_model.set(previous_selection)
            else:
                # Préserver la valeur courante (même si absente) pour ne pas surprendre l'utilisateur
                # Optionnel: on pourrait logger un avertissement ici.
                pass
        else:
            if models:
                self.cv_selected_model.set(models[0])
            else:
                self.cv_selected_model.set("")
    
    def get_selected_rvt_products(self):
        """Retourne la liste des produits RVT sélectionnés"""
        selected_rvt = []
        if self.product_m_hs.get():
            selected_rvt.append('M_HS')
        if self.product_svf.get():
            selected_rvt.append('SVF')
        if self.product_slo.get():
            selected_rvt.append('SLO')
        if self.product_ld.get():
            selected_rvt.append('LD')
        if self.product_vat.get():
            selected_rvt.append('VAT')
        return selected_rvt
    
    def update_available_rvt(self):
        """Met à jour la liste des RVT disponibles pour la CV selon les produits sélectionnés"""
        selected_rvt = self.get_selected_rvt_products()
        
        # Mettre à jour la combobox RVT
        self.cv_rvt_combo['values'] = selected_rvt
        
        # Si le RVT sélectionné n'est plus disponible, prendre le premier disponible
        if self.cv_target_rvt.get() not in selected_rvt:
            if selected_rvt:
                self.cv_target_rvt.set(selected_rvt[0])
            else:
                self.cv_target_rvt.set("")
        
        # Mettre à jour l'état de la CV
        self.update_cv_availability()
    
    def update_cv_availability(self):
        """Met à jour la disponibilité de la CV selon les RVT sélectionnés"""
        selected_rvt = self.get_selected_rvt_products()
        has_rvt = len(selected_rvt) > 0
        
        # Si aucun RVT sélectionné, désactiver la CV
        if not has_rvt and self.cv_enabled.get():
            self.cv_enabled.set(False)
            messagebox.showwarning("Computer Vision", 
                                 "La détection par computer vision a été désactivée car aucun produit RVT n'est sélectionné.")
        
        # Activer/désactiver les contrôles CV selon la disponibilité des RVT
        self.on_cv_enabled_changed()
    
    def on_rvt_products_changed(self):
        """Appelé quand les produits RVT changent"""
        self.update_available_rvt()
    
    def on_cv_enabled_changed(self):
        """Appelé quand l'état de la détection CV change"""
        enabled = self.cv_enabled.get()
        selected_rvt = self.get_selected_rvt_products()
        has_rvt = len(selected_rvt) > 0
        
        # Vérifier si on peut activer la CV
        if enabled and not has_rvt:
            self.cv_enabled.set(False)
            messagebox.showwarning("Computer Vision", 
                                 "Impossible d'activer la détection par computer vision : aucun produit RVT n'est sélectionné.")
            enabled = False
        
        # Activer/désactiver les contrôles selon l'état et la disponibilité
        controls_enabled = enabled and has_rvt
        self.cv_model_combo.config(state="readonly" if controls_enabled else "disabled")
        self.cv_rvt_combo.config(state="readonly" if controls_enabled else "disabled")
    
    def disable_ui_controls(self):
        """Désactive tous les contrôles UI pendant l'exécution du pipeline"""
        # Désactiver les widgets de configuration SAHI pour éviter les modifications
        for widget in self.scrollable_frame.winfo_children():
            self._disable_widget_recursive(widget)
    
    def enable_ui_controls(self):
        """Réactive tous les contrôles UI après l'exécution du pipeline"""
        # Réactiver les widgets de configuration
        for widget in self.scrollable_frame.winfo_children():
            self._enable_widget_recursive(widget)
        
        # Restaurer l'état correct des contrôles CV
        self.on_cv_enabled_changed()
    
    def _disable_widget_recursive(self, widget):
        """Désactive récursivement un widget et ses enfants"""
        try:
            # Ne pas désactiver les widgets de log, boutons stop, et barre de progression
            if widget == self.log_text or widget == self.stop_button or widget == self.progress_bar or widget == self.status_label:
                return
            
            # Ne pas désactiver les frames contenant les logs
            if hasattr(widget, 'winfo_children'):
                children = widget.winfo_children()
                if self.log_text in children or self.stop_button in children or self.progress_bar in children:
                    # Désactiver seulement les enfants, pas le frame parent
                    for child in children:
                        if child not in [self.log_text, self.stop_button, self.progress_bar, self.status_label]:
                            self._disable_widget_recursive(child)
                    return
            
            # Désactiver le widget s'il a un état
            if hasattr(widget, 'config') and 'state' in widget.keys():
                widget.config(state='disabled')
            
            # Récursion sur les enfants
            for child in widget.winfo_children():
                self._disable_widget_recursive(child)
        except:
            pass  # Ignorer les erreurs de widgets détruits
    
    def _enable_widget_recursive(self, widget):
        """Réactive récursivement un widget et ses enfants"""
        try:
            # Ne pas toucher aux widgets de log qui doivent rester actifs
            if widget == self.log_text or widget == self.stop_button or widget == self.progress_bar or widget == self.status_label:
                return
            
            # Réactiver le widget s'il a un état
            if hasattr(widget, 'config') and 'state' in widget.keys():
                widget_type = widget.winfo_class()
                if widget_type in ['TCombobox']:
                    widget.config(state='readonly')
                else:
                    widget.config(state='normal')
            
            # Récursion sur les enfants
            for child in widget.winfo_children():
                self._enable_widget_recursive(child)
        except:
            pass  # Ignorer les erreurs de widgets détruits


def main_cli():
    """Fonction principale pour l'interface en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement LIDAR archéologique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run_pipeline.py                    # Interface graphique
  python run_pipeline.py --cli              # Interface en ligne de commande
  python run_pipeline.py --cli --work-dir /path # Répertoire de travail spécifique
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Utiliser l'interface en ligne de commande"
    )
    
    parser.add_argument(
        "--work-dir", "-w",
        help="Répertoire de travail (par défaut: répertoire courant)"
    )
    
    
    args = parser.parse_args()
    
    # Configuration du logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Changement de répertoire si spécifié
    if args.work_dir:
        work_dir = Path(args.work_dir)
        if not work_dir.exists():
            logger.error(f"Répertoire de travail inexistant: {work_dir}")
            return 1
        os.chdir(work_dir)
        logger.info(f"Répertoire de travail: {work_dir}")
    
    # Lancement du pipeline
    try:
        logger.info("Initialisation du pipeline LIDAR...")
        pipeline = LidarPipeline(work_dir=args.work_dir)
        
        # Exécution du pipeline
        logger.info("Démarrage du traitement...")
        success = pipeline.run()
        
        if success:
            logger.info("Pipeline terminé avec succès!")
            return 0
        else:
            logger.error("Échec du pipeline")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}", exc_info=True)
        return 1


def main():
    """Fonction principale"""
    # Vérifier si l'argument --cli est fourni pour l'interface en ligne de commande
    if len(sys.argv) > 1 and "--cli" in sys.argv:
        return main_cli()
    
    # Sinon, lancer l'interface graphique avec optimisations
    try:
        # Utiliser Tkinter standard avec style personnalisé pour de meilleures performances
        root = tk.Tk()
        
        # Appliquer un style moderne mais léger
        style = ttk.Style()
        
        # Définir une palette de couleurs moderne
        bg_color = "#f0f0f0"
        accent_color = "#0078d4"
        text_color = "#323130"
        
        # Configurer les couleurs de base
        root.configure(bg=bg_color)
        
        # Styles personnalisés légers
        style.configure('TLabel', background=bg_color, foreground=text_color)
        style.configure('TFrame', background=bg_color)
        style.configure('TLabelFrame', background=bg_color, foreground=text_color)
        style.configure('TButton', focuscolor='none')
        style.configure('Accent.TButton', 
                       background=accent_color, 
                       foreground='black',
                       borderwidth=2,
                       relief='raised',
                       font=('Segoe UI', 9, 'bold'))
        style.map('Accent.TButton',
                 background=[('active', '#106ebe'), ('pressed', '#005a9e')],
                 foreground=[('active', 'black'), ('pressed', 'black')],
                 relief=[('pressed', 'sunken'), ('active', 'raised')])
        
        # Optimisations de performance
        root.option_add('*TCombobox*Listbox.selectBackground', accent_color)
        
        app = PipelineApp(root)
        root.mainloop()
        return 0
    except Exception as e:
        print(f"Erreur lors du lancement de l'interface graphique: {e}")
        return 1
        print("Utilisez --cli pour l'interface en ligne de commande")
        return 1


if __name__ == "__main__":
    sys.exit(main())
