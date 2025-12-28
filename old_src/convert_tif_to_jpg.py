#!/usr/bin/env python3
"""
Script de conversion d'images TIF vers JPG
Convertit tous les fichiers .tif d'un répertoire en .jpg avec optimisation de qualité

Interface graphique Tkinter pour sélectionner les dossiers d'entrée et de sortie.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import logging
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
from typing import Optional

def setup_logging():
    """Configure le système de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('conversion_tif_jpg.log')
        ]
    )
    return logging.getLogger(__name__)

def convert_tif_to_jpg(input_path, output_path, quality=95, create_world_file=True, reference_tif_path=None):
    """
    Convertit un fichier TIF en JPG avec création optionnelle du fichier world (.jgw)
    
    Args:
        input_path (str): Chemin vers le fichier TIF source
        output_path (str): Chemin vers le fichier JPG de sortie
        quality (int): Qualité JPEG (1-100, défaut: 95)
        create_world_file (bool): Créer le fichier .jgw pour le géoréférencement
        reference_tif_path (str): TIF de référence pour le géoréférencement (si différent de input_path)
    
    Returns:
        bool: True si la conversion a réussi, False sinon
    """
    try:
        with Image.open(input_path) as img:
            # Convertir en RGB si nécessaire (pour éviter les erreurs avec les modes non supportés)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Créer un fond blanc pour les images avec transparence
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Sauvegarder en JPG avec la qualité spécifiée
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        # Créer le fichier world (.jgw) si demandé
        if create_world_file:
            # Utiliser le TIF de référence si fourni, sinon le TIF source
            ref_tif = reference_tif_path if reference_tif_path else input_path
            success_world = create_world_file_from_tif(ref_tif, output_path)
            if not success_world:
                logging.warning(f"Impossible de créer le fichier world pour {output_path}")
        
        return True
            
    except Exception as e:
        logging.error(f"Erreur lors de la conversion de {input_path}: {e}")
        return False


def create_world_file_from_tif(input_tif_path, output_jpg_path):
    """
    Crée un fichier world (.jgw) à partir des informations géographiques d'un TIF
    
    Args:
        input_tif_path (str): Chemin vers le fichier TIF source
        output_jpg_path (str): Chemin vers le fichier JPG de sortie
    
    Returns:
        bool: True si le fichier world a été créé, False sinon
    """
    try:
        # Essayer d'utiliser rasterio en priorité (plus fiable)
        try:
            import rasterio
            from rasterio.transform import Affine
            
            # Ouvrir le fichier TIF avec rasterio
            with rasterio.open(input_tif_path) as dataset:
                transform = dataset.transform
                
                # Extraire les paramètres de géotransformation
                # Format rasterio: Affine(a, b, c, d, e, f) où:
                # a = pixel width, b = row rotation, c = x coordinate of upper-left corner
                # d = column rotation, e = pixel height (negative), f = y coordinate of upper-left corner
                pixel_width = transform.a
                row_rotation = transform.b
                x_origin = transform.c
                col_rotation = transform.d
                pixel_height = transform.e  # négatif
                y_origin = transform.f
                
                logging.info("Géoréférencement extrait avec rasterio")
                
        except Exception as rasterio_err:
            # Fallback vers GDAL (bindings Python)
            try:
                from osgeo import gdal
                
                # Ouvrir le fichier TIF source
                dataset = gdal.Open(input_tif_path)
                if not dataset:
                    logging.warning(f"Impossible d'ouvrir le fichier TIF: {input_tif_path}")
                    return False
                
                # Extraire les informations de géoréférencement
                geotransform = dataset.GetGeoTransform()
                if not geotransform or geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                    logging.warning(f"Pas d'informations de géoréférencement dans: {input_tif_path}")
                    dataset = None
                    return False
                
                # Format GDAL: (x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height)
                x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height = geotransform
                dataset = None
                
                logging.info("Géoréférencement extrait avec GDAL (bindings)")
                
            except Exception as gdal_py_err:
                # Fallback final: utiliser gdalinfo -json (CLI) si disponible
                import json
                import shutil
                import subprocess
                
                gdalinfo = shutil.which('gdalinfo') or shutil.which(r'C:\OSGeo4W\bin\gdalinfo.exe')
                if not gdalinfo:
                    logging.warning("Ni rasterio ni GDAL disponibles - impossible de créer le fichier world")
                    return False
                try:
                    proc = subprocess.run([gdalinfo, '-json', input_tif_path], capture_output=True, text=True, check=True)
                    info = json.loads(proc.stdout)
                    gt = info.get('geoTransform') or info.get('geoTransform', None)
                    if not gt or len(gt) < 6:
                        logging.warning(f"gdalinfo n'a pas renvoyé de geoTransform utilisable pour: {input_tif_path}")
                        return False
                    # geoTransform: [x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height]
                    x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height = gt[:6]
                    logging.info("Géoréférencement extrait avec gdalinfo (CLI)")
                except Exception as cli_err:
                    logging.warning(f"Échec gdalinfo -json pour {input_tif_path}: {cli_err}")
                    return False
        
        # Créer le chemin du fichier .jgw
        jgw_path = Path(output_jpg_path).with_suffix('.jgw')
        
        # Écrire le fichier world
        with open(jgw_path, 'w') as f:
            f.write(f"{pixel_width:.10f}\n")      # Pixel size X
            f.write(f"{row_rotation:.10f}\n")     # Rotation Y
            f.write(f"{col_rotation:.10f}\n")     # Rotation X
            f.write(f"{pixel_height:.10f}\n")     # Pixel size Y (généralement négatif)
            f.write(f"{x_origin:.10f}\n")         # X origine (coin supérieur gauche)
            f.write(f"{y_origin:.10f}\n")         # Y origine (coin supérieur gauche)
        logging.info(f"Fichier world créé: {jgw_path}")
        return True
        
    except Exception as e:
        logging.error(f"Erreur lors de la création du fichier world: {e}")
        return False

def convert_directory(input_dir, output_dir=None, quality=95, overwrite=False):
    """
    Convertit tous les fichiers TIF d'un répertoire en JPG
    
    Args:
        input_dir (str): Répertoire contenant les fichiers TIF
        output_dir (str): Répertoire de sortie (optionnel, par défaut même répertoire)
        quality (int): Qualité JPEG (1-100)
        overwrite (bool): Écraser les fichiers existants
    
    Returns:
        tuple: (nombre_convertis, nombre_erreurs)
    """
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Le répertoire d'entrée n'existe pas: {input_dir}")
        return 0, 0
    
    # Définir le répertoire de sortie
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Trouver tous les fichiers TIF
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.TIF")) + \
                list(input_path.glob("*.tiff")) + list(input_path.glob("*.TIFF"))
    
    if not tif_files:
        logger.warning(f"Aucun fichier TIF trouvé dans {input_dir}")
        return 0, 0
    
    logger.info(f"Trouvé {len(tif_files)} fichier(s) TIF à convertir")
    
    converted = 0
    errors = 0
    
    for tif_file in tif_files:
        # Créer le nom du fichier JPG
        jpg_name = tif_file.stem + ".jpg"
        jpg_path = output_path / jpg_name
        
        # Vérifier si le fichier existe déjà
        if jpg_path.exists() and not overwrite:
            logger.info(f"Fichier existant ignoré: {jpg_path} (utilisez --overwrite pour écraser)")
            continue
        
        logger.info(f"Conversion: {tif_file.name} -> {jpg_name}")
        
        if convert_tif_to_jpg(str(tif_file), str(jpg_path), quality, create_world_file=True):
            converted += 1
            logger.info(f"✓ Converti: {jpg_name}")
        else:
            errors += 1
            logger.error(f"✗ Échec: {tif_file.name}")
    
    return converted, errors

class TifToJpgApp:
    """Application Tkinter pour convertir les images TIF en JPG"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Conversion TIF vers JPG")
        self.root.geometry("600x450")
        self.root.resizable(True, True)
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.quality_var = tk.IntVar(value=95)
        self.overwrite_var = tk.BooleanVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Prêt à convertir")
        self.processing = False
        
        # Créer l'interface
        self.create_widgets()
        
    def create_widgets(self):
        """Crée les widgets de l'interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Titre
        title_label = ttk.Label(main_frame, text="Conversion TIF vers JPG", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Dossier d'entrée
        ttk.Label(main_frame, text="Dossier d'entrée:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=1, column=1, 
                                                                         sticky=(tk.W, tk.E), 
                                                                         padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Parcourir", 
                  command=self.browse_input_dir).grid(row=1, column=2, pady=5)
        
        # Dossier de sortie
        ttk.Label(main_frame, text="Dossier de sortie:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, 
                                                                          sticky=(tk.W, tk.E), 
                                                                          padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Parcourir", 
                  command=self.browse_output_dir).grid(row=2, column=2, pady=5)
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(1, weight=1)
        
        # Qualité JPEG
        ttk.Label(options_frame, text="Qualité JPEG:").grid(row=0, column=0, sticky=tk.W, pady=5)
        quality_frame = ttk.Frame(options_frame)
        quality_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        quality_scale = ttk.Scale(quality_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.quality_var, length=200)
        quality_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        quality_frame.columnconfigure(0, weight=1)
        
        quality_label = ttk.Label(quality_frame, textvariable=self.quality_var)
        quality_label.grid(row=0, column=1, padx=(10, 0))
        
        # Écraser les fichiers existants
        ttk.Checkbutton(options_frame, text="Écraser les fichiers existants", 
                       variable=self.overwrite_var).grid(row=1, column=0, columnspan=2, 
                                                         sticky=tk.W, pady=5)
        
        # Bouton de traitement
        self.process_button = ttk.Button(main_frame, text="Démarrer la conversion", 
                                        command=self.start_processing,
                                        style='Accent.TButton')
        self.process_button.grid(row=4, column=0, columnspan=3, pady=20)
        
        # Barre de progression
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        ttk.Label(progress_frame, text="Progression:").grid(row=0, column=0, sticky=tk.W)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Statut
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        status_frame.columnconfigure(0, weight=1)
        
        ttk.Label(status_frame, text="Statut:").grid(row=0, column=0, sticky=tk.W)
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                foreground="blue")
        status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal", padding="5")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def browse_input_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier d'entrée"""
        directory = filedialog.askdirectory(title="Sélectionner le dossier contenant les images TIF")
        if directory:
            self.input_dir.set(directory)
    
    def browse_output_dir(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie"""
        directory = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message):
        """Ajoute un message au journal"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, value, status_text):
        """Met à jour la barre de progression et le texte de statut"""
        self.progress_var.set(value)
        self.status_var.set(status_text)
        self.log_message(status_text)
        # Force la mise à jour de l'interface
        self.root.update_idletasks()
    
    def start_processing(self):
        """Démarre la conversion des images dans un thread séparé"""
        # Vérifier les dossiers
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier d'entrée valide.")
            return
        
        if not output_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de sortie.")
            return
        
        # Désactiver les contrôles pendant le traitement
        self.process_button.config(state=tk.DISABLED)
        self.processing = True
        
        # Vider le journal
        self.log_text.delete(1.0, tk.END)
        
        # Démarrer le traitement dans un thread séparé
        thread = threading.Thread(target=self.process_thread, args=(input_dir, output_dir))
        thread.daemon = True
        thread.start()
    
    def process_thread(self, input_dir, output_dir):
        """Thread de conversion des images"""
        try:
            # Réinitialiser la barre de progression
            self.update_progress(0, "Démarrage de la conversion...")
            
            # Obtenir les paramètres
            quality = int(self.quality_var.get())
            overwrite = self.overwrite_var.get()
            
            # Trouver tous les fichiers TIF
            input_path = Path(input_dir)
            tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.TIF")) + \
                        list(input_path.glob("*.tiff")) + list(input_path.glob("*.TIFF"))
            
            if not tif_files:
                self.update_progress(0, "Aucun fichier TIF trouvé dans le dossier d'entrée")
                messagebox.showwarning("Attention", "Aucun fichier TIF trouvé dans le dossier d'entrée.")
                return
            
            # Créer le dossier de sortie si nécessaire
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            converted = 0
            errors = 0
            total_files = len(tif_files)
            
            self.update_progress(0, f"Conversion de {total_files} fichier(s)...")
            
            for i, tif_file in enumerate(tif_files):
                # Créer le nom du fichier JPG
                jpg_name = tif_file.stem + ".jpg"
                jpg_path = output_path / jpg_name
                
                # Vérifier si le fichier existe déjà
                if jpg_path.exists() and not overwrite:
                    self.update_progress((i + 1) / total_files * 100, 
                                       f"Fichier existant ignoré: {jpg_name}")
                    continue
                
                self.update_progress(i / total_files * 100, 
                                   f"Conversion: {tif_file.name} -> {jpg_name}")
                
                if convert_tif_to_jpg(str(tif_file), str(jpg_path), quality, create_world_file=True):
                    converted += 1
                    self.update_progress((i + 1) / total_files * 100, 
                                       f"✓ Converti: {jpg_name}")
                else:
                    errors += 1
                    self.update_progress((i + 1) / total_files * 100, 
                                       f"✗ Échec: {tif_file.name}")
            
            # Conversion terminée
            final_message = f"Conversion terminée. {converted} fichiers convertis, {errors} erreurs."
            self.update_progress(100, final_message)
            messagebox.showinfo("Terminé", final_message)
            
        except Exception as e:
            error_message = f"Erreur lors de la conversion: {str(e)}"
            self.update_progress(0, error_message)
            messagebox.showerror("Erreur", f"Une erreur s'est produite lors de la conversion:\n{str(e)}")
        
        finally:
            # Réactiver les contrôles
            self.process_button.config(state=tk.NORMAL)
            self.processing = False

def main_cli():
    """Fonction principale pour l'interface en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Convertit des images TIF en JPG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python convert_tif_to_jpg.py /chemin/vers/images
  python convert_tif_to_jpg.py /chemin/vers/images --output /chemin/sortie
  python convert_tif_to_jpg.py /chemin/vers/images --quality 85 --overwrite
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Répertoire contenant les fichiers TIF à convertir"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Répertoire de sortie (par défaut: même répertoire que l'entrée)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="Qualité JPEG (1-100, défaut: 95)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écraser les fichiers JPG existants"
    )
    
    args = parser.parse_args()
    
    # Validation des arguments
    if not (1 <= args.quality <= 100):
        print("Erreur: La qualité doit être entre 1 et 100")
        return 1
    
    # Configuration du logging
    logger = setup_logging()
    
    logger.info("=== Début de la conversion TIF vers JPG ===")
    logger.info(f"Répertoire d'entrée: {args.input_dir}")
    logger.info(f"Répertoire de sortie: {args.output or 'même répertoire'}")
    logger.info(f"Qualité JPEG: {args.quality}")
    logger.info(f"Écraser fichiers existants: {args.overwrite}")
    
    # Effectuer la conversion
    converted, errors = convert_directory(
        args.input_dir,
        args.output,
        args.quality,
        args.overwrite
    )
    
    # Résumé
    logger.info("=== Résumé de la conversion ===")
    logger.info(f"Fichiers convertis: {converted}")
    logger.info(f"Erreurs: {errors}")
    
    if errors > 0:
        logger.warning("Des erreurs se sont produites pendant la conversion")
        return 1
    else:
        logger.info("Conversion terminée avec succès!")
        return 0

def main():
    """Fonction principale"""
    # Vérifier si des arguments sont fournis pour l'interface en ligne de commande
    if len(sys.argv) > 1:
        return main_cli()
    
    # Sinon, lancer l'interface graphique
    root = tk.Tk()
    app = TifToJpgApp(root)
    root.mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main())
