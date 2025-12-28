"""
Module de computer vision pour la détection d'objets sur les images RVT
Utilise YOLO pour la détection et génère des shapefiles des résultats
"""

import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile

logger = logging.getLogger(__name__)


def _get_ultralytics_task(model_path: str) -> Optional[str]:
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        task = getattr(model, "task", None)
        if isinstance(task, str) and task:
            return task
    except Exception as e:
        logger.debug(f"Impossible de déterminer la tâche Ultralytics pour {model_path}: {e}")
    return None


def _save_empty_outputs(image_path: str, output_path: str, jpg_folder_path: Optional[str] = None):
    try:
        image_name = Path(image_path).stem
        if jpg_folder_path:
            labels_txt = Path(jpg_folder_path) / f"{image_name}.txt"
            labels_json = Path(jpg_folder_path) / f"{image_name}.json"
        else:
            output_path_obj = Path(output_path)
            labels_txt = output_path_obj.with_suffix('.txt')
            labels_json = output_path_obj.with_suffix('.json')

        if not labels_txt.exists():
            labels_txt.write_text("")
            logger.info(f"Fichier labels vide créé (aucune détection): {labels_txt}")

        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            img_width, img_height = None, None

        payload = {
            "image_path": image_path,
            "image_dimensions": {"width": img_width, "height": img_height},
            "detections": []
        }
        labels_json.write_text(json.dumps(payload, indent=2))
        logger.info(f"JSON vide créé (aucune détection): {labels_json}")
    except Exception as write_e:
        logger.warning(f"Impossible d'écrire les sorties vides pour {image_path}: {write_e}")

def get_detection_output_path(jpg_path: str, target_rvt: str, annotated_output_dir: str = None) -> str:
    """
    Génère le chemin de sortie pour les détections dans le dossier annotated_images
    
    Args:
        jpg_path: Chemin vers l'image JPG
        target_rvt: Type de produit RVT (LDO, SVF, etc.)
        annotated_output_dir: Répertoire de sortie pour les images annotées
    
    Returns:
        Chemin vers le fichier de détection de sortie dans annotated_images
    """
    jpg_file = Path(jpg_path)
    # Remplacer .jpg par _detections.jpg
    detection_name = jpg_file.stem + "_detections.jpg"
    
    if annotated_output_dir:
        # Utiliser le répertoire annotated_images fourni
        detection_path = Path(annotated_output_dir) / detection_name
    else:
        # Fallback: utiliser le même répertoire que l'image source
        detection_path = jpg_file.parent / detection_name
    
    return str(detection_path)

def run_inference(image_path: str, model_path: str, args_path: str, output_path: str, 
                 confidence_threshold: float = 0.5, slice_height: int = 640, 
                 slice_width: int = 640, overlap_ratio: float = 0.2, 
                 generate_annotated_images: bool = False, annotated_output_dir: str = None,
                 iou_threshold: float = 0.5, jpg_folder_path: str = None,
                 max_det: int = 1000) -> bool:
    """
    Exécute l'inférence YOLO sur une image
    
    Args:
        image_path: Chemin vers l'image d'entrée
        model_path: Chemin vers le modèle YOLO
        args_path: Chemin vers le fichier d'arguments
        output_path: Chemin de sortie pour l'image avec détections
        confidence_threshold: Seuil de confiance pour les détections
        slice_height: Hauteur des tuiles pour SAHI
        slice_width: Largeur des tuiles pour SAHI
        overlap_ratio: Ratio de chevauchement pour les tuiles
        generate_annotated_images: Générer des images annotées
        annotated_output_dir: Répertoire de sortie pour les images annotées
        iou_threshold: Seuil IoU pour la suppression des doublons
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers .txt
    
    Returns:
        True si l'inférence a réussi, False sinon
    """
    try:
        # Vérifier que les fichiers existent
        if not os.path.exists(image_path):
            logger.error(f"Image non trouvée: {image_path}")
            return False

        if not os.path.exists(model_path):
            logger.error(f"Modèle non trouvé: {model_path}")
            return False

        # Déterminer la tâche (detect/segment) depuis les métadonnées Ultralytics
        task = _get_ultralytics_task(model_path)
        if task:
            logger.info(f"Tâche du modèle détectée (Ultralytics): {task}")
        else:
            task = "detect"
            logger.info("Tâche du modèle non déterminée, fallback: detect")

        if task == "segment":
            return _run_segmentation_inference(
                image_path=image_path,
                model_path=model_path,
                output_path=output_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                generate_annotated_images=generate_annotated_images,
                jpg_folder_path=jpg_folder_path,
                max_det=max_det,
            )

        # Sinon: détection bbox (comportement actuel SAHI)
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            sahi_available = True
        except ImportError:
            logger.warning("SAHI n'est pas installé. La computer vision (détection bbox) est désactivée.")
            return False

        # Charger le modèle
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cpu"  # Utiliser CPU par défaut
        )
        
        # Régler la limite maximale de détections sur le modèle Ultralytics sous-jacent (si possible)
        try:
            # Ultralytics v8 utilise overrides pour stocker les paramètres
            if hasattr(detection_model.model, "overrides") and isinstance(detection_model.model.overrides, dict):
                detection_model.model.overrides["max_det"] = max_det
            # Certaines versions exposent les args via predictor
            if hasattr(detection_model.model, "predictor") and hasattr(detection_model.model.predictor, "args"):
                try:
                    detection_model.model.predictor.args["max_det"] = max_det
                except Exception:
                    pass
            logger.info(f"Paramètre max_det réglé à {max_det}")
        except Exception as e:
            logger.debug(f"Impossible de régler max_det: {e}")
        
        # Exécuter la prédiction avec découpage
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )
        
        # Vérifier s'il y a des détections
        has_detections = len(result.object_prediction_list) > 0
        
        if has_detections:
            # Sauvegarder l'image avec les détections seulement si des détections existent
            result.export_visuals(export_dir=str(Path(output_path).parent))
            
            # Renommer le fichier de sortie pour correspondre au nom attendu
            exported_path = Path(output_path).parent / "prediction_visual.png"
            output_path_obj = Path(output_path)
            
            if exported_path.exists():
                # Supprimer le fichier de destination s'il existe déjà
                if output_path_obj.exists():
                    output_path_obj.unlink()
                    logger.info(f"Fichier existant supprimé: {output_path}")
                
                # Renommer le fichier
                exported_path.rename(output_path)
            # Sauvegarder les détections au format YOLO pour les shapefiles
            save_detections_to_yolo_format(result, image_path, output_path, jpg_folder_path)
        else:
            logger.info(f"Aucune détection trouvée dans {Path(image_path).name}, image non sauvegardée")
            # Écrire quand même des sorties vides pour marquer l'image comme traitée
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)

        return has_detections
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence YOLO: {e}")
        return False


def _run_segmentation_inference(
    image_path: str,
    model_path: str,
    output_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    generate_annotated_images: bool,
    jpg_folder_path: Optional[str],
    max_det: int,
) -> bool:
    try:
        from ultralytics import YOLO
        from PIL import Image

        model = YOLO(model_path)

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        results = model.predict(
            source=image_path,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
        )

        if not results:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        r0 = results[0]
        masks = getattr(r0, "masks", None)
        if masks is None or getattr(masks, "xy", None) is None:
            logger.warning("Le modèle est marqué segment mais aucun masque n'a été produit")
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        polygons_xy = list(masks.xy)
        if not polygons_xy:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        classes = []
        confidences = []
        try:
            if getattr(r0, "boxes", None) is not None:
                b = r0.boxes
                if getattr(b, "cls", None) is not None:
                    classes = [int(x) for x in b.cls.cpu().numpy().tolist()]
                if getattr(b, "conf", None) is not None:
                    confidences = [float(x) for x in b.conf.cpu().numpy().tolist()]
        except Exception:
            classes = []
            confidences = []

        while len(classes) < len(polygons_xy):
            classes.append(0)
        while len(confidences) < len(polygons_xy):
            confidences.append(None)

        image_name = Path(image_path).stem
        if jpg_folder_path:
            txt_path = Path(jpg_folder_path) / f"{image_name}.txt"
        else:
            txt_path = Path(output_path).with_suffix('.txt')

        detections_data = []
        with open(txt_path, 'w') as f:
            for idx, (cls_id, poly) in enumerate(zip(classes, polygons_xy)):
                if poly is None or len(poly) < 3:
                    continue

                coords = []
                for x, y in poly:
                    x_rel = float(x) / float(img_width)
                    y_rel = float(y) / float(img_height)
                    coords.extend([x_rel, y_rel])

                f.write(f"{int(cls_id)} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

                detections_data.append({
                    "class_id": int(cls_id),
                    "confidence": confidences[idx],
                    "polygon": coords,
                })

        json_path = txt_path.with_suffix('.json')
        json_path.write_text(json.dumps({
            "image_path": image_path,
            "image_dimensions": {"width": img_width, "height": img_height},
            "detections": detections_data,
            "task": "segment",
        }, indent=2))

        logger.info(f"Segmentations sauvegardées au format YOLO-seg: {txt_path}")
        logger.info(f"Données complètes sauvegardées en JSON: {json_path}")

        if generate_annotated_images:
            try:
                im = r0.plot()
                from PIL import Image as PILImage
                out_img = PILImage.fromarray(im[..., ::-1]) if hasattr(im, 'shape') else None
                if out_img is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    out_img.save(output_path)
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder l'image annotée (segmentation): {e}")

        return len(detections_data) > 0

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence segmentation: {e}")
        return False

def get_class_names_from_model(model_path: str) -> Optional[List[str]]:
    """
    Récupère les noms des classes depuis un modèle YOLO
    
    Args:
        model_path: Chemin vers le modèle YOLO
    
    Returns:
        Liste des noms de classes ou None si échec
    """
    try:
        # Essayer d'importer ultralytics pour lire le modèle
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            if hasattr(model, 'names'):
                return list(model.names.values())
        except ImportError:
            logger.warning("ultralytics n'est pas installé")
        
        # Fallback: noms de classes par défaut pour COCO
        default_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        logger.info("Utilisation des noms de classes par défaut (COCO)")
        return default_classes
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des noms de classes: {e}")
        return None

def save_detections_to_yolo_format(result, image_path: str, output_path: str, jpg_folder_path: str = None):
    """
    Sauvegarde les détections SAHI au format YOLO (.txt) pour la création de shapefiles
    
    Args:
        result: Résultat de prédiction SAHI
        image_path: Chemin vers l'image source
        output_path: Chemin vers l'image avec détections
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers .txt
    """
    try:
        from PIL import Image
        
        # Obtenir les dimensions de l'image source
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Créer le fichier .txt dans le dossier jpg si spécifié, sinon dans le même dossier que l'image avec détections
        if jpg_folder_path:
            # Utiliser le nom de fichier de l'image source pour créer le .txt dans le dossier jpg
            image_name = Path(image_path).stem
            txt_path = Path(jpg_folder_path) / f"{image_name}.txt"
        else:
            # Fallback: utiliser le même répertoire que l'image avec détections
            output_path_obj = Path(output_path)
            txt_path = output_path_obj.with_suffix('.txt')
        
        with open(txt_path, 'w') as f:
            for detection in result.object_prediction_list:
                bbox = detection.bbox
                
                # Convertir les coordonnées absolues en coordonnées relatives YOLO
                x_center_rel = (bbox.minx + bbox.maxx) / 2 / img_width
                y_center_rel = (bbox.miny + bbox.maxy) / 2 / img_height
                width_rel = (bbox.maxx - bbox.minx) / img_width
                height_rel = (bbox.maxy - bbox.miny) / img_height
                
                # Format YOLO standard: class_id x_center y_center width height
                f.write(f"{detection.category.id} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}\n")
        
        logger.info(f"Détections sauvegardées au format YOLO: {txt_path}")
        
        # Sauvegarder également les données complètes avec confiance en JSON
        json_path = txt_path.with_suffix('.json')
        detections_data = []
        
        for detection in result.object_prediction_list:
            bbox = detection.bbox
            
            # Convertir les coordonnées absolues en coordonnées relatives YOLO
            x_center_rel = (bbox.minx + bbox.maxx) / 2 / img_width
            y_center_rel = (bbox.miny + bbox.maxy) / 2 / img_height
            width_rel = (bbox.maxx - bbox.minx) / img_width
            height_rel = (bbox.maxy - bbox.miny) / img_height
            
            detections_data.append({
                "class_id": detection.category.id,
                "x_center": x_center_rel,
                "y_center": y_center_rel,
                "width": width_rel,
                "height": height_rel,
                "confidence": detection.score.value,
                "bbox_absolute": {
                    "minx": bbox.minx,
                    "miny": bbox.miny,
                    "maxx": bbox.maxx,
                    "maxy": bbox.maxy
                }
            })
        
        # Sauvegarder en JSON
        import json
        with open(json_path, 'w') as json_file:
            json.dump({
                "image_path": image_path,
                "image_dimensions": {"width": img_width, "height": img_height},
                "detections": detections_data
            }, json_file, indent=2)
        
        logger.info(f"Données complètes sauvegardées en JSON: {json_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des détections YOLO: {e}")

def create_shapefile_from_detections(detection_results: Dict, output_shapefile: str, 
                                   class_names: Optional[List[str]] = None) -> bool:
    """
    Crée un shapefile à partir des résultats de détection
    
    Args:
        detection_results: Résultats de détection SAHI
        output_shapefile: Chemin de sortie pour le shapefile
        class_names: Noms des classes (optionnel)
    
    Returns:
        True si le shapefile a été créé avec succès, False sinon
    """
    try:
        # Vérifier si les dépendances géospatiales sont disponibles
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            import pandas as pd
        except ImportError:
            logger.warning("geopandas ou shapely non disponibles. Shapefile non créé.")
            return False
        
        # Extraire les détections
        detections = []
        if hasattr(detection_results, 'object_prediction_list'):
            for detection in detection_results.object_prediction_list:
                bbox = detection.bbox
                
                # Créer un polygone à partir de la bounding box
                polygon = Polygon([
                    (bbox.minx, bbox.miny),
                    (bbox.maxx, bbox.miny),
                    (bbox.maxx, bbox.maxy),
                    (bbox.minx, bbox.maxy)
                ])
                
                # Déterminer le nom de la classe
                class_name = "unknown"
                if class_names and detection.category.id < len(class_names):
                    class_name = class_names[detection.category.id]
                
                detections.append({
                    'geometry': polygon,
                    'class_id': detection.category.id,
                    'class_name': class_name,
                    'confidence': detection.score.value
                })
        
        if not detections:
            logger.info("Aucune détection trouvée pour créer le shapefile")
            return False
        
        # Créer le GeoDataFrame
        gdf = gpd.GeoDataFrame(detections)
        
        # Sauvegarder le shapefile
        try:
            # DIAGNOSTIC DÉTAILLÉ POUR PYINSTALLER
            logger.info(f"🔍 DIAGNOSTIC: Tentative de sauvegarde shapefile...")
            logger.info(f"   GeoDataFrame shape: {gdf.shape}")
            logger.info(f"   Colonnes: {list(gdf.columns)}")
            logger.info(f"   CRS: {getattr(gdf, 'crs', 'Non défini')}")
            logger.info(f"   Chemin de sortie: {output_shapefile}")
            
            # Test des modules critiques avant sauvegarde
            try:
                import pandas._libs.window.aggregations
                logger.info("   ✅ pandas._libs.window.aggregations disponible")
            except Exception as mod_e:
                logger.error(f"   ❌ pandas._libs.window.aggregations: {mod_e}")
            
            # Tentative de sauvegarde
            gdf.to_file(output_shapefile)
            logger.info(f"✅ Shapefile créé avec succès: {output_shapefile}")
            
        except Exception as save_e:
            logger.error(f"❌ ERREUR SAUVEGARDE SHAPEFILE: {save_e}")
            logger.error(f"   Type d'erreur: {type(save_e).__name__}")
            
            # Traceback détaillé pour PyInstaller
            import traceback
            tb_lines = traceback.format_exc().split('\n')
            for i, line in enumerate(tb_lines):
                if line.strip():
                    logger.error(f"   TB[{i}]: {line}")
            
            # Essayer des alternatives de sauvegarde
            try:
                logger.info("🔄 Tentative de sauvegarde alternative...")
                # Essayer sans CRS
                gdf_no_crs = gdf.copy()
                gdf_no_crs.crs = None
                gdf_no_crs.to_file(output_shapefile)
                logger.info("✅ Sauvegarde alternative réussie (sans CRS)")
            except Exception as alt_e:
                logger.error(f"❌ Sauvegarde alternative échouée: {alt_e}")
                raise save_e
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du shapefile: {e}")
        return False

def setup_computer_vision_environment():
    """
    Configure l'environnement pour la computer vision
    Vérifie les dépendances et affiche des avertissements si nécessaire
    """
    missing_deps = []
    
    try:
        import sahi
    except ImportError:
        missing_deps.append("sahi")
    
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import geopandas
    except ImportError:
        missing_deps.append("geopandas")
    
    try:
        import shapely
    except ImportError:
        missing_deps.append("shapely")
    
    if missing_deps:
        logger.warning(f"Dépendances manquantes pour la computer vision: {', '.join(missing_deps)}")
        logger.warning("Installez-les avec: pip install sahi ultralytics geopandas shapely")
        return False
    
    logger.info("Environnement computer vision configuré avec succès")
    return True
