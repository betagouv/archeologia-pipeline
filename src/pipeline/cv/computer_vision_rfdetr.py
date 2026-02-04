"""
Module de computer vision RF-DETR pour la détection d'objets sur les images RVT.
Ce module est dédié à l'inférence RF-DETR et est utilisé par cv_runner_rfdetr.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .class_utils import get_num_classes_from_model
from .cv_output import (
    get_detection_output_path,
    save_empty_outputs,
    save_detections_to_files,
    save_annotated_image,
)

logger = logging.getLogger(__name__)


def _load_rfdetr_model(model_path: str):
    """
    Charge un modèle RF-DETR en détectant automatiquement la variante et la résolution.
    
    Returns:
        Tuple (model, resolution) ou lève une exception si le chargement échoue.
    """
    import torch
    
    try:
        from rfdetr import RFDETRBase
    except ImportError as e:
        raise ImportError(f"Le package rfdetr n'est pas installé. Erreur: {e}")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Détecter la résolution à partir de la taille des position_embeddings
    resolution = 560  # défaut
    try:
        state_dict = checkpoint.get('model', checkpoint)
        for key in state_dict.keys():
            if 'position_embeddings' in key and 'backbone' in key:
                pos_embed_shape = state_dict[key].shape
                num_positions = pos_embed_shape[1]
                num_patches = num_positions - 1
                patch_size = 16
                grid_size = int(num_patches ** 0.5)
                resolution = grid_size * patch_size
                logger.debug(f"RF-DETR: détecté resolution={resolution} (grid={grid_size}x{grid_size})")
                break
    except Exception as e:
        logger.debug(f"RF-DETR: impossible de détecter la résolution, utilisation de 560: {e}")
    
    # Essayer différentes variantes RF-DETR
    model_classes = []
    try:
        from rfdetr import RFDETRMedium
        model_classes.append(('RFDETRMedium', RFDETRMedium))
    except ImportError:
        pass
    
    model_classes.append(('RFDETRBase', RFDETRBase))
    
    for name, cls in [('RFDETRLarge', 'RFDETRLarge'), ('RFDETRSmall', 'RFDETRSmall'), ('RFDETRNano', 'RFDETRNano')]:
        try:
            from rfdetr import __dict__ as rfdetr_dict
            if name in rfdetr_dict:
                model_classes.append((name, rfdetr_dict[name]))
        except Exception:
            pass
    
    last_error = None
    for model_name, model_cls in model_classes:
        try:
            logger.debug(f"RF-DETR: essai avec {model_name} resolution={resolution}")
            model = model_cls(pretrain_weights=model_path, resolution=resolution)
            logger.info(f"RF-DETR: modèle {model_name} chargé avec succès (resolution={resolution})")
            return model, resolution
        except Exception as e:
            last_error = e
            logger.debug(f"RF-DETR: échec {model_name}: {e}")
            continue
    
    raise last_error or RuntimeError("Impossible de charger le modèle RF-DETR")


def _apply_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Applique Non-Maximum Suppression sur les détections."""
    if not detections:
        return []
    
    try:
        import numpy as np
        
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        class_ids = np.array([d["class_id"] for d in detections])
        
        # NMS par classe
        keep_indices = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # Trier par score décroissant
            order = cls_scores.argsort()[::-1]
            cls_boxes = cls_boxes[order]
            cls_indices = cls_indices[order]
            
            keep = []
            while len(cls_boxes) > 0:
                keep.append(cls_indices[0])
                if len(cls_boxes) == 1:
                    break
                
                # Calculer IoU avec la première box
                ious = _compute_iou(cls_boxes[0], cls_boxes[1:])
                
                # Garder les boxes avec IoU < threshold
                mask = ious < iou_threshold
                cls_boxes = cls_boxes[1:][mask]
                cls_indices = cls_indices[1:][mask]
            
            keep_indices.extend(keep)
        
        return [detections[i] for i in keep_indices]
    
    except Exception as e:
        logger.warning(f"Erreur NMS, retour des détections brutes: {e}")
        return detections


def _compute_iou(box1, boxes):
    """Calcule l'IoU entre une box et un ensemble de boxes."""
    import numpy as np
    
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areas2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = area1 + areas2 - intersection
    
    return intersection / np.maximum(union, 1e-6)


def run_rfdetr_inference(
    *,
    image_path: str,
    model_path: str,
    output_path: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.2,
    generate_annotated_images: bool = False,
    annotated_output_dir: Optional[str] = None,
    jpg_folder_path: Optional[str] = None,
    max_det: int = 1000,
) -> bool:
    """
    Exécute l'inférence avec un modèle RF-DETR en utilisant SAHI pour le slicing.
    
    Le slicing est crucial pour les grandes images LiDAR car RF-DETR a une résolution
    d'entrée fixe (typiquement 560-576px). Sans slicing, les petits objets seraient
    perdus lors du redimensionnement.
    """
    try:
        from PIL import Image
        import numpy as np

        with Image.open(image_path) as img:
            img_width, img_height = img.size
            pil_image = img.convert("RGB")

        # Charger le modèle RF-DETR
        model, model_resolution = _load_rfdetr_model(model_path)
        
        # Utiliser SAHI pour le slicing si disponible
        all_detections = []
        use_sahi = False
        
        try:
            from sahi.slicing import slice_image
            print(f"[RF-DETR DEBUG] SAHI import OK, slicing {slice_width}x{slice_height}, overlap={overlap_ratio}")
            
            # Découper l'image en tuiles
            slice_result = slice_image(
                image=pil_image,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
            )
            
            # Vérifier la structure du résultat
            if hasattr(slice_result, 'sliced_image_list'):
                slices = slice_result.sliced_image_list
            elif hasattr(slice_result, 'images'):
                slices = slice_result.images
            else:
                slices = slice_result if isinstance(slice_result, list) else []
            
            print(f"[RF-DETR DEBUG] {len(slices)} tuiles à traiter")
            use_sahi = len(slices) > 0
            
            # Inférence sur chaque tuile
            for idx, slice_data in enumerate(slices):
                # Extraire l'image de la tuile
                if hasattr(slice_data, 'image'):
                    slice_img = slice_data.image
                else:
                    slice_img = slice_data
                
                # Convertir en PIL si nécessaire
                if not isinstance(slice_img, Image.Image):
                    slice_img = Image.fromarray(np.array(slice_img))
                
                # Prédiction sur la tuile
                detections = model.predict(slice_img, threshold=confidence_threshold)
                
                if detections is not None and len(detections) > 0:
                    # Récupérer l'offset de la tuile
                    if hasattr(slice_data, 'starting_pixel'):
                        offset_x, offset_y = slice_data.starting_pixel
                    elif hasattr(slice_data, 'starting_pixels'):
                        offset_x, offset_y = slice_data.starting_pixels
                    else:
                        offset_x, offset_y = 0, 0
                    
                    # Ajuster les coordonnées des détections
                    for i in range(len(detections.xyxy)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                        
                        # Ajouter l'offset pour obtenir les coordonnées dans l'image complète
                        all_detections.append({
                            "bbox": [float(x1 + offset_x), float(y1 + offset_y), 
                                    float(x2 + offset_x), float(y2 + offset_y)],
                            "class_id": class_id,
                            "confidence": conf,
                        })
            
            # Appliquer NMS pour fusionner les détections qui se chevauchent
            if all_detections:
                all_detections = _apply_nms(all_detections, iou_threshold)
                print(f"[RF-DETR DEBUG] {len(all_detections)} détections après NMS")
            
        except Exception as e:
            print(f"[RF-DETR DEBUG] SAHI slicing échoué: {type(e).__name__}: {e}")
            use_sahi = False
        
        # Fallback: inférence directe sans slicing
        if not use_sahi:
            print("[RF-DETR DEBUG] Inférence directe sur l'image entière (sans SAHI)")
            detections = model.predict(pil_image, threshold=confidence_threshold)
            
            if detections is not None and len(detections) > 0:
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    all_detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class_id": class_id,
                        "confidence": conf,
                    })

        # Sauvegarder les résultats
        if not all_detections:
            logger.info(f"Aucune détection RF-DETR trouvée dans {Path(image_path).name}")
            save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        # Limiter au max_det
        if len(all_detections) > max_det:
            all_detections = sorted(all_detections, key=lambda x: x["confidence"], reverse=True)[:max_det]

        # RF-DETR utilise des class IDs 1-indexés → convertir en 0-indexé
        num_classes = get_num_classes_from_model(model_path)
        for det in all_detections:
            raw_id = det["class_id"]
            # Décaler de -1 (RF-DETR est 1-indexé)
            normalized_id = max(0, raw_id - 1)
            if num_classes:
                normalized_id = min(normalized_id, num_classes - 1)
            det["class_id"] = normalized_id
        
        logger.info(f"RF-DETR: class IDs normalisés (1-indexé → 0-indexé)")

        # Utiliser le module commun pour sauvegarder les fichiers
        txt_path, json_path = save_detections_to_files(
            image_path=image_path,
            output_path=output_path,
            detections=all_detections,
            img_width=img_width,
            img_height=img_height,
            jpg_folder_path=jpg_folder_path,
            task="detect",
            model_type="rfdetr",
        )

        logger.info(f"RF-DETR: {len(all_detections)} détections sauvegardées pour {Path(image_path).name}")

        # Générer l'image annotée si demandé
        if generate_annotated_images and all_detections:
            save_annotated_image(pil_image, all_detections, output_path)

        return len(all_detections) > 0

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence RF-DETR: {e}")
        import traceback
        traceback.print_exc()
        return False
