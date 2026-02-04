"""
Module de computer vision ONNX pour la détection d'objets sur les images RVT.
Ce module utilise ONNX Runtime pour l'inférence, permettant un runner léger et unifié.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cv_output import (
    get_detection_output_path,
    save_empty_outputs,
    save_detections_to_files,
    save_annotated_image,
)
from .sahi_lite import (
    slice_image as sahi_lite_slice_image,
    Detection,
    merge_sliced_detections,
)

logger = logging.getLogger(__name__)


def _load_onnx_model(model_path: str):
    """
    Charge un modèle ONNX avec onnxruntime.
    
    Returns:
        Tuple (session, input_name, input_shape, model_meta)
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"onnxruntime n'est pas installé: {e}")
    
    # Configurer les providers (GPU si disponible, sinon CPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers = ort.get_available_providers()
    providers = [p for p in providers if p in available_providers]
    
    if not providers:
        providers = ['CPUExecutionProvider']
    
    logger.info(f"ONNX: providers disponibles: {available_providers}")
    logger.info(f"ONNX: utilisation de: {providers}")
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Récupérer les infos du modèle
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # [batch, channels, height, width]
    
    # Charger les métadonnées si disponibles
    model_meta = {}
    meta_path = Path(model_path).with_suffix('.json')
    if meta_path.exists():
        try:
            model_meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    
    return session, input_name, input_shape, model_meta


def _preprocess_image(pil_image, target_size: Tuple[int, int], model_type: str = "yolo") -> np.ndarray:
    """
    Prétraite une image PIL pour l'inférence ONNX.
    
    Args:
        pil_image: Image PIL
        target_size: (width, height) cible
        model_type: "yolo" ou "rfdetr" - détermine la normalisation
    
    Returns:
        Tensor numpy [1, 3, H, W] normalisé
    """
    from PIL import Image
    
    # Redimensionner
    img_resized = pil_image.resize(target_size, Image.BILINEAR)
    
    # Convertir en numpy
    img_array = np.array(img_resized).astype(np.float32)
    
    # Normaliser [0, 255] -> [0, 1]
    img_array = img_array / 255.0
    
    # RF-DETR utilise la normalisation ImageNet
    if model_type == "rfdetr":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
    
    # HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Ajouter dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def _postprocess_yolo(
    outputs: List[np.ndarray],
    img_width: int,
    img_height: int,
    model_width: int,
    model_height: int,
    confidence_threshold: float,
) -> List[Dict]:
    """
    Post-traite les sorties YOLO ONNX.
    
    YOLO ONNX output: [1, num_detections, 4+num_classes] ou [1, 4+num_classes, num_detections]
    """
    detections = []
    
    output = outputs[0]  # Premier output
    
    # Gérer les différents formats de sortie YOLO
    if len(output.shape) == 3:
        output = output[0]  # Enlever batch dimension
    
    # Vérifier l'orientation (transposer si nécessaire)
    if output.shape[0] < output.shape[1] and output.shape[0] < 10:
        output = output.T  # [4+nc, num] -> [num, 4+nc]
    
    for row in output:
        if len(row) < 5:
            continue
        
        # Format YOLO: x_center, y_center, width, height, class_scores...
        x_center, y_center, w, h = row[:4]
        class_scores = row[4:]
        
        # Trouver la meilleure classe
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        
        if confidence < confidence_threshold:
            continue
        
        # Convertir vers coordonnées absolues
        scale_x = img_width / model_width
        scale_y = img_height / model_height
        
        x1 = (x_center - w / 2) * scale_x
        y1 = (y_center - h / 2) * scale_y
        x2 = (x_center + w / 2) * scale_x
        y2 = (y_center + h / 2) * scale_y
        
        # Clamp aux limites de l'image
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": class_id,
            "confidence": confidence,
        })
    
    return detections


def _postprocess_rfdetr(
    outputs: List[np.ndarray],
    img_width: int,
    img_height: int,
    model_width: int,
    model_height: int,
    confidence_threshold: float,
) -> List[Dict]:
    """
    Post-traite les sorties RF-DETR ONNX.
    
    RF-DETR outputs: 
      - pred_boxes [1, N, 4] en format [cx, cy, w, h] normalisé (0-1)
      - pred_logits [1, N, num_classes] (logits, pas de softmax appliqué)
    """
    detections = []
    
    # Debug: afficher les shapes des outputs
    logger.info(f"RF-DETR outputs: {len(outputs)} tenseurs")
    for i, out in enumerate(outputs):
        logger.info(f"  Output {i}: shape={out.shape}, dtype={out.dtype}, min={out.min():.4f}, max={out.max():.4f}")
    
    if len(outputs) == 2:
        # Format standard RF-DETR: [boxes, logits]
        boxes = outputs[0][0]   # [N, 4] - format cxcywh normalisé
        logits = outputs[1][0]  # [N, num_classes] - logits bruts
        
        logger.info(f"RF-DETR: boxes shape={boxes.shape}, logits shape={logits.shape}")
        
        # Appliquer sigmoid pour convertir logits en probabilités
        scores = 1 / (1 + np.exp(-logits))  # sigmoid
        
        # Trouver la meilleure classe et son score pour chaque détection
        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        
        # Debug: afficher les stats des scores
        logger.info(f"RF-DETR: max_scores min={max_scores.min():.4f}, max={max_scores.max():.4f}, mean={max_scores.mean():.4f}")
        logger.info(f"RF-DETR: {(max_scores >= confidence_threshold).sum()} détections au-dessus du seuil {confidence_threshold}")
        
    elif len(outputs) >= 3:
        boxes = outputs[0][0]   # [N, 4]
        scores_raw = outputs[1][0]  # [N] ou [N, num_classes]
        labels = outputs[2][0]  # [N]
        
        if scores_raw.ndim == 1:
            max_scores = scores_raw
            class_ids = labels.astype(int)
        else:
            max_scores = scores_raw.max(axis=1)
            class_ids = scores_raw.argmax(axis=1)
    else:
        return detections
    
    # Vérifier si les boxes sont normalisées (0-1)
    boxes_normalized = boxes.max() <= 1.0
    
    for i in range(len(max_scores)):
        confidence = float(max_scores[i])
        if confidence < confidence_threshold:
            continue
        
        # Format RF-DETR: [cx, cy, w, h] normalisé
        cx, cy, w, h = boxes[i]
        
        if boxes_normalized:
            # Dénormaliser vers coordonnées du modèle
            cx = cx * model_width
            cy = cy * model_height
            w = w * model_width
            h = h * model_height
        
        # Convertir cxcywh -> xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Rescale vers la taille de l'image originale
        scale_x = img_width / model_width
        scale_y = img_height / model_height
        
        x1 = float(x1) * scale_x
        y1 = float(y1) * scale_y
        x2 = float(x2) * scale_x
        y2 = float(y2) * scale_y
        
        # Clamp aux limites de l'image
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Ignorer les boxes invalides
        if x2 <= x1 or y2 <= y1:
            continue
        
        # RF-DETR: les class IDs sont 1-indexés (0 = background), donc on soustrait 1
        # pour obtenir des indices 0-indexés compatibles avec classes.txt
        class_id = int(class_ids[i]) - 1
        if class_id < 0:
            continue  # Ignorer les détections de classe "background"
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "confidence": confidence,
        })
    
    return detections


def _apply_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Applique Non-Maximum Suppression sur les détections."""
    if not detections:
        return []
    
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["confidence"] for d in detections])
    class_ids = np.array([d["class_id"] for d in detections])
    
    keep_indices = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        order = cls_scores.argsort()[::-1]
        cls_boxes = cls_boxes[order]
        cls_indices = cls_indices[order]
        
        keep = []
        while len(cls_boxes) > 0:
            keep.append(cls_indices[0])
            if len(cls_boxes) == 1:
                break
            
            # Calculer IoU
            x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
            y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
            x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
            y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area1 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (cls_boxes[0, 3] - cls_boxes[0, 1])
            areas2 = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
            union = area1 + areas2 - intersection
            ious = intersection / np.maximum(union, 1e-6)
            
            mask = ious < iou_threshold
            cls_boxes = cls_boxes[1:][mask]
            cls_indices = cls_indices[1:][mask]
        
        keep_indices.extend(keep)
    
    return [detections[i] for i in keep_indices]


def run_onnx_inference(
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
    return_count: bool = False,
    class_names: List[str] = None,
    class_colors: List[int] = None,
):
    """
    Exécute l'inférence avec un modèle ONNX en utilisant SAHI pour le slicing.
    
    Args:
        image_path: Chemin vers l'image
        model_path: Chemin vers le modèle ONNX
        output_path: Chemin de sortie pour l'image annotée
        confidence_threshold: Seuil de confiance
        iou_threshold: Seuil IoU pour NMS
        slice_height: Hauteur des tuiles SAHI
        slice_width: Largeur des tuiles SAHI
        overlap_ratio: Ratio de chevauchement SAHI
        generate_annotated_images: Générer les images annotées
        annotated_output_dir: Dossier de sortie pour les images annotées
        jpg_folder_path: Dossier pour les fichiers .txt/.json
        max_det: Nombre maximum de détections
        return_count: Si True, retourne (bool, int) avec le nombre de détections
    
    Returns:
        Si return_count=False: True si des détections ont été trouvées
        Si return_count=True: (True/False, nombre_detections)
    """
    try:
        from PIL import Image
        
        # Charger l'image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            pil_image = img.convert("RGB")
        
        # Charger le modèle ONNX
        session, input_name, input_shape, model_meta = _load_onnx_model(model_path)
        
        # Déterminer la taille du modèle
        if len(input_shape) >= 4:
            model_height = input_shape[2] if isinstance(input_shape[2], int) else 640
            model_width = input_shape[3] if isinstance(input_shape[3], int) else 640
        else:
            model_height = model_width = 640
        
        # Détecter le type de modèle
        model_type = model_meta.get("model_type", "yolo")
        logger.info(f"ONNX: modèle {model_type}, taille {model_width}x{model_height}")
        
        # Utiliser sahi_lite pour le slicing (numpy-only, pas de dépendance torch)
        img_array = np.array(pil_image)
        sliced_images, orig_height, orig_width = sahi_lite_slice_image(
            image=img_array,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )
        
        logger.info(f"ONNX: {len(sliced_images)} tuiles à traiter")
        
        all_slice_detections = []
        slice_starting_pixels = []
        
        total_raw_detections = 0
        for idx, sliced_img in enumerate(sliced_images):
            slice_pil = Image.fromarray(sliced_img.image)
            slice_w, slice_h = slice_pil.size
            
            # Prétraiter (avec normalisation ImageNet pour RF-DETR)
            input_tensor = _preprocess_image(slice_pil, (model_width, model_height), model_type)
            
            # Inférence
            outputs = session.run(None, {input_name: input_tensor})
            
            # Post-traiter selon le type de modèle
            if model_type == "rfdetr":
                dets = _postprocess_rfdetr(outputs, slice_w, slice_h, model_width, model_height, confidence_threshold)
            else:
                dets = _postprocess_yolo(outputs, slice_w, slice_h, model_width, model_height, confidence_threshold)
            
            total_raw_detections += len(dets)
            if len(dets) > 0:
                logger.info(f"ONNX: slice {idx+1}/{len(sliced_images)} -> {len(dets)} détections")
            
            # Convertir en objets Detection
            slice_detections = [
                Detection(
                    bbox=d["bbox"],
                    score=d["confidence"],
                    class_id=d["class_id"],
                )
                for d in dets
            ]
            
            all_slice_detections.append(slice_detections)
            slice_starting_pixels.append(sliced_img.starting_pixel)
        
        logger.info(f"ONNX: {total_raw_detections} détections brutes avant fusion")
        
        # Fusionner les détections de toutes les slices avec Greedy NMM
        merged_detections = merge_sliced_detections(
            all_detections=all_slice_detections,
            slice_starting_pixels=slice_starting_pixels,
            original_height=orig_height,
            original_width=orig_width,
            match_threshold=iou_threshold,
            match_metric="IOS",
            class_agnostic=False,
        )
        
        # Convertir en format dict
        all_detections = [
            {
                "bbox": det.bbox,
                "class_id": det.class_id,
                "confidence": det.score,
            }
            for det in merged_detections
        ]
        
        logger.info(f"ONNX: {len(all_detections)} détections après fusion")
        
        # Sauvegarder les résultats
        if not all_detections:
            logger.info(f"ONNX: aucune détection dans {Path(image_path).name}")
            save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return (False, 0) if return_count else False
        
        # Limiter au max_det
        if len(all_detections) > max_det:
            all_detections = sorted(all_detections, key=lambda x: x["confidence"], reverse=True)[:max_det]
        
        num_detections = len(all_detections)
        
        # Sauvegarder les fichiers
        save_detections_to_files(
            image_path=image_path,
            output_path=output_path,
            detections=all_detections,
            img_width=img_width,
            img_height=img_height,
            jpg_folder_path=jpg_folder_path,
            task="detect",
            model_type="onnx",
        )
        
        logger.info(f"ONNX: {num_detections} détections sauvegardées pour {Path(image_path).name}")
        
        # Générer l'image annotée
        if generate_annotated_images and all_detections:
            save_annotated_image(pil_image, all_detections, output_path, class_names=class_names, class_colors=class_colors)
        
        return (True, num_detections) if return_count else True
    
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence ONNX: {e}")
        import traceback
        traceback.print_exc()
        return (False, 0) if return_count else False
