"""
Module de computer vision ONNX pour la détection d'objets et la segmentation sémantique sur les images RVT.
Ce module utilise ONNX Runtime pour l'inférence, permettant un runner léger et unifié.
Supporte: YOLO (détection), RF-DETR (détection), SegFormer/SMP (segmentation sémantique).
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
        model_type: "yolo", "rfdetr", "segformer" ou "smp" - détermine la normalisation
    
    Returns:
        Tensor numpy [1, 3, H, W] normalisé
    """
    import cv2
    
    # Redimensionner avec cv2 (INTER_LINEAR) pour correspondre à Albumentations
    # utilisé pendant l'entraînement (A.Resize utilise cv2.INTER_LINEAR par défaut)
    img_array = np.array(pil_image)
    img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convertir en float32
    img_array = img_array.astype(np.float32)
    
    # Normaliser [0, 255] -> [0, 1]
    img_array = img_array / 255.0
    
    # RF-DETR, SegFormer et SMP utilisent la normalisation ImageNet
    if model_type in ("rfdetr", "segformer", "smp"):
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
    class_offset: int = 1,
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
        
        # Ignorer les boxes invalides ou trop petites (< 10 pixels)
        box_width = x2 - x1
        box_height = y2 - y1
        min_box_size = 10  # pixels
        if box_width < min_box_size or box_height < min_box_size:
            logger.debug(f"RF-DETR: box trop petite ignorée - w={box_width:.1f}, h={box_height:.1f}, class={int(class_ids[i])}")
            continue
        
        # RF-DETR: appliquer le décalage de classe (class_offset)
        # - class_offset=1 (défaut): modèles avec background à l'index 0, on soustrait 1
        # - class_offset=0: modèles sans background, on garde les indices tels quels
        class_id = int(class_ids[i]) - class_offset
        if class_id < 0:
            continue  # Ignorer les détections de classe "background"
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "confidence": confidence,
        })
    
    return detections


def _postprocess_rfdetr_seg(
    outputs: List[np.ndarray],
    img_width: int,
    img_height: int,
    model_width: int,
    model_height: int,
    confidence_threshold: float,
    class_offset: int = 1,
) -> List[Dict]:
    """
    Post-traite les sorties RF-DETR Seg ONNX (instance segmentation).

    RF-DETR Seg outputs:
      - outputs[0]: pred_boxes  [1, N, 4]  cxcywh normalisé
      - outputs[1]: pred_logits [1, N, num_classes]
      - outputs[2]: pred_masks  [1, N, Mh, Mw]  masques d'instances (logits)
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV non disponible, segmentation impossible — fallback bbox")
        return _postprocess_rfdetr(outputs[:2], img_width, img_height, model_width, model_height, confidence_threshold, class_offset)

    detections = []

    if len(outputs) < 3:
        return _postprocess_rfdetr(outputs, img_width, img_height, model_width, model_height, confidence_threshold, class_offset)

    boxes_out  = outputs[0][0]   # [N, 4]
    logits_out = outputs[1][0]   # [N, num_classes]
    masks_out  = outputs[2][0]   # [N, Mh, Mw]

    logger.info(f"RF-DETR Seg: boxes={boxes_out.shape}, logits={logits_out.shape}, masks={masks_out.shape}")

    scores = 1.0 / (1.0 + np.exp(-logits_out))  # sigmoid
    max_scores = scores.max(axis=1)
    class_ids  = scores.argmax(axis=1)

    scale_x = img_width  / model_width
    scale_y = img_height / model_height

    for i in range(len(max_scores)):
        confidence = float(max_scores[i])
        if confidence < confidence_threshold:
            continue

        class_id = int(class_ids[i]) - class_offset
        if class_id < 0:
            continue

        # ----- Masque d'instance → polygone -----
        mask_logit = masks_out[i]                       # [Mh, Mw]
        mask_prob  = 1.0 / (1.0 + np.exp(-mask_logit)) # sigmoid

        # Upscale les probabilités (float) directement vers la taille image finale
        # INTER_LINEAR sur les probas avant seuillage → contours lisses (évite la pixelisation)
        mask_prob_full = cv2.resize(mask_prob.astype(np.float32), (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        binary_full = (mask_prob_full >= 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback bbox si pas de contour
            cx, cy, w, h = boxes_out[i]
            if boxes_out.max() <= 1.0:
                cx, cy, w, h = cx * model_width, cy * model_height, w * model_width, h * model_height
            x1 = max(0.0, (cx - w / 2) * scale_x)
            y1 = max(0.0, (cy - h / 2) * scale_y)
            x2 = min(img_width,  (cx + w / 2) * scale_x)
            y2 = min(img_height, (cy + h / 2) * scale_y)
            detections.append({"bbox": [x1, y1, x2, y2], "class_id": class_id, "confidence": confidence})
            continue

        # Garder le plus grand contour par instance
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10 or len(contour) < 3:
            continue

        polygon = []
        for pt in contour:
            polygon.extend([float(pt[0][0]) / img_width, float(pt[0][1]) / img_height])

        x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
        detections.append({
            "class_id":  class_id,
            "confidence": confidence,
            "polygon":   polygon,
            "bbox":      [float(x_b), float(y_b), float(x_b + w_b), float(y_b + h_b)],
            "area":      float(area),
        })

    logger.info(f"RF-DETR Seg: {len(detections)} instances extraites")
    return detections


def _run_rfdetr_seg_with_sahi(
    pil_image,
    session,
    input_name: str,
    model_width: int,
    model_height: int,
    slice_width: int,
    slice_height: int,
    overlap_ratio: float,
    confidence_threshold: float,
    class_offset: int = 1,
) -> List[Dict]:
    """
    Exécute RF-DETR Seg avec SAHI slicing en accumulant les masques de probabilité
    par classe dans l'espace image global, puis extrait les polygones une seule fois.

    Inspiré de _run_segformer_with_sahi : évite les doublons et gère correctement
    les formes linéaires qui débordent sur plusieurs slices.

    Pour chaque classe, on accumule max(prob) dans le masque global — une zone couverte
    par plusieurs slices conserve la proba la plus haute observée, ce qui est correct
    pour l'instance segmentation (évite la moyenne qui diluerait les détections).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV non disponible, _run_rfdetr_seg_with_sahi impossible")
        return []

    from PIL import Image as _PILImage

    img_width, img_height = pil_image.size
    img_array = np.array(pil_image)

    sliced_images, orig_height, orig_width = sahi_lite_slice_image(
        image=img_array,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )
    logger.info(f"RF-DETR Seg SAHI: {len(sliced_images)} tuiles")

    # global_class_probs[c] = masque de probabilité max pour la classe c, espace image global
    global_class_probs = None  # [num_classes, H, W], initialisé à la 1ère tuile
    num_classes = None
    # Conserver la confiance de détection pour chaque classe (max sur les instances détectées)
    global_class_conf = None   # [num_classes, H, W]

    for idx, sliced_img in enumerate(sliced_images):
        slice_pil = _PILImage.fromarray(sliced_img.image)
        slice_w, slice_h = slice_pil.size
        start_x, start_y = sliced_img.starting_pixel

        input_tensor = _preprocess_image(slice_pil, (model_width, model_height), "rfdetr")
        outputs = session.run(None, {input_name: input_tensor})

        if len(outputs) < 3:
            continue

        boxes_out  = outputs[0][0]   # [N, 4]
        logits_out = outputs[1][0]   # [N, num_classes]
        masks_out  = outputs[2][0]   # [N, Mh, Mw]

        scores    = 1.0 / (1.0 + np.exp(-logits_out))  # sigmoid [N, C]
        max_scores = scores.max(axis=1)                  # [N]
        class_ids  = scores.argmax(axis=1)               # [N]

        n_cls = logits_out.shape[1]
        # Nombre de classes réelles (sans background offsetté)
        n_real = max(1, n_cls - class_offset)

        if global_class_probs is None:
            num_classes = n_real
            global_class_probs = np.zeros((n_real, orig_height, orig_width), dtype=np.float32)
            global_class_conf  = np.zeros((n_real, orig_height, orig_width), dtype=np.float32)

        end_x = min(start_x + slice_w, orig_width)
        end_y = min(start_y + slice_h, orig_height)
        actual_w = end_x - start_x
        actual_h = end_y - start_y

        for i in range(len(max_scores)):
            confidence = float(max_scores[i])
            if confidence < confidence_threshold:
                continue

            class_id = int(class_ids[i]) - class_offset
            if class_id < 0 or class_id >= n_real:
                continue

            mask_logit = masks_out[i]                        # [Mh, Mw]
            mask_prob  = 1.0 / (1.0 + np.exp(-mask_logit))  # sigmoid, float [Mh, Mw]

            # Upscale vers la taille de la slice (float → lisse)
            mask_prob_slice = cv2.resize(
                mask_prob.astype(np.float32),
                (slice_w, slice_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # Accumulation par max dans l'espace global
            existing = global_class_probs[class_id, start_y:end_y, start_x:end_x]
            new_vals  = mask_prob_slice[:actual_h, :actual_w]
            global_class_probs[class_id, start_y:end_y, start_x:end_x] = np.maximum(existing, new_vals)

            # Confiance associée : propagée uniquement où le masque est actif
            conf_slice = np.where(new_vals >= 0.5, confidence, 0.0).astype(np.float32)
            existing_conf = global_class_conf[class_id, start_y:end_y, start_x:end_x]
            global_class_conf[class_id, start_y:end_y, start_x:end_x] = np.maximum(existing_conf, conf_slice)

        if (idx + 1) % 10 == 0:
            logger.info(f"RF-DETR Seg SAHI: {idx + 1}/{len(sliced_images)} tuiles traitées")

    if global_class_probs is None:
        return []

    # Extraire les polygones depuis les masques globaux fusionnés
    detections = []
    for class_id in range(num_classes):
        prob_map = global_class_probs[class_id]   # [H, W]
        binary   = (prob_map >= 0.5).astype(np.uint8)

        if not binary.any():
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10 or len(contour) < 3:
                continue

            polygon = []
            for pt in contour:
                polygon.extend([float(pt[0][0]) / orig_width, float(pt[0][1]) / orig_height])

            x_b, y_b, w_b, h_b = cv2.boundingRect(contour)

            # Confiance = max de la confiance observée dans la zone du contour
            mask_contour = np.zeros((orig_height, orig_width), dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 1, -1)
            conf_vals = global_class_conf[class_id][mask_contour == 1]
            if len(conf_vals) > 0 and conf_vals.max() > 0:
                confidence = float(conf_vals.max())
            else:
                prob_vals = prob_map[mask_contour == 1]
                confidence = float(prob_vals.max()) if len(prob_vals) > 0 else confidence_threshold

            detections.append({
                "class_id":   class_id,
                "confidence": confidence,
                "polygon":    polygon,
                "bbox":       [float(x_b), float(y_b), float(x_b + w_b), float(y_b + h_b)],
                "area":       float(area),
            })

    logger.info(f"RF-DETR Seg SAHI: {len(detections)} polygones extraits (masques globaux fusionnés)")
    return detections


def _postprocess_segformer(
    outputs: List[np.ndarray],
    img_width: int,
    img_height: int,
    model_width: int,
    model_height: int,
    confidence_threshold: float,
    bg_bias: float = 0.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Post-traite les sorties SegFormer ONNX.
    
    SegFormer output: logits [1, num_classes, H, W] où H, W sont la taille du modèle / 4
    
    Returns:
        Tuple (segmentation_mask, detections_as_polygons)
        - segmentation_mask: [H, W] avec les class IDs
        - detections_as_polygons: Liste de détections avec polygones pour les shapefiles
    """
    import cv2
    
    output = outputs[0]  # [1, num_classes, H, W]
    
    if len(output.shape) == 4:
        output = output[0]  # [num_classes, H, W]
    
    num_classes, mask_h, mask_w = output.shape
    logger.info(f"SegFormer: output shape = {output.shape}, num_classes = {num_classes}")
    
    # Appliquer softmax pour obtenir les probabilités
    exp_output = np.exp(output - np.max(output, axis=0, keepdims=True))
    probs = exp_output / np.sum(exp_output, axis=0, keepdims=True)  # [num_classes, H, W]
    
    # Redimensionner les probabilités par classe à la taille de l'image originale
    # Utiliser cv2 (INTER_LINEAR) pour cohérence avec Albumentations (entraînement)
    probs_resized = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
    for c in range(num_classes):
        probs_resized[c] = cv2.resize(probs[c], (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Appliquer le biais anti-background (réduit la proba du background avant argmax)
    if bg_bias > 0.0:
        probs_resized[0] -= bg_bias
        probs_resized[0] = np.clip(probs_resized[0], 0.0, 1.0)
        logger.info(f"SegFormer: bg_bias={bg_bias} appliqué (pénalité background)")
    
    # Obtenir la classe prédite et la confiance pour chaque pixel
    segmentation_mask = np.argmax(probs_resized, axis=0).astype(np.uint8)  # [H, W]
    confidence_mask = np.max(probs_resized, axis=0)  # [H, W]
    
    # Convertir le masque en polygones pour chaque classe (sauf background = 0)
    detections = _mask_to_polygons(
        segmentation_mask, 
        confidence_mask, 
        confidence_threshold,
        img_width,
        img_height,
    )
    
    logger.info(f"SegFormer: {len(detections)} polygones extraits")
    
    return segmentation_mask, detections


def _mask_to_polygons(
    segmentation_mask: np.ndarray,
    confidence_mask: np.ndarray,
    confidence_threshold: float,
    img_width: int,
    img_height: int,
    min_area: int = 25,
) -> List[Dict]:
    """
    Convertit un masque de segmentation en polygones.
    
    Args:
        segmentation_mask: [H, W] avec les class IDs
        confidence_mask: [H, W] avec les probabilités
        confidence_threshold: Seuil de confiance minimum
        img_width, img_height: Dimensions de l'image
        min_area: Aire minimum pour un polygone (en pixels)
    
    Returns:
        Liste de détections avec polygones
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV non disponible, conversion masque->polygones impossible")
        return []
    
    detections = []
    unique_classes = np.unique(segmentation_mask)
    
    for class_id in unique_classes:
        if class_id == 0:  # Ignorer le background
            continue
        
        # Créer un masque binaire pour cette classe
        binary_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        # Appliquer le seuil de confiance
        class_confidence = confidence_mask * binary_mask
        binary_mask = (class_confidence >= confidence_threshold).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            continue
        
        # Trouver les contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            if len(contour) < 3:
                continue
            
            # Convertir en liste de coordonnées normalisées [x1, y1, x2, y2, ...]
            # Pas de simplification pour garder la précision maximale
            polygon = []
            for point in contour:
                x = float(point[0][0]) / img_width
                y = float(point[0][1]) / img_height
                polygon.extend([x, y])
            
            # Segmentation sémantique: confiance fixée à 1.00
            # (pas de confiance per-instance, chaque polygone est une région segmentée)
            mean_confidence = 1.0
            
            # Calculer la bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(x + w), float(y + h)]
            
            detections.append({
                "class_id": int(class_id) - 1,  # 0-indexed (background était 0)
                "confidence": mean_confidence,
                "polygon": polygon,
                "bbox": bbox,
                "area": float(area),
            })
    
    return detections


def _save_segmentation_annotated_image(
    pil_image,
    segmentation_mask: np.ndarray,
    detections: List[Dict],
    output_path: str,
    class_names: List[str] = None,
    class_colors: List[int] = None,
    alpha: float = 0.5,
    jpeg_quality: int = 95,
) -> None:
    """
    Sauvegarde une image annotée avec le masque de segmentation superposé.
    
    Args:
        pil_image: Image PIL source
        segmentation_mask: [H, W] avec les class IDs
        detections: Liste des détections avec polygones (pour les contours)
        output_path: Chemin de sortie
        class_names: Liste des noms de classes
        class_colors: Liste des indices de couleurs par classe
        alpha: Transparence du masque (0-1)
        jpeg_quality: Qualité JPEG
    """
    try:
        from PIL import Image, ImageDraw
        from .class_utils import get_class_color, BASE_COLOR_PALETTE
        
        img_copy = pil_image.copy().convert("RGBA")
        img_width, img_height = img_copy.size
        
        # Créer un overlay pour le masque de segmentation
        overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        overlay_pixels = overlay.load()
        
        # Colorier chaque pixel selon sa classe
        unique_classes = np.unique(segmentation_mask)
        for class_id in unique_classes:
            if class_id == 0:  # Ignorer le background
                continue
            
            # Obtenir la couleur pour cette classe (0-indexed dans class_colors)
            color = get_class_color(int(class_id) - 1, class_colors)
            rgba_color = (*color, int(255 * alpha))
            
            # Appliquer la couleur aux pixels de cette classe
            mask_indices = np.where(segmentation_mask == class_id)
            for y, x in zip(mask_indices[0], mask_indices[1]):
                if 0 <= x < img_width and 0 <= y < img_height:
                    overlay_pixels[x, y] = rgba_color
        
        # Fusionner l'overlay avec l'image
        img_with_mask = Image.alpha_composite(img_copy, overlay)
        
        # Dessiner les contours des polygones
        draw = ImageDraw.Draw(img_with_mask)
        for det in detections:
            if "polygon" not in det:
                continue
            
            polygon = det["polygon"]
            class_id = det.get("class_id", 0)
            color = get_class_color(class_id, class_colors)
            
            # Convertir les coordonnées normalisées en pixels
            points = []
            for i in range(0, len(polygon), 2):
                x = int(polygon[i] * img_width)
                y = int(polygon[i + 1] * img_height)
                points.append((x, y))
            
            if len(points) >= 3:
                # Dessiner le contour du polygone
                draw.polygon(points, outline=color, width=2)
        
        # Convertir en RGB pour sauvegarder en JPEG
        img_final = img_with_mask.convert("RGB")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            img_final.save(output_path, quality=jpeg_quality, optimize=True)
        else:
            img_final.save(output_path)
        
        logger.info(f"Image segmentation annotée sauvegardée: {output_path}")
        
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder l'image annotée: {e}")
        import traceback
        traceback.print_exc()


def _run_segformer_with_sahi(
    pil_image,
    session,
    input_name: str,
    model_width: int,
    model_height: int,
    slice_width: int,
    slice_height: int,
    overlap_ratio: float,
    confidence_threshold: float,
    merge_polygons: bool = True,
    model_type: str = "smp",
    bg_bias: float = 0.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Exécute SegFormer avec SAHI slicing et fusionne les masques.
    
    Args:
        pil_image: Image PIL source
        session: Session ONNX
        input_name: Nom de l'entrée du modèle
        model_width, model_height: Taille d'entrée du modèle
        slice_width, slice_height: Taille des tuiles SAHI
        overlap_ratio: Ratio de chevauchement
        confidence_threshold: Seuil de confiance
        merge_polygons: Fusionner les polygones adjacents (formes linéaires)
    
    Returns:
        Tuple (segmentation_mask, detections)
    """
    import cv2
    from PIL import Image
    
    img_width, img_height = pil_image.size
    img_array = np.array(pil_image)
    
    # Découper l'image en tuiles avec SAHI
    sliced_images, orig_height, orig_width = sahi_lite_slice_image(
        image=img_array,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )
    
    logger.info(f"SegFormer SAHI: {len(sliced_images)} tuiles à traiter")
    
    # Accumuler les probabilités par classe séparément [num_classes, H, W]
    # C'est la méthode correcte : on somme les probas par classe puis on prend argmax
    global_probs = None  # Initialisé à la première tuile (quand on connaît num_classes)
    vote_count = np.zeros((img_height, img_width), dtype=np.float32)
    num_classes = None
    
    for idx, sliced_img in enumerate(sliced_images):
        slice_pil = Image.fromarray(sliced_img.image)
        slice_w, slice_h = slice_pil.size
        start_x, start_y = sliced_img.starting_pixel
        
        # Prétraiter et inférer
        input_tensor = _preprocess_image(slice_pil, (model_width, model_height), model_type)
        outputs = session.run(None, {input_name: input_tensor})
        
        # Post-traiter pour obtenir les probabilités de cette tuile
        output = outputs[0]
        if len(output.shape) == 4:
            output = output[0]  # [num_classes, H, W]
        
        # Initialiser global_probs à la première tuile
        if global_probs is None:
            num_classes = output.shape[0]
            global_probs = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
        
        # Softmax pour obtenir les probabilités
        exp_output = np.exp(output - np.max(output, axis=0, keepdims=True))
        probs = exp_output / np.sum(exp_output, axis=0, keepdims=True)  # [num_classes, H, W]
        
        # Redimensionner chaque canal de probabilité à la taille de la tuile originale
        # Utiliser cv2 (INTER_LINEAR) pour cohérence avec le preprocessing
        probs_resized = np.zeros((num_classes, slice_h, slice_w), dtype=np.float32)
        for c in range(num_classes):
            probs_resized[c] = cv2.resize(probs[c], (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)
        
        # Accumuler dans le masque global
        end_x = min(start_x + slice_w, img_width)
        end_y = min(start_y + slice_h, img_height)
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        # Sommer les probabilités par classe (vote correct)
        global_probs[:, start_y:end_y, start_x:end_x] += probs_resized[:, :actual_h, :actual_w]
        vote_count[start_y:end_y, start_x:end_x] += 1.0
        
        if (idx + 1) % 10 == 0:
            logger.info(f"SegFormer SAHI: {idx + 1}/{len(sliced_images)} tuiles traitées")
    
    # Normaliser par le nombre de votes (moyenne des probabilités)
    vote_count = np.maximum(vote_count, 1.0)  # Éviter division par zéro
    for c in range(num_classes):
        global_probs[c] /= vote_count
    
    # Appliquer le biais anti-background (réduit la proba du background avant argmax)
    if bg_bias > 0.0:
        global_probs[0] -= bg_bias
        global_probs[0] = np.clip(global_probs[0], 0.0, 1.0)
        logger.info(f"SegFormer SAHI: bg_bias={bg_bias} appliqué (pénalité background)")
    
    # Classe finale = argmax des probabilités moyennes par classe
    final_mask = np.argmax(global_probs, axis=0).astype(np.uint8)  # [H, W]
    final_confidence = np.max(global_probs, axis=0)  # [H, W]
    
    # Appliquer le seuil de confiance
    final_mask[final_confidence < confidence_threshold] = 0
    
    logger.info(f"SegFormer SAHI: masque fusionné {final_mask.shape}")
    
    # Convertir le masque en polygones
    detections = _mask_to_polygons(
        final_mask, final_confidence, confidence_threshold, img_width, img_height
    )
    
    # Fusionner les polygones adjacents si demandé (pour les formes linéaires)
    if merge_polygons and detections:
        detections = _merge_adjacent_polygons(detections, img_width, img_height)
    
    return final_mask, detections


def _merge_adjacent_polygons(
    detections: List[Dict],
    img_width: int,
    img_height: int,
    buffer_distance: float = 0.001,  # 0.1% de l'image (précision maximale)
) -> List[Dict]:
    """
    Fusionne les polygones adjacents de même classe (pour les formes linéaires).
    
    Utilise une approche conservative pour éviter les artefacts géométriques:
    - Buffer petit pour connecter uniquement les polygones vraiment adjacents
    - Simplification des géométries après fusion
    - Filtrage des polygones aberrants (triangles fins, etc.)
    
    Args:
        detections: Liste des détections avec polygones
        img_width, img_height: Dimensions de l'image
        buffer_distance: Distance de buffer pour la fusion (en fraction de l'image)
    
    Returns:
        Liste des détections avec polygones fusionnés
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError:
        logger.warning("shapely non disponible, fusion des polygones ignorée")
        return detections
    
    # Grouper les détections par classe
    by_class = {}
    for det in detections:
        class_id = det.get("class_id", 0)
        if class_id not in by_class:
            by_class[class_id] = []
        by_class[class_id].append(det)
    
    merged_detections = []
    buffer_px = buffer_distance * max(img_width, img_height)
    min_area = 25  # Aire minimale en pixels²
    
    for class_id, class_dets in by_class.items():
        # Convertir les polygones normalisés en géométries Shapely (SANS buffer initial)
        shapely_polys = []
        confidences = []
        original_areas = []
        
        for det in class_dets:
            polygon = det.get("polygon", [])
            if len(polygon) < 6:  # Minimum 3 points
                continue
            
            # Convertir coordonnées normalisées en pixels
            coords = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] * img_width
                y = polygon[i + 1] * img_height
                coords.append((x, y))
            
            if len(coords) >= 3:
                try:
                    poly = ShapelyPolygon(coords)
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    if poly.is_valid and not poly.is_empty and poly.area >= min_area:
                        shapely_polys.append(poly)
                        confidences.append(det.get("confidence", 0.5))
                        original_areas.append(poly.area)
                except Exception:
                    continue
        
        if not shapely_polys:
            continue
        
        # Calculer l'aire totale originale pour validation
        total_original_area = sum(original_areas)
        
        # Fusionner les polygones qui se touchent ou sont très proches
        try:
            # Appliquer un petit buffer, fusionner, puis réduire
            buffered_polys = [p.buffer(buffer_px, join_style=2) for p in shapely_polys]  # join_style=2 = mitre
            merged = unary_union(buffered_polys)
            
            # Réduire le buffer pour revenir à la taille originale
            merged = merged.buffer(-buffer_px, join_style=2)
            
            # Extraire les polygones résultants (pas de simplification)
            if merged.is_empty:
                # Fallback: garder les originaux
                merged_detections.extend(class_dets)
                continue
            
            # Gérer MultiPolygon, Polygon, ou GeometryCollection
            if merged.geom_type == 'MultiPolygon':
                polys = list(merged.geoms)
            elif merged.geom_type == 'Polygon':
                polys = [merged]
            elif merged.geom_type == 'GeometryCollection':
                polys = [g for g in merged.geoms if g.geom_type == 'Polygon']
            else:
                merged_detections.extend(class_dets)
                continue
            
            # Confiance moyenne pour cette classe
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            for poly in polys:
                if poly.is_empty or poly.area < min_area:
                    continue
                
                # Filtrer les polygones aberrants (triangles fins, artefacts)
                # Ratio isopérimétrique: 4*pi*area/perimeter² = 1 pour un cercle, ~0.78 pour un carré
                # Les triangles très fins ont un ratio très bas
                if poly.length > 0:
                    compactness = 4 * 3.14159 * poly.area / (poly.length ** 2)
                    if compactness < 0.005:  # Filtrer uniquement les artefacts extrêmes
                        logger.debug(f"Polygone filtré (compactness={compactness:.3f})")
                        continue
                
                # Vérifier que le polygone n'est pas aberrant (aire >> aire originale)
                if poly.area > total_original_area * 2:
                    logger.warning(f"Polygone aberrant filtré (aire={poly.area:.0f} >> originale={total_original_area:.0f})")
                    continue
                
                # Convertir en coordonnées normalisées
                try:
                    coords = list(poly.exterior.coords)
                except Exception:
                    continue
                    
                polygon_norm = []
                for x, y in coords[:-1]:  # Exclure le point de fermeture
                    # Clamp aux limites de l'image
                    x = max(0, min(x, img_width))
                    y = max(0, min(y, img_height))
                    polygon_norm.extend([x / img_width, y / img_height])
                
                if len(polygon_norm) < 6:  # Minimum 3 points
                    continue
                
                # Calculer la bbox
                minx, miny, maxx, maxy = poly.bounds
                
                merged_detections.append({
                    "class_id": class_id,
                    "confidence": avg_confidence,
                    "polygon": polygon_norm,
                    "bbox": [minx, miny, maxx, maxy],
                    "area": poly.area,
                })
                
        except Exception as e:
            logger.warning(f"Erreur fusion polygones classe {class_id}: {e}")
            # Fallback: garder les détections originales
            merged_detections.extend(class_dets)
    
    logger.info(f"Fusion polygones: {len(detections)} -> {len(merged_detections)}")
    return merged_detections


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
        task = model_meta.get("task", "detect")
        logger.info(f"ONNX: modèle {model_type}, taille {model_width}x{model_height}, task={task}")
        
        # =====================================================================
        # Mode SEGMENTATION SÉMANTIQUE (SegFormer)
        # =====================================================================
        if model_type in ("segformer", "smp") or task == "semantic_segmentation":
            logger.info(f"ONNX: mode segmentation sémantique")
            
            # Paramètres de segmentation depuis les métadonnées du modèle
            use_sahi = model_meta.get("use_sahi", True)
            merge_polygons = model_meta.get("merge_polygons", True)
            bg_bias = float(model_meta.get("bg_bias", 0.0))
            
            # Override per-model du confidence_threshold
            model_confidence = model_meta.get("confidence_threshold")
            if model_confidence is not None:
                confidence_threshold = float(model_confidence)
            logger.info(f"Segmentation: confidence_threshold={confidence_threshold}, bg_bias={bg_bias}")
            
            if use_sahi and (img_width > model_width * 1.5 or img_height > model_height * 1.5):
                # =========================================================
                # Mode SAHI: découper l'image en tuiles et fusionner les masques
                # =========================================================
                logger.info(f"SegFormer SAHI: image {img_width}x{img_height} -> tuiles {slice_width}x{slice_height}")
                
                segmentation_mask, all_detections = _run_segformer_with_sahi(
                    pil_image=pil_image,
                    session=session,
                    input_name=input_name,
                    model_width=model_width,
                    model_height=model_height,
                    slice_width=slice_width,
                    slice_height=slice_height,
                    overlap_ratio=overlap_ratio,
                    confidence_threshold=confidence_threshold,
                    merge_polygons=merge_polygons,
                    model_type=model_type,
                    bg_bias=bg_bias,
                )
            else:
                # =========================================================
                # Mode direct: image entière (petites images)
                # =========================================================
                logger.info(f"SegFormer direct: image {img_width}x{img_height}")
                
                input_tensor = _preprocess_image(pil_image, (model_width, model_height), model_type)
                outputs = session.run(None, {input_name: input_tensor})
                
                segmentation_mask, all_detections = _postprocess_segformer(
                    outputs, img_width, img_height, model_width, model_height, confidence_threshold,
                    bg_bias=bg_bias,
                )
            
            logger.info(f"SegFormer: {len(all_detections)} polygones détectés")
            
            # Sauvegarder les résultats
            if not all_detections:
                logger.info(f"ONNX: aucune détection dans {Path(image_path).name}")
                save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
                return (False, 0) if return_count else False
            
            num_detections = len(all_detections)
            
            # Sauvegarder les fichiers (avec polygones)
            save_detections_to_files(
                image_path=image_path,
                output_path=output_path,
                detections=all_detections,
                img_width=img_width,
                img_height=img_height,
                jpg_folder_path=jpg_folder_path,
                task="segment",
                model_type=model_type,
            )
            
            logger.info(f"SegFormer: {num_detections} polygones sauvegardés pour {Path(image_path).name}")
            
            # Générer l'image annotée avec masque de segmentation
            if generate_annotated_images:
                _save_segmentation_annotated_image(
                    pil_image, segmentation_mask, all_detections, output_path, 
                    class_names=class_names, class_colors=class_colors
                )
            
            return (True, num_detections) if return_count else True
        
        # =====================================================================
        # Mode INSTANCE SEGMENTATION (RF-DETR Seg)
        # =====================================================================
        if model_type == "rfdetr" and task == "instance_segmentation":
            logger.info("ONNX: mode instance segmentation RF-DETR Seg")
            class_offset = model_meta.get("class_offset", 1)

            # Accumulation des masques de probabilité par classe dans l'espace global
            # → polygones extraits une seule fois, pas de doublons ni d'offsets
            all_detections = _run_rfdetr_seg_with_sahi(
                pil_image=pil_image,
                session=session,
                input_name=input_name,
                model_width=model_width,
                model_height=model_height,
                slice_width=slice_width,
                slice_height=slice_height,
                overlap_ratio=overlap_ratio,
                confidence_threshold=confidence_threshold,
                class_offset=class_offset,
            )
            orig_width, orig_height = pil_image.size
            logger.info(f"RF-DETR Seg: {len(all_detections)} instances après fusion globale")

            if not all_detections:
                save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
                return (False, 0) if return_count else False

            num_detections = len(all_detections)
            save_detections_to_files(
                image_path=image_path,
                output_path=output_path,
                detections=all_detections,
                img_width=orig_width,
                img_height=orig_height,
                jpg_folder_path=jpg_folder_path,
                task="segment",
                model_type=model_type,
            )
            logger.info(f"RF-DETR Seg: {num_detections} instances sauvegardées pour {Path(image_path).name}")

            if generate_annotated_images and all_detections:
                save_annotated_image(pil_image, all_detections, output_path, class_names=class_names, class_colors=class_colors)

            return (True, num_detections) if return_count else True

        # =====================================================================
        # Mode DÉTECTION (YOLO, RF-DETR)
        # =====================================================================
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
                class_offset = model_meta.get("class_offset", 1)  # défaut: 1 pour compatibilité
                dets = _postprocess_rfdetr(outputs, slice_w, slice_h, model_width, model_height, confidence_threshold, class_offset)
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
