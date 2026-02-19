"""
SAHI-lite: Slicing Aided Hyper Inference - Version légère numpy-only.

Basé sur le travail de:
- Fatih C Akyon (SAHI original)
- UTokyo-FieldPhenomics-Lab/EasyAMS (version numpy)

Cette implémentation supprime la dépendance à torch pour permettre
un runner ONNX léger.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# =============================================================================
# Slicing
# =============================================================================

def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    Calcule les bounding boxes des slices pour une image.
    
    Args:
        image_height: Hauteur de l'image originale
        image_width: Largeur de l'image originale
        slice_height: Hauteur de chaque slice
        slice_width: Largeur de chaque slice
        overlap_height_ratio: Ratio de chevauchement vertical (0.0-1.0)
        overlap_width_ratio: Ratio de chevauchement horizontal (0.0-1.0)
    
    Returns:
        Liste de bboxes [x_min, y_min, x_max, y_max] pour chaque slice
    """
    slice_bboxes = []
    
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    
    y_min = 0
    while y_min < image_height:
        y_max = y_min + slice_height
        
        x_min = 0
        while x_min < image_width:
            x_max = x_min + slice_width
            
            # Ajuster si on dépasse les bords
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            
            x_min = x_max - x_overlap
        
        y_min = y_max - y_overlap
    
    return slice_bboxes


@dataclass
class SlicedImage:
    """Représente une slice d'image."""
    image: np.ndarray
    starting_pixel: List[int]  # [x, y]


def slice_image(
    image: np.ndarray,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> Tuple[List[SlicedImage], int, int]:
    """
    Découpe une image en slices avec chevauchement.
    
    Args:
        image: Image numpy (H, W, C)
        slice_height: Hauteur de chaque slice
        slice_width: Largeur de chaque slice
        overlap_height_ratio: Ratio de chevauchement vertical
        overlap_width_ratio: Ratio de chevauchement horizontal
    
    Returns:
        Tuple (liste de SlicedImage, hauteur originale, largeur originale)
    """
    image_height, image_width = image.shape[:2]
    
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    
    sliced_images = []
    for bbox in slice_bboxes:
        x_min, y_min, x_max, y_max = bbox
        slice_img = image[y_min:y_max, x_min:x_max]
        sliced_images.append(SlicedImage(
            image=slice_img,
            starting_pixel=[x_min, y_min]
        ))
    
    return sliced_images, image_height, image_width


# =============================================================================
# Detection result
# =============================================================================

@dataclass
class Detection:
    """Représente une détection."""
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: Optional[str] = None
    
    def to_shifted(self, shift_x: int, shift_y: int) -> "Detection":
        """Retourne une copie avec les coordonnées décalées."""
        return Detection(
            bbox=[
                self.bbox[0] + shift_x,
                self.bbox[1] + shift_y,
                self.bbox[2] + shift_x,
                self.bbox[3] + shift_y,
            ],
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
        )
    
    def clip_to_bounds(self, width: int, height: int) -> "Detection":
        """Clip les coordonnées aux limites de l'image."""
        return Detection(
            bbox=[
                max(0, min(width, self.bbox[0])),
                max(0, min(height, self.bbox[1])),
                max(0, min(width, self.bbox[2])),
                max(0, min(height, self.bbox[3])),
            ],
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
        )
    
    def is_valid(self) -> bool:
        """Vérifie si la bbox est valide (aire > 0)."""
        return self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]
    
    @property
    def area(self) -> float:
        """Calcule l'aire de la bbox."""
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


# =============================================================================
# NMS / Greedy NMM
# =============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calcule l'IoU entre deux boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_ios(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calcule l'IoS (Intersection over Smaller) entre deux boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    smaller_area = min(area1, area2)
    
    if smaller_area == 0:
        return 0.0
    
    return inter_area / smaller_area


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> List[int]:
    """
    Non-Maximum Suppression en numpy pur.
    
    Args:
        boxes: Array (N, 4) de bboxes [x1, y1, x2, y2]
        scores: Array (N,) de scores
        iou_threshold: Seuil IoU pour la suppression
    
    Returns:
        Liste des indices à conserver
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calcul IoU avec les boxes restantes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-10)
        
        # Garder les boxes avec IoU < seuil
        mask = iou < iou_threshold
        order = order[1:][mask]
    
    return keep


def greedy_nmm(
    detections: List[Detection],
    match_threshold: float = 0.5,
    match_metric: str = "IOS",
    class_agnostic: bool = False,
) -> List[Detection]:
    """
    Greedy Non-Maximum Merging.
    
    Fusionne les détections qui se chevauchent au lieu de simplement les supprimer.
    
    Args:
        detections: Liste de détections
        match_threshold: Seuil de correspondance
        match_metric: "IOU" ou "IOS"
        class_agnostic: Si True, ignore les classes lors du matching
    
    Returns:
        Liste de détections après fusion
    """
    if len(detections) == 0:
        return []
    
    # Convertir en arrays numpy pour le traitement
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.score for d in detections])
    class_ids = np.array([d.class_id for d in detections])
    
    if class_agnostic:
        # Traiter toutes les classes ensemble
        keep_indices = _greedy_nmm_single_class(
            boxes, scores, match_threshold, match_metric
        )
    else:
        # Traiter chaque classe séparément
        keep_indices = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_indices = np.where(class_mask)[0]
            
            class_keep = _greedy_nmm_single_class(
                boxes[class_mask],
                scores[class_mask],
                match_threshold,
                match_metric,
            )
            
            keep_indices.extend(class_indices[class_keep].tolist())
    
    return [detections[i] for i in keep_indices]


def _greedy_nmm_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    match_threshold: float,
    match_metric: str,
) -> List[int]:
    """Greedy NMM pour une seule classe."""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)
    
    keep = []
    
    while len(order) > 0:
        idx = order[-1]
        keep.append(idx)
        order = order[:-1]
        
        if len(order) == 0:
            break
        
        # Calcul intersection
        xx1 = np.maximum(x1[order], x1[idx])
        yy1 = np.maximum(y1[order], y1[idx])
        xx2 = np.minimum(x2[order], x2[idx])
        yy2 = np.minimum(y2[order], y2[idx])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        if match_metric == "IOU":
            union = areas[order] + areas[idx] - inter
            match_value = inter / np.maximum(union, 1e-10)
        elif match_metric == "IOS":
            smaller = np.minimum(areas[order], areas[idx])
            match_value = inter / np.maximum(smaller, 1e-10)
        else:
            raise ValueError(f"Unknown match_metric: {match_metric}")
        
        # Garder les boxes avec match < seuil
        mask = match_value < match_threshold
        order = order[mask]
    
    return keep


# =============================================================================
# Sliced prediction
# =============================================================================

def merge_sliced_detections(
    all_detections: List[List[Detection]],
    slice_starting_pixels: List[List[int]],
    original_height: int,
    original_width: int,
    match_threshold: float = 0.5,
    match_metric: str = "IOS",
    class_agnostic: bool = False,
) -> List[Detection]:
    """
    Fusionne les détections de toutes les slices.
    
    Args:
        all_detections: Liste de listes de détections (une par slice)
        slice_starting_pixels: Pixels de départ de chaque slice [[x, y], ...]
        original_height: Hauteur de l'image originale
        original_width: Largeur de l'image originale
        match_threshold: Seuil pour la fusion
        match_metric: "IOU" ou "IOS"
        class_agnostic: Ignorer les classes lors de la fusion
    
    Returns:
        Liste de détections fusionnées
    """
    # Décaler toutes les détections vers les coordonnées de l'image originale
    shifted_detections = []
    
    for slice_detections, starting_pixel in zip(all_detections, slice_starting_pixels):
        shift_x, shift_y = starting_pixel
        for det in slice_detections:
            shifted = det.to_shifted(shift_x, shift_y)
            clipped = shifted.clip_to_bounds(original_width, original_height)
            if clipped.is_valid():
                shifted_detections.append(clipped)
    
    if len(shifted_detections) == 0:
        return []
    
    # Appliquer Greedy NMM pour fusionner les détections qui se chevauchent
    merged = greedy_nmm(
        shifted_detections,
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=class_agnostic,
    )
    
    return merged


# =============================================================================
# Utilitaires
# =============================================================================

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convertit les boxes de format [x, y, w, h] vers [x1, y1, x2, y2]."""
    result = boxes.copy()
    result[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    result[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return result


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convertit les boxes de format [x1, y1, x2, y2] vers [x, y, w, h]."""
    result = boxes.copy()
    result[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    result[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
    return result


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convertit les boxes de format [cx, cy, w, h] vers [x1, y1, x2, y2]."""
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return result
