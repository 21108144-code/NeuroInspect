"""
NeuroInspect - Pixel-Level Defect Localization
Generates segmentation masks and heatmaps from anomaly detection output.
"""
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class DefectRegion:
    """Represents a localized defect region."""
    id: int
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) normalized
    mask: np.ndarray  # Binary mask for this region
    area_pixels: int
    area_percentage: float
    centroid: Tuple[float, float]
    mean_intensity: float
    max_intensity: float
    contour: np.ndarray


class DefectLocalizer:
    """
    Localizes defects from anomaly heatmaps.
    Converts reconstruction error maps to actionable bounding boxes and masks.
    """
    
    def __init__(self, threshold: float = 0.3, min_area: int = 100,
                 max_regions: int = 20):
        """
        Args:
            threshold: Anomaly intensity threshold (0-1)
            min_area: Minimum defect area in pixels
            max_regions: Maximum number of defect regions to return
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_regions = max_regions
    
    def localize(self, error_map: np.ndarray, 
                 original_image: np.ndarray = None) -> Dict[str, Any]:
        """
        Extract defect regions from anomaly heatmap.
        
        Args:
            error_map: Normalized anomaly map (H, W) with values in [0, 1]
            original_image: Original image for overlay generation
        
        Returns:
            Dictionary with:
            - regions: List of DefectRegion objects
            - binary_mask: Combined binary mask
            - heatmap_colored: Colored heatmap visualization
            - overlay: Heatmap overlaid on original (if provided)
        """
        h, w = error_map.shape[:2]
        
        # Apply threshold to create binary mask
        binary_mask = (error_map > self.threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract regions
        regions = []
        total_area = h * w
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = (x / w, y / h, (x + bw) / w, (y + bh) / h)  # Normalized
            
            # Create region mask
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            
            # Calculate statistics
            masked_values = error_map[region_mask > 0]
            mean_intensity = float(np.mean(masked_values)) if len(masked_values) > 0 else 0
            max_intensity = float(np.max(masked_values)) if len(masked_values) > 0 else 0
            
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"] / w
                cy = M["m01"] / M["m00"] / h
            else:
                cx, cy = (x + bw/2) / w, (y + bh/2) / h
            
            regions.append(DefectRegion(
                id=idx,
                bbox=bbox,
                mask=region_mask,
                area_pixels=int(area),
                area_percentage=area / total_area * 100,
                centroid=(cx, cy),
                mean_intensity=mean_intensity,
                max_intensity=max_intensity,
                contour=contour
            ))
        
        # Sort by area (largest first) and limit
        regions.sort(key=lambda r: r.area_pixels, reverse=True)
        regions = regions[:self.max_regions]
        
        # Generate colored heatmap
        heatmap_colored = self._create_colored_heatmap(error_map)
        
        # Generate overlay if original provided
        overlay = None
        if original_image is not None:
            overlay = self._create_overlay(original_image, heatmap_colored, regions)
        
        return {
            "regions": regions,
            "binary_mask": binary_mask,
            "heatmap_colored": heatmap_colored,
            "overlay": overlay,
            "total_defect_area": sum(r.area_percentage for r in regions),
        }
    
    def _create_colored_heatmap(self, error_map: np.ndarray) -> np.ndarray:
        """Convert grayscale error map to colored heatmap."""
        # Normalize to 0-255
        heatmap = (error_map * 255).astype(np.uint8)
        
        # Apply colormap (JET for industrial visibility)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def _create_overlay(self, original: np.ndarray, heatmap: np.ndarray,
                        regions: List[DefectRegion], alpha: float = 0.4) -> np.ndarray:
        """Create heatmap overlay with bounding boxes."""
        # Ensure same size
        if original.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Ensure 3 channels
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Blend
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        
        # Draw bounding boxes
        h, w = overlay.shape[:2]
        for region in regions:
            x1 = int(region.bbox[0] * w)
            y1 = int(region.bbox[1] * h)
            x2 = int(region.bbox[2] * w)
            y2 = int(region.bbox[3] * h)
            
            # Color based on intensity (green to red)
            intensity = region.max_intensity
            color = (
                int(255 * (1 - intensity)),  # B
                int(255 * (1 - intensity)),  # G
                255,                          # R
            )
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"D{region.id}: {region.area_percentage:.1f}%"
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay
    
    def set_threshold(self, threshold: float) -> None:
        """Update localization threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Localization threshold set to {self.threshold}")
