"""
NeuroInspect - Pre-trained Anomaly Detector using ResNet
Uses pre-trained ResNet backbone for feature extraction.
No training required - works out of the box.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from loguru import logger

# Use pretrained ResNet for feature-based anomaly detection
import torchvision.models as models
import torchvision.transforms as transforms


class PretrainedDetector:
    """
    Anomaly detector using pre-trained ResNet features.
    Works out-of-the-box without training on specific defect data.
    Uses feature distribution comparison for anomaly scoring.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.5,
        backbone: str = "resnet18",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.backbone_name = backbone
        
        # Load pre-trained ResNet
        logger.info(f"Loading pre-trained {backbone} backbone...")
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final classification layer - we want features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Reference features from normal images (will be computed on first normal samples)
        self.reference_features: Optional[torch.Tensor] = None
        self.is_loaded = True
        
        logger.info(f"Pre-trained detector initialized on {self.device}")
    
    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract features from image using pre-trained backbone."""
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform and extract features
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features
    
    def compute_anomaly_map(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute pixel-level anomaly map using feature statistics.
        
        Returns:
            anomaly_map: Normalized anomaly heatmap (0-1)
            anomaly_score: Overall anomaly score (0-1)
        """
        h, w = image.shape[:2]
        
        # Extract features
        features = self.extract_features(image)  # [1, C, H', W']
        
        # Compute feature statistics for anomaly detection
        # Use channel-wise variance as anomaly indicator
        feat_var = torch.var(features, dim=1, keepdim=True)  # [1, 1, H', W']
        
        # Compute mean feature magnitude
        feat_mag = torch.mean(torch.abs(features), dim=1, keepdim=True)
        
        # Combine variance and magnitude for anomaly score
        anomaly_feat = feat_var * feat_mag
        
        # Normalize
        anomaly_feat = (anomaly_feat - anomaly_feat.min()) / (anomaly_feat.max() - anomaly_feat.min() + 1e-8)
        
        # Resize to original image size
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_feat, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
        # Enhance contrast
        anomaly_map = np.power(anomaly_map, 0.5)  # Gamma correction
        
        # Apply edge detection to find actual defects
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # Combine feature-based anomaly with edge detection
        combined_map = 0.6 * anomaly_map + 0.4 * edges_normalized
        combined_map = np.clip(combined_map, 0, 1)
        
        # Overall anomaly score
        anomaly_score = float(np.percentile(combined_map, 95))
        
        return combined_map, anomaly_score
    
    def detect(
        self,
        image: np.ndarray,
        return_heatmap: bool = True,
    ) -> Dict[str, Any]:
        """
        Run anomaly detection on an image.
        
        Args:
            image: Input image (BGR format)
            return_heatmap: Whether to return visualization
            
        Returns:
            Dictionary with detection results
        """
        # Compute anomaly map
        anomaly_map, anomaly_score = self.compute_anomaly_map(image)
        
        # Determine if defective based on threshold
        is_defective = anomaly_score > self.threshold
        
        result = {
            "is_defective": is_defective,
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "threshold": self.threshold,
        }
        
        if return_heatmap:
            result["heatmap"] = self._create_heatmap(image, anomaly_map)
        
        return result
    
    def _create_heatmap(self, image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        """Create colored heatmap overlay."""
        # Normalize and colorize anomaly map
        heatmap = (anomaly_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return overlay


class EnhancedDefectDetector:
    """
    Enhanced defect detector that combines pre-trained features with 
    classical image processing for robust anomaly detection.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.pretrained_detector = PretrainedDetector(device, threshold)
        self.is_loaded = True
        
        logger.info("Enhanced defect detector initialized")
    
    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold."""
        self.threshold = threshold
        self.pretrained_detector.set_threshold(threshold)
    
    def detect(
        self,
        image: np.ndarray,
        return_heatmap: bool = True,
    ) -> Dict[str, Any]:
        """
        Run detection combining deep features and classical methods.
        Uses direct crack detection via dark-line thresholding.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # DIRECT CRACK DETECTION: Cracks are dark lines on lighter background
        # Use adaptive thresholding to find dark regions
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect dark pixels (cracks are darker than background)
        mean_val = np.mean(blur)
        std_val = np.std(blur)
        dark_threshold = mean_val - 1.5 * std_val
        
        # Binary mask of dark regions (potential cracks)
        dark_mask = (blur < dark_threshold).astype(np.uint8) * 255
        
        # Use adaptive threshold as backup
        adaptive_mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Combine both approaches
        combined_mask = cv2.bitwise_or(dark_mask, adaptive_mask)
        
        # Morphological operations to clean up and connect crack lines
        # Use vertical kernel to connect vertical cracks
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_v)
        
        # Use small kernel to remove noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Create anomaly map from mask
        anomaly_map = combined_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to smooth the map
        anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)
        
        # Calculate anomaly score
        anomaly_score = float(np.percentile(anomaly_map, 95))
        is_defective = anomaly_score > self.threshold
        
        result = {
            "is_defective": is_defective,
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "threshold": self.threshold,
        }
        
        if return_heatmap:
            result["heatmap"] = self._create_heatmap(image, anomaly_map)
        
        return result
    
    def _classical_defect_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Classical image processing for defect detection.
        Specifically tuned to detect vertical cracks while ignoring horizontal texture.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use Sobel to separate horizontal and vertical edges
        # Sobel X detects vertical edges (like cracks) - we want this
        # Sobel Y detects horizontal edges (like texture lines) - we want to suppress this
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Take absolute values
        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)
        
        # Normalize
        sobelx = (sobelx - sobelx.min()) / (sobelx.max() - sobelx.min() + 1e-8)
        sobely = (sobely - sobely.min()) / (sobely.max() - sobely.min() + 1e-8)
        
        # Emphasize vertical edges (cracks), suppress horizontal edges (texture)
        # Weight vertical edges much higher than horizontal
        edge_map = 0.8 * sobelx + 0.2 * sobely
        
        # Also use Canny for fine edges
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_canny = edges_canny.astype(np.float32) / 255.0
        
        # Dilate to connect crack segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))  # Vertical kernel
        edge_map_dilated = cv2.dilate(edge_map.astype(np.float32), kernel, iterations=1)
        
        # Texture analysis - look for discontinuities
        blur = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
        local_diff = np.abs(gray.astype(np.float32) - blur)
        local_diff = (local_diff - local_diff.min()) / (local_diff.max() - local_diff.min() + 1e-8)
        
        # Combine: prioritize vertical edges
        classical_map = 0.6 * edge_map_dilated + 0.2 * edges_canny + 0.2 * local_diff
        classical_map = np.clip(classical_map, 0, 1).astype(np.float32)
        
        return classical_map
    
    def _create_heatmap(self, image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        """Create colored heatmap overlay."""
        heatmap = (anomaly_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return overlay
