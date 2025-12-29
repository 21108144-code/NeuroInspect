"""
NeuroInspect - Explainability Module
Grad-CAM implementation for visual explanations of defect detections.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for autoencoder explanations.
    
    Modified for anomaly detection:
    - Uses reconstruction error as the "class" to explain
    - Highlights regions contributing most to high reconstruction error
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Args:
            model: The autoencoder model
            target_layer: Name of layer to extract gradients from.
                         If None, uses last encoder conv layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        # Find and hook target layer
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward and backward hooks on target layer."""
        target = self._find_target_layer()
        if target is None:
            logger.warning("Could not find target layer for Grad-CAM")
            return
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))
        logger.debug(f"Grad-CAM hooks registered on {type(target).__name__}")
    
    def _find_target_layer(self) -> Optional[torch.nn.Module]:
        """Find the target layer for gradient extraction."""
        if self.target_layer:
            # Find by name
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    return module
        
        # Default: last conv layer in encoder
        last_conv = None
        for module in self.model.encoder.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_size: Tuple[int, int] = None
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_size: Optional output size (W, H)
        
        Returns:
            Dictionary with:
            - heatmap: Normalized heatmap (H, W) in [0, 1]
            - attention_regions: List of high-attention bounding boxes
            - explanation_text: Textual explanation
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        reconstruction, _ = self.model(input_tensor)
        
        # Compute reconstruction error as target
        error = F.mse_loss(reconstruction, input_tensor, reduction='none')
        target = error.mean()
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return self._empty_result(input_tensor.shape[2:])
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        if target_size is None:
            target_size = (input_tensor.shape[3], input_tensor.shape[2])
        
        heatmap = cv2.resize(cam, target_size)
        
        # Extract attention regions
        attention_regions = self._extract_attention_regions(heatmap)
        
        # Generate explanation text
        explanation = self._generate_explanation(heatmap, attention_regions)
        
        return {
            "heatmap": heatmap,
            "attention_regions": attention_regions,
            "explanation_text": explanation,
            "raw_cam": cam,
        }
    
    def _extract_attention_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Extract bounding boxes of high-attention regions."""
        h, w = heatmap.shape
        
        # Threshold to binary
        binary = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small regions
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = {
                "x_min": x / w,
                "y_min": y / h,
                "x_max": (x + bw) / w,
                "y_max": (y + bh) / h,
            }
            
            # Calculate attention intensity in region
            mask = np.zeros_like(heatmap)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            intensity = float(np.mean(heatmap[mask > 0]))
            
            regions.append({
                "id": i,
                "bbox": bbox,
                "intensity": intensity,
                "area_percentage": (area / (h * w)) * 100,
            })
        
        regions.sort(key=lambda r: r["intensity"], reverse=True)
        return regions[:10]  # Top 10 regions
    
    def _generate_explanation(
        self,
        heatmap: np.ndarray,
        regions: List[Dict]
    ) -> str:
        """Generate human-readable explanation."""
        if not regions:
            return "No significant anomalous regions detected. The reconstruction closely matches the input."
        
        explanation_parts = [
            f"Detected {len(regions)} regions of interest contributing to anomaly detection."
        ]
        
        # Describe top regions
        for i, region in enumerate(regions[:3]):
            bbox = region["bbox"]
            cx = (bbox["x_min"] + bbox["x_max"]) / 2
            cy = (bbox["y_min"] + bbox["y_max"]) / 2
            
            # Describe location
            loc_x = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
            loc_y = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
            
            explanation_parts.append(
                f"Region {i+1}: Located in {loc_x}-{loc_y} area, "
                f"covering {region['area_percentage']:.1f}% of image "
                f"with {region['intensity']*100:.0f}% attention intensity."
            )
        
        # Overall assessment
        total_attention = sum(r["area_percentage"] for r in regions)
        if total_attention > 20:
            explanation_parts.append(
                "Large affected area suggests significant deviation from normal patterns."
            )
        elif total_attention > 5:
            explanation_parts.append(
                "Moderate affected area indicates localized anomaly."
            )
        else:
            explanation_parts.append(
                "Small affected area suggests minor or subtle deviation."
            )
        
        return " ".join(explanation_parts)
    
    def _empty_result(self, size: Tuple[int, int]) -> Dict[str, Any]:
        """Return empty result when Grad-CAM fails."""
        return {
            "heatmap": np.zeros(size),
            "attention_regions": [],
            "explanation_text": "Unable to generate explanation. Gradients not available.",
            "raw_cam": None,
        }
    
    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create visualization overlay of Grad-CAM on original image."""
        # Ensure same size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to color
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Ensure original is 3-channel
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Blend
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def __del__(self):
        """Clean up hooks."""
        for hook in self._hooks:
            hook.remove()


class ExplainabilityEngine:
    """
    High-level explainability interface for the inspection system.
    Provides multiple explanation methods and aggregates results.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.grad_cam = GradCAM(model)
    
    @torch.no_grad()
    def explain(
        self,
        image: np.ndarray,
        method: str = "gradcam"
    ) -> Dict[str, Any]:
        """
        Generate explanation for an image.
        
        Args:
            image: Input image (H, W, C) in uint8
            method: Explanation method ("gradcam", "occlusion", "attention")
        
        Returns:
            Explanation results
        """
        # Preprocess
        tensor = self._preprocess(image)
        
        if method == "gradcam":
            return self._explain_gradcam(tensor, image)
        elif method == "occlusion":
            return self._explain_occlusion(tensor, image)
        else:
            return self._explain_gradcam(tensor, image)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (256, 256))
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _explain_gradcam(
        self,
        tensor: torch.Tensor,
        original: np.ndarray
    ) -> Dict[str, Any]:
        """Generate Grad-CAM explanation."""
        # Need gradients for Grad-CAM
        tensor.requires_grad_(True)
        
        # Enable gradients temporarily
        with torch.enable_grad():
            result = self.grad_cam.generate(tensor, (original.shape[1], original.shape[0]))
        
        # Create overlay
        overlay = self.grad_cam.create_overlay(original, result["heatmap"])
        result["overlay"] = overlay
        
        return result
    
    def _explain_occlusion(
        self,
        tensor: torch.Tensor,
        original: np.ndarray,
        patch_size: int = 32,
        stride: int = 16
    ) -> Dict[str, Any]:
        """Generate occlusion-based explanation."""
        _, _, h, w = tensor.shape
        
        # Get baseline reconstruction error
        with torch.no_grad():
            recon, _ = self.model(tensor)
            baseline_error = F.mse_loss(recon, tensor).item()
        
        # Compute importance for each patch
        importance_map = np.zeros((h, w))
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Occlude patch
                occluded = tensor.clone()
                occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.5  # Gray occlusion
                
                with torch.no_grad():
                    recon, _ = self.model(occluded)
                    occluded_error = F.mse_loss(recon, tensor).item()
                
                # Importance = how much error changes
                importance = abs(occluded_error - baseline_error)
                importance_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                    importance_map[y:y+patch_size, x:x+patch_size], importance
                )
        
        # Normalize
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
        
        # Resize to original
        heatmap = cv2.resize(importance_map, (original.shape[1], original.shape[0]))
        
        return {
            "heatmap": heatmap,
            "method": "occlusion",
            "explanation_text": "Occlusion sensitivity shows regions where masking causes largest reconstruction change.",
        }
    
    def get_feature_importance(
        self,
        defect_result: Dict
    ) -> Dict[str, float]:
        """Get importance of different features for detection."""
        # This would analyze which features contributed most
        # For now, return placeholder based on detection result
        return {
            "reconstruction_error": 0.45,
            "texture_deviation": 0.25,
            "edge_consistency": 0.15,
            "color_variance": 0.15,
        }
