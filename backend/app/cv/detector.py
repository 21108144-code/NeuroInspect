"""
NeuroInspect - Convolutional Autoencoder for Anomaly Detection
Production-grade unsupervised defect detection using reconstruction error.

Architecture:
- Encoder: Convolutional layers reducing spatial dimensions
- Latent: Compressed representation (bottleneck)
- Decoder: Transposed convolutions reconstructing input
- Anomaly Detection: High reconstruction error = defect

Key Design Decisions:
1. Trained on GOOD samples only - learns normal patterns
2. Defects cause high reconstruction error (anomalies)
3. GPU-optimized with batch inference support
4. Supports both grayscale and RGB images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
from loguru import logger


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed convolutional block for decoder."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.deconv(x)))


class DefectAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for industrial defect detection.
    
    Learns to reconstruct normal (defect-free) images.
    Defects manifest as regions with high reconstruction error.
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, 
                 image_size: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),      # 256 -> 128
            ConvBlock(32, 64, stride=2),               # 128 -> 64
            ConvBlock(64, 128, stride=2),              # 64 -> 32
            ConvBlock(128, 256, stride=2),             # 32 -> 16
            ConvBlock(256, 512, stride=2),             # 16 -> 8
        )
        
        # Latent space
        self.fc_encode = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            DeconvBlock(512, 256),                     # 8 -> 16
            DeconvBlock(256, 128),                     # 16 -> 32
            DeconvBlock(128, 64),                      # 32 -> 64
            DeconvBlock(64, 32),                       # 64 -> 128
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_encode(features)
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image."""
        features = self.fc_decode(z)
        features = features.view(features.size(0), 512, 8, 8)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-wise reconstruction error (anomaly map).
        Higher values indicate potential defects.
        """
        reconstruction, _ = self.forward(x)
        # Per-pixel L2 error
        error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
        return error


class DefectDetector:
    """
    Production wrapper for defect detection using autoencoder.
    Handles model loading, preprocessing, inference, and post-processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda",
                 threshold: float = 0.5, image_size: int = 256):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.image_size = image_size
        self.model: Optional[DefectAutoencoder] = None
        self.is_loaded = False
        
        # Statistics for normalization (computed during training)
        self.mean_error: float = 0.0
        self.std_error: float = 1.0
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._initialize_fresh_model()
    
    def _initialize_fresh_model(self):
        """Initialize model with random weights (for demo/training)."""
        self.model = DefectAutoencoder(in_channels=3, image_size=self.image_size)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        logger.info(f"Initialized fresh autoencoder model on {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = DefectAutoencoder(
                in_channels=checkpoint.get("in_channels", 3),
                latent_dim=checkpoint.get("latent_dim", 128),
                image_size=checkpoint.get("image_size", 256)
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Load normalization statistics if available
            self.mean_error = checkpoint.get("mean_error", 0.0)
            self.std_error = checkpoint.get("std_error", 1.0)
            
            self.is_loaded = True
            logger.info(f"Loaded model from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_fresh_model()
    
    def save_model(self, model_path: str) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.model.in_channels,
            "latent_dim": self.model.latent_dim,
            "image_size": self.model.image_size,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved model to {model_path}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C) in BGR or RGB format, uint8
        
        Returns:
            Preprocessed tensor (1, C, H, W) normalized to [0, 1]
        """
        import cv2
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1] and convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect defects in a single image.
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            Dictionary with detection results:
            - is_defective: bool
            - anomaly_score: float (0-1)
            - reconstruction: np.ndarray
            - error_map: np.ndarray (normalized anomaly heatmap)
            - latent: np.ndarray (feature vector for clustering)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        tensor = self.preprocess(image)
        original_shape = image.shape[:2]
        
        # Inference
        reconstruction, latent = self.model(tensor)
        error_map = self.model.get_reconstruction_error(tensor)
        
        # Compute anomaly score (mean reconstruction error)
        raw_score = error_map.mean().item()
        
        # Normalize score using training statistics
        normalized_score = (raw_score - self.mean_error) / (self.std_error + 1e-8)
        anomaly_score = float(torch.sigmoid(torch.tensor(normalized_score)).item())
        
        # Resize error map to original size
        error_map_np = error_map.squeeze().cpu().numpy()
        error_map_np = (error_map_np - error_map_np.min()) / (error_map_np.max() - error_map_np.min() + 1e-8)
        
        import cv2
        error_map_resized = cv2.resize(error_map_np, (original_shape[1], original_shape[0]))
        
        # Reconstruction for visualization
        recon_np = reconstruction.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).astype(np.uint8)
        recon_resized = cv2.resize(recon_np, (original_shape[1], original_shape[0]))
        
        return {
            "is_defective": anomaly_score > self.threshold,
            "anomaly_score": anomaly_score,
            "reconstruction": recon_resized,
            "error_map": error_map_resized,
            "latent": latent.squeeze().cpu().numpy(),
        }
    
    @torch.no_grad()
    def detect_batch(self, images: list) -> list:
        """Batch detection for multiple images."""
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
    
    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold."""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Detection threshold set to {self.threshold}")
