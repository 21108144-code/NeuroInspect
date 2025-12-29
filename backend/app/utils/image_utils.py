"""
NeuroInspect - Image Processing Utilities
Helper functions for image loading, preprocessing, and encoding.
"""
import base64
import io
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
from loguru import logger


def load_image(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        source: File path, base64 string, bytes, or numpy array
    
    Returns:
        Image as numpy array (H, W, C) in BGR format
    """
    if isinstance(source, np.ndarray):
        return source
    
    if isinstance(source, bytes):
        # Decode bytes
        nparr = np.frombuffer(source, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return image
    
    if isinstance(source, str):
        # Check if base64
        if source.startswith("data:image"):
            # Remove data URL prefix
            source = source.split(",")[1]
        
        try:
            # Try base64 decode
            img_bytes = base64.b64decode(source)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            pass
        
        # Try file path
        if Path(source).exists():
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"Failed to load image from {source}")
            return image
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def resize_image(
    image: np.ndarray,
    max_size: int = 1024,
    min_size: int = 256
) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension
        min_size: Minimum dimension
    
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    scale = 1.0
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
    elif min(h, w) < min_size:
        scale = min_size / min(h, w)
    
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image, scale


def encode_image_base64(
    image: np.ndarray,
    format: str = "png",
    quality: int = 90
) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image: Image as numpy array
        format: Output format (png, jpg)
        quality: JPEG quality (0-100)
    
    Returns:
        Base64 encoded string with data URL prefix
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL
    pil_image = Image.fromarray(image_rgb)
    
    # Encode
    buffer = io.BytesIO()
    if format.lower() == "jpg" or format.lower() == "jpeg":
        pil_image.save(buffer, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        pil_image.save(buffer, format="PNG")
        mime = "image/png"
    
    # Base64 encode
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:{mime};base64,{b64}"


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    # Remove data URL prefix if present
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",")[1]
    
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode base64 image")
    
    return image


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image: Image in [0, 255] range
        mean: Channel means
        std: Channel standard deviations
    
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    normalized = (image - mean) / std
    return normalized


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Reverse normalization."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    denorm = (image * std + mean) * 255.0
    return np.clip(denorm, 0, 255).astype(np.uint8)


def create_comparison_grid(
    original: np.ndarray,
    reconstruction: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray
) -> np.ndarray:
    """Create 2x2 comparison grid for visualization."""
    h, w = original.shape[:2]
    
    # Ensure all same size
    reconstruction = cv2.resize(reconstruction, (w, h))
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.resize(overlay, (w, h))
    
    # Create grid
    top_row = np.hstack([original, reconstruction])
    bottom_row = np.hstack([heatmap, overlay])
    grid = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Reconstruction", (w + 10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Anomaly Map", (10, h + 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "Overlay", (w + 10, h + 25), font, 0.7, (255, 255, 255), 2)
    
    return grid


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for contrast enhancement."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
