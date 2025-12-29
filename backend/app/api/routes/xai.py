"""
NeuroInspect - Explainability API Routes
Endpoints for Grad-CAM and model interpretability.
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, Query
from loguru import logger
import uuid

from app.models.schemas import XAIRequest, XAIResponse, BoundingBox
from app.api.deps import get_detector, get_explainability
from app.cv.detector import DefectDetector
from app.cv.explainability import ExplainabilityEngine
from app.utils.image_utils import load_image, encode_image_base64, resize_image

router = APIRouter(prefix="/xai", tags=["Explainability"])


@router.post("/explain", response_model=XAIResponse)
async def explain_detection(
    file: UploadFile = File(...),
    method: str = Form(default="gradcam"),
    target_layer: Optional[str] = Form(default=None),
    return_overlay: bool = Form(default=True),
    detector: DefectDetector = Depends(get_detector),
    explainer: ExplainabilityEngine = Depends(get_explainability),
):
    """
    Generate explanation for defect detection.
    
    - **file**: Image file to explain
    - **method**: Explanation method (gradcam, occlusion, attention)
    - **target_layer**: Target layer for Grad-CAM (optional)
    - **return_overlay**: Include visual overlay in response
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load and preprocess image
        contents = await file.read()
        image = load_image(contents)
        image, scale = resize_image(image, max_size=1024)
        
        # Run detection first to get context
        detection_result = detector.detect(image)
        
        # Generate explanation
        explanation = explainer.explain(image, method=method)
        
        # Get feature importance
        feature_importance = explainer.get_feature_importance(detection_result)
        
        # Extract attention regions as bounding boxes
        attention_regions = []
        for region in explanation.get("attention_regions", []):
            bbox = region.get("bbox", {})
            attention_regions.append(BoundingBox(
                x_min=bbox.get("x_min", 0),
                y_min=bbox.get("y_min", 0),
                x_max=bbox.get("x_max", 0),
                y_max=bbox.get("y_max", 0),
            ))
        
        # Encode images
        heatmap_b64 = encode_image_base64(
            (explanation["heatmap"] * 255).astype("uint8"),
            format="png"
        ) if "heatmap" in explanation else ""
        
        overlay_b64 = None
        if return_overlay and "overlay" in explanation:
            overlay_b64 = encode_image_base64(explanation["overlay"], format="png")
        
        return XAIResponse(
            inspection_id=str(uuid.uuid4()),
            method=method,
            target_layer=target_layer or "auto",
            heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            attention_regions=attention_regions[:10],
            explanation_text=explanation.get("explanation_text", ""),
            feature_importance=feature_importance,
        )
        
    except Exception as e:
        logger.exception(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def list_methods():
    """List available explanation methods."""
    return {
        "methods": [
            {
                "id": "gradcam",
                "name": "Grad-CAM",
                "description": "Gradient-weighted Class Activation Mapping. Highlights regions that contribute most to the anomaly detection decision.",
                "supports_target_layer": True,
            },
            {
                "id": "occlusion",
                "name": "Occlusion Sensitivity",
                "description": "Systematically occludes parts of the image to identify which regions are most important for detection.",
                "supports_target_layer": False,
            },
            {
                "id": "attention",
                "name": "Attention Maps",
                "description": "Visualizes the attention weights from transformer-based models (if available).",
                "supports_target_layer": False,
            },
        ]
    }


@router.get("/layers")
async def list_target_layers(
    detector: DefectDetector = Depends(get_detector),
):
    """List available target layers for Grad-CAM."""
    if not detector.model:
        return {"layers": []}
    
    layers = []
    for name, module in detector.model.named_modules():
        if "conv" in name.lower() or "bn" in name.lower():
            layers.append({
                "name": name,
                "type": type(module).__name__,
            })
    
    return {"layers": layers}


@router.post("/compare")
async def compare_explanations(
    file: UploadFile = File(...),
    detector: DefectDetector = Depends(get_detector),
    explainer: ExplainabilityEngine = Depends(get_explainability),
):
    """
    Compare multiple explanation methods on the same image.
    
    Useful for understanding which regions different methods highlight.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = load_image(contents)
        image, scale = resize_image(image, max_size=512)  # Smaller for multiple methods
        
        results = {}
        
        # Grad-CAM
        try:
            gradcam_result = explainer.explain(image, method="gradcam")
            results["gradcam"] = {
                "heatmap": encode_image_base64(
                    (gradcam_result["heatmap"] * 255).astype("uint8")
                ),
                "explanation": gradcam_result.get("explanation_text", ""),
            }
        except Exception as e:
            results["gradcam"] = {"error": str(e)}
        
        # Occlusion
        try:
            occlusion_result = explainer.explain(image, method="occlusion")
            results["occlusion"] = {
                "heatmap": encode_image_base64(
                    (occlusion_result["heatmap"] * 255).astype("uint8")
                ),
                "explanation": occlusion_result.get("explanation_text", ""),
            }
        except Exception as e:
            results["occlusion"] = {"error": str(e)}
        
        # Original image
        results["original"] = {
            "image": encode_image_base64(image),
        }
        
        return {
            "comparison_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "methods": results,
        }
        
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
