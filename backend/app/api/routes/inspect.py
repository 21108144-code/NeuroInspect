"""
NeuroInspect - Inspection API Routes
Endpoints for image/video defect inspection using pre-trained detector.
Updated: Fixed classification to default to CRACK instead of ANOMALY.
"""
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, Query
from loguru import logger

from app.models.schemas import (
    InspectionRequest,
    InspectionResult,
    DefectDetection,
    BoundingBox,
    SeverityLevel,
    DefectType,
)
from app.api.deps import (
    get_detector,
    get_localizer,
    get_severity_scorer,
    get_database,
)
from app.cv.pretrained_detector import EnhancedDefectDetector
from app.cv.localizer import DefectLocalizer
from app.cv.severity import SeverityScorer
from app.models.database import DatabaseManager, Inspection, Defect
from app.utils.image_utils import load_image, encode_image_base64, resize_image

router = APIRouter(prefix="/inspect", tags=["Inspection"])


@router.post("", response_model=InspectionResult)
async def inspect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5),
    return_heatmap: bool = Form(default=True),
    return_mask: bool = Form(default=True),
    return_explanation: bool = Form(default=False),
    detector: EnhancedDefectDetector = Depends(get_detector),
    scorer: SeverityScorer = Depends(get_severity_scorer),
    db: DatabaseManager = Depends(get_database),
):
    """
    Inspect a single image for defects using pre-trained anomaly detection.
    
    - **file**: Image file (JPEG, PNG, BMP)
    - **confidence_threshold**: Minimum confidence for detection (0-1)
    - **return_heatmap**: Include anomaly heatmap in response
    - **return_mask**: Include binary defect mask in response
    - **return_explanation**: Include textual explanation
    """
    import time
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        contents = await file.read()
        image = load_image(contents)
        image, scale = resize_image(image, max_size=1024)
        
        h, w = image.shape[:2]
        
        # Run detection with pre-trained model
        detection_result = detector.detect(image, return_heatmap=True)
        
        anomaly_map = detection_result["anomaly_map"]
        anomaly_score = detection_result["anomaly_score"]
        is_defective = detection_result["is_defective"]
        
        # Localize defects from anomaly map
        defects = _localize_defects_from_map(
            anomaly_map, 
            image, 
            scorer, 
            threshold=confidence_threshold
        )
        
        # Create heatmap visualization
        heatmap_colored = None
        if return_heatmap and "heatmap" in detection_result:
            heatmap_colored = detection_result["heatmap"]
        elif return_heatmap:
            heatmap_colored = _create_heatmap(image, anomaly_map)
        
        # Create binary mask
        binary_mask = None
        if return_mask:
            binary_mask = ((anomaly_map > confidence_threshold) * 255).astype(np.uint8)
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Prepare response
        processing_time = (time.time() - start_time) * 1000
        inspection_id = str(uuid.uuid4())
        
        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=datetime.utcnow(),
            image_name=file.filename or "uploaded_image",
            image_width=w,
            image_height=h,
            processing_time_ms=processing_time,
            is_defective=is_defective,
            overall_score=anomaly_score,
            defects=defects,
            heatmap_base64=encode_image_base64(heatmap_colored) if return_heatmap else None,
            mask_base64=encode_image_base64(binary_mask) if return_mask else None,
            explanation=f"Detected {len(defects)} defect(s) with overall anomaly score {anomaly_score:.2%}" if return_explanation else None,
        )
        
        # Store in database
        await _store_inspection(db, result, detection_result)
        
        logger.info(f"Inspection complete: {inspection_id}, {len(defects)} defects, {processing_time:.1f}ms")
        return result
        
    except Exception as e:
        logger.exception(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _localize_defects_from_map(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    scorer: SeverityScorer,
    threshold: float = 0.5,
) -> List[DefectDetection]:
    """
    Localize individual defects from the anomaly map.
    Uses connected components analysis with aggressive merging.
    """
    h, w = anomaly_map.shape[:2]
    
    # Use lower threshold for better detection
    actual_threshold = max(0.1, threshold * 0.5)  # Use half the threshold
    
    # Threshold the anomaly map
    binary = (anomaly_map > actual_threshold).astype(np.uint8)
    
    # AGGRESSIVE morphological operations to merge nearby crack segments
    # Use large vertical kernel to connect vertical crack pieces
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))  # Tall vertical kernel
    binary = cv2.dilate(binary, kernel_v, iterations=2)
    
    # Close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Erode back a bit to tighten bounding boxes
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.erode(binary, kernel_erode, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    defects = []
    
    # Collect all component areas to find the largest
    component_areas = []
    for i in range(1, num_labels):
        area = stats[i][4]  # Area is the 5th element
        component_areas.append((i, area))
    
    # Sort by area descending - largest first
    component_areas.sort(key=lambda x: x[1], reverse=True)
    
    # ONLY KEEP THE SINGLE LARGEST COMPONENT (the main crack)
    # The crack should always be the largest connected region
    if len(component_areas) == 0:
        return []
    
    # Take only the top 1 component (the crack)
    for idx, (i, area) in enumerate(component_areas[:1]):
        # Get component stats
        x, y, comp_w, comp_h, _ = stats[i]
        cx, cy = centroids[i]
        
        # Must have some minimum area (but not too strict)
        if area < 100:
            continue
        
        # Calculate normalized bounding box
        x_min = x / w
        y_min = y / h
        x_max = (x + comp_w) / w
        y_max = (y + comp_h) / h
        
        # Area percentage
        area_pct = (area / (h * w)) * 100
        
        # Get mean anomaly score in region
        region_mask = (labels == i)
        mean_score = float(np.mean(anomaly_map[region_mask]))
        max_score = float(np.max(anomaly_map[region_mask]))
        
        # Calculate severity
        severity_result = scorer.calculate_severity(
            area_percentage=area_pct,
            max_intensity=max_score,
            centroid=(cx / w, cy / h),
        )
        
        # Classify defect type
        defect_type = _classify_defect_type_from_region(
            aspect_ratio=comp_w / (comp_h + 1e-6),
            area_pct=area_pct,
            intensity=mean_score,
        )
        
        defects.append(DefectDetection(
            id=str(uuid.uuid4())[:8],
            defect_type=defect_type,
            confidence=mean_score,
            severity=SeverityLevel(severity_result["level"].value),
            severity_score=severity_result["score"],
            bounding_box=BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            ),
            area_percentage=area_pct,
            pixel_count=area,
        ))
    
    # Sort by severity score
    defects.sort(key=lambda d: d.severity_score, reverse=True)
    
    return defects[:20]  # Limit to top 20 defects


def _classify_defect_type_from_region(
    aspect_ratio: float,
    area_pct: float,
    intensity: float,
) -> DefectType:
    """
    Classify defect type based on region characteristics.
    
    Priority order:
    1. High intensity = Crack (most serious)
    2. Elongated horizontal = Scratch
    3. Elongated vertical = Crack  
    4. Large area = Stain/Corrosion
    5. Compact medium = Dent
    6. Default = Crack (safer to flag as serious)
    """
    # High intensity is almost always a crack or serious defect
    if intensity > 0.5:
        return DefectType.CRACK
    
    # Very elongated patterns
    if aspect_ratio > 4:
        return DefectType.SCRATCH  # Horizontal scratch
    elif aspect_ratio < 0.25:
        return DefectType.CRACK    # Vertical crack
    
    # Large area patterns
    if area_pct > 5:
        if intensity > 0.4:
            return DefectType.CORROSION
        return DefectType.STAIN
    
    # Medium-sized compact regions
    if 0.5 < aspect_ratio < 2.0 and area_pct > 0.5 and intensity < 0.4:
        return DefectType.DENT
    
    # Default to crack (better to over-report serious defects)
    return DefectType.CRACK


def _create_heatmap(image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    """Create colored heatmap overlay."""
    heatmap = (anomaly_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


@router.post("/batch", response_model=List[InspectionResult])
async def inspect_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(default=0.5),
    detector: EnhancedDefectDetector = Depends(get_detector),
    scorer: SeverityScorer = Depends(get_severity_scorer),
):
    """
    Batch inspect multiple images.
    
    - **files**: List of image files (max 32)
    - **confidence_threshold**: Minimum confidence for detection
    """
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per batch")
    
    results = []
    for file in files:
        try:
            result = await inspect_image(
                file=file,
                confidence_threshold=confidence_threshold,
                return_heatmap=False,
                return_mask=False,
                return_explanation=False,
                detector=detector,
                scorer=scorer,
                db=None,
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process {file.filename}: {e}")
    
    return results


@router.get("/health")
async def inspection_health(detector: EnhancedDefectDetector = Depends(get_detector)):
    """Check inspection service health."""
    import torch
    
    return {
        "status": "healthy" if detector.is_loaded else "degraded",
        "model_loaded": detector.is_loaded,
        "device": str(detector.device),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


async def _store_inspection(
    db: DatabaseManager,
    result: InspectionResult,
    detection_result: dict
) -> None:
    """Store inspection result in database."""
    if db is None:
        return
    
    try:
        async for session in db.get_session():
            # Create inspection record
            inspection = Inspection(
                id=result.inspection_id,
                timestamp=result.timestamp,
                image_name=result.image_name,
                image_width=result.image_width,
                image_height=result.image_height,
                processing_time_ms=result.processing_time_ms,
                is_defective=result.is_defective,
                overall_score=result.overall_score,
            )
            session.add(inspection)
            
            # Create defect records
            for defect in result.defects:
                defect_record = Defect(
                    id=defect.id,
                    inspection_id=result.inspection_id,
                    defect_type=defect.defect_type.value,
                    confidence=defect.confidence,
                    severity=defect.severity.value,
                    severity_score=defect.severity_score,
                    bbox_x_min=defect.bounding_box.x_min,
                    bbox_y_min=defect.bounding_box.y_min,
                    bbox_x_max=defect.bounding_box.x_max,
                    bbox_y_max=defect.bounding_box.y_max,
                    area_percentage=defect.area_percentage,
                    pixel_count=defect.pixel_count,
                )
                session.add(defect_record)
            
            await session.commit()
    except Exception as e:
        logger.warning(f"Failed to store inspection: {e}")
