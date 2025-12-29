"""
NeuroInspect - Pydantic Schemas
Request/Response models for API endpoints.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Defect severity classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DefectType(str, Enum):
    """Common industrial defect categories."""
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    STAIN = "stain"
    CORROSION = "corrosion"
    DEFORMATION = "deformation"
    MISSING_PART = "missing_part"
    FOREIGN_OBJECT = "foreign_object"
    SURFACE_ROUGHNESS = "surface_roughness"
    ANOMALY = "anomaly"


# ============ Request Schemas ============

class InspectionRequest(BaseModel):
    """Request for single image inspection."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_heatmap: bool = Field(default=True)
    return_mask: bool = Field(default=True)
    return_explanation: bool = Field(default=False)


class BatchInspectionRequest(BaseModel):
    """Request for batch image inspection."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_images: int = Field(default=32, ge=1, le=100)


class RootCauseRequest(BaseModel):
    """Request for root cause analysis."""
    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int = Field(default=3, ge=1)
    time_range_hours: Optional[int] = Field(default=24, ge=1)


class XAIRequest(BaseModel):
    """Request for explainability output."""
    method: str = Field(default="gradcam", pattern="^(gradcam|attention|occlusion)$")
    target_layer: Optional[str] = None
    return_overlay: bool = Field(default=True)


class SettingsUpdateRequest(BaseModel):
    """Request to update detection settings."""
    detection_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    localization_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    severity_weights: Optional[Dict[str, float]] = None


# ============ Response Schemas ============

class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized 0-1)."""
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class DefectDetection(BaseModel):
    """Single defect detection result."""
    id: str
    defect_type: DefectType
    confidence: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel
    severity_score: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    area_percentage: float = Field(ge=0.0, le=100.0)
    pixel_count: int = Field(ge=0)


class InspectionResult(BaseModel):
    """Complete inspection result for single image."""
    inspection_id: str
    timestamp: datetime
    image_name: str
    image_width: int
    image_height: int
    processing_time_ms: float
    is_defective: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    defects: List[DefectDetection]
    heatmap_base64: Optional[str] = None
    mask_base64: Optional[str] = None
    explanation: Optional[str] = None


class DefectSummary(BaseModel):
    """Summary statistics for defect listing."""
    total_inspections: int
    total_defects: int
    defects_by_type: Dict[str, int]
    defects_by_severity: Dict[str, int]
    avg_severity_score: float
    detection_rate: float


class DefectListResponse(BaseModel):
    """Response for defect listing endpoint."""
    summary: DefectSummary
    defects: List[Dict[str, Any]]
    page: int
    page_size: int
    total_pages: int


class ClusterInfo(BaseModel):
    """Information about a defect cluster."""
    cluster_id: int
    size: int
    centroid: List[float]
    dominant_type: DefectType
    avg_severity: float
    time_range: Dict[str, datetime]
    potential_cause: str
    confidence: float


class RootCauseResponse(BaseModel):
    """Response for root cause analysis."""
    analysis_id: str
    timestamp: datetime
    total_defects_analyzed: int
    num_clusters: int
    noise_points: int
    clusters: List[ClusterInfo]
    recommendations: List[str]


class XAIResponse(BaseModel):
    """Response for explainability endpoint."""
    inspection_id: str
    method: str
    target_layer: str
    heatmap_base64: str
    overlay_base64: Optional[str] = None
    attention_regions: List[BoundingBox]
    explanation_text: str
    feature_importance: Dict[str, float]


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None


class SettingsResponse(BaseModel):
    """Current settings response."""
    detection_threshold: float
    localization_threshold: float
    severity_weights: Dict[str, float]
    model_device: str
    max_batch_size: int
