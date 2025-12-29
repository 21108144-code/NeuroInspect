"""NeuroInspect Models Package"""
from app.models.schemas import (
    SeverityLevel,
    DefectType,
    InspectionRequest,
    BatchInspectionRequest,
    RootCauseRequest,
    XAIRequest,
    SettingsUpdateRequest,
    BoundingBox,
    DefectDetection,
    InspectionResult,
    DefectSummary,
    DefectListResponse,
    ClusterInfo,
    RootCauseResponse,
    XAIResponse,
    HealthResponse,
    SettingsResponse,
)
from app.models.database import (
    Base,
    Inspection,
    Defect,
    ClusterAnalysis,
    SystemSettings,
    DatabaseManager,
)

__all__ = [
    # Enums
    "SeverityLevel",
    "DefectType",
    # Request Schemas
    "InspectionRequest",
    "BatchInspectionRequest",
    "RootCauseRequest",
    "XAIRequest",
    "SettingsUpdateRequest",
    # Response Schemas
    "BoundingBox",
    "DefectDetection",
    "InspectionResult",
    "DefectSummary",
    "DefectListResponse",
    "ClusterInfo",
    "RootCauseResponse",
    "XAIResponse",
    "HealthResponse",
    "SettingsResponse",
    # Database
    "Base",
    "Inspection",
    "Defect",
    "ClusterAnalysis",
    "SystemSettings",
    "DatabaseManager",
]
