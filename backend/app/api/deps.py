"""
NeuroInspect - API Dependencies
Shared dependencies for FastAPI routes.
"""
from typing import Optional
from functools import lru_cache
from fastapi import Depends

from app.config import Settings, get_settings
from app.models.database import DatabaseManager
from app.cv.pretrained_detector import EnhancedDefectDetector
from app.cv.localizer import DefectLocalizer
from app.cv.severity import SeverityScorer, SeverityConfig
from app.cv.clustering import RootCauseAnalyzer
from app.cv.explainability import ExplainabilityEngine


# Global instances (initialized on startup)
_detector: Optional[EnhancedDefectDetector] = None
_localizer: Optional[DefectLocalizer] = None
_severity_scorer: Optional[SeverityScorer] = None
_root_cause: Optional[RootCauseAnalyzer] = None
_explainability: Optional[ExplainabilityEngine] = None
_database: Optional[DatabaseManager] = None


def init_dependencies(settings: Settings) -> None:
    """Initialize all service dependencies."""
    global _detector, _localizer, _severity_scorer, _root_cause, _explainability, _database
    
    # Initialize pre-trained detector
    _detector = EnhancedDefectDetector(
        device=settings.model_device,
        threshold=settings.detection_threshold,
    )
    
    # Initialize localizer
    _localizer = DefectLocalizer(
        threshold=settings.localization_threshold,
    )
    
    # Initialize severity scorer
    _severity_scorer = SeverityScorer(
        SeverityConfig(
            weight_area=settings.severity_weights_area,
            weight_intensity=settings.severity_weights_intensity,
            weight_location=settings.severity_weights_location,
        )
    )
    
    # Initialize root cause analyzer
    _root_cause = RootCauseAnalyzer()
    
    # Initialize explainability (disabled for pre-trained detector)
    # The pre-trained detector doesn't expose the internal model for Grad-CAM
    _explainability = None
    
    # Initialize database
    _database = DatabaseManager(settings.database_url)


def get_detector() -> EnhancedDefectDetector:
    """Get defect detector instance."""
    if _detector is None:
        raise RuntimeError("Detector not initialized")
    return _detector


def get_localizer() -> DefectLocalizer:
    """Get defect localizer instance."""
    if _localizer is None:
        raise RuntimeError("Localizer not initialized")
    return _localizer


def get_severity_scorer() -> SeverityScorer:
    """Get severity scorer instance."""
    if _severity_scorer is None:
        raise RuntimeError("Severity scorer not initialized")
    return _severity_scorer


def get_root_cause_analyzer() -> RootCauseAnalyzer:
    """Get root cause analyzer instance."""
    if _root_cause is None:
        raise RuntimeError("Root cause analyzer not initialized")
    return _root_cause


def get_explainability() -> ExplainabilityEngine:
    """Get explainability engine instance."""
    if _explainability is None:
        raise RuntimeError("Explainability engine not initialized")
    return _explainability


def get_database() -> DatabaseManager:
    """Get database manager instance."""
    if _database is None:
        raise RuntimeError("Database not initialized")
    return _database
