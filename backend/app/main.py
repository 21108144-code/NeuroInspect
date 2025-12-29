"""
NeuroInspect - FastAPI Application Entry Point
Production-ready API server with middleware, error handling, and lifecycle management.
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import get_settings
from app.core.logging_config import setup_logging
from app.core.exceptions import (
    NeuroInspectException,
    neuroinspect_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from app.api.deps import init_dependencies, get_database, get_detector, get_localizer, get_severity_scorer
from app.api.routes import inspect, defects, root_cause, xai
from app.models.schemas import HealthResponse, SettingsResponse, SettingsUpdateRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    settings = get_settings()
    
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.app_name} v1.0.0")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Device: {settings.model_device}")
    
    # Initialize dependencies
    init_dependencies(settings)
    
    # Initialize database
    db = get_database()
    if db:
        await db.init_db()
        logger.info("Database initialized")
    
    # Create directories
    Path("./data").mkdir(exist_ok=True)
    Path("./weights").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    logger.info(f"{settings.app_name} started successfully")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")
    if db:
        await db.close()


# Create FastAPI application
settings = get_settings()
app = FastAPI(
    title="NeuroInspect API",
    description="Enterprise-grade industrial AI inspection system for defect detection, root cause analysis, and explainable AI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(NeuroInspectException, neuroinspect_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Include routers
app.include_router(inspect.router)
app.include_router(defects.router)
app.include_router(root_cause.router)
app.include_router(xai.router)


# Health and settings endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    import torch
    from app.api.deps import get_detector
    
    detector = get_detector()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=detector.is_loaded if detector else False,
        device=str(detector.device) if detector else "unknown",
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )


@app.get("/settings", response_model=SettingsResponse, tags=["System"])
async def get_current_settings():
    """Get current system settings."""
    settings = get_settings()
    return SettingsResponse(
        detection_threshold=settings.detection_threshold,
        localization_threshold=settings.localization_threshold,
        severity_weights={
            "area": settings.severity_weights_area,
            "intensity": settings.severity_weights_intensity,
            "location": settings.severity_weights_location,
        },
        model_device=settings.model_device,
        max_batch_size=settings.max_batch_size,
    )


@app.put("/settings", response_model=SettingsResponse, tags=["System"])
async def update_settings(request: SettingsUpdateRequest):
    """
    Update detection settings.
    
    Note: Some settings require application restart to take effect.
    """
    from app.api.deps import get_detector, get_localizer, get_severity_scorer
    
    if request.detection_threshold is not None:
        detector = get_detector()
        detector.set_threshold(request.detection_threshold)
    
    if request.localization_threshold is not None:
        localizer = get_localizer()
        localizer.set_threshold(request.localization_threshold)
    
    if request.severity_weights:
        scorer = get_severity_scorer()
        scorer.update_weights(
            area=request.severity_weights.get("area"),
            intensity=request.severity_weights.get("intensity"),
            location=request.severity_weights.get("location"),
        )
    
    return await get_current_settings()


@app.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "name": "NeuroInspect API",
        "version": "1.0.0",
        "description": "Industrial AI Inspection System",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
    )
