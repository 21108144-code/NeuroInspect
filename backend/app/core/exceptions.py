"""
NeuroInspect - Custom Exception Handlers
Standardized error responses for the API.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger


class NeuroInspectException(Exception):
    """Base exception for NeuroInspect application."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InferenceError(NeuroInspectException):
    """Exception raised during model inference."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class ImageProcessingError(NeuroInspectException):
    """Exception raised during image preprocessing."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)


class ModelNotLoadedError(NeuroInspectException):
    """Exception raised when model is not available."""
    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503)


class InvalidInputError(NeuroInspectException):
    """Exception raised for invalid input data."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)


async def neuroinspect_exception_handler(
    request: Request, exc: NeuroInspectException
) -> JSONResponse:
    """Global exception handler for NeuroInspect exceptions."""
    logger.error(f"NeuroInspect Error: {exc.message}", extra={"details": exc.details})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for standard HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "path": str(request.url),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "path": str(request.url),
        },
    )
