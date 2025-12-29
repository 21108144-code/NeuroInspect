"""
NeuroInspect - Structured Logging Configuration
Production-ready logging with JSON format support and rotation.
"""
import sys
from loguru import logger
from app.config import get_settings


def setup_logging() -> None:
    """Configure loguru for production-grade logging."""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Console format based on environment
    if settings.log_format == "json":
        log_format = (
            '{{"timestamp":"{time:YYYY-MM-DDTHH:mm:ss.SSS}","level":"{level}",'
            '"module":"{module}","function":"{function}","line":{line},'
            '"message":"{message}"}}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=settings.log_format != "json",
        backtrace=settings.debug,
        diagnose=settings.debug,
    )
    
    # Add file handler for production
    if settings.is_production:
        logger.add(
            "logs/neuroinspect_{time:YYYY-MM-DD}.log",
            format=log_format,
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
        )
    
    logger.info(f"Logging configured: level={settings.log_level}, format={settings.log_format}")


# Export configured logger
__all__ = ["logger", "setup_logging"]
