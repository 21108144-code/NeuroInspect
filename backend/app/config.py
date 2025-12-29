"""
NeuroInspect Configuration Management
Centralized settings using Pydantic for validation and environment loading.
"""
from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "NeuroInspect"
    app_env: str = "development"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Model Configuration
    model_device: str = "cuda"
    model_weights_dir: str = "./weights"
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    localization_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    severity_weights_area: float = Field(default=0.4, ge=0.0, le=1.0)
    severity_weights_intensity: float = Field(default=0.3, ge=0.0, le=1.0)
    severity_weights_location: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/neuroinspect.db"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Performance
    max_batch_size: int = 32
    inference_workers: int = 4
    image_max_size: int = 1024
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    return Settings()
