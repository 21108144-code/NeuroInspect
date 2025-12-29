"""
NeuroInspect - Database Models
SQLAlchemy models for defect history persistence.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Inspection(Base):
    """Inspection record for each analyzed image/frame."""
    __tablename__ = "inspections"
    
    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    image_name = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=True)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    is_defective = Column(Boolean, default=False, index=True)
    overall_score = Column(Float, nullable=False)
    heatmap_path = Column(String(500), nullable=True)
    mask_path = Column(String(500), nullable=True)
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    defects = relationship("Defect", back_populates="inspection", cascade="all, delete-orphan")


class Defect(Base):
    """Individual defect detected in an inspection."""
    __tablename__ = "defects"
    
    id = Column(String(36), primary_key=True)
    inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=False, index=True)
    defect_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    severity_score = Column(Float, nullable=False)
    bbox_x_min = Column(Float, nullable=False)
    bbox_y_min = Column(Float, nullable=False)
    bbox_x_max = Column(Float, nullable=False)
    bbox_y_max = Column(Float, nullable=False)
    area_percentage = Column(Float, nullable=False)
    pixel_count = Column(Integer, nullable=False)
    feature_vector = Column(JSON, nullable=True)  # For clustering
    cluster_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    inspection = relationship("Inspection", back_populates="defects")


class ClusterAnalysis(Base):
    """Root cause cluster analysis results."""
    __tablename__ = "cluster_analyses"
    
    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_defects = Column(Integer, nullable=False)
    num_clusters = Column(Integer, nullable=False)
    noise_points = Column(Integer, nullable=False)
    algorithm = Column(String(50), default="hdbscan")
    parameters = Column(JSON, nullable=True)
    clusters_info = Column(JSON, nullable=False)
    recommendations = Column(JSON, nullable=True)


class SystemSettings(Base):
    """Persisted system settings."""
    __tablename__ = "system_settings"
    
    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database connection management
class DatabaseManager:
    """Async database connection manager."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(
            database_url,
            echo=False,
            future=True,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def init_db(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.async_session() as session:
            yield session
    
    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()
