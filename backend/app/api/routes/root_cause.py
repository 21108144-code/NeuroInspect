"""
NeuroInspect - Root Cause Analysis API Routes
Endpoints for defect clustering and pattern analysis.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select
from loguru import logger
import uuid

from app.models.schemas import (
    RootCauseRequest,
    RootCauseResponse,
    ClusterInfo,
    DefectType,
)
from app.api.deps import get_database, get_root_cause_analyzer
from app.models.database import DatabaseManager, Defect, Inspection, ClusterAnalysis
from app.cv.clustering import RootCauseAnalyzer

router = APIRouter(prefix="/root_cause", tags=["Root Cause Analysis"])


@router.post("/analyze", response_model=RootCauseResponse)
async def analyze_root_causes(
    request: RootCauseRequest = None,
    db: DatabaseManager = Depends(get_database),
    analyzer: RootCauseAnalyzer = Depends(get_root_cause_analyzer),
):
    """
    Perform root cause analysis on recent defects.
    
    Uses HDBSCAN clustering to identify patterns in defects
    and infer potential root causes.
    """
    request = request or RootCauseRequest()
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=request.time_range_hours)
    
    async for session in db.get_session():
        # Get defects in time range
        query = (
            select(Defect, Inspection.timestamp)
            .join(Inspection)
            .where(Inspection.timestamp >= start_time)
            .where(Inspection.timestamp <= end_time)
        )
        
        result = await session.execute(query)
        rows = result.all()
        
        if len(rows) < request.min_cluster_size:
            return RootCauseResponse(
                analysis_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                total_defects_analyzed=len(rows),
                num_clusters=0,
                noise_points=len(rows),
                clusters=[],
                recommendations=[
                    f"Insufficient defects for clustering ({len(rows)} < {request.min_cluster_size})",
                    "Expand time range or lower minimum cluster size",
                ],
            )
        
        # Prepare defect data for clustering
        defects_data = []
        for defect, timestamp in rows:
            defects_data.append({
                "id": defect.id,
                "centroid": (
                    (defect.bbox_x_min + defect.bbox_x_max) / 2,
                    (defect.bbox_y_min + defect.bbox_y_max) / 2,
                ),
                "defect_type": defect.defect_type,
                "severity_score": defect.severity_score,
                "severity_level": defect.severity,
                "timestamp": timestamp,
                "latent": defect.feature_vector,
            })
        
        # Run clustering
        analyzer.min_cluster_size = request.min_cluster_size
        analyzer.min_samples = request.min_samples
        
        analysis_result = analyzer.analyze(defects_data)
        
        # Convert clusters to response format
        clusters = []
        for cluster in analysis_result.get("clusters", []):
            clusters.append(ClusterInfo(
                cluster_id=cluster.cluster_id,
                size=cluster.size,
                centroid=cluster.centroid,
                dominant_type=DefectType(cluster.dominant_type) if cluster.dominant_type in [e.value for e in DefectType] else DefectType.ANOMALY,
                avg_severity=cluster.avg_severity,
                time_range={
                    "start": cluster.time_range[0],
                    "end": cluster.time_range[1],
                },
                potential_cause=cluster.potential_causes[0] if cluster.potential_causes else "Unknown",
                confidence=cluster.confidence,
            ))
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        analysis_record = ClusterAnalysis(
            id=analysis_id,
            timestamp=datetime.utcnow(),
            total_defects=len(defects_data),
            num_clusters=len(clusters),
            noise_points=analysis_result.get("noise_count", 0),
            parameters={
                "min_cluster_size": request.min_cluster_size,
                "min_samples": request.min_samples,
                "time_range_hours": request.time_range_hours,
            },
            clusters_info=[{
                "cluster_id": c.cluster_id,
                "size": c.size,
                "dominant_type": c.dominant_type.value,
                "avg_severity": c.avg_severity,
            } for c in clusters],
            recommendations=analysis_result.get("recommendations", []),
        )
        session.add(analysis_record)
        
        # Update defects with cluster assignments
        for cluster in analysis_result.get("clusters", []):
            for defect_id in cluster.defect_ids:
                update_query = (
                    select(Defect)
                    .where(Defect.id == defect_id)
                )
                defect_result = await session.execute(update_query)
                defect = defect_result.scalar_one_or_none()
                if defect:
                    defect.cluster_id = cluster.cluster_id
        
        await session.commit()
        
        return RootCauseResponse(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            total_defects_analyzed=len(defects_data),
            num_clusters=len(clusters),
            noise_points=analysis_result.get("noise_count", 0),
            clusters=clusters,
            recommendations=analysis_result.get("recommendations", []),
        )


@router.get("/history")
async def get_analysis_history(
    limit: int = Query(default=10, ge=1, le=50),
    db: DatabaseManager = Depends(get_database),
):
    """Get history of root cause analyses."""
    async for session in db.get_session():
        query = (
            select(ClusterAnalysis)
            .order_by(ClusterAnalysis.timestamp.desc())
            .limit(limit)
        )
        
        result = await session.execute(query)
        analyses = result.scalars().all()
        
        return {
            "analyses": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "total_defects": a.total_defects,
                    "num_clusters": a.num_clusters,
                    "noise_points": a.noise_points,
                    "parameters": a.parameters,
                }
                for a in analyses
            ]
        }


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    db: DatabaseManager = Depends(get_database),
):
    """Get details for a specific analysis."""
    async for session in db.get_session():
        query = select(ClusterAnalysis).where(ClusterAnalysis.id == analysis_id)
        result = await session.execute(query)
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "id": analysis.id,
            "timestamp": analysis.timestamp.isoformat(),
            "total_defects": analysis.total_defects,
            "num_clusters": analysis.num_clusters,
            "noise_points": analysis.noise_points,
            "algorithm": analysis.algorithm,
            "parameters": analysis.parameters,
            "clusters": analysis.clusters_info,
            "recommendations": analysis.recommendations,
        }


@router.get("/visualization/{analysis_id}")
async def get_visualization_data(
    analysis_id: str,
    db: DatabaseManager = Depends(get_database),
    analyzer: RootCauseAnalyzer = Depends(get_root_cause_analyzer),
):
    """
    Get 2D visualization data for cluster plotting.
    
    Uses UMAP to project high-dimensional features to 2D.
    """
    async for session in db.get_session():
        # Get analysis
        analysis_query = select(ClusterAnalysis).where(ClusterAnalysis.id == analysis_id)
        analysis_result = await session.execute(analysis_query)
        analysis = analysis_result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get defects from that time period
        # Parse time range from parameters
        time_range = analysis.parameters.get("time_range_hours", 24)
        end_time = analysis.timestamp
        start_time = end_time - timedelta(hours=time_range)
        
        defect_query = (
            select(Defect, Inspection.timestamp)
            .join(Inspection)
            .where(Inspection.timestamp >= start_time)
            .where(Inspection.timestamp <= end_time)
        )
        
        result = await session.execute(defect_query)
        rows = result.all()
        
        if not rows:
            return {"points": [], "labels": [], "clusters": []}
        
        # Prepare data
        defects_data = []
        for defect, timestamp in rows:
            defects_data.append({
                "id": defect.id,
                "centroid": (
                    (defect.bbox_x_min + defect.bbox_x_max) / 2,
                    (defect.bbox_y_min + defect.bbox_y_max) / 2,
                ),
                "defect_type": defect.defect_type,
                "severity_score": defect.severity_score,
                "cluster_id": defect.cluster_id,
                "latent": defect.feature_vector,
            })
        
        # Get visualization coordinates
        viz_data = analyzer.get_cluster_visualization_data(defects_data)
        
        # Add cluster assignments
        viz_data["cluster_ids"] = [d.get("cluster_id", -1) for d in defects_data]
        
        return viz_data
