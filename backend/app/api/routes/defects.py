"""
NeuroInspect - Defects API Routes
Endpoints for defect listing and analytics.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, desc
from loguru import logger

from app.models.schemas import (
    DefectListResponse,
    DefectSummary,
    SeverityLevel,
    DefectType,
)
from app.api.deps import get_database
from app.models.database import DatabaseManager, Defect, Inspection

router = APIRouter(prefix="/defects", tags=["Defects"])


@router.get("", response_model=DefectListResponse)
async def list_defects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    severity: Optional[str] = Query(default=None),
    defect_type: Optional[str] = Query(default=None),
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    sort_by: str = Query(default="created_at", pattern="^(created_at|severity_score|area_percentage)$"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    db: DatabaseManager = Depends(get_database),
):
    """
    List defects with filtering and pagination.
    
    - **page**: Page number (starting from 1)
    - **page_size**: Items per page (max 100)
    - **severity**: Filter by severity level
    - **defect_type**: Filter by defect type
    - **start_date**: Filter by detection date (from)
    - **end_date**: Filter by detection date (to)
    - **sort_by**: Sort field
    - **sort_order**: Sort direction
    """
    async for session in db.get_session():
        # Build query
        query = select(Defect).join(Inspection)
        
        # Apply filters
        if severity:
            query = query.where(Defect.severity == severity)
        if defect_type:
            query = query.where(Defect.defect_type == defect_type)
        if start_date:
            query = query.where(Inspection.timestamp >= start_date)
        if end_date:
            query = query.where(Inspection.timestamp <= end_date)
        
        # Get total count
        count_query = select(func.count(Defect.id)).join(Inspection)
        if severity:
            count_query = count_query.where(Defect.severity == severity)
        if defect_type:
            count_query = count_query.where(Defect.defect_type == defect_type)
        if start_date:
            count_query = count_query.where(Inspection.timestamp >= start_date)
        if end_date:
            count_query = count_query.where(Inspection.timestamp <= end_date)
        
        total_result = await session.execute(count_query)
        total_count = total_result.scalar() or 0
        
        # Apply sorting
        sort_column = getattr(Defect, sort_by, Defect.created_at)
        if sort_order == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute
        result = await session.execute(query)
        defects = result.scalars().all()
        
        # Convert to response format
        defect_list = []
        for d in defects:
            defect_list.append({
                "id": d.id,
                "inspection_id": d.inspection_id,
                "defect_type": d.defect_type,
                "confidence": d.confidence,
                "severity": d.severity,
                "severity_score": d.severity_score,
                "area_percentage": d.area_percentage,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })
        
        # Get summary
        summary = await _get_defect_summary(session, start_date, end_date)
        
        return DefectListResponse(
            summary=summary,
            defects=defect_list,
            page=page,
            page_size=page_size,
            total_pages=(total_count + page_size - 1) // page_size,
        )


@router.get("/summary", response_model=DefectSummary)
async def get_summary(
    hours: int = Query(default=24, ge=1, le=720),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get defect summary statistics.
    
    - **hours**: Time range in hours (default 24)
    """
    start_date = datetime.utcnow() - timedelta(hours=hours)
    
    async for session in db.get_session():
        return await _get_defect_summary(session, start_date)


@router.get("/trends")
async def get_trends(
    period: str = Query(default="day", pattern="^(hour|day|week)$"),
    limit: int = Query(default=30, ge=1, le=365),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get defect trends over time.
    
    - **period**: Aggregation period (hour, day, week)
    - **limit**: Number of periods to return
    """
    # Calculate time buckets based on period
    if period == "hour":
        interval = timedelta(hours=1)
    elif period == "day":
        interval = timedelta(days=1)
    else:
        interval = timedelta(weeks=1)
    
    now = datetime.utcnow()
    start_date = now - (interval * limit)
    
    async for session in db.get_session():
        # Get defects in time range
        query = (
            select(Defect, Inspection.timestamp)
            .join(Inspection)
            .where(Inspection.timestamp >= start_date)
            .order_by(Inspection.timestamp)
        )
        
        result = await session.execute(query)
        rows = result.all()
        
        # Aggregate by period
        buckets = {}
        for defect, timestamp in rows:
            if period == "hour":
                bucket_key = timestamp.strftime("%Y-%m-%d %H:00")
            elif period == "day":
                bucket_key = timestamp.strftime("%Y-%m-%d")
            else:
                # Week - use Monday as start
                week_start = timestamp - timedelta(days=timestamp.weekday())
                bucket_key = week_start.strftime("%Y-%m-%d")
            
            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "period": bucket_key,
                    "total": 0,
                    "by_severity": {},
                    "by_type": {},
                }
            
            buckets[bucket_key]["total"] += 1
            
            sev = defect.severity
            buckets[bucket_key]["by_severity"][sev] = buckets[bucket_key]["by_severity"].get(sev, 0) + 1
            
            dtype = defect.defect_type
            buckets[bucket_key]["by_type"][dtype] = buckets[bucket_key]["by_type"].get(dtype, 0) + 1
        
        # Fill in missing periods
        current = start_date
        filled_buckets = []
        while current <= now:
            if period == "hour":
                bucket_key = current.strftime("%Y-%m-%d %H:00")
            elif period == "day":
                bucket_key = current.strftime("%Y-%m-%d")
            else:
                week_start = current - timedelta(days=current.weekday())
                bucket_key = week_start.strftime("%Y-%m-%d")
            
            if bucket_key in buckets:
                filled_buckets.append(buckets[bucket_key])
            else:
                filled_buckets.append({
                    "period": bucket_key,
                    "total": 0,
                    "by_severity": {},
                    "by_type": {},
                })
            
            current += interval
        
        return {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": now.isoformat(),
            "data": filled_buckets[-limit:],
        }


@router.get("/{defect_id}")
async def get_defect(
    defect_id: str,
    db: DatabaseManager = Depends(get_database),
):
    """Get details for a specific defect."""
    async for session in db.get_session():
        query = select(Defect).where(Defect.id == defect_id)
        result = await session.execute(query)
        defect = result.scalar_one_or_none()
        
        if not defect:
            raise HTTPException(status_code=404, detail="Defect not found")
        
        return {
            "id": defect.id,
            "inspection_id": defect.inspection_id,
            "defect_type": defect.defect_type,
            "confidence": defect.confidence,
            "severity": defect.severity,
            "severity_score": defect.severity_score,
            "bbox": {
                "x_min": defect.bbox_x_min,
                "y_min": defect.bbox_y_min,
                "x_max": defect.bbox_x_max,
                "y_max": defect.bbox_y_max,
            },
            "area_percentage": defect.area_percentage,
            "pixel_count": defect.pixel_count,
            "cluster_id": defect.cluster_id,
            "created_at": defect.created_at.isoformat() if defect.created_at else None,
        }


async def _get_defect_summary(
    session,
    start_date: datetime = None,
    end_date: datetime = None,
) -> DefectSummary:
    """Calculate defect summary statistics."""
    # Base query
    query = select(Defect).join(Inspection)
    count_query = select(func.count(Defect.id)).join(Inspection)
    inspection_count = select(func.count(Inspection.id))
    
    if start_date:
        query = query.where(Inspection.timestamp >= start_date)
        count_query = count_query.where(Inspection.timestamp >= start_date)
        inspection_count = inspection_count.where(Inspection.timestamp >= start_date)
    if end_date:
        query = query.where(Inspection.timestamp <= end_date)
        count_query = count_query.where(Inspection.timestamp <= end_date)
        inspection_count = inspection_count.where(Inspection.timestamp <= end_date)
    
    # Get total counts
    total_result = await session.execute(count_query)
    total_defects = total_result.scalar() or 0
    
    inspection_result = await session.execute(inspection_count)
    total_inspections = inspection_result.scalar() or 0
    
    # Get all defects for aggregation
    result = await session.execute(query)
    defects = result.scalars().all()
    
    # Aggregate by type
    by_type = {}
    by_severity = {}
    total_severity = 0
    
    for d in defects:
        by_type[d.defect_type] = by_type.get(d.defect_type, 0) + 1
        by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
        total_severity += d.severity_score
    
    avg_severity = total_severity / len(defects) if defects else 0
    detection_rate = total_defects / total_inspections if total_inspections > 0 else 0
    
    return DefectSummary(
        total_inspections=total_inspections,
        total_defects=total_defects,
        defects_by_type=by_type,
        defects_by_severity=by_severity,
        avg_severity_score=avg_severity,
        detection_rate=detection_rate,
    )
