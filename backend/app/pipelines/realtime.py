"""
NeuroInspect - Real-time Processing Pipeline
Low-latency streaming inference for live inspection.
"""
import asyncio
import time
from typing import Optional, Callable, Dict, Any, AsyncGenerator
from datetime import datetime
import uuid
from loguru import logger

from app.cv.detector import DefectDetector
from app.cv.localizer import DefectLocalizer
from app.cv.severity import SeverityScorer
from app.utils.video_utils import VideoProcessor, FrameBuffer, annotate_frame
from app.utils.image_utils import resize_image


class RealtimePipeline:
    """
    Pipeline for real-time video stream processing.
    Optimized for low latency and consistent frame rates.
    """
    
    def __init__(
        self,
        detector: DefectDetector,
        localizer: DefectLocalizer,
        scorer: SeverityScorer,
        target_fps: float = 30.0,
        skip_frames: int = 0,
        buffer_size: int = 30,
    ):
        self.detector = detector
        self.localizer = localizer
        self.scorer = scorer
        self.target_fps = target_fps
        self.skip_frames = skip_frames
        self.buffer = FrameBuffer(max_size=buffer_size)
        
        self._running = False
        self._frame_count = 0
        self._defect_count = 0
        self._start_time: Optional[float] = None
    
    async def process_stream(
        self,
        source: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_frames: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process video stream in real-time.
        
        Args:
            source: Video source (file path or camera index)
            callback: Optional callback for each result
            max_frames: Maximum frames to process (None = unlimited)
        
        Yields:
            Processing results for each frame
        """
        self._running = True
        self._frame_count = 0
        self._defect_count = 0
        self._start_time = time.time()
        
        frame_time = 1.0 / self.target_fps
        skip_counter = 0
        
        try:
            with VideoProcessor(source) as video:
                logger.info(f"Started real-time processing: {video.get_info()}")
                
                while self._running:
                    # Check frame limit
                    if max_frames and self._frame_count >= max_frames:
                        break
                    
                    # Read frame
                    frame = video.read_frame()
                    if frame is None:
                        if video.is_streaming:
                            await asyncio.sleep(0.01)
                            continue
                        else:
                            break
                    
                    # Skip frames if needed
                    skip_counter += 1
                    if skip_counter <= self.skip_frames:
                        continue
                    skip_counter = 0
                    
                    # Process frame
                    process_start = time.time()
                    result = await self._process_frame(frame, self._frame_count)
                    process_time = time.time() - process_start
                    
                    # Update statistics
                    self._frame_count += 1
                    self._defect_count += result.get("defect_count", 0)
                    
                    # Add to buffer
                    self.buffer.add(frame)
                    
                    # Callback
                    if callback:
                        callback(result)
                    
                    yield result
                    
                    # Rate limiting
                    sleep_time = frame_time - process_time
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        
        finally:
            self._running = False
            logger.info(f"Real-time processing stopped: {self.get_statistics()}")
    
    async def _process_frame(self, frame, frame_idx: int) -> Dict[str, Any]:
        """Process a single frame."""
        # Resize for faster processing
        frame_resized, scale = resize_image(frame, max_size=512)
        
        # Detect anomalies
        detection_result = self.detector.detect(frame_resized)
        
        # Localize if defective
        defects = []
        if detection_result["is_defective"]:
            localization = self.localizer.localize(
                detection_result["error_map"],
                original_image=frame_resized
            )
            
            for region in localization["regions"]:
                severity = self.scorer.calculate_severity(
                    area_percentage=region.area_percentage,
                    max_intensity=region.max_intensity,
                    centroid=region.centroid,
                )
                
                defects.append({
                    "id": str(uuid.uuid4())[:8],
                    "bbox": {
                        "x_min": region.bbox[0],
                        "y_min": region.bbox[1],
                        "x_max": region.bbox[2],
                        "y_max": region.bbox[3],
                    },
                    "confidence": region.mean_intensity,
                    "severity": severity["level"].value,
                    "severity_score": severity["score"],
                })
        
        return {
            "frame_idx": frame_idx,
            "timestamp": datetime.utcnow().isoformat(),
            "is_defective": detection_result["is_defective"],
            "anomaly_score": detection_result["anomaly_score"],
            "defect_count": len(defects),
            "defects": defects,
        }
    
    def stop(self) -> None:
        """Stop the processing loop."""
        self._running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if self._start_time is None:
            return {"status": "not started"}
        
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "status": "running" if self._running else "stopped",
            "frames_processed": self._frame_count,
            "total_defects": self._defect_count,
            "elapsed_seconds": elapsed,
            "actual_fps": fps,
            "target_fps": self.target_fps,
        }
    
    def get_annotated_frame(self, result: Dict[str, Any]) -> Optional[Any]:
        """Get latest frame with detection annotations."""
        frame = self.buffer.get_latest()
        if frame is None:
            return None
        
        return annotate_frame(frame, result.get("defects", []))


class StreamManager:
    """
    Manages multiple real-time processing streams.
    Useful for multi-camera inspection setups.
    """
    
    def __init__(self):
        self.streams: Dict[str, RealtimePipeline] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def add_stream(
        self,
        stream_id: str,
        source: str,
        pipeline: RealtimePipeline,
        callback: Optional[Callable] = None,
    ) -> None:
        """Add and start a new stream."""
        if stream_id in self.streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        self.streams[stream_id] = pipeline
        
        async def run_stream():
            async for result in pipeline.process_stream(source, callback):
                pass  # Results handled by callback
        
        self.tasks[stream_id] = asyncio.create_task(run_stream())
        logger.info(f"Started stream: {stream_id}")
    
    async def remove_stream(self, stream_id: str) -> None:
        """Stop and remove a stream."""
        if stream_id not in self.streams:
            return
        
        self.streams[stream_id].stop()
        
        if stream_id in self.tasks:
            self.tasks[stream_id].cancel()
            try:
                await self.tasks[stream_id]
            except asyncio.CancelledError:
                pass
            del self.tasks[stream_id]
        
        del self.streams[stream_id]
        logger.info(f"Removed stream: {stream_id}")
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all streams."""
        return {
            sid: stream.get_statistics()
            for sid, stream in self.streams.items()
        }
    
    async def stop_all(self) -> None:
        """Stop all streams."""
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            await self.remove_stream(stream_id)
