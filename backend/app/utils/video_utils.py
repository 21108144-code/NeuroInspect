"""
NeuroInspect - Video Processing Utilities
Frame extraction and video handling for inspection pipelines.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Dict, Any
from loguru import logger


class VideoProcessor:
    """
    Video frame extraction and processing for inspection.
    Supports file-based and streaming sources.
    """
    
    def __init__(self, source: str = None):
        """
        Args:
            source: Video file path or camera index (as string)
        """
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.is_streaming = False
    
    def open(self, source: str = None) -> bool:
        """Open video source."""
        if source:
            self.source = source
        
        if self.source is None:
            raise ValueError("No video source specified")
        
        # Check if camera index
        try:
            camera_idx = int(self.source)
            self.capture = cv2.VideoCapture(camera_idx)
            self.is_streaming = True
        except ValueError:
            # File path
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")
            self.capture = cv2.VideoCapture(self.source)
            self.is_streaming = False
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Get properties
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Opened video: {self.width}x{self.height} @ {self.fps}fps, "
                   f"{self.frame_count if not self.is_streaming else 'streaming'} frames")
        return True
    
    def close(self) -> None:
        """Release video capture."""
        if self.capture:
            self.capture.release()
            self.capture = None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read single frame."""
        if not self.capture or not self.capture.isOpened():
            return None
        
        ret, frame = self.capture.read()
        return frame if ret else None
    
    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: int = None,
        interval: int = 1,
        max_frames: int = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end of video)
            interval: Extract every Nth frame
            max_frames: Maximum frames to extract
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        if not self.capture:
            self.open()
        
        # Set start position
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        extracted = 0
        
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            if end_frame and frame_idx > end_frame:
                break
            
            if max_frames and extracted >= max_frames:
                break
            
            if (frame_idx - start_frame) % interval == 0:
                yield frame_idx, frame
                extracted += 1
            
            frame_idx += 1
    
    def extract_keyframes(
        self,
        threshold: float = 30.0,
        max_frames: int = 100
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract keyframes based on scene change detection.
        
        Args:
            threshold: Scene change threshold (lower = more keyframes)
            max_frames: Maximum keyframes to extract
        
        Returns:
            List of (frame_index, frame_array) tuples
        """
        if not self.capture:
            self.open()
        
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        keyframes = []
        prev_frame = None
        frame_idx = 0
        
        while len(keyframes) < max_frames:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            if prev_frame is None:
                # First frame is always a keyframe
                keyframes.append((frame_idx, frame.copy()))
            else:
                # Compute frame difference
                diff = cv2.absdiff(frame, prev_frame)
                diff_score = np.mean(diff)
                
                if diff_score > threshold:
                    keyframes.append((frame_idx, frame.copy()))
            
            prev_frame = frame.copy()
            frame_idx += 1
        
        logger.info(f"Extracted {len(keyframes)} keyframes from {frame_idx} total frames")
        return keyframes
    
    def get_info(self) -> Dict[str, Any]:
        """Get video information."""
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.frame_count / self.fps if self.fps > 0 else 0,
            "is_streaming": self.is_streaming,
        }
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FrameBuffer:
    """
    Fixed-size frame buffer for real-time processing.
    Useful for temporal analysis and motion detection.
    """
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.frames: List[np.ndarray] = []
        self.timestamps: List[float] = []
    
    def add(self, frame: np.ndarray, timestamp: float = None) -> None:
        """Add frame to buffer."""
        import time
        
        self.frames.append(frame)
        self.timestamps.append(timestamp or time.time())
        
        # Remove oldest if buffer full
        while len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.timestamps.pop(0)
    
    def get_latest(self) -> Optional[np.ndarray]:
        """Get most recent frame."""
        return self.frames[-1] if self.frames else None
    
    def get_all(self) -> List[np.ndarray]:
        """Get all frames in buffer."""
        return self.frames.copy()
    
    def clear(self) -> None:
        """Clear buffer."""
        self.frames.clear()
        self.timestamps.clear()
    
    def compute_motion(self, threshold: int = 25) -> Optional[np.ndarray]:
        """
        Compute motion mask from buffer.
        Uses background subtraction.
        """
        if len(self.frames) < 2:
            return None
        
        # Use MOG2 background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=len(self.frames),
            varThreshold=threshold,
            detectShadows=False
        )
        
        for frame in self.frames:
            fg_mask = bg_subtractor.apply(frame)
        
        return fg_mask
    
    def __len__(self) -> int:
        return len(self.frames)


def create_video_writer(
    output_path: str,
    fps: float = 30.0,
    size: Tuple[int, int] = (640, 480),
    codec: str = "mp4v"
) -> cv2.VideoWriter:
    """Create video writer for saving inspection results."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, size)


def annotate_frame(
    frame: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Annotate frame with detection results.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries with bbox, label, confidence
        show_labels: Draw class labels
        show_confidence: Show confidence scores
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    for det in detections:
        bbox = det.get("bbox", det.get("bounding_box", {}))
        
        # Handle both normalized and pixel coordinates
        x1 = int(bbox.get("x_min", 0) * w if bbox.get("x_min", 0) <= 1 else bbox.get("x_min", 0))
        y1 = int(bbox.get("y_min", 0) * h if bbox.get("y_min", 0) <= 1 else bbox.get("y_min", 0))
        x2 = int(bbox.get("x_max", 0) * w if bbox.get("x_max", 0) <= 1 else bbox.get("x_max", 0))
        y2 = int(bbox.get("y_max", 0) * h if bbox.get("y_max", 0) <= 1 else bbox.get("y_max", 0))
        
        # Color based on severity
        severity = det.get("severity", "medium")
        colors = {
            "critical": (0, 0, 255),    # Red
            "high": (0, 127, 255),      # Orange
            "medium": (0, 255, 255),    # Yellow
            "low": (0, 255, 0),         # Green
            "info": (255, 255, 0),      # Cyan
        }
        color = colors.get(severity, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels or show_confidence:
            label_parts = []
            if show_labels:
                label_parts.append(det.get("defect_type", det.get("label", "defect")))
            if show_confidence:
                conf = det.get("confidence", 0)
                label_parts.append(f"{conf:.0%}")
            
            label = " ".join(label_parts)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated
