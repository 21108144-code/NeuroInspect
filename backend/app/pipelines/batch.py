"""
NeuroInspect - Batch Processing Pipeline
High-throughput batch inference for industrial inspection.
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import uuid
from loguru import logger

from app.cv.detector import DefectDetector
from app.cv.localizer import DefectLocalizer
from app.cv.severity import SeverityScorer
from app.utils.image_utils import load_image, resize_image


class BatchPipeline:
    """
    Pipeline for batch processing multiple images.
    Optimized for throughput over latency.
    """
    
    def __init__(
        self,
        detector: DefectDetector,
        localizer: DefectLocalizer,
        scorer: SeverityScorer,
        batch_size: int = 16,
        max_workers: int = 4,
    ):
        self.detector = detector
        self.localizer = localizer
        self.scorer = scorer
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = None,
        recursive: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Path to input directory
            output_dir: Optional path for result outputs
            extensions: Image file extensions to process
            recursive: Whether to search subdirectories
        
        Yields:
            Processing results for each image
        """
        extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find all image files
        pattern = "**/*" if recursive else "*"
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"{pattern}{ext}"))
            image_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_results = self._process_batch(batch_files)
            
            for result in batch_results:
                if output_dir:
                    self._save_result(result, output_path)
                yield result
    
    def _process_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self._process_single(image_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                results.append({
                    "path": str(image_path),
                    "success": False,
                    "error": str(e),
                })
        
        return results
    
    def _process_single(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image."""
        import time
        start_time = time.time()
        
        # Load and preprocess
        image = load_image(str(image_path))
        image, scale = resize_image(image, max_size=1024)
        
        # Detect
        detection_result = self.detector.detect(image)
        
        # Localize
        localization = self.localizer.localize(
            detection_result["error_map"],
            original_image=image
        )
        
        # Score severity for each region
        defects = []
        for region in localization["regions"]:
            severity = self.scorer.calculate_severity(
                area_percentage=region.area_percentage,
                max_intensity=region.max_intensity,
                centroid=region.centroid,
            )
            
            defects.append({
                "id": str(uuid.uuid4())[:8],
                "bbox": region.bbox,
                "area_percentage": region.area_percentage,
                "confidence": region.mean_intensity,
                "severity": severity["level"].value,
                "severity_score": severity["score"],
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "path": str(image_path),
            "success": True,
            "inspection_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "is_defective": detection_result["is_defective"],
            "anomaly_score": detection_result["anomaly_score"],
            "defect_count": len(defects),
            "defects": defects,
            "processing_time_ms": processing_time,
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
        }
    
    def _save_result(self, result: Dict[str, Any], output_dir: Path) -> None:
        """Save processing result to output directory."""
        import json
        
        if not result.get("success"):
            return
        
        # Create result file
        result_path = output_dir / f"{result['inspection_id']}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from batch results."""
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
            }
        
        defective = [r for r in successful if r.get("is_defective")]
        total_defects = sum(r.get("defect_count", 0) for r in successful)
        processing_times = [r.get("processing_time_ms", 0) for r in successful]
        
        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "defective_count": len(defective),
            "defective_rate": len(defective) / len(successful),
            "total_defects": total_defects,
            "avg_defects_per_image": total_defects / len(successful),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
        }
