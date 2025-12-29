"""
NeuroInspect - Severity Scoring System
Multi-factor severity assessment for detected defects.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class SeverityLevel(str, Enum):
    """Severity classification levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SeverityConfig:
    """Configuration for severity scoring."""
    # Weight factors (must sum to 1.0)
    weight_area: float = 0.4
    weight_intensity: float = 0.3
    weight_location: float = 0.3
    
    # Thresholds for severity levels
    threshold_critical: float = 0.85
    threshold_high: float = 0.65
    threshold_medium: float = 0.40
    threshold_low: float = 0.20
    
    # Critical zones (normalized coordinates)
    # Format: [(x_min, y_min, x_max, y_max, multiplier), ...]
    critical_zones: List[Tuple[float, float, float, float, float]] = None
    
    def __post_init__(self):
        if self.critical_zones is None:
            # Default: center region is more critical (common in industrial inspection)
            self.critical_zones = [
                (0.25, 0.25, 0.75, 0.75, 1.5),  # Center zone, 1.5x severity
            ]


class SeverityScorer:
    """
    Calculates defect severity using multiple factors:
    1. Area: Larger defects are more severe
    2. Intensity: Higher anomaly scores indicate more severe defects
    3. Location: Defects in critical zones are weighted higher
    
    Also supports rule-based adjustments for domain-specific requirements.
    """
    
    def __init__(self, config: SeverityConfig = None):
        self.config = config or SeverityConfig()
        
        # Validate weights sum to 1
        total_weight = (self.config.weight_area + 
                       self.config.weight_intensity + 
                       self.config.weight_location)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Severity weights sum to {total_weight}, normalizing...")
            self.config.weight_area /= total_weight
            self.config.weight_intensity /= total_weight
            self.config.weight_location /= total_weight
    
    def calculate_severity(
        self,
        area_percentage: float,
        max_intensity: float,
        centroid: Tuple[float, float],
        defect_type: str = None,
        custom_rules: Dict = None
    ) -> Dict[str, any]:
        """
        Calculate severity score for a defect.
        
        Args:
            area_percentage: Defect area as percentage of total image (0-100)
            max_intensity: Maximum anomaly intensity in defect region (0-1)
            centroid: Defect centroid (normalized x, y)
            defect_type: Optional defect type for rule-based adjustments
            custom_rules: Optional custom scoring rules
        
        Returns:
            Dictionary with:
            - score: Overall severity score (0-1)
            - level: SeverityLevel enum
            - components: Individual factor scores
            - adjustments: Any rule-based adjustments applied
        """
        # Calculate component scores
        area_score = self._calculate_area_score(area_percentage)
        intensity_score = self._calculate_intensity_score(max_intensity)
        location_score = self._calculate_location_score(centroid)
        
        # Weighted combination
        base_score = (
            area_score * self.config.weight_area +
            intensity_score * self.config.weight_intensity +
            location_score * self.config.weight_location
        )
        
        # Apply rule-based adjustments
        adjustments = []
        final_score = base_score
        
        if defect_type:
            adjustment = self._apply_type_rules(defect_type)
            if adjustment != 1.0:
                adjustments.append(f"Type '{defect_type}': {adjustment}x")
                final_score *= adjustment
        
        if custom_rules:
            rule_adjustment = self._apply_custom_rules(custom_rules, area_percentage, max_intensity)
            if rule_adjustment != 1.0:
                adjustments.append(f"Custom rules: {rule_adjustment}x")
                final_score *= rule_adjustment
        
        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        # Determine severity level
        level = self._score_to_level(final_score)
        
        return {
            "score": final_score,
            "level": level,
            "components": {
                "area": area_score,
                "intensity": intensity_score,
                "location": location_score,
            },
            "adjustments": adjustments,
        }
    
    def _calculate_area_score(self, area_percentage: float) -> float:
        """
        Convert area percentage to severity score.
        Uses sigmoid-like curve: small defects score low, large defects saturate.
        """
        # Parameters for the curve
        midpoint = 5.0  # Area % where score is 0.5
        steepness = 0.5  # How sharp the transition is
        
        # Sigmoid transformation
        score = 1 / (1 + np.exp(-steepness * (area_percentage - midpoint)))
        return float(score)
    
    def _calculate_intensity_score(self, max_intensity: float) -> float:
        """
        Convert anomaly intensity to severity score.
        Higher intensity = more severe.
        """
        # Apply slight non-linearity to emphasize high intensities
        score = max_intensity ** 0.8
        return float(min(1.0, score))
    
    def _calculate_location_score(self, centroid: Tuple[float, float]) -> float:
        """
        Calculate location-based severity multiplier.
        Defects in critical zones score higher.
        """
        cx, cy = centroid
        max_multiplier = 1.0
        
        for zone in self.config.critical_zones:
            x_min, y_min, x_max, y_max, multiplier = zone
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                max_multiplier = max(max_multiplier, multiplier)
        
        # Normalize to [0, 1] range
        # Assumes multipliers are in [1.0, 2.0] range
        return float(min(1.0, (max_multiplier - 1.0) * 2))
    
    def _apply_type_rules(self, defect_type: str) -> float:
        """
        Apply type-specific severity adjustments.
        Returns multiplier (1.0 = no change).
        """
        # Industrial defect type severity multipliers
        type_multipliers = {
            "crack": 1.5,        # Cracks are serious structural issues
            "corrosion": 1.4,   # Corrosion indicates degradation
            "missing_part": 1.6, # Missing parts are critical
            "deformation": 1.3,  # Deformation affects function
            "scratch": 0.8,      # Scratches often cosmetic
            "stain": 0.6,        # Stains usually minor
            "dent": 0.9,         # Dents vary in severity
            "foreign_object": 1.2,
            "surface_roughness": 0.7,
            "anomaly": 1.0,      # Generic, no adjustment
        }
        return type_multipliers.get(defect_type.lower(), 1.0)
    
    def _apply_custom_rules(self, rules: Dict, area: float, intensity: float) -> float:
        """Apply custom rule-based adjustments."""
        multiplier = 1.0
        
        # Example custom rules
        if rules.get("reject_large_defects") and area > 10:
            multiplier *= 2.0
        
        if rules.get("high_precision_mode") and intensity > 0.8:
            multiplier *= 1.5
        
        return multiplier
    
    def _score_to_level(self, score: float) -> SeverityLevel:
        """Convert numeric score to severity level."""
        if score >= self.config.threshold_critical:
            return SeverityLevel.CRITICAL
        elif score >= self.config.threshold_high:
            return SeverityLevel.HIGH
        elif score >= self.config.threshold_medium:
            return SeverityLevel.MEDIUM
        elif score >= self.config.threshold_low:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO
    
    def batch_score(self, defects: List[Dict]) -> List[Dict]:
        """
        Score multiple defects at once.
        
        Args:
            defects: List of defect dictionaries with required fields
        
        Returns:
            List of severity results
        """
        results = []
        for defect in defects:
            result = self.calculate_severity(
                area_percentage=defect.get("area_percentage", 0),
                max_intensity=defect.get("max_intensity", 0),
                centroid=defect.get("centroid", (0.5, 0.5)),
                defect_type=defect.get("defect_type"),
            )
            results.append(result)
        return results
    
    def update_weights(self, area: float = None, intensity: float = None, 
                       location: float = None) -> None:
        """Update severity weight factors."""
        if area is not None:
            self.config.weight_area = area
        if intensity is not None:
            self.config.weight_intensity = intensity
        if location is not None:
            self.config.weight_location = location
        
        # Normalize
        total = (self.config.weight_area + 
                self.config.weight_intensity + 
                self.config.weight_location)
        self.config.weight_area /= total
        self.config.weight_intensity /= total
        self.config.weight_location /= total
        
        logger.info(f"Severity weights updated: area={self.config.weight_area:.2f}, "
                   f"intensity={self.config.weight_intensity:.2f}, "
                   f"location={self.config.weight_location:.2f}")
