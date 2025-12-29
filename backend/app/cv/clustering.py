"""
NeuroInspect - Root Cause Analysis via Clustering
Identifies patterns in defects to determine potential manufacturing issues.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from loguru import logger

# Suppress HDBSCAN warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ClusterResult:
    """Result for a single cluster of defects."""
    cluster_id: int
    size: int
    centroid: List[float]
    defect_ids: List[str]
    dominant_type: str
    type_distribution: Dict[str, int]
    avg_severity: float
    severity_distribution: Dict[str, int]
    time_range: Tuple[datetime, datetime]
    spatial_center: Tuple[float, float]
    potential_causes: List[str]
    confidence: float


class RootCauseAnalyzer:
    """
    Clusters defects to identify patterns indicating root causes.
    
    Uses HDBSCAN for density-based clustering:
    - Works without predefined cluster count
    - Handles noise (outlier defects)
    - Identifies varying density clusters
    
    Features used for clustering:
    - Defect location (x, y centroid)
    - Defect type (one-hot encoded)
    - Severity score
    - Time of detection (optional)
    - Feature vector from autoencoder latent space
    """
    
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self._clusterer = None
        self._umap_reducer = None
    
    def analyze(
        self,
        defects: List[Dict[str, Any]],
        use_latent_features: bool = True,
        time_weight: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform root cause analysis on defect dataset.
        
        Args:
            defects: List of defect dictionaries with:
                - id: Unique identifier
                - centroid: (x, y) normalized position
                - defect_type: Category string
                - severity_score: 0-1 severity
                - timestamp: datetime of detection
                - latent: Optional feature vector from autoencoder
            use_latent_features: Whether to include latent vectors
            time_weight: Weight for temporal clustering
        
        Returns:
            Dictionary with clustering results and recommendations
        """
        if len(defects) < self.min_cluster_size:
            return {
                "clusters": [],
                "noise_count": len(defects),
                "total_analyzed": len(defects),
                "recommendations": ["Insufficient defects for clustering analysis"],
            }
        
        # Extract and prepare features
        features, defect_types = self._prepare_features(
            defects, use_latent_features, time_weight
        )
        
        # Perform clustering
        try:
            import hdbscan
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = self._clusterer.fit_predict(features)
        except ImportError:
            logger.warning("HDBSCAN not available, using KMeans fallback")
            from sklearn.cluster import KMeans
            n_clusters = max(2, len(defects) // 10)
            self._clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = self._clusterer.fit_predict(features)
        
        # Analyze clusters
        clusters = self._analyze_clusters(defects, labels, features, defect_types)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(clusters, defects)
        
        # Count noise points
        noise_count = sum(1 for l in labels if l == -1)
        
        return {
            "clusters": clusters,
            "noise_count": noise_count,
            "total_analyzed": len(defects),
            "unique_cluster_count": len(set(labels)) - (1 if -1 in labels else 0),
            "recommendations": recommendations,
            "feature_dim": features.shape[1],
        }
    
    def _prepare_features(
        self,
        defects: List[Dict],
        use_latent: bool,
        time_weight: float
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for clustering."""
        features_list = []
        defect_types = []
        
        # Collect all unique defect types for one-hot encoding
        all_types = list(set(d.get("defect_type", "anomaly") for d in defects))
        type_to_idx = {t: i for i, t in enumerate(all_types)}
        
        for defect in defects:
            feat = []
            
            # Spatial features (x, y centroid)
            centroid = defect.get("centroid", (0.5, 0.5))
            feat.extend([centroid[0], centroid[1]])
            
            # Severity
            feat.append(defect.get("severity_score", 0.5))
            
            # Defect type (one-hot)
            type_onehot = [0] * len(all_types)
            dtype = defect.get("defect_type", "anomaly")
            if dtype in type_to_idx:
                type_onehot[type_to_idx[dtype]] = 1
            feat.extend(type_onehot)
            
            # Temporal feature (if available)
            if "timestamp" in defect and time_weight > 0:
                ts = defect["timestamp"]
                if isinstance(ts, datetime):
                    # Normalize to hours since epoch
                    hours = (ts - datetime(2024, 1, 1)).total_seconds() / 3600
                    feat.append(hours * time_weight)
            
            # Latent features from autoencoder
            if use_latent and "latent" in defect and defect["latent"] is not None:
                latent = np.array(defect["latent"])
                # Reduce dimensionality if too high
                if len(latent) > 16:
                    latent = latent[:16]  # Take first 16 dimensions
                feat.extend(latent.tolist())
            
            features_list.append(feat)
            defect_types.append(dtype)
        
        # Pad features to same length
        max_len = max(len(f) for f in features_list)
        features_list = [f + [0] * (max_len - len(f)) for f in features_list]
        
        # Normalize features
        features = np.array(features_list, dtype=np.float32)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return features, defect_types
    
    def _analyze_clusters(
        self,
        defects: List[Dict],
        labels: np.ndarray,
        features: np.ndarray,
        defect_types: List[str]
    ) -> List[ClusterResult]:
        """Analyze each cluster to extract insights."""
        clusters = []
        unique_labels = set(labels) - {-1}  # Exclude noise
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_defects = [d for d, m in zip(defects, mask) if m]
            cluster_types = [t for t, m in zip(defect_types, mask) if m]
            cluster_features = features[mask]
            
            # Calculate centroid in feature space
            centroid = cluster_features.mean(axis=0).tolist()
            
            # Type distribution
            type_dist = {}
            for t in cluster_types:
                type_dist[t] = type_dist.get(t, 0) + 1
            dominant_type = max(type_dist, key=type_dist.get)
            
            # Severity statistics
            severities = [d.get("severity_score", 0.5) for d in cluster_defects]
            avg_severity = float(np.mean(severities))
            
            severity_levels = [d.get("severity_level", "medium") for d in cluster_defects]
            severity_dist = {}
            for s in severity_levels:
                severity_dist[s] = severity_dist.get(s, 0) + 1
            
            # Time range
            timestamps = [d.get("timestamp") for d in cluster_defects if d.get("timestamp")]
            if timestamps:
                time_range = (min(timestamps), max(timestamps))
            else:
                time_range = (datetime.utcnow(), datetime.utcnow())
            
            # Spatial center (average of defect centroids)
            centroids = [d.get("centroid", (0.5, 0.5)) for d in cluster_defects]
            spatial_center = (
                float(np.mean([c[0] for c in centroids])),
                float(np.mean([c[1] for c in centroids]))
            )
            
            # Infer potential causes
            causes = self._infer_causes(
                dominant_type, type_dist, spatial_center, avg_severity, len(cluster_defects)
            )
            
            # Cluster confidence (based on density and consistency)
            confidence = self._calculate_cluster_confidence(
                cluster_features, type_dist, severities
            )
            
            clusters.append(ClusterResult(
                cluster_id=int(cluster_id),
                size=len(cluster_defects),
                centroid=centroid[:3],  # First 3 dims for visualization
                defect_ids=[d.get("id", "") for d in cluster_defects],
                dominant_type=dominant_type,
                type_distribution=type_dist,
                avg_severity=avg_severity,
                severity_distribution=severity_dist,
                time_range=time_range,
                spatial_center=spatial_center,
                potential_causes=causes,
                confidence=confidence,
            ))
        
        # Sort by size
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters
    
    def _infer_causes(
        self,
        dominant_type: str,
        type_dist: Dict[str, int],
        spatial_center: Tuple[float, float],
        avg_severity: float,
        cluster_size: int
    ) -> List[str]:
        """Infer potential root causes based on cluster characteristics."""
        causes = []
        
        # Type-based causes
        type_causes = {
            "scratch": ["Tool wear", "Material handling issue", "Fixture misalignment"],
            "dent": ["Impact during transport", "Press calibration issue", "Material collision"],
            "crack": ["Material fatigue", "Thermal stress", "Excessive force"],
            "stain": ["Contamination", "Cleaning process issue", "Material defect"],
            "corrosion": ["Environmental exposure", "Coating failure", "Material composition"],
            "deformation": ["Temperature variance", "Tooling wear", "Material properties"],
            "missing_part": ["Assembly error", "Supplier issue", "Station malfunction"],
            "foreign_object": ["Contamination", "Previous process debris", "Environment"],
            "surface_roughness": ["Tool condition", "Feed rate issue", "Material quality"],
        }
        
        if dominant_type.lower() in type_causes:
            causes.extend(type_causes[dominant_type.lower()])
        
        # Location-based inference
        x, y = spatial_center
        if x < 0.2:
            causes.append("Left edge alignment issue")
        elif x > 0.8:
            causes.append("Right edge processing variance")
        if y < 0.2 or y > 0.8:
            causes.append("Edge region sensitivity")
        if 0.4 < x < 0.6 and 0.4 < y < 0.6:
            causes.append("Center tooling pattern")
        
        # Size-based inference
        if cluster_size > 10:
            causes.append("Systematic process issue")
        
        # Severity-based inference
        if avg_severity > 0.7:
            causes.append("Major equipment malfunction")
        
        return causes[:5]  # Top 5 causes
    
    def _calculate_cluster_confidence(
        self,
        features: np.ndarray,
        type_dist: Dict[str, int],
        severities: List[float]
    ) -> float:
        """Calculate confidence score for cluster validity."""
        # Type homogeneity (higher = more confident)
        type_homogeneity = max(type_dist.values()) / sum(type_dist.values())
        
        # Feature compactness (lower variance = more confident)
        feature_variance = np.mean(np.var(features, axis=0))
        compactness = 1 / (1 + feature_variance)
        
        # Severity consistency
        severity_std = np.std(severities) if len(severities) > 1 else 0
        severity_consistency = 1 / (1 + severity_std)
        
        # Combined confidence
        confidence = 0.4 * type_homogeneity + 0.3 * compactness + 0.3 * severity_consistency
        return float(min(1.0, confidence))
    
    def _generate_recommendations(
        self,
        clusters: List[ClusterResult],
        defects: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not clusters:
            recommendations.append("No significant defect patterns detected. Continue monitoring.")
            return recommendations
        
        # Analyze top clusters
        for i, cluster in enumerate(clusters[:3]):
            rec = f"Cluster {cluster.cluster_id} ({cluster.size} defects, {cluster.dominant_type}): "
            if cluster.potential_causes:
                rec += f"Investigate {cluster.potential_causes[0].lower()}"
            recommendations.append(rec)
        
        # Overall recommendations
        total_defects = len(defects)
        clustered = sum(c.size for c in clusters)
        cluster_rate = clustered / total_defects if total_defects > 0 else 0
        
        if cluster_rate > 0.7:
            recommendations.append("High clustering rate indicates systematic issues. Prioritize process review.")
        
        # High severity clusters
        high_sev_clusters = [c for c in clusters if c.avg_severity > 0.6]
        if high_sev_clusters:
            recommendations.append(f"{len(high_sev_clusters)} high-severity clusters require immediate attention.")
        
        return recommendations
    
    def get_cluster_visualization_data(
        self,
        defects: List[Dict],
        use_umap: bool = True
    ) -> Dict[str, Any]:
        """
        Get 2D/3D coordinates for cluster visualization.
        Uses UMAP for dimensionality reduction.
        """
        if len(defects) < 3:
            return {"points": [], "labels": []}
        
        features, _ = self._prepare_features(defects, use_latent=True, time_weight=0.1)
        
        # Reduce to 2D for visualization
        try:
            if use_umap:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
                coords_2d = reducer.fit_transform(features)
            else:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                coords_2d = reducer.fit_transform(features)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(features)
        
        return {
            "points": coords_2d.tolist(),
            "defect_ids": [d.get("id", "") for d in defects],
            "types": [d.get("defect_type", "anomaly") for d in defects],
            "severities": [d.get("severity_score", 0.5) for d in defects],
        }
