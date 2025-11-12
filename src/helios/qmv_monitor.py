"""
QMV Monitor (Quality Metric Variance)
Ensures C < 0.03 for model quality consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QMVMetrics:
    """Container for QMV monitoring metrics."""
    timestamp: datetime
    qmv_value: float
    is_valid: bool
    threshold: float = 0.02
    metric_variances: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class QMVMonitor:
    """
    QMV Monitor for Helios ML Framework.
    
    Monitors Quality Metric Variance (QMV) with C < 0.03 threshold.
    Ensures model performance consistency across folds and iterations.
    
    Attributes:
        threshold (float): Maximum QMV threshold (default: 0.03)
        audit_trail (List[QMVMetrics]): Historical QMV monitoring records
    """
    
    def __init__(self, threshold: float = 0.02):
        """
        Initialize QMV Monitor.

        Args:
            threshold: Maximum QMV threshold (default: 0.02 for enhanced validation)
        """
        self.threshold = threshold
        self.audit_trail: List[QMVMetrics] = []
        self._performance_history = []
        
    def compute_qmv(
        self,
        scores: List[float],
        metric_name: str = "accuracy"
    ) -> float:
        """
        Compute Quality Metric Variance for a set of scores.
        
        The QMV measures the consistency of model performance.
        Lower QMV values indicate greater consistency.
        
        Args:
            scores: List of performance scores (e.g., from cross-validation)
            metric_name: Name of the metric being monitored
            
        Returns:
            QMV value (lower is more consistent)
        """
        if len(scores) < 2:
            return 0.0
            
        scores_array = np.array(scores)
        
        # Calculate coefficient of variation
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array, ddof=1)
        
        # Avoid division by zero
        if abs(mean_score) < 1e-10:
            return 1.0 if std_score > 1e-10 else 0.0
            
        # Coefficient of variation as QMV
        qmv_value = std_score / abs(mean_score)
        
        return qmv_value
    
    def compute_multi_metric_qmv(
        self,
        metric_scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute QMV for multiple metrics.
        
        Args:
            metric_scores: Dictionary mapping metric names to score lists
            
        Returns:
            Dictionary mapping metric names to QMV values
        """
        qmv_values = {}
        
        for metric_name, scores in metric_scores.items():
            qmv_values[metric_name] = self.compute_qmv(scores, metric_name)
            
        return qmv_values
    
    def validate(
        self,
        scores: List[float],
        metric_name: str = "accuracy",
        metadata: Optional[Dict] = None
    ) -> QMVMetrics:
        """
        Validate QMV for given scores and record in audit trail.
        
        Args:
            scores: List of performance scores
            metric_name: Name of the metric
            metadata: Optional metadata for audit trail
            
        Returns:
            QMVMetrics object with validation results
        """
        qmv_value = self.compute_qmv(scores, metric_name)
        
        # Create metrics object
        metrics = QMVMetrics(
            timestamp=datetime.now(),
            qmv_value=qmv_value,
            is_valid=qmv_value < self.threshold,
            threshold=self.threshold,
            metric_variances={metric_name: qmv_value},
            metadata=metadata or {}
        )
        
        # Add to audit trail
        self.audit_trail.append(metrics)
        self._performance_history.append({
            'scores': scores,
            'metric_name': metric_name,
            'qmv': qmv_value
        })
        
        return metrics
    
    def validate_multi_metric(
        self,
        metric_scores: Dict[str, List[float]],
        metadata: Optional[Dict] = None
    ) -> QMVMetrics:
        """
        Validate QMV for multiple metrics.
        
        Args:
            metric_scores: Dictionary mapping metric names to score lists
            metadata: Optional metadata for audit trail
            
        Returns:
            QMVMetrics object with validation results
        """
        qmv_values = self.compute_multi_metric_qmv(metric_scores)
        
        # Aggregate QMV (maximum variance across metrics)
        max_qmv = max(qmv_values.values()) if qmv_values else 0.0
        
        # Create metrics object
        metrics = QMVMetrics(
            timestamp=datetime.now(),
            qmv_value=max_qmv,
            is_valid=max_qmv < self.threshold,
            threshold=self.threshold,
            metric_variances=qmv_values,
            metadata=metadata or {}
        )
        
        # Add to audit trail
        self.audit_trail.append(metrics)
        
        return metrics
    
    def get_audit_summary(self) -> pd.DataFrame:
        """
        Get summary of QMV monitoring audit trail.
        
        Returns:
            DataFrame with audit trail summary
        """
        if not self.audit_trail:
            return pd.DataFrame()
            
        records = []
        for metrics in self.audit_trail:
            records.append({
                'timestamp': metrics.timestamp,
                'qmv_value': metrics.qmv_value,
                'is_valid': metrics.is_valid,
                'threshold': metrics.threshold,
                'num_metrics': len(metrics.metric_variances)
            })
            
        return pd.DataFrame(records)
    
    def export_audit_trail(self, filepath: str):
        """
        Export audit trail to file.
        
        Args:
            filepath: Path to save audit trail
        """
        summary = self.get_audit_summary()
        summary.to_csv(filepath, index=False)
        
    def check_threshold(self, qmv_value: float) -> bool:
        """
        Check if QMV value meets threshold.
        
        Args:
            qmv_value: QMV value to check
            
        Returns:
            True if QMV < threshold
        """
        return qmv_value < self.threshold
    
    def get_performance_statistics(self) -> Dict:
        """
        Get statistical summary of performance history.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._performance_history:
            return {}
            
        stats = {
            'total_validations': len(self._performance_history),
            'valid_count': sum(1 for m in self.audit_trail if m.is_valid),
            'invalid_count': sum(1 for m in self.audit_trail if not m.is_valid),
            'mean_qmv': np.mean([m.qmv_value for m in self.audit_trail]),
            'max_qmv': np.max([m.qmv_value for m in self.audit_trail]),
            'min_qmv': np.min([m.qmv_value for m in self.audit_trail])
        }
        
        return stats
