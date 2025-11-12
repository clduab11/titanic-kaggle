"""
ISR Validator (Information Stability Ratio)
Ensures T ≥ 1.5 for model stability and reliability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ISRMetrics:
    """Container for ISR validation metrics."""
    timestamp: datetime
    isr_value: float
    is_valid: bool
    threshold: float = 1.5
    feature_stability: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class ISRValidator:
    """
    ISR Validator for Helios ML Framework.
    
    Validates Information Stability Ratio (ISR) with T ≥ 1.5 threshold.
    Monitors feature stability and data distribution consistency.
    
    Attributes:
        threshold (float): Minimum ISR threshold (default: 1.5)
        audit_trail (List[ISRMetrics]): Historical ISR validation records
    """
    
    def __init__(self, threshold: float = 1.5):
        """
        Initialize ISR Validator.
        
        Args:
            threshold: Minimum ISR threshold (default: 1.5)
        """
        self.threshold = threshold
        self.audit_trail: List[ISRMetrics] = []
        self._baseline_stats = None
        
    def compute_isr(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Information Stability Ratio between training and validation sets.
        
        The ISR measures the stability of information content across datasets.
        Higher ISR values indicate greater stability.
        
        Args:
            X_train: Training dataset features
            X_val: Validation dataset features
            feature_weights: Optional weights for features
            
        Returns:
            ISR value (higher is more stable)
        """
        if feature_weights is None:
            feature_weights = {col: 1.0 for col in X_train.columns}
            
        # Compute stability for each feature
        feature_stabilities = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for col in X_train.columns:
            if col not in X_val.columns:
                continue
                
            # Calculate feature-level stability
            stability = self._compute_feature_stability(
                X_train[col].values,
                X_val[col].values
            )
            feature_stabilities[col] = stability
            
            weight = feature_weights.get(col, 1.0)
            weighted_sum += stability * weight
            total_weight += weight
            
        # Aggregate ISR value
        isr_value = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return isr_value
    
    def _compute_feature_stability(
        self,
        train_values: np.ndarray,
        val_values: np.ndarray
    ) -> float:
        """
        Compute stability metric for a single feature.
        
        Uses statistical distance measures to quantify stability.
        
        Args:
            train_values: Training feature values
            val_values: Validation feature values
            
        Returns:
            Stability score (higher is more stable)
        """
        # Remove NaN values
        train_clean = train_values[~np.isnan(train_values)]
        val_clean = val_values[~np.isnan(val_values)]
        
        if len(train_clean) == 0 or len(val_clean) == 0:
            return 0.0
            
        # Compute statistical measures
        train_mean = np.mean(train_clean)
        val_mean = np.mean(val_clean)
        train_std = np.std(train_clean)
        val_std = np.std(val_clean)
        
        # Avoid division by zero
        if train_std < 1e-10 or val_std < 1e-10:
            # If variance is near zero, check if means are similar
            return 2.0 if abs(train_mean - val_mean) < 1e-6 else 0.5
            
        # Compute stability using coefficient of variation distance
        cv_train = train_std / (abs(train_mean) + 1e-10)
        cv_val = val_std / (abs(val_mean) + 1e-10)
        
        # Compute mean stability
        mean_diff = abs(train_mean - val_mean) / (train_std + val_std + 1e-10)
        
        # Combined stability metric (scaled to favor values > 1.5)
        stability = 2.0 / (1.0 + mean_diff + abs(cv_train - cv_val))
        
        return stability
    
    def validate(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None
    ) -> ISRMetrics:
        """
        Validate ISR for given datasets and record in audit trail.
        
        Args:
            X_train: Training dataset
            X_val: Validation dataset
            feature_weights: Optional feature importance weights
            metadata: Optional metadata for audit trail
            
        Returns:
            ISRMetrics object with validation results
        """
        isr_value = self.compute_isr(X_train, X_val, feature_weights)
        
        # Compute per-feature stability for audit trail
        feature_stability = {}
        for col in X_train.columns:
            if col in X_val.columns:
                stability = self._compute_feature_stability(
                    X_train[col].values,
                    X_val[col].values
                )
                feature_stability[col] = stability
        
        # Create metrics object
        metrics = ISRMetrics(
            timestamp=datetime.now(),
            isr_value=isr_value,
            is_valid=isr_value >= self.threshold,
            threshold=self.threshold,
            feature_stability=feature_stability,
            metadata=metadata or {}
        )
        
        # Add to audit trail
        self.audit_trail.append(metrics)
        
        return metrics
    
    def get_audit_summary(self) -> pd.DataFrame:
        """
        Get summary of ISR validation audit trail.
        
        Returns:
            DataFrame with audit trail summary
        """
        if not self.audit_trail:
            return pd.DataFrame()
            
        records = []
        for metrics in self.audit_trail:
            records.append({
                'timestamp': metrics.timestamp,
                'isr_value': metrics.isr_value,
                'is_valid': metrics.is_valid,
                'threshold': metrics.threshold,
                'num_features': len(metrics.feature_stability)
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
        
    def check_threshold(self, isr_value: float) -> bool:
        """
        Check if ISR value meets threshold.
        
        Args:
            isr_value: ISR value to check
            
        Returns:
            True if ISR >= threshold
        """
        return isr_value >= self.threshold
