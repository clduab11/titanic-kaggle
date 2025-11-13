"""
Helios ML Configuration
Default configuration for ISR/QMV thresholds and model parameters.
"""

# ISR (Information Stability Ratio) Configuration
# The threshold of 1.5 is appropriate for datasets with 500-1000+ samples.
# For smaller datasets, the ISR validator includes automatic sample-size adjustments
# that make the metric more forgiving by applying a multiplicative factor.
# The adjustment accounts for higher natural variance in small datasets like Titanic (891 samples).
# See ISRValidator class in src/helios/isr_validator.py for details on the adjustment formula.
ISR_THRESHOLD = 1.5  # T ≥ 1.5

# Enable automatic sample-size adjustment for datasets with fewer than 1000 samples
ISR_SAMPLE_SIZE_ADJUSTMENT_ENABLED = True

ISR_VALIDATION_ENABLED = True

# QMV (Quality Metric Variance) Configuration
QMV_THRESHOLD = 0.03  # C < 0.03
QMV_VALIDATION_ENABLED = True

# Cross-Validation Configuration
CV_N_FOLDS = 5
CV_STRATIFIED = True
CV_RANDOM_STATE = 42

# Adversarial Validation Configuration
ADV_VALIDATION_ENABLED = True
ADV_THRESHOLD = 0.75  # AUC threshold for distribution shift detection

# Ensemble Configuration
ENSEMBLE_VOTING = 'weighted'  # Options: 'weighted', 'majority', 'soft'
ENSEMBLE_MODELS = [
    'random_forest',
    'xgboost',
    'lightgbm',
    'logistic',
    'gradient_boosting'
]

# Feature Engineering Configuration
FEATURE_SCALING = True
FEATURE_SELECTION_ENABLED = True

# Audit Trail Configuration
AUDIT_TRAIL_ENABLED = True
AUDIT_TRAIL_DIR = 'audit_trails'

# Target Performance
TARGET_ACCURACY_MIN = 0.78
TARGET_ACCURACY_MAX = 0.82

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_DIR = 'logs'
