"""
Helios ML Configuration
Default configuration for ISR/QMV thresholds and model parameters.
"""

# ISR (Information Stability Ratio) Configuration
ISR_THRESHOLD = 1.5  # T ≥ 1.5
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
