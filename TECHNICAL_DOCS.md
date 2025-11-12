# Helios ML Framework - Technical Documentation

## Overview

Helios ML is an advanced machine learning framework implementing rigorous validation and monitoring standards for the Kaggle Titanic competition. The framework enforces Information Stability Ratio (ISR) governance (T≥1.5) and Quality Metric Variance (QMV) monitoring (C<0.03).

## Architecture

### Core Components

#### 1. ISR Validator (`src/helios/isr_validator.py`)

**Purpose**: Ensures data stability between training and validation sets.

**Key Methods**:
- `compute_isr(X_train, X_val, feature_weights)`: Computes ISR value
- `validate(X_train, X_val, feature_weights, metadata)`: Validates and records ISR
- `export_audit_trail(filepath)`: Exports validation history

**ISR Calculation**:
```
ISR = weighted_average(feature_stabilities)

feature_stability = 2.0 / (1.0 + mean_diff + cv_diff)

where:
  mean_diff = |μ_train - μ_val| / (σ_train + σ_val)
  cv_diff = |CV_train - CV_val|
  CV = coefficient of variation
```

**Threshold**: T ≥ 1.5

**Audit Trail**: Records timestamp, ISR value, validity, per-feature stability

#### 2. QMV Monitor (`src/helios/qmv_monitor.py`)

**Purpose**: Monitors consistency of model performance across cross-validation folds.

**Key Methods**:
- `compute_qmv(scores, metric_name)`: Computes QMV for given scores
- `validate(scores, metric_name, metadata)`: Validates and records QMV
- `compute_multi_metric_qmv(metric_scores)`: QMV for multiple metrics

**QMV Calculation**:
```
QMV = σ / |μ|

where:
  σ = standard deviation of scores
  μ = mean of scores
```

**Threshold**: C < 0.03

**Audit Trail**: Records timestamp, QMV value, validity, metric variances

#### 3. Feature Engineer (`src/helios/feature_engineer.py`)

**Purpose**: Implements RLAD (Robust Learning with Adversarial Defense) feature engineering.

**Key Methods**:
- `engineer_features(df, is_training)`: Creates engineered features
- `prepare_for_training(df, target_col, scale_features)`: Prepares final feature set
- `adversarial_validation(X_train, X_test, n_folds)`: Detects distribution shift

**RLAD Features**:
1. **Title Extraction**: Extracts titles from names (Mr, Mrs, Miss, Master, Rare)
2. **Family Features**: FamilySize, IsAlone
3. **Fare Engineering**: FarePerPerson, FareBin
4. **Age Engineering**: Age imputation by Title/Pclass, AgeBin, IsChild
5. **Cabin Features**: Deck extraction, HasCabin
6. **Interaction Features**: Pclass_Sex, Pclass_Age

**Adversarial Validation**:
- Trains classifier to distinguish train vs test
- AUC < 0.6: Similar distributions (good)
- AUC > 0.75: Significant shift (concerning)

#### 4. Ensemble Orchestrator (`src/helios/ensemble_orchestrator.py`)

**Purpose**: Implements MoT (Mixture of Techniques) ensemble voting.

**Base Models**:
1. **Random Forest**: 200 estimators, max_depth=8
2. **XGBoost**: 200 estimators, learning_rate=0.05
3. **LightGBM**: 200 estimators, learning_rate=0.05
4. **Logistic Regression**: C=1.0, L2 regularization
5. **Gradient Boosting**: 200 estimators, learning_rate=0.05

**Key Methods**:
- `train_with_cv(X, y, n_folds, verbose)`: Train with stratified k-fold CV
- `predict(X, voting)`: Make predictions using ensemble
- `predict_proba(X)`: Get prediction probabilities
- `evaluate(X, y, voting)`: Evaluate ensemble performance

**Voting Strategies**:
1. **Weighted**: Performance-based weights via softmax
   ```
   weight_i = exp(score_i) / Σ(exp(score_j))
   prediction = argmax(Σ(weight_i * prob_i))
   ```
2. **Majority**: Simple majority vote
3. **Soft**: Average probabilities

**Cross-Validation**:
- Stratified k-fold (default: 5 folds)
- Maintains class distribution
- Performance tracked per fold

## Workflow

### 1. Data Loading
```python
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
```

### 2. Feature Engineering
```python
feature_engineer = FeatureEngineer()
train_eng = feature_engineer.engineer_features(train_df, is_training=True)
test_eng = feature_engineer.engineer_features(test_df, is_training=False)
X_train, y_train = feature_engineer.prepare_for_training(train_eng)
```

### 3. ISR Validation
```python
isr_validator = ISRValidator(threshold=1.5)
isr_metrics = isr_validator.validate(X_train[:700], X_train[700:])
assert isr_metrics.is_valid, "ISR validation failed"
```

### 4. Ensemble Training
```python
ensemble = EnsembleOrchestrator(random_state=42)
cv_scores = ensemble.train_with_cv(X_train, y_train, n_folds=5)
```

### 5. QMV Monitoring
```python
qmv_monitor = QMVMonitor(threshold=0.03)
for model_name, scores in cv_scores.items():
    qmv_metrics = qmv_monitor.validate(scores, metric_name=model_name)
    assert qmv_metrics.is_valid, f"QMV validation failed for {model_name}"
```

### 6. Prediction
```python
predictions = ensemble.predict(X_test, voting='weighted')
```

### 7. Audit Trail Export
```python
isr_validator.export_audit_trail('audit_trails/isr_audit.csv')
qmv_monitor.export_audit_trail('audit_trails/qmv_audit.csv')
```

## Configuration

Edit `configs/config.py` to customize:

```python
# ISR Configuration
ISR_THRESHOLD = 1.5
ISR_VALIDATION_ENABLED = True

# QMV Configuration
QMV_THRESHOLD = 0.03
QMV_VALIDATION_ENABLED = True

# Cross-Validation
CV_N_FOLDS = 5
CV_STRATIFIED = True
CV_RANDOM_STATE = 42

# Ensemble
ENSEMBLE_VOTING = 'weighted'  # 'weighted', 'majority', 'soft'

# Target Performance
TARGET_ACCURACY_MIN = 0.78
TARGET_ACCURACY_MAX = 0.82
```

## Performance Metrics

### Expected Results
- **Cross-validation accuracy**: 78-82%
- **Test accuracy**: 78-82%
- **ISR compliance**: T ≥ 1.5 across validations
- **QMV compliance**: C < 0.03 for all models

### Evaluation Metrics
- **Accuracy**: Correct predictions / Total predictions
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

## Best Practices

### 1. Data Preparation
- Always use `is_training=True` for training data on first call
- Scale features consistently using same scaler
- Handle missing values before feature engineering

### 2. ISR Validation
- Validate before training (pre-training check)
- Validate periodically during iterative refinement
- Monitor per-feature stability for feature selection

### 3. QMV Monitoring
- Check all models individually
- Aggregate QMV for ensemble health
- Re-tune models with high variance

### 4. Ensemble Usage
- Use weighted voting for best performance
- Check CV summary to identify weak models
- Consider removing consistently poor models

### 5. Audit Trails
- Export after each major milestone
- Review trends over multiple runs
- Use for compliance documentation

## Troubleshooting

### ISR Validation Fails (ISR < 1.5)
**Causes**:
- High distribution shift between train/val
- Insufficient data preprocessing
- Feature scaling issues

**Solutions**:
- Check data quality and outliers
- Ensure consistent preprocessing
- Add more data augmentation

### QMV Exceeds Threshold (QMV ≥ 0.03)
**Causes**:
- High variance across folds
- Overfitting to specific folds
- Small dataset size

**Solutions**:
- Increase regularization
- Use more cross-validation folds
- Add more training data
- Tune model hyperparameters

### Low Ensemble Accuracy
**Causes**:
- Poor feature engineering
- Weak base models
- Inappropriate hyperparameters

**Solutions**:
- Review and enhance feature set
- Tune individual model parameters
- Try different ensemble strategies
- Check for data leakage

## API Reference

### ISRValidator

```python
class ISRValidator(threshold: float = 1.5)
    compute_isr(X_train, X_val, feature_weights=None) -> float
    validate(X_train, X_val, feature_weights=None, metadata=None) -> ISRMetrics
    get_audit_summary() -> pd.DataFrame
    export_audit_trail(filepath: str)
    check_threshold(isr_value: float) -> bool
```

### QMVMonitor

```python
class QMVMonitor(threshold: float = 0.03)
    compute_qmv(scores, metric_name='accuracy') -> float
    validate(scores, metric_name='accuracy', metadata=None) -> QMVMetrics
    compute_multi_metric_qmv(metric_scores) -> Dict[str, float]
    get_audit_summary() -> pd.DataFrame
    export_audit_trail(filepath: str)
    check_threshold(qmv_value: float) -> bool
```

### FeatureEngineer

```python
class FeatureEngineer()
    engineer_features(df, is_training=True) -> pd.DataFrame
    prepare_for_training(df, target_col='Survived', scale_features=True) -> Tuple[pd.DataFrame, pd.Series]
    select_features(df, target_col=None) -> List[str]
    adversarial_validation(X_train, X_test, n_folds=5) -> Dict
```

### EnsembleOrchestrator

```python
class EnsembleOrchestrator(random_state: int = 42)
    train_with_cv(X, y, n_folds=5, verbose=True) -> Dict[str, List[float]]
    predict(X, voting='weighted') -> np.ndarray
    predict_proba(X) -> np.ndarray
    evaluate(X, y, voting='weighted') -> Dict[str, float]
    get_feature_importance(method='mean') -> pd.Series
    get_cv_summary() -> pd.DataFrame
```

## Contributing

When contributing to this framework:
1. Maintain ISR/QMV compliance standards
2. Add tests for new features
3. Update documentation
4. Export audit trails for validation

## License

See LICENSE file for details.
