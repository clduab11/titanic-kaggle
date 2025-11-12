# Helios ML Framework - Titanic Kaggle Competition

**ISR-governed (T≥1.5), QMV-monitored (C<0.03) multi-agent system for Kaggle Titanic competition**

Helios ML is an advanced machine learning framework implementing rigorous validation and monitoring standards through Information Stability Ratio (ISR) governance and Quality Metric Variance (QMV) monitoring.

## 🎯 Target Performance

- **Accuracy**: 78-82% on test set
- **ISR Threshold**: T ≥ 1.5 (information stability)
- **QMV Threshold**: C < 0.03 (quality consistency)
- **Full audit trail**: Complete ISR/QMV compliance documentation

## 🏗️ Architecture

### Core Components

1. **ISR Validator** (`src/helios/isr_validator.py`)
   - Ensures T ≥ 1.5 for model stability
   - Monitors feature stability and data distribution consistency
   - Maintains audit trail of all validations

2. **QMV Monitor** (`src/helios/qmv_monitor.py`)
   - Ensures C < 0.03 for performance consistency
   - Tracks model quality variance across folds
   - Provides statistical performance summaries

3. **Feature Engineer** (`src/helios/feature_engineer.py`)
   - RLAD (Robust Learning with Adversarial Defense) abstractions
   - Advanced feature engineering for Titanic dataset
   - Adversarial validation for distribution shift detection

4. **Ensemble Orchestrator** (`src/helios/ensemble_orchestrator.py`)
   - MoT (Mixture of Techniques) ensemble voting
   - Multiple base models: Random Forest, XGBoost, LightGBM, Logistic Regression, Gradient Boosting
   - Stratified k-fold cross-validation
   - Weighted voting based on model performance

## 📁 Project Structure

```
titanic-kaggle/
├── src/
│   └── helios/
│       ├── __init__.py
│       ├── isr_validator.py
│       ├── qmv_monitor.py
│       ├── feature_engineer.py
│       └── ensemble_orchestrator.py
├── notebooks/
│   └── helios_demo.ipynb
├── data/
│   └── (place train.csv and test.csv here)
├── configs/
│   ├── __init__.py
│   └── config.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/clduab11/titanic-kaggle.git
cd titanic-kaggle

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- scikit-learn ≥ 1.3.0
- xgboost ≥ 2.0.0
- lightgbm ≥ 4.0.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0

### Data Setup

Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place the files in the `data/` directory:
- `train.csv`
- `test.csv`

### Usage

#### Python Script

```python
from helios import ISRValidator, QMVMonitor, FeatureEngineer, EnsembleOrchestrator
import pandas as pd

# Load data
train_df = pd.read_csv('data/train.csv')

# Feature engineering with RLAD
feature_engineer = FeatureEngineer()
train_engineered = feature_engineer.engineer_features(train_df, is_training=True)
X_train, y_train = feature_engineer.prepare_for_training(train_engineered)

# ISR validation
isr_validator = ISRValidator(threshold=1.5)
isr_metrics = isr_validator.validate(X_train[:700], X_train[700:])
print(f"ISR Valid: {isr_metrics.is_valid}")

# Train ensemble with stratified k-fold CV
ensemble = EnsembleOrchestrator(random_state=42)
cv_scores = ensemble.train_with_cv(X_train, y_train, n_folds=5)

# QMV monitoring
qmv_monitor = QMVMonitor(threshold=0.03)
for model_name, scores in cv_scores.items():
    qmv_metrics = qmv_monitor.validate(scores, metric_name=model_name)
    print(f"{model_name} QMV: {qmv_metrics.qmv_value:.6f}, Valid: {qmv_metrics.is_valid}")

# Make predictions
predictions = ensemble.predict(X_test, voting='weighted')
```

#### Jupyter Notebook

Open and run `notebooks/helios_demo.ipynb` for a complete walkthrough of the framework.

## 🔬 Key Features

### RLAD Abstractions
- **Robust feature engineering**: Title extraction, family size, fare binning, age imputation
- **Adversarial validation**: Detects distribution shift between train and test sets
- **Composite features**: Interaction features for enhanced model performance

### Stratified K-Fold Cross-Validation
- Maintains class distribution across folds
- 5-fold cross-validation by default
- Performance tracking per fold

### MoT Ensemble Voting
- **Multiple base models**: Combines diverse model architectures
- **Weighted voting**: Performance-based weight assignment using softmax
- **Voting strategies**: Weighted, majority, and soft voting options

### Audit Trail
- Complete ISR validation history
- QMV monitoring records
- Exportable to CSV for compliance documentation

## 📊 Expected Results

The Helios ML framework is designed to achieve:

- **Cross-validation accuracy**: 78-82%
- **Test accuracy**: 78-82%
- **ISR compliance**: All validations meet T ≥ 1.5
- **QMV compliance**: All models meet C < 0.03

## 🔍 Validation Metrics

### ISR (Information Stability Ratio)
- **Threshold**: T ≥ 1.5
- **Purpose**: Ensures training and validation data have consistent information content
- **Interpretation**: Higher values indicate greater stability

### QMV (Quality Metric Variance)
- **Threshold**: C < 0.03
- **Purpose**: Ensures consistent model performance across cross-validation folds
- **Interpretation**: Lower values indicate greater consistency

## 📈 Model Performance

The ensemble includes:

1. **Random Forest**: 200 estimators, max_depth=8
2. **XGBoost**: 200 estimators, learning_rate=0.05
3. **LightGBM**: 200 estimators, learning_rate=0.05
4. **Logistic Regression**: C=1.0, L2 regularization
5. **Gradient Boosting**: 200 estimators, learning_rate=0.05

Performance weights are automatically calculated based on cross-validation results.

## 🛡️ Quality Assurance

- ✅ ISR validation ensures data stability
- ✅ QMV monitoring ensures model consistency
- ✅ Adversarial validation detects distribution shift
- ✅ Stratified CV maintains class balance
- ✅ Full audit trail for reproducibility

## 📝 License

This project is licensed under the terms included in the LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please ensure that any changes maintain ISR/QMV compliance standards.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
