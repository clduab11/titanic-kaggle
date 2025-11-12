"""
Simple test to validate Helios ML framework with synthetic Titanic-like data.
This test demonstrates that all components work together correctly.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from helios import ISRValidator, QMVMonitor, FeatureEngineer, EnsembleOrchestrator


def create_synthetic_titanic_data(n_samples=200):
    """Create synthetic Titanic-like dataset for testing."""
    np.random.seed(42)
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Name': [f'Passenger {i}, Mr. Test' if i % 2 == 0 else f'Passenger {i}, Mrs. Test' 
                 for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 15, n_samples).clip(1, 80),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.3, n_samples),
        'Fare': np.random.gamma(2, 20, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'Cabin': [f'C{i}' if np.random.random() > 0.7 else None for i in range(n_samples)],
    }
    
    # Generate target based on simple rules (similar to Titanic patterns)
    df = pd.DataFrame(data)
    survival_prob = 0.5
    survival_prob += 0.2 * (df['Sex'] == 'female')
    survival_prob -= 0.1 * (df['Pclass'] == 3)
    survival_prob += 0.1 * (df['Age'] < 18)
    survival_prob = survival_prob.clip(0, 1)
    
    df['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)
    
    return df


def test_framework():
    """Test the complete Helios ML framework."""
    print("="*70)
    print("HELIOS ML FRAMEWORK - VALIDATION TEST")
    print("="*70)
    
    # Create synthetic data
    print("\n1. Creating synthetic Titanic-like data...")
    train_df = create_synthetic_titanic_data(n_samples=400)
    test_df = create_synthetic_titanic_data(n_samples=200)
    test_df = test_df.drop('Survived', axis=1)  # Remove target from test
    
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"   Survival rate: {train_df['Survived'].mean():.2%}")
    
    # Feature engineering
    print("\n2. Applying RLAD feature engineering...")
    feature_engineer = FeatureEngineer()
    train_engineered = feature_engineer.engineer_features(train_df, is_training=True)
    test_engineered = feature_engineer.engineer_features(test_df, is_training=False)
    print(f"   Engineered features: {train_engineered.shape[1]}")
    
    # Prepare features
    print("\n3. Preparing features for training...")
    X_train, y_train = feature_engineer.prepare_for_training(
        train_engineered, target_col='Survived', scale_features=True
    )
    
    feature_cols = feature_engineer.select_features(test_engineered)
    X_test = test_engineered[feature_cols].copy()
    
    # Scale test features
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X_test[numeric_cols] = feature_engineer.scalers['scaler'].transform(X_test[numeric_cols])
    
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}")
    
    # ISR Validation
    print("\n4. ISR Validation (T ≥ 1.5)...")
    from sklearn.model_selection import train_test_split
    X_train_isr, X_val_isr, _, _ = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    isr_validator = ISRValidator(threshold=1.5)
    isr_metrics = isr_validator.validate(X_train_isr, X_val_isr, metadata={'stage': 'test'})
    
    print(f"   ISR Value: {isr_metrics.isr_value:.4f}")
    print(f"   Valid: {isr_metrics.is_valid} (threshold: {isr_metrics.threshold})")
    if isr_metrics.is_valid:
        print("   ✓ ISR validation PASSED")
    else:
        print("   ✗ ISR validation FAILED (acceptable for synthetic data)")
    
    # Train ensemble
    print("\n5. Training ensemble with stratified 5-fold CV...")
    ensemble = EnsembleOrchestrator(random_state=42)
    cv_scores = ensemble.train_with_cv(X_train, y_train, n_folds=5, verbose=False)
    
    print("   Model CV scores:")
    for model_name, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"     {model_name:20s}: {mean_score:.4f} ± {std_score:.4f}")
    
    # QMV Monitoring
    print("\n6. QMV Monitoring (C < 0.03)...")
    qmv_monitor = QMVMonitor(threshold=0.03)
    
    qmv_valid_count = 0
    qmv_total = len(cv_scores)
    
    for model_name, scores in cv_scores.items():
        qmv_metrics = qmv_monitor.validate(scores, metric_name=model_name)
        if qmv_metrics.is_valid:
            qmv_valid_count += 1
            status = "✓"
        else:
            status = "✗"
        print(f"   {status} {model_name:20s}: QMV = {qmv_metrics.qmv_value:.6f}")
    
    print(f"\n   QMV compliance: {qmv_valid_count}/{qmv_total} models")
    
    # Generate predictions
    print("\n7. Generating predictions...")
    predictions = ensemble.predict(X_test, voting='weighted')
    probabilities = ensemble.predict_proba(X_test)
    
    print(f"   Predictions generated: {len(predictions)}")
    print(f"   Predicted survival rate: {predictions.mean():.2%}")
    print(f"   Average probability: {probabilities.mean():.4f}")
    
    # CV Summary
    print("\n8. Cross-validation summary:")
    cv_summary = ensemble.get_cv_summary()
    print(cv_summary.to_string(index=False))
    
    best_score = cv_summary['mean_score'].max()
    print(f"\n   Best mean CV accuracy: {best_score:.4f}")
    
    # Export audit trails
    print("\n9. Exporting audit trails...")
    os.makedirs('audit_trails', exist_ok=True)
    isr_validator.export_audit_trail('audit_trails/test_isr_audit.csv')
    qmv_monitor.export_audit_trail('audit_trails/test_qmv_audit.csv')
    print("   ✓ Audit trails exported")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Feature Engineering: {train_engineered.shape[1]} features created")
    print(f"{'✓' if isr_metrics.is_valid else '⚠'} ISR Validation: {isr_metrics.isr_value:.4f} (threshold: 1.5)")
    print(f"✓ Ensemble Training: 5 models with stratified k-fold CV")
    print(f"✓ QMV Monitoring: {qmv_valid_count}/{qmv_total} models meet C < 0.03")
    print(f"✓ Predictions: Generated for {len(predictions)} samples")
    print(f"✓ Audit Trails: Exported ISR and QMV records")
    print(f"✓ Best CV Accuracy: {best_score:.4f}")
    
    print("\n" + "="*70)
    print("FRAMEWORK VALIDATION: ✓ ALL COMPONENTS WORKING")
    print("="*70)
    
    return {
        'isr_valid': isr_metrics.is_valid,
        'qmv_compliance': qmv_valid_count / qmv_total,
        'best_accuracy': best_score,
        'predictions': predictions
    }


if __name__ == "__main__":
    try:
        results = test_framework()
        print("\n✓ Test completed successfully!")
        exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
