"""
Helios ML Framework - Example Usage Script
Demonstrates basic usage of the framework for Titanic competition.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from helios import ISRValidator, QMVMonitor, FeatureEngineer, EnsembleOrchestrator


def load_data():
    """Load Titanic dataset."""
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    return train_df, test_df


def engineer_features(train_df, test_df):
    """Apply RLAD feature engineering."""
    print("\nApplying RLAD feature engineering...")
    feature_engineer = FeatureEngineer()
    
    train_engineered = feature_engineer.engineer_features(train_df, is_training=True)
    test_engineered = feature_engineer.engineer_features(test_df, is_training=False)
    
    print(f"  Engineered train shape: {train_engineered.shape}")
    print(f"  Engineered test shape: {test_engineered.shape}")
    
    return feature_engineer, train_engineered, test_engineered


def prepare_features(feature_engineer, train_engineered, test_engineered):
    """Prepare features for training."""
    print("\nPreparing features for training...")
    X_train, y_train = feature_engineer.prepare_for_training(
        train_engineered,
        target_col='Survived',
        scale_features=True
    )
    
    feature_cols = feature_engineer.select_features(test_engineered)
    X_test = test_engineered[feature_cols].copy()
    
    # Scale test features
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X_test[numeric_cols] = feature_engineer.scalers['scaler'].transform(X_test[numeric_cols])
    
    print(f"  Training features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")
    
    return X_train, y_train, X_test


def validate_isr(X_train, y_train):
    """Validate Information Stability Ratio."""
    print("\n" + "="*60)
    print("ISR VALIDATION (T ≥ 1.5)")
    print("="*60)
    
    from sklearn.model_selection import train_test_split
    
    # Split for ISR validation
    X_train_isr, X_val_isr, _, _ = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    isr_validator = ISRValidator(threshold=1.5)
    isr_metrics = isr_validator.validate(
        X_train_isr,
        X_val_isr,
        metadata={'stage': 'pre-training'}
    )
    
    print(f"ISR Value: {isr_metrics.isr_value:.4f}")
    print(f"Threshold: {isr_metrics.threshold}")
    print(f"Valid: {isr_metrics.is_valid}")
    
    if isr_metrics.is_valid:
        print("✓ ISR validation PASSED")
    else:
        print("✗ ISR validation FAILED")
    
    return isr_validator


def train_ensemble(X_train, y_train):
    """Train ensemble with stratified k-fold CV."""
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING (Stratified 5-Fold CV)")
    print("="*60)
    
    ensemble = EnsembleOrchestrator(random_state=42)
    cv_scores = ensemble.train_with_cv(
        X_train,
        y_train,
        n_folds=5,
        verbose=True
    )
    
    return ensemble, cv_scores


def validate_qmv(cv_scores):
    """Validate Quality Metric Variance."""
    print("\n" + "="*60)
    print("QMV MONITORING (C < 0.03)")
    print("="*60)
    
    qmv_monitor = QMVMonitor(threshold=0.03)
    
    for model_name, scores in cv_scores.items():
        qmv_metrics = qmv_monitor.validate(
            scores,
            metric_name=model_name,
            metadata={'stage': 'cross-validation'}
        )
        
        status = "✓ PASS" if qmv_metrics.is_valid else "✗ FAIL"
        print(f"{model_name:20s} QMV: {qmv_metrics.qmv_value:.6f}  {status}")
    
    return qmv_monitor


def generate_predictions(ensemble, X_test, test_df):
    """Generate predictions for test set."""
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    predictions = ensemble.predict(X_test, voting='weighted')
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    submission.to_csv('data/submission.csv', index=False)
    
    print(f"Predictions saved to data/submission.csv")
    print(f"Prediction distribution:")
    print(f"  Survived = 0: {(predictions == 0).sum()}")
    print(f"  Survived = 1: {(predictions == 1).sum()}")
    
    return submission


def export_audit_trails(isr_validator, qmv_monitor):
    """Export ISR and QMV audit trails."""
    print("\n" + "="*60)
    print("EXPORTING AUDIT TRAILS")
    print("="*60)
    
    os.makedirs('audit_trails', exist_ok=True)
    
    isr_validator.export_audit_trail('audit_trails/isr_audit_trail.csv')
    print("ISR audit trail exported to audit_trails/isr_audit_trail.csv")
    
    qmv_monitor.export_audit_trail('audit_trails/qmv_audit_trail.csv')
    print("QMV audit trail exported to audit_trails/qmv_audit_trail.csv")


def print_summary(ensemble):
    """Print final summary."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    cv_summary = ensemble.get_cv_summary()
    print("\nCross-Validation Summary:")
    print(cv_summary.to_string(index=False))
    
    best_score = cv_summary['mean_score'].max()
    print(f"\nBest mean CV accuracy: {best_score:.4f}")
    
    if 0.78 <= best_score <= 0.82:
        print("✓ Target accuracy range (78-82%) ACHIEVED")
    elif best_score > 0.82:
        print("⚠ Accuracy exceeds target range")
    else:
        print("✗ Accuracy below target range")
    
    print("\nHelios ML Framework Complete!")
    print("  ✓ ISR-governed (T≥1.5)")
    print("  ✓ QMV-monitored (C<0.03)")
    print("  ✓ RLAD feature engineering")
    print("  ✓ MoT ensemble voting")
    print("  ✓ Full audit trail")


def main():
    """Main execution function."""
    print("="*60)
    print("HELIOS ML FRAMEWORK - TITANIC COMPETITION")
    print("="*60)
    print("ISR-governed (T≥1.5), QMV-monitored (C<0.03)")
    print("Multi-agent system with RLAD and MoT ensemble")
    print("="*60)
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Feature engineering
        feature_engineer, train_engineered, test_engineered = engineer_features(train_df, test_df)
        
        # Prepare features
        X_train, y_train, X_test = prepare_features(feature_engineer, train_engineered, test_engineered)
        
        # ISR validation
        isr_validator = validate_isr(X_train, y_train)
        
        # Train ensemble
        ensemble, cv_scores = train_ensemble(X_train, y_train)
        
        # QMV monitoring
        qmv_monitor = validate_qmv(cv_scores)
        
        # Generate predictions
        submission = generate_predictions(ensemble, X_test, test_df)
        
        # Export audit trails
        export_audit_trails(isr_validator, qmv_monitor)
        
        # Print summary
        print_summary(ensemble)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure train.csv and test.csv are in the data/ directory.")
        print("Download from: https://www.kaggle.com/c/titanic/data")
        return 1
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
