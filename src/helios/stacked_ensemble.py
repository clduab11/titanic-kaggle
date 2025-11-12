"""
Stacked Ensemble Orchestrator with enhanced models for 83-85% accuracy
Implements multi-level stacking with CatBoost, ExtraTrees, and SVC.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb

# Try to import CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")


class StackedEnsembleOrchestrator:
    """
    Stacked Ensemble Orchestrator for Helios ML Framework.

    Implements sophisticated stacking strategy with:
    - Level 1: 7 diverse base models with optimized hyperparameters
    - Level 2: Meta-learner using LogisticRegression
    - 10-fold stratified cross-validation
    - Adversarial validation for distribution shift detection
    - Weighted voting and rank averaging

    Designed to achieve 83-85% accuracy on Titanic dataset.
    """

    def __init__(self, random_state: int = 42, n_folds: int = 10):
        """
        Initialize Stacked Ensemble Orchestrator.

        Args:
            random_state: Random seed for reproducibility
            n_folds: Number of CV folds (default: 10)
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.models: Dict[str, Any] = {}
        self.meta_model: Any = None
        self.weights: Dict[str, float] = {}
        self.cv_scores: Dict[str, List[float]] = {}
        self._feature_importance: Dict[str, np.ndarray] = {}
        self._oof_predictions = None

    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize optimized base models for Level 1."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=0.05,
                reg_lambda=1,
                scale_pos_weight=1.5,  # Handle class imbalance
                random_state=self.random_state,
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=1000,
                num_leaves=31,
                learning_rate=0.01,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                lambda_l1=0.05,
                lambda_l2=1,
                min_child_samples=20,
                class_weight='balanced',
                random_state=self.random_state,
                verbosity=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                C=0.1,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            ),
            'svc': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            )
        }

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=3,
                bagging_temperature=1,
                random_strength=1,
                border_count=128,
                auto_class_weights='Balanced',
                random_state=self.random_state,
                verbose=False
            )

        return models

    def train_with_stacking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Tuple[Dict[str, List[float]], np.ndarray]:
        """
        Train ensemble using stacking with 10-fold stratified CV.

        Args:
            X: Training features
            y: Training target
            verbose: Whether to print progress

        Returns:
            Tuple of (CV scores dict, out-of-fold predictions)
        """
        if verbose:
            print("=" * 70)
            print("STACKED ENSEMBLE TRAINING (10-Fold Stratified CV)")
            print("=" * 70)

        models = self._initialize_base_models()
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Initialize storage
        cv_scores = {name: [] for name in models.keys()}
        oof_predictions = np.zeros((len(X), len(models)))
        model_columns = list(models.keys())

        # Level 1: Train base models with OOF predictions
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if verbose:
                print(f"\nFold {fold_idx + 1}/{self.n_folds}")

            for model_idx, (name, model) in enumerate(models.items()):
                # Handle early stopping for tree models
                if name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'lightgbm':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)]
                    )
                elif name == 'catboost' and CATBOOST_AVAILABLE:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                # Get predictions for validation fold
                if hasattr(model, 'predict_proba'):
                    val_preds_proba = model.predict_proba(X_val)[:, 1]
                    oof_predictions[val_idx, model_idx] = val_preds_proba

                # Get class predictions for scoring
                val_preds = model.predict(X_val)
                acc = accuracy_score(y_val, val_preds)
                cv_scores[name].append(acc)

                if verbose:
                    print(f"  {name:20s}: {acc:.4f}")

        # Calculate mean scores and display
        if verbose:
            print("\n" + "=" * 70)
            print("LEVEL 1 MODELS - CROSS-VALIDATION SUMMARY")
            print("=" * 70)

        for name in models.keys():
            mean_score = np.mean(cv_scores[name])
            std_score = np.std(cv_scores[name])
            if verbose:
                print(f"{name:20s}: {mean_score:.4f} (+/- {std_score:.4f})")

        self.cv_scores = cv_scores
        self._oof_predictions = oof_predictions

        # Calculate weights based on performance
        self._calculate_weights()

        # Level 2: Train meta-learner on OOF predictions
        if verbose:
            print("\n" + "=" * 70)
            print("LEVEL 2 META-LEARNER TRAINING")
            print("=" * 70)

        self.meta_model = LogisticRegression(
            C=1.0,
            solver='liblinear',
            random_state=self.random_state
        )

        # Create meta-features dataframe
        meta_features_df = pd.DataFrame(
            oof_predictions,
            columns=model_columns
        )

        # Train meta-learner
        self.meta_model.fit(meta_features_df, y)

        if verbose:
            # Evaluate stacked predictions
            stacked_preds = self.meta_model.predict(meta_features_df)
            stacked_acc = accuracy_score(y, stacked_preds)
            print(f"\nStacked Model OOF Accuracy: {stacked_acc:.4f}")

            # Show meta-model coefficients
            print("\nMeta-Model Coefficients:")
            for model_name, coef in zip(model_columns, self.meta_model.coef_[0]):
                print(f"  {model_name:20s}: {coef:7.4f}")

        # Train final models on full dataset
        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING FINAL MODELS ON FULL DATASET")
            print("=" * 70)

        # Re-initialize models without early stopping for final training
        final_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,  # Reduced for final training
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=0.05,
                reg_lambda=1,
                scale_pos_weight=1.5,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,  # Reduced for final training
                num_leaves=31,
                learning_rate=0.01,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                lambda_l1=0.05,
                lambda_l2=1,
                min_child_samples=20,
                class_weight='balanced',
                random_state=self.random_state,
                verbosity=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                C=0.1,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            ),
            'svc': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            )
        }

        if CATBOOST_AVAILABLE:
            final_models['catboost'] = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=3,
                bagging_temperature=1,
                random_strength=1,
                border_count=128,
                auto_class_weights='Balanced',
                random_state=self.random_state,
                verbose=False
            )

        for name, model in final_models.items():
            model.fit(X, y)
            self.models[name] = model

            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self._feature_importance[name] = model.feature_importances_

        if verbose:
            print("✓ Training complete!")

        return cv_scores, oof_predictions

    def _calculate_weights(self):
        """Calculate ensemble weights based on CV performance."""
        if not self.cv_scores:
            return

        # Calculate mean scores
        mean_scores = {name: np.mean(scores) for name, scores in self.cv_scores.items()}

        # Apply softmax for weights
        scores_array = np.array(list(mean_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))
        softmax_weights = exp_scores / np.sum(exp_scores)

        # Assign weights
        for idx, name in enumerate(mean_scores.keys()):
            self.weights[name] = softmax_weights[idx]

    def predict_stacked(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using stacked ensemble.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        if not self.models or self.meta_model is None:
            raise ValueError("Models not trained. Call train_with_stacking first.")

        # Get Level 1 predictions
        level1_preds = np.zeros((len(X), len(self.models)))

        for idx, (name, model) in enumerate(self.models.items()):
            if hasattr(model, 'predict_proba'):
                level1_preds[:, idx] = model.predict_proba(X)[:, 1]
            else:
                level1_preds[:, idx] = model.predict(X)

        # Create meta-features
        meta_features_df = pd.DataFrame(
            level1_preds,
            columns=list(self.models.keys())
        )

        # Level 2 predictions
        final_preds = self.meta_model.predict(meta_features_df)

        return final_preds

    def predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using weighted voting.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_with_stacking first.")

        weighted_probs = np.zeros(len(X))

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
                weight = self.weights.get(name, 1.0 / len(self.models))
                weighted_probs += weight * probs

        return (weighted_probs >= 0.5).astype(int)

    def predict_rank_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using rank averaging (robust to outliers).

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_with_stacking first.")

        # Get all predictions
        all_preds = []

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
                all_preds.append(preds)

        # Convert to array
        pred_array = np.array(all_preds).T  # Shape: (n_samples, n_models)

        # Rank predictions
        ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, pred_array)

        # Average ranks
        avg_ranks = np.mean(ranked, axis=1)

        # Convert to binary predictions
        threshold = np.median(avg_ranks)
        return (avg_ranks >= threshold).astype(int)

    def predict(
        self,
        X: pd.DataFrame,
        method: str = 'stacked'
    ) -> np.ndarray:
        """
        Make predictions using specified method.

        Args:
            X: Features for prediction
            method: Prediction method ('stacked', 'weighted', 'rank_average')

        Returns:
            Array of predictions
        """
        if method == 'stacked':
            return self.predict_stacked(X)
        elif method == 'weighted':
            return self.predict_weighted(X)
        elif method == 'rank_average':
            return self.predict_rank_average(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_cv_summary(self) -> pd.DataFrame:
        """Get summary of cross-validation results."""
        if not self.cv_scores:
            return pd.DataFrame()

        summary = []
        for name, scores in self.cv_scores.items():
            summary.append({
                'model': name,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'weight': self.weights.get(name, 0.0)
            })

        return pd.DataFrame(summary).sort_values('mean_score', ascending=False)

    def adversarial_validation(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform adversarial validation to detect distribution shift.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Dictionary with adversarial validation results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        print("\n" + "=" * 70)
        print("ADVERSARIAL VALIDATION")
        print("=" * 70)

        # Create adversarial labels
        X_train_adv = X_train.copy()
        X_test_adv = X_test.copy()

        X_train_adv['is_test'] = 0
        X_test_adv['is_test'] = 1

        # Combine datasets
        X_combined = pd.concat([X_train_adv, X_test_adv], axis=0)
        y_combined = X_combined['is_test']
        X_combined = X_combined.drop('is_test', axis=1)

        # Train adversarial model
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_combined, y_combined):
            X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
            y_tr, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx]

            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)

        mean_auc = np.mean(auc_scores)

        print(f"Adversarial Validation AUC: {mean_auc:.4f}")
        print(f"Interpretation: ", end='')

        if mean_auc < 0.55:
            print("Excellent - Very similar distributions")
        elif mean_auc < 0.60:
            print("Good - Minor distribution differences")
        elif mean_auc < 0.70:
            print("Warning - Moderate distribution shift")
        else:
            print("Alert - Significant distribution shift detected!")

        return {
            'mean_auc': mean_auc,
            'auc_scores': auc_scores,
            'distribution_shift': mean_auc > 0.60,
            'interpretation': 'Low shift' if mean_auc < 0.60 else 'Moderate shift' if mean_auc < 0.70 else 'High shift'
        }
