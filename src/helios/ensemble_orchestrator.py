"""
Ensemble Orchestrator with MoT (Mixture of Techniques) voting
Implements multi-model ensemble with sophisticated voting strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb


class EnsembleOrchestrator:
    """
    Ensemble Orchestrator for Helios ML Framework.
    
    Implements MoT (Mixture of Techniques) ensemble voting with:
    - Multiple base models (RF, XGBoost, LightGBM, LogReg, GBM)
    - Stratified k-fold cross-validation
    - Weighted voting strategies
    - Performance tracking and model selection
    
    Attributes:
        models (Dict): Dictionary of trained base models
        weights (Dict): Learned weights for each model
        cv_scores (Dict): Cross-validation scores for each model
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Ensemble Orchestrator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.cv_scores: Dict[str, List[float]] = {}
        self._feature_importance: Dict[str, np.ndarray] = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize base models for ensemble.
        
        Returns:
            Dictionary of initialized models
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1
            ),
            'logistic': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train ensemble models using stratified k-fold cross-validation.
        
        Args:
            X: Training features
            y: Training target
            n_folds: Number of CV folds
            verbose: Whether to print progress
            
        Returns:
            Dictionary of CV scores for each model
        """
        models = self._initialize_models()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {name: [] for name in models.keys()}
        cv_predictions = {name: np.zeros(len(X)) for name in models.keys()}
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if verbose:
                print(f"\nFold {fold_idx + 1}/{n_folds}")
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Calculate accuracy
                acc = accuracy_score(y_val, y_pred)
                cv_scores[name].append(acc)
                
                # Store predictions for ensemble
                cv_predictions[name][val_idx] = y_pred
                
                if verbose:
                    print(f"  {name}: {acc:.4f}")
        
        # Calculate mean CV scores and weights
        for name in models.keys():
            mean_score = np.mean(cv_scores[name])
            if verbose:
                print(f"\n{name} mean CV score: {mean_score:.4f} (+/- {np.std(cv_scores[name]):.4f})")
        
        # Store CV scores
        self.cv_scores = cv_scores
        
        # Calculate weights based on performance
        self._calculate_weights()
        
        # Train final models on full dataset
        if verbose:
            print("\nTraining final models on full dataset...")
        
        for name, model in models.items():
            model.fit(X, y)
            self.models[name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self._feature_importance[name] = model.feature_importances_
        
        return cv_scores
    
    def _calculate_weights(self):
        """
        Calculate ensemble weights based on CV performance.
        
        Uses softmax of mean CV scores for weighting.
        """
        if not self.cv_scores:
            return
            
        # Calculate mean scores
        mean_scores = {name: np.mean(scores) for name, scores in self.cv_scores.items()}
        
        # Apply softmax for weights
        scores_array = np.array(list(mean_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Assign weights
        for idx, name in enumerate(mean_scores.keys()):
            self.weights[name] = softmax_weights[idx]
    
    def predict(
        self,
        X: pd.DataFrame,
        voting: str = 'weighted'
    ) -> np.ndarray:
        """
        Make predictions using ensemble voting.
        
        Args:
            X: Features for prediction
            voting: Voting strategy ('weighted', 'majority', 'soft')
            
        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_with_cv first.")
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X)[:, 1]
        
        # Apply voting strategy
        if voting == 'weighted':
            return self._weighted_voting(predictions, probabilities)
        elif voting == 'majority':
            return self._majority_voting(predictions)
        elif voting == 'soft':
            return self._soft_voting(probabilities)
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")
    
    def _weighted_voting(
        self,
        predictions: Dict[str, np.ndarray],
        probabilities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Weighted voting using learned weights.
        
        Args:
            predictions: Dictionary of model predictions
            probabilities: Dictionary of model probabilities
            
        Returns:
            Final predictions
        """
        if not probabilities:
            # Fallback to majority voting if no probabilities
            return self._majority_voting(predictions)
        
        # Weighted average of probabilities
        weighted_probs = np.zeros(len(next(iter(probabilities.values()))))
        
        for name, probs in probabilities.items():
            weight = self.weights.get(name, 1.0 / len(probabilities))
            weighted_probs += weight * probs
        
        return (weighted_probs >= 0.5).astype(int)
    
    def _majority_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple majority voting.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Final predictions
        """
        pred_array = np.array(list(predictions.values()))
        return (np.mean(pred_array, axis=0) >= 0.5).astype(int)
    
    def _soft_voting(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Soft voting using average probabilities.
        
        Args:
            probabilities: Dictionary of model probabilities
            
        Returns:
            Final predictions
        """
        if not probabilities:
            raise ValueError("No probability predictions available for soft voting")
        
        prob_array = np.array(list(probabilities.values()))
        avg_probs = np.mean(prob_array, axis=0)
        
        return (avg_probs >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities for positive class
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_with_cv first.")
        
        weighted_probs = np.zeros(len(X))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
                weight = self.weights.get(name, 1.0 / len(self.models))
                weighted_probs += weight * probs
        
        return weighted_probs
    
    def get_feature_importance(self, method: str = 'mean') -> pd.Series:
        """
        Get aggregated feature importance across models.
        
        Args:
            method: Aggregation method ('mean', 'median', 'max')
            
        Returns:
            Series with feature importance scores
        """
        if not self._feature_importance:
            return pd.Series()
        
        # Stack importance arrays
        importance_df = pd.DataFrame(self._feature_importance).T
        
        if method == 'mean':
            aggregated = importance_df.mean(axis=0)
        elif method == 'median':
            aggregated = importance_df.median(axis=0)
        elif method == 'max':
            aggregated = importance_df.max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated.sort_values(ascending=False)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        voting: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Test features
            y: Test target
            voting: Voting strategy
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X, voting=voting)
        y_pred_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def get_cv_summary(self) -> pd.DataFrame:
        """
        Get summary of cross-validation results.
        
        Returns:
            DataFrame with CV statistics for each model
        """
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
