"""
Feature Engineer with RLAD (Robust Learning with Adversarial Defense) abstractions
Implements advanced feature engineering for Titanic dataset.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold


class FeatureEngineer:
    """
    Feature Engineer for Helios ML Framework.
    
    Implements RLAD abstractions and advanced feature engineering
    for the Titanic dataset.
    
    Attributes:
        scalers (Dict): Dictionary of fitted scalers for numeric features
        encoders (Dict): Dictionary of fitted encoders for categorical features
    """
    
    def __init__(self):
        """Initialize Feature Engineer."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._feature_importance = {}
        
    def engineer_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Engineer features from raw Titanic dataset.
        
        Applies RLAD-based feature transformations including:
        - Title extraction from names
        - Family size features
        - Fare categorization
        - Age imputation and binning
        - Cabin deck extraction
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data (for fitting transformations)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Extract title from name
        df['Title'] = self._extract_title(df)
        
        # Family size features (RLAD: composite features)
        df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fare engineering
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=False, duplicates='drop')
        
        # Age engineering with RLAD-based imputation
        if 'Age' in df.columns:
            df['Age'] = self._impute_age(df, is_training)
            df['AgeBin'] = pd.cut(df['Age'], bins=5, labels=False)
            df['IsChild'] = (df['Age'] < 18).astype(int)
        
        # Cabin deck extraction
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
            df['HasCabin'] = df['Cabin'].notna().astype(int)
        
        # Embarked encoding
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna('S')
        
        # Sex encoding
        if 'Sex' in df.columns:
            if is_training:
                self.encoders['Sex'] = LabelEncoder()
                df['Sex_encoded'] = self.encoders['Sex'].fit_transform(df['Sex'])
            else:
                df['Sex_encoded'] = self.encoders['Sex'].transform(df['Sex'])
        
        # Title encoding
        if is_training:
            self.encoders['Title'] = LabelEncoder()
            df['Title_encoded'] = self.encoders['Title'].fit_transform(df['Title'])
        else:
            # Handle unseen titles
            df['Title_encoded'] = df['Title'].apply(
                lambda x: self.encoders['Title'].transform([x])[0] 
                if x in self.encoders['Title'].classes_ else -1
            )
        
        # Embarked encoding
        if 'Embarked' in df.columns:
            if is_training:
                self.encoders['Embarked'] = LabelEncoder()
                df['Embarked_encoded'] = self.encoders['Embarked'].fit_transform(df['Embarked'])
            else:
                df['Embarked_encoded'] = self.encoders['Embarked'].transform(df['Embarked'])
        
        # Deck encoding
        if 'Deck' in df.columns:
            if is_training:
                self.encoders['Deck'] = LabelEncoder()
                df['Deck_encoded'] = self.encoders['Deck'].fit_transform(df['Deck'])
            else:
                df['Deck_encoded'] = df['Deck'].apply(
                    lambda x: self.encoders['Deck'].transform([x])[0] 
                    if x in self.encoders['Deck'].classes_ else -1
                )
        
        # RLAD: Interaction features
        if 'Pclass' in df.columns and 'Sex_encoded' in df.columns:
            df['Pclass_Sex'] = df['Pclass'] * 10 + df['Sex_encoded']
        
        if 'Pclass' in df.columns and 'AgeBin' in df.columns:
            df['Pclass_Age'] = df['Pclass'] * 10 + df['AgeBin']
        
        return df
    
    def _extract_title(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract title from passenger name.
        
        Args:
            df: DataFrame with Name column
            
        Returns:
            Series with extracted titles
        """
        if 'Name' not in df.columns:
            return pd.Series(['Mr'] * len(df))
            
        titles = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
            'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
            'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
            'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        
        titles = titles.map(title_mapping).fillna(titles)
        return titles
    
    def _impute_age(
        self,
        df: pd.DataFrame,
        is_training: bool
    ) -> pd.Series:
        """
        Impute missing age values using RLAD-based strategy.
        
        Uses median age by Title and Pclass for robust imputation.
        
        Args:
            df: DataFrame with Age column
            is_training: Whether this is training data
            
        Returns:
            Series with imputed ages
        """
        age = df['Age'].copy()
        
        if age.isna().sum() == 0:
            return age
            
        # Group by Title and Pclass for imputation
        for title in df['Title'].unique():
            for pclass in df.get('Pclass', [1, 2, 3]).unique():
                mask = (df['Title'] == title) & (df.get('Pclass', pclass) == pclass)
                if mask.sum() > 0:
                    median_age = df.loc[mask, 'Age'].median()
                    if pd.notna(median_age):
                        age.loc[mask & age.isna()] = median_age
        
        # Fill remaining with overall median
        age = age.fillna(df['Age'].median())
        
        return age
    
    def select_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> List[str]:
        """
        Select optimal feature set using RLAD principles.
        
        Args:
            df: Engineered DataFrame
            target_col: Optional target column name to exclude
            
        Returns:
            List of selected feature names
        """
        # Define core features
        numeric_features = [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'FarePerPerson',
            'AgeBin', 'IsChild', 'HasCabin'
        ]
        
        encoded_features = [
            'Sex_encoded', 'Title_encoded', 'Embarked_encoded',
            'Deck_encoded', 'Pclass_Sex', 'Pclass_Age'
        ]
        
        # Filter to available features
        available_features = [
            f for f in numeric_features + encoded_features
            if f in df.columns and f != target_col
        ]
        
        return available_features
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = 'Survived',
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare engineered features for model training.
        
        Args:
            df: Engineered DataFrame
            target_col: Name of target column
            scale_features: Whether to scale numeric features
            
        Returns:
            Tuple of (X, y) - features and target
        """
        # Select features
        feature_cols = self.select_features(df, target_col)
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Scale numeric features
        if scale_features:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                if 'scaler' not in self.scalers:
                    self.scalers['scaler'] = StandardScaler()
                    X[numeric_cols] = self.scalers['scaler'].fit_transform(X[numeric_cols])
                else:
                    X[numeric_cols] = self.scalers['scaler'].transform(X[numeric_cols])
        
        return X, y
    
    def adversarial_validation(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        n_folds: int = 5
    ) -> Dict:
        """
        Perform adversarial validation to detect distribution shift.
        
        RLAD component: Validates that test set is from same distribution.
        
        Args:
            X_train: Training features
            X_test: Test features
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with adversarial validation results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        
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
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X_combined, y_combined):
            X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
            y_tr, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_tr, y_tr)
            
            y_pred = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores)
        
        return {
            'mean_auc': mean_auc,
            'auc_scores': auc_scores,
            'distribution_shift': mean_auc > 0.75,  # Threshold for significant shift
            'interpretation': 'Low shift' if mean_auc < 0.6 else 
                            'Moderate shift' if mean_auc < 0.75 else 'High shift'
        }
