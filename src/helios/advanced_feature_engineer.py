"""
Advanced Feature Engineer with enhanced RLAD abstractions for 83-85% accuracy
Implements sophisticated feature engineering with leave-one-out encoding to prevent leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineer for Helios ML Framework.

    Implements sophisticated RLAD abstractions and advanced feature engineering
    designed to achieve 83-85% accuracy on Titanic dataset.

    Key Features:
    - Family survival rates with leave-one-out encoding
    - MICE age imputation
    - Complex interaction features
    - Advanced title and cabin extraction
    - Ticket pattern analysis
    """

    def __init__(self):
        """Initialize Advanced Feature Engineer."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._family_survival_map = {}
        self._ticket_survival_map = {}
        self._age_imputer = None
        self._feature_importance = {}

    def engineer_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Engineer advanced features from raw Titanic dataset.

        Args:
            df: Raw dataframe
            is_training: Whether this is training data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # 1. Title extraction with enhanced categories
        df['Title'] = self._extract_advanced_title(df)

        # 2. Basic family features
        df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['FamilySizeCategory'] = df['FamilySize'].apply(self._categorize_family_size)

        # 3. Social features
        df['Mother'] = ((df.get('Sex', '') == 'female') &
                       (df.get('Parch', 0) > 0) &
                       (df.get('Age', 0) > 18) &
                       (df['Title'] != 'Miss')).astype(int)

        df['Child'] = (df.get('Age', 100) < 14).astype(int)
        df['ChildWithNanny'] = ((df['Child'] == 1) &
                                (df.get('Pclass', 3) == 1) &
                                (df.get('SibSp', 0) == 0)).astype(int)

        # 4. Surname extraction
        if 'Name' in df.columns:
            df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
            df['SurnameCount'] = df.groupby('Surname')['Surname'].transform('count')

        # 5. Ticket features
        if 'Ticket' in df.columns:
            df['TicketPrefix'] = df['Ticket'].apply(self._extract_ticket_prefix)
            df['TicketNumber'] = df['Ticket'].apply(self._extract_ticket_number)
            df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
            df['SharedTicket'] = (df['TicketGroupSize'] > 1).astype(int)

        # 6. Cabin features
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['CabinNumber'] = df['Cabin'].apply(self._extract_cabin_number)
            df['MultipleCabins'] = df['Cabin'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

        # 7. Embarked handling
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna('S')  # Most common port

        # 8. Fare engineering
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            df['LogFare'] = np.log1p(df['Fare'])
            df['FareOutlier'] = (df['Fare'] > df['Fare'].quantile(0.95)).astype(int)

            # Fare bins within each class
            for pclass in [1, 2, 3]:
                mask = df.get('Pclass', 3) == pclass
                if mask.sum() > 0:
                    df.loc[mask, f'FareBin_Class{pclass}'] = pd.qcut(
                        df.loc[mask, 'Fare'],
                        q=4,
                        labels=False,
                        duplicates='drop'
                    )

            # Fill NaN fare bins with median
            for pclass in [1, 2, 3]:
                col = f'FareBin_Class{pclass}'
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

        # 9. Age imputation with MICE
        if 'Age' in df.columns:
            df = self._impute_age_mice(df, is_training)
            df['AgeBin'] = df['Age'].apply(self._categorize_age)
            df['IsChild'] = (df['Age'] < 18).astype(int)
            df['IsSenior'] = (df['Age'] >= 60).astype(int)
            df['Teen'] = ((df['Age'] >= 13) & (df['Age'] < 20)).astype(int)

            # Age * Class interaction
            if 'Pclass' in df.columns:
                df['AgeClass'] = df['AgeBin'].astype(str) + '_' + df['Pclass'].astype(str)

        # 10. Family survival rates with LEAVE-ONE-OUT encoding (CRITICAL)
        if is_training and 'Survived' in df.columns:
            df = self._create_family_survival_features_loo(df)
        elif not is_training:
            df = self._apply_family_survival_features(df)

        # 11. Encode categorical features
        categorical_features = {
            'Sex': df.get('Sex'),
            'Title': df['Title'],
            'Embarked': df.get('Embarked'),
            'Deck': df.get('Deck'),
            'TicketPrefix': df.get('TicketPrefix'),
            'FamilySizeCategory': df['FamilySizeCategory'],
            'AgeClass': df.get('AgeClass')
        }

        for feat_name, feat_series in categorical_features.items():
            if feat_series is not None and feat_name in df.columns:
                encoded_name = f'{feat_name}_encoded'
                if is_training:
                    self.encoders[feat_name] = LabelEncoder()
                    df[encoded_name] = self.encoders[feat_name].fit_transform(feat_series.astype(str))
                else:
                    # Handle unseen categories
                    df[encoded_name] = feat_series.apply(
                        lambda x: self.encoders[feat_name].transform([str(x)])[0]
                        if str(x) in self.encoders[feat_name].classes_ else -1
                    )

        # 12. Advanced interaction features (KEY for 83%+ accuracy)
        if all(col in df.columns for col in ['Sex_encoded', 'Pclass', 'AgeBin']):
            df['Sex_Pclass_Age'] = (df['Sex_encoded'].astype(str) + '_' +
                                    df['Pclass'].astype(str) + '_' +
                                    df['AgeBin'].astype(str))
            if is_training:
                self.encoders['Sex_Pclass_Age'] = LabelEncoder()
                df['Sex_Pclass_Age_encoded'] = self.encoders['Sex_Pclass_Age'].fit_transform(df['Sex_Pclass_Age'])
            else:
                df['Sex_Pclass_Age_encoded'] = df['Sex_Pclass_Age'].apply(
                    lambda x: self.encoders['Sex_Pclass_Age'].transform([x])[0]
                    if x in self.encoders['Sex_Pclass_Age'].classes_ else -1
                )

        if all(col in df.columns for col in ['FamilySize', 'Pclass']):
            df['FamilySize_Pclass'] = df['FamilySize'] * 10 + df['Pclass']

        if all(col in df.columns for col in ['IsAlone', 'Sex_encoded', 'Pclass']):
            df['IsAlone_Sex_Pclass'] = (df['IsAlone'].astype(str) + '_' +
                                        df['Sex_encoded'].astype(str) + '_' +
                                        df['Pclass'].astype(str))
            if is_training:
                self.encoders['IsAlone_Sex_Pclass'] = LabelEncoder()
                df['IsAlone_Sex_Pclass_encoded'] = self.encoders['IsAlone_Sex_Pclass'].fit_transform(df['IsAlone_Sex_Pclass'])
            else:
                df['IsAlone_Sex_Pclass_encoded'] = df['IsAlone_Sex_Pclass'].apply(
                    lambda x: self.encoders['IsAlone_Sex_Pclass'].transform([x])[0]
                    if x in self.encoders['IsAlone_Sex_Pclass'].classes_ else -1
                )

        if all(col in df.columns for col in ['FarePerPerson', 'Embarked_encoded']):
            df['FarePerPerson_Embarked'] = df['FarePerPerson'] * df['Embarked_encoded']

        if all(col in df.columns for col in ['Deck_encoded', 'Sex_encoded']):
            df['Deck_Sex'] = df['Deck_encoded'] * 10 + df['Sex_encoded']

        if all(col in df.columns for col in ['Title_encoded', 'Pclass']):
            df['Title_Pclass'] = df['Title_encoded'] * 10 + df['Pclass']

        return df

    def _extract_advanced_title(self, df: pd.DataFrame) -> pd.Series:
        """Extract and categorize titles from passenger names."""
        if 'Name' not in df.columns:
            return pd.Series(['Mr'] * len(df))

        titles = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Enhanced title mapping
        title_mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare_Female', 'Countess': 'Rare_Female', 'Dona': 'Rare_Female',
            'Capt': 'Rare_Male', 'Col': 'Rare_Male', 'Don': 'Rare_Male',
            'Major': 'Rare_Male', 'Sir': 'Rare_Male', 'Jonkheer': 'Rare_Male',
            'Rev': 'Clergy', 'Dr': 'Professional'
        }

        titles = titles.map(title_mapping).fillna(titles)
        return titles

    def _categorize_family_size(self, size: int) -> str:
        """Categorize family size."""
        if size == 1:
            return 'Single'
        elif size <= 3:
            return 'Small'
        else:
            return 'Large'

    def _categorize_age(self, age: float) -> int:
        """Categorize age into bins."""
        if age < 13:
            return 0  # Child
        elif age < 20:
            return 1  # Teen
        elif age < 60:
            return 2  # Adult
        else:
            return 3  # Senior

    def _extract_ticket_prefix(self, ticket: str) -> str:
        """Extract ticket prefix."""
        if pd.isna(ticket):
            return 'NONE'
        ticket_str = str(ticket).strip()
        # Extract alphabetic prefix
        prefix = ''.join([c for c in ticket_str if c.isalpha()])
        return prefix if prefix else 'NUMERIC'

    def _extract_ticket_number(self, ticket: str) -> int:
        """Extract ticket number."""
        if pd.isna(ticket):
            return 0
        ticket_str = str(ticket).strip()
        # Extract numeric part
        numbers = ''.join([c for c in ticket_str if c.isdigit()])
        return int(numbers) if numbers else 0

    def _extract_cabin_number(self, cabin: str) -> int:
        """Extract cabin number."""
        if pd.isna(cabin):
            return 0
        cabin_str = str(cabin).strip()
        numbers = ''.join([c for c in cabin_str if c.isdigit()])
        return int(numbers) if numbers else 0

    def _impute_age_mice(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Impute age using MICE (Multivariate Imputation by Chained Equations)."""
        if 'Age' not in df.columns:
            return df

        # Features to use for imputation
        impute_features = ['Pclass', 'SibSp', 'Parch', 'Fare']

        # Add encoded sex if available
        if 'Sex' in df.columns:
            df['Sex_temp'] = df['Sex'].map({'male': 0, 'female': 1})
            impute_features.append('Sex_temp')

        # Select features that exist
        available_features = [f for f in impute_features if f in df.columns]

        if len(available_features) > 0 and df['Age'].isna().sum() > 0:
            if is_training:
                # Fit MICE imputer
                self._age_imputer = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    min_value=0,
                    max_value=100
                )

                # Prepare data for imputation
                impute_data = df[available_features + ['Age']].copy()
                imputed_values = self._age_imputer.fit_transform(impute_data)
                df['Age'] = imputed_values[:, -1]
            else:
                if self._age_imputer is not None:
                    # Transform test data
                    impute_data = df[available_features + ['Age']].copy()
                    imputed_values = self._age_imputer.transform(impute_data)
                    df['Age'] = imputed_values[:, -1]
                else:
                    # Fallback to median
                    df['Age'] = df['Age'].fillna(df['Age'].median())

        # Clean up temporary column
        if 'Sex_temp' in df.columns:
            df.drop('Sex_temp', axis=1, inplace=True)

        return df

    def _create_family_survival_features_loo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create family survival features using LEAVE-ONE-OUT encoding.

        This is CRITICAL to prevent data leakage while still capturing family survival patterns.
        Each passenger's family survival rate excludes their own survival status.
        """
        df = df.copy()

        # Initialize features
        df['FamilySurvivalRate_LOO'] = 0.5  # Default to population mean
        df['TicketSurvivalRate_LOO'] = 0.5

        # Family survival by Surname + Fare (proxy for family unit)
        if all(col in df.columns for col in ['Surname', 'Fare', 'Survived']):
            # Create family groups
            df['FamilyGroup'] = df['Surname'] + '_' + df['Fare'].round(2).astype(str)

            # Calculate leave-one-out survival rates
            for idx, row in df.iterrows():
                family_group = row['FamilyGroup']

                # Get all family members except current passenger
                family_mask = (df['FamilyGroup'] == family_group) & (df.index != idx)

                if family_mask.sum() > 0:
                    # Calculate survival rate excluding this passenger
                    survival_rate = df.loc[family_mask, 'Survived'].mean()
                    df.at[idx, 'FamilySurvivalRate_LOO'] = survival_rate
                else:
                    # Single passenger - use overall rate adjusted by features
                    sex_rate = df[df['Sex'] == row['Sex']]['Survived'].mean()
                    class_rate = df[df['Pclass'] == row['Pclass']]['Survived'].mean()
                    df.at[idx, 'FamilySurvivalRate_LOO'] = (sex_rate + class_rate) / 2

        # Ticket group survival (leave-one-out)
        if all(col in df.columns for col in ['Ticket', 'Survived']):
            for idx, row in df.iterrows():
                ticket = row['Ticket']

                # Get all ticket members except current passenger
                ticket_mask = (df['Ticket'] == ticket) & (df.index != idx)

                if ticket_mask.sum() > 0:
                    survival_rate = df.loc[ticket_mask, 'Survived'].mean()
                    df.at[idx, 'TicketSurvivalRate_LOO'] = survival_rate

        # Store mapping for test set (using all training data)
        self._family_survival_map = df.groupby('FamilyGroup')['Survived'].mean().to_dict()
        self._ticket_survival_map = df.groupby('Ticket')['Survived'].mean().to_dict()

        return df

    def _apply_family_survival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned family survival features to test set."""
        df = df.copy()

        # Initialize features
        df['FamilySurvivalRate_LOO'] = 0.5
        df['TicketSurvivalRate_LOO'] = 0.5

        # Apply family survival mapping
        if 'Surname' in df.columns and 'Fare' in df.columns:
            df['FamilyGroup'] = df['Surname'] + '_' + df['Fare'].round(2).astype(str)

            for idx, row in df.iterrows():
                family_group = row['FamilyGroup']

                if family_group in self._family_survival_map:
                    df.at[idx, 'FamilySurvivalRate_LOO'] = self._family_survival_map[family_group]
                else:
                    # Use conditional defaults for unseen families
                    df.at[idx, 'FamilySurvivalRate_LOO'] = 0.5

        # Apply ticket survival mapping
        if 'Ticket' in df.columns:
            for idx, row in df.iterrows():
                ticket = row['Ticket']

                if ticket in self._ticket_survival_map:
                    df.at[idx, 'TicketSurvivalRate_LOO'] = self._ticket_survival_map[ticket]

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> List[str]:
        """Select optimal feature set for modeling."""
        # Core numeric features
        numeric_features = [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'FarePerPerson', 'LogFare',
            'HasCabin', 'SurnameCount', 'TicketGroupSize',
            'SharedTicket', 'IsChild', 'IsSenior', 'Teen',
            'Mother', 'Child', 'ChildWithNanny',
            'FareOutlier', 'MultipleCabins', 'CabinNumber',
            'FamilySurvivalRate_LOO', 'TicketSurvivalRate_LOO',
            'FareBin_Class1', 'FareBin_Class2', 'FareBin_Class3'
        ]

        # Encoded categorical features
        encoded_features = [
            'Sex_encoded', 'Title_encoded', 'Embarked_encoded',
            'Deck_encoded', 'TicketPrefix_encoded', 'FamilySizeCategory_encoded',
            'AgeClass_encoded'
        ]

        # Interaction features
        interaction_features = [
            'Sex_Pclass_Age_encoded', 'FamilySize_Pclass',
            'IsAlone_Sex_Pclass_encoded', 'FarePerPerson_Embarked',
            'Deck_Sex', 'Title_Pclass'
        ]

        # Filter to available features
        all_features = numeric_features + encoded_features + interaction_features
        available_features = [
            f for f in all_features
            if f in df.columns and f != target_col
        ]

        return available_features

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = 'Survived',
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare engineered features for model training."""
        # Select features
        feature_cols = self.select_features(df, target_col)

        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None

        # Handle any remaining NaN values
        X = X.fillna(X.median())

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
