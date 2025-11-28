"""
Data Processing Module for PCA-Agent
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder
from src.utils import detect_column_types, logger


class DataProcessor:
    """
    Handles all data loading, cleaning, and preprocessing operations
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
        self.column_types = None
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath, **kwargs)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   handle_nulls: str = 'drop',
                   null_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean the dataset
        
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            handle_nulls: How to handle nulls ('drop', 'fill_mean', 'fill_median', 'fill_mode')
            null_threshold: Drop columns with null percentage above this
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Remove duplicates
        if drop_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle columns with too many nulls
        null_percentages = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{null_threshold*100}% nulls: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Handle remaining nulls
        if handle_nulls == 'drop':
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows with nulls. Remaining rows: {len(df_clean)}")
        elif handle_nulls == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            logger.info("Filled numeric nulls with mean")
        elif handle_nulls == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            logger.info("Filled numeric nulls with median")
        elif handle_nulls == 'fill_mode':
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            logger.info("Filled nulls with mode")
        
        return df_clean
    
    def detect_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and convert column types appropriately
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected types
        """
        logger.info("Detecting and converting column types...")
        df_typed = df.copy()
        
        # Detect column types
        self.column_types = detect_column_types(df_typed)
        
        # Convert date columns if detected
        for col in df_typed.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    logger.info(f"Converted {col} to datetime")
                except:
                    pass
        
        return df_typed
    
    def encode_categorical(self, df: pd.DataFrame, 
                          target_col: Optional[str] = None,
                          method: str = 'target',
                          categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            target_col: Target column name (for target encoding)
            method: Encoding method ('target', 'onehot', 'label')
            categorical_cols: List of categorical columns (auto-detected if None)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info(f"Encoding categorical variables using {method} encoding...")
        df_encoded = df.copy()
        
        if categorical_cols is None:
            categorical_cols = self.column_types['categorical'] if self.column_types else []
            categorical_cols = [col for col in categorical_cols if col in df_encoded.columns and col != target_col]
        
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df_encoded
        
        if method == 'target' and target_col:
            self.target_encoder = TargetEncoder(cols=categorical_cols)
            df_encoded[categorical_cols] = self.target_encoder.fit_transform(
                df_encoded[categorical_cols], 
                df_encoded[target_col]
            )
            logger.info(f"Target encoded {len(categorical_cols)} columns")
            
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
            logger.info(f"One-hot encoded {len(categorical_cols)} columns")
            
        elif method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            logger.info(f"Label encoded {len(categorical_cols)} columns")
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, 
                          exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from normalization
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing numerical features...")
        df_normalized = df.copy()
        
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if numeric_cols:
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
            logger.info(f"Normalized {len(numeric_cols)} numerical columns")
        
        return df_normalized


class DataValidator:
    """
    Handles data validation and quality checks
    """
    
    def __init__(self):
        self.required_columns = ['date', 'cost']  # Minimal set
        
    def validate_schema(self, df: pd.DataFrame, required_columns: list = None) -> dict:
        """
        Check if required columns exist
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation status and missing columns
        """
        if required_columns is None:
            required_columns = self.required_columns
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        return {
            'valid': len(missing_cols) == 0,
            'missing_columns': missing_cols
        }
    
    def check_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a data quality report
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with quality metrics per column
        """
        quality_report = pd.DataFrame({
            'dtype': df.dtypes,
            'null_count': df.isnull().sum(),
            'null_pct': (df.isnull().sum() / len(df)) * 100,
            'unique_count': df.nunique(),
            'duplicate_count': df.duplicated().sum()
        })
        
        # Add basic stats for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe().T
            quality_report = quality_report.join(desc[['min', 'max', 'mean', 'std']], how='left')
            
        return quality_report.sort_values('null_pct', ascending=False)

    def generate_quality_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a text summary of data quality
        
        Args:
            df: Input DataFrame
            
        Returns:
            String summary
        """
        report = self.check_data_quality(df)
        
        summary = []
        summary.append("DATA QUALITY SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Total Rows: {len(df)}")
        summary.append(f"Total Columns: {len(df.columns)}")
        summary.append(f"Duplicate Rows: {df.duplicated().sum()}")
        summary.append("-" * 50)
        
        # Nulls
        high_nulls = report[report['null_pct'] > 0]
        if not high_nulls.empty:
            summary.append("\nColumns with Missing Values:")
            for idx, row in high_nulls.iterrows():
                summary.append(f"  - {idx}: {row['null_count']} ({row['null_pct']:.1f}%)")
        else:
            summary.append("\nNo missing values found.")
            
        # Outliers (basic check using Z-score > 3 for numeric)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary.append("\nPotential Outliers (Z-score > 3):")
            from scipy import stats
            try:
                z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(df[numeric_cols].mean())))
                outliers = (z_scores > 3).sum(axis=0)
                outliers = outliers[outliers > 0]
                
                if not outliers.empty:
                    for col, count in outliers.items():
                        summary.append(f"  - {col}: {count} rows")
                else:
                    summary.append("  None detected.")
            except:
                summary.append("  Could not calculate outliers (possible constant values).")
                
        return "\n".join(summary)
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                           target_col: str,
                           clean: bool = True,
                           encode: bool = True,
                           normalize: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            clean: Whether to clean data
            encode: Whether to encode categorical variables
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (processed features DataFrame, target Series)
        """
        logger.info("Starting preprocessing pipeline...")
        
        df_processed = df.copy()
        
        # Clean data
        if clean:
            df_processed = self.clean_data(df_processed)
        
        # Detect types
        df_processed = self.detect_and_convert_types(df_processed)
        
        # Separate target
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        y = df_processed[target_col].copy()
        X = df_processed.drop(columns=[target_col])
        
        # Encode categorical
        if encode:
            X = self.encode_categorical(X, target_col=None, method='target')
        
        # Normalize
        if normalize:
            X = self.normalize_features(X, exclude_cols=[])
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        
        return X, y
