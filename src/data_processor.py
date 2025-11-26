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
                    df_typed[col] = pd.to_datetime(df_typed[col])
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
