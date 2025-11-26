"""
Utility functions for PCA-Agent
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect categorical and numerical columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with 'categorical' and 'numerical' column lists
    """
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category', 'bool']:
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numerical_cols.append(col)
    
    logger.info(f"Detected {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
    
    return {
        'categorical': categorical_cols,
        'numerical': numerical_cols
    }


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity detection
    
    Args:
        df: Input DataFrame
        features: List of feature column names
        
    Returns:
        DataFrame with VIF scores
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(df[features].values, i) 
        for i in range(len(features))
    ]
    
    return vif_data.sort_values('VIF', ascending=False)


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove highly correlated features
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold
        
    Returns:
        List of columns to keep
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > threshold)
    ]
    
    logger.info(f"Removing {len(to_drop)} highly correlated features")
    
    return [col for col in df.columns if col not in to_drop]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_features: Number of features used
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    n = len(y_true)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adjusted_R2': adjusted_r2,
        'MAPE': mape
    }


def create_output_directory(path: str) -> None:
    """
    Create output directory if it doesn't exist
    
    Args:
        path: Directory path
    """
    import os
    os.makedirs(path, exist_ok=True)
    logger.info(f"Output directory created/verified: {path}")
