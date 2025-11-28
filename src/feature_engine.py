"""
Feature Engineering Module for PCA-Agent
Creates digital media-specific features for campaign analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging
from src.utils import logger


class FeatureEngineer:
    """
    Handles feature engineering for digital media campaign data
    """
    
    def __init__(self):
        self.created_features = []
        
    def create_media_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create standard digital media performance metrics
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional media metrics
        """
        logger.info("Creating digital media metrics...")
        df_features = df.copy()
        
        # Cost per Click (CPC)
        if 'cost' in df_features.columns and 'clicks' in df_features.columns:
            df_features['cpc'] = df_features['cost'] / (df_features['clicks'] + 1)
            self.created_features.append('cpc')
        
        # Cost per Mille (CPM)
        if 'cost' in df_features.columns and 'impressions' in df_features.columns:
            df_features['cpm'] = (df_features['cost'] / (df_features['impressions'] + 1)) * 1000
            self.created_features.append('cpm')
        
        # Click-Through Rate (CTR)
        if 'clicks' in df_features.columns and 'impressions' in df_features.columns:
            df_features['ctr'] = (df_features['clicks'] / (df_features['impressions'] + 1)) * 100
            self.created_features.append('ctr')
        
        # Conversion Rate (CVR)
        if 'conversions' in df_features.columns and 'clicks' in df_features.columns:
            df_features['cvr'] = (df_features['conversions'] / (df_features['clicks'] + 1)) * 100
            self.created_features.append('cvr')
        
        # Cost per Conversion (CPA)
        if 'cost' in df_features.columns and 'conversions' in df_features.columns:
            df_features['cpa'] = df_features['cost'] / (df_features['conversions'] + 1)
            self.created_features.append('cpa')
        
        # Return on Ad Spend (ROAS)
        if 'revenue' in df_features.columns and 'cost' in df_features.columns:
            df_features['roas'] = df_features['revenue'] / (df_features['cost'] + 1)
            self.created_features.append('roas')
        
        # Video Completion Rate
        if 'video_views' in df_features.columns and 'impressions' in df_features.columns:
            df_features['video_completion_rate'] = (df_features['video_views'] / (df_features['impressions'] + 1)) * 100
            self.created_features.append('video_completion_rate')
        
        # Engagement Rate
        if 'engagements' in df_features.columns and 'impressions' in df_features.columns:
            df_features['engagement_rate'] = (df_features['engagements'] / (df_features['impressions'] + 1)) * 100
            self.created_features.append('engagement_rate')
        
        # Frequency (if reach is available)
        if 'impressions' in df_features.columns and 'reach' in df_features.columns:
            df_features['frequency'] = df_features['impressions'] / (df_features['reach'] + 1)
            self.created_features.append('frequency')
        
        logger.info(f"Created {len(self.created_features)} media metrics")
        return df_features
    
    def create_share_metrics(self, df: pd.DataFrame, group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create share of spend/impressions metrics
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by (e.g., ['date', 'campaign'])
            
        Returns:
            DataFrame with share metrics
        """
        logger.info("Creating share metrics...")
        df_features = df.copy()
        
        if group_by is None:
            group_by = []
        
        # Share of Spend
        if 'cost' in df_features.columns:
            if group_by:
                df_features['share_of_spend'] = df_features.groupby(group_by)['cost'].transform(
                    lambda x: x / x.sum() * 100
                )
            else:
                total_cost = df_features['cost'].sum()
                df_features['share_of_spend'] = (df_features['cost'] / total_cost) * 100
            self.created_features.append('share_of_spend')
        
        # Share of Impressions
        if 'impressions' in df_features.columns:
            if group_by:
                df_features['share_of_impressions'] = df_features.groupby(group_by)['impressions'].transform(
                    lambda x: x / x.sum() * 100
                )
            else:
                total_impr = df_features['impressions'].sum()
                df_features['share_of_impressions'] = (df_features['impressions'] / total_impr) * 100
            self.created_features.append('share_of_impressions')
        
        # Share of Conversions
        if 'conversions' in df_features.columns:
            if group_by:
                df_features['share_of_conversions'] = df_features.groupby(group_by)['conversions'].transform(
                    lambda x: x / (x.sum() + 1) * 100
                )
            else:
                total_conv = df_features['conversions'].sum()
                df_features['share_of_conversions'] = (df_features['conversions'] / (total_conv + 1)) * 100
            self.created_features.append('share_of_conversions')
        
        return df_features
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time-based features...")
        df_features = df.copy()
        
        if date_col not in df_features.columns:
            logger.warning(f"Date column '{date_col}' not found. Skipping time features.")
            return df_features
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[date_col]):
            df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        # Day of week
        df_features['day_of_week'] = df_features[date_col].dt.dayofweek
        self.created_features.append('day_of_week')
        
        # Is weekend
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        self.created_features.append('is_weekend')
        
        # Month
        df_features['month'] = df_features[date_col].dt.month
        self.created_features.append('month')
        
        # Quarter
        df_features['quarter'] = df_features[date_col].dt.quarter
        self.created_features.append('quarter')
        
        # Week of year
        df_features['week_of_year'] = df_features[date_col].dt.isocalendar().week
        self.created_features.append('week_of_year')
        
        # Day of month
        df_features['day_of_month'] = df_features[date_col].dt.day
        self.created_features.append('day_of_month')
        
        # Seasonality (sin/cos transformations)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        self.created_features.extend(['month_sin', 'month_cos'])
        
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        self.created_features.extend(['day_sin', 'day_cos'])
        
        logger.info(f"Created {10} time-based features")
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lags: List[int] = [1, 7, 14],
                           group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create lag features for time series analysis
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            group_by: Columns to group by
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {len(columns)} columns...")
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                continue
                
            for lag in lags:
                lag_col_name = f'{col}_lag_{lag}'
                
                if group_by:
                    df_features[lag_col_name] = df_features.groupby(group_by)[col].shift(lag)
                else:
                    df_features[lag_col_name] = df_features[col].shift(lag)
                
                self.created_features.append(lag_col_name)
        
        # Fill NaN values created by lagging
        df_features = df_features.fillna(0)
        
        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int] = [7, 14, 30],
                               group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            group_by: Columns to group by
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {len(columns)} columns...")
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                continue
                
            for window in windows:
                # Rolling mean
                mean_col_name = f'{col}_rolling_mean_{window}'
                if group_by:
                    df_features[mean_col_name] = df_features.groupby(group_by)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                else:
                    df_features[mean_col_name] = df_features[col].rolling(window=window, min_periods=1).mean()
                self.created_features.append(mean_col_name)
                
                # Rolling std
                std_col_name = f'{col}_rolling_std_{window}'
                if group_by:
                    df_features[std_col_name] = df_features.groupby(group_by)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                else:
                    df_features[std_col_name] = df_features[col].rolling(window=window, min_periods=1).std()
                self.created_features.append(std_col_name)
        
        # Fill NaN values
        df_features = df_features.fillna(0)
        
        logger.info(f"Created {len(columns) * len(windows) * 2} rolling features")
        return df_features
    
    def create_adstock_features(self, df: pd.DataFrame,
                               columns: List[str],
                               decay_rates: List[float] = [0.5, 0.7, 0.9],
                               max_lag: int = 7) -> pd.DataFrame:
        """
        Create adstock transformation features (carryover effect)
        
        Args:
            df: Input DataFrame
            columns: Columns to apply adstock to
            decay_rates: List of decay rates (0-1)
            max_lag: Maximum lag period
            
        Returns:
            DataFrame with adstock features
        """
        logger.info(f"Creating adstock features for {len(columns)} columns...")
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                continue
                
            for decay in decay_rates:
                adstock_col_name = f'{col}_adstock_{int(decay*100)}'
                adstock_values = []
                
                for i in range(len(df_features)):
                    adstock = 0
                    for lag in range(max_lag + 1):
                        if i - lag >= 0:
                            adstock += df_features[col].iloc[i - lag] * (decay ** lag)
                    adstock_values.append(adstock)
                
                df_features[adstock_col_name] = adstock_values
                self.created_features.append(adstock_col_name)
        
        logger.info(f"Created {len(columns) * len(decay_rates)} adstock features")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples with feature pairs to interact
            
        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating {len(feature_pairs)} interaction features...")
        df_features = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_features.columns and feat2 in df_features.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                df_features[interaction_name] = df_features[feat1] * df_features[feat2]
                self.created_features.append(interaction_name)
        
        return df_features
    
    def engineer_features_pipeline(self, df: pd.DataFrame,
                                  date_col: Optional[str] = 'date',
                                  create_media: bool = True,
                                  create_time: bool = True,
                                  create_lags: bool = False,
                                  create_rolling: bool = False,
                                  create_adstock: bool = False,
                                  lag_columns: Optional[List[str]] = None,
                                  rolling_columns: Optional[List[str]] = None,
                                  adstock_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            create_media: Create media metrics
            create_time: Create time features
            create_lags: Create lag features
            create_rolling: Create rolling features
            create_adstock: Create adstock features
            lag_columns: Columns for lag features
            rolling_columns: Columns for rolling features
            adstock_columns: Columns for adstock features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        df_engineered = df.copy()
        
        # Media metrics
        if create_media:
            df_engineered = self.create_media_metrics(df_engineered)
        
        # Time features
        if create_time and date_col and date_col in df_engineered.columns:
            df_engineered = self.create_time_features(df_engineered, date_col)
        
        # Lag features
        if create_lags and lag_columns:
            df_engineered = self.create_lag_features(df_engineered, lag_columns)
        
        # Rolling features
        if create_rolling and rolling_columns:
            df_engineered = self.create_rolling_features(df_engineered, rolling_columns)
        
        # Adstock features
        if create_adstock and adstock_columns:
            df_engineered = self.create_adstock_features(df_engineered, adstock_columns)
        
        logger.info(f"Feature engineering complete. Total features created: {len(self.created_features)}")
        logger.info(f"Final shape: {df_engineered.shape}")
        
        return df_engineered
    
    def select_features(self, df: pd.DataFrame, 
                       target_col: str,
                       method: str = 'correlation',
                       threshold: float = 0.95) -> pd.DataFrame:
        """
        Select features to reduce multicollinearity and noise
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            method: Selection method ('correlation', 'vif')
            threshold: Threshold for removal (correlation coef or VIF value)
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features using {method} method...")
        
        # Separate target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df.copy()
            
        # Keep only numeric columns for selection
        X_numeric = X.select_dtypes(include=[np.number])
        dropped_cols = []
        
        if method == 'correlation':
            # Calculate correlation matrix
            corr_matrix = X_numeric.corr().abs()
            
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            dropped_cols = [column for column in upper.columns if any(upper[column] > threshold)]
            
            logger.info(f"Dropping {len(dropped_cols)} features due to high correlation (> {threshold})")
            
        elif method == 'vif':
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Iteratively drop features with high VIF
            X_vif = X_numeric.copy()
            X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()
            
            while True:
                vif_data = pd.DataFrame()
                vif_data["feature"] = X_vif.columns
                try:
                    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                                      for i in range(len(X_vif.columns))]
                except:
                    break
                
                max_vif = vif_data["VIF"].max()
                if max_vif > threshold:
                    feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
                    X_vif = X_vif.drop(columns=[feature_to_drop])
                    dropped_cols.append(feature_to_drop)
                else:
                    break
            
            logger.info(f"Dropping {len(dropped_cols)} features due to high VIF (> {threshold})")
            
        # Drop selected columns from original dataframe
        df_selected = df.drop(columns=dropped_cols)
        
        return df_selected
