"""
Main Entry Point for PCA-Agent
Post-Campaign Analysis AI Agent
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngineer
from src.modeler import CampaignModeler
from src.analyzer import CampaignAnalyzer
from src.config import RANDOM_STATE, TEST_SIZE, MODELS_TO_TRAIN
from src.utils import create_output_directory, logger


def create_sample_campaign_data(n_rows: int = 1000) -> pd.DataFrame:
    """
    Create sample digital media campaign data for testing
    
    Args:
        n_rows: Number of rows to generate
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(RANDOM_STATE)
    
    # Generate sample data
    data = {
        'date': pd.date_range('2024-01-01', periods=n_rows, freq='H'),
        'platform': np.random.choice(['Google Ads', 'Meta Ads', 'DV360', 'Snapchat'], n_rows),
        'campaign': np.random.choice(['Campaign_A', 'Campaign_B', 'Campaign_C', 'Campaign_D'], n_rows),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_rows),
        'placement': np.random.choice(['Search', 'Display', 'Video', 'Social'], n_rows),
        'impressions': np.random.randint(1000, 100000, n_rows),
        'clicks': np.random.randint(10, 5000, n_rows),
        'cost': np.random.uniform(50, 5000, n_rows),
        'video_views': np.random.randint(0, 3000, n_rows),
        'engagements': np.random.randint(5, 1000, n_rows),
        'reach': np.random.randint(500, 50000, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable (conversions) with some realistic relationships
    df['conversions'] = (
        df['clicks'] * 0.05 +  # Base conversion from clicks
        df['engagements'] * 0.1 +  # Engagement impact
        np.where(df['platform'] == 'Google Ads', 10, 0) +  # Platform effect
        np.where(df['device'] == 'Desktop', 5, 0) +  # Device effect
        np.random.normal(0, 5, n_rows)  # Random noise
    ).clip(0)  # No negative conversions
    
    # Generate revenue
    df['revenue'] = df['conversions'] * np.random.uniform(20, 100, n_rows)
    
    return df


def run_pca_agent(data_path: str = None, 
                 target_column: str = 'conversions',
                 use_sample_data: bool = True,
                 tune_hyperparameters: bool = True,
                 models_to_train: list = None):
    """
    Run the complete PCA-Agent pipeline
    
    Args:
        data_path: Path to campaign data CSV
        target_column: Name of target column to predict
        use_sample_data: Whether to use sample data (for testing)
        tune_hyperparameters: Whether to tune model hyperparameters
        models_to_train: List of models to train (None = all)
    """
    logger.info("=" * 80)
    logger.info("PCA-AGENT: POST-CAMPAIGN ANALYSIS AI")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = 'output'
    create_output_directory(output_dir)
    
    # ========== STEP 1: LOAD DATA ==========
    logger.info("\n[STEP 1] Loading Data...")
    
    if use_sample_data or data_path is None:
        logger.info("Generating sample campaign data...")
        df = create_sample_campaign_data(n_rows=2000)
    else:
        processor = DataProcessor()
        df = processor.load_data(data_path)
    
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nData Preview:\n{df.head()}")
    
    # ========== STEP 2: DATA PREPROCESSING ==========
    logger.info("\n[STEP 2] Preprocessing Data...")
    
    processor = DataProcessor()
    
    # Clean data first
    df_clean = processor.clean_data(df)
    df_typed = processor.detect_and_convert_types(df_clean)
    
    # Separate target
    y = df_typed[target_column].copy()
    X = df_typed.drop(columns=[target_column])
    
    # Encode categorical variables with label encoding (simpler than target encoding)
    X_encoded = processor.encode_categorical(X, target_col=None, method='label')
    
    logger.info(f"Preprocessed data: {X_encoded.shape[1]} features")
    
    # ========== STEP 3: FEATURE ENGINEERING ==========
    logger.info("\n[STEP 3] Engineering Features...")
    
    engineer = FeatureEngineer()
    X_engineered = engineer.engineer_features_pipeline(
        X_encoded,
        date_col='date' if 'date' in X_encoded.columns else None,
        create_media=True,
        create_time=True,
        create_lags=False,  # Set to True for time series
        create_rolling=False,
        create_adstock=False
    )
    
    logger.info(f"Feature engineering complete: {X_engineered.shape[1]} features")
    
    # Remove date column if present (can't use in modeling)
    if 'date' in X_engineered.columns:
        X_engineered = X_engineered.drop(columns=['date'])
    
    # ========== STEP 4: TRAIN-TEST SPLIT ==========
    logger.info("\n[STEP 4] Splitting Data...")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # ========== STEP 5: MODEL TRAINING ==========
    logger.info("\n[STEP 5] Training Models...")
    
    modeler = CampaignModeler(random_state=RANDOM_STATE)
    modeler.initialize_models()
    
    if models_to_train is None:
        models_to_train = MODELS_TO_TRAIN
    
    # Train subset of fast models for demo (you can train all)
    quick_models = ['linear', 'lasso', 'ridge', 'elasticnet', 'random_forest', 
                   'gradient_boosting', 'xgboost']
    models_to_use = [m for m in quick_models if m in models_to_train]
    
    logger.info(f"Training {len(models_to_use)} models: {models_to_use}")
    
    modeler.train_all_models(
        X_train, y_train,
        model_names=models_to_use,
        tune_hyperparameters=tune_hyperparameters,
        tuning_method='grid' if tune_hyperparameters else None
    )
    
    # ========== STEP 6: MODEL EVALUATION ==========
    logger.info("\n[STEP 6] Evaluating Models...")
    
    results_df = modeler.evaluate_all_models(X_test, y_test)
    print(f"\n{results_df.to_string()}")
    
    # Save results
    results_df.to_csv(f'{output_dir}/model_results.csv', index=False)
    logger.info(f"Results saved to {output_dir}/model_results.csv")
    
    # ========== STEP 7: SELECT BEST MODEL ==========
    logger.info("\n[STEP 7] Selecting Best Model...")
    
    best_model_name, best_model = modeler.select_best_model(metric='R2')
    
    # Get feature importance
    feature_importance = modeler.get_feature_importance(
        best_model_name,
        feature_names=X_train.columns.tolist()
    )
    
    print(f"\nTop 10 Important Features:\n{feature_importance.head(10)}")
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    # ========== STEP 8: GENERATE INSIGHTS ==========
    logger.info("\n[STEP 8] Generating Insights...")
    
    analyzer = CampaignAnalyzer(
        best_model,
        best_model_name,
        X_train.columns.tolist()
    )
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    
    # Generate insights
    df_test = df.iloc[y_test.index].copy()
    insights = analyzer.generate_performance_insights(
        df_test,
        y_pred,
        actual_col=target_column,
        cost_col='cost'
    )
    
    print(f"\nOverall Performance:")
    for key, value in insights['overall'].items():
        print(f"  {key}: {value}")
    
    # ========== STEP 9: BUDGET OPTIMIZATION ==========
    logger.info("\n[STEP 9] Budget Optimization...")
    
    current_budget = df_test['cost'].sum()
    optimization = analyzer.simulate_budget_optimization(
        df_test,
        current_budget,
        target_col=target_column,
        cost_col='cost'
    )
    
    print(f"\nBudget Optimization Scenarios:")
    for scenario in optimization['scenarios']:
        print(f"\n{scenario['name']}:")
        print(f"  New Budget: ${scenario['new_budget']:,.2f}")
        print(f"  Expected Change: {scenario['target_change_pct']:.1f}%")
    
    # ========== STEP 10: EXECUTIVE SUMMARY ==========
    logger.info("\n[STEP 10] Generating Executive Summary...")
    
    summary = analyzer.generate_executive_summary(
        results_df,
        insights,
        optimization
    )
    
    print(f"\n{summary}")
    
    # Save summary
    with open(f'{output_dir}/executive_summary.txt', 'w') as f:
        f.write(summary)
    
    logger.info(f"\nExecutive summary saved to {output_dir}/executive_summary.txt")
    
    # ========== COMPLETE ==========
    logger.info("\n" + "=" * 80)
    logger.info("PCA-AGENT ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nAll outputs saved to: {output_dir}/")
    
    return {
        'model': best_model,
        'model_name': best_model_name,
        'results': results_df,
        'insights': insights,
        'optimization': optimization,
        'feature_importance': feature_importance
    }


if __name__ == "__main__":
    # Run with sample data
    results = run_pca_agent(
        use_sample_data=True,
        target_column='conversions',
        tune_hyperparameters=False  # Set to True for better results (slower)
    )
    
    print("\n[SUCCESS] PCA-Agent completed successfully!")
    print(f"Best Model: {results['model_name']}")
    print(f"R2 Score: {results['results'].iloc[0]['R2']:.4f}")
