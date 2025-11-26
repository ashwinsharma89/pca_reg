"""
Unit Tests for PCA-Agent
Pytest test suite for all modules
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngineer
from src.modeler import CampaignModeler
from src.analyzer import CampaignAnalyzer
from main import create_sample_campaign_data


class TestDataProcessor:
    """Tests for DataProcessor class"""
    
    def test_load_csv(self, tmp_path):
        """Test CSV loading"""
        # Create temp CSV
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and verify
        processor = DataProcessor()
        loaded_df = processor.load_data(str(csv_path))
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['a', 'b']
    
    def test_clean_data(self):
        """Test data cleaning"""
        # Create dirty data
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 4],
            'b': [1, 2, 3, 4, 4]
        })
        
        processor = DataProcessor()
        cleaned = processor.clean_data(df, drop_duplicates=True, handle_nulls='drop')
        
        assert len(cleaned) == 3  # Removed null and duplicate
        assert cleaned.isnull().sum().sum() == 0
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4]
        })
        
        processor = DataProcessor()
        encoded = processor.encode_categorical(df, method='label', categorical_cols=['category'])
        
        assert encoded['category'].dtype in [np.int32, np.int64]


class TestFeatureEngineer:
    """Tests for FeatureEngineer class"""
    
    def test_create_media_metrics(self):
        """Test media metrics creation"""
        df = pd.DataFrame({
            'impressions': [1000, 2000, 3000],
            'clicks': [10, 20, 30],
            'cost': [100, 200, 300],
            'conversions': [1, 2, 3]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_media_metrics(df)
        
        assert 'cpc' in result.columns
        assert 'cpm' in result.columns
        assert 'ctr' in result.columns
        assert 'cpa' in result.columns
    
    def test_create_time_features(self):
        """Test time feature creation"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': range(10)
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_time_features(df, date_col='date')
        
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'month' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
    
    def test_create_lag_features(self):
        """Test lag feature creation"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_lag_features(df, columns=['value'], lags=[1, 2])
        
        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns


class TestCampaignModeler:
    """Tests for CampaignModeler class"""
    
    def test_initialize_models(self):
        """Test model initialization"""
        modeler = CampaignModeler()
        models = modeler.initialize_models()
        
        assert len(models) == 15
        assert 'linear' in models
        assert 'xgboost' in models
        assert 'random_forest' in models
    
    def test_train_model(self):
        """Test single model training"""
        # Create sample data
        X = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
        y = pd.Series(np.random.rand(100))
        
        modeler = CampaignModeler()
        modeler.initialize_models()
        
        # Train linear model
        model = modeler.train_model('linear', X, y, tune_hyperparameters=False)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create sample data
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.rand(100))
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = pd.Series(np.random.rand(20))
        
        modeler = CampaignModeler()
        modeler.initialize_models()
        modeler.train_model('linear', X_train, y_train)
        
        metrics = modeler.evaluate_model('linear', X_test, y_test)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'MAPE' in metrics


class TestCampaignAnalyzer:
    """Tests for CampaignAnalyzer class"""
    
    def test_generate_insights(self):
        """Test insights generation"""
        # Create sample data
        df = pd.DataFrame({
            'platform': ['Google Ads', 'Meta Ads'] * 50,
            'cost': np.random.rand(100) * 1000,
            'conversions': np.random.rand(100) * 100
        })
        
        predictions = np.random.rand(100) * 100
        
        # Create dummy model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.rand(100)
        model.fit(X_dummy, y_dummy)
        
        analyzer = CampaignAnalyzer(model, 'linear', ['a', 'b', 'c', 'd', 'e'])
        insights = analyzer.generate_performance_insights(
            df, predictions, 'conversions', 'cost'
        )
        
        assert 'overall' in insights
        assert 'mean_prediction' in insights['overall']
        assert 'mean_actual' in insights['overall']
    
    def test_budget_optimization(self):
        """Test budget optimization simulation"""
        df = pd.DataFrame({
            'cost': [1000] * 10,
            'conversions': [50] * 10
        })
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        
        analyzer = CampaignAnalyzer(model, 'linear', ['a', 'b', 'c', 'd', 'e'])
        optimization = analyzer.simulate_budget_optimization(
            df, 10000, 'conversions', 'cost'
        )
        
        assert 'current' in optimization
        assert 'scenarios' in optimization
        assert len(optimization['scenarios']) > 0


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline"""
        # Create sample data
        df = create_sample_campaign_data(n_rows=200)
        
        # Data processing
        processor = DataProcessor()
        df_clean = processor.clean_data(df)
        df_typed = processor.detect_and_convert_types(df_clean)
        
        y = df_typed['conversions'].copy()
        X = df_typed.drop(columns=['conversions'])
        X_encoded = processor.encode_categorical(X, method='label')
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_engineered = engineer.engineer_features_pipeline(
            X_encoded,
            create_media=True,
            create_time=False,
            create_lags=False
        )
        
        # Remove date column
        if 'date' in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=['date'])
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42
        )
        
        # Model training
        modeler = CampaignModeler(random_state=42)
        modeler.initialize_models()
        modeler.train_model('linear', X_train, y_train)
        
        # Evaluation
        metrics = modeler.evaluate_model('linear', X_test, y_test)
        
        assert metrics['R2'] > 0  # Should have some predictive power
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0


# Pytest fixtures
@pytest.fixture
def sample_data():
    """Fixture for sample campaign data"""
    return create_sample_campaign_data(n_rows=100)


@pytest.fixture
def trained_modeler(sample_data):
    """Fixture for trained modeler"""
    processor = DataProcessor()
    df_clean = processor.clean_data(sample_data)
    df_typed = processor.detect_and_convert_types(df_clean)
    
    y = df_typed['conversions'].copy()
    X = df_typed.drop(columns=['conversions'])
    X_encoded = processor.encode_categorical(X, method='label')
    
    if 'date' in X_encoded.columns:
        X_encoded = X_encoded.drop(columns=['date'])
    
    modeler = CampaignModeler(random_state=42)
    modeler.initialize_models()
    modeler.train_model('linear', X_encoded, y)
    
    return modeler


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
