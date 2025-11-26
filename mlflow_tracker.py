"""
MLflow Integration for PCA-Agent
Track experiments, models, and metrics
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from datetime import datetime
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking for PCA-Agent
    """
    
    def __init__(self, experiment_name: str = "PCA-Agent", tracking_uri: str = "mlruns"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (local path or remote server)
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")
    
    def start_run(self, run_name: str = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, model_name: str, signature=None):
        """
        Log model to MLflow
        
        Args:
            model: Trained model
            model_name: Name for the model
            signature: MLflow model signature
        """
        # Determine model type and log accordingly
        model_type = type(model).__name__
        
        if 'XGB' in model_type:
            mlflow.xgboost.log_model(model, model_name, signature=signature)
        elif 'LGBM' in model_type or 'LightGBM' in model_type:
            mlflow.lightgbm.log_model(model, model_name, signature=signature)
        elif 'CatBoost' in model_type:
            mlflow.catboost.log_model(model, model_name, signature=signature)
        else:
            mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        logger.info(f"Model logged: {model_name}")
    
    def log_artifact(self, filepath: str):
        """Log artifact file"""
        mlflow.log_artifact(filepath)
    
    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """Log DataFrame as artifact"""
        temp_path = f"temp_{filename}"
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        import os
        os.unlink(temp_path)
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")
    
    def log_campaign_analysis(
        self,
        model,
        model_name: str,
        results_df: pd.DataFrame,
        feature_importance: pd.DataFrame,
        config: Dict[str, Any]
    ):
        """
        Log complete campaign analysis to MLflow
        
        Args:
            model: Best trained model
            model_name: Name of the best model
            results_df: Model comparison results
            feature_importance: Feature importance DataFrame
            config: Configuration parameters
        """
        try:
            # Start run
            self.start_run(f"campaign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Log parameters
            self.log_params({
                'best_model': model_name,
                'target_column': config.get('target_column', 'unknown'),
                'tune_hyperparameters': config.get('tune_hyperparameters', False),
                'test_size': config.get('test_size', 0.2),
                'random_state': config.get('random_state', 42)
            })
            
            # Log metrics from best model
            best_result = results_df.iloc[0]
            self.log_metrics({
                'r2_score': float(best_result['R2']),
                'rmse': float(best_result['RMSE']),
                'mae': float(best_result['MAE']),
                'mape': float(best_result['MAPE']),
                'adjusted_r2': float(best_result['Adjusted_R2'])
            })
            
            # Log model
            self.log_model(model, model_name)
            
            # Log artifacts
            self.log_dataframe(results_df, 'model_results.csv')
            self.log_dataframe(feature_importance, 'feature_importance.csv')
            
            # Log output files if they exist
            import os
            if os.path.exists('output/executive_summary.txt'):
                self.log_artifact('output/executive_summary.txt')
            
            # End run
            self.end_run()
            
            logger.info("Campaign analysis logged to MLflow successfully")
            
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
            mlflow.end_run(status='FAILED')
            raise


def track_model_training(func):
    """
    Decorator to automatically track model training with MLflow
    
    Usage:
        @track_model_training
        def train_models(...):
            ...
    """
    def wrapper(*args, **kwargs):
        tracker = MLflowTracker()
        tracker.start_run()
        
        try:
            result = func(*args, **kwargs)
            mlflow.end_run(status='FINISHED')
            return result
        except Exception as e:
            mlflow.end_run(status='FAILED')
            raise e
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = MLflowTracker(experiment_name="PCA-Agent-Test")
    
    # Start a run
    tracker.start_run("test_run")
    
    # Log some parameters
    tracker.log_params({
        'model': 'ridge',
        'alpha': 1.0,
        'target': 'conversions'
    })
    
    # Log some metrics
    tracker.log_metrics({
        'r2': 0.993,
        'rmse': 6.38,
        'mae': 5.19
    })
    
    # End run
    tracker.end_run()
    
    print("MLflow tracking example completed")
