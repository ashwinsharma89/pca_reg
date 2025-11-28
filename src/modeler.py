"""
Modeling Module for PCA-Agent
Handles model training, hyperparameter tuning, and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet, 
    HuberRegressor, BayesianRidge
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, PARAM_GRIDS
from src.utils import calculate_metrics, logger


class CampaignModeler:
    """
    Handles model training, tuning, and evaluation for campaign analysis
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all regression models
        
        Returns:
            Dictionary of model instances
        """
        logger.info("Initializing regression models...")
        
        self.models = {
            'linear': LinearRegression(),
            'lasso': Lasso(random_state=self.random_state),
            'ridge': Ridge(random_state=self.random_state),
            'elasticnet': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'extra_trees': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'adaboost': AdaBoostRegressor(random_state=self.random_state),
            'xgboost': XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0),
            'lightgbm': LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1),
            'catboost': CatBoostRegressor(random_state=self.random_state, verbose=0),
            'svr': SVR(),
            'knn': KNeighborsRegressor(n_jobs=-1),
            'huber': HuberRegressor(),
            'bayesian_ridge': BayesianRidge()
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   tune_hyperparameters: bool = False,
                   tuning_method: str = 'grid',
                   cv_folds: int = CV_FOLDS) -> Any:
        """
        Train a single model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to tune hyperparameters
            tuning_method: 'grid', 'random', or 'optuna'
            cv_folds: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Initialize models first.")
        
        model = self.models[model_name]
        
        if tune_hyperparameters and model_name in PARAM_GRIDS:
            logger.info(f"Tuning hyperparameters for {model_name} using {tuning_method}...")
            
            if tuning_method == 'grid':
                search = GridSearchCV(
                    model,
                    PARAM_GRIDS[model_name],
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                logger.info(f"Best params for {model_name}: {search.best_params_}")
                
            elif tuning_method == 'random':
                search = RandomizedSearchCV(
                    model,
                    PARAM_GRIDS[model_name],
                    n_iter=20,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                logger.info(f"Best params for {model_name}: {search.best_params_}")
                
            elif tuning_method == 'optuna':
                model = self._tune_with_optuna(model_name, X_train, y_train, cv_folds)
        else:
            model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        logger.info(f"Training complete for {model_name}")
        
        return model
    
    def _tune_with_optuna(self, model_name: str, X_train: pd.DataFrame, 
                         y_train: pd.Series, cv_folds: int) -> Any:
        """
        Tune hyperparameters using Optuna
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            
        Returns:
            Tuned model
        """
        def objective(trial):
            params = {}
            
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBRegressor(**params, random_state=self.random_state, n_jobs=-1, verbosity=0)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100)
                }
                model = LGBMRegressor(**params, random_state=self.random_state, n_jobs=-1, verbose=-1)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                model = RandomForestRegressor(**params, random_state=self.random_state, n_jobs=-1)
            else:
                # Default to grid search params
                return 0
            
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        logger.info(f"Best params for {model_name}: {study.best_params}")
        
        # Train final model with best params
        if model_name == 'xgboost':
            model = XGBRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1, verbosity=0)
        elif model_name == 'lightgbm':
            model = LGBMRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1, verbose=-1)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1)
        
        model.fit(X_train, y_train)
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        model_names: Optional[List[str]] = None,
                        tune_hyperparameters: bool = True,
                        tuning_method: str = 'grid') -> Dict[str, Any]:
        """
        Train all specified models
        
        Args:
            X_train: Training features
            y_train: Training target
            model_names: List of model names to train (None = all)
            tune_hyperparameters: Whether to tune hyperparameters
            tuning_method: Tuning method to use
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Starting training for all models...")
        
        if not self.models:
            self.initialize_models()
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        for model_name in model_names:
            try:
                self.train_model(model_name, X_train, y_train, 
                               tune_hyperparameters, tuning_method)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Training complete for {len(self.trained_models)} models")
        return self.trained_models
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test.values, y_pred, X_test.shape[1])
        metrics['Model'] = model_name
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, X_test: pd.DataFrame, 
                           y_test: pd.Series,
                           X_full: Optional[pd.DataFrame] = None,
                           y_full: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Evaluate all trained models with CV support
        
        Args:
            X_test: Test features
            y_test: Test target
            X_full: Full features (for CV)
            y_full: Full target (for CV)
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info("Evaluating all trained models...")
        
        results_list = []
        
        from sklearn.model_selection import cross_validate
        
        for model_name in self.trained_models.keys():
            try:
                # Standard evaluation
                metrics = self.evaluate_model(model_name, X_test, y_test)
                
                # Cross-validation if full data provided
                if X_full is not None and y_full is not None:
                    cv_results = cross_validate(
                        self.trained_models[model_name], 
                        X_full, 
                        y_full, 
                        cv=CV_FOLDS,
                        scoring=['r2', 'neg_root_mean_squared_error'],
                        n_jobs=-1
                    )
                    metrics['CV_R2_Mean'] = cv_results['test_r2'].mean()
                    metrics['CV_R2_Std'] = cv_results['test_r2'].std()
                    metrics['CV_RMSE_Mean'] = -cv_results['test_neg_root_mean_squared_error'].mean()
                
                results_list.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results_list)
        if not results_df.empty:
            results_df = results_df.sort_values('R2', ascending=False)
        
        logger.info("Evaluation complete")
        
        return results_df
    
    def select_best_model(self, metric: str = 'R2', 
                         ascending: bool = False) -> Tuple[str, Any]:
        """
        Select the best performing model
        
        Args:
            metric: Metric to use for selection
            ascending: Sort order
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        results_df = pd.DataFrame(list(self.results.values()))
        results_df = results_df.sort_values(metric, ascending=ascending)
        
        best_model_name = results_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with {metric}={results_df.iloc[0][metric]:.4f}")
        
        return best_model_name, self.best_model
    
    def get_feature_importance(self, model_name: Optional[str] = None,
                              feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based or linear models
        
        Args:
            model_name: Name of the model (uses best model if None)
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Run select_best_model first.")
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models[model_name]
        
        importance_df = pd.DataFrame()
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_df['Feature'] = feature_names if feature_names else range(len(model.feature_importances_))
            importance_df['Importance'] = model.feature_importances_
            
        # Linear models
        elif hasattr(model, 'coef_'):
            importance_df['Feature'] = feature_names if feature_names else range(len(model.coef_))
            importance_df['Importance'] = np.abs(model.coef_)
        
        if not importance_df.empty:
            importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def run_diagnostics(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Run diagnostics on the best model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of diagnostic results
        """
        if self.best_model is None:
            raise ValueError("No best model selected")
            
        from src.analyzer import ModelDiagnostics
        
        predictions = self.best_model.predict(X_test)
        residuals = y_test - predictions
        
        diagnostics = {
            'normality': ModelDiagnostics.check_normality(residuals),
            'heteroscedasticity': ModelDiagnostics.check_heteroscedasticity(residuals, predictions),
            'vif': ModelDiagnostics.check_multicollinearity(X_test)
        }
        
        return diagnostics
