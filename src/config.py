"""
Configuration settings for PCA-Agent
"""

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Regression Models to Train
MODELS_TO_TRAIN = [
    'linear',
    'lasso',
    'ridge',
    'elasticnet',
    'random_forest',
    'extra_trees',
    'gradient_boosting',
    'adaboost',
    'xgboost',
    'lightgbm',
    'catboost',
    'svr',
    'knn',
    'huber',
    'bayesian_ridge'
]

# Hyperparameter Grids
PARAM_GRIDS = {
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    'elasticnet': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    'extra_trees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0]
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['rbf', 'linear']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'huber': {
        'epsilon': [1.1, 1.35, 1.5, 2.0],
        'alpha': [0.0001, 0.001, 0.01, 0.1]
    },
    'bayesian_ridge': {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    }
}

# Feature Engineering Settings
ADSTOCK_DECAY_RATES = [0.3, 0.5, 0.7, 0.9]
LAG_PERIODS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Evaluation Metrics
METRICS = ['rmse', 'mae', 'mape', 'r2', 'adjusted_r2']

# Output Settings
OUTPUT_DIR = 'output'
REPORT_FORMAT = 'html'  # 'html', 'pdf', 'excel'
