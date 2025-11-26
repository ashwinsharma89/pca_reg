# PCA-Agent: Post-Campaign Analysis AI

Enterprise-grade Post-Campaign Analysis AI system for digital media campaign analytics.

## Features

- **15 Regression Models**: Linear, Lasso, Ridge, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, SVR, KNN, Huber, Bayesian Ridge
- **Automated Feature Engineering**: Media metrics (CPC, CPM, CTR, ROAS), time features, lag/rolling windows, adstock transformations
- **Hyperparameter Tuning**: Grid Search, Random Search, Optuna (Bayesian optimization)
- **Model Interpretation**: SHAP values, feature importance, permutation importance
- **Budget Optimization**: What-if analysis, optimal allocation recommendations
- **Comprehensive Reporting**: Executive summaries, performance insights, wasted spend analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
python main.py
```

This will run the agent with sample campaign data and generate:
- Model comparison results
- Feature importance analysis
- Performance insights
- Budget optimization recommendations
- Executive summary

## Usage with Your Data

```python
from main import run_pca_agent

results = run_pca_agent(
    data_path='your_campaign_data.csv',
    target_column='conversions',  # or 'revenue', 'leads', etc.
    tune_hyperparameters=True
)
```

## Project Structure

```
PCA_Regression/
├── src/
│   ├── config.py           # Configuration settings
│   ├── data_processor.py   # Data cleaning and preprocessing
│   ├── feature_engine.py   # Feature engineering
│   ├── modeler.py          # Model training and evaluation
│   ├── analyzer.py         # Insights and optimization
│   └── utils.py            # Helper functions
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── output/                 # Generated reports and results
```

## Supported Platforms

- Google Ads
- Meta Ads (Facebook/Instagram)
- DV360
- CM360
- Snapchat Ads
- Any digital media platform with performance data

## Output

The agent generates:
- `model_results.csv` - Comparison of all models
- `feature_importance.csv` - Top features driving performance
- `executive_summary.txt` - Non-technical summary for stakeholders

## License

MIT
