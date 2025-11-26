"""
Analysis and Insights Module for PCA-Agent
Handles model interpretation, insights generation, and budget optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import logger


class CampaignAnalyzer:
    """
    Handles model interpretation, insights generation, and optimization recommendations
    """
    
    def __init__(self, model, model_name: str, feature_names: List[str]):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
        
    def calculate_shap_values(self, X: pd.DataFrame, 
                             sample_size: Optional[int] = 100) -> np.ndarray:
        """
        Calculate SHAP values for model interpretation
        
        Args:
            X: Feature DataFrame
            sample_size: Number of samples to use (for performance)
            
        Returns:
            SHAP values array
        """
        logger.info("Calculating SHAP values...")
        
        # Sample data if too large
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Choose appropriate explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            # For classifiers (shouldn't happen in regression, but just in case)
            self.explainer = shap.Explainer(self.model, X_sample)
        elif self.model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting']:
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Linear models and others
            self.explainer = shap.Explainer(self.model.predict, X_sample)
        
        self.shap_values = self.explainer(X_sample)
        
        logger.info("SHAP values calculated successfully")
        return self.shap_values
    
    def get_feature_importance_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Get mean absolute SHAP values
        shap_importance = np.abs(self.shap_values.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': shap_importance
        })
        
        importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
        
        return importance_df
    
    def plot_shap_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create SHAP summary plot
        
        Args:
            X: Feature DataFrame
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_shap_waterfall(self, X: pd.DataFrame, index: int = 0, 
                           save_path: Optional[str] = None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X: Feature DataFrame
            index: Index of the sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(self.shap_values[index], show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        plt.close()
    
    def generate_performance_insights(self, df: pd.DataFrame, 
                                     predictions: np.ndarray,
                                     actual_col: str,
                                     cost_col: str = 'cost') -> Dict[str, Any]:
        """
        Generate performance insights from the data
        
        Args:
            df: Original DataFrame with features
            predictions: Model predictions
            actual_col: Name of actual target column
            cost_col: Name of cost column
            
        Returns:
            Dictionary of insights
        """
        logger.info("Generating performance insights...")
        
        insights = {}
        
        # Add predictions to dataframe
        df_analysis = df.copy()
        df_analysis['predictions'] = predictions
        df_analysis['actual'] = df_analysis[actual_col]
        
        # Calculate residuals
        df_analysis['residual'] = df_analysis['actual'] - df_analysis['predictions']
        df_analysis['abs_residual'] = np.abs(df_analysis['residual'])
        df_analysis['pct_error'] = (df_analysis['abs_residual'] / (df_analysis['actual'] + 1)) * 100
        
        # Overall performance
        insights['overall'] = {
            'mean_prediction': df_analysis['predictions'].mean(),
            'mean_actual': df_analysis['actual'].mean(),
            'total_cost': df_analysis[cost_col].sum() if cost_col in df_analysis.columns else None,
            'mean_residual': df_analysis['residual'].mean(),
            'mean_abs_error': df_analysis['abs_residual'].mean(),
            'mean_pct_error': df_analysis['pct_error'].mean()
        }
        
        # Platform/Channel insights (if available)
        if 'platform' in df_analysis.columns:
            platform_insights = df_analysis.groupby('platform').agg({
                'actual': 'sum',
                'predictions': 'sum',
                cost_col: 'sum' if cost_col in df_analysis.columns else 'count',
                'pct_error': 'mean'
            }).round(2)
            
            insights['by_platform'] = platform_insights.to_dict('index')
        
        # Device insights (if available)
        if 'device' in df_analysis.columns:
            device_insights = df_analysis.groupby('device').agg({
                'actual': 'sum',
                'predictions': 'sum',
                cost_col: 'sum' if cost_col in df_analysis.columns else 'count',
                'pct_error': 'mean'
            }).round(2)
            
            insights['by_device'] = device_insights.to_dict('index')
        
        # Top performers (lowest error)
        insights['top_performers'] = df_analysis.nsmallest(10, 'pct_error')[
            ['predictions', 'actual', 'pct_error']
        ].to_dict('records')
        
        # Worst performers (highest error)
        insights['worst_performers'] = df_analysis.nlargest(10, 'pct_error')[
            ['predictions', 'actual', 'pct_error']
        ].to_dict('records')
        
        logger.info("Performance insights generated")
        return insights
    
    def identify_wasted_spend(self, df: pd.DataFrame,
                             cost_col: str = 'cost',
                             efficiency_metric: str = 'cpa',
                             threshold_percentile: float = 75) -> pd.DataFrame:
        """
        Identify campaigns/placements with wasted spend
        
        Args:
            df: DataFrame with campaign data
            cost_col: Name of cost column
            efficiency_metric: Metric to use for efficiency (cpa, cpc, cpm)
            threshold_percentile: Percentile threshold for inefficiency
            
        Returns:
            DataFrame with wasted spend analysis
        """
        logger.info("Identifying wasted spend...")
        
        if efficiency_metric not in df.columns:
            logger.warning(f"{efficiency_metric} not found in data")
            return pd.DataFrame()
        
        # Calculate threshold
        threshold = df[efficiency_metric].quantile(threshold_percentile / 100)
        
        # Identify inefficient spend
        wasted_df = df[df[efficiency_metric] > threshold].copy()
        wasted_df['wasted_amount'] = wasted_df[cost_col]
        wasted_df['efficiency_vs_threshold'] = (
            (wasted_df[efficiency_metric] - threshold) / threshold * 100
        )
        
        wasted_df = wasted_df.sort_values('wasted_amount', ascending=False)
        
        total_wasted = wasted_df['wasted_amount'].sum()
        total_spend = df[cost_col].sum()
        wasted_pct = (total_wasted / total_spend) * 100
        
        logger.info(f"Identified ${total_wasted:,.2f} ({wasted_pct:.1f}%) in potentially wasted spend")
        
        return wasted_df
    
    def simulate_budget_optimization(self, df: pd.DataFrame,
                                    current_budget: float,
                                    target_col: str,
                                    cost_col: str = 'cost',
                                    scenarios: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Simulate budget optimization scenarios
        
        Args:
            df: DataFrame with campaign data
            current_budget: Current total budget
            target_col: Target metric (conversions, revenue, etc.)
            cost_col: Cost column name
            scenarios: List of scenario dictionaries
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running budget optimization simulation...")
        
        if scenarios is None:
            scenarios = [
                {'name': 'Increase 10%', 'budget_change': 1.10},
                {'name': 'Increase 20%', 'budget_change': 1.20},
                {'name': 'Decrease 10%', 'budget_change': 0.90},
                {'name': 'Decrease 20%', 'budget_change': 0.80}
            ]
        
        results = {
            'current': {
                'budget': current_budget,
                'total_target': df[target_col].sum(),
                'efficiency': df[target_col].sum() / current_budget if current_budget > 0 else 0
            },
            'scenarios': []
        }
        
        for scenario in scenarios:
            new_budget = current_budget * scenario['budget_change']
            budget_change_pct = (scenario['budget_change'] - 1) * 100
            
            # Simple linear projection (can be enhanced with model predictions)
            projected_target = df[target_col].sum() * scenario['budget_change']
            projected_efficiency = projected_target / new_budget if new_budget > 0 else 0
            
            scenario_result = {
                'name': scenario['name'],
                'new_budget': new_budget,
                'budget_change_pct': budget_change_pct,
                'projected_target': projected_target,
                'target_change': projected_target - df[target_col].sum(),
                'target_change_pct': ((projected_target / df[target_col].sum()) - 1) * 100,
                'projected_efficiency': projected_efficiency,
                'efficiency_change_pct': ((projected_efficiency / results['current']['efficiency']) - 1) * 100
            }
            
            results['scenarios'].append(scenario_result)
        
        logger.info("Budget optimization simulation complete")
        return results
    
    def recommend_budget_allocation(self, df: pd.DataFrame,
                                   total_budget: float,
                                   group_by: str,
                                   target_col: str,
                                   cost_col: str = 'cost') -> pd.DataFrame:
        """
        Recommend optimal budget allocation across groups
        
        Args:
            df: DataFrame with campaign data
            total_budget: Total budget to allocate
            group_by: Column to group by (platform, campaign, etc.)
            target_col: Target metric
            cost_col: Cost column
            
        Returns:
            DataFrame with recommended allocation
        """
        logger.info(f"Calculating optimal budget allocation by {group_by}...")
        
        # Calculate efficiency by group
        group_stats = df.groupby(group_by).agg({
            cost_col: 'sum',
            target_col: 'sum'
        }).reset_index()
        
        group_stats['current_efficiency'] = group_stats[target_col] / group_stats[cost_col]
        group_stats['current_budget_share'] = (group_stats[cost_col] / group_stats[cost_col].sum()) * 100
        
        # Allocate budget proportional to efficiency
        total_efficiency = group_stats['current_efficiency'].sum()
        group_stats['recommended_budget'] = (
            (group_stats['current_efficiency'] / total_efficiency) * total_budget
        )
        group_stats['recommended_budget_share'] = (
            group_stats['recommended_budget'] / total_budget
        ) * 100
        
        group_stats['budget_change'] = (
            group_stats['recommended_budget'] - group_stats[cost_col]
        )
        group_stats['budget_change_pct'] = (
            (group_stats['recommended_budget'] / group_stats[cost_col]) - 1
        ) * 100
        
        # Expected impact
        group_stats['expected_target'] = (
            group_stats['recommended_budget'] * group_stats['current_efficiency']
        )
        
        group_stats = group_stats.sort_values('recommended_budget', ascending=False)
        
        logger.info("Budget allocation recommendations generated")
        return group_stats
    
    def generate_executive_summary(self, model_results: pd.DataFrame,
                                  insights: Dict[str, Any],
                                  optimization: Dict[str, Any]) -> str:
        """
        Generate executive summary report
        
        Args:
            model_results: Model evaluation results
            insights: Performance insights
            optimization: Budget optimization results
            
        Returns:
            Formatted executive summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append("POST-CAMPAIGN ANALYSIS - EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Model Performance
        summary.append("MODEL PERFORMANCE")
        summary.append("-" * 80)
        best_model = model_results.iloc[0]
        summary.append(f"Best Model: {best_model['Model']}")
        summary.append(f"R² Score: {best_model['R2']:.4f}")
        summary.append(f"RMSE: {best_model['RMSE']:.2f}")
        summary.append(f"MAPE: {best_model['MAPE']:.2f}%")
        summary.append("")
        
        # Overall Performance
        if 'overall' in insights:
            summary.append("CAMPAIGN PERFORMANCE")
            summary.append("-" * 80)
            overall = insights['overall']
            summary.append(f"Mean Actual: {overall['mean_actual']:.2f}")
            summary.append(f"Mean Predicted: {overall['mean_prediction']:.2f}")
            if overall['total_cost']:
                summary.append(f"Total Spend: ${overall['total_cost']:,.2f}")
            summary.append(f"Mean Absolute Error: {overall['mean_abs_error']:.2f}")
            summary.append(f"Mean % Error: {overall['mean_pct_error']:.2f}%")
            summary.append("")
        
        # Platform Performance
        if 'by_platform' in insights:
            summary.append("PERFORMANCE BY PLATFORM")
            summary.append("-" * 80)
            for platform, stats in insights['by_platform'].items():
                summary.append(f"{platform}:")
                summary.append(f"  Total Results: {stats.get('actual', 0):.2f}")
                summary.append(f"  Avg % Error: {stats.get('pct_error', 0):.2f}%")
            summary.append("")
        
        # Budget Optimization
        if 'scenarios' in optimization:
            summary.append("BUDGET OPTIMIZATION SCENARIOS")
            summary.append("-" * 80)
            for scenario in optimization['scenarios']:
                summary.append(f"{scenario['name']}:")
                summary.append(f"  New Budget: ${scenario['new_budget']:,.2f}")
                summary.append(f"  Expected Change: {scenario['target_change_pct']:.1f}%")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
