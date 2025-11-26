"""
Streamlit Dashboard for PCA-Agent
Interactive web interface for campaign analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from main import run_pca_agent
import os

# Page configuration
st.set_page_config(
    page_title="PCA-Agent Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🎯 PCA-Agent: Post-Campaign Analysis Dashboard")
st.markdown("**Enterprise-grade AI for digital media campaign optimization**")

# Sidebar
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Campaign Data (CSV)",
    type=['csv']
)

# Use sample data option
use_sample = st.sidebar.checkbox("Use Sample Data", value=True)

# Target column selection
target_column = st.sidebar.selectbox(
    "Target Variable",
    ["conversions", "revenue", "leads", "sales", "roas"]
)

# Model selection
tune_hyperparameters = st.sidebar.checkbox("Tune Hyperparameters", value=False)

# Run analysis button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# Main content
if run_analysis:
    with st.spinner("Running PCA-Agent analysis..."):
        try:
            # Save uploaded file if provided
            data_path = None
            if uploaded_file and not use_sample:
                data_path = f"temp_{uploaded_file.name}"
                with open(data_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Run analysis
            results = run_pca_agent(
                data_path=data_path,
                target_column=target_column,
                use_sample_data=use_sample,
                tune_hyperparameters=tune_hyperparameters
            )
            
            # Clean up temp file
            if data_path and os.path.exists(data_path):
                os.unlink(data_path)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            best_result = results['results'].iloc[0]
            
            with col1:
                st.metric("Best Model", results['model_name'])
            with col2:
                st.metric("R² Score", f"{best_result['R2']:.4f}")
            with col3:
                st.metric("RMSE", f"{best_result['RMSE']:.2f}")
            with col4:
                st.metric("MAPE", f"{best_result['MAPE']:.2f}%")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Model Comparison",
                "🎯 Feature Importance",
                "💰 Budget Optimization",
                "📈 Platform Performance"
            ])
            
            with tab1:
                st.subheader("Model Performance Comparison")
                
                # Model comparison chart
                fig = px.bar(
                    results['results'],
                    x='Model',
                    y='R2',
                    title='R² Score by Model',
                    color='R2',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.dataframe(results['results'], use_container_width=True)
            
            with tab2:
                st.subheader("Top Feature Importance")
                
                # Feature importance chart
                top_features = results['feature_importance'].head(10)
                
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Full table
                st.dataframe(results['feature_importance'], use_container_width=True)
            
            with tab3:
                st.subheader("Budget Optimization Scenarios")
                
                # Budget scenarios
                scenarios = results['optimization']['scenarios']
                
                scenario_df = pd.DataFrame(scenarios)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=scenario_df['name'],
                    y=scenario_df['target_change_pct'],
                    marker_color=['green' if x > 0 else 'red' for x in scenario_df['target_change_pct']]
                ))
                fig.update_layout(
                    title='Expected Impact of Budget Changes',
                    xaxis_title='Scenario',
                    yaxis_title='Expected Change (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scenario details
                for scenario in scenarios:
                    with st.expander(f"📊 {scenario['name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("New Budget", f"${scenario['new_budget']:,.2f}")
                            st.metric("Budget Change", f"{scenario['budget_change_pct']:.1f}%")
                        with col2:
                            st.metric("Projected Target", f"{scenario['projected_target']:.2f}")
                            st.metric("Expected Change", f"{scenario['target_change_pct']:.1f}%")
            
            with tab4:
                st.subheader("Platform Performance Analysis")
                
                if 'by_platform' in results['insights']:
                    platform_data = pd.DataFrame(results['insights']['by_platform']).T
                    platform_data = platform_data.reset_index()
                    platform_data.columns = ['Platform'] + list(platform_data.columns[1:])
                    
                    # Platform performance chart
                    fig = px.bar(
                        platform_data,
                        x='Platform',
                        y='actual',
                        title='Total Results by Platform',
                        color='pct_error',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Platform table
                    st.dataframe(platform_data, use_container_width=True)
                else:
                    st.info("Platform-level data not available in this dataset")
            
            # Download section
            st.subheader("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists("output/model_results.csv"):
                    with open("output/model_results.csv", "rb") as f:
                        st.download_button(
                            "📊 Model Results (CSV)",
                            f,
                            file_name="model_results.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if os.path.exists("output/feature_importance.csv"):
                    with open("output/feature_importance.csv", "rb") as f:
                        st.download_button(
                            "🎯 Feature Importance (CSV)",
                            f,
                            file_name="feature_importance.csv",
                            mime="text/csv"
                        )
            
            with col3:
                if os.path.exists("output/executive_summary.txt"):
                    with open("output/executive_summary.txt", "rb") as f:
                        st.download_button(
                            "📄 Executive Summary (TXT)",
                            f,
                            file_name="executive_summary.txt",
                            mime="text/plain"
                        )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:
    # Welcome screen
    st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to get started")
    
    st.markdown("""
    ### Features
    - 🤖 **15 Regression Models** - Compare multiple algorithms automatically
    - 📊 **Feature Engineering** - Automatic creation of media-specific features
    - 🎯 **SHAP Analysis** - Understand what drives your results
    - 💰 **Budget Optimization** - Get actionable recommendations
    - 📈 **Platform Analysis** - Compare performance across channels
    
    ### Supported Platforms
    - Google Ads
    - Meta Ads (Facebook/Instagram)
    - DV360
    - CM360
    - Snapchat Ads
    - Any digital media platform
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**PCA-Agent v1.0**")
st.sidebar.markdown("Enterprise AI for Campaign Analysis")
