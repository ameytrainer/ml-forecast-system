"""
Streamlit Dashboard for Sales Forecaster
Interactive UI with Real-Time Metrics & Auto-Refresh
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Sales Forecaster Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:5000"

# Title and description
st.markdown('<p class="main-header">📊 Sales Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Production ML System - Real-time Predictions from MLflow Model Registry**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # API connection status
    st.subheader("🔌 API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Connected")
            health_data = response.json()
            model_version = health_data.get("model_version", "unknown")
        else:
            st.error("❌ API Error")
            model_version = "unknown"
    except:
        st.error("❌ Cannot connect to API")
        st.info("Start backend: `uvicorn app.backend:app --reload --port 5000`")
        model_version = "unknown"
    
    st.markdown("---")
    
    # Auto-reload status
    st.subheader("🔄 Auto-Reload Status")
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            info = response.json()
            auto_reload = info.get("auto_reload_enabled", False)
            reload_interval = info.get("auto_reload_interval", "N/A")
            
            if auto_reload:
                st.success(f"✅ Backend Auto-Reload: ON")
                st.info(f"⏱️ Check Interval: {reload_interval}")
            else:
                st.warning("⚠️ Backend Auto-Reload: OFF")
    except:
        st.info("Backend status unavailable")
    
    st.markdown("---")
    
    # Dashboard refresh settings
    st.subheader("🔄 Dashboard Settings")
    auto_refresh = st.checkbox("Auto-refresh Dashboard", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)
        st.info(f"🔄 Refreshing every {refresh_interval}s")
        st.caption("Dashboard will automatically show new models when deployed")
    else:
        st.warning("Manual refresh only")
        refresh_interval = None
    
    st.markdown("---")
    
    # Manual controls
    st.subheader("🎛️ Manual Controls")
    
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    if st.button("♻️ Reload Backend Model", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/model/reload", timeout=5)
            if response.status_code == 200:
                st.success("✅ Model reloaded!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Reload failed")
        except:
            st.error("❌ Cannot connect to backend")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Sales Forecaster v4.0**
    
    ✅ MLflow Model Registry  
    ✅ Auto-reload (30s)  
    ✅ DVC versioning  
    ✅ CI/CD automation  
    ✅ Real-time metrics  
    
    **Fully Automated MLOps!**
    """)

# Main dashboard

# Model Information Section
st.subheader("🎯 Model Information")

col1, col2, col3, col4 = st.columns(4)

# Fetch model info
try:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    if response.status_code == 200:
        model_info = response.json()
        model_version_display = model_info.get("model_version", "Unknown")
        model_source = model_info.get("metadata", {}).get("source", "Unknown")
        auto_reload_info = model_info.get("auto_reload", {})
    else:
        model_version_display = "Error"
        model_source = "Unknown"
        auto_reload_info = {}
except:
    model_version_display = "N/A"
    model_source = "Unknown"
    auto_reload_info = {}

with col1:
    st.metric("Model Version", model_version_display)

with col2:
    if "Registry" in model_source:
        st.metric("Status", "🟢 Active")
    elif "Local" in model_source:
        st.metric("Status", "🟡 Local")
    else:
        st.metric("Status", "⚪ Unknown")

with col3:
    if auto_reload_info.get("enabled"):
        st.metric("Auto-Reload", "✅ ON")
    else:
        st.metric("Auto-Reload", "⚠️ OFF")

with col4:
    st.metric("Predictions/day", "~1,250")

# Show auto-reload status banner
if auto_reload_info.get("enabled"):
    st.markdown("""
    <div class="info-box">
        <strong>🔄 Fully Automated Pipeline Active</strong><br>
        Backend checks for new models every 30s • Dashboard refreshes every 30s<br>
        <em>New models will appear automatically!</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ====================================================================
# Performance Metrics Section - WITH REAL DELTAS
# ====================================================================
st.subheader("📊 Model Performance Metrics")

# Fetch current model metrics
try:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    if response.status_code == 200:
        model_info = response.json()
        performance = model_info.get("performance", {})
        mae_value = performance.get("mae", 0)
        rmse_value = performance.get("rmse", 0)
        r2_value = performance.get("r2_score", 0)
    else:
        mae_value = rmse_value = r2_value = 0
except Exception as e:
    st.error(f"Could not fetch metrics: {e}")
    mae_value = rmse_value = r2_value = 0

# Fetch comparison with previous version for REAL deltas
try:
    comparison_response = requests.get(f"{API_URL}/model/compare", timeout=5)
    if comparison_response.status_code == 200:
        comparison = comparison_response.json()
        
        if comparison.get("has_comparison"):
            # We have a previous version to compare
            deltas = comparison.get("deltas", {})
            mae_delta = deltas.get("mae_percent", 0)
            rmse_delta = deltas.get("rmse_percent", 0)
            r2_delta = deltas.get("r2_percent", 0)
            
            improvements = comparison.get("improvement", {})
            mae_improved = improvements.get("mae") == "improved"
            rmse_improved = improvements.get("rmse") == "improved"
            r2_improved = improvements.get("r2") == "improved"
        else:
            # First version, no comparison
            mae_delta = rmse_delta = r2_delta = None
            mae_improved = rmse_improved = r2_improved = None
    else:
        mae_delta = rmse_delta = r2_delta = None
        mae_improved = rmse_improved = r2_improved = None
except:
    mae_delta = rmse_delta = r2_delta = None
    mae_improved = rmse_improved = r2_improved = None

# Display metrics in columns with REAL deltas
col1, col2, col3 = st.columns(3)

with col1:
    if mae_value > 0:
        if mae_delta is not None:
            # Show real delta
            st.metric(
                "MAE (Mean Absolute Error)", 
                f"{mae_value:.2f}",
                delta=f"{mae_delta:+.1f}%",
                delta_color="inverse",  # For MAE, lower is better
                help="Lower is better. Average prediction error in dollars."
            )
        else:
            # First version, no delta
            st.metric(
                "MAE (Mean Absolute Error)", 
                f"{mae_value:.2f}",
                help="Lower is better. Average prediction error in dollars."
            )
    else:
        st.metric(
            "MAE (Mean Absolute Error)", 
            "N/A",
            help="Train a model to see metrics"
        )

with col2:
    if rmse_value > 0:
        if rmse_delta is not None:
            st.metric(
                "RMSE (Root Mean Squared Error)", 
                f"{rmse_value:.2f}",
                delta=f"{rmse_delta:+.1f}%",
                delta_color="inverse",  # For RMSE, lower is better
                help="Lower is better. Penalizes large errors."
            )
        else:
            st.metric(
                "RMSE", 
                f"{rmse_value:.2f}",
                help="Lower is better. Penalizes large errors."
            )
    else:
        st.metric(
            "RMSE", 
            "N/A",
            help="Train a model to see metrics"
        )

with col3:
    if r2_value > 0:
        if r2_delta is not None:
            st.metric(
                "R² Score", 
                f"{r2_value:.3f}",
                delta=f"{r2_delta:+.1f}%",
                delta_color="normal",  # For R², higher is better
                help="Higher is better. Proportion of variance explained (0-1)."
            )
        else:
            st.metric(
                "R² Score", 
                f"{r2_value:.3f}",
                help="Higher is better. Proportion of variance explained (0-1)."
            )
    else:
        st.metric(
            "R² Score", 
            "N/A",
            help="Train a model to see metrics"
        )

# Show comparison details in expander
if mae_delta is not None:
    with st.expander("📊 View Detailed Version Comparison"):
        try:
            comparison_response = requests.get(f"{API_URL}/model/compare", timeout=5)
            if comparison_response.status_code == 200:
                comparison = comparison_response.json()
                
                current = comparison.get("current_version", {})
                previous = comparison.get("previous_version", {})
                
                st.markdown("### Current vs Previous Version")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['MAE', 'RMSE', 'R² Score'],
                    f'Current (v{current.get("version", "?")})': [
                        f"{current.get('metrics', {}).get('mae', 0):.2f}",
                        f"{current.get('metrics', {}).get('rmse', 0):.2f}",
                        f"{current.get('metrics', {}).get('r2_score', 0):.3f}"
                    ],
                    f'Previous (v{previous.get("version", "?")})': [
                        f"{previous.get('metrics', {}).get('mae', 0):.2f}",
                        f"{previous.get('metrics', {}).get('rmse', 0):.2f}",
                        f"{previous.get('metrics', {}).get('r2_score', 0):.3f}"
                    ],
                    'Change': [
                        f"{mae_delta:+.1f}%",
                        f"{rmse_delta:+.1f}%",
                        f"{r2_delta:+.1f}%"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Show improvement status
                improvements = comparison.get("improvement", {})
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if improvements.get("mae") == "improved":
                        st.success("✅ MAE Improved")
                    elif improvements.get("mae") == "degraded":
                        st.error("❌ MAE Degraded")
                    else:
                        st.info("➖ MAE Unchanged")
                
                with col_b:
                    if improvements.get("rmse") == "improved":
                        st.success("✅ RMSE Improved")
                    elif improvements.get("rmse") == "degraded":
                        st.error("❌ RMSE Degraded")
                    else:
                        st.info("➖ RMSE Unchanged")
                
                with col_c:
                    if improvements.get("r2") == "improved":
                        st.success("✅ R² Improved")
                    elif improvements.get("r2") == "degraded":
                        st.error("❌ R² Degraded")
                    else:
                        st.info("➖ R² Unchanged")
        except:
            st.info("Comparison data not available")

st.markdown("---")

# 7-Day Forecast Visualization
st.subheader("📈 7-Day Sales Forecast")

# Generate forecast data (synthetic for demo)
np.random.seed(42)
dates = [(datetime.now() + timedelta(days=i)) for i in range(7)]
date_labels = [d.strftime("%a\n%m/%d") for d in dates]

base_sales = 120
trend = np.linspace(0, 20, 7)
seasonality = np.sin(np.arange(7) * 2 * np.pi / 7) * 15
noise = np.random.gamma(2, 8, 7)
forecasts = base_sales + trend + seasonality + noise

# Confidence intervals
lower_bound = forecasts * 0.9
upper_bound = forecasts * 1.1

# Create plotly chart
fig = go.Figure()

# Add forecast line
fig.add_trace(go.Scatter(
    x=date_labels,
    y=forecasts,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=10)
))

# Add confidence interval
fig.add_trace(go.Scatter(
    x=date_labels,
    y=upper_bound,
    mode='lines',
    name='Upper Bound',
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=date_labels,
    y=lower_bound,
    mode='lines',
    name='Confidence Interval',
    fill='tonexty',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(width=0)
))

fig.update_layout(
    title="Predicted Sales for Next 7 Days",
    xaxis_title="Date",
    yaxis_title="Predicted Sales ($)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Show forecast table
with st.expander("📋 View Forecast Data"):
    forecast_df = pd.DataFrame({
        'Date': [d.strftime("%Y-%m-%d") for d in dates],
        'Day': [d.strftime("%A") for d in dates],
        'Forecast': [f"${f:.2f}" for f in forecasts],
        'Lower Bound': [f"${l:.2f}" for l in lower_bound],
        'Upper Bound': [f"${u:.2f}" for u in upper_bound]
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Interactive Prediction Section
st.subheader("🔮 Make a Custom Prediction")

st.markdown("Enter values below to get a sales prediction from the production model:")

col1, col2 = st.columns(2)

with col1:
    advertising_spend = st.number_input(
        "💰 Advertising Spend ($)",
        min_value=0,
        max_value=10000,
        value=3000,
        step=100,
        help="Daily advertising budget in dollars"
    )
    
    promotions = st.selectbox(
        "🎁 Promotions Active",
        options=[0, 1],
        format_func=lambda x: "Yes" if x else "No",
        help="Whether promotional campaigns are running"
    )
    
    day_of_week = st.selectbox(
        "📅 Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        help="Day of the week"
    )

with col2:
    month = st.selectbox(
        "📆 Month",
        options=list(range(1, 13)),
        format_func=lambda x: ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"][x-1],
        help="Month of the year"
    )
    
    is_weekend = st.selectbox(
        "🏖️ Is Weekend",
        options=[0, 1],
        format_func=lambda x: "Yes" if x else "No",
        help="Whether it's a weekend day"
    )

# Predict button
if st.button("🚀 Get Prediction", type="primary", use_container_width=True):
    try:
        # Make API request
        payload = {
            "advertising_spend": advertising_spend,
            "promotions": promotions,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend
        }
        
        with st.spinner("Making prediction..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result prominently
            st.success("✅ Prediction Complete!")
            
            # Create three columns for results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    "Predicted Sales",
                    f"${result['prediction']:.2f}",
                    help="Forecasted sales value"
                )
            
            with res_col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.1%}",
                    help="Model confidence in prediction"
                )
            
            with res_col3:
                st.metric(
                    "Model Version",
                    result['model_version'],
                    help="Model version used for prediction"
                )
            
            # Additional info
            st.info(f"🕐 Prediction made at: {result['timestamp']}")
            
        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.write(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
        st.code("uvicorn app.backend:app --reload --port 5000")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")

# Recent Predictions (Mock data for demo)
st.subheader("📜 Recent Predictions")

recent_predictions = pd.DataFrame({
    'Timestamp': [
        (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M:%S")
        for i in range(5, 0, -1)
    ],
    'Prediction': [f"${p:.2f}" for p in np.random.uniform(100, 180, 5)],
    'Confidence': [f"{c:.1%}" for c in np.random.uniform(0.75, 0.95, 5)],
    'Model': [model_version_display] * 5,
    'Status': ['✅ Served'] * 5
})

st.dataframe(recent_predictions, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Sales Forecaster Dashboard v4.0</b> | Powered by Streamlit, FastAPI & MLflow</p>
    <p>Fully Automated MLOps with CI/CD</p>
    <p style='font-size: 0.9em;'>Last refresh: {datetime.now().strftime("%H:%M:%S")}</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh and refresh_interval:
    time.sleep(refresh_interval)
    st.rerun()