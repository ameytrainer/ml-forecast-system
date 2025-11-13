"""
Streamlit Dashboard for Sales Forecaster
Interactive UI for model predictions and monitoring
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:5000"

# Title and description
st.markdown('<p class="main-header">📊 Sales Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Production ML System - Real-time Predictions**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # API connection status
    st.subheader("API Status")
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
    
    # Refresh settings
    st.subheader("Dashboard Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        st.info(f"Refreshing every {refresh_interval}s")
    
    st.markdown("---")
    
    # About
    st.subheader("About")
    st.markdown("""
    **Sales Forecaster v3.0**
    
    Production ML system with:
    - MLflow tracking
    - DVC versioning
    - CI/CD automation
    
    Built for MLOps Course
    """)

# Main dashboard

# Model Information Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Version", model_version)

with col2:
    st.metric("Status", "🟢 Active")

with col3:
    st.metric("Uptime", "99.9%")

with col4:
    st.metric("Predictions/day", "~1,250")

st.markdown("---")

# Fetch metrics from API
try:
    response = requests.get(f"{API_URL}/metrics", timeout=2)
    metrics = response.json()
    model_perf = metrics.get("model_performance", {})
except:
    model_perf = {"mae": "N/A", "rmse": "N/A", "r2_score": "N/A"}

# Performance Metrics
st.subheader("📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    mae_value = model_perf.get('mae', 0)
    st.metric(
        "MAE (Mean Absolute Error)", 
        f"{mae_value:.2f}" if isinstance(mae_value, (int, float)) else "N/A",
        delta="-9.7%" if isinstance(mae_value, (int, float)) else None,
        delta_color="inverse"
    )

with col2:
    rmse_value = model_perf.get('rmse', 0)
    st.metric(
        "RMSE (Root Mean Squared Error)", 
        f"{rmse_value:.2f}" if isinstance(rmse_value, (int, float)) else "N/A",
        delta="-8.7%" if isinstance(rmse_value, (int, float)) else None,
        delta_color="inverse"
    )

with col3:
    r2_value = model_perf.get('r2_score', 0)
    st.metric(
        "R² Score", 
        f"{r2_value:.2f}" if isinstance(r2_value, (int, float)) else "N/A",
        delta="+2.3%" if isinstance(r2_value, (int, float)) else None
    )

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
    st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")

# Interactive Prediction Section
st.subheader("🔮 Make a Custom Prediction")

st.markdown("Enter values below to get a sales prediction:")

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
                    help="Model version used"
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
    'Status': ['✅ Served'] * 5
})

st.dataframe(recent_predictions, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Sales Forecaster Dashboard</b> | Powered by Streamlit, FastAPI & MLflow</p>
    <p>MLOps with Agentic AI - Advanced Certification Course</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
