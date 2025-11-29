# app.py - Stock Signal Generator Multi-Page App

import streamlit as st

# Page config
st.set_page_config(
    page_title="Stock Signal Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 Stock Signal Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered Stock Predictions with XGBoost</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("ROC-AUC", "0.6442", delta="0.04")
st.sidebar.metric("Win Rate", "62.4%", delta="12.4%")
st.sidebar.metric("Sharpe Ratio", "3.64", delta="2.64")
st.sidebar.markdown("---")
st.sidebar.info("""
**Model Details:**
- Algorithm: XGBoost Classifier
- Features: 17 technical indicators
- Training: 2020-2023 (4.5 years)
- Test: Jul 2023 - Nov 2024
- Trades: 2,829 signals
""")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 What This App Does")
    st.write("""
    This application uses machine learning to predict profitable stock trades based on technical indicators:
    
    - **Exploratory Data Analysis**: Visualize stock correlations, Sharpe ratios, RSI, MACD, and more
    - **Live Predictions**: Get buy/sell signals with confidence scores for any stock
    - **Backtested Performance**: 444% return over 17 months (equal-weight strategy)
    """)
    
    st.markdown("### 📚 How It Works")
    st.write("""
    1. **Features**: 17 technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
    2. **Model**: XGBoost trained on 4.5 years of historical data
    3. **Prediction**: Binary classification - will stock return >3% in next 7 days?
    4. **Output**: Probability score (0-100%) and buy/sell recommendation
    """)

with col2:
    st.markdown("### 🚀 Quick Start")
    st.info("""
    **👈 Use the sidebar** to navigate between pages:
    
    1. **📊 EDA**: Explore stock correlations, technical indicators, and historical patterns
    2. **🔮 Predictions**: Get live predictions for any stock ticker
    """)
    
    st.markdown("### ⚠️ Important Disclaimers")
    st.warning("""
    - **Not Financial Advice**: This is an educational project
    - **Backtest Assumptions**: Perfect execution at close prices, no slippage/fees
    - **Real Trading**: Would require entry/exit rules, risk management, transaction costs
    - **Signal Only**: Model generates signals, not a complete trading system
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Built with XGBoost, Streamlit, and PostgreSQL | Data: Yahoo Finance</p>
    <p>For educational purposes only - Not financial advice</p>
</div>
""", unsafe_allow_html=True)