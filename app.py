# app.py - Stock Signal Generator Multi-Page App

import streamlit as st

# Page config
st.set_page_config(
    page_title="Stock Signal Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom styling ----------
st.markdown(
    """
<style>
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding-bottom: 3rem;
}

.hero-title {
  font-size: 3rem;
  font-weight: 800;
  color: #38bdf8;
  text-align: center;
  margin-bottom: 0.4rem;
}

.hero-subtitle {
  text-align: center;
  color: #e5e7eb;
  font-size: 0.98rem;
  margin-bottom: 1.2rem;
}

.hero-tagline {
  text-align: center;
  color: #d1d5db;
  font-size: 0.95rem;
  margin-bottom: 2rem;
}

.tag-row {
  display: flex;
  justify-content: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 2.2rem;
}

.tag-pill {
  background: #020617;
  border-radius: 999px;
  padding: 0.25rem 0.9rem;
  border: 1px solid #1f2937;
  font-size: 0.8rem;
  color: #e5e7eb;
}

.card-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1.3rem;
  margin-bottom: 1.8rem;
}

.card {
  flex: 1 1 0;
  min-width: 260px;
  background: #020617;
  border-radius: 0.9rem;
  padding: 1.4rem 1.6rem;
  border: 1px solid #1f2937;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
}

.card h3 {
  margin-top: 0;
  margin-bottom: 0.6rem;
  font-size: 1.15rem;
}

.card p,
.card li {
  font-size: 0.95rem;
}

.card ul {
  padding-left: 1.1rem;
  margin-top: 0.35rem;
}

.section-footer {
  margin-top: 1.8rem;
  text-align: center;
  font-size: 0.85rem;
  color: #9ca3af;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Homepage content ----------
st.markdown('<div class="page">', unsafe_allow_html=True)

st.markdown(
    '<div class="hero-title">Stock Portfolio ML Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-subtitle">
Front-end dashboard for our Machine Learning term project.
</div>
<div class="hero-tagline">
We show how raw stock market data becomes structured, model-ready features and clear,
interpretable signals that can plug directly into ML experiments.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="tag-row">
  <div class="tag-pill">Feature Engineering Pipeline</div>
  <div class="tag-pill">Technical Indicators</div>
  <div class="tag-pill">Model-Ready Datasets</div>
  <div class="tag-pill">Explainable Signals</div>
</div>
""",
    unsafe_allow_html=True,
)

# Row 1: Purpose + Why It Matters
st.markdown(
    """
<div class="card-row">

  <div class="card">
    <h3>Project Purpose</h3>
    <p>
    This project is about building a <strong>transparent, reusable workflow</strong> for equity data,
    not a black-box trading bot.
    </p>
    <ul>
      <li>Make the <strong>data pipeline</strong> visible and reproducible.</li>
      <li>Show exactly how <strong>indicators and labels</strong> are engineered.</li>
      <li>Produce clean <strong>feature datasets</strong> that can be shared across models.</li>
    </ul>
  </div>

  <div class="card">
    <h3>Why It Matters</h3>
    <p>
    Real markets are noisy and high-dimensional. This app demonstrates how to turn that noise into
    structured information that:
    </p>
    <ul>
      <li>Supports <strong>systematic analysis</strong> instead of ad-hoc chart reading.</li>
      <li>Gives a consistent basis for <strong>classification or forecasting models</strong>.</li>
      <li>Can be reused for new research questions without rebuilding everything.</li>
    </ul>
  </div>

</div>
""",
    unsafe_allow_html=True,
)

# Row 2: What it shows + How to use
st.markdown(
    """
<div class="card-row">

  <div class="card">
    <h3>What This App Demonstrates</h3>
    <ul>
      <li><strong>End-to-end feature engineering</strong> from historical price data.</li>
      <li>
        A library of <strong>technical indicators</strong>
        (SMA, EMA, RSI, MACD, Bollinger Bands, volatility, and more).
      </li>
      <li>
        <strong>Return-based targets</strong> that define what a “good outcome” looks like for a stock.
      </li>
      <li>
        Exportable <strong>CSV files</strong> that match the features shown here,
        ready for models such as Random Forest or XGBoost.
      </li>
    </ul>
    <p>
    In short, the app connects financial intuition (indicators and signals) with
    <strong>machine-learning practice</strong> (structured, labeled datasets).
    </p>
  </div>

  <div class="card">
    <h3>How to Use This Dashboard</h3>
    <ol>
      <li>Select a ticker and date range in the sidebar.</li>
      <li>On the <strong>EDA</strong> page, review price history, indicator behavior, and correlations.</li>
      <li>Inspect the <strong>feature tables</strong> and labels that would feed ML models.</li>
      <li>In the full pipeline, regenerate cached CSVs from the <strong>PostgreSQL ingestion</strong> step.</li>
    </ol>
    <p>
    This gives both technical and non-technical readers a concrete view of what our pipeline produces.
    </p>
  </div>

</div>
""",
    unsafe_allow_html=True,
)

# Row 3: Contribution + Scope
st.markdown(
    """
<div class="card-row">

  <div class="card">
    <h3>Technical Contribution</h3>
    <ul>
      <li>Unified feature set built from multiple technical indicators.</li>
      <li>Consistent labeling scheme based on forward returns.</li>
      <li>
        Clear separation between <strong>data ingestion</strong>,
        <strong>feature engineering</strong>, and <strong>visualization</strong>.
      </li>
      <li>
        Design that supports comparison of several ML models on the same feature space.
      </li>
    </ul>
  </div>

  <div class="card">
    <h3>Scope and Limitations</h3>
    <ul>
      <li>Educational term project, not a live trading system.</li>
      <li>Relies on historical data and simplifying assumptions.</li>
      <li>Does not model transaction costs, slippage, or execution rules.</li>
      <li>
        Outputs are intended for <strong>analysis and experimentation</strong>,
        not real-money decisions.
      </li>
    </ul>
  </div>

</div>

<div class="section-footer">
Portfolio ML Analyzer — Machine Learning Term Project · Built with Streamlit | Data source: Yahoo Finance (yfinance)
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

