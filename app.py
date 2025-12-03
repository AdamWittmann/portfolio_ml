import streamlit as st
import pandas as pd
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Stock Signal Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hardcoded project root (your machine)
PROJECT_ROOT = Path("")
METRICS_PATH = PROJECT_ROOT / "model_metrics.csv"
TRADING_METRICS_PATH = PROJECT_ROOT / "trading_metrics.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "model_pipeline.py"
GENERATE_METRICS_PATH = PROJECT_ROOT / "models" / "generate_metrics.py"


# ---------- Load Metrics ----------
@st.cache_data
def load_trading_metrics():
    """Load trading metrics safely with flexible CSV format support."""
    try:
        if TRADING_METRICS_PATH.exists():
            df = pd.read_csv(TRADING_METRICS_PATH)

            # Case 1: key/value format (metric,value)
            if "metric" in df.columns and "value" in df.columns:
                return df.set_index("metric")["value"]

            # Case 2: single-row dataframe with columns
            if df.shape[0] == 1:
                return df.iloc[0]

            # Case 3: unknown format → attempt first row
            return df.iloc[0]

    except Exception as e:
        st.sidebar.error(f"Error loading trading metrics: {e}")

    # Fallback
    return pd.Series({
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "avg_return": 0.0,
        "n_trades": 0
    })



@st.cache_data
def load_model_metrics():
    """Load final-fold model performance metrics in flexible formats."""
    try:
        if METRICS_PATH.exists():
            df = pd.read_csv(METRICS_PATH, index_col=0)

            # Case 1: Already in final-fold format (metric/value)
            if "value" in df.columns:
                return df

            # Case 2: Train/Test format → use Test column
            if "Test" in df.columns:
                return df[["Test"]].rename(columns={"Test": "value"})

            # Case 3: Multiple columns → assume last is final fold
            if df.shape[1] >= 1:
                last_col = df.columns[-1]
                return df[[last_col]].rename(columns={last_col: "value"})

    except Exception as e:
        st.sidebar.error(f"Error loading metrics: {e}")

    # Default fallback
    return pd.DataFrame({"value": [0, 0, 0, 0, 0]},
                        index=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])



def get_last_training_time():
    """Get last time metrics file was updated (proxy for last training)."""
    try:
        if METRICS_PATH.exists():
            mod_time = METRICS_PATH.stat().st_mtime
            return datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "Never"


def run_model_pipeline():
    """Run the model training pipeline (walk-forward pipeline)."""
    try:
        if not PIPELINE_PATH.exists():
            return False, f"❌ Could not find pipeline at {PIPELINE_PATH}"

        result = subprocess.run(
            [sys.executable, str(PIPELINE_PATH)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Training timed out after 10 minutes"
    except Exception as e:
        return False, f"Exception: {str(e)}"


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Model Performance (Final Walk-Forward Fold)")

    # Initialize session state flags
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False
    if "metrics_in_progress" not in st.session_state:
        st.session_state.metrics_in_progress = False

    # Buttons: Train + Metrics
    col1, col2 = st.columns([1, 1])

    # Train button
    with col1:
        train_button = st.button(
            "Train",
            type="primary",
            use_container_width=True,
            help="Run full walk-forward training pipeline",
            disabled=st.session_state.training_in_progress,
        )

        if train_button and not st.session_state.training_in_progress:
            st.session_state.training_in_progress = True

            load_model_metrics.clear()
            load_trading_metrics.clear()

            with st.spinner("Training model... This may take several minutes"):
                success, output = run_model_pipeline()

            st.session_state.training_in_progress = False

            if success:
                st.success("Training complete!")
                st.info("Click Metrics to recompute metrics, then Refresh.")
            else:
                st.error("Training failed")
                with st.expander("Show error details"):
                    st.code(output)

    # Metrics button
    with col2:
        metrics_button = st.button(
            "Metrics",
            use_container_width=True,
            help="Recompute metrics for the saved model on the final fold window",
            disabled=st.session_state.metrics_in_progress,
        )

        if metrics_button and not st.session_state.metrics_in_progress:
            st.session_state.metrics_in_progress = True

            load_model_metrics.clear()
            load_trading_metrics.clear()

            if not GENERATE_METRICS_PATH.exists():
                st.error("generate_metrics.py not found")
                st.session_state.metrics_in_progress = False
            else:
                with st.spinner("Generating metrics..."):
                    result = subprocess.run(
                        [sys.executable, str(GENERATE_METRICS_PATH)],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=str(PROJECT_ROOT),
                    )

                st.session_state.metrics_in_progress = False

                if result.returncode == 0:
                    st.success("Metrics updated! Click Refresh to see latest values.")
                else:
                    st.error("Failed to generate metrics")
                    with st.expander("Show error"):
                        st.code(result.stderr)

    # Refresh button
    if st.button("Refresh", use_container_width=True):
        load_model_metrics.clear()
        load_trading_metrics.clear()
        st.rerun()

    last_trained = get_last_training_time()
    st.caption(f"Last metrics update (final fold): {last_trained}")

    st.divider()

    # Load metrics
    model_metrics = load_model_metrics()
    trading_metrics = load_trading_metrics()

    has_real_metrics = model_metrics["value"].max() > 0

    if not has_real_metrics:
        st.warning("No metrics yet. Click Train, then Metrics to generate them.")

    # Classification Metrics (Final Fold)
    st.markdown("#### Classification Metrics (Final Fold)")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{model_metrics.loc['Accuracy', 'value']:.2%}",
        )
        st.metric(
            "Precision",
            f"{model_metrics.loc['Precision', 'value']:.2%}",
        )

    with col2:
        st.metric(
            "Recall",
            f"{model_metrics.loc['Recall', 'value']:.2%}",
        )
        st.metric(
            "F1 Score",
            f"{model_metrics.loc['F1 Score', 'value']:.2%}",
        )

    st.metric("ROC AUC", f"{model_metrics.loc['ROC AUC', 'value']:.3f}")

    st.caption("Final walk-forward evaluation window: 2024-07-01 → 2024-12-31")

    st.divider()

    # Trading Performance
    st.markdown("#### Trading Performance (Final Fold)")
    st.metric("Win Rate", f"{trading_metrics['win_rate']:.1%}")
    st.metric("Sharpe Ratio", f"{trading_metrics['sharpe_ratio']:.2f}")
    st.metric("Avg Return per Trade", f"{trading_metrics['avg_return']:.2%}")
    st.metric("Total Trades", f"{int(trading_metrics['n_trades']):,}")

    st.divider()

    # Debug info
    with st.expander("Debug Info"):
        st.text(f"Project Root: {PROJECT_ROOT}")
        st.text(f"Pipeline exists: {PIPELINE_PATH.exists()}")
        st.text(f"Generate Script exists: {GENERATE_METRICS_PATH.exists()}")
        st.text(f"Metrics CSV exists: {METRICS_PATH.exists()}")
        st.text(f"Trading CSV exists: {TRADING_METRICS_PATH.exists()}")


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
        <strong>Return-based targets</strong> that define what a "good outcome" looks like for a stock.
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
      <li>Click <strong>🚀 Train</strong> in the sidebar to train using walk-forward validation.</li>
      <li>Click <strong>Metrics</strong> to recompute final-fold performance from the saved model.</li>
      <li>Use the <strong>EDA</strong> and other pages (if added) to explore data and features.</li>
      <li>Monitor the <strong>classification and trading metrics</strong> in the sidebar.</li>
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
        <strong>feature engineering</strong>, and <strong>model evaluation</strong>.
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
