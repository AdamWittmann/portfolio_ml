# models/generate_metrics.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import logging

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")

engine = create_engine(DATABASE_URL)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "model_metrics.csv"
TRADING_METRICS_PATH = PROJECT_ROOT / "trading_metrics.csv"


# -------------------------------------------------------------------
# Helper functions (mirroring your pipeline logic)
# -------------------------------------------------------------------
def create_binary_target(df: pd.DataFrame, threshold: float = 0.03, days_ahead: int = 7):
    """Create binary target: 1 if forward return > threshold, else 0."""
    df = df.sort_values(["symbol", "date"]).copy()
    df["future_close"] = df.groupby("symbol")["close"].shift(-days_ahead)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    df["target"] = (df["future_return"] > threshold).astype(int)
    df = df.dropna(subset=["future_close", "future_return"])
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix X and metadata (no y here; we grab df['target'] directly)."""
    exclude_cols = ["id", "symbol", "date", "target", "future_return", "future_close"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df["target"].copy()
    metadata = df[["symbol", "date", "target", "future_return"]].copy()

    return X, y, metadata


def evaluate_trading_performance(test_metadata: pd.DataFrame, y_pred_proba, threshold: float = 0.5):
    """
    Evaluate the model as a trading strategy.

    Signal = 1 if predicted probability >= threshold.
    """
    test_metadata = test_metadata.copy()
    test_metadata["pred_proba"] = y_pred_proba
    test_metadata["signal"] = (y_pred_proba >= threshold).astype(int)

    trades = test_metadata[test_metadata["signal"] == 1].copy()

    if len(trades) == 0:
        logger.warning(f"⚠️  No trades generated at threshold {threshold}!")
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
        }

    n_trades = len(trades)
    n_winners = (trades["future_return"] > 0.03).sum()
    win_rate = n_winners / n_trades

    avg_return = trades["future_return"].mean()
    median_return = trades["future_return"].median()
    std_return = trades["future_return"].std()

    if std_return > 0:
        sharpe = avg_return / std_return
    else:
        sharpe = 0.0

    logger.info("\n💰 TRADING PERFORMANCE (threshold=0.5)")
    logger.info(f"   Total signals: {n_trades}")
    logger.info(f"   Winners: {n_winners} ({win_rate:.2%})")
    logger.info(f"   Average return: {avg_return:.2%}")
    logger.info(f"   Median return: {median_return:.2%}")
    logger.info(f"   Sharpe ratio: {sharpe:.2f}")

    return {
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "avg_return": float(avg_return),
        "sharpe_ratio": float(sharpe),
    }


# -------------------------------------------------------------------
# Main: recompute metrics for the final walk-forward window
# -------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("📊 GENERATING METRICS FOR SAVED MODEL (WALK-FORWARD FINAL FOLD)")
    logger.info("=" * 60)

    # 1. Load model
    model_path = PROJECT_ROOT / "models" / "stock_classifier.json"
    logger.info(f"\n📥 Loading model from {model_path} ...")
    if not model_path.exists():
        logger.error("❌ Model not found! Run model_pipeline.py first.")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    logger.info("✅ Model loaded")

    # 2. Load data
    logger.info("\n📥 Loading price data from database...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded {len(df):,} rows")

    # 3. Create target + features
    logger.info("\n🎯 Creating binary target...")
    df = create_binary_target(df, threshold=0.03, days_ahead=7)

    # 4. Select FINAL walk-forward test window
    # These dates should match your last fold in the training pipeline.
    test_start = pd.to_datetime("2024-07-01")
    test_end = pd.to_datetime("2024-12-31")

    mask = (df["date"] >= test_start) & (df["date"] <= test_end)
    test_df = df[mask].copy()

    if test_df.empty:
        logger.error(
            f"❌ No rows found between {test_start.date()} and {test_end.date()} "
            "for evaluation. Check your price data range."
        )
        return

    logger.info(
        f"\n🧪 Evaluation window: {test_df['date'].min().date()} → {test_df['date'].max().date()} "
        f"({len(test_df):,} rows)"
    )

    X_test, y_test, test_metadata = prepare_features(test_df)

    # 5. Predict
    logger.info("\n🔮 Generating predictions for evaluation window...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 6. Compute classification metrics
    logger.info("\n📊 CLASSIFICATION METRICS (FINAL FOLD)")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"ROC AUC  : {auc:.4f}")

    metrics_df = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "value": [acc, prec, rec, f1, auc],
        }
    ).set_index("metric")

    METRICS_PATH.write_text(metrics_df.to_csv())
    logger.info(f"\n💾 Saved classification metrics to {METRICS_PATH}")

    # 7. Trading metrics
    trading_metrics = evaluate_trading_performance(test_metadata, y_pred_proba, threshold=0.5)
    trading_df = pd.DataFrame([trading_metrics])
    TRADING_METRICS_PATH.write_text(trading_df.to_csv(index=False))
    logger.info(f"💾 Saved trading metrics to {TRADING_METRICS_PATH}")

    logger.info("\n✅ Metrics generation complete.")


if __name__ == "__main__":
    main()
