# predict.py - Generate predictions using walk-forward final model

import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

engine = create_engine(os.getenv("DATABASE_URL"))


def create_binary_target(df, threshold=0.03, days_ahead=7):
    """Create binary target: 1 if return > threshold, else 0"""
    df = df.sort_values(['symbol', 'date']).copy()
    df['future_close'] = df.groupby('symbol')['close'].shift(-days_ahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    df = df.dropna(subset=['future_return', 'future_close'])
    return df


def prepare_features(df):
    """Prepare feature matrix X and metadata"""
    exclude_cols = ['id', 'symbol', 'date', 'target', 'future_return', 'future_close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    metadata = df[['symbol', 'date', 'target', 'future_return']].copy()
    
    return X, metadata


def main():
    logger.info("="*60)
    logger.info("🔮 GENERATING PREDICTIONS (WALK-FORWARD MODEL)")
    logger.info("="*60)

    # 1. Load model
    logger.info("\n📥 Loading model...")
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/stock_classifier.json')
        logger.info("✅ Loaded final walk-forward model")
    except FileNotFoundError:
        logger.error("❌ Model not found! Run model_pipeline.py first.")
        return
    
    # 2. Load data
    logger.info("\n📥 Loading data...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df):,} rows")

    # 3. Create target & features
    logger.info("\n🎯 Creating target + features...")
    df = create_binary_target(df, threshold=0.03, days_ahead=7)

    # 4. Select FUTURE data only (after last training fold)
    prediction_start = pd.to_datetime("2025-01-01")
    pred_df = df[df['date'] >= prediction_start].copy()

    if pred_df.empty:
        logger.error("❌ No future rows available for predictions!")
        return

    logger.info(
        f"Prediction window: {pred_df['date'].min().date()} → "
        f"{pred_df['date'].max().date()} | {len(pred_df):,} rows"
    )

    # 5. Prepare features
    X_test, metadata = prepare_features(pred_df)

    # 6. Predict
    logger.info("\n🔮 Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 7. Save results
    metadata = metadata.copy()
    metadata['pred_proba'] = y_pred_proba
    metadata['prediction'] = y_pred

    metadata.columns = [
        'ticker', 'date', 'target', 'future_return',
        'pred_proba', 'prediction'
    ]

    metadata.to_csv('future_predictions.csv', index=False)

    logger.info(f"\n💾 Saved future_predictions.csv")
    logger.info(f"   Total predictions: {len(metadata):,}")
    logger.info(f"   Buy signals: {y_pred.sum():,} ({y_pred.mean():.1%})")

    logger.info("\n" + "="*60)
    logger.info("✅ DONE! Predictions generated for the FUTURE window.")
    logger.info("="*60)


if __name__ == "__main__":
    main()
