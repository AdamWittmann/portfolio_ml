# predict.py - Generate predictions from saved model

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
    """Prepare feature matrix X and target y"""
    exclude_cols = ['id', 'symbol', 'date', 'target', 'future_return', 'future_close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    metadata = df[['symbol', 'date', 'target', 'future_return']].copy()
    
    return X, y, metadata


def split_data_by_time(df, train_end_date, test_start_date):
    """Split data temporally"""
    train_end = pd.to_datetime(train_end_date).date()
    test_start = pd.to_datetime(test_start_date).date()
    
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, test_metadata = prepare_features(test_df)
    
    return X_train, X_test, y_train, y_test, test_metadata


def main():
    logger.info("="*60)
    logger.info("🔮 GENERATING PREDICTIONS FROM SAVED MODEL")
    logger.info("="*60)
    
    # 1. Load model
    logger.info("\n📥 Loading model...")
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/stock_classifier.json')
        logger.info("✅ Loaded models/stock_classifier.json")
    except FileNotFoundError:
        logger.error("❌ Model not found! Run model_pipeline.py first.")
        return
    
    # 2. Load data
    logger.info("\n📥 Loading data...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    logger.info(f"Loaded {len(df):,} rows")
    
    # 3. Create target
    logger.info("\n🎯 Creating target...")
    df = create_binary_target(df, threshold=0.03, days_ahead=7)
    
    # 4. Split (same as training)
    logger.info("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test, test_metadata = split_data_by_time(
        df,
        train_end_date='2023-06-30',
        test_start_date='2023-07-01'
    )
    
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info(f"Date range: {test_metadata['date'].min()} to {test_metadata['date'].max()}")
    
    # 5. Predict
    logger.info("\n🔮 Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 6. Save
    test_metadata = test_metadata.copy()
    test_metadata['pred_proba'] = y_pred_proba
    test_metadata['prediction'] = y_pred
    test_metadata.columns = ['ticker', 'date', 'target', 'future_return', 'pred_proba', 'prediction']
    test_metadata.to_csv('test_predictions.csv', index=False)
    
    logger.info(f"\n💾 Saved test_predictions.csv")
    logger.info(f"   Total predictions: {len(test_metadata):,}")
    logger.info(f"   Buy signals: {y_pred.sum():,} ({y_pred.mean():.1%})")
    
    logger.info("\n" + "="*60)
    logger.info("✅ DONE! Now run: python position_sizing_analysis.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()