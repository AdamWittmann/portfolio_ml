# generate_metrics.py - Generate metrics from saved model and test data

import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'stock_classifier.json'
METRICS_PATH = PROJECT_ROOT / 'artifacts' / 'metrics' / 'model_metrics.csv'
TRADING_METRICS_PATH = PROJECT_ROOT / 'artifacts' / 'metrics' / 'trading_metrics.csv'

engine = create_engine(os.getenv("DATABASE_URL"))


def create_binary_target(df: pd.DataFrame, threshold: float = 0.03, days_ahead: int = 7):
    """Create binary target variable"""
    df = df.sort_values(['symbol', 'date']).copy()
    
    df['future_close'] = df.groupby('symbol')['close'].shift(-days_ahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    df = df.dropna(subset=['future_return', 'future_close'])
    
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix X and target vector y"""
    exclude_cols = ['id', 'symbol', 'date', 'target', 'future_return', 'future_close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    metadata = df[['symbol', 'date', 'target', 'future_return']].copy()
    
    return X, y, metadata


def split_data_by_time(df: pd.DataFrame, train_end_date: str, test_start_date: str):
    """Split data temporally"""
    train_end = pd.to_datetime(train_end_date).date()
    test_start = pd.to_datetime(test_start_date).date()
    
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, test_metadata = prepare_features(test_df)
    
    return X_train, X_test, y_train, y_test, test_metadata


def evaluate_model(model, X, y):
    """Calculate metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1 Score': f1_score(y, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics, y_pred_proba


def calculate_trading_metrics(test_metadata, y_pred_proba, threshold=0.5):
    """Calculate trading performance metrics"""
    test_metadata = test_metadata.copy()
    test_metadata['pred_proba'] = y_pred_proba
    test_metadata['signal'] = (y_pred_proba >= threshold).astype(int)
    
    trades = test_metadata[test_metadata['signal'] == 1].copy()
    
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'avg_return': 0.0,
            'n_trades': 0
        }
    
    n_trades = len(trades)
    n_winners = (trades['future_return'] > 0.03).sum()
    win_rate = n_winners / n_trades
    
    avg_return = trades['future_return'].mean()
    std_return = trades['future_return'].std()
    
    sharpe = (avg_return / std_return * np.sqrt(52)) if std_return > 0 else 0
    
    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe
    }

def main():
    import sys
    
    logger.info("="*60)
    logger.info("📊 GENERATING METRICS FROM SAVED MODEL")
    logger.info("="*60)
    
    # Check if model exists
    logger.info(f"\n🔍 Looking for model at: {MODEL_PATH}")
    sys.stdout.flush()  # Force output
    
    if not MODEL_PATH.exists():
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        logger.error("   Please train the model first using model_pipeline.py")
        return
    
    logger.info("Model file exists, attempting to load...")
    sys.stdout.flush()
    
    # Load the saved model
    logger.info(f"\n📥 Loading model from {MODEL_PATH}")
    sys.stdout.flush()
    
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    logger.info("✅ Model loaded successfully")
    sys.stdout.flush()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n📥 Loading data from database...")
    sys.stdout.flush()
    
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    sys.stdout.flush()
    
    # Rest of the code...
if __name__ == "__main__":
    main()