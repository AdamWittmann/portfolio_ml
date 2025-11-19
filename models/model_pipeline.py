# model_pipeline.py - ENHANCED VERSION

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.getenv("DATABASE_URL"))

def create_binary_target(df: pd.DataFrame, threshold: float = 0.03, days_ahead: int = 5) -> pd.DataFrame:
    """
    Create binary target variable: will the stock exceed threshold return in N days?
    
    Args:
        df: DataFrame with columns ['date', 'symbol', 'close', ...]
        threshold: Minimum return to classify as 'buy' (default 3%)
        days_ahead: Lookahead window in days (default 5)
        
    Returns:
        DataFrame with 'target' and 'future_return' columns added
    """
    df = df.sort_values(['symbol', 'date']).copy()
    
    df['future_return'] = df.groupby('symbol')['close'].pct_change(periods=days_ahead).shift(-days_ahead)
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    rows_before = len(df)
    df = df.dropna(subset=['future_return'])
    rows_after = len(df)
    
    logger.info(f"Dropped {rows_before - rows_after} rows (last {days_ahead} days per ticker)")
    logger.info(f"\nTarget distribution:\n{df['target'].value_counts()}")
    logger.info(f"\nBuy rate: {df['target'].mean():.2%}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y.
    Drops non-feature columns.
    """
    # Columns to exclude from features
    exclude_cols = ['id', 'symbol', 'date', 'target', 'future_return']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # Keep metadata for analysis
    metadata = df[['symbol', 'date', 'target']]
    
    logger.info(f"\nFeatures: {len(feature_cols)} columns")
    logger.info(f"Feature names: {feature_cols}")
    
    return X, y, metadata


def split_data_by_time(df: pd.DataFrame, train_end_date: str, test_start_date: str):
    """
    Split data temporally - critical for time series!
    Never train on future data.
    
    Args:
        df: Full dataset with 'date' column
        train_end_date: Last date to include in training (e.g., '2023-12-31')
        test_start_date: First date to include in testing (e.g., '2024-01-01')
    
    Returns:
        X_train, X_test, y_train, y_test, test_metadata
    """
    # Convert string dates to datetime for comparison
    train_end = pd.to_datetime(train_end_date).date()
    test_start = pd.to_datetime(test_start_date).date()
    
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    logger.info(f"\n📅 Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"📅 Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    logger.info(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, test_metadata = prepare_features(test_df)
    
    logger.info(f"\nTrain buy rate: {y_train.mean():.2%}")
    logger.info(f"Test buy rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, test_metadata

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier with class weighting for imbalanced data.
    """
    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    logger.info(f"\n🎯 Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        early_stopping_rounds=10,  # Moved here from fit()
        eval_metric='logloss'
    )
    
    logger.info("\n🚀 Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info("✅ Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive evaluation with multiple metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n" + "="*60)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info("="*60)
    
    # Basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    for metric, value in metrics.items():
        logger.info(f"{metric:12}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                 Predicted: No  | Predicted: Yes")
    logger.info(f"Actual: No    |    {cm[0,0]:6d}     |    {cm[0,1]:6d}")
    logger.info(f"Actual: Yes   |    {cm[1,0]:6d}     |    {cm[1,1]:6d}")
    
    # Classification report
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Dont Buy', 'Buy'])}")
    
    return metrics, y_pred, y_pred_proba


def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance.png'):
    """
    Visualize which features matter most.
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n📈 Feature importance plot saved to {save_path}")
    plt.close()
    
    return importance_df


def save_model(model, filepath='models/stock_classifier.json'):
    """
    Save trained model to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    logger.info(f"\n💾 Model saved to {filepath}")


def main():
    """
    Full training pipeline.
    """
    logger.info("="*60)
    logger.info("🎯 STOCK CLASSIFIER TRAINING PIPELINE")
    logger.info("="*60)
    
    # 1. Load data
    logger.info("\n📥 Loading data from database...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Create target variable
    logger.info("\n🎯 Creating target variable...")
    df = create_binary_target(df, threshold=0.03, days_ahead=5)
    
    # 3. Train/test split (time-based)
    logger.info("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test, test_metadata = split_data_by_time(
        df,
        train_end_date='2024-06-30',  # Train on data through mid-2024
        test_start_date='2024-07-01'  # Test on recent data
    )
    
    # 4. Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # 5. Evaluate
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 6. Feature importance
    importance_df = plot_feature_importance(model, X_train.columns)
    logger.info(f"\nTop 5 features:\n{importance_df.head()}")
    
    # 7. Save model
    save_model(model)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()