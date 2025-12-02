# model_pipeline.py - OPTIMIZED VERSION

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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # your 6650M (adjust 0 or 1)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {torch.cuda.get_device_name(0)}")

engine = create_engine(os.getenv("DATABASE_URL"))


def create_binary_target(df: pd.DataFrame, threshold: float = 0.03, days_ahead: int = 7) -> pd.DataFrame:
    """
    Create binary target variable: will the stock exceed threshold return in N days?
    
    Args:
        df: DataFrame with columns ['date', 'symbol', 'close', ...]
        threshold: Minimum return to classify as 'buy' (default 3%)
        days_ahead: Lookahead window in days (default 7 for 1 week)
        
    Returns:
        DataFrame with 'target' and 'future_return' columns added
    """
    df = df.sort_values(['symbol', 'date']).copy()
    
    df['future_close'] = df.groupby('symbol')['close'].shift(-days_ahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    rows_before = len(df)
    df = df.dropna(subset=['future_return', 'future_close'])
    rows_after = len(df)
    
    logger.info(f"Dropped {rows_before - rows_after} rows (last {days_ahead} days per ticker)")
    logger.info(f"\nTarget distribution:\n{df['target'].value_counts()}")
    logger.info(f"Buy rate: {df['target'].mean():.2%}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y.
    Drops non-feature columns.
    """
    exclude_cols = ['id', 'symbol', 'date', 'target', 'future_return', 'future_close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    metadata = df[['symbol', 'date', 'target', 'future_return']].copy()
    
    logger.info(f"\nFeatures: {len(feature_cols)} columns")
    logger.info(f"Feature names: {feature_cols}")
    
    return X, y, metadata


def split_data_by_time(df: pd.DataFrame, train_end_date: str, test_start_date: str):
    train_end = pd.to_datetime(train_end_date).date()
    test_start = pd.to_datetime(test_start_date).date()
    
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    logger.info(f"\n Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    logger.info(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    
    X_train, y_train, _ = prepare_features(train_df)
    X_test, y_test, test_metadata = prepare_features(test_df)
    
    logger.info(f"\nTrain buy rate: {y_train.mean():.2%}")
    logger.info(f"Test buy rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, test_metadata


def train_xgboost_randomized_search(X_train, y_train, X_test, y_test, n_iter=30):
    """
    Train XGBoost with RandomizedSearchCV and TimeSeriesSplit cross-validation.
    Optimized parameter space for stock prediction.
    """
    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    logger.info(f"\n Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # Parameter distributions - original ranges that worked
    param_distributions = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.2),
        'min_child_weight': randint(3, 25),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0.5, 2.0),
    }
    
    # Base model
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        eval_metric='logloss'
    )
    
    # TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # RandomizedSearchCV
    logger.info(f"\n🚀 Starting RandomizedSearchCV: {n_iter} iterations × 3 CV folds = {n_iter * 3} total fits")
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=1,  # GPU training must be sequential
        verbose=0,  # Silent mode
        random_state=42,
        return_train_score=True
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"\nSearch complete!")
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    logger.info(f"Best parameters:")
    for param, value in search.best_params_.items():
        logger.info(f"   {param}: {value}")
    
    # Show top 5 parameter combinations
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    logger.info(f"\n Top 5 parameter combinations:")
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        logger.info(f"   Rank {idx+1}: CV Score = {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    
    return search.best_estimator_, search.best_params_, search.cv_results_

# def train_xgboost_grid_search(
#     X_train, y_train, X_test, y_test,
#     param_grid=None,
#     cv_splits=3
#     ):
#     """
#     Train an XGBoost model using GridSearchCV with TimeSeriesSplit.
#     Provides a clean template for experimentation.
#     """

#     # ---------------------------------------------------------
#     # 1) Class Weight (optional but useful for imbalance)
#     # ---------------------------------------------------------
#     n_neg = (y_train == 0).sum()
#     n_pos = (y_train == 1).sum()
#     scale_pos_weight = n_neg / max(n_pos, 1)

#     print(f"\n🎯 Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

#     # ---------------------------------------------------------
#     # 2) Default Parameter Grid (modify freely)
#     # ---------------------------------------------------------
#     if param_grid is None:
#         param_grid = {
#             "n_estimators": [100, 150, 200],
#             "max_depth": [3, 4, 5],
#             "learning_rate": [0.01, 0.05, 0.1],
#             "min_child_weight": [1, 5, 10],
#             "gamma": [0, 0.1, 0.2],
#             "subsample": [0.6, 0.8, 1.0],
#             "colsample_bytree": [0.6, 0.8, 1.0],
#             "reg_alpha": [0, 0.1, 0.5],
#             "reg_lambda": [1, 1.5, 2],
#         }

#     # ---------------------------------------------------------
#     # 3) Base Model
#     # ---------------------------------------------------------
#     base_model = XGBClassifier(
#         random_state=42,
#         tree_method="hist",
#         eval_metric="logloss",
#         scale_pos_weight=scale_pos_weight
#     )

#     # ---------------------------------------------------------
#     # 4) TimeSeries CV
#     # ---------------------------------------------------------
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     total_fits = (
#         np.prod([len(v) for v in param_grid.values()]) * cv_splits
#     )

#     print(f"\n🚀 Starting GridSearchCV: ~{total_fits} total fits")

#     # ---------------------------------------------------------
#     # 5) GridSearchCV
#     # ---------------------------------------------------------
#     search = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         scoring="roc_auc",
#         cv=tscv,
#         n_jobs=1,   # sequential = GPU-safe
#         verbose=1,
#         return_train_score=True,
#     )

#     search.fit(X_train, y_train)

#     # ---------------------------------------------------------
#     # 6) Results Summary
#     # ---------------------------------------------------------
#     print("\n✅ GridSearch Complete!")
#     print(f"🏆 Best CV ROC-AUC: {search.best_score_:.4f}")
#     print(f"🏆 Best Parameters:\n{search.best_params_}")

#     # Top results
#     results_df = pd.DataFrame(search.cv_results_)
#     results_df = results_df.sort_values("rank_test_score")

#     print("\n📊 Top Parameter Combinations:")
#     print(results_df[
#         ["rank_test_score", "mean_test_score", "std_test_score", "params"]
#     ].head())

#     return search.best_estimator_, search.best_params_, search.cv_results_

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive evaluation with multiple metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("\n" + "="*60)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info("="*60)
    
    # Basic metric
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
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=["Don't Buy", 'Buy'])}")
    
    return metrics, y_pred, y_pred_proba


def evaluate_trading_performance(test_metadata, y_pred_proba, threshold=0.5):
    """
    Evaluate model as a trading strategy.
    """
    test_metadata = test_metadata.copy()
    test_metadata['pred_proba'] = y_pred_proba
    test_metadata['signal'] = (y_pred_proba >= threshold).astype(int)
    
    trades = test_metadata[test_metadata['signal'] == 1].copy()
    
    if len(trades) == 0:
        logger.warning(f"⚠️  No trades generated at threshold {threshold}!")
        return None
    
    n_trades = len(trades)
    n_winners = (trades['future_return'] > 0.03).sum()
    win_rate = n_winners / n_trades
    
    avg_return = trades['future_return'].mean()
    median_return = trades['future_return'].median()
    std_return = trades['future_return'].std()
    
    sharpe = (avg_return / std_return * np.sqrt(52)) if std_return > 0 else 0
    
    logger.info(f"\n💰 TRADING PERFORMANCE (threshold={threshold})")
    logger.info(f"   Total signals: {n_trades}")
    logger.info(f"   Winners: {n_winners} ({win_rate:.2%})")
    logger.info(f"   Average return: {avg_return:.2%}")
    logger.info(f"   Median return: {median_return:.2%}")
    logger.info(f"   Sharpe ratio: {sharpe:.2f}")
    
    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe
    }


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
    logger.info(f"\nModel saved to {filepath}")


def main():
    """
    Full training pipeline with RandomizedSearchCV optimization.
    """
    logger.info("="*60)
    logger.info(" STOCK CLASSIFIER TRAINING PIPELINE - OPTIMIZED")
    logger.info("="*60)
    
    # 1. Load data
    logger.info("\nLoading data from database...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Create target variable
    logger.info("\nCreating target variable...")
    df = create_binary_target(df, threshold=0.03, days_ahead=7)
    
    # 3. Train/test split (time-based)
    logger.info("\n Splitting data...")
    X_train, X_test, y_train, y_test, test_metadata = split_data_by_time(
        df,
        train_end_date='2024-12-25',  # Train on data through mid-2023
        test_start_date='2024-12-26'  # Test on recent data
    )
    
    # 4. Train model with RandomizedSearchCV
    model, best_params, cv_results = train_xgboost_randomized_search(
        X_train, y_train, X_test, y_test, 
    )
    
    # 5. Evaluate
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 6. Trading performance
    trading_metrics = evaluate_trading_performance(test_metadata, y_pred_proba, threshold=0.5)
    
    # Test different thresholds
    logger.info("\n" + "="*60)
    logger.info("🔍 THRESHOLD ANALYSIS")
    logger.info("="*60)
    for thresh in [0.4, 0.5, 0.6]:
        evaluate_trading_performance(test_metadata, y_pred_proba, threshold=thresh)
    
    # 7. Feature importance
    importance_df = plot_feature_importance(model, X_train.columns)
    logger.info(f"\nTop 5 features:\n{importance_df.head()}")
    
    # 8. Save model
    save_model(model)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return model, metrics, best_params


if __name__ == "__main__":
    model, metrics, best_params = main()