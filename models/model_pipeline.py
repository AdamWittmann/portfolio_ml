# model_pipeline.py - OPTIMIZED VERSION

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV, learning_curve
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # your 6650M (adjust 0 or 1)

#Uncomment below to train on personal gpus (AMD)
# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using: {torch.cuda.get_device_name(0)}")

engine = create_engine(os.getenv("DATABASE_URL"))


def create_binary_target(df: pd.DataFrame, threshold: float = 0.03, days_ahead: int = 7) -> pd.DataFrame:
    """
    Create binary target variable: will the stock exceed threshold return in N days?
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


def generate_walkforward_folds(df: pd.DataFrame, fold_specs, train_start_date=None):
    """
    Build walk-forward train/test folds based on date ranges.

    fold_specs: list of (test_start_str, test_end_str) in 'YYYY-MM-DD' format.
    train_start_date: earliest date to use as training start (inclusive).
    """
    df = df.sort_values(['symbol', 'date']).copy()

    if train_start_date is None:
        train_start = df['date'].min()
    else:
        train_start = pd.to_datetime(train_start_date)

    folds = []

    for test_start_str, test_end_str in fold_specs:
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)

        train_mask = (df['date'] >= train_start) & (df['date'] < test_start)
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if train_df.empty or test_df.empty:
            logger.warning(
                f"Skipping fold {test_start_str} to {test_end_str}: "
                f"train_rows={len(train_df)}, test_rows={len(test_df)}"
            )
            continue

        X_train, y_train, _ = prepare_features(train_df)
        X_test, y_test, test_metadata = prepare_features(test_df)

        logger.info(
            f"\nFold {test_start_str} → {test_end_str}: "
            f"train_samples={len(X_train)}, test_samples={len(X_test)}"
        )

        folds.append({
            'test_start': test_start,
            'test_end': test_end,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': test_metadata,
        })

    if not folds:
        raise ValueError("No valid walk-forward folds were created. Check your date ranges.")

    logger.info(f"\nTotal walk-forward folds created: {len(folds)}")
    return folds


def train_xgboost_walkforward_search(folds, n_iter=30, random_state=42):
    """
    Manual RandomizedSearch over XGBoost hyperparameters using walk-forward folds.
    For each sampled param set, we train on each fold's train data and
    evaluate ROC-AUC on that fold's test data, then average over folds.
    """
    rng = np.random.default_rng(random_state)

    # Compute a global class weight based on all training data across folds
    y_all = pd.concat([f['y_train'] for f in folds], axis=0)
    n_neg = (y_all == 0).sum()
    n_pos = (y_all == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"\nGlobal class weight (scale_pos_weight): {scale_pos_weight:.2f}")

    # Parameter ranges (same spirit as before)
    def sample_params():
        return {
            'n_estimators': int(rng.integers(100, 300)),
            'max_depth': int(rng.integers(3, 8)),
            'learning_rate': float(rng.uniform(0.01, 0.2)),
            'min_child_weight': int(rng.integers(3, 25)),
            'gamma': float(rng.uniform(0.0, 0.5)),
            'subsample': float(rng.uniform(0.6, 1.0)),
            'colsample_bytree': float(rng.uniform(0.6, 1.0)),
            'reg_alpha': float(rng.uniform(0.0, 1.0)),
            'reg_lambda': float(rng.uniform(0.5, 2.0)),
        }

    best_params = None
    best_score = -np.inf
    results = []

    logger.info(f"\n🚀 Starting walk-forward RandomizedSearch: {n_iter} parameter sets")

    for i in range(n_iter):
        params = sample_params()
        fold_scores = []

        for fold_idx, fold in enumerate(folds):
            model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                tree_method='hist',
                eval_metric='logloss',
                **params
            )

            model.fit(fold['X_train'], fold['y_train'])
            y_proba = model.predict_proba(fold['X_test'])[:, 1]
            score = roc_auc_score(fold['y_test'], y_proba)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        logger.info(
            f"Iter {i+1}/{n_iter}: mean ROC-AUC = {mean_score:.4f} "
            f"(±{std_score:.4f}) with params={params}"
        )

        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    logger.info(f"\n🏆 Best mean ROC-AUC over folds: {best_score:.4f}")
    logger.info("🏆 Best parameters:")
    for k, v in best_params.items():
        logger.info(f"   {k}: {v}")

    results_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)
    return best_params, results_df


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


# ---------- NEW: ROC + regression helpers ----------

def plot_roc_curve_with_auc(y_true, y_scores, save_path='roc_curve.png'):
    """
    Plot ROC curve and report AUC for binary classifiers.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value:.3f})', color='royalblue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"📈 ROC curve saved to {save_path} (AUC = {auc_value:.3f})")
    return auc_value


def plot_regression_fit(y_true, y_pred, save_path='regression_fit.png'):
    """
    Plot predicted vs. actual values for regression diagnostics.
    """
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Ideal Fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Regression Fit (R²={r2:.3f}, RMSE={rmse:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"📊 Regression fit plot saved to {save_path} (R²={r2:.3f}, RMSE={rmse:.3f})")


def plot_residuals(y_true, y_pred, save_path='residuals.png'):
    """
    Plot residuals (actual - predicted) against predictions to inspect patterns.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"🧭 Residual plot saved to {save_path}")


def plot_learning_curves(
    estimator,
    X,
    y,
    cv=None,
    scoring='roc_auc',
    train_sizes=None,
    save_path='learning_curve.png'
):
    """
    Plot training and validation scores as a function of training set size.
    """
    if cv is None:
        cv = TimeSeriesSplit(n_splits=3)
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=1,
        shuffle=False,
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training Score')
    plt.plot(train_sizes_abs, valid_mean, 'o-', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    results_df = pd.DataFrame({
        'train_size': train_sizes_abs,
        'train_score_mean': train_mean,
        'validation_score_mean': valid_mean,
    })

    logger.info(f"📉 Learning curve saved to {save_path}\n{results_df}")
    return results_df


# ---------- UPDATED: evaluation uses optional ROC path ----------

def evaluate_model(model, X_test, y_test, roc_path=None):
    """
    Comprehensive evaluation with multiple metrics and optional ROC visualization.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logger.info("\n" + "=" * 60)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info("=" * 60)

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
    logger.info(f"Actual: No    |    {cm[0, 0]:6d}     |    {cm[0, 1]:6d}")
    logger.info(f"Actual: Yes   |    {cm[1, 0]:6d}     |    {cm[1, 1]:6d}")

    # Classification report (fixed quoting)
    report = classification_report(y_test, y_pred, target_names=["Don't Buy", "Buy"])
    logger.info("\n" + report)

    # Optional ROC curve
    if roc_path:
        plot_roc_curve_with_auc(y_test, y_pred_proba, save_path=roc_path)

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
    Full training pipeline with walk-forward validation and
    custom randomized hyperparameter search over XGBoost.
    """
    logger.info("="*60)
    logger.info(" STOCK CLASSIFIER TRAINING PIPELINE - WALK-FORWARD")
    logger.info("="*60)

    # 1. Load data
    logger.info("\nLoading data from database...")
    df = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", engine)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Create target variable
    logger.info("\nCreating target variable...")
    df = create_binary_target(df, threshold=0.03, days_ahead=7)

    # 3. Define walk-forward folds (you can tweak these ranges)
    logger.info("\nDefining walk-forward folds...")
    fold_specs = [
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
    ]

    train_start_date = df['date'].min().date()
    logger.info(f"Using {train_start_date} as initial training start date")

    folds = generate_walkforward_folds(
        df,
        fold_specs=fold_specs,
        train_start_date=train_start_date
    )

    # 4. Hyperparameter search across all folds
    best_params, cv_results = train_xgboost_walkforward_search(
        folds,
        n_iter=30,
        random_state=42
    )

    # 5. Evaluate best params on each fold (full walk-forward)
    logger.info("\n" + "="*60)
    logger.info("📊 WALK-FORWARD EVALUATION WITH BEST PARAMS")
    logger.info("="*60)

    final_model = None
    final_metrics = None
    final_metadata = None
    final_y_pred_proba = None
    final_class_weight = None  # for learning curve model

    for i, fold in enumerate(folds):
        logger.info(
            f"\n===== Fold {i+1}/{len(folds)}: "
            f"Train <= {fold['test_start'].date() - pd.Timedelta(days=1)}, "
            f"Test {fold['test_start'].date()} → {fold['test_end'].date()} ====="
        )

        # Recompute class weight per fold
        y_train = fold['y_train']
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            tree_method='hist',
            eval_metric='logloss',
            **best_params
        )

        model.fit(fold['X_train'], fold['y_train'])

        # NEW: pass roc_path to save per-fold ROC curve
        metrics, y_pred, y_pred_proba = evaluate_model(
            model,
            fold['X_test'],
            fold['y_test'],
            roc_path=f"roc_curve_fold_{i+1}.png",
        )

        evaluate_trading_performance(fold['metadata'], y_pred_proba, threshold=0.5)

        # Keep the last fold's model/metrics as "final" (most recent period)
        if i == len(folds) - 1:
            final_model = model
            final_metrics = metrics
            final_metadata = fold['metadata']
            final_y_pred_proba = y_pred_proba
            final_class_weight = scale_pos_weight

    # 6. Feature importance + learning curves based on final (most recent) model
    if final_model is not None:
        importance_df = plot_feature_importance(final_model, folds[-1]['X_train'].columns)
        logger.info(f"\nTop 5 features:\n{importance_df.head()}")

        # NEW: learning curve on final training set
        plot_learning_curves(
            XGBClassifier(
                scale_pos_weight=final_class_weight,
                random_state=42,
                tree_method='hist',
                eval_metric='logloss',
                **best_params,
            ),
            folds[-1]['X_train'],
            folds[-1]['y_train'],
            cv=TimeSeriesSplit(n_splits=3),
            scoring='roc_auc',
            save_path='learning_curve.png',
        )

        # 7. Save final model
        save_model(final_model)
    else:
        logger.warning("No final model was trained — check fold definitions.")
        final_metrics = {}
        best_params = {}

    logger.info("\n" + "="*60)
    logger.info("✅ WALK-FORWARD PIPELINE COMPLETE!")
    logger.info("="*60)

    return final_model, final_metrics, best_params


if __name__ == "__main__":
    model, metrics, best_params = main()
