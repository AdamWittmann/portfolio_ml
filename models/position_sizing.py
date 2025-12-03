# position_sizing_analysis.py - Standalone position sizing analysis
"""
Loads predictions from test_predictions.csv and applies ML-based position sizing
to optimize portfolio allocation. Compares dynamic sizing vs equal-weight strategy.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / 'artifacts' / 'predictions' / 'test_predictions.csv'
SIZED_PORTFOLIO_PATH = PROJECT_ROOT / 'artifacts' / 'predictions' / 'portfolio_results_with_sizing.csv'


def train_position_sizing_model(signals_df):
    """
    Train a model to predict optimal position size based on signal strength
    and portfolio state.
    
    Features:
    - signal_strength: How confident is the prediction (pred_proba - 0.5)
    - recent_win_rate: Rolling 20-trade win rate
    - recent_volatility: Rolling std of returns
    - drawdown_pct: Current drawdown from peak
    - n_positions: Number of concurrent positions
    
    Target:
    - optimal_size: Kelly Criterion-based position size (5-25% of portfolio)
    """
    logger.info("\n" + "="*60)
    logger.info("🎯 TRAINING POSITION SIZING MODEL")
    logger.info("="*60)
    
    df = signals_df.copy()
    
    # Feature 1: Signal strength (how confident is the prediction)
    df['signal_strength'] = df['pred_proba'] - 0.5
    
    # Sort by ticker and date for rolling calculations
    df = df.sort_values(['ticker', 'date'])
    
    # Feature 2: Recent win rate (rolling 20 trades)
    df['recent_win_rate'] = df.groupby('ticker')['future_return'].transform(
        lambda x: (x > 0).rolling(20, min_periods=5).mean()
    )
    
    # Feature 3: Recent volatility
    df['recent_volatility'] = df.groupby('ticker')['future_return'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    
    # Feature 4: Drawdown from peak
    df['cumulative_return'] = df.groupby('ticker')['future_return'].transform('cumsum')
    df['peak'] = df.groupby('ticker')['cumulative_return'].transform('cummax')
    df['drawdown_pct'] = (df['cumulative_return'] - df['peak']) / (df['peak'].abs() + 1)
    
    # Feature 5: Number of concurrent positions (approximate)
    df['n_positions'] = df.groupby('date').cumcount() + 1
    
    # Target: Optimal position size using Kelly Criterion heuristic
    # Kelly = (p * b - q) / b, where p=win_prob, q=loss_prob, b=win/loss ratio
    win_rate = df['recent_win_rate'].fillna(0.5)
    avg_win = df[df['future_return'] > 0].groupby('ticker')['future_return'].transform('mean').fillna(0.01)
    avg_loss = df[df['future_return'] < 0].groupby('ticker')['future_return'].transform('mean').fillna(-0.01)
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
    df['optimal_size'] = kelly_fraction.clip(0.05, 0.25)  # Constrain to 5-25% position size
    
    # Prepare features
    features = ['signal_strength', 'recent_win_rate', 'recent_volatility', 
                'drawdown_pct', 'n_positions']
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=features + ['optimal_size'])
    
    X = df_clean[features]
    y = df_clean['optimal_size']
    
    logger.info(f"Training samples: {len(X):,}")
    logger.info(f"Target range: {y.min():.1%} - {y.max():.1%}")
    
    # Train XGBoost regressor
    sizer_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    
    sizer_model.fit(X, y, verbose=False)
    
    # Show feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': sizer_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n📊 Position Sizing Feature Importance:")
    for _, row in importance.iterrows():
        logger.info(f"   {row['feature']:20s}: {row['importance']:.1%}")
    
    return sizer_model


def apply_position_sizing(signals_df, sizer_model, portfolio_value=20000):
    """
    Apply position sizing model to signals and calculate portfolio returns.
    
    Returns:
    - DataFrame with position sizes, capital allocation, and profits per trade
    """
    logger.info("\n" + "="*60)
    logger.info("💰 APPLYING DYNAMIC POSITION SIZING")
    logger.info("="*60)
    
    df = signals_df.copy()
    
    # Recreate features (same as training)
    df['signal_strength'] = df['pred_proba'] - 0.5
    df = df.sort_values(['ticker', 'date'])
    df['recent_win_rate'] = df.groupby('ticker')['future_return'].transform(
        lambda x: (x > 0).rolling(20, min_periods=5).mean()
    )
    df['recent_volatility'] = df.groupby('ticker')['future_return'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    df['cumulative_return'] = df.groupby('ticker')['future_return'].transform('cumsum')
    df['peak'] = df.groupby('ticker')['cumulative_return'].transform('cummax')
    df['drawdown_pct'] = (df['cumulative_return'] - df['peak']) / (df['peak'].abs() + 1)
    df['n_positions'] = df.groupby('date').cumcount() + 1
    
    features = ['signal_strength', 'recent_win_rate', 'recent_volatility', 
                'drawdown_pct', 'n_positions']
    
    # Predict position sizes
    df_clean = df.dropna(subset=features)
    X = df_clean[features]
    df_clean['position_size'] = sizer_model.predict(X).clip(0.05, 0.25)
    
    # Calculate capital allocation and profit
    df_clean['capital_allocated'] = portfolio_value * df_clean['position_size']
    df_clean['profit'] = df_clean['capital_allocated'] * df_clean['future_return']
    
    # Summary stats
    total_profit = df_clean['profit'].sum()
    total_return = total_profit / portfolio_value
    sharpe = df_clean['profit'].mean() / df_clean['profit'].std() * np.sqrt(252)
    avg_position = df_clean['position_size'].mean()
    win_rate = (df_clean['future_return'] > 0).mean()
    
    logger.info(f"\n💵 Portfolio Performance:")
    logger.info(f"   Initial capital: ${portfolio_value:,.0f}")
    logger.info(f"   Total profit: ${total_profit:,.0f}")
    logger.info(f"   Total return: {total_return:.2%}")
    logger.info(f"   Sharpe ratio: {sharpe:.2f}")
    logger.info(f"   Win rate: {win_rate:.1%}")
    logger.info(f"   Avg position size: {avg_position:.1%}")
    logger.info(f"   Number of trades: {len(df_clean):,}")
    
    return df_clean


def compare_strategies(signals_df, portfolio_results, portfolio_value=20000):
    """
    Compare dynamic position sizing vs equal-weight strategy.
    """
    logger.info("\n" + "="*60)
    logger.info("📊 COMPARISON: Dynamic Sizing vs Equal Weight")
    logger.info("="*60)
    
    # Equal weight strategy (fixed % per trade)
    fixed_size = 0.10  # 10% per trade
    signals_df['fixed_capital'] = portfolio_value * fixed_size
    signals_df['fixed_profit'] = signals_df['fixed_capital'] * signals_df['future_return']
    
    fixed_total = signals_df['fixed_profit'].sum()
    fixed_return = fixed_total / portfolio_value
    fixed_sharpe = signals_df['fixed_profit'].mean() / signals_df['fixed_profit'].std() * np.sqrt(252)
    fixed_win_rate = (signals_df['future_return'] > 0).mean()
    
    # Dynamic sizing stats
    dynamic_total = portfolio_results['profit'].sum()
    dynamic_return = dynamic_total / portfolio_value
    dynamic_sharpe = portfolio_results['profit'].mean() / portfolio_results['profit'].std() * np.sqrt(252)
    
    logger.info(f"\n💰 Equal Weight (10% per trade):")
    logger.info(f"   Total profit: ${fixed_total:,.0f}")
    logger.info(f"   Total return: {fixed_return:.2%}")
    logger.info(f"   Sharpe ratio: {fixed_sharpe:.2f}")
    logger.info(f"   Win rate: {fixed_win_rate:.1%}")
    
    logger.info(f"\n🎯 Dynamic Sizing (ML-optimized):")
    logger.info(f"   Total profit: ${dynamic_total:,.0f}")
    logger.info(f"   Total return: {dynamic_return:.2%}")
    logger.info(f"   Sharpe ratio: {dynamic_sharpe:.2f}")
    
    logger.info(f"\n📈 Improvement:")
    logger.info(f"   Profit difference: ${dynamic_total - fixed_total:+,.0f}")
    logger.info(f"   Return difference: {dynamic_return - fixed_return:+.2%}")
    logger.info(f"   Sharpe improvement: {dynamic_sharpe - fixed_sharpe:+.2f}")


def main():
    """
    Main execution: Load predictions, train position sizing model, compare strategies.
    """
    logger.info("="*60)
    logger.info("📊 POSITION SIZING ANALYSIS")
    logger.info("="*60)
    
    # Load saved predictions
    logger.info("\n📥 Loading saved predictions...")
    try:
        test_results = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        logger.error(f"\n❌ ERROR: {PREDICTIONS_PATH} not found!")
        logger.error("   Please run model_pipeline.py first to generate predictions.")
        return
    
    logger.info(f"Loaded {len(test_results):,} predictions")
    
    # Filter for buy signals only
    signals = test_results[test_results['prediction'] == 1].copy()
    logger.info(f"Buy signals: {len(signals):,} ({len(signals)/len(test_results):.1%} of total)")
    
    if len(signals) == 0:
        logger.error("\n❌ ERROR: No buy signals found in predictions!")
        return
    
    # Train position sizing model
    sizer_model = train_position_sizing_model(signals)
    
    # Apply position sizing
    portfolio_results = apply_position_sizing(
        signals, 
        sizer_model, 
        portfolio_value=20000
    )
    
    # Compare strategies
    compare_strategies(signals, portfolio_results, portfolio_value=20000)
    
    # Save results
    SIZED_PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    portfolio_results.to_csv(SIZED_PORTFOLIO_PATH, index=False)
    logger.info(f"\n💾 Saved {SIZED_PORTFOLIO_PATH}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ POSITION SIZING ANALYSIS COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()