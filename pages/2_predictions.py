# pages/2_📮_Predictions.py - Multi-Stock Forecast Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

alpaca_key    = os.getenv("ALPACA_API_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

st.set_page_config(page_title="Stock Predictions", page_icon="", layout="wide")

# ── Alpaca Setup ──────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

@st.cache_resource
def load_alpaca(api_key, secret_key):
    return TradingClient(api_key, secret_key, paper=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/stock_classifier.json')
        return model
    except FileNotFoundError:
        return None

model = load_model()

st.title("Multi-Stock Forecast Dashboard")
st.markdown("### Compare ML predictions across multiple stocks simultaneously")

if model is None:
    st.error("❌ Model not found! Please run `model_pipeline.py` first to train the model.")
    st.stop()
else:
    st.success("✅ Model loaded successfully")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Forecast Settings")

default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
ticker_input = st.sidebar.text_area(
    "Stock Tickers (one per line)",
    value='\n'.join(default_tickers),
    height=200,
    help="Enter stock ticker symbols, one per line"
)
tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Minimum confidence to trigger a BUY signal"
)

lookback_days = st.sidebar.number_input(
    "Lookback Period (days)", min_value=200, max_value=1000, value=365,
    help="How many days of historical data to fetch"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Filters")
show_only_buys = st.sidebar.checkbox("Show only BUY signals", value=False)
min_confidence_display = st.sidebar.slider(
    "Minimum Confidence to Display", min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

# ── Alpaca Trading Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Alpaca Paper Trading")

if not ALPACA_AVAILABLE:
    st.sidebar.warning("alpaca-py not installed. Run: pip install alpaca-py")
else:

    trading_enabled      = st.sidebar.toggle("Enable Paper Trading", value=False)
    trade_amount         = st.sidebar.number_input("Trade Amount (USD)", min_value=10, max_value=10000, value=1000)
    min_trade_confidence = st.sidebar.slider("Min Confidence to Trade", 0.5, 1.0, 0.70, 0.05)
    hold_days            = st.sidebar.number_input("Hold Period (days)", min_value=1, max_value=30, value=5,
                                                   help="Automatically close positions after this many days")

    alpaca = None
    if alpaca_key and alpaca_secret:
        try:
            alpaca = load_alpaca(alpaca_key, alpaca_secret)
            account = alpaca.get_account()
            st.sidebar.success(f"✅ Connected")
            st.sidebar.metric("Buying Power",    f"${float(account.buying_power):,.2f}")
            st.sidebar.metric("Portfolio Value",  f"${float(account.portfolio_value):,.2f}")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")
            alpaca = None

# ── Feature Calculation ───────────────────────────────────────────────────────
def calculate_features(df):
    df = df.copy()
    df['sma_20']  = df['close'].rolling(20).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['daily_return'] = df['close'].pct_change()
    df['volatility']   = df['daily_return'].rolling(20).std()
    df['peak']     = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['peak']) / df['peak']

    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    sma20  = df['close'].rolling(20).mean()
    std20  = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    return df

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'Volume',
    'sma_20', 'sma_50', 'sma_200', 'daily_return', 'volatility',
    'drawdown', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width'
]

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, days):
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            return None
        df = pd.DataFrame(index=stock_data.index)
        for col, src in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('Volume','Volume')]:
            v = stock_data[src].values
            df[col] = v.flatten() if hasattr(v, 'flatten') else v
        df = calculate_features(df).dropna()
        return df if len(df) > 0 else None
    except Exception:
        return None

# ── Auto-Close Positions Past Hold Period ─────────────────────────────────────
def close_expired_positions(alpaca_client, hold_days):
    """Close any bot-opened positions older than hold_days."""
    closed = []
    try:
        cutoff = datetime.now() - timedelta(days=hold_days)
        orders = alpaca_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED))
        # Find filled BUY orders placed by this bot (magic tag in client_order_id)
        bot_buys = [
            o for o in orders
            if o.client_order_id and o.client_order_id.startswith("xgb_")
            and o.side == OrderSide.BUY
            and o.filled_at and o.filled_at.replace(tzinfo=None) < cutoff
        ]
        positions = {p.symbol: p for p in alpaca_client.get_all_positions()}
        for order in bot_buys:
            sym = order.symbol
            if sym in positions:
                close_order = MarketOrderRequest(
                    symbol=sym,
                    qty=positions[sym].qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"xgb_close_{sym}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                alpaca_client.submit_order(close_order)
                closed.append(sym)
    except Exception as e:
        st.warning(f"Auto-close check failed: {e}")
    return closed

# ── Run Predictions ───────────────────────────────────────────────────────────
st.header("Forecast Results")
trade_log = []

with st.spinner(f"Analyzing {len(tickers)} stocks..."):

    # Auto-close expired positions before placing new ones
    if ALPACA_AVAILABLE and alpaca and trading_enabled:
        closed_syms = close_expired_positions(alpaca, hold_days)
        if closed_syms:
            st.info(f"🔄 Auto-closed positions past {hold_days}-day hold: {', '.join(closed_syms)}")

    results = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, lookback_days)
        if df is None or len(df) == 0:
            continue

        latest    = df[FEATURE_COLS].iloc[-1:].copy()
        pred_proba = model.predict_proba(latest)[:, 1][0]
        pred_class = 1 if pred_proba >= threshold else 0

        current_price    = df['close'].iloc[-1]
        prev_price       = df['close'].iloc[-2]
        price_change_pct = ((current_price - prev_price) / prev_price) * 100

        results.append({
            'Ticker':        ticker,
            'Signal':        'BUY' if pred_class == 1 else 'HOLD',
            'Confidence':    pred_proba,
            'Current Price': current_price,
            'Price Change %': price_change_pct,
            'Target Price':  current_price * 1.03,
            'RSI':           latest['rsi'].values[0],
            'MACD':          latest['macd'].values[0],
            'Volatility':    latest['volatility'].values[0],
            'Drawdown %':    latest['drawdown'].values[0] * 100,
            'SMA 20':        latest['sma_20'].values[0],
            'SMA 50':        latest['sma_50'].values[0],
        })

        # ── Execute Trade ─────────────────────────────────────────────────────
        if (ALPACA_AVAILABLE and alpaca and trading_enabled
                and pred_class == 1
                and pred_proba >= min_trade_confidence):
            try:
                order = MarketOrderRequest(
                    symbol=ticker,
                    notional=trade_amount,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"xgb_{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                result = alpaca.submit_order(order)
                trade_log.append({
                    'Ticker':     ticker,
                    'Action':     'BUY',
                    'Amount':     f"${trade_amount:,.0f}",
                    'Confidence': f"{pred_proba:.1%}",
                    'Status':     '✅ Placed',
                    'Order ID':   str(result.id)[:8] + "...",
                    'Hold Until': (datetime.now() + timedelta(days=hold_days)).strftime('%Y-%m-%d')
                })
                st.toast(f"✅ BUY ${trade_amount:,.0f} of {ticker} ({pred_proba:.1%} confidence)", icon="🟢")
            except Exception as e:
                trade_log.append({
                    'Ticker':     ticker,
                    'Action':     'BUY',
                    'Amount':     f"${trade_amount:,.0f}",
                    'Confidence': f"{pred_proba:.1%}",
                    'Status':     f"❌ Failed: {str(e)[:40]}",
                    'Order ID':   '-',
                    'Hold Until': '-'
                })

if len(results) == 0:
    st.error("❌ No valid data found for any tickers.")
    st.stop()

# ── Trade Log ─────────────────────────────────────────────────────────────────
if trade_log:
    st.subheader("📋 Today's Trade Log")
    st.dataframe(pd.DataFrame(trade_log), use_container_width=True)

# ── Open Positions ────────────────────────────────────────────────────────────
if ALPACA_AVAILABLE and alpaca:
    try:
        positions = alpaca.get_all_positions()
        bot_positions = [p for p in positions if p.symbol in tickers]
        if bot_positions:
            st.subheader("📊 Open Positions")
            pos_data = [{
                'Symbol':     p.symbol,
                'Qty':        p.qty,
                'Avg Entry':  f"${float(p.avg_entry_price):.2f}",
                'Current':    f"${float(p.current_price):.2f}",
                'P&L':        f"${float(p.unrealized_pl):.2f}",
                'P&L %':      f"{float(p.unrealized_plpc)*100:.2f}%",
                'Market Val': f"${float(p.market_value):.2f}",
            } for p in bot_positions]
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True)
    except Exception:
        pass

# ── Results Table ─────────────────────────────────────────────────────────────
results_df  = pd.DataFrame(results)
filtered_df = results_df.copy()
if show_only_buys:
    filtered_df = filtered_df[filtered_df['Signal'] == 'BUY']
if min_confidence_display > 0:
    filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence_display]
filtered_df = filtered_df.sort_values('Confidence', ascending=False)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("BUY Signals", f"{(results_df['Signal']=='BUY').sum()}/{len(results_df)}")
with col2:
    st.metric("Average Confidence", f"{results_df['Confidence'].mean():.1%}")
with col3:
    st.metric("High Confidence (>70%)", f"{(results_df['Confidence']>=0.7).sum()}")
with col4:
    st.metric("Avg Price Change", f"{results_df['Price Change %'].mean():+.2f}%")

st.subheader(f"🎯 Predictions ({len(filtered_df)} stocks)")

display_df = filtered_df.copy()
display_df['Confidence']    = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
display_df['Price Change %']= display_df['Price Change %'].apply(lambda x: f"{x:+.2f}%")
display_df['Target Price']  = display_df['Target Price'].apply(lambda x: f"${x:.2f}")
display_df['RSI']           = display_df['RSI'].apply(lambda x: f"{x:.1f}")
display_df['MACD']          = display_df['MACD'].apply(lambda x: f"{x:.2f}")
display_df['Volatility']    = display_df['Volatility'].apply(lambda x: f"{x:.4f}")
display_df['Drawdown %']    = display_df['Drawdown %'].apply(lambda x: f"{x:.2f}%")
display_df['SMA 20']        = display_df['SMA 20'].apply(lambda x: f"${x:.2f}")
display_df['SMA 50']        = display_df['SMA 50'].apply(lambda x: f"${x:.2f}")

def color_signal(val):
    return 'background-color: #90EE90' if val == 'BUY' else 'background-color: #FFB6C1'

st.dataframe(display_df.style.applymap(color_signal, subset=['Signal']), use_container_width=True, height=400)

# ── Visualizations ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Confidence Distribution", "Top Picks", "Price vs Confidence"])

with tab1:
    fig = px.histogram(
        results_df, x='Confidence', nbins=20, color='Signal',
        title="Confidence Score Distribution",
        color_discrete_map={'BUY': 'green', 'HOLD': 'red'}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold: {threshold:.0%}")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Top 5 BUY Opportunities (Highest Confidence)")
    buy_signals = results_df[results_df['Signal'] == 'BUY'].nlargest(5, 'Confidence')
    if len(buy_signals) == 0:
        st.warning("No BUY signals at current threshold.")
    else:
        for idx, row in buy_signals.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
                with c1:
                    st.markdown(f"### {row['Ticker']}")
                    st.markdown(f"**{row['Confidence']:.1%}** confidence")
                with c2:
                    st.metric("Current", f"${row['Current Price']:.2f}", delta=f"{row['Price Change %']:+.2f}%")
                with c3:
                    st.metric("Target", f"${row['Target Price']:.2f}", delta="+3.0%")
                with c4:
                    st.write(f"**RSI:** {row['RSI']:.1f} | **MACD:** {row['MACD']:.2f}")
                    st.write(f"**Vol:** {row['Volatility']:.4f} | **DD:** {row['Drawdown %']:.2f}%")
                st.markdown("---")

with tab3:
    fig = px.scatter(
        results_df, x='Confidence', y='Price Change %', color='Signal',
        size='Volatility', hover_data=['Ticker', 'RSI', 'MACD'],
        title="Price Change vs Model Confidence",
        color_discrete_map={'BUY': 'green', 'HOLD': 'red'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ How to Interpret This Dashboard"):
    st.markdown("""
    **Signal Types:**
    - 🟢 **BUY**: Model confidence ≥ threshold, predicts >3% return in 7 days
    - 🔴 **HOLD**: Model confidence < threshold or predicts <3% return

    **Confidence Levels:**
    - **50-60%**: Weak signal (barely above random chance)
    - **60-70%**: Moderate signal (meaningful edge)
    - **70%+**: Strong signal (high conviction)

    **Trading:**
    - Enter API keys in the sidebar to enable paper trading
    - Trades are tagged with `xgb_` prefix so the bot can track and auto-close them
    - Positions are automatically sold after the configured hold period
    - Only signals above the **Min Confidence to Trade** threshold trigger orders

    **Key Metrics:**
    - **RSI**: >70 overbought, <30 oversold
    - **MACD**: positive = bullish, negative = bearish
    - **Volatility**: higher = riskier
    - **Drawdown**: distance from recent peak

    **Important:** This is a signal generator, not financial advice.
    Past performance doesn't guarantee future results.
    """)