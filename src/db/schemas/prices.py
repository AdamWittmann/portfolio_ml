from sqlalchemy import MetaData, Table, Column, Integer, String, Date, Float

metadata = MetaData()

prices = Table(
    "prices",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String),
    Column("date", Date),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", Float),
    Column("sma_20", Float),
    Column("sma_50", Float),
    Column("sma_200", Float),
    Column("daily_return", Float),
    Column("volatility", Float),
    Column("drawdown", Float),
    Column("rsi", Float),
    Column("macd", Float),
    Column("macd_signal", Float),
    Column("bb_upper", Float),
    Column("bb_lower", Float),
    Column("bb_width", Float),
    Column("sentiment_score", Float),
)