Portfolio ML Analyzer

A lightweight end-to-end project that ingests stock market data, engineers features, stores them in Postgres, and prepares everything for a machine-learning driven portfolio dashboard.

What This Project Does

Pulls historical OHLCV data for a set of tickers using yfinance

Caches data locally for quick, repeatable runs

Computes core technical + risk features:

Moving Averages (20, 50, 200)

Daily returns

Rolling volatility

Drawdown

Stores cleaned + enriched data in a PostgreSQL database

Ensures one clean row per (symbol, date) to support:

Performance scoring

Risk profiling

Portfolio optimization

Future ML models (e.g. classification of high/medium/low/underperforming)

Tech Stack

Language: Python

Data: yfinance, pandas

Database: PostgreSQL + SQLAlchemy

Planned UI: Streamlit dashboard (local web app)

Planned ML: Tree-based and/or linear models on engineered features for:

Stock-level performance scores

Risk tags

Suggested allocations based on watchlist/portfolio

How It Fits Together

Data Loader

Fetches/updates OHLCV data

Computes features

Syncs into Postgres (idempotent: avoids duplicate rows)

Database

Central source of truth for historical prices + features

Designed to be easily consumed by ML training scripts and the dashboard

Dashboard (up next)

Will read from Postgres

Show per-stock status (high/medium/low/underperforming)

Display basic risk metrics and trends

Goals

Build a realistic, production-style pipeline (not just a notebook)

Learn full-stack ML patterns: data → features → DB → UI → ML

Provide a foundation for experimenting with:

Sentiment signals

Fundamental data

Portfolio optimization

Simple predictive models (with honest evaluation)

Status

✅ Data ingestion + caching

✅ Feature engineering for core technical metrics

✅ PostgreSQL integration + de-duplicated storage

🔜 Streamlit dashboard

🔜 ML scoring + recommendations
