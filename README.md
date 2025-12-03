Portfolio ML Analyzer

Welcome to the walkthrough for my professor! This repository contains the code, notebooks, and trained artifacts for my machine-learning-driven portfolio dashboard.

🔗 Live demo: https://freebee.streamlit.app/

What to look at first:
- Notebooks: end-to-end experimentation in `notebooks/` (data cleaning, feature engineering, model evaluation).
- Models: saved inference artifacts in `models/` (used directly by the app).
- App entry point: `app.py` and Streamlit `pages/` for the UI.
- Reports: generated CSV metrics in `artifacts/metrics/`, prediction outputs in `artifacts/predictions/`, and figures in `artifacts/figures/` for quick grading reference.

⚠️ Note on training: the full retraining pipeline expects a local PostgreSQL database that is not hosted for this submission, so running fresh training end to end will fail. The saved model artifacts still load for prediction using the cached CSV data shipped in `data_cache/`.

Users can either:

Run ONLY the Streamlit dashboard UI using the provided CSVs

Run the FULL ingestion pipeline using their own local PostgreSQL database

🚀 Quick Start — Choose Your Setup
👉 To run JUST the dashboard UI (NO database needed):

Jump to UI-Only Setup

👉 To run the FULL pipeline (PostgreSQL + ingestion + CSV generation):

Jump to Full Pipeline With PostgreSQL

🧭 Guided tour for grading
- Open the notebooks in `notebooks/` to review the experiments.
- Inspect `models/` for the exported estimator files used in the app.
- Launch the Streamlit app (UI-only setup) to verify predictions load from the saved model and cached data.

🎨 UI-Only Setup (No Database Required)

This is the fastest and easiest way to use the project.
The Streamlit UI reads exclusively from CSV backups in data_cache/.

No database setup needed.
No .env file needed.
No ingestion pipeline needed.

1. Clone the Repository
git clone https://github.com/AdamWittmann/portfolio_ml.git
cd portfolio_ml

2. Create a Python Virtual Environment

macOS / Linux / Git Bash

python3 -m venv .venv
source .venv/bin/activate


Windows PowerShell

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Ensure data_cache/ Contains CSV Data

The dashboard loads data such as:

data_cache/
   AAPL.csv
   MSFT.csv
   ...


The repo already includes these.

5. Run the Streamlit Dashboard
streamlit run streamlit_app.py


(Replace streamlit_app.py with your actual filename.)

Dashboard will launch automatically at:

http://localhost:8501

🎉 That's All You Need for the UI!

You now have:

Stock charts

Risk metrics

Technical indicators

Data tables

Right out of the box — no Postgres required.

🔥 Full Pipeline With PostgreSQL

Use this version if you want:

Fresh data updates

Daily OHLCV ingestion

Data engineering

A real relational database backend

ML-ready datasets

This requires installing and configuring PostgreSQL locally.

1. Prerequisites

Install:

Python 3.10+

PostgreSQL 14+

Git

2. Clone the Repo
git clone https://github.com/AdamWittmann/portfolio_ml.git
cd portfolio_ml

3. Create & Activate Virtual Environment

macOS / Linux / Git Bash

python3 -m venv .venv
source .venv/bin/activate


Windows PowerShell

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

4. Install Python Dependencies
pip install --upgrade pip
pip install -r requirements.txt

5. Install PostgreSQL (Native Install Only — No Docker)
macOS (Homebrew)
brew install postgresql
brew services start postgresql

Ubuntu/Debian Linux
sudo apt-get update
sudo apt-get install postgresql

Windows

Install from the official PostgreSQL website:
https://www.postgresql.org/download/

6. Create Your Local Database

Enter psql:

macOS/Linux

sudo -u postgres psql


Windows

Open “SQL Shell (psql)” from Start Menu.

Inside psql, run:

CREATE USER mluser WITH PASSWORD 'mlpassword';
CREATE DATABASE portfolio_ml OWNER mluser;
GRANT ALL PRIVILEGES ON DATABASE portfolio_ml TO mluser;
\q


Feel free to change username/password — just keep them consistent with your .env.

7. Create Your .env File

At the project root:

touch .env


Add:

POSTGRES_USER=mluser
POSTGRES_PASSWORD=mlpassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=portfolio_ml

DATABASE_URL=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

8. Run the Data Loader Pipeline

This pulls market data, writes CSV backups, and populates your database.

bash run_data_loader.sh


If using Windows PowerShell (no bash), open the script and run the Python command inside manually.

9. Verify Everything Works
Check database tables:
psql $DATABASE_URL -c "\dt"

Check CSV cache:
ls data_cache

Check Python dependencies:
python -c "import pandas, sqlalchemy, yfinance; print('OK')"

🧭 Workflow Summary
UI-Only Users:

Clone repo

Install dependencies

Run Streamlit

Done

Full-Pipeline Users:

Clone repo

Install dependencies

Install PostgreSQL

Create database + .env

Run data loader

Launch Streamlit (optional)

📝 Troubleshooting
Streamlit not found
pip install streamlit

psql “connection refused”

Postgres service might not be running.

ModuleNotFoundError

Activate your venv:

source .venv/bin/activate

Permission denied when creating DB

Make sure you're running psql as the correct user (postgres superuser on Linux/macOS).
