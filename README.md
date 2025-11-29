🚀 Quick Start — Choose Your Setup
👉 If you ONLY want to run the Streamlit UI (NO database required):

Jump directly to: UI-Only Setup

👉 If you want the FULL pipeline (data ingestion → CSVs → PostgreSQL):

Jump to: Full Pipeline With PostgreSQL

🎨 UI-Only Setup (No Database Required)

This is the simplest and fastest way to use the project.
You do not need Postgres, a .env file, or the ingestion pipeline.

The Streamlit dashboard reads exclusively from the CSV backups in data_cache/.

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

4. Ensure Cached Data Exists

The UI loads data from:

portfolio_ml/
 └─ data_cache/
      AAPL.csv
      MSFT.csv
      ...


These files are already included in the repo.
If missing, run the ingestion pipeline (below).

5. Run the Streamlit Dashboard
streamlit run streamlit_app.py


(Replace streamlit_app.py with your actual filename if different.)

This launches the UI at:

http://localhost:8501

🎉 That’s It!

You now have the full dashboard running with:

Technical indicators

Charts

Tables

Risk metrics

No local database required.

🔥 Full Pipeline With PostgreSQL

This option is for developers who want to:

Pull fresh stock data

Generate CSV backups

Load data into a local PostgreSQL database

Run ML-ready analytics

Prepare full-feature datasets

If you're building backend features or ML workflows, use this mode.

1. Prerequisites

Install:

Python 3.10+

PostgreSQL 14+

Git

(Optional) Docker

2. Clone the Repo
git clone https://github.com/AdamWittmann/portfolio_ml.git
cd portfolio_ml

3. Create and Activate a Virtual Environment

macOS / Linux / Git Bash

python3 -m venv .venv
source .venv/bin/activate


Windows PowerShell

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

4. Install Python Dependencies
pip install --upgrade pip
pip install -r requirements.txt

5. Set Up PostgreSQL

You must create your own local database.
You are not connecting to Adam’s database.

Option A — Install PostgreSQL Natively
Linux (Debian/Ubuntu)
sudo apt-get install postgresql

macOS (Homebrew)
brew install postgresql
brew services start postgresql

Windows

Install from:
https://www.postgresql.org/download/

Option B — Run PostgreSQL in Docker
docker run --name portfolio_ml_db \
  -e POSTGRES_USER=mluser \
  -e POSTGRES_PASSWORD=mlpassword \
  -e POSTGRES_DB=portfolio_ml \
  -p 5432:5432 \
  -d postgres:16

6. Create Your Local Database (If Not Using Docker’s AUTO DB)

Enter psql:

sudo -u postgres psql


Then run:

CREATE USER mluser WITH PASSWORD 'mlpassword';
CREATE DATABASE portfolio_ml OWNER mluser;
GRANT ALL PRIVILEGES ON DATABASE portfolio_ml TO mluser;
\q

7. Create Your .env File

At the repo root:

touch .env


Add:

POSTGRES_USER=mluser
POSTGRES_PASSWORD=mlpassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=portfolio_ml

DATABASE_URL=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

8. Run the Data Loader Pipeline

This pulls fresh OHLCV data, writes CSV backups, and populates your DB.

bash run_data_loader.sh


If you're on Windows without bash, open the script and run the equivalent Python command inside it manually.

9. Verify Setup
CSVs Created?
ls data_cache

Database reachable?
psql $DATABASE_URL -c "\dt"

Python dependencies?
python -c "import pandas, yfinance, sqlalchemy; print('OK')"

🧭 Typical Developer Workflow

Activate your virtual environment

Pull latest project changes

Run the data loader to update datasets

Build or test ML models

Launch Streamlit (optional)

Commit code changes

📝 Troubleshooting
Database errors

Ensure Postgres is running

Check .env correctness

Ensure your DB user has privileges

command not found: streamlit

Install it:

pip install streamlit

ModuleNotFoundError

Your venv may not be active.

📌 Notes

The CSV backup layer allows users without Postgres to fully use the UI.

The database pipeline is only required for backend and ML feature development.

Your teammate does NOT need Adam’s credentials.

Everything is self-contained on their own local machine.
