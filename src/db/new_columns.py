from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    # Add new columns
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS rsi FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS macd FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS macd_signal FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS bb_upper FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS bb_lower FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS bb_width FLOAT"))
    conn.execute(text("ALTER TABLE prices ADD COLUMN IF NOT EXISTS sentiment_score FLOAT"))
    conn.commit()

print("✅ All columns added successfully!")