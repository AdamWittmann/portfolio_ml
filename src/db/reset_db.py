from sqlalchemy import create_engine, text
from schemas.prices import metadata
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

print("⚠️  WARNING: This will DELETE all data in the prices table!")
print("The table will be recreated with new columns (RSI, MACD, Bollinger Bands, sentiment)")
confirm = input("Type 'YES' to continue: ")

if confirm == "YES":
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS prices CASCADE"))
        conn.commit()
    
    metadata.create_all(engine)
    print("✅ Table dropped and recreated with updated schema!")
    print("Now run: python src/data_loader.py")
else:
    print("❌ Aborted. No changes made.")