#Run this script once to set up db connection
import sqlalchemy as sa
from sqlalchemy import create_engine,text

engine = create_engine("postgresql://adam_ml:Woodruff5614!@localhost:5432/portfolio_ml")
with engine.connect() as conn:
    print("Connected!", conn)
    print(conn.execute(text("SELECT current_database();")).scalar())