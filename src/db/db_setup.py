#Run this script once to set up db connection
import sqlalchemy as sa
from sqlalchemy import create_engine,text
from schemas.prices import metadata
import os
from dotenv import load_dotenv
load_dotenv()
#Connect to the database
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    print("Connected!", conn)
    print(conn.execute(text("SELECT current_database();")).scalar())

metadata.create_all(engine)
print("**Tables Created**")