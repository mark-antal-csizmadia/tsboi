import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
PG_DATABASE = os.environ.get("PG_DATABASE")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")
