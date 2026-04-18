import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env.local'))

MONGO_URI = os.environ['MONGODB_URI']
DB_NAME   = os.environ.get('MONGODB_DB_NAME', 'pulse-arc')

MLB_API_BASE  = 'https://statsapi.mlb.com/api/v1'
MLB_API_BASE2 = 'https://statsapi.mlb.com/api/v1.1'

# Seasons to ingest (regular season only)
DEFAULT_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]
