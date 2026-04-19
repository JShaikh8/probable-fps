import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Load .env.local for local dev; in production (Render) DATABASE_URL is
# injected from the service's env var group, so the file is absent and
# load_dotenv is a silent no-op.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env.local'))

_raw = os.environ.get('DATABASE_URL') or os.environ.get('DATABASE_URI')
if not _raw:
    raise RuntimeError(
        'DATABASE_URL is not set. Add it to .env.local locally, or to the '
        'service environment in production.'
    )

# Render's managed Postgres exposes a connection string starting with
# "postgres://". SQLAlchemy 2.0 requires "postgresql://" or the driver-
# specific "postgresql+psycopg://" form. Normalize here so both work.
if _raw.startswith('postgres://'):
    _raw = _raw.replace('postgres://', 'postgresql+psycopg://', 1)
elif _raw.startswith('postgresql://') and '+psycopg' not in _raw.split('://', 1)[0]:
    _raw = _raw.replace('postgresql://', 'postgresql+psycopg://', 1)

DATABASE_URL = _raw

MLB_API_BASE  = 'https://statsapi.mlb.com/api/v1'
MLB_API_BASE2 = 'https://statsapi.mlb.com/api/v1.1'

# Seasons to ingest (regular season only)
DEFAULT_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Session:
    return SessionLocal()


def get_engine():
    return engine
