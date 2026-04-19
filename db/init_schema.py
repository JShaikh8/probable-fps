"""
Bootstrap script — creates every table defined in db.models against DATABASE_URL.

Usage:
    python -m db.init_schema           # create missing tables
    python -m db.init_schema --drop    # drop all tables first (DANGER)
"""
from __future__ import annotations

import argparse
import sys

from config import get_engine
from db.models import Base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop', action='store_true',
                        help='Drop all tables before creating (destroys data)')
    args = parser.parse_args()

    engine = get_engine()

    if args.drop:
        confirm = input('About to DROP ALL TABLES. Type the DB host to confirm: ')
        if confirm not in str(engine.url):
            print('Host mismatch — aborting.')
            sys.exit(1)
        print('Dropping…')
        Base.metadata.drop_all(engine)

    print(f'Creating tables on {engine.url}')
    Base.metadata.create_all(engine)
    print(f'✓ {len(Base.metadata.tables)} tables ready: {", ".join(sorted(Base.metadata.tables))}')


if __name__ == '__main__':
    main()
