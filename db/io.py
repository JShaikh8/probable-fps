"""Shared helpers for bulk upserts and reads against the Postgres schema."""
from __future__ import annotations

from typing import Iterable, Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session


CHUNK = 500


def bulk_upsert(session: Session, model, rows: list[dict],
                pk_cols: Sequence[str], update_cols: Sequence[str] | None = None):
    """
    Upsert rows into `model`, conflict on `pk_cols`, update `update_cols`
    (defaults to every non-PK column present on rows).
    """
    if not rows:
        return

    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        stmt = pg_insert(model.__table__).values(chunk)

        if update_cols is None:
            sample = chunk[0]
            cols = [c for c in sample.keys() if c not in pk_cols]
        else:
            cols = list(update_cols)

        set_map = {c: getattr(stmt.excluded, c) for c in cols}
        stmt = stmt.on_conflict_do_update(
            index_elements=list(pk_cols),
            set_=set_map,
        )
        session.execute(stmt)


def truncate(session: Session, *models):
    """Clear the given tables (CASCADE, RESTART IDENTITY)."""
    if not models:
        return
    names = ', '.join(m.__tablename__ for m in models)
    from sqlalchemy import text
    session.execute(text(f'TRUNCATE {names} RESTART IDENTITY CASCADE'))
