"""Reset the TEST Postgres: Absurd's per-queue tables, the ledger, and the projections.

The per-session hook in `tests/conftest.py` already deletes non-terminal TASKS, which is the flake
source — a leftover claimable task steals `work_batch` claims from later tests. This is the fuller
reset, for the two things that hook deliberately leaves alone:

- **completed rows accumulate.** Thousands are harmless (nothing claims them) but they make manual
  inspection miserable and slow a `TRUNCATE`-free debug session down.
- **projections and the ledger carry state across runs.** A projection test that inserts a fixed
  key fails on the *second* run with a duplicate-key error that has nothing to do with the change
  under test.

Deliberately a script and not inline in the justfile: the SQL is generated (Absurd's tables are
per-queue, so the table list is a query) and a heredoc inside a `just` recipe is the wrong place
for that.

**Scoped to the test DSN by the caller** (`just pgt-clean` builds it from `PGTEST_PORT`). It
truncates; run it against a database you are willing to lose.
"""

import os
import sys

import psycopg

QUEUE_TABLE_PREFIXES = ("t_", "c_", "e_", "r_")
"""Absurd's per-queue tables: tasks, checkpoints, events, runs (`absurd.sql`). One set per queue,
so the list has to be discovered rather than hard-coded."""

APP_TABLES = ("ledger",)
"""The substrate's canonical record. A consumer that builds a projection from it adds that table
here, so the two are truncated together: a ledger without its projection is a state no rebuild
produces."""


def clean(dsn: str) -> str:
    with psycopg.connect(dsn, autocommit=True) as conn:
        marker = conn.execute("SELECT to_regclass('_pgtest_disposable')").fetchone()
        if marker is None or marker[0] is None:
            raise SystemExit(
                "refusing to truncate: this database has not declared itself disposable.\n"
                "`scripts/pgtest_up.sh` creates the `_pgtest_disposable` marker; a dev database "
                "will not have it, and DATABASE_URL cannot tell the two apart."
            )
        queue_tables = [
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'absurd' "
                "AND (tablename LIKE 't\\_%' OR tablename LIKE 'c\\_%' "
                "     OR tablename LIKE 'e\\_%' OR tablename LIKE 'r\\_%')"
            ).fetchall()
        ]
        for table in queue_tables:
            # `{table:i}` — psycopg quotes the identifier; the schema rides the static text. The
            # same t-string SQL boundary the readers use, dogfooded on generated DDL.
            conn.execute(t"TRUNCATE absurd.{table:i} CASCADE")
        app = []
        for table in APP_TABLES:
            found = conn.execute(t"SELECT to_regclass({table})").fetchone()
            if found is not None and found[0] is not None:
                app.append(table)
        for table in app:
            # TRUNCATE is not blocked by the append-only trigger (which fires on UPDATE/DELETE) —
            # deliberately, since the trigger protects HISTORY and this resets a test fixture.
            conn.execute(t"TRUNCATE {table:i} CASCADE")
    return (
        f"pgt-clean: truncated {len(queue_tables)} absurd queue table(s) + {len(app)} app table(s)"
    )


if __name__ == "__main__":
    default = "postgresql://effective:effective@localhost:5432/effective"
    print(clean(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATABASE_URL", default)))
