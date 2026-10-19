"""A run banked under an earlier public release resumes here as it did there, or fails by name
before any effect where the recorded format broke.

`expected.json` holds how each run resumes under the release that banked it.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
from _public_runs import SHAPES, Calls, Shape, register

from effective.engines.sqlite import SqliteApp

BANK = Path(__file__).parent / "fixtures" / "public-runs"
RECORD = json.loads((BANK / "expected.json").read_text())
EXPECTED = RECORD["shapes"]

BROKEN = {
    "55333986efa9ffaea26d8b3e0480ad4ea0865a77": ("UnreadFormat", frozenset(EXPECTED)),
}
"""Per release, the error a run it banked fails with here, and the shapes that fail with it: that
release marks no run with the recorded format's key."""


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.name)
def test_a_banked_run_resumes_as_it_did_there_or_fails_by_name(tmp_path: Path, shape: Shape):
    expected = EXPECTED[shape.name]
    db = tmp_path / "task.db"
    with closing(sqlite3.connect(db)) as store:
        store.executescript((BANK / f"{shape.name}.sql").read_text())
    app = SqliteApp(str(db))
    domain = Calls()
    register(app, shape, domain)
    recorded = _rows(app)
    snapshot = app.run_until_result(UUID(expected["task_id"]))
    written = _rows(app) - recorded
    app.close()
    assert snapshot is not None
    match BROKEN.get(RECORD["release"]):
        case (kind, shapes) if shape.name in shapes:
            assert snapshot.failure is not None
            assert (snapshot.failure.kind, domain.calls, written) == (kind, [], 0)
        case _:
            resumed = {"state": snapshot.state, "calls": domain.calls, "result": snapshot.result}
            assert resumed == {key: expected[key] for key in resumed}


def _rows(app: SqliteApp) -> int:
    """The checkpoint and ledger rows the store holds."""
    (checkpoints,) = app.conn.execute("SELECT count(*) FROM checkpoints").fetchone()
    (ledger,) = app.conn.execute("SELECT count(*) FROM ledger").fetchone()
    return checkpoints + ledger
