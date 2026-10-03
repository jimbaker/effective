"""StoreArtifact is content-addressed.

Two contracts build identity on ``op_key(StoreArtifact)``: ``ctx.step``'s at-most-once
checkpointing and the human tier's approval event ``approve;{op_key(op)}``. The engine's
duplicate-name occurrence suffixing (``name#2``) backstops the checkpoint but not the approval
event, so a key on ``content_type`` alone would let one approval cover two gated stores.
Content-addressing (``artifact:{ct},{digest(value)}``) makes the key injective. It also keeps
per-child artifact ids distinct across a RecordingHandler gather, so no child artifact is dropped
on the join.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from effective.api import gather, store_artifact
from effective.domain import DomainOp
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import artifact_id, content_digest, op_key
from effective.handlers.recording import RecordingHandler
from effective.ops import StoreArtifact
from effective.sqlite import SqliteApp

# --- op_key injectivity + the approval-event contract -------------------------


def test_op_key_injective_for_store_artifact():
    ct = "application/json"
    k1 = op_key(StoreArtifact(value={"a": 1}, content_type=ct))
    k2 = op_key(StoreArtifact(value={"a": 2}, content_type=ct))
    k_same = op_key(StoreArtifact(value={"a": 1}, content_type=ct))
    assert k1 != k2  # distinct content -> distinct key
    assert k1 == k_same  # identical content -> identical key (idempotent, correct)
    assert k1.stored().startswith(f"artifact:{ct},sha256-")


def test_two_gated_artifacts_get_distinct_approval_events():
    """The teeth: the human tier's approval event is ``approve;{op_key(op)}``.
    Two same-typed, different-valued stores must demand two distinct approvals."""
    ct = "application/json"
    op1 = StoreArtifact(value={"doc": "A"}, content_type=ct)
    op2 = StoreArtifact(value={"doc": "B"}, content_type=ct)
    # equal events would let one approval cover both gated stores
    assert f"approve;{op_key(op1)}" != f"approve;{op_key(op2)}"


def _digest_under_seed(seed: str, snippet: str) -> str:
    """content_digest of `snippet` (an expression) in a fresh process at PYTHONHASHSEED=seed."""
    src = Path(__file__).parent.parent / "src"
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(src)!r})
        from effective.handlers.base import content_digest
        print(content_digest({snippet}))
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONHASHSEED": seed, "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_content_digest_is_stable_across_processes_for_set_values():
    """A value containing a set hashes identically across processes.

    Otherwise the checkpoint name, artifact id and approval-event name diverge when a fresh
    worker resumes a dead one. Sets iterate in PYTHONHASHSEED order, so only canonicalization
    makes the digest seed-independent."""
    snippet = "{'tags': {'red', 'orange', 'yellow', 'green', 'blue'}, 'n': 1}"
    digests = {_digest_under_seed(seed, snippet) for seed in ("1", "2", "17", "424242")}
    assert len(digests) == 1, digests


# --- RecordingHandler: content-addressed id, value stored ---------------------


def test_recording_returns_content_addressed_id_and_stores_value():
    def wf():
        a = yield from store_artifact({"x": 1}, "application/json")
        b = yield from store_artifact({"x": 2}, "application/json")
        return a, b

    h = RecordingHandler()
    out = h.run(wf)
    assert isinstance(out, tuple)  # completes (no suspension)
    a, b = out
    assert isinstance(a, str)
    assert isinstance(b, str)
    assert a != b  # distinct values -> distinct ids
    assert a == f"application/json,sha256-{content_digest({'x': 1})}"
    assert len(h.artifacts) == 2  # both persisted, no stale-read shadowing
    assert h.artifacts[a] == ("application/json", {"x": 1})
    assert h.artifacts[b] == ("application/json", {"x": 2})


def test_recording_identical_value_dedups():
    def wf():
        a = yield from store_artifact({"x": 1}, "application/json")
        b = yield from store_artifact({"x": 1}, "application/json")
        return a, b

    out = RecordingHandler().run(wf)
    assert isinstance(out, tuple)
    a, b = out
    assert a == b  # same content -> same id (content-addressing dedups)


# --- gather: child artifacts merged (were dropped), ids never collide ---------


def test_gather_merges_distinct_child_artifacts():
    def branch(v):
        def thunk():
            aid = yield from store_artifact({"v": v}, "application/json")
            return aid

        return thunk

    def parent():
        return (yield from gather([branch(1), branch(2)]))

    h = RecordingHandler()
    ids = h.run(parent)
    assert isinstance(ids, list)
    assert ids[0] != ids[1]  # no per-child sequence id that resets to 0
    assert len(h.artifacts) == 2  # child artifacts survive the join
    assert {value["v"] for (_ct, value) in h.artifacts.values()} == {1, 2}


def test_gather_identical_child_artifacts_dedup():
    def branch():
        def thunk():
            aid = yield from store_artifact({"v": 7}, "application/json")
            return aid

        return thunk

    def parent():
        return (yield from gather([branch(), branch()]))

    h = RecordingHandler()
    ids = h.run(parent)
    assert isinstance(ids, list)
    assert ids[0] == ids[1]  # same content across branches -> one id
    assert len(h.artifacts) == 1  # dedups; the merge does not clobber


# --- artifact_id helper -------------------------------------------------------


def test_artifact_id_is_op_key_without_prefix():
    op = StoreArtifact(value={"x": 1}, content_type="application/json")
    assert op_key(op).stored() == f"artifact:{artifact_id(op)}"


# --- durable (SQLite engine, infra-free): persistence + no shadowing ----------


class _NoDomain:
    def run(self, op: DomainOp):
        raise AssertionError("no domain op expected in this workflow")


@pytest.fixture
def app():
    """A fresh in-memory durable engine, closed on teardown (no leaked sqlite3 connection)."""
    a = SqliteApp(":memory:")
    yield a
    a.close()


def test_durable_store_artifact_content_addressed_and_persisted(app):
    """On the durable path (DurableHandler over the SQLite engine): two same-typed
    distinct-value stores resolve to distinct ids AND persist both values — the
    old path returned a constant id and discarded the value."""

    def wf():
        a = yield from store_artifact({"doc": "A"}, "message/rfc822")
        b = yield from store_artifact({"doc": "B"}, "message/rfc822")
        return {"a": a, "b": b}

    @app.register_task("art")
    def task(params, ctx):
        return DurableHandler(ctx, _NoDomain()).run(wf)

    snap = app.run_until_result(app.spawn("art", {"run_id": "r"}))
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result["a"] != snap.result["b"]  # distinct content -> distinct ids

    # Both values are durably persisted, under two distinct artifact checkpoints
    # (no stale-read shadowing) — the discard-value + collision fix on the durable path.
    art = app.conn.execute(
        "SELECT name, state FROM checkpoints WHERE name LIKE 'artifact:%' ORDER BY name"
    ).fetchall()
    assert len(art) == 2, [n for n, _ in art]
    values = [json.loads(state) for _, state in art]
    assert {"doc": "A"} in values
    assert {"doc": "B"} in values


def test_durable_gather_branch_artifacts_persist_without_collision(app):
    """Artifacts stored inside gather branches persist durably under distinct
    ``gather:{i}:`` prefixed checkpoints — no drop, no cross-branch id collision."""

    def branch(v):
        def thunk():
            return (yield from store_artifact({"v": v}, "application/json"))

        return thunk

    def wf():
        return (yield from gather([branch(1), branch(2)]))

    @app.register_task("g")
    def task(params, ctx):
        return DurableHandler(ctx, _NoDomain()).run(wf)

    snap = app.run_until_result(app.spawn("g", {"run_id": "r"}))
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result[0] != snap.result[1]  # distinct branch values -> distinct ids
    art = app.conn.execute(
        "SELECT name FROM checkpoints WHERE name LIKE '%artifact%' ORDER BY name"
    ).fetchall()
    # one prefixed artifact checkpoint per branch, both present (no drop, no collision)
    assert len(art) == 2, [n for (n,) in art]
    assert all(name.startswith("gather:") for (name,) in art)
