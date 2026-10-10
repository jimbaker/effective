"""`ForkLedger`: a fork child's LEDGER IDENTITY.

A fork's divergent tail row authors the SAME event id the base did (`reviewed:{message_id}`
embeds no run id), so unscoped it collides with the base's canonical row on the ledger's global
`UNIQUE(event_id)` and is silently dropped by `ON CONFLICT DO NOTHING`, on both engines.
`ForkLedger` gives the child its own identity by rescoping every event id to
`hyp:{child_run_id};{event_id}`; `effective.lineage.marginal` re-aligns by stripping the same scope
token. These pins cover the rescope, the merge-gate guard, the single-source-of-truth
scheme, and that the write-scheme and the strip-token compose into a correct marginal.

Infra-free: an in-memory `SqliteApp` stands up the real ledger table (UNIQUE + append-only
triggers), so the collision is exercised against the genuine constraint, not a mock.
"""

import json

import pytest
from pydantic_core import to_jsonable_python

from effective.counterfactual import (
    FORK_SCOPE,
    ForkLedger,
    fork_scoped,
    fork_sealed_name,
    fork_unscoped,
)
from effective.engines.sqlite import SqliteLedger
from effective.keys import Key, Segment, compose_key
from effective.keys.grammar import KeySyntaxError
from effective.lineage import marginal
from effective.ops import LedgerRow


def _reviewed(event_id: str, decision: str) -> LedgerRow:
    return LedgerRow(event_id=Key.parse(event_id), kind="reviewed", decision=decision)


# --- the scheme is one source of truth (write ⇄ strip) -------------------------------------


def test_fork_scoped_and_its_inverse_are_derived_from_one_tag():
    """`fork_scoped` writes the scope; `fork_unscoped` removes it. If they drifted, alignment
    would fail silently — so they must be exact inverses, derived from the same `FORK_SCOPE`."""
    cid, eid = Segment("r-fork-0"), "reviewed:m1"
    scoped = fork_scoped(cid, Key.parse(eid))
    # The TAG and the wrapped id, read off the structure — not a hand-spelled join. Spelling it
    # was what let the join drift: the assertion said `{FORK_SCOPE}:{cid}:{eid}` while the writer
    # had moved to `;`, so a test named for the derivation was restating a form instead.
    assert [term.tag for term in scoped.terms()] == [FORK_SCOPE, "reviewed"]
    assert fork_unscoped(scoped.stored(), cid) == eid

    # The inverse matches the FRAME, so it is exact where a prefix strip was approximate: a
    # different child's scope, and a scope that is not leading, both come back untouched.
    assert fork_unscoped(scoped.stored(), "another-child") == scoped.stored()
    assert fork_unscoped(eid, cid) == eid


# --- the collision fix, against the real UNIQUE constraint ---------------------------------


def test_a_forks_divergent_row_survives_the_unique_constraint_instead_of_being_dropped(sqlite_app):
    """Base commits `reviewed:m1` (approve); the fork re-runs and authors `reviewed:m1`
    (reject). Through a bare writer the fork's row would collide and be dropped, leaving only the
    base's approve. Through `ForkLedger` the fork's row is rescoped by `fork_scoped`: a distinct
    row that survives, and the base's canonical row is untouched.

    It ASKS `fork_scoped` for the rescoped id rather than spelling it, and the hazard is
    specific: a rewrite that turns BOTH separators into `;` still PARSES — `hyp(r-fork)` then a
    content-free `reviewed` qualifier then a content-free `m1`, three terms where the writer
    mints two. A legal key that means something else is what a careful rewrite cannot see and
    what asking the writer makes
    unwritable."""
    app = sqlite_app(":memory:")
    base = SqliteLedger(app.conn, "r-base", app.write_lock)
    fork = ForkLedger(
        SqliteLedger(app.conn, "r-fork", app.write_lock, hypothetical=True),
        child_run_id=Segment("r-fork"),
    )

    base.append(_reviewed("reviewed:m1", "approve"))
    fork.append(_reviewed("reviewed:m1", "reject"))  # same authored id as the base

    rows = app.conn.execute(
        "SELECT event_id, hypothetical, payload FROM ledger ORDER BY event_id"
    ).fetchall()
    by_id = {eid: (hyp, json.loads(payload)) for eid, hyp, payload in rows}

    rescoped = fork_scoped(Segment("r-fork"), Key.parse("reviewed:m1")).stored()
    assert set(by_id) == {"reviewed:m1", rescoped}  # both survive, distinct
    assert by_id["reviewed:m1"] == (
        0,
        to_jsonable_python(_reviewed("reviewed:m1", "approve")),
    )  # base untouched
    assert by_id[rescoped][0] == 1  # the fork row is hypothetical
    assert by_id[rescoped][1]["decision"] == "reject"  # and carries the delta


def test_fork_ledger_refuses_a_non_hypothetical_base(sqlite_app):
    """The merge gate: a fork must never write a canonical row. Rescoping a canonical writer's ids
    would be exactly that, so `ForkLedger` refuses one loudly at construction."""
    app = sqlite_app(":memory:")
    canonical = SqliteLedger(app.conn, "r-fork", app.write_lock)  # hypothetical defaults False
    with pytest.raises(ValueError, match="hypothetical"):
        ForkLedger(canonical, child_run_id=Segment("r-fork"))


def test_fork_ledger_leaves_the_callers_row_untouched(sqlite_app):
    """`append` rescopes into a copy — the caller's row (also the payload source) is untouched."""
    app = sqlite_app(":memory:")
    fork = ForkLedger(
        SqliteLedger(app.conn, "r-fork", app.write_lock, hypothetical=True),
        child_run_id=Segment("r-fork"),
    )
    row = _reviewed("reviewed:m1", "reject")
    fork.append(row)
    assert row.event_id.stored() == "reviewed:m1"  # unmutated


# --- the write-scheme and the strip-token compose into a correct marginal -------------------


def test_marginal_realigns_a_fork_tail_once_the_scope_is_removed():
    """Base approves and commits; the fork rejects and ledgers a rejection.
    The lineages share their prefix by IDENTITY once the scope is removed, and the marginal
    is `committed` (base) vs `rejected` (fork). This proves `fork_scoped` (write) and
    `fork_unscoped` (remove) are one working scheme."""
    cid = Segment("r-fork")
    # STORED rows (what `marginal` reads back off a ledger), so the reviewed row is dumped to its
    # wire form rather than left as the `LedgerRow` a WRITER would hand `append`.
    base = [
        {"event_id": "processed:m1", "kind": "processed"},
        to_jsonable_python(_reviewed("reviewed:m1", "approve")),
        {"event_id": "committed:m1", "kind": "committed", "amount": "42.00"},
    ]
    # the fork's rows as ForkLedger wrote them (rescoped), diverging at the consequence
    fork = [
        {"event_id": fork_scoped(cid, Key.parse("processed:m1")).stored(), "kind": "processed"},
        {
            "event_id": fork_scoped(cid, Key.parse("reviewed:m1")).stored(),
            "kind": "reviewed",
            "decision": "reject",
        },
        {"event_id": fork_scoped(cid, Key.parse("rejected:m1")).stored(), "kind": "rejected"},
    ]

    unscoped = [{**row, "event_id": fork_unscoped(row["event_id"], cid)} for row in fork]
    m = marginal(base, unscoped)
    assert m.shared_prefix == 2  # processed + reviewed align by identity
    assert [r["kind"] for r in m.base_tail] == ["committed"]
    assert [r["kind"] for r in m.fork_tail] == ["rejected"]


def test_without_the_scope_token_the_fork_diverges_at_its_first_row():
    """The token is load-bearing: without it the rescoped ids match nothing in the base, so the
    marginal diverges at index 0 — the failure the strip token exists to prevent."""
    cid = Segment("r-fork")
    base = [{"event_id": "processed:m1", "kind": "processed"}]
    fork = [
        {"event_id": fork_scoped(cid, Key.parse("processed:m1")).stored(), "kind": "processed"}
    ]
    m = marginal(base, fork)  # no scrub
    assert m.shared_prefix == 0


# --- injectivity of the two lineage-scoping schemes ----------------------------------------
#
# Both schemes are `tag:{child_run_id}:{rest}`, injective ONLY while `child_run_id` is `:`-free.
# `child_run_id=f"{base}:{i}"` is the obvious naming for a parallel sweep, and unchecked it lets
# two sibling forks mint one ledger `event_id` (silently dropped, since appends are idempotent by
# id) or park on one queue-global event name (one answer, two forks). These pin the PRECONDITION,
# at every site that mints a name.


def test_a_colon_bearing_child_run_id_is_refused_at_every_minting_site():
    from typing import cast

    from effective.counterfactual import ForkLedger
    from effective.handlers.base import TaskContext
    from effective.handlers.durable import RenamedAwaitCtx

    # A construction-site TYPE carries this guarantee: `Segment` refuses a `:`-bearing lineage id
    # once, where the value is created.
    with pytest.raises(ValueError, match="delimiter"):
        Segment("sweep:0")

    class _Hyp:
        hypothetical = True

        # Accepts `writer=`: a double that takes MORE than the contract demands still satisfies
        # the narrower one. A LEAF stub, so ignoring the value is correct; a delegating wrapper
        # would have to forward it.
        def append(self, row, *, writer=None):
            pass

    with pytest.raises(ValueError, match="delimiter"):
        ForkLedger(_Hyp(), child_run_id="sweep:0")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="delimiter"):
        RenamedAwaitCtx(cast(TaskContext, object()), "sweep:0")


def test_the_refusal_names_a_usable_replacement():
    """An error message that names the culprit AND the fix (the DX doctrine's error rule)."""
    with pytest.raises(ValueError, match="delimiter") as caught:
        Segment("sweep:0")
    assert "'sweep-0'" in str(caught.value)


def test_distinct_pairs_give_distinct_names_and_the_old_counterexample_is_unwritable():
    """Distinct (child, id) pairs give distinct names, and the counterexample is unwritable.

    The counterexample is `("sweep-0", "reviewed:m1")` against `("sweep", "0-reviewed:m1")`:
    under a naive join the two flatten to the same bytes. Injectivity rests on the GRAMMAR:
    `0-reviewed` is not a well-formed tag, because a tag is lower-kebab and cannot open with a
    digit, so the collision is refused at construction."""
    from effective.counterfactual import fork_scoped

    pairs = [("sweep-0", "reviewed:m1"), ("sweep", "committed:m1"), ("sweep-0", "committed:m1")]
    assert len({fork_scoped(Segment(c), Key.parse(e)) for c, e in pairs}) == len(pairs)

    with pytest.raises(KeySyntaxError, match="not a well-formed tag"):
        fork_scoped(Segment("sweep"), Key.parse("0-reviewed:m1")).stored()


def test_a_fork_ledger_cannot_wrap_another_fork_ledger():
    """Double-rescoping would give `hyp:{outer};hyp:{inner};{id}`, and a fork's marginal is
    defined against ONE child. Refused, and the message names nesting as the culprit: the base
    IS hypothetical, so "construct it with hypothetical=True" would mislead."""
    from effective.counterfactual import ForkLedger

    class _Hyp:
        hypothetical = True

        # Accepts `writer=`: a double that takes MORE than the contract demands still satisfies
        # the narrower one. A LEAF stub, so ignoring the value is correct; a delegating wrapper
        # would have to forward it.
        def append(self, row, *, writer=None):
            pass

    inner = ForkLedger(_Hyp(), child_run_id=Segment("cf-a"))
    with pytest.raises(ValueError, match="cannot wrap another ForkLedger"):
        ForkLedger(inner, child_run_id=Segment("cf-b"))


# --- the terminal seal ----------------------------------------------------------------------
#
# A fork's genesis is appended BEFORE the handler runs and its boundary checks run AFTER it
# completes, so a fork declared corrupt has already committed its whole tail, byte-identical in
# shape to a valid one. Without a seal the refusal is loud to the caller and SILENT in the
# ledger, which is the bookkeeper `effective.checkpoints` reads. A positive attestation on the
# clean path closes that, and the polarity is forced by the crash lens: a worker dying mid-fork
# cannot write "I failed", so absence must mean "not known valid".


def _seal_scenario(
    tmp_path, sqlite_app, *, seed_extra: dict | None = None, fault: str | None = None
):
    """Run a fork to completion (or to refusal / crash) and return its ledger kinds."""

    from effective.api import append_ledger, ask_llm, await_event
    from effective.checkpoints import read_sqlite_task
    from effective.cost import Usage
    from effective.fork import fork_seed, run_fork
    from effective.handlers.durable import DurableHandler

    mid = "m1"

    def wf(_rid):
        yield from ask_llm("extract", [], dict)
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"extracted:{Segment(mid)}"), kind="extracted")
        )
        approval = yield from await_event(f"review:{mid}", dict)
        yield from append_ledger(
            LedgerRow(
                event_id=compose_key(t"reviewed:{Segment(mid)}"),
                kind="reviewed",
                decision=approval["decision"],
            )
        )
        return approval["decision"]

    class _Dom:
        def run(self, op):
            return {"amount": "5.00"}

        def run_metered(self, op):
            return self.run(op), Usage()

    app = sqlite_app(str(tmp_path / "seal.db"))

    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, _Dom(), ledger=led).run(lambda: wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(f"review:{mid}", {"decision": "reject"})
    app.run_until_result(base_id)
    seed = dict(
        fork_seed(
            read_sqlite_task(str(tmp_path / "seal.db"), base_id), through=f"ledger;extracted:{mid}"
        )
    )
    seed.update(seed_extra or {})

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_Dom(),
            forked_from="r-base",
            forked_at_event=f"extracted:{mid}",
            fork_point=compose_key(t"review:{Segment(mid)}"),
            delta={"decision": "approve"},
        )

    fid = app.spawn("fork", {"run_id": "r-fork"}, max_attempts=4)
    app.run_until_result(fid)
    app.emit_event(f"fork:r-fork;review:{mid}", {"decision": "approve"})
    snap = app.run_until_result(fid)
    kinds = [
        k
        for (k,) in app.conn.execute(
            "SELECT kind FROM ledger WHERE workflow_run_id='r-fork' ORDER BY seq"
        )
    ]
    return snap, kinds


def _lineage_rows(tmp_path, run_id: str) -> list[dict]:
    """One lineage's ledger rows, as the mappings `fork_marginal`/`marginal` consume."""
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "seal.db"))
    try:
        return [
            {"event_id": event_id, "kind": kind}
            for event_id, kind in conn.execute(
                "SELECT event_id, kind FROM ledger WHERE workflow_run_id=? ORDER BY seq", (run_id,)
            )
        ]
    finally:
        conn.close()


def test_a_valid_fork_is_SEALED_in_the_ledger(tmp_path, sqlite_app):
    snap, kinds = _seal_scenario(tmp_path, sqlite_app)
    assert snap is not None
    assert snap.state == "completed"
    assert kinds[0] == "forked"  # the genesis is still first
    assert kinds[-1] == "fork_sealed"  # and the seal is last
    assert "reviewed" in kinds


def test_a_REFUSED_fork_is_not_sealed_so_the_ledger_alone_distinguishes_it(tmp_path, sqlite_app):
    """The seal's property: a VOI consumer reading only the ledger can tell. No crossing to the
    task row, since canonical truth is never derived from it."""
    snap, kinds = _seal_scenario(tmp_path, sqlite_app, seed_extra={"bogus:never": 1})
    assert snap is not None
    assert snap.state == "failed"
    assert "fork_sealed" not in kinds, (
        f"a refused fork was sealed — the ledger cannot distinguish it: {kinds}"
    )


def test_the_seal_is_idempotent_under_replay(tmp_path, sqlite_app):
    """Deterministic id, so a crash-replay of the tail cannot write a second seal — the same
    property the genesis already has."""
    from effective.counterfactual import sealed_row

    assert sealed_row("cf-a", forked_from="r-base").event_id == fork_sealed_name("cf-a")
    snap, kinds = _seal_scenario(tmp_path, sqlite_app)
    assert snap is not None
    assert kinds.count("fork_sealed") == 1


# --- the marginal, by ADDRESS ---------------------------------------------------------------
#
# `marginal` is the DISCOVERY form and cannot work on a real `run_fork` lineage: the genesis sits
# at index 0 with no base counterpart, and the shared prefix's rows are SEEDED so they never
# appear in the fork lineage at all. On such a lineage it gives `shared_prefix == 0`, the whole
# base reading as divergent, and a hand-built fixture can pin a shape `run_fork` cannot emit.
# `fork_marginal` slices at the KNOWN boundary instead.


def test_fork_marginal_works_on_a_real_run_fork_lineage(tmp_path, sqlite_app):
    """The deliverable, against output `run_fork` actually produced — not a fixture."""
    from effective.lineage import fork_marginal

    snap, _kinds = _seal_scenario(tmp_path, sqlite_app)
    assert snap is not None

    base, fork = _lineage_rows(tmp_path, "r-base"), _lineage_rows(tmp_path, "r-fork")
    result = fork_marginal(base, fork, child_run_id="r-fork", forked_at_event="extracted:m1")
    assert result.shared_prefix == 1, f"the seeded prefix was not recognised: {result}"
    # the divergence surfaces at the CONSEQUENCE: base reviewed=reject and stopped; fork
    # reviewed=approve. Same event id on both sides, different content — alignment by identity.
    assert [r["kind"] for r in result.base_tail] == ["reviewed"]
    assert [r["kind"] for r in result.fork_tail] == ["reviewed"]
    # the fork's bookkeeping rows are not part of the marginal
    assert not any(r["kind"] in ("forked", "fork_sealed") for r in result.fork_tail)


def test_fork_marginal_refuses_a_provenance_address_the_base_does_not_contain(
    tmp_path, sqlite_app
):
    """A genesis naming an address absent from the base is a provenance error, not something to
    silently align at 0 — which is precisely how the discovery form failed."""
    from effective.lineage import fork_marginal

    snap, _ = _seal_scenario(tmp_path, sqlite_app)
    assert snap is not None
    base, fork = _lineage_rows(tmp_path, "r-base"), _lineage_rows(tmp_path, "r-fork")
    with pytest.raises(ValueError, match="not in the base lineage"):
        fork_marginal(base, fork, child_run_id="r-fork", forked_at_event="no-such:event")
