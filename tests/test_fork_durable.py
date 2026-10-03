"""The durable spawned fork end-to-end: the decision fork on the real SQLite engine.

`run_fork` composes the fork pieces into a crash-surviving counterfactual: re-run a base
workflow over a seeded prefix, substitute a different decision at the fork point, and commit the
divergent tail to a hypothetical lineage the canonical view never sees. This proves it on a genuine
durable SQLite engine (file-backed, so `read_sqlite_task` reads the base's checkpoints), with a
substrate-pure decision workflow (a prefix step + prefix ledger + review await + tail-ledger
decision branch). `test_fork_durable_absurd.py` is the Absurd half.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import BaseModel

from effective.api import append_ledger, ask_llm, await_event, call_tool, gather, sleep_until
from effective.checkpoints import read_sqlite_task
from effective.counterfactual import fork_scoped
from effective.domain import SPAWN_TOOL, SpawnResult
from effective.fork import (
    fork_seed,
    join_fork,
    run_fork,
    spawn_fork,
)
from effective.handlers.absurd import (
    DurableHandler,
    RenamedAwaitCtx,
    SeedingCtx,
    _supports_peek,
    fork_event_name,
    spawn_done_name,
)
from effective.handlers.base import step_key
from effective.keys import Key, Run, Segment, compose_key
from effective.ops import Addressing, LedgerRow, Writer
from effective.sandbox import WorldMutation
from effective.sqlite import SqliteApp, SqliteLedger

# A ctx that is constructed and never touched (the refusal under test precedes every use of it),
# so its task id is a placeholder. The id is a `UUID`, so the name says what it is for.
_UNUSED_TASK = UUID("019fa000-0000-7000-8000-0000000000ff")

MID = "m1"

# **Composed ONCE, then referenced.** The file states each identity exactly once and the
# COMPOSER decides how it renders, so a separator change has one place to reach.
EXTRACTED = compose_key(t"extracted:{Segment(MID)}")
REVIEW = compose_key(t"review:{Segment(MID)}")
REVIEWED = compose_key(t"reviewed:{Segment(MID)}")
COMMITTED = compose_key(t"committed:{Segment(MID)}")
# lint: terminal-hole — a `Key` splice needs no `Segment` wrapper. The marker is per LINE,
# not per block, which is why it is repeated.
LEDGER_EXTRACTED = compose_key(t"ledger;{EXTRACTED:domain=address}")
# lint: terminal-hole
LEDGER_REVIEWED = compose_key(t"ledger;{REVIEWED:domain=address}")
# lint: terminal-hole
LEDGER_COMMITTED = compose_key(t"ledger;{COMMITTED:domain=address}")
KA = compose_key(t"ka:{Segment(MID)}")
KB = compose_key(t"kb:{Segment(MID)}")


def _forked(child: str, name: Key) -> str:
    """The name a fork CHILD awaits — the parent's identity re-scoped to the child run.

    A splice, not a concatenation: `name` arrives as a `Key`, so a child id carrying a delimiter
    cannot forge a frame, and the fork arm's shape lives in one template instead of twelve."""
    # lint: terminal-hole — `name` is a `Key`, which is why it can be spliced at all.
    return compose_key(t"fork:{Segment(child)};{name:domain=address}").stored()


def _decision_wf(_rid: str):
    """A minimal decision workflow: an extract Step, an `extracted` ledger row, a review await (the
    fork point), then a `reviewed` row and a branch — reject returns, approve commits."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    yield from append_ledger(
        LedgerRow(
            event_id=REVIEWED,
            kind="reviewed",
            decision=approval["decision"],
        )
    )
    if approval["decision"] == "reject":
        return "rejected"
    yield from append_ledger(LedgerRow(event_id=COMMITTED, kind="committed"))
    return "committed"


def _await_free_wf(_rid: str):
    """The same shape minus the await: the one workflow family for which a wrong `fork_point`
    produces no earlier symptom than the completion phase-match (pinned below)."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    return "done"


class _Dom:
    """A trivial domain answering the one `extract` Step. In the fork it is never called (extract
    is seeded, the tail is ledger-only), so `DryRun` around it has nothing to refuse."""

    def run(self, op) -> dict:
        return {"amount": "5.00"}

    def run_metered(self, op) -> tuple[dict, object]:
        from effective.cost import Usage

        return self.run(op), Usage()


def _rows(app: SqliteApp, run_id: str) -> list[tuple[str, str, int]]:
    """(event_id, kind, hypothetical) for a lineage, in commit order."""
    return list(
        app.conn.execute(
            "SELECT event_id, kind, hypothetical FROM ledger WHERE workflow_run_id=? ORDER BY seq",
            (run_id,),
        )
    )


def test_run_fork_substitutes_the_decision_and_diverges_on_the_hypothetical_ledger(
    tmp_path, sqlite_app
):
    db = tmp_path / "fork.db"
    app = sqlite_app(str(db))
    dom = _Dom()

    # --- BASE run: review -> reject (canonical) --------------------------------------------
    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, dom, ledger=led).run(lambda: _decision_wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)  # parks at review:m1
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    base_snap = app.run_until_result(base_id)
    assert base_snap is not None
    assert base_snap.result == "rejected"
    base_before = _rows(app, "r-base")
    assert [(e, k) for e, k, _ in base_before] == [
        (EXTRACTED.stored(), "extracted"),
        (REVIEWED.stored(), "reviewed"),
    ]  # rejected: no commit
    assert all(hyp == 0 for _, _, hyp in base_before)  # base is canonical

    # --- FORK: substitute approve, run the tail hypothetically ------------------------------
    base_ckpts = read_sqlite_task(str(db), base_id)
    seed = fork_seed(base_ckpts, through=LEDGER_EXTRACTED.stored())  # prefix only

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _decision_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=dom,
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
        )

    fork_id = app.spawn("fork", {"run_id": "r-fork"})
    app.run_until_result(fork_id)  # parks at the CHILD's renamed event
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})  # the delta
    fork_snap = app.run_until_result(fork_id)
    assert fork_snap is not None
    assert fork_snap.result == "committed"  # the counterfactual: approve -> commit

    # the base lineage is UNTOUCHED (poison-provable: the fork never wrote a canonical row)
    assert _rows(app, "r-base") == base_before

    # the fork lineage: genesis + the DIVERGENT TAIL ONLY (the shared prefix `extracted` is seeded,
    # so its append-thunk is replaced by SeedingCtx and never re-copied: a fork stores the
    # divergence, not a copy of history). ALL hypothetical; ids lineage-scoped so the fork's
    # `reviewed:m1` cannot collide with the base's canonical one.
    # Bookended: the `forked` genesis first (provenance), the `fork_sealed` attestation last
    # ("every boundary check passed", so `sealed ⇒ valid marginal` from the ledger alone).
    fork_rows = _rows(app, "r-fork")
    assert [k for _, k, _ in fork_rows] == ["forked", "reviewed", "committed", "fork_sealed"]
    assert [k for _, k, _ in fork_rows].count("forked") == 1  # exactly one genesis
    assert [k for _, k, _ in fork_rows].count("fork_sealed") == 1  # and exactly one seal
    assert all(hyp == 1 for _, _, hyp in fork_rows)  # fenced off from the canonical view
    assert all(e.startswith("hyp:r-fork;") for e, _, _ in fork_rows)  # lineage-scoped ids

    # the divergence proper: base reviewed->reject and committed NOTHING; fork reviewed->approve
    # and COMMITTED — the marginal (approve -> commits; reject -> a rejection, nothing posted).
    assert json.loads(_payload(app, REVIEWED.stored()))["decision"] == "reject"  # base
    assert json.loads(_payload(app, _rescoped(REVIEWED.stored())))["decision"] == "approve"  # fork
    base_kinds = [k for _, k, _ in base_before]
    fork_kinds = [k for _, k, _ in fork_rows]
    assert "committed" not in base_kinds  # the base never committed
    assert "committed" in fork_kinds  # the fork would have


def _rescoped(event_id: str) -> str:
    """A fork child's lineage-scoped ledger id, asked of the writer rather than spelled.

    A hand spelling such as `hyp:r-fork;reviewed;{MID}` still parses (three terms where
    `fork_scoped` mints two), so a wrong one survives both a careful read and a sweep over plain
    string literals."""
    from effective.counterfactual import fork_scoped
    from effective.keys import Key, Segment

    return fork_scoped(Segment("r-fork"), Key.parse(event_id)).stored()


def _payload(app: SqliteApp, event_id: str) -> str:
    return app.conn.execute("SELECT payload FROM ledger WHERE event_id=?", (event_id,)).fetchone()[
        0
    ]


class _RecFault:
    """One-shot fault STATE that persists across a task's retries (like `_conformance.Fault`), plus
    a record of the op it fired at, so the crash test can assert de-aliasing."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.armed = True
        self.fired: list[str] = []


class _RecFaultCtx:
    """A per-attempt ctx wrapper driving a persistent `_RecFault`. Name-targeted faults ALIAS when
    the run id embeds the target and the renamed await embeds that run id: `reviewed`/`committed`
    then crash the AWAIT instead of the tail append. Recording the fired-at op and asserting it
    makes de-aliasing regression-proof."""

    def __init__(self, ctx, fault: _RecFault) -> None:
        self._ctx = ctx
        self._f = fault

    def _trip(self, name: Key | str) -> None:
        # `Key | str` for `_conformance.FaultCtx`'s reason: `step` carries a `Key` while
        # `await_event`/`sleep` carry a `str`. The target is a SUBSTRING probe, so the identity
        # is unwrapped once here rather than at each call site.
        text = name if isinstance(name, str) else name.stored()
        if self._f.armed and self._f.target in text:
            self._f.armed = False
            self._f.fired.append(text)
            from _conformance import FaultInjected

            raise FaultInjected(f"crash at {text}")

    def step(self, name, thunk, /):
        self._trip(name)
        return self._ctx.step(name, thunk)

    def await_event(self, name, /):
        self._trip(name)
        return self._ctx.await_event(name)

    def sleep_until(self, when, /, *, name: Key | None = None):
        self._trip("sleep")
        return self._ctx.sleep_until(when, name=name)

    def __getattr__(self, attr):
        return getattr(self._ctx, attr)


def test_the_fork_child_survives_crash_at_every_op(tmp_path, sqlite_app):
    """The headline durability gate: a crash at EACH distinct ctx op of the fork child resumes by
    replay to the IDENTICAL hypothetical outcome (result `committed`, lineage [forked, reviewed
    (approve), committed]), and the base lineage stays untouched. Exact op-key targets, run ids
    disjoint from every target string, and a fired-at assertion together make the tail-append
    crash points exercised in their own right instead of aliased to the await."""
    db = tmp_path / "crash.db"
    app = sqlite_app(str(db))
    dom = _Dom()

    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, dom, ledger=led).run(lambda: _decision_wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    app.run_until_result(base_id)
    base_before = _rows(app, "r-base")
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    # the five distinct ctx ops, by EXACT match key. `review:m1` targets the renamed await
    # `fork:{rid}:review:m1` and is NOT a substring of `ledger;reviewed:m1`; the run id `cfN`
    # shares no substring with any target, so nothing aliases.
    targets = [
        "extract",
        LEDGER_EXTRACTED.stored(),
        REVIEW.stored(),
        LEDGER_REVIEWED.stored(),
        LEDGER_COMMITTED.stored(),
    ]
    for i, target in enumerate(targets):
        rid = f"cf{i}"
        fault = _RecFault(target)  # persists across the task's retries (one-shot)

        @app.register_task(f"crash-{i}")
        def fork_task(params, ctx, _fault=fault):
            hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
            return run_fork(
                _RecFaultCtx(ctx, _fault),
                lambda: _decision_wf(params["run_id"]),
                child_run_id=params["run_id"],
                seed=seed,
                hypothetical_ledger=hyp,
                domain=dom,
                forked_from="r-base",
                forked_at_event=EXTRACTED.stored(),
                fork_point=REVIEW,
                delta={"decision": "approve"},
            )

        fid = app.spawn(f"crash-{i}", {"run_id": rid}, max_attempts=8)
        app.run_until_result(fid)  # crash+retry -> park (prefix crash), or park then crash (tail)
        app.emit_event(
            fork_event_name(rid, Key.parse(REVIEW.stored())).stored(), {"decision": "approve"}
        )
        snap = app.run_until_result(fid)
        assert fault.fired, f"target={target}: the fault never fired"
        assert target in fault.fired[0], f"target={target}: fired at {fault.fired[0]!r} (aliased!)"
        assert snap is not None
        assert snap.result == "committed", f"target={target}: {snap.result}"
        kinds = [k for _, k, _ in _rows(app, rid)]
        # `fork_sealed` is the terminal attestation: present here on every crash target,
        # because each of these forks completed with all boundary checks passing. Its presence IS
        # the "this marginal is valid" claim, and it is idempotent, so N crashes give exactly one.
        assert kinds == ["forked", "reviewed", "committed", "fork_sealed"], (
            f"target={target}: {kinds}"
        )
        decision = json.loads(
            _payload(app, fork_scoped(Segment(rid), Key.parse(REVIEWED.stored())).stored())
        )["decision"]
        assert decision == "approve", f"target={target}: reviewed={decision}"  # right VALUE, too

    # the canonical base lineage is untouched after five crashing forks
    assert _rows(app, "r-base") == base_before


def _run_base(app, db_path: str, wf, *, contract=None):
    """Run the base decision workflow to a canonical reject; return its task id."""
    from effective.cost import Contract

    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        h = DurableHandler(ctx, _Dom(), ledger=led, contract=contract or Contract.V0)
        return h.run(lambda: wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    app.run_until_result(base_id)
    return base_id


def test_forking_an_iterative_workflow_seeds_each_occurrence(tmp_path, sqlite_app):
    """A repeated step name (an agent loop) has `extract`, `extract#2` checkpoints; the fork must
    seed each occurrence with ITS OWN base value. The tail `reviewed` row carries both prefix
    values, so the fork's row shows (a, b) = (1, 2), matching the base, and never (1, 1)."""
    db = tmp_path / "dup.db"
    app = sqlite_app(str(db))

    class _Counting:
        def __init__(self):
            self.n = 0

        def run(self, op):
            self.n += 1
            return {"n": self.n}

        def run_metered(self, op):
            from effective.cost import Usage

            return self.run(op), Usage()

    def wf(_r):
        a = yield from ask_llm("extract", [], dict)
        b = yield from ask_llm("extract", [], dict)  # duplicate step name
        yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
        approval = yield from await_event(REVIEW.stored(), dict)
        yield from append_ledger(
            LedgerRow(
                event_id=REVIEWED,
                kind="reviewed",
                a=a["n"],
                b=b["n"],
            )
        )
        return "reject" if approval["decision"] == "reject" else "done"

    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, _Counting(), ledger=led).run(lambda: wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    app.run_until_result(base_id)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())
    # the seed map is keyed by IDENTITY — a `str` lookup would simply miss, silently
    # Ask the minter: the arm landed at step 8, so a bare author name keys as `step:extract`.
    assert step_key("extract").occurrence(2) in seed  # the base recorded a distinct 2nd

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            _RecFaultCtx(ctx, _RecFault("__never__")),  # no crash; just drive the fork
            lambda: wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_Counting(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
        )

    fid = app.spawn("fork", {"run_id": "r-fork"})
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    app.run_until_result(fid)
    reviewed = json.loads(_payload(app, _rescoped(REVIEWED.stored())))
    assert (reviewed["a"], reviewed["b"]) == (1, 2)  # each occurrence its own value, not (1, 1)


def _forked_snap(app, seed, *, contract=None, fork_point=None):
    from effective.cost import Contract

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _decision_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_Dom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=fork_point or REVIEW,
            delta={"decision": "approve"},
            contract=contract or Contract.V0,
        )

    fid = app.spawn("fork", {"run_id": "r-fork"}, max_attempts=2)
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    return app.run_until_result(fid)


def test_a_tail_through_boundary_fails_loudly(tmp_path, sqlite_app):
    """A `through` naming a TAIL op → SeedingCtx's seal catches the seeded tail step (a divergent
    row would otherwise be silently dropped). The fork fails rather than committing a corrupt
    lineage."""
    db = tmp_path / "tail.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    base_ckpts = read_sqlite_task(str(db), base_id)
    seed = fork_seed(base_ckpts, through=LEDGER_REVIEWED.stored())  # a TAIL op!
    snap = _forked_snap(app, seed)
    assert snap is not None
    assert snap.state == "failed"
    assert "SeedBoundary" in str(snap.failure)


def test_a_bogus_seed_key_fails_loudly(tmp_path, sqlite_app):
    """An extra seed key the child never yields → the wired `unconsumed()` check catches it on
    completion instead of accepting it silently."""
    db = tmp_path / "bogus.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    base_ckpts = read_sqlite_task(str(db), base_id)
    # `Key.parse` for the injected bogus key: the seed map is keyed by IDENTITY, so a `str`
    # would sit in the map unmatched by construction and `unconsumed()` would report it for
    # the wrong reason. Parsing keeps the fixture in the same world as the real keys.
    seed = {
        **fork_seed(base_ckpts, through=LEDGER_EXTRACTED.stored()),
        Key.parse("bogus:never"): "y",
    }
    snap = _forked_snap(app, seed)
    assert snap is not None
    assert snap.state == "failed"
    assert "unconsumed" in str(snap.failure)


def test_a_wrong_fork_point_fails_loudly(tmp_path, sqlite_app):
    """A `fork_point` that never fires leaves the phase in `Seeding`, so the `Live` tail-guard is
    inert: a tail `through` would then slip through as a silently-corrupt lineage. `run_fork`
    matches the phase on completion, so a completed fork still `Seeding` fails loudly rather than
    committing the corrupt lineage. `fork_point` cross-checks like `seed`/`through`.
    """
    db = tmp_path / "wrongfp.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    base_ckpts = read_sqlite_task(str(db), base_id)
    seed = fork_seed(base_ckpts, through=LEDGER_REVIEWED.stored())  # a TAIL through...
    snap = _forked_snap(
        app, seed, fork_point=Key.parse("never-fires")
    )  # ...that the wrong fp can't guard
    assert snap is not None
    assert snap.state == "failed"
    # The diagnosis fires at the first observable symptom. Three guards could catch this: the
    # completion check, the `Seeding` arm for a tail op judged as prefix, and the prefix-await
    # refusal. The refusal fires first, because this workflow's next op after the seeded prefix
    # is the `review:m1` await, which with the fork point misnamed is a prefix await. Its message
    # is the sharpest of the three: it names the await AND the fork_point the caller meant.
    assert "ForkedPrefixAwait" in str(snap.failure)
    assert f"fork_point='review:{MID}'" in str(snap.failure)


def test_forking_a_v1_metered_base_decodes_the_envelope(tmp_path, sqlite_app):
    """A v1 base records the raw {result, usage} envelope in the extract checkpoint. Forked under
    Contract.V1 (the passthrough), the child folds the envelope and the workflow sees the DECODED
    result instead of the envelope leaking into the hypothetical `amount`."""
    from effective.cost import Contract

    db = tmp_path / "v1.db"
    app = sqlite_app(str(db))

    def wf(_r):
        assessment = yield from ask_llm("extract", [], dict)
        yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
        approval = yield from await_event(REVIEW.stored(), dict)
        yield from append_ledger(
            LedgerRow(
                event_id=REVIEWED,
                kind="reviewed",
                amount=assessment["amount"],
            )
        )
        return "reject" if approval["decision"] == "reject" else "done"

    base_id = _run_base(app, str(db), wf, contract=Contract.V1)
    base_ckpts = read_sqlite_task(str(db), base_id)
    seed = fork_seed(base_ckpts, through=LEDGER_EXTRACTED.stored())

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
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            contract=Contract.V1,
        )

    fid = app.spawn("fork", {"run_id": "r-fork"})
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    app.run_until_result(fid)
    reviewed = json.loads(_payload(app, _rescoped(REVIEWED.stored())))
    assert reviewed["amount"] == "5.00"  # the decoded result, not the {result, usage} envelope


# --- a counterfactual consumes no real time, on the DURABLE path -------------------------
#
# A fork ctx wrapper that delegates `sleep_until` untouched parks a forked tail against the wall
# clock (state='sleeping' for 24h), whatever the in-process refusal says. These pin the CLASS (a
# counterfactual consumes no real time) on both sides of the phase boundary, which is the
# distinction that makes the rule correct.


def _sleeping_wf(rid: str, wake):
    """Approve leads to a durable sleep; reject returns immediately. So the BASE (which
    rejects) never sleeps and the FORK (which approves) reaches the sleep in its tail."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    if approval["decision"] == "approve":
        yield from sleep_until(wake)
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"slept:{Segment(MID)}"), kind="slept")
        )
    return approval["decision"]


def test_a_sleep_in_the_forked_tail_is_refused_not_slept(tmp_path, sqlite_app):
    """The Inception refusal, on the engine: a fork explores a decision, not a clock."""
    wake = datetime.now(UTC) + timedelta(days=1)
    db = tmp_path / "d7.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), lambda rid: _sleeping_wf(rid, wake))
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _sleeping_wf(params["run_id"], wake),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_Dom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
        )

    fid = app.spawn("fork", {"run_id": "r-fork"})
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fid)

    assert snap is not None
    assert snap.state == "failed", f"the fork did not refuse: {snap.state}"
    assert "ForkedSleep" in str(snap.failure)
    # the task must NOT be parked against the real clock — the whole point
    state, available_at = app.conn.execute(
        "SELECT state, available_at FROM tasks WHERE task_id=?", (fid,)
    ).fetchone()
    assert state != "sleeping", "the counterfactual parked against reality's clock"
    assert available_at <= time.time(), "a wake time in the future survived the refusal"
    # and it refused BEFORE committing the post-sleep row
    assert [k for _, k, _ in _rows(app, "r-fork")] == ["forked"]


def test_an_already_elapsed_sleep_in_the_replayed_PREFIX_passes_through(tmp_path, sqlite_app):
    """The other side of the phase boundary, and the reason the refusal is phase-gated rather
    than blanket: a prefix sleep HAPPENED in reality, so its wake time is already past and
    replaying it is a no-op. Refusing it would make any run that ever slept unforkable."""
    past = datetime.now(UTC) - timedelta(days=1)

    def wf(rid: str):
        yield from ask_llm("extract", [], dict)
        yield from sleep_until(past)  # in the PREFIX, and already elapsed
        yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
        approval = yield from await_event(REVIEW.stored(), dict)
        yield from append_ledger(
            LedgerRow(
                event_id=REVIEWED,
                kind="reviewed",
                decision=approval["decision"],
            )
        )
        return approval["decision"]

    db = tmp_path / "d7prefix.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

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
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
        )

    fid = app.spawn("fork", {"run_id": "r-fork"})
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fid)

    assert snap is not None, "the fork never completed"
    assert snap.result == "approve"
    assert [k for _, k, _ in _rows(app, "r-fork")] == ["forked", "reviewed", "fork_sealed"]


def test_supports_peek_unwraps_every_fork_ctx_wrapper():
    """`run_fork` stacks `SeedingCtx` OUTERMOST, so a probe that stops at it finds
    `RenamedAwaitCtx.peek_event` via `__getattr__` (always defined on the class) and falsely
    reports peek support over a base ctx that has none, replacing the legible await-in-gather
    `NotImplementedError` with an `AttributeError`. Pinned as the CLASS: the full fork stack
    reports exactly what the BASE ctx supports."""

    class _NoPeekCtx:
        def step(self, name, thunk, /):
            return thunk()

        def await_event(self, name, /):
            return None

        def sleep_until(self, when, /, *, name: Key | None = None):
            return None

    bare = _NoPeekCtx()
    stack = SeedingCtx(RenamedAwaitCtx(bare, "cf"), {}, fork_point=Key.parse("review"))
    assert _supports_peek(bare) is False
    assert _supports_peek(stack) is False, "the fork stack falsely reported peek support"


def test_a_wrong_fork_point_is_caught_by_the_completion_check_when_every_op_is_seeded(
    tmp_path, sqlite_app
):
    """The completion phase-match stays REACHABLE, and gives the sharper diagnosis when it is.

    The `Seeding` arm fires at the first *unseeded* op and the prefix-await refusal at the first
    prefix await, so between them they preempt the completion check for any workflow that awaits
    at all. What they cannot see is an AWAIT-FREE workflow with everything seeded: no unseeded op,
    no prefix await, `unconsumed()` empty. The only remaining evidence of a wrong `fork_point` is
    that the phase never crossed, which is exactly what the completion check reads; each of the
    three guards covers a case the other two miss."""
    db = tmp_path / "newtwo.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _await_free_wf)
    ckpts = read_sqlite_task(str(db), base_id)
    # `through` is a `str` (`fork_seed`'s callers all compose it as an f-string), so the
    # boundary key unwraps to its durable form here.
    seed = fork_seed(ckpts, through=ckpts[-1].key.stored())  # seed EVERYTHING

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _await_free_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_Dom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=Key.parse("never-fires"),
            delta={"decision": "reject"},  # nothing diverges: every op is seeded
        )

    fid = app.spawn("fork", {"run_id": "r-fork"}, max_attempts=2)
    app.run_until_result(fid)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "reject"})
    snap = app.run_until_result(fid)

    assert snap is not None
    assert snap.state == "failed"
    assert "without crossing fork_point" in str(snap.failure), snap.failure


# --- a prefix `budget-grant:` park would deadlock the fork forever, silently -----------------
#
# The one class of failure a durable substrate cannot shrug off: the child parks in its own event
# namespace on a grant nobody will emit, so there is no exception, no failure row, no attempt
# burned, nothing to read afterwards. "Prefix awaits are unsupported" speaks of WORKFLOW-authored
# awaits and so does not cover this one: the park is injected by the budget layer DURING the
# replay, because the seeded prefix re-folds each raw envelope's usage and the meter re-crosses
# the same ceiling at the same op.

BUDGET_LIMIT = 0.0015  # crossed after two asks at 0.001 each -> parks before ask2
GRANT_NAME = compose_key(t"budget-grant:{Segment('r-base')},{0}")


def _metered_domain():
    from effective.cost import MeteredInterpreter, Usage

    return MeteredInterpreter(
        llm=lambda _op: ("ans", Usage(prompt_tokens=10, completion_tokens=5, cost=0.001)),
        tools=lambda _op: "tool-done",
    )


def _budgeted_wf(_rid: str):
    """Three metered asks, a ledger row, then the review await — so the budget trips inside what
    the fork will replay as its PREFIX, exactly as a real measured run does."""
    from effective.api import step
    from effective.domain import AskLLM

    for i in range(3):
        yield from step(f"ask{i}", AskLLM(messages="m", response_schema=str))
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    yield from append_ledger(
        LedgerRow(
            event_id=REVIEWED,
            kind="reviewed",
            decision=approval["decision"],
        )
    )
    return approval["decision"]


def _budgeted_base(app, db: str) -> UUID:
    """A base that really parked for a grant mid-run, got it, then parked at the fork point."""
    from effective.budget import MeasuredBudget
    from effective.cost import Contract

    @app.register_task("base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        budget = MeasuredBudget(overall=BUDGET_LIMIT, run_id="r-base", on_exhaust="park")
        return DurableHandler(
            ctx, _metered_domain(), ledger=led, contract=Contract.V1, budget=budget
        ).run(lambda: _budgeted_wf(params["run_id"]))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    assert _waiting_on(app, base_id) == GRANT_NAME.stored()  # the mid-prefix park, for real
    app.emit_event(GRANT_NAME.stored(), {"add_dollars": 0.01})
    app.run_until_result(base_id)
    assert _waiting_on(app, base_id) == REVIEW.stored()  # now at the fork point
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    snap = app.run_until_result(base_id)
    assert snap is not None
    assert snap.result == "reject"
    return base_id


def _waiting_on(app, task_id: UUID) -> str | None:
    from effective.sql import bind

    return app.conn.execute(
        *bind(t"SELECT waiting_event FROM tasks WHERE task_id={task_id}")
    ).fetchone()[0]


def _budgeted_fork(app, seed, *, transplanted=frozenset()) -> UUID:
    from effective.budget import MeasuredBudget
    from effective.cost import Contract

    @app.register_task("fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        budget = MeasuredBudget(overall=BUDGET_LIMIT, run_id="r-base", on_exhaust="park")
        return run_fork(
            ctx,
            lambda: _budgeted_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_metered_domain(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            contract=Contract.V1,
            budget=budget,
            transplanted=transplanted,
        )

    return app.spawn("fork", {"run_id": "r-fork"}, max_attempts=1)


def test_a_prefix_budget_grant_park_is_refused_not_deadlocked(tmp_path, sqlite_app):
    """Unrefused, this fork reaches `waiting` on `fork:r-fork;budget-grant:r-base,0` with
    `attempt=0`, no failure and no error, and emitting the delta at the fork point changes
    nothing, because the child never gets that far. The pin asserts the three things that
    distinguish a refusal from a deadlock: it is terminal, it is NOT parked, and it says which
    await and what to do about it."""
    db = tmp_path / "s3.db"
    app = sqlite_app(str(db))
    base_id = _budgeted_base(app, str(db))
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    fork_id = _budgeted_fork(app, seed)
    snap = app.run_until_result(fork_id)

    assert snap is not None
    assert snap.state == "failed"  # terminal, not `waiting` forever
    assert _waiting_on(app, fork_id) is None  # and not parked on anything
    assert "ForkedPrefixAwait" in str(snap.failure)
    assert GRANT_NAME.stored() in str(snap.failure)  # the await that would have deadlocked
    assert _forked("r-fork", GRANT_NAME) in str(snap.failure)  # the name it would have parked on
    assert "transplanted" in str(snap.failure)  # the opt-out, named

    # The delta emit is provably irrelevant: the run is over. (Under a deadlock this emit tells
    # you nothing; the task simply stays `waiting`.)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    after = app.run_until_result(fork_id)
    assert after is not None
    assert after.state == "failed"

    # The canonical bookkeeper agrees: a genesis, and no seal. `sealed => valid marginal`
    # holds: a refused fork leaves an unsealed lineage, so no consumer reads its marginal.
    kinds = [k for _, k, _ in _rows(app, "r-fork")]
    assert kinds == ["forked"]


def test_a_transplanted_prefix_grant_lets_the_fork_through(tmp_path, sqlite_app):
    """The opt-out is a real seam, not a costume: deliver the base's grant under the CHILD's name,
    name it in `transplanted`, and the same fork runs to completion — replaying the prefix,
    substituting the decision, and sealing. This is the shape the unbuilt prefix-await transplant
    will automate (a reader that re-emits the base's recorded answers); pinning it now means the
    refusal above is bounded by something that works, and that the budget layer resumes correctly
    from a transplanted grant."""
    db = tmp_path / "s3ok.db"
    app = sqlite_app(str(db))
    base_id = _budgeted_base(app, str(db))
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    fork_id = _budgeted_fork(app, seed, transplanted=frozenset({GRANT_NAME}))
    # the transplant: the base's answer, re-emitted under the child's renamed event
    app.emit_event(_forked("r-fork", GRANT_NAME), {"add_dollars": 0.01})

    app.run_until_result(fork_id)
    assert _waiting_on(app, fork_id) == _forked("r-fork", REVIEW)  # reached the fork point
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fork_id)

    assert snap is not None
    assert snap.result == "approve"  # the counterfactual ran
    kinds = [k for _, k, _ in _rows(app, "r-fork")]
    assert kinds == ["forked", "reviewed", "fork_sealed"]  # divergent tail only, and SEALED
    assert all(hyp == 1 for _, _, hyp in _rows(app, "r-fork"))  # fenced off from canonical


def test_a_fork_point_inside_a_gather_region_is_refused(tmp_path, sqlite_app):
    """A gather region is ATOMIC for cutting, so a fork point inside one is refused at the entry
    rather than half-working.

    It half-works, which is why it needs a refusal rather than a docstring: a gather branch's
    await is resolved by `peek_event`, not `await_event`, so the `Seeding -> Live` phase crosses on
    the FIRST pass (at `_join`'s re-arm) and never again on a replay; and a durable fork is
    replay. Without the refusal, the first pass parks correctly at
    `fork:r-fork;gather:0,1;ev:m1`, and the resume dies with a `SeedBoundaryError` blaming
    `through`, whose message enumerates three causes, none of them this one."""
    from effective.counterfactual import ForkPointInGather

    db = tmp_path / "fpg.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    from effective.sqlite import SqliteTaskContext

    with pytest.raises(ForkPointInGather) as caught:
        run_fork(
            # a real ctx, never touched: the refusal precedes every other input's use, which is
            # what makes it a precondition rather than a failure discovered mid-run
            SqliteTaskContext(app.conn, _UNUSED_TASK),
            lambda: _decision_wf("r-fork"),
            child_run_id="r-fork",
            seed=seed,
            fork_point=compose_key(t"gather:0,1;review:{Segment(MID)}"),
            hypothetical_ledger=SqliteLedger(
                app.conn, "r-fork", app.write_lock, hypothetical=True
            ),
            domain=_Dom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
        )
    assert "INSIDE a gather region" in str(caught.value)
    assert "peek_event" in str(caught.value)  # the mechanism, not just the verdict


# --- a `gather` in the forked TAIL: from "runs, unverified" to pinned -------------------------
#
# These pin forking a gather-bearing region as a supported case. The shape matters: the
# counterfactual's own tail fans out, so the fork's machinery (SeedingCtx, RenamedAwaitCtx,
# ForkLedger, DryRun) has to survive branches running in real threads under `asyncio.to_thread`.


def _tail_gather_wf(run_id: str):
    """The decision workflow whose TAIL fans out: prefix + fork point, then two branches that each
    call a tool and append a row."""

    def branch(tool: str, kind: str):
        def thunk():
            value = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"{kind}:{MID}"), kind=kind))
            return value

        return thunk

    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    results = yield from gather([branch("a", "ka"), branch("b", "kb")])
    return {"decision": approval["decision"], "results": results}


def _nested_gather_wf(run_id: str):
    """A gather whose branch is itself a gather: the graph scheme nests depth inside width."""

    def leaf(tool: str, kind: str):
        def thunk():
            value = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"{kind}:{MID}"), kind=kind))
            return value

        return thunk

    def inner():
        def thunk():
            return (yield from gather([leaf("a", "ka"), leaf("b", "kb")]))

        return thunk

    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    results = yield from gather([inner()])
    return {"decision": approval["decision"], "results": results}


class _ToolDom(_Dom):
    """`_Dom` plus a counting tool runner — so a crash test can assert EXACTLY-ONCE across
    retries, which is the property a re-run branch would violate."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op):
        name = getattr(op, "name", None)
        if name is None:
            return super().run(op)
        self.calls.append(name)
        return {"a": 10, "b": 20}[name]


def _gather_fork(app, seed, rid: str, wf, *, ctx_wrap=None, max_attempts: int = 8):
    """Register + spawn one gather-bearing fork; returns `(task_id, domain)` so a caller can
    assert exactly-once on the domain the child actually used."""
    dom = _ToolDom()

    @app.register_task(f"gfork-{rid}")
    def fork_task(params, ctx, _dom=dom):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx_wrap(ctx) if ctx_wrap is not None else ctx,
            lambda: wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_dom,
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            allow=frozenset({"a", "b"}),  # read-only tools the counterfactual may call
        )

    return app.spawn(f"gfork-{rid}", {"run_id": rid}, max_attempts=max_attempts), dom


def _base_for(app, db: str, wf) -> UUID:
    @app.register_task("gbase")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, _ToolDom(), ledger=led).run(lambda: wf(params["run_id"]))

    base_id = app.spawn("gbase", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(REVIEW.stored(), {"decision": "reject"})
    app.run_until_result(base_id)
    return base_id


def test_a_gather_in_the_forked_tail_runs_hypothetically_and_seals(tmp_path, sqlite_app):
    """The baseline the docstring described but nothing held: the counterfactual's tail fans out,
    each branch's rows land hypothetical and lineage-scoped, `DryRun` governs each branch, the
    base is untouched, and the fork seals."""
    db = tmp_path / "tg.db"
    app = sqlite_app(str(db))
    base_id = _base_for(app, str(db), _tail_gather_wf)
    base_before = _rows(app, "r-base")
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    fid, _dom = _gather_fork(app, seed, "gf", _tail_gather_wf)
    app.run_until_result(fid)
    app.emit_event(_forked("gf", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fid)

    assert snap is not None
    assert snap.result == {"decision": "approve", "results": [10, 20]}
    kinds = [k for _, k, _ in _rows(app, "gf")]
    assert kinds[0] == "forked"
    assert kinds[-1] == "fork_sealed"
    assert sorted(kinds[1:-1]) == ["ka", "kb"]  # both branches committed, order is a race
    assert all(hyp == 1 for _, _, hyp in _rows(app, "gf"))
    assert all(e.startswith("hyp:gf;") for e, _, _ in _rows(app, "gf"))
    assert _rows(app, "r-base") == base_before


def _colliding_tail_gather_wf(run_id: str):
    """`_tail_gather_wf` with the ONE difference under test: both branches author the SAME
    `event_id` instead of hand-disambiguating to `ka:`/`kb:`. Everything else is identical, so a
    difference in outcome is attributable to the id alone."""

    def branch(tool: str):
        def thunk():
            value = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"done:{MID}"), kind="done"))
            return value

        return thunk

    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    results = yield from gather([branch("a"), branch("b")])
    return {"decision": approval["decision"], "results": results}


def test_a_fork_whose_tail_collides_on_one_event_id_does_not_seal(tmp_path, sqlite_app):
    """The seal is a POSITIVE attestation on the clean path, so absence means "not known valid".
    A lineage that silently lost a row is not valid, and must not carry the attestation."""
    db = tmp_path / "tgx.db"
    app = sqlite_app(str(db))
    base_id = _base_for(app, str(db), _colliding_tail_gather_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    fid, _dom = _gather_fork(app, seed, "gf", _colliding_tail_gather_wf)
    app.run_until_result(fid)
    app.emit_event(_forked("gf", REVIEW), {"decision": "approve"})
    app.run_until_result(fid)

    rows = _rows(app, "gf")
    kinds = [k for _, k, _ in rows]
    # Both branches committed a row, so both must be on the record — or the fork must not seal.
    assert kinds.count("done") == 2 or "fork_sealed" not in kinds, (
        f"the fork sealed over a lineage holding {kinds.count('done')} of 2 committed rows: {rows}"
    )


def test_the_forked_tails_gather_survives_a_crash_at_every_branch_op(tmp_path, sqlite_app):
    """The durability gate, inside the fan-out. A crash at each distinct branch op resumes by
    replay to the IDENTICAL hypothetical outcome, and each tool runs EXACTLY ONCE across the
    crash — the property a re-run branch would break, and the reason a parked branch must never
    be re-executed in-process."""
    db = tmp_path / "tgc.db"
    app = sqlite_app(str(db))
    base_id = _base_for(app, str(db), _tail_gather_wf)
    base_before = _rows(app, "r-base")
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    targets = [
        "extract",  # a SEEDED prefix op (the replay commits it through the child's ctx)
        LEDGER_EXTRACTED.stored(),
        REVIEW.stored(),  # the fork point, renamed
        "gather:0,0;step;tool:a",  # ...and then each op INSIDE the fan-out
        # lint: terminal-hole — a `Key` splice
        compose_key(t"gather:{0},{0};ledger;{KA:domain=address}").stored(),
        "gather:0,1;step;tool:b",
        # lint: terminal-hole — a `Key` splice
        compose_key(t"gather:{0},{1};ledger;{KB:domain=address}").stored(),
    ]
    for i, target in enumerate(targets):
        rid = f"cg{i}"  # shares no substring with any target
        fault = _RecFault(target)
        fid, dom = _gather_fork(
            app, seed, rid, _tail_gather_wf, ctx_wrap=lambda c, _f=fault: _RecFaultCtx(c, _f)
        )
        app.run_until_result(fid)
        app.emit_event(
            fork_event_name(rid, Key.parse(REVIEW.stored())).stored(), {"decision": "approve"}
        )
        snap = app.run_until_result(fid)

        assert fault.fired, f"target={target}: never fired"
        assert target in fault.fired[0], f"target={target}: fired at {fault.fired[0]!r}"
        assert snap is not None
        assert snap.result == {"decision": "approve", "results": [10, 20]}, f"target={target}"
        kinds = [k for _, k, _ in _rows(app, rid)]
        assert kinds[0] == "forked", f"target={target}: {kinds}"
        assert kinds[-1] == "fork_sealed", f"target={target}: {kinds}"
        assert sorted(kinds[1:-1]) == ["ka", "kb"], f"target={target}: {kinds}"
        # exactly-once across the crash: a replayed branch re-binds its checkpoint, never re-runs
        assert sorted(dom.calls) == ["a", "b"], f"target={target}"

    assert _rows(app, "r-base") == base_before


def test_a_nested_gather_in_the_forked_tail_composes(tmp_path, sqlite_app):
    """Depth inside width, inside a counterfactual: the inner gather's keys compose under the
    outer branch's prefix, so nothing collides with the seed or with a sibling, and the fork
    seals as usual."""
    db = tmp_path / "tgn.db"
    app = sqlite_app(str(db))
    base_id = _base_for(app, str(db), _nested_gather_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    fid, _dom = _gather_fork(app, seed, "gn", _nested_gather_wf)
    app.run_until_result(fid)
    app.emit_event(_forked("gn", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fid)

    assert snap is not None
    assert snap.result == {"decision": "approve", "results": [[10, 20]]}
    keys = [c.key.stored() for c in read_sqlite_task(str(db), fid)]
    # the inner gather nests under the outer branch, and the arm sits on the LEAF
    assert "gather:0,0;gather:0,0;step;tool:a" in keys
    assert "gather:0,0;gather:0,1;step;tool:b" in keys
    kinds = [k for _, k, _ in _rows(app, "gn")]
    assert kinds[0] == "forked"
    assert kinds[-1] == "fork_sealed"


def test_a_dry_run_violation_inside_a_forked_gather_is_caught_by_except_star(tmp_path, sqlite_app):
    """The refusal contract across the fan-out: a branch raises inside an
    `asyncio.TaskGroup`, so the typed refusal arrives WRAPPED in an `ExceptionGroup`, while the
    same refusal outside a gather arrives bare. `except*` catches both, which is why
    `run_fork_as_task` classifies with it. The failed task is reported as the refusal itself."""
    db = tmp_path / "tgd.db"
    app = sqlite_app(str(db))
    base_id = _base_for(app, str(db), _tail_gather_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    @app.register_task("nodry")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _tail_gather_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_ToolDom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            allow=frozenset(),  # NO tool is allowed — the branch must be refused
        )

    fid = app.spawn("nodry", {"run_id": "gd"}, max_attempts=1)
    app.run_until_result(fid)
    app.emit_event(_forked("gd", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fid)

    assert snap is not None
    assert snap.state == "failed"
    assert "WorldMutation" in str(snap.failure)

    # ...and `except*` is what classifies it either way — the shape `run_fork_as_task` relies on.
    caught: list[str] = []
    try:
        raise ExceptionGroup("tg", [WorldMutation("a", frozenset())])
    except* WorldMutation as eg:
        caught.append(type(eg).__name__)
    try:
        raise WorldMutation("a", frozenset())  # the same refusal, bare (a fork with no gather)
    except* WorldMutation as eg:
        caught.append(type(eg).__name__)
    assert caught == ["ExceptionGroup", "ExceptionGroup"]  # `except*` wraps a bare one to match
    assert [k for _, k, _ in _rows(app, "gd")] == ["forked"]  # genesis only; no tail, no seal


# --- a counterfactual that spawns a counterfactual -------------------------------------------
#
# A DIAGONAL cell of the composition grid (`fork ∘ fork`), which the pairs grid cannot reach
# because `fork` is not a registry member. The child's event world (`RenamedAwaitCtx`) rescopes
# every awaited name to `fork:{child}:`, while the grandchild emits its `done_event` from its own
# params, so the join would park forever on a name nothing produces, on both engines. It is
# refused, because both frames are needed and neither can give way until rename-aware emit exists
# (`ops.refuse_absolute_await_under_a_rename`).


_GRANDCHILD = "g1"


def _spawning_tail_wf(rid: str):
    """The decision workflow whose TAIL spawns a counterfactual of its own and joins it.

    Not a contrived shape: `spawn_fork`'s `budget` parameter exists precisely to bound "a
    counterfactual that spawns counterfactuals" (`fork.py`), so the substrate designs for this."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    handle = yield from spawn_fork(
        "grandchild",
        child_run_id=_GRANDCHILD,
        base_task_id=_UNUSED_TASK,
        through=LEDGER_EXTRACTED.stored(),
        fork_point=REVIEW,
        forked_from=rid,
        forked_at_event=EXTRACTED.stored(),
        delta={"decision": "reject"},
    )
    outcome = yield from join_fork(handle)
    return f"{approval['decision']}/{outcome.child_run_id}"


class _SpawnDom(_Dom):
    """`_Dom` plus a `spawn` that answers without enqueuing: the grandchild's own execution
    is not what is under test, and canning it keeps the base's half a two-line setup."""

    def __init__(self) -> None:
        self.spawned: list[str] = []
        self.done_events: list[str] = []

    def run(self, op):
        if getattr(op, "name", None) == SPAWN_TOOL:
            self.spawned.append(op.args["task_name"])
            self.done_events.append(op.args["params"]["done_event"])
            return SpawnResult(task_id=_UNUSED_TASK)
        return super().run(op)


def test_spawn_and_join_at_the_BASE_run_completes(tmp_path, sqlite_app):
    """The order that must stay open, pinned FIRST: the discipline the composition table's
    disjunction law encodes. A run that spawns a fork and joins it parks on the done event its
    child was handed, the exact name the child emits, and resumes to completion. The refusal below
    is narrow only if this still works; a guard that took both orders would refuse a composition
    that is name-correct end to end."""
    db = tmp_path / "f2base.db"
    app = sqlite_app(str(db))
    dom = _SpawnDom()

    @app.register_task("f2base")
    def base_task(params, ctx):
        led = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, dom, ledger=led).run(
            lambda: _spawning_tail_wf(params["run_id"])
        )

    task_id = app.spawn("f2base", {"run_id": "r-base"})
    app.run_until_result(task_id)
    app.emit_event(REVIEW.stored(), {"decision": "approve"})
    app.run_until_result(task_id)

    # Parked on the child's own name, unframed because nothing framed it.
    (handed,) = dom.done_events
    assert _waiting_on(app, task_id) == handed
    app.emit_event(handed, {"answer": {"kind": "returned", "value": "completed"}})
    snap = app.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == f"approve/{_GRANDCHILD}"
    assert dom.spawned == ["grandchild"]


def test_a_fork_child_that_spawns_and_joins_a_grandchild_is_refused_not_deadlocked(
    tmp_path, sqlite_app
):
    """The same tail, one level down. Unrefused, it is the quietest failure the substrate has:
    the spawn succeeds, the join parks, and the task sits `waiting` forever on
    `fork:r-fork;fork_done:g1` while the grandchild emits `fork_done:g1`.

    Asserts the three things that separate a refusal from a deadlock (the shape
    `test_a_prefix_budget_grant_park_is_refused_not_deadlocked` established): terminal, NOT parked,
    and the message names both halves of the mismatch plus a way out."""
    db = tmp_path / "f2fork.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())
    dom = _SpawnDom()

    @app.register_task("f2fork")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _spawning_tail_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=dom,
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            allow=frozenset({SPAWN_TOOL}),  # the `DryRun` exit its own message recommends
        )

    fork_id = app.spawn("f2fork", {"run_id": "r-fork"}, max_attempts=1)
    app.run_until_result(fork_id)
    # The child parks at its OWN fork point (the rename doing its intended job), so the delta
    # arrives under the child's name — the same handoff every fork test here performs.
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    snap = app.run_until_result(fork_id)

    assert snap is not None
    assert snap.state == "failed"  # terminal, not `waiting` forever
    assert _waiting_on(app, fork_id) is None  # and not parked on anything
    failure = str(snap.failure)
    (handed,) = dom.done_events
    assert f"'{handed}'" in failure  # what the grandchild emits
    assert "'fork:r-fork;'" in failure  # what this child would have prepended
    assert "transplanted" in failure  # the substrate's own opt-out, named
    assert "ALREADY enqueued" in failure  # the orphan, per N1's honesty lever

    # The counterexample, kept checkable now that the guard makes it unreachable — and COMPUTED
    # through the same function the ctx applies, so it cannot drift from what the rename does.
    parked = fork_event_name("r-fork", Key.parse(handed)).stored()
    assert parked == _forked("r-fork", Key.parse(handed))
    assert parked != handed, "the rename is what makes the join unwakeable"

    assert [k for _, k, _ in _rows(app, "r-fork")] == ["forked"]  # genesis only; no seal


class Answered(BaseModel):
    """What these joins read of an answer the caller emits by hand."""

    child_run_id: str


def _transplanted_join_wf(_rid: str):
    """A tail whose ABSOLUTE join the CALLER answers under the child's renamed name."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(LedgerRow(event_id=EXTRACTED, kind="extracted"))
    approval = yield from await_event(REVIEW.stored(), dict)
    outcome = yield from await_event(_done(_GRANDCHILD), Answered, addressing=Addressing.ABSOLUTE)
    return f"{approval['decision']}/{outcome.child_run_id}"


def test_a_transplanted_absolute_await_under_a_rename_is_NOT_refused(tmp_path, sqlite_app):
    """The rename guard honors `transplanted`, as every refusal in the fork family does.

    `transplanted` is the substrate's own word for *"the caller HAS delivered this under the
    child's names"*, so the emitter IS rename-aware, the name is produced, and the refusal's
    message ("the await would park forever on a name nothing produces") would be FALSE for this
    shape. With the guard off, this shape completes end to end; a guard that refused it would
    be refusing a name-correct composition."""
    db = tmp_path / "transplant.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())
    done = _done(_GRANDCHILD)

    @app.register_task("tp")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _transplanted_join_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_SpawnDom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
            transplanted=frozenset({done}),  # the caller emits it under the CHILD's scope
        )

    fork_id = app.spawn("tp", {"run_id": "r-fork"}, max_attempts=1)
    app.run_until_result(fork_id)
    app.emit_event(_forked("r-fork", REVIEW), {"decision": "approve"})
    app.run_until_result(fork_id)

    # It parked on the RENAMED name — which is exactly what `transplanted` declares the caller
    # will emit — rather than being refused.
    assert _waiting_on(app, fork_id) == _forked("r-fork", done)
    app.emit_event(_forked("r-fork", done), {"child_run_id": _GRANDCHILD, "status": "completed"})
    snap = app.run_until_result(fork_id)

    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == f"approve/{_GRANDCHILD}"


def _done(child_run_id: str):
    """A grandchild's join name, asked of the minter the handler names done events with."""
    placement = compose_key(t"tool:spawn,{Run(child_run_id)}")
    return spawn_done_name(Writer(task=str(_UNUSED_TASK), placement=placement))


def _prefix_absolute_wf(_rid: str):
    """An ABSOLUTE await in the replayed PREFIX — before the fork point."""
    yield from ask_llm("extract", [], dict)
    yield from await_event(_done(_GRANDCHILD), Answered, addressing=Addressing.ABSOLUTE)
    approval = yield from await_event(REVIEW.stored(), dict)
    return approval["decision"]


def test_an_absolute_await_in_the_PREFIX_still_gets_the_prefix_diagnosis(tmp_path, sqlite_app):
    """The generic rename guard must not SHADOW a better, purpose-built refusal.

    An await in a fork's replayed prefix is `SeedingCtx`'s business: it raises
    `ForkedPrefixAwait`, which names *which* await, names `fork_point`, points at `transplanted`,
    and is in `fork.REFUSALS` so a parent gets an answer. The rename guard sits in the handler,
    above that ctx, so firing on every ABSOLUTE name would replace the specific diagnosis with a
    generic one, and a base whose prefix contains a `join_fork` or a `spawn_subagent_task` (an
    ordinary shape) would lose both the diagnosis and the parent's answer.

    A generic guard displacing a specific one is a regression even when both refuse."""
    db = tmp_path / "prefixabs.db"
    app = sqlite_app(str(db))
    base_id = _run_base(app, str(db), _decision_wf)
    seed = fork_seed(read_sqlite_task(str(db), base_id), through=LEDGER_EXTRACTED.stored())

    @app.register_task("pa")
    def fork_task(params, ctx):
        hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
        return run_fork(
            ctx,
            lambda: _prefix_absolute_wf(params["run_id"]),
            child_run_id=params["run_id"],
            seed=seed,
            hypothetical_ledger=hyp,
            domain=_SpawnDom(),
            forked_from="r-base",
            forked_at_event=EXTRACTED.stored(),
            fork_point=REVIEW,
            delta={"decision": "approve"},
        )

    fork_id = app.spawn("pa", {"run_id": "r-fork"}, max_attempts=1)
    snap = app.run_until_result(fork_id)

    assert snap is not None
    assert snap.state == "failed"
    failure = str(snap.failure)
    assert "ForkedPrefixAwait" in failure, "the SPECIFIC refusal must not be shadowed"
    assert REVIEW.stored() in failure  # it names the fork point the caller meant
    assert "transplanted" in failure  # and the opt-out
