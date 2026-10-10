"""`respawn` on a real engine: the generation boundary, and the property it exists for.

A long-lived loop keeps running without its replay history growing without bound. That is an
ENGINE property, not a control-flow one: it is about what the checkpoint store holds after the
cut, so a recorder test cannot show it. Hence this file, on the embedded SQLite engine, which is
enough because the property under test is "the new task's store starts empty", true by
construction on both engines (`_conformance` covers the cross-engine op semantics).

What a budget addresses once a run is a chain is an open question, and
`tests/test_carrier_audit.py` is its red test.
"""

from itertools import pairwise
from typing import Any
from uuid import UUID

import pytest
from _spawning import sqlite_spawner
from pydantic import BaseModel

from effective.api import append_ledger, ask_llm, call_tool, gather, scoped
from effective.budget import Budget, MeasuredBudget
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.cost import CONTRACT_PARAM, Contract, Usage
from effective.domain import SPAWN_TOOL, CallTool, DomainOp
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler, Respawned
from effective.interpreters.tools import spawn_tool
from effective.keys import compose_key
from effective.ops import LedgerRow


class Watch(BaseModel):
    """The carry — everything that survives a generation, and nothing else does."""

    seen: int = 0


BATCH = 2


class _SpendingDomain:
    """Answers the spawn tool and a metered `AskLLM` — the `(value, Usage)` V1 shape, which is
    what makes the handler's meter move at all."""

    def __init__(self, app: SqliteApp, cost: float) -> None:
        self._spawn = spawn_tool(sqlite_spawner(app))
        self.cost = cost

    def run(self, op: DomainOp[Any]) -> Any:
        # Tools go through `run`; only `run_metered` reports usage, and `metered_call` requires
        # BOTH an `AskLLM` and a domain that implements it. A `CallTool` therefore accrues
        # nothing — the measured ceiling is a MODEL-spend ceiling by construction.
        assert isinstance(op, CallTool)
        return self._spawn(op)

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        """The v1 usage-in-checkpoint seam — the third condition `metered_call` requires."""
        return "ans", Usage(prompt_tokens=10, completion_tokens=5, cost=self.cost)


class _Domain:
    """Answers the substrate's spawn tool and one observational call per batch item."""

    def __init__(self, app: SqliteApp) -> None:
        self._spawn = spawn_tool(sqlite_spawner(app))
        self.polls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        # `DomainOp`, not `CallTool`: the protocol's alphabet includes `AskLLM`, and narrowing
        # the parameter would make this domain unassignable to `DomainInterpreter`.
        assert isinstance(op, CallTool)
        if op.name == SPAWN_TOOL:
            return self._spawn(op)
        self.polls.append(op.args["at"])
        return "item"


def watch_batch(state: Watch, turn: Turn):
    """One generation: drain BATCH items, then hand the cursor forward.

    Note what the author does NOT write: no scope, no generation in any name. The ops are named
    plainly and the generation is a *task* boundary, which is why the folded view of this
    workflow is the same at `generations=3` as at `generations=None`."""
    seen = state.seen
    for i in range(BATCH):
        yield from call_tool(f"poll{i}", {"at": f"{turn.generation}:{i}"}, str)
        seen += 1
    # SUBJECT-scoped: the same item triaged twice across generations is ONE row, and a
    # run-id-scoped id here would lose a row silently.
    yield from append_ledger(
        # lint: terminal-hole: `generation: int` on the respawn turn (`combinators.Turn`).
        LedgerRow(event_id=compose_key(t"batch:{turn.generation}"), kind="batch")
    )
    if turn.final:
        return Done({"seen": seen, "ended_at": turn.generation})
    return Again(Watch(seen=seen))


def _register(
    app: SqliteApp, domain: _Domain, generations: int, *, budget: Budget | None = None
) -> None:
    @app.register_task("watch")
    def task(params, ctx):
        run_id = params["run_id"]
        chain = Chain.from_params(params, task="watch", schema=Watch, initial=Watch())
        ledger = SqliteLedger(app.conn, run_id, app.write_lock)
        return DurableHandler(ctx, domain, ledger=ledger, params=params).run(
            lambda: respawn(watch_batch, chain, budget=budget or Budget(generations=generations))
        )


def _drain(app: SqliteApp, rounds: int = 30) -> None:
    for _ in range(rounds):
        if not app.work_batch():
            return


def _checkpoint_keys(app: SqliteApp, task_id: UUID) -> list[str]:
    from effective.checkpoints import keys, read_sqlite_conn

    # `read_sqlite_conn`, not `read_sqlite_task`: an `:memory:`-style store has no path to reopen
    # (the same note `_conformance.SqliteBackend` carries).
    return [str(k) for k in keys(read_sqlite_conn(app.conn, task_id))]


def _tasks(app: SqliteApp) -> list[tuple[str, str]]:
    rows = app.conn.execute("SELECT task_id, state FROM tasks ORDER BY rowid").fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def _result(app: SqliteApp, task_id: str) -> Any:
    import json

    row = app.conn.execute("SELECT result FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] is not None else None


def test_a_chain_runs_to_done_and_each_generation_is_its_own_task(tmp_path, sqlite_app):
    """Three generations, three tasks, one run — and the carry is the only thing that crosses."""
    app = sqlite_app(str(tmp_path / "respawn.db"))
    domain = _Domain(app)
    _register(app, domain, generations=3)
    first = app.spawn("watch", {"run_id": "r1"})
    _drain(app)

    tasks = _tasks(app)
    assert len(tasks) == 3, tasks
    assert [state for _, state in tasks] == ["completed"] * 3

    # The chain's value lands in the LAST task, not the one we spawned, so an operator surface
    # has to follow the chain to find it.
    assert _result(app, str(first)) == {"next_generation": 1, "task_id": tasks[1][0]}
    last = _result(app, tasks[2][0])
    assert last == {"seen": BATCH * 3, "ended_at": 2}

    # The carry is what survived: six polls across three generations, each seeing its own turn.
    assert domain.polls == [f"{g}:{i}" for g in range(3) for i in range(BATCH)]


def test_replay_history_is_BOUNDED_which_is_the_whole_point(tmp_path, sqlite_app):
    """The claim the design exists to make, measured: generation *n+1* starts with an empty
    store, so a chain's history does not grow with its length.

    Asserted as a bound rather than an equality so it cannot pass by accident: every generation
    holds the SAME number of checkpoints, and that number is independent of how many have run.
    A single unbounded task would instead accumulate `generations x per_generation`."""
    app = sqlite_app(str(tmp_path / "bounded.db"))
    _register(app, _Domain(app), generations=4)
    app.spawn("watch", {"run_id": "r1"})
    _drain(app)

    per_generation = [len(_checkpoint_keys(app, UUID(tid))) for tid, _ in _tasks(app)]

    assert len(per_generation) == 4
    # Generations 0..2 each spawn a successor (one extra checkpoint); the final one does not.
    assert len(set(per_generation[:-1])) == 1, per_generation
    assert max(per_generation) <= per_generation[0], per_generation
    assert sum(per_generation) > max(per_generation), "sanity: the chain really did run"


def test_the_generation_boundary_is_a_ledger_event(tmp_path, sqlite_app):
    """A granted generation is a governance decision, so it belongs in the CANONICAL
    record. Deriving it from checkpoint rotation would derive the canonical bookkeeper from the
    disposable one, which is the two-bookkeepers rule.

    One row per boundary, none for the final generation: three generations means two cuts."""
    app = sqlite_app(str(tmp_path / "ledger.db"))
    _register(app, _Domain(app), generations=3)
    app.spawn("watch", {"run_id": "r1"})
    _drain(app)

    rows = app.conn.execute(
        "SELECT event_id, kind FROM ledger WHERE workflow_run_id = ? ORDER BY seq", ("r1",)
    ).fetchall()
    kinds = [str(r[1]) for r in rows]
    ids = [str(r[0]) for r in rows]

    assert kinds.count("respawned") == 2, kinds
    assert [i for i in ids if i.startswith("respawned:")] == ["respawned:r1,1", "respawned:r1,2"]
    # The author's own rows are subject-scoped and coexist with the substrate's.
    assert kinds.count("batch") == 3


def test_again_at_a_final_turn_is_a_loud_error(tmp_path, sqlite_app):
    """The same shape `descend` raises for `Deeper` past an exhausted budget: the combinator
    holds no `T`, so a step that will not answer when asked for the last time is a dev error,
    not a silent extra generation."""

    def never_done(state: Watch, turn: Turn):
        yield from call_tool("poll0", {"at": "x"}, str)
        return Again(state)  # ignores `turn.final` — the mistake under test

    app = sqlite_app(str(tmp_path / "loud.db"))
    domain = _Domain(app)

    @app.register_task("bad")
    def task(params, ctx):
        chain = Chain.from_params(params, task="bad", schema=Watch, initial=Watch())
        return DurableHandler(ctx, domain, ledger=None, params=params).run(
            lambda: respawn(never_done, chain, budget=Budget(generations=1))
        )

    task_id = app.spawn("bad", {"run_id": "r1"}, max_attempts=1)
    _drain(app)

    state, failure = app.conn.execute(
        "SELECT state, failure FROM tasks WHERE task_id = ?", (str(task_id),)
    ).fetchone()
    assert state == "failed"
    assert "returned Again at generation 0" in str(failure)
    assert "turn.final" in str(failure), "the message must name what to check"


# --- the two carries ---------------------------------------------------------------------


def test_the_generation_bound_DECREMENTS_across_the_chain(tmp_path, sqlite_app):
    """The generation bound carries. A budget rebuilt from a literal every generation never
    decrements, and the chain runs past any bound.

    `generations=N` means N generations RUN, so the declared bound is the count of tasks."""
    for bound in (1, 2, 5):
        app = sqlite_app(str(tmp_path / f"bound{bound}.db"))
        _register(app, _Domain(app), generations=bound)
        app.spawn("watch", {"run_id": "r1"})
        _drain(app, rounds=80)
        assert len(_tasks(app)) == bound, (bound, _tasks(app))


def test_the_measured_accrual_CROSSES_a_generation_boundary(tmp_path, sqlite_app):
    """`overall` binds ACROSS the generations of a chain.

    The substrate carries `(spent, granted, trips)` in reserved params, invisible to the author
    exactly as checkpoint ids are, so generation *n+1* starts where *n* left off. Re-arming
    instead lets four generations under one ceiling spend 5.3x it.

    Params carry it because a chain is strictly SEQUENTIAL; a concurrent fleet would need a
    shared spend store, which is deferred and outside this mechanism.

    The workflow must ACTUALLY SPEND for this to test anything: `call_tool` accrues nothing, so
    every seed would be 0.0 and the assertion would pass with the carry deleted. Metered spend
    rides `ask_llm` under `Contract.V1`."""
    app = sqlite_app(str(tmp_path / "accrual.db"))
    domain = _SpendingDomain(app, cost=0.001)

    def spend_batch(state: Watch, turn: Turn):
        yield from ask_llm("think", "x", str)
        if turn.final:
            return Done({"seen": state.seen + 1})
        return Again(Watch(seen=state.seen + 1))

    @app.register_task("spend")
    def task(params, ctx):
        run_id = params["run_id"]
        chain = Chain.from_params(params, task="spend", schema=Watch, initial=Watch())
        budget = MeasuredBudget(run_id=run_id, overall=100.0, on_exhaust="fail")
        return DurableHandler(
            ctx, domain, ledger=None, budget=budget, params=params, contract=Contract.V1
        ).run(lambda: respawn(spend_batch, chain, budget=Budget(generations=3)))

    app.spawn("spend", {"run_id": "r1", CONTRACT_PARAM: Contract.V1.value})
    _drain(app)

    tasks = _tasks(app)
    assert len(tasks) == 3, tasks
    seeded = [_prior_accrual_of(app, task_id) for task_id, _ in tasks]

    # Generation 0 starts at zero; each later one is seeded with everything spent before it, so
    # the seeds are STRICTLY increasing. All-zero (the carry deleted) fails both assertions.
    assert seeded[0] == 0.0
    assert all(b > a for a, b in pairwise(seeded)), seeded
    assert seeded[-1] > 0.0, "a chain that never accrued cannot show a carry"


def _prior_accrual_of(app: SqliteApp, task_id: str) -> float:
    """The spend this task was SEEDED with, read off its own params."""
    import json

    from effective.ops import ACCRUAL_PARAM

    row = app.conn.execute("SELECT params FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    carried = json.loads(row[0]).get(ACCRUAL_PARAM)
    return 0.0 if not carried else float(carried[0])


# --- known defects, pinned until fixed ----------------------------------------------------
#
# `xfail(strict)` so the suite stays green and a premature pass is a FAILURE. Each flips to a
# positive assertion in the commit that fixes it.


@pytest.mark.xfail(strict=True, reason="enforce_generation is wired into no handler")
def test_a_per_generation_ceiling_is_ACTUALLY_ENFORCED(tmp_path, sqlite_app):
    """`MeasuredBudget(per_generation=...)` is accepted and limits nothing.

    `enforce_generation` exists and is called by no handler, so a governance surface that reads
    as enforced is not. Two asks at 0.001 each under a 0.0015 per-generation ceiling must trip;
    today the task completes."""
    app = sqlite_app(str(tmp_path / "pergen.db"))
    domain = _SpendingDomain(app, cost=0.001)

    def spend_twice(state: Watch, turn: Turn):
        yield from ask_llm("a", "x", str)
        yield from ask_llm("b", "x", str)
        return Done({"seen": 2})

    @app.register_task("pergen")
    def task(params, ctx):
        chain = Chain.from_params(params, task="pergen", schema=Watch, initial=Watch())
        budget = MeasuredBudget(run_id=params["run_id"], per_generation=0.0015, on_exhaust="fail")
        return DurableHandler(
            ctx, domain, ledger=None, budget=budget, params=params, contract=Contract.V1
        ).run(lambda: respawn(spend_twice, chain, budget=Budget(generations=1)))

    app.spawn("pergen", {"run_id": "r1", CONTRACT_PARAM: Contract.V1.value})
    _drain(app)

    state = _tasks(app)[0][1]
    assert state == "failed", "the per-generation ceiling must bind"

    # Folded in rather than pinned separately: the meter is seeded with the CHAIN's prior spend
    # so `overall` binds across generations, so a per-generation check fed `self._meter.cost`
    # would compare the chain total against a per-generation ceiling and trip on generation 2
    # whatever it spent. A standalone xfail would XPASS while nothing trips at all, for the wrong
    # reason; here only a fix that keeps the two figures apart satisfies it.
    app2 = sqlite_app(str(tmp_path / "pergen2.db"))
    domain2 = _SpendingDomain(app2, cost=0.001)

    def spend_once(state: Watch, turn: Turn):
        yield from ask_llm("a", "x", str)
        if turn.final:
            return Done({"seen": state.seen + 1})
        return Again(Watch(seen=state.seen + 1))

    @app2.register_task("pergen2")
    def chained(params, ctx):
        chain = Chain.from_params(params, task="pergen2", schema=Watch, initial=Watch())
        budget = MeasuredBudget(
            run_id=params["run_id"], overall=100.0, per_generation=0.0015, on_exhaust="fail"
        )
        return DurableHandler(
            ctx, domain2, ledger=None, budget=budget, params=params, contract=Contract.V1
        ).run(lambda: respawn(spend_once, chain, budget=Budget(generations=2)))

    app2.spawn("pergen2", {"run_id": "r1", CONTRACT_PARAM: Contract.V1.value})
    _drain(app2)
    # Each generation spends 0.001 under a 0.0015 per-generation ceiling and must SURVIVE, while
    # the chain total (0.002) already exceeds it.
    assert [st for _, st in _tasks(app2)] == ["completed", "completed"], _tasks(app2)


# --- Family A: respawn's row is empty, and now something enforces it --------------------


def test_respawn_in_a_gather_branch_is_refused_BEFORE_the_spawn(tmp_path, sqlite_app):
    """B1/B3 — the blocker. Measured before this guard: `worker_deaths=2`, two tasks left
    `running` with NO failure recorded, and the sibling side effect fired 3x.

    Two properties, and the second is the one the ordering buys:

    1. It is REFUSED — not silently duplicated. `_ChainContinues` is a `BaseException`, and on the
       concurrent path a `TaskGroup` wraps it into a `BaseExceptionGroup` that `_run` cannot
       catch, so the group escaped `work_batch` and killed the worker while the successor was
       already committed.
    2. NO CHILD IS ENQUEUED. The guard fires before `ctx.step`, so there is no orphan running
       detached. A guard downstream of the spawn would satisfy (1) and not this.
    """
    app = sqlite_app(str(tmp_path / "branch.db"))
    domain = _Domain(app)

    def branch_body(state: Watch, turn: Turn):
        yield from call_tool("watch", {"at": "w"}, str)
        return Again(Watch(seen=state.seen + 1))

    @app.register_task("branch")
    def task(params, ctx):
        chain = Chain.from_params(params, task="branch", schema=Watch, initial=Watch())

        def respawning_branch():
            return (yield from respawn(branch_body, chain, budget=Budget(generations=3)))

        def sibling():
            return (yield from call_tool("other", {"at": "SIBLING"}, str))

        def wf():
            return (yield from gather([respawning_branch, sibling]))

        return DurableHandler(ctx, domain, ledger=None, params=params).run(wf)

    app.spawn("branch", {"run_id": "r1"}, max_attempts=1)
    _drain(app)

    tasks = _tasks(app)
    assert len(tasks) == 1, f"a refused branch must enqueue NO successor: {tasks}"
    assert tasks[0][1] == "failed", tasks
    # The sibling ran once, in the one generation that existed — not once per generation.
    assert domain.polls.count("SIBLING") == 1, domain.polls


def test_the_two_interpreters_agree_that_respawn_under_a_scope_travels_OUT(tmp_path, sqlite_app):
    """`scoped ∘ respawn` is LEGAL, and both interpreters give it one meaning.

    A scope is ordinary namespacing; the durable engine ends the task and keys the ops under the
    scope. A recorder that handed the `Respawned` back as the scope's VALUE would let the workflow
    resume past it.

    Asserted as agreement between the two, not as a recorder detail, because that is the
    invariant: a structural op must mean the same thing on both paths."""

    # in memory
    def body(state: Watch, turn: Turn):
        yield from call_tool("poll", {"at": "p"}, str)
        return Done({"seen": 1}) if turn.final else Again(Watch(seen=1))

    def wf_mem():
        chain = Chain(task="w", state=Watch(), run_id="r1")
        out = yield from scoped(
            compose_key(t"s:{0}"), lambda: respawn(body, chain, budget=Budget(generations=3))
        )
        return ["RESUMED", out]

    handler = RecordingHandler(responses={"s:0;tool:poll": "x"})
    assert isinstance(handler.run(wf_mem), Respawned), (
        "the recorder must END the run, not resume past the scope"
    )

    # durably: the same shape ends the task and spawns the successor
    app = sqlite_app(str(tmp_path / "scoped.db"))
    domain = _Domain(app)

    @app.register_task("scoped-chain")
    def task(params, ctx):
        chain = Chain.from_params(params, task="scoped-chain", schema=Watch, initial=Watch())
        return DurableHandler(ctx, domain, ledger=None, params=params).run(
            lambda: scoped(
                compose_key(t"s:{0}"),
                lambda: respawn(body, chain, budget=Budget(generations=3)),
            )
        )

    app.spawn("scoped-chain", {"run_id": "r1"})
    _drain(app)
    assert len(_tasks(app)) == 3, _tasks(app)


def test_the_respawned_row_carries_every_declared_field(tmp_path, sqlite_app):
    """A `kind="respawned"` row carries `generation`, **the authorizing grant**, and **a digest
    of the carry-state**: what happened, what authorized it, and what crossed the boundary.

    A granted generation is a governance decision, so a row that cannot answer "who allowed
    this, and for how many more" leaves the canonical record unable to audit the thing it exists
    for."""
    import json

    app = sqlite_app(str(tmp_path / "fields.db"))
    _register(app, _Domain(app), generations=3)
    app.spawn("watch", {"run_id": "r1"})
    _drain(app)

    payloads = [
        json.loads(p)
        for (k, p) in app.conn.execute(
            "SELECT kind, payload FROM ledger WHERE workflow_run_id = ? ORDER BY seq", ("r1",)
        ).fetchall()
        if k == "respawned"
    ]
    assert len(payloads) == 2, payloads
    for row in payloads:
        assert set(row) >= {"generation", "granted", "carry_digest"}, row
        assert isinstance(row["granted"], int)
        assert row["carry_digest"], "the carry crossed the boundary; the row must say what"

    # The digest tracks the CARRY, so two boundaries carrying different state differ.
    assert payloads[0]["carry_digest"] != payloads[1]["carry_digest"]
    assert all(row["granted"] == 0 for row in payloads), "no park was asked, so nothing granted"


def test_the_respawned_row_records_the_grant_that_authorized_it(tmp_path, sqlite_app):
    """When a human answers a generation park, the
    canonical record must say so. `granted=0` everywhere would satisfy the field's presence and
    none of its purpose, which is why this case exists beside the one above."""
    import json

    from effective.budget import chain_grant_name

    app = sqlite_app(str(tmp_path / "granted.db"))
    _register(app, _Domain(app), 0, budget=Budget(generations=1, on_exhaust="park", run_id="r1"))
    app.spawn("watch", {"run_id": "r1"})  # the id was only ever the emit's addressee
    _drain(app)

    # Generation 0 holds `remaining == 1`, so it asks before its step runs. Answer with 2 more.
    app.emit_event(chain_grant_name("r1", 0).stored(), {"add_generations": 2})
    _drain(app)

    granted = [
        json.loads(p)["granted"]
        for (k, p) in app.conn.execute(
            "SELECT kind, payload FROM ledger WHERE workflow_run_id = ? ORDER BY seq", ("r1",)
        ).fetchall()
        if k == "respawned"
    ]
    assert granted != [], "the chain crossed a boundary after the grant"
    assert granted[0] == 2, granted
