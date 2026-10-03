"""The parallel marginal sweep: N counterfactuals as their own durable tasks.

`spawn_fork` / `join_fork` are two EDGES of the graph, kept unfused: the parent
creates N children with one checkpointed `Step` each, then waits for their answers. The children
are separate tasks, so they are already concurrent; the parent's fan-out is a LOOP, not a
`gather`; `gather` is for in-TASK concurrency, and using it here would also break the
qualified-emitter contract (a branch's await is rescoped, the child's emit is not).

Three properties, one per test: the sweep produces one marginal per delta from one base; a
refused child ANSWERS rather than hanging its parent at the join edge; and an unjoined spawn is a
detached lineage that still runs and seals, which is the promotion path.

SQLite (0<->1), so it runs infra-free in `test-core`. The Absurd half of the same shape is
`test_fork_sweep_absurd.py`.
"""

from functools import partial
from uuid import UUID, uuid4

import pytest
from _conformance import refusals_of
from pydantic import BaseModel

from agent.runtime import make_tool_runner, spawn_tool
from effective.api import append_ledger, ask_llm, await_event
from effective.checkpoints import read_sqlite_conn
from effective.combinators import Again, Chain, respawn
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, SpawnResult
from effective.fork import ForkOutcome, join_fork, marginal_sweep, run_fork_as_task, spawn_fork
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Segment, compose_key
from effective.ops import LedgerRow
from effective.parked import read_sqlite_parked_conn
from effective.spawning import Failed, Returned
from effective.sqlite import SqliteApp, SqliteLedger


class ChainCarry(BaseModel):
    """The carry across a generation boundary, for the `fork ∘ respawn` pin."""

    n: int = 0


DELTAS = {"fork-approve": "approve", "fork-escalate": "escalate", "fork-reject": "reject"}


def new_message_id() -> str:
    """A fresh subject for ONE base run: the run-UNIQUE half of the naming rule.

    On Absurd events are queue-global and an answer is permanent (exactly one NULL→payload
    transition per name, ever: `infra/absurd/absurd.sql`), so a message id reused across runs
    lets a stale `review:` answer complete a fresh run *before it ever parks*. Measured on the
    live engine 2026-07-25: with a module-constant message id, a second base run went straight to
    `completed` with the first run's decision, having never registered a wake. A test that runs
    only on SQLite cannot see this.

    **It returns `str`, not `Segment`, and that is deliberate.** Branding the mint looks like the
    tidy move — one edit would cover every `compose_key(t"…:{message_id}")` downstream, since a
    `Segment` IS a `str`. It does not survive the trip: this id is handed to `app.spawn(...,
    {"message_id": message_id})`, Absurd stores task arguments as JSON, and the workflow reads a
    plain `str` back out of `params`. The brand would hold on the in-process walks and be gone on
    the durable one — a defect visible on exactly one engine, which is the class the repo's
    "a green SQLite pass is not a port" rule exists for.

    And it goes SILENTLY, which is the sharp half. Measured 2026-08-06:
    `json.dumps({"message_id": Segment("m1")})` is `{"message_id": "m1"}` and reads back a plain
    `str` — because a `Segment` IS a `str`, the encoder has nothing to object to. Compare
    `_spawner` below, where a `UUID` in the same payload raises `Object of type UUID is not JSON
    serializable`; that one is an explicit exit precisely because it is loud. So the wrap goes
    where the key is COMPOSED, which is downstream of every boundary."""
    return f"m{uuid4().hex[:8]}"


def decision_wf(message_id: str):
    """The exemplar's shape: a prefix step + a prefix ledger row, the review await (the fork
    point), then a decision-dependent tail.

    **`message_id` is the run's SUBJECT, not its run id** — and the difference is the whole naming
    rule a forkable workflow obeys. Every name authored here is FORK-STABLE: a child re-runs this
    same generator over the BASE's message id, so its step keys match the seed it was handed and
    its await matches the one `fork_point` the driver passed. Scope either on the run id instead
    and the fork is refused — both arms measured 2026-07-25 on a real fork:

    - run-scoped ledger ids → `SeedBoundaryError: step 'ledger;r-fork:extracted' ran LIVE during
      the Seeding phase` (the child's key is not in the base's seed);
    - a run-scoped await name → `ForkedPrefixAwait: … awaited 'review:r-fork', which is not the
      fork point 'review:r-base'` (the phase never crosses).

    Run-UNIQUENESS is the other axis and it lives OUTSIDE the workflow: the base gets it from a
    fresh `new_message_id()`, and the child gets it from `RenamedAwaitCtx`, which parks it at
    `fork:{child_run_id};review:{message_id}`. A workflow that reaches for the run id to get
    uniqueness is doing the substrate's job with the one variable a fork changes."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    approval = yield from await_event(f"review:{message_id}", dict)
    decision = approval["decision"]
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"reviewed:{Segment(message_id)}"),
            kind="reviewed",
            decision=decision,
        )
    )
    if decision == "approve":
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"committed:{Segment(message_id)}"), kind="committed")
        )
    return decision


def _domain():
    return MeteredInterpreter(
        llm=lambda _op: ({"amount": "5.00"}, Usage()), tools=lambda _op: "tool-done"
    )


def _rows(app: SqliteApp, run_id: str) -> list[tuple[str, str, int]]:
    from effective.sql import bind

    return list(
        app.conn.execute(
            *bind(
                t"SELECT event_id, kind, hypothetical FROM ledger "
                t"WHERE workflow_run_id={run_id} ORDER BY seq"
            )
        )
    )


def failure_of(app: SqliteApp, outcome: ForkOutcome) -> tuple[tuple[str, str], tuple[str, int]]:
    """A fork child's `Failed` answer, and the state and attempt its task ended on."""
    match outcome.answer:
        case Failed(error=error):
            state, attempt = app.conn.execute(
                "SELECT state, attempt FROM tasks "
                "WHERE json_extract(params, '$.child_run_id') = ?",
                (outcome.child_run_id,),
            ).fetchone()
            return error, (state, attempt)
    raise AssertionError(f"the fork child did not answer a failure: {outcome.answer!r}")


def _spawner(app: SqliteApp):
    """The engine-specific injection point.

    ONE engine fact lives here now: the SQLite spawn takes an `idempotency_key`, so a crash
    between the enqueue and the `Step`'s commit cannot enqueue a second child.

    Delivery is broadcast on both engines, so the done-event name is the whole address and the
    child needs no `reply_to`. The `idempotency_key`/`queue` difference is why the two spawners
    are two."""

    def spawn(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> UUID:
        return app.spawn(
            task_name, params, idempotency_key=idempotency_key, max_attempts=max_attempts
        )

    return spawn


def _register_base(app: SqliteApp, message_id: str) -> None:
    @app.register_task("base")
    def base_task(params, ctx):
        ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, _domain(), ledger=ledger).run(lambda: decision_wf(message_id))


def _register(app: SqliteApp, base_task_id: UUID, message_id: str) -> None:
    """Wire the fork child and the sweeping parent (the base has already run — its task id is
    what the children fork from)."""
    app.register_task("child")(
        partial(
            run_fork_as_task,
            # The child inherits the BASE's message id. `run_fork_as_task` hands the workflow the
            # CHILD's run id, so a workflow that used its argument as a scope would mint names
            # neither the seed nor the fork point match (see `decision_wf`). The fork's own
            # identity is carried by `child_run_id`, one layer down, where it belongs.
            workflow=lambda _child_run_id: decision_wf(message_id),
            domain=_domain(),
            hypothetical_ledger=lambda rid: SqliteLedger(
                app.conn, rid, app.write_lock, hypothetical=True
            ),
            read_base=lambda tid: read_sqlite_conn(app.conn, tid),
        )
    )

    @app.register_task("sweep")
    def sweep_task(params, ctx):
        def sweep():
            # The `src/` combinator, not a hand-rolled loop: spawn all N, then join all N.
            return (
                yield from marginal_sweep(
                    "child",
                    base_task_id=base_task_id,
                    through=f"ledger;extracted:{message_id}",
                    fork_point=compose_key(t"review:{Segment(message_id)}"),
                    forked_from="r-base",
                    forked_at_event=f"extracted:{message_id}",
                    deltas={cid: {"decision": d} for cid, d in DELTAS.items()},
                )
            )

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)


def _run_base(app: SqliteApp, message_id: str, decision: str = "reject") -> UUID:
    _register_base(app, message_id)
    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    # It PARKED — asserted, not assumed. A base that answered without ever registering a wake is
    # the M-2 failure (`test_fork_sweep_absurd.py`), and the seed a fork is about to take would
    # then be a prefix nobody ever paused at.
    assert [(p.task_name, p.wake_event) for p in read_sqlite_parked_conn(app.conn)] == [
        ("base", f"review:{message_id}")
    ]
    app.emit_event(f"review:{message_id}", {"decision": decision})
    snap = app.run_until_result(base_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == decision
    return base_id


def test_a_sweep_returns_one_marginal_per_delta_from_one_base(tmp_path, sqlite_app):
    """A sweep end to end: one base, three counterfactuals, three answers.

    Each child is its own task with its own checkpoints, its own attempts and its own
    hypothetical lineage; the base's canonical rows are untouched by all three. The parent never
    ran a counterfactual itself — it only holds N spawn checkpoints and N joins."""
    app = sqlite_app(str(tmp_path / "sweep.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)
    base_rows = _rows(app, "r-base")
    _register(app, base_id, message_id)

    sweep_id = app.spawn("sweep", {"run_id": "r-sweep"})
    snap = app.run_until_result(sweep_id, max_batches=256)

    assert snap is not None
    assert snap.state == "completed", snap
    outcomes = [ForkOutcome.model_validate(o) for o in snap.result]
    assert [o.child_run_id for o in outcomes] == list(DELTAS)
    assert [o.answer for o in outcomes] == [  # the marginals
        Returned(value="approve"),
        Returned(value="escalate"),
        Returned(value="reject"),
    ]

    # each child wrote its OWN hypothetical, sealed lineage: `sealed => valid marginal`
    for child_run_id, decision in DELTAS.items():
        kinds = [k for _, k, _ in _rows(app, child_run_id)]
        assert kinds[0] == "forked"
        assert kinds[-1] == "fork_sealed"
        assert ("committed" in kinds) == (decision == "approve")  # the divergence, per delta
        assert all(hyp == 1 for _, _, hyp in _rows(app, child_run_id))
        assert all(e.startswith(f"hyp:{child_run_id};") for e, _, _ in _rows(app, child_run_id))

    assert _rows(app, "r-base") == base_rows  # the canonical lineage never moved


def test_a_refused_child_answers_instead_of_hanging_its_parent(tmp_path, sqlite_app):
    """A refused fork must not hang: the liveness rule, one level up at the join edge.

    A fork whose `fork_point` never fires is refused by `run_fork`. If that refusal only raised,
    the child task would fail and the parent would wait on `fork-done:` forever: a sweep that
    silently never finishes. `run_fork_as_task` converts the typed refusal into a reported
    outcome and emits it, so the parent learns *which* child failed and *why*, and the sweep
    completes with a hole rather than a hang."""
    app = sqlite_app(str(tmp_path / "refused.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)
    _register(app, base_id, message_id)

    @app.register_task("bad-sweep")
    def bad_sweep(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-bad",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=Key.parse(
                    "never-fires"
                ),  # the workflow awaits review:{message_id}, not this
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},
            )
            return (yield from join_fork(handle))

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    sweep_id = app.spawn("bad-sweep", {"run_id": "r-bad"})
    snap = app.run_until_result(sweep_id, max_batches=256)

    assert snap is not None
    assert snap.state == "completed", snap  # the PARENT completed — no hang
    [(kind, message)] = refusals_of(ForkOutcome.model_validate(snap.result))
    assert kind == "ForkedPrefixAwait"  # the named refusal, relayed across the edge
    assert "never-fires" in message

    # and the refused lineage is NOT sealed, so no consumer reads its marginal
    assert [k for _, k, _ in _rows(app, "fork-bad")] == ["forked"]


def test_an_unjoined_spawn_is_a_detached_lineage_that_still_runs(tmp_path, sqlite_app):
    """The promotion path, which is why the two edges are unfused: spawn without joining and the
    child is a lineage of its own — it runs, diverges and seals while the parent finishes without
    waiting. This is the shape a fork takes when it stops being a probe and becomes a real,
    possibly human-in-the-loop session: no join edge, so nobody is blocked on it."""
    app = sqlite_app(str(tmp_path / "detached.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)
    _register(app, base_id, message_id)

    @app.register_task("detach")
    def detach_task(params, ctx):
        def detach():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-detached",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},
            )
            return handle.child_run_id  # ...and no `join_fork`

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(detach)

    parent_id = app.spawn("detach", {"run_id": "r-detach"})
    snap = app.run_until_result(parent_id, max_batches=64)
    assert snap is not None
    assert snap.state == "completed"  # the parent did not wait
    assert snap.result == "fork-detached"

    while app.work_batch():  # the child lives on: drain what is left claimable
        pass

    kinds = [k for _, k, _ in _rows(app, "fork-detached")]
    assert kinds == ["forked", "reviewed", "committed", "fork_sealed"]  # ran and sealed anyway


def test_ask_leaves_the_fork_point_open_for_a_human(tmp_path, sqlite_app):
    """`Ask()` is the promoted fork, and it is a REQUIRED, NAMED choice — not an omitted argument.

    With a delta the child pre-delivers its own answer; with `Ask()` it parks at its fork point
    exactly as the base did, in its own event namespace, and waits. Answering it by hand — the
    same gesture `just approve` makes on a canonical run — resumes it to a real marginal.

    The marker exists because inferring "ask a human" from a MISSING delta would turn a forgotten
    argument into a task parked forever on a question nobody was told to answer. Here the mistake
    is a `TypeError` instead, which is why the last assertion is part of the test."""

    from effective.fork import Ask
    from effective.handlers.absurd import fork_event_name

    app = sqlite_app(str(tmp_path / "ask.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)
    _register(app, base_id, message_id)

    @app.register_task("ask-sweep")
    def ask_sweep(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-hitl",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta=Ask(),  # no substitution — a human decides
            )
            return handle.child_run_id

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    parent_id = app.spawn("ask-sweep", {"run_id": "r-ask"})
    parent = app.run_until_result(parent_id, max_batches=64)
    assert parent is not None
    assert parent.result == "fork-hitl"
    while app.work_batch():  # let the child run as far as it can
        pass

    # it got as far as its OWN fork point and stopped there — genesis written, nothing decided
    assert [k for _, k, _ in _rows(app, "fork-hitl")] == ["forked"]
    # The child is found by the run id its params carry: that is the lineage's name, and the
    # name the answering emit is addressed under.
    (waiting_event,) = app.conn.execute(
        "SELECT waiting_event FROM tasks "
        "WHERE name=? AND json_extract(params, '$.child_run_id')=?",
        ("child", "fork-hitl"),
    ).fetchone()
    assert (
        waiting_event
        == fork_event_name("fork-hitl", compose_key(t"review:{Segment(message_id)}")).stored()
    )

    # a human answers, in the child's own namespace — and the counterfactual completes
    app.emit_event(waiting_event, {"decision": "approve"})
    while app.work_batch():
        pass
    assert [k for _, k, _ in _rows(app, "fork-hitl")] == [
        "forked",
        "reviewed",
        "committed",
        "fork_sealed",
    ]

    # and the mistake the marker exists to prevent is now a call-site error, not a parked task
    with pytest.raises(TypeError):
        spawn_fork(  # ty: ignore[missing-argument]  (the point of the test)
            "child",
            child_run_id="fork-oops",
            base_task_id=base_id,
            through=f"ledger;extracted:{message_id}",
            fork_point=compose_key(t"review:{Segment(message_id)}"),
            forked_from="r-base",
            forked_at_event=f"extracted:{message_id}",
        )


def test_the_workflow_argument_is_the_BASE_run_id_so_a_workflow_may_use_it(tmp_path, sqlite_app):
    """`run_fork_as_task` hands the workflow the base's run id, not the child's.

    The argument means "the run id the author sees", and authored names computed from it must
    match the seed the child was handed. Passing `forked_from` makes it fork-stable, so every
    child of one base sees exactly what the base saw.

    Pinned by CAPTURE rather than by outcome: a workflow that ignores the value cannot
    distinguish the two, so an outcome assertion would pass either way. The child's own identity
    is asserted unchanged alongside, because that is what must NOT move."""
    app = sqlite_app(str(tmp_path / "argid.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)
    seen: list[str] = []

    app.register_task("child")(
        partial(
            run_fork_as_task,
            workflow=lambda run_id: (seen.append(run_id), decision_wf(message_id))[1],
            domain=_domain(),
            hypothetical_ledger=lambda rid: SqliteLedger(
                app.conn, rid, app.write_lock, hypothetical=True
            ),
            read_base=lambda tid: read_sqlite_conn(app.conn, tid),
        )
    )

    @app.register_task("one-fork")
    def one_fork(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-argid",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},
            )
            return (yield from join_fork(handle))

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    snap = app.run_until_result(app.spawn("one-fork", {"run_id": "r-argid"}), max_batches=256)
    assert snap is not None
    assert snap.state == "completed", snap

    assert seen, "the child never ran, so this pins nothing"
    assert set(seen) == {"r-base"}  # the BASE's run id — fork-stable
    assert "fork-argid" not in seen  # ...and never the child's, which is what broke forkability

    # the child's OWN identity is untouched — it travels by the paths that need it
    assert next(k for _, k, _ in _rows(app, "fork-argid")) == "forked"


def _chaining_tail_wf(message_id: str):
    """`decision_wf`'s shape, but the TAIL is a long-lived CHAIN — `fork ∘ respawn`."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    yield from await_event(f"review:{message_id}", dict)

    def step(state: ChainCarry, _turn):
        return Again(ChainCarry(n=state.n + 1))
        yield  # pragma: no cover — generator marker

    return (
        yield from respawn(step, Chain(task="base", state=ChainCarry(), run_id="fork-approve"))
    )


def test_a_generation_boundary_inside_a_fork_is_refused_and_names_promotion(tmp_path, sqlite_app):
    """`fork ∘ respawn` is LOUD, and the refusal names PROMOTION.

    Everything that makes a counterfactual one is built per TASK by `run_fork`: the `ForkLedger`
    writing to a hypothetical lineage, the `RenamedAwaitCtx` event world, the `DryRun` sandbox,
    the `SeedingCtx` phase boundary. A respawn is *defined* as ending the task, and `_respawn`
    puts only the generation, the carry and the accrual on the wire — so generation n+1 would run
    as an ordinary registered task with none of the four: canonical ledger, unrenamed awaits
    (absorbing the base's answers), real world writes. The fork would also SEAL a lineage whose
    marginal does not exist, because a chain has no answer until its last generation.

    Before this it was guarded only by ACCIDENT — `DryRun` refused the spawn as a
    `WorldMutation`, a rule about world writes rather than counterfactual scope, whose message
    recommends adding `spawn` to `allow`, removing the guard. Measured: with that flag the
    composition proceeded. So this pin supplies the flag, to prove the refusal is the SUBSTRATE's
    and not the sandbox's.

    Asserted at the sweep level, because the property is that a refused child ANSWERS."""
    app = sqlite_app(str(tmp_path / "chainfork.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)

    app.register_task("child")(
        partial(
            run_fork_as_task,
            workflow=lambda _child_run_id: _chaining_tail_wf(message_id),
            domain=_domain(),
            hypothetical_ledger=lambda rid: SqliteLedger(
                app.conn, rid, app.write_lock, hypothetical=True
            ),
            read_base=lambda tid: read_sqlite_conn(app.conn, tid),
            # The sandbox exit its own message recommends — so what refuses below is the
            # composition rule, not `DryRun`.
            allow=frozenset({SPAWN_TOOL}),
        )
    )

    @app.register_task("chain-sweep")
    def sweep_task(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-approve",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},
            )
            return (yield from join_fork(handle))

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    sweep_id = app.spawn("chain-sweep", {"run_id": "r-chain"})
    snap = app.run_until_result(sweep_id, max_batches=256)

    assert snap is not None
    assert snap.state == "completed", snap  # the parent ANSWERS
    # A programming error, so the child answers `Failed` and fails once. Not a `WorldMutation`:
    # the sandbox was told to allow the spawn, so this is the substrate's composition rule.
    (kind, message), task = failure_of(app, ForkOutcome.model_validate(snap.result))
    assert kind == "CompositionRefused"
    assert "PROMOTION" in message, "the refusal must name what the author wanted"
    assert "respawn` outside, `fork` inside" in message  # the order that composes
    assert task == ("failed", 1)
    # And no generation escaped: nothing but the fork's own genesis, no second task.
    assert [k for _, k, _ in _rows(app, "fork-approve")] == ["forked"]
    assert list(read_sqlite_parked_conn(app.conn)) == []


def _nesting_wf(message_id: str):
    """`decision_wf`'s shape, but the TAIL spawns a counterfactual of its own and joins it.

    The composition the substrate designs for and refuses today: `spawn_fork`'s `budget` exists
    to bound "a counterfactual that spawns counterfactuals". Run as a fork child, its join is an
    ABSOLUTE await under `RenamedAwaitCtx`, which is the refusal under test."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    approval = yield from await_event(f"review:{message_id}", dict)
    handle = yield from spawn_fork(
        "child",
        child_run_id="grandchild",
        base_task_id=uuid4(),
        through=f"ledger;extracted:{message_id}",
        fork_point=compose_key(t"review:{Segment(message_id)}"),
        forked_from="fork-approve",
        forked_at_event=f"extracted:{message_id}",
        delta={"decision": "reject"},
    )
    outcome = yield from join_fork(handle)
    return f"{approval['decision']}/{outcome.child_run_id}"


def test_a_composition_refusal_in_a_child_answers_instead_of_hanging_the_sweep(
    tmp_path, sqlite_app
):
    """A fork child's spawn+join is refused, and the refusal must reach the parent as an answer.

    `run_fork_as_task` classifies what escapes a child into `REFUSALS` (a deterministic ANSWER,
    reported through `ForkOutcome` and emitted) versus a CRASH (retried to death, `done_event`
    never emitted). A refusal outside that tuple reads as a crash and the SWEEP parks forever on
    `fork-done:` (measured on both engines), with the orphan grandchild running to completion and
    every other child's marginal lost too, since `marginal_sweep` joins in order and never
    returns. That silent forever-park in the PARENT is exactly what `ForkOutcome`'s docstring
    forbids.

    **This is the class assertion.** Calling `run_fork` directly and asserting the CHILD fails
    terminally pins the instance. The invariant is that a refused child ANSWERS, which only
    `run_fork_as_task` + a joining parent can show. It is the twin of
    `test_a_refused_child_answers_instead_of_hanging_its_parent` above, and the reason
    `CompositionRefused` is one base type rather than a type per refusal site: the
    property is "a composition refusal is an answer", so the next one inherits the relay."""
    app = sqlite_app(str(tmp_path / "nested.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)

    app.register_task("child")(
        partial(
            run_fork_as_task,
            workflow=lambda _child_run_id: _nesting_wf(message_id),
            domain=_domain(),
            hypothetical_ledger=lambda rid: SqliteLedger(
                app.conn, rid, app.write_lock, hypothetical=True
            ),
            read_base=lambda tid: read_sqlite_conn(app.conn, tid),
            # `canned`, not `allow`: the grandchild's own execution is not what is under test —
            # the JOIN is — and canning the spawn keeps this pin to one moving part. That the
            # grandchild really is enqueued (and orphaned) is measured separately, on both
            # engines, and stated in the refusal's own message.
            canned={SPAWN_TOOL: SpawnResult(task_id=uuid4())},
        )
    )

    @app.register_task("nest-sweep")
    def sweep_task(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-approve",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},
            )
            return (yield from join_fork(handle))

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    sweep_id = app.spawn("nest-sweep", {"run_id": "r-nest"})
    snap = app.run_until_result(sweep_id, max_batches=256)

    assert snap is not None
    assert snap.state == "completed", snap  # the PARENT completed — no hang
    # A programming error, answered `Failed` across the edge, and the child fails once.
    (kind, message), task = failure_of(app, ForkOutcome.model_validate(snap.result))
    assert kind == "CompositionRefused"
    assert "ABSOLUTE event name" in message
    assert task == ("failed", 1)

    # Nothing is left parked — the liveness half, asserted positively rather than by the parent's
    # state alone (a task can be `completed` while a sibling hangs).
    assert list(read_sqlite_parked_conn(app.conn)) == []


def test_a_GOVERNED_denial_in_a_forked_tail_answers_the_parent(tmp_path, sqlite_app):
    """A `rules`-tier denial is a deterministic ANSWER, so the child must report it — not crash.

    **The invariant is that the PARENT gets an answer**, which is why that is what this asserts.
    Pinning "the child ends failed" would pin the instance. The expensive failure is a
    `done_event` never emitted: the engine retries the child to death (the policy is pure, so it
    re-derives the same denial every attempt) and the parent sits on `fork-done:` forever with
    `attempt=0`, no failure row, and nothing to read.

    `govern.Refused` meets `fork.REFUSALS`' stated criterion (`permission.rules` contractually
    forbids reading the clock, random or live state), so it belongs in the list.
    """
    from effective.govern import Refused
    from effective.permission import Allow, Deny, cascade, rules

    app = sqlite_app(str(tmp_path / "denied.db"))
    message_id = new_message_id()
    base_id = _run_base(app, message_id)

    def deny_the_commit(op):
        """Pure: a function of the op alone, so every retry re-derives it identically."""
        row = getattr(getattr(op, "row", None), "event_id", None)
        if row is not None and "committed" in row.stored():
            return Deny("committing is not permitted in this counterfactual")
        return Allow()

    app.register_task("child")(
        partial(
            run_fork_as_task,
            workflow=lambda _child_run_id: decision_wf(message_id),
            domain=_domain(),
            hypothetical_ledger=lambda rid: SqliteLedger(
                app.conn, rid, app.write_lock, hypothetical=True
            ),
            read_base=lambda tid: read_sqlite_conn(app.conn, tid),
            op_layers=[cascade([rules(deny_the_commit)])],
        )
    )

    @app.register_task("denied-sweep")
    def sweep_task(params, ctx):
        def sweep():
            handle = yield from spawn_fork(
                "child",
                child_run_id="fork-denied",
                base_task_id=base_id,
                through=f"ledger;extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                delta={"decision": "approve"},  # the approve path reaches the denied commit
            )
            return (yield from join_fork(handle))

        domain = MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(_spawner(app))}),
        )
        return DurableHandler(ctx, domain).run(sweep)

    snap = app.run_until_result(app.spawn("denied-sweep", {"run_id": "r-denied"}), max_batches=256)

    assert snap is not None
    assert snap.state == "completed", snap  # THE INVARIANT: the parent answers rather than hanging
    # A denial is an OUTCOME, and the parent must learn of it, and why.
    [(kind, message)] = refusals_of(ForkOutcome.model_validate(snap.result))
    assert kind == "Refused"
    assert "committing is not permitted" in message

    # ANTI-VACUITY: the layer really did deny, and `Refused` really is classified as an answer.
    from effective.fork import REFUSALS

    assert issubclass(Refused, REFUSALS)
    # and the counterfactual wrote nothing canonical — the denial did not cost the two-bookkeepers
    assert [r for r in _rows(app, "fork-denied") if r[2] == 0] == []
