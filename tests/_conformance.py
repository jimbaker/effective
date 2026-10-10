"""Shared cross-backend durable-conformance harness.

ONE self-contained workflow + domain + fault machinery, plus a backend adapter
per engine, so the same properties run against both the embedded SQLite engine
(0↔1, always) and Absurd/Postgres (0↔N, gated on a reachable PG). The point is
engine-independence: identical workflow, identical assertions, same
``DurableHandler`` + ``TaskContext`` contract; only the engine differs.

Not a test module (leading underscore). No ``psycopg``/``absurd_sdk``
at import time: the Absurd adapter imports them lazily inside its methods, so
SQLite-only runs (and ``test-core``) stay clean and never need the server.
"""

import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any, assert_never
from uuid import UUID, uuid4

from pydantic import BaseModel

from effective.api import (
    Effect,
    append_ledger,
    ask_llm,
    await_event,
    call_tool,
    gather,
    scoped,
    step,
)
from effective.budget import MeasuredBudget, OnExhaust
from effective.code import EXECUTE_TOOL, run_code
from effective.coding.states import (
    ReviewVerdict,
    State,
    Verdict,
)
from effective.coding.transition import transition
from effective.coding.verdicts import mechanical_judges
from effective.combinators import Answered, Deeper, descend, recurse
from effective.cost import CONTRACT_PARAM, Contract, MeteredInterpreter, Usage
from effective.domain import CallTool, DomainOp
from effective.fork import ForkOutcome
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Run, Segment, compose_key, scope_prefix
from effective.layers import TransientError
from effective.ledgerread import pg_payloads, sqlite_payloads
from effective.machine.evidence import CommandRun, Predicate
from effective.machine.outcomes import (
    Advance,
    Exhausted,
    Finish,
    Outcome,
    Park,
    ParkReason,
)
from effective.machine.spec import Ctx, Evidence, Report, StateSpec
from effective.machine.specs import build_specs, fuse
from effective.machine.trampoline import run_machine, stop_record
from effective.ops import Addressing, LedgerRow, Writer
from effective.spawning import Refusal, run_child
from effective.sqlite import SqliteApp, SqliteLedger, SqliteTaskContext

PG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective"
)
IMMEDIATE_RETRY = {"kind": "fixed", "base_seconds": 0}  # no backoff -> retry claimable at once


# ── the identities this fixture and its drivers share ────────────────────────
#
# **One minter, two consumers**: whoever emits asks this function, because a second speller
# here is a park that never wakes.
# The fixture composed these and `test_conformance.py` re-spelled them by f-string, so the two
# agreed only by everyone remembering the same separators.


def private(stem: str) -> str:
    """A task name no other test registers, since the Postgres queue is shared: `stem` and a fresh
    uuid, composed as a key."""
    return compose_key(t"private:{Segment(stem)},{Run(str(uuid4()))}").stored()


def review_name(run_id: str) -> Key:
    """The ADDRESS a conformance workflow's gate awaits — the name a driver must emit to."""
    return compose_key(t"review:{Segment(run_id)}")


def sub_scope(run_id: str) -> Key:
    """The subagent frame — a NAMING scope, so a projection keeps two subagents apart."""
    return compose_key(t"sub:{Segment(run_id)}")


def done_id(run_id: str) -> Key:
    """The ledger id a conformance run finishes on, and the one identity here that CANNOT be
    composed.

    Its shape is `{run_id}:done`: the run id sits in the TAG position, and `compose_key` refuses
    a template beginning with an interpolation, since an untyped leading hole is invisible to the
    one-tag-per-shape registry. So `Key.parse` is correct here, and this function exists so that
    the parse happens ONCE rather than at five call sites."""
    return Key.parse(f"{run_id}:done")


def ev_name(run_id: str) -> Key:
    """The bare branch-await address the gather fixtures park on."""
    return compose_key(t"ev:{Segment(run_id)}")


def scoped_by(scope: Key, name: Key) -> Key:
    """`name` as it reads INSIDE `scope` — the frame a handler applies, via the named exit.

    NOT `compose_key(t"rec:{i};{name}")`: a frame is not a coordinate of the scope's namespace,
    and composing it that way claims a two-hole shape under a tag that mints one — which
    `--key-borrowing` refuses, correctly. `Key.prefixed` is the one place a key is re-composed
    by a scope, and it is what `_PrefixedCtx` itself calls."""
    return name.prefixed(scope_prefix(scope))


def branched(g: int, i: int, name: Key) -> Key:
    """`name` as it reads from INSIDE gather branch `(g, i)` — the frame the handler applies.

    A splice rather than a prefix concatenation, so the branch coordinates are `int`-typed and
    the frame's shape lives here instead of at every assertion that quotes one."""
    # lint: terminal-hole — a `Key` splice
    return compose_key(t"gather:{g},{i};{name:domain=address}")


def refusals_of(outcome: ForkOutcome) -> list[tuple[str, str]]:
    """The refusals a fork child answered with, each as its type name and message."""
    match outcome.answer:
        case Refusal(refusals=refusals):
            return refusals
    raise AssertionError(f"the fork child did not answer a refusal: {outcome.answer!r}")


def _spawn_params(run_id: str, contract: Contract) -> dict:
    """Build a task's immutable spawn params, where the accrual contract rides.

    v0 carries NO key: an in-flight task spawned before the envelope existed looks exactly
    like this, so `Contract.from_params` returns v0 and the task stays on the bare path."""
    params = {"run_id": run_id}
    if contract is not Contract.V0:
        params[CONTRACT_PARAM] = contract.value
    return params


# ── the self-contained workflow + domain ─────────────────────────────────────
class CountingDomain:
    """Stable value per CallTool, recording each call (for exactly-once checks).

    ``delay`` makes the tool work take time (to show concurrent gather branches overlap);
    ``list.append`` is atomic under CPython, so concurrent calls don't corrupt ``calls`` —
    only its order is a race (assert ``sorted``).

    ``spans`` records each call's in-flight interval, which is how overlap is PROVEN rather
    than timed. A wall-clock bound on the whole run says "fast enough to have been concurrent",
    which is a claim about the machine: it read 1.975s against a 0.55s ceiling at load average
    15 and 0.49s alone, on unchanged code. Two intervals intersect or they do not, at any
    speed."""

    def __init__(
        self, flaky: bool = False, delay: float = 0.0, incrementing: bool = False
    ) -> None:
        self.calls: list[str] = []
        self.spans: list[tuple[str, float, float]] = []
        self.flaky = flaky  # raise TransientError on the FIRST 'a'
        self.delay = delay
        # incrementing: each call to a tool returns a *distinct* value, so a stale
        # checkpoint read (e.g. a key collision) is visible in the result, not just the count.
        self.incrementing = incrementing

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        entered = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        self.spans.append((op.name, entered, time.perf_counter()))
        self.calls.append(op.name)
        if self.flaky and op.name == "a" and self.calls.count("a") == 1:
            raise TransientError("flaky first 'a'")
        base = {"a": 10, "b": 20}[op.name]
        if self.incrementing:
            return base + self.calls.count(op.name)  # 11, 12, ... per repeated call
        return base  # retry-independent — result checks exactly-once


def two_step_wf(run_id: str):
    """4 ctx ops: tool:a, ledger;<run>:e1, tool:b, ledger;<run>:e2."""
    a = yield from call_tool("a", {}, int)
    yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:e1"), kind="k1"))
    b = yield from call_tool("b", {}, int)
    yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:e2"), kind="k2"))
    return {"a": a, "b": b}


def gated_wf(run_id: str):
    """Awaits a run-scoped event so the name correlates on either backend."""
    a = yield from call_tool("a", {}, int)
    decision = yield from await_event(review_name(run_id), dict)
    yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="done"))
    return {"a": a, "decision": decision}


def gather_wf(run_id: str):
    """Fan out two independent branches under `gather`; each does a tool step + a ledger
    append. On the durable path each branch's ops are checkpointed under `gather:{i}:`,
    so a crash mid-gather keeps committed branch-steps and re-runs only the rest. 4 ctx
    ops total; results join in branch order."""

    def branch(tool: str, kind: str):
        def thunk():
            v = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:{kind}"), kind=kind))
            return v

        return thunk

    results = yield from gather([branch("a", "ka"), branch("b", "kb")])
    return {"results": results}


def colliding_gather_wf(run_id: str):
    """`gather_wf` with the one difference under test: both branches author the SAME `event_id`
    instead of hand-disambiguating to `:ka`/`:kb`.

    The substrate places the two appends distinctly (`gather:0,0;ledger;<run>:done` vs
    `gather:0,1;…`), so the TAPE separates them; the ledger's `UNIQUE(event_id)` does not, and
    the second row is dropped by `ON CONFLICT DO NOTHING`. 4 ctx ops."""

    def branch(tool: str):
        def thunk():
            v = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="done"))
            return v

        return thunk

    results = yield from gather([branch("a"), branch("b")])
    return {"results": results}


def sequential_collision_wf(run_id: str):
    """The same collision with NO gather anywhere — two appends of one authored id, in sequence.

    This is what makes the family *placed-writer collision* rather than a gather defect: the
    engine's duplicate-name rule gives the second append its own checkpoint (`…;done#2`), so the
    two writers are distinct on the tape and identical on the canonical record. 2 ctx ops."""
    yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="first"))
    yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="second"))
    return {"appends": 2}


def scoped_wf(run_id: str):
    """Two same-shaped bodies under DIFFERENT scopes: the structural `scoped(...)`.

    Each body calls the same tool under the same bare name, so without the handler-applied
    prefix the second would re-bind the first's checkpoint and return a stale result. The keys
    are `rec:0;tool:a` and `rec:1;tool:a`; nothing in the workflow spells either.

    Also pins that a scope's namespacing reaches the LEDGER arm, not just steps: each body
    appends under a bare `{run_id}:e`, and the two rows can only coexist because the checkpoint
    keys differ (`ledger;` ids are the author's, but the append's checkpoint is scoped)."""

    def body(i: int):
        def thunk():
            v = yield from call_tool("a", {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:e{i}"), kind=f"k{i}"))
            return v

        return thunk

    first = yield from scoped(compose_key(t"rec:{0}"), body(0))
    second = yield from scoped(compose_key(t"rec:{1}"), body(1))
    return {"first": first, "second": second}


def scoped_await_wf(run_id: str):
    """A durable await INSIDE a scope: the engine parks on the SCOPED event name
    (`rec:0;review:{run_id}`), which is the whole point — the author writes the bare name and
    the handler namespaces it, so two scopes awaiting the same bare name are two questions.

    The park/resume path is where the two engines could diverge (an Absurd park writes a
    pending-await marker with a NULL payload; SQLite uses `tasks.waiting_event`), so this runs
    on both."""

    def body():
        decision = yield from await_event(review_name(run_id), dict)
        yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="done"))
        return decision

    result = yield from scoped(compose_key(t"rec:{0}"), body)
    return {"decision": result}


def scoped_nested_gather_await_wf(run_id: str):
    """gather → scoped → gather → await: the deepest await composition, and the one the suite
    did not have.

    A park travels outward as a VALUE (`_GatherPark` → `BranchParked`), so every frame it crosses
    must re-add its own prefix: both `_branch_await` and `_join`'s NESTED-gather re-raise carry
    the scope path. Drop it from either and this shape parks on
    `gather:0,0;gather:0,0;ev:{run_id}` while the recorder and `qualified_event_name` both say
    `gather:0,0;s:0;gather:0,0;ev:{run_id}`, and no emitter following the documented contract can
    wake the run.

    The assertion that matters is that the ENGINE agrees with `qualified_event_name`, so the test
    composes the expected name rather than spelling it."""

    def innermost():
        def thunk():
            return (yield from await_event(ev_name(run_id), dict))

        return thunk

    def scoped_branch():
        def thunk():
            return (yield from scoped(compose_key(t"s:{0}"), lambda: gather([innermost()])))

        return thunk

    results = yield from gather([scoped_branch()])
    return {"results": results}


def recurse_fold_park_wf(run_id: str):
    """An op and a park inside `recurse`'s COMBINE: the fold frames, on a real engine.

    `combinators.recurse` merges groups under `gather:{g},{k};fold:{level},{k}`.
    `descend_recurse_wf` runs a fold gather, but its `combine` is pure, so its fold frames never
    reach a checkpoint key or a park name; this is the composition table's `recurse@fold` member
    on a durable engine.

    Note the coordinate the engine must produce: `gather:1:0`, not `gather:0:0`. `recurse` issues
    its leaf gather first and its fold gather second at the SAME level, so the ordinal is the
    only thing distinguishing two gathers a workflow never named, which is the job of the `{g}`
    discriminator."""

    def decompose(_ctx):
        return ["a", "b"]  # two chunks at fanin 2 -> exactly one fold level runs
        yield  # pragma: no cover — generator marker

    def leaf(chunk):
        return (yield from step(f"leaf-{chunk}", CallTool(name="a", args={}, result_schema=int)))

    def combine(group):
        yield from step("merge", CallTool(name="a", args={}, result_schema=int))
        answer = yield from await_event(ev_name(run_id), dict)
        return sum(group) + answer["extra"]

    total = yield from recurse("ctx", decompose, leaf, combine, fanin=2)
    return {"total": total}


def gather_await_wf(run_id: str):
    """A gather branch that parks on a durable await and joins its payload (the
    await-in-gather park). The round completes (the
    sibling's tool step commits), the whole task parks on the branch's PREFIXED
    event (``gather:0,1;ev:{run_id}``), and on wake the replay re-binds the
    committed sibling and delivers the payload by name."""

    def tool_branch():
        def thunk():
            return (yield from call_tool("a", {}, int))

        return thunk

    def await_branch():
        def thunk():
            return (yield from await_event(ev_name(run_id), dict))

        return thunk

    results = yield from gather([tool_branch(), await_branch()])
    return {"results": results}


def descend_recurse_wf(run_id: str):
    """``recurse`` whose leaf DRILLS, the recursive-language-model shape.

    The 'easy' chunk answers at d0; the 'hard' chunk exhausts a budget-1 ``descend`` and parks
    on the gather-qualified, run-scoped grant ``gather:0,1;rec:1;depth-grant:{run_id}:1``. Pins
    the composition contract: the ``{g}:{i}`` half comes from the ctx prefix (invisible at the
    descend call site), the run-scoping half from the caller, and the two rules are orthogonal."""

    def decompose(ctx):
        return ["easy", "hard"]
        yield  # pragma: no cover — generator marker

    def judge_for(chunk):
        def judge(current, level):
            v = yield from step("judge", CallTool(name="a", args={}, result_schema=int))
            if chunk == "hard" and not level.final:
                return Deeper(current)  # drills until a final level forces the answer
            return Answered(v + level.depth)

        return judge

    def leaf(chunk):
        return (yield from descend(chunk, judge_for(chunk), budget=1, run_id=run_id))

    def combine(group):
        return sum(group)
        yield  # pragma: no cover — generator marker

    total = yield from recurse("ctx", decompose, leaf, combine)
    return {"total": total}


def wake_race_collision_wf(run_id: str):
    """A raced branch whose author step is named exactly ``wake-race``: the
    aliasing probe for the repark checkpoint name. Were the repark minted as
    ``gather:{g},{i};wake-race``, run 2's ``begin_step`` would read the repark's
    wake-time checkpoint as the step's value: silent wrong data, the tool never
    executed, the engines diverging. The step name is legal: op_key rejects only
    reserved arm tags at the top level."""
    from effective.api import step

    def racing_branch():
        def thunk():
            payload = yield from await_event(ev_name(run_id), dict)
            v = yield from step("wake-race", CallTool(name="a", args={}, result_schema=int))
            return [payload, v]

        return thunk

    results = yield from gather([racing_branch()])
    return {"results": results}


def double_await_wf(run_id: str):
    """One branch with TWO sequential awaits, each raced mid-round, so the same
    gather hits the wake race twice: 'at most once' holds per await, not per
    branch. The repark
    name must be fresh per wake CONDITION or the second race finds the first's
    stale checkpoint and burns an attempt via the fallback."""

    def double_branch():
        def thunk():
            first = yield from await_event(f"ev1:{run_id}", dict)
            second = yield from await_event(f"ev2:{run_id}", dict)
            return [first, second]

        return thunk

    def tool_branch():
        def thunk():
            return (yield from call_tool("a", {}, int))

        return thunk

    results = yield from gather([double_branch(), tool_branch()])
    return {"results": results}


def multi_park_wf(run_id: str):
    """THREE branches: two park on DIFFERENT events (indices 0 and 2), one runs a
    tool (index 1). Pins the serialized-wake contract: the task parks on the
    LOWEST parked branch's event; an out-of-order emission of branch 2's event
    does not wake it; branch 0's emission does — and the wake's replay resolves
    branch 2 from the already-delivered event without a further park."""

    def await_branch(i: int):
        def thunk():
            return (yield from await_event(f"ev{i}:{run_id}", dict))

        return thunk

    def tool_branch():
        def thunk():
            return (yield from call_tool("a", {}, int))

        return thunk

    results = yield from gather([await_branch(0), tool_branch(), await_branch(2)])
    return {"results": results}


def two_parking_gathers_wf(run_id: str):
    """Two same-shaped gathers, EACH with a parking branch awaiting the same bare
    event name. Injectivity of the ``{g}`` discriminator on EVENT names: the two
    parks are ``gather:0,1;ev:{run}`` then ``gather:1,1;ev:{run}`` — distinct
    events needing distinct emissions, each payload landing in its own gather."""

    def tool_branch():
        def thunk():
            return (yield from call_tool("a", {}, int))

        return thunk

    def await_branch():
        def thunk():
            return (yield from await_event(ev_name(run_id), dict))

        return thunk

    g1 = yield from gather([tool_branch(), await_branch()])
    g2 = yield from gather([tool_branch(), await_branch()])
    return {"g1": g1, "g2": g2}


def nested_gather_park_wf(run_id: str):
    """A park inside a NESTED gather: outer branch 0 is itself a gather whose
    branch 0 awaits. The park's event name composes the full path —
    ``gather:0,0;gather:0,0;ev:{run_id}`` — injectivity under composition."""

    def inner_await():
        def thunk():
            return (yield from await_event(ev_name(run_id), dict))

        return thunk

    def outer_gathering():
        def thunk():
            inner = yield from gather([inner_await()])
            return inner[0]

        return thunk

    def tool_branch():
        def thunk():
            return (yield from call_tool("b", {}, int))

        return thunk

    results = yield from gather([outer_gathering(), tool_branch()])
    return {"results": results}


def make_gather_sleep_wf(wake_at):
    """A gather branch that sleeps durably until ``wake_at`` (decided by the TEST,
    not the workflow — a clock read between yields would breach the determinism
    boundary). The sibling's tool step commits in the same round; the task parks
    'sleeping' and completes once the deadline passes."""

    def gather_sleep_wf(run_id: str):
        from effective.api import sleep_until

        def sleeping_branch():
            def thunk():
                yield from sleep_until(wake_at)
                return "woke"

            return thunk

        def tool_branch():
            def thunk():
                return (yield from call_tool("a", {}, int))

            return thunk

        results = yield from gather([sleeping_branch(), tool_branch()])
        return {"results": results}

    return gather_sleep_wf


def make_top_level_sleep_wf(wake_at):
    """A durable sleep at the TASK ROOT: the path `make_gather_sleep_wf` above does not take.

    A top-level `sleep_until` reaches `ctx.sleep_until`, while a sleep inside a gather branch
    routes to `DurableHandler._branch_sleep`, a pure clock compare that calls neither ctx. So a
    defect in `SdkCtx.sleep_until` (forwarding one argument where the SDK wants two) makes every
    top-level durable sleep on Absurd raise `TypeError` and retry to death while every
    branch-sleep test stays green. `sleep ∘ gather` and `sleep` alone are separate rows of the
    composition table."""

    def top_level_sleep_wf(run_id: str):
        from effective.api import sleep_until

        before = yield from call_tool("a", {}, int)
        yield from sleep_until(wake_at)
        after = yield from call_tool("b", {}, int)
        return {"before": before, "after": after}

    return top_level_sleep_wf


def scoped_absolute_await_wf(run_id: str):
    """An ABSOLUTE await inside a `scoped(...)`: the frame is NOT prepended, on both engines.

    A scope completes RELATIVE names. An absolute one is already whole, and its emitter is a
    different task that has never seen this scope, so the handler resolves the await at the ctx
    it was constructed with. `budget-grant` resolves at `_root_ctx` for the same reason, and the
    rule is a property of the NAME, so no call site wires it.

    The test emits the BARE name; if the reroute regressed, the task would park forever on
    `s:0;{name}` and the drain would time out rather than complete."""

    def body():
        return (yield from await_event(sub_scope(run_id), dict, addressing=Addressing.ABSOLUTE))

    decision = yield from scoped(compose_key(t"s:{0}"), body)
    return {"decision": decision}


def branch_absolute_await_wf(run_id: str):
    """The same absolute await inside a gather BRANCH — REFUSED, on both engines.

    A branch coordinate is not a naming choice; it is a concurrency slot, and a branch await
    peeks and parks as a value the barrier re-arms with the coordinate re-added. There is
    nothing to reroute to, so this refuses and names the fix: a cross-task fan-in wants a
    loop."""

    def branch():
        return (yield from await_event(sub_scope(run_id), dict, addressing=Addressing.ABSOLUTE))

    results = yield from gather([branch])
    return {"results": results}


def two_gathers_wf(run_id: str):
    """Two *same-shaped* gathers in one workflow — same branch count, same inner step names.
    Without a per-gather discriminator in the durable key, the second gather would read the
    first's checkpoints and return stale results (the regression this pins). Each branch's
    tool returns a value that increments per call, so a stale read is visible: a correct run
    makes 4 distinct calls and g2 != g1."""

    def branch(tool: str):
        def thunk():
            return (yield from call_tool(tool, {}, int))

        return thunk

    g1 = yield from gather([branch("a"), branch("b")])
    g2 = yield from gather([branch("a"), branch("b")])
    return {"g1": g1, "g2": g2}


# ── measured-spend accrual: the usage envelope + trip on the durable path ──
class MeteringDomain:
    """A domain with AskLLM *usage*, for the measured-trip conformance. Each AskLLM costs
    a fixed ``cost`` and records one LIVE call — replay re-binds a committed step without
    touching the domain, so ``len(calls)`` is the *real* spend (x cost) and never the
    replayed prefix. Wraps a ``MeteredInterpreter`` so it exposes both ``run`` (v0/CallTool)
    and ``run_metered`` (the v1 usage-in-checkpoint seam)."""

    def __init__(self, cost: float = 0.001) -> None:
        self.cost = cost
        self.calls: list[str] = []

        def llm(_op: Any) -> tuple[str, Usage]:
            self.calls.append("ask")  # list.append is atomic under CPython (concurrent-safe)
            return "ans", Usage(prompt_tokens=10, completion_tokens=5, cost=self.cost)

        def tools(_op: Any) -> Any:
            raise AssertionError("metered conformance workflows yield no CallTool")

        self._interp = MeteredInterpreter(llm=llm, tools=tools)

    def run(self, op: DomainOp) -> Any:
        return self._interp.run(op)

    def run_metered(self, op: DomainOp) -> tuple[Any, Usage]:
        return self._interp.run_metered(op)


class FlakyMeteringDomain(MeteringDomain):
    """A metered domain whose first ``fail_first`` AskLLM attempts raise ``TransientError``.

    The flaky metered *service* behind a durable both-engine row: `measured_drive` is pinned
    in-process, and this covers the `DurableHandler` v1 arm under a `serve(retry_domain(...))`
    stack, the arm production runs."""

    def __init__(self, cost: float = 0.001, fail_first: int = 1) -> None:
        super().__init__(cost)
        self.fail_first = fail_first
        self.failures = 0

    def run_metered(self, op: DomainOp) -> tuple[Any, Usage]:
        if self.failures < self.fail_first:
            self.failures += 1
            raise TransientError(f"flaky metered call {self.failures}")
        return super().run_metered(op)


def metered_trip_wf(run_id: str):
    """3 sequential asks; with a ceiling between 2x and 3x the per-ask cost, the 3rd trips."""
    a = yield from ask_llm("m1", "x", str)
    b = yield from ask_llm("m2", "x", str)
    c = yield from ask_llm("m3", "x", str)
    return {"asks": [a, b, c]}


def scoped_metered_trip_wf(run_id: str):
    """``metered_trip_wf`` inside a ``scoped(...)`` — the measured trip's park must take NO scope
    frame.

    `budget-grant:{run_id},{trip_n}` addresses the RUN, not a place in it: the trip point is
    spend-dependent, so a frame in the name would vary run to run for identical code and no
    emitter could compute it. Awaiting through the scoped ctx made it `rec:0;budget-grant:…`,
    which nothing emits — the run parked unwakeably.

    Contrast an `approve;{…}:{op_key}` park, which stays frame-qualified because it names an OP
    OCCURRENCE and the frame is what stops one emission approving a sibling."""

    def body():
        return (yield from metered_trip_wf(run_id))

    result = yield from scoped(compose_key(t"rec:{0}"), body)
    return result


def metered_gather_trip_wf(run_id: str):
    """A CONCURRENT gather of two spending branches, then a sequential ask that trips because
    the folded branch spend + this ask crosses the ceiling: the *concurrent* pin. The
    per-branch subtotals aggregate confluently, so the trip fires deterministically regardless
    of branch completion order."""

    def branch(name: str):
        def thunk():
            return (yield from ask_llm(name, "x", str))

        return thunk

    pair = yield from gather([branch("g0"), branch("g1")])
    tail = yield from ask_llm("tail", "x", str)
    return {"pair": pair, "tail": tail}


# ── run_code: the segment machine on the durable path ─────────────
class Approval(BaseModel):
    """The reviewer's ruling for the `human` cascade tier (duck-typed fields)."""

    decision: str
    rationale: str = ""


class CodeActionDomain:
    """Domain for the run_code conformance cases: the reserved execute tool runs a
    real ``MontyEngine`` whose one function records its LIVE calls (a re-bound call
    never lands here — the within-run replay-by-re-execution proof), and the
    ``send_email`` action records its calls and returns an *incrementing* ack,
    so a doubled or collided action is visible in the result, not just the count."""

    def __init__(self) -> None:
        from effective.monty import MontyEngine, execute_tool

        self.fn_calls: list[str] = []
        self.action_calls: list[Any] = []

        def llm_query(prompt: str) -> str:
            self.fn_calls.append(prompt)
            return f"summary({prompt})"

        self._execute = execute_tool(MontyEngine(functions={"llm_query": llm_query}))

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        if op.name == EXECUTE_TOOL:
            return self._execute(op)
        assert op.name == "send_email"
        self.action_calls.append(op.args)
        return {"id": f"msg-{len(self.action_calls)}"}


# One code body serves every case: the action result lands in the output, and a
# cascade DENIAL re-enters as PermissionError the code routes around in-sandbox.
RUN_CODE_BODY = (
    "s = llm_query(topic)\n"
    "try:\n"
    "    ack = send_email('approver@example.com', s)\n"
    "except PermissionError as e:\n"
    "    ack = f'denied: {e}'\n"
    "{'summary': s, 'ack': ack}"
)


def _one_run_code(topic: str):
    return run_code(
        "c",
        RUN_CODE_BODY,
        schema=dict,
        inputs={"topic": topic},
        functions=("llm_query",),
        actions={"send_email": dict},
    )


def run_code_wf(run_id: str):
    """4 ctx ops: code:seg,0,c (runs llm_query live, pauses at the action),
    code:action,0,c;tool:send_email, code:seg,1,c (llm_query re-binds from fn_log,
    completes), ledger;<run>:done."""
    out = yield from _one_run_code(run_id)
    yield from append_ledger(LedgerRow(event_id=done_id(run_id), kind="code-done"))
    return out


def duplicate_run_code_wf(run_id: str):
    """Two SAME-NAMED run_code calls with DIFFERENT code. Duplicate step names
    occurrence-suffix (`name#2`) on the real Absurd SDK and on the SQLite ctx
    alike; without the suffix SQLite would silently serve the first call's
    checkpoints to the second and the engines would diverge. Unique
    names remain the convention; the suffix is the safety net."""
    first = yield from run_code(
        "c", "{'who': llm_query('first')}", schema=dict, functions=("llm_query",)
    )
    second = yield from run_code(
        "c", "{'who': llm_query('second')}", schema=dict, functions=("llm_query",)
    )
    return {"first": first, "second": second}


def gather_run_code_wf(run_id: str):
    """Two gather branches each run_code with the SAME name ('c') — only the
    ``gather:{g},{i};`` ctx prefix keeps their ``code:seg,{j},c`` and
    ``code:action,{j},c;tool:{tool}`` keys disjoint (injectivity under composition). The
    incrementing action ack makes a collided/stale read visible in the results."""

    def branch(topic: str):
        def thunk():
            return (yield from _one_run_code(topic))

        return thunk

    results = yield from gather([branch(f"{run_id}-b0"), branch(f"{run_id}-b1")])
    return {"results": results}


# ── the coding machine (`effective.coding`) on a durable engine ───────────────
#
# A second-engine port is verified on the REAL engine, not only under
# `RecordingHandler`/`ReplayHandler`. The machine belongs HERE rather than in
# `tests/_walks.py` for the reason that file's own docstring gives: `_walks._Tool` answers
# `STEP_RESULT = "v"` for every step whatever its schema, and the machine's postamble yields
# `call_tool(SUITE_TOOL, …, CommandRun)` and reads `run.green` — *"a probe that needs a step to
# return something particular is asking a domain question and belongs in `_conformance`."*

MACHINE_GOAL = "make the target test pass"
MACHINE_TARGET = "test_add"
MACHINE_MODULE = "mod.py"
BROKEN_SOURCE = "def add(a, b):\n    return a - b\n"
FIXED_SOURCE = "def add(a, b):\n    return a + b\n"
MACHINE_TREE = {MACHINE_MODULE: BROKEN_SOURCE}

RED_SUITE = CommandRun(exit_code=1, failures=(f"tests/test_mod.py::{MACHINE_TARGET}",))
GREEN_SUITE = CommandRun(exit_code=0)


class CodingSuiteDomain:
    """Serves the machine's tools with CANNED measurements.

    The real predicate (`coding.runners.run_suite`) shells out to pytest, and a durable case that
    spawned a subprocess per checkpoint would be measuring the fixture rather than the engines.
    What this file is for is whether the same op stream commits identically on both — so the
    measurement is canned and the WALK is real: red until the edit lands, green after.

    **`TEST -> DRAFT` is a FORWARD edge.** The path is `test, draft, finalize, review`, four
    DISTINCT states, so nothing here re-enters a `state:` frame, the one case `visit:` exists to
    disambiguate; `StubbornSuiteDomain` below is the run that re-enters.

    The state is handler-side and survives a crash, exactly as a real workspace would, which is
    what makes `calls` a proof: if a resume wrongly re-ran a committed step, `apply_fix` would
    appear twice."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fixed = False

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        self.calls.append(op.name)
        match op.name:
            case "run_suite":
                return GREEN_SUITE if self.fixed else RED_SUITE
            case "read_file":
                return FIXED_SOURCE if self.fixed else BROKEN_SOURCE
            case "apply_fix":
                self.fixed = True
                return {MACHINE_MODULE: FIXED_SOURCE}
            case _:
                raise KeyError(f"CodingSuiteDomain: nothing serves {op.name!r}")


class StubbornSuiteDomain(CodingSuiteDomain):
    """The same fixture, except the first fix does not take, so DRAFT is entered TWICE.

    `route_draft` maps `STILL_RED` to `Advance(State.DRAFT)`, a self-edge, so one extra red suite
    is the whole difference, and it is the only shape in this fixture that re-enters a `state:`
    frame, the case `visit:` exists to keep injective. Without the
    counter the two DRAFT visits mint the same key, because a repeated scope frame gets no
    occurrence suffix on the in-memory tape."""

    def __init__(self, fixes_needed: int = 2) -> None:
        super().__init__()
        self.fixes_needed = fixes_needed
        self.fixes = 0

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        if op.name == "apply_fix":
            self.fixes += 1
            self.fixed = self.fixes >= self.fixes_needed
            self.calls.append(op.name)
            return {MACHINE_MODULE: FIXED_SOURCE if self.fixed else BROKEN_SOURCE}
        return super().run(op)


def _measuring_worker(ctx: Ctx) -> Effect[Evidence]:
    """Run the predicate and hand the measurement forward — TEST, FINALIZE and REVIEW all do
    exactly this, since none of them needs to change the tree to reach a verdict."""
    run: CommandRun = yield from call_tool("run_suite", {}, CommandRun)
    return Evidence(summary=f"{ctx.state.value}: exit {run.exit_code}", measured=run)


def _editing_worker(ctx: Ctx) -> Effect[Evidence]:
    """DRAFT: read, edit, re-measure — and RETURN the tree.

    The return is the load-bearing part on a durable engine. A workflow may only learn about a
    handler-side change through a recorded op result, because a resume re-executes and no tool
    runs for a committed step; a machine that read a shared workspace instead re-derived a
    different artifact id at the postamble."""
    yield from call_tool("read_file", {}, str)
    tree: dict[str, str] = yield from call_tool("apply_fix", {}, dict)
    run: CommandRun = yield from call_tool("run_suite", {}, CommandRun)
    return Evidence(summary=f"{ctx.state.value}: exit {run.exit_code}", measured=run, tree=tree)


def _unreachable_worker(ctx: Ctx) -> Effect[Evidence]:
    """PLAN, EXPLORE and REPL have no work in this fixture. They RAISE rather than returning
    something harmless: a state that quietly did nothing would let a wrong walk pass, and the
    walk is half of what this case measures."""
    raise AssertionError(f"{ctx.state.value} should not run in the conformance walk")
    yield  # pragma: no cover  -- generator, as `Worker` requires


def _judge_review(_ctx: Ctx, _evidence: Evidence) -> Effect[ReviewVerdict]:
    return ReviewVerdict.APPROVED
    yield  # pragma: no cover


def _unreachable_judge(_ctx: Ctx, _evidence: Evidence) -> Effect[Verdict]:
    raise AssertionError("no judge should run for a deferred state")
    yield  # pragma: no cover


def machine_specs() -> dict[State, StateSpec]:
    """The four states this walk reaches, assembled by the SUBSTRATE's builder.

    `build_specs` iterates `State`, so this map is total without the fixture saying so, and the
    three mechanical judges come from `mechanical_judges` rather than being wired here — the
    wiring was identical to the fixture benchmark's, which is exactly the duplication
    `effective.coding.specs` exists to end. Scripting the judges would make the walk a story the
    fixture tells about itself; computing them from the (canned) measurement means the path is
    DISCOVERED, so a wrong resume shows up as a different path rather than a narrated success.

    `canonical=frozenset()` overrides the substrate default deliberately: this walk asserts the
    postamble's two rows and nothing else, so no state here declares a ledger reach."""
    return build_specs(
        State,
        workers={
            State.TEST: _measuring_worker,
            State.DRAFT: _editing_worker,
            State.FINALIZE: _measuring_worker,
            State.REVIEW: _measuring_worker,
        },
        judges=mechanical_judges(MACHINE_TARGET, debt_remains=False)
        | {State.REVIEW: _judge_review},
        default_worker=_unreachable_worker,
        default_judge=_unreachable_judge,
        canonical=frozenset(),
    )


def coding_machine_wf(run_id: str):
    """10 ctx ops: 1 (TEST) + 3 (DRAFT) + 1 (FINALIZE) + 1 (REVIEW) + 4 postamble
    (artifact, predicate, `ledger;machine:<run>;commit`, `ledger;machine:<run>`).

    Returns a PROJECTION rather than the `Session`, and that is required rather than tidy:
    `DurableHandler._run` calls `_dump` exactly once at the task boundary, so a durable run hands
    back JSON — asserting against a dumped `Finish | Park` or `Verdict | Exhausted` would be
    asserting against a serializer. What the projection keeps is what the case is about: the
    PATH (the trajectory, not just the terminal state), the content-addressed artifact id, and
    the predicate's word."""
    session = yield from run_machine(
        # Promoted HERE, at the task boundary: spawn params arrive as `str`, so this is the
        # one place a malformed run id can be refused before the machine performs an op.
        Run(run_id),
        MACHINE_GOAL,
        machine_specs(),
        transition,
        start=State.TEST,
        budget=8,
        tree=MACHINE_TREE,
    )
    commitment = session.commitment
    assert commitment is not None, "the postamble is unconditional; a Session always carries one"
    return {
        "path": [state.value for state in session.path],
        "stopped": stop_record(session.stopped).kind,
        "artifact_id": commitment.artifact_id,
        "passed": commitment.passed,
        "files": list(commitment.files),
    }


# ── the ruling machine: a judge that YIELDS, fused and split ─────────────────
#
# Shared by `tests/test_machine_fuse_or_split.py` and the durable conformance lane. These are plain
# callables, not pytest fixtures, so they live in the module the lane already imports and pytest
# does not collect, rather than in `conftest`.
#
# **This is the only embodiment in the tree whose judges YIELD.** Every judge in
# `machine_specs()` above, in `tests/test_coding_*` and in the other embodiments is
# `return X; yield`, which mints no op and no frame, so every assertion about a judge's position on
# the tape holds vacuously of an empty set unless it runs against this walk. The anti-emptiness
# guards in the drivers are not decoration; they are why this exists.

RULING_GOAL = "both forms, one machine"
RULING_NOTE = "ruling.txt"
GATE_TOOL = "gate"
GATE_PREDICATE = Predicate(GATE_TOOL, CommandRun)


class RulingState(StrEnum):
    WORK = "work"
    REVIEW = "review"
    RULE = "rule"


class WorkVerdict(StrEnum):
    DONE = "done"


class GatherVerdict(StrEnum):
    """REVIEW's fibre is a SINGLETON, which is what a worker-state's fibre looks like once its
    judgement moves out. `ty` still narrows it to `Never`, so a dropped arm is still a type error —
    the dependent sum does not care that this fibre has one member."""

    GATHERED = "gathered"


class RuleVerdict(StrEnum):
    APPROVED = "approved"
    BREACH = "breach"


type RulingVerdict = WorkVerdict | GatherVerdict | RuleVerdict


def route_ruling(
    state: RulingState, verdict: RulingVerdict | Exhausted[RulingState]
) -> Outcome[RulingState]:
    match verdict:
        case Exhausted():
            return Park(verdict.state, ParkReason.EXHAUSTED)
        case WorkVerdict.DONE:
            return Advance(RulingState.REVIEW)
        case GatherVerdict.GATHERED:
            return Advance(RulingState.RULE)
        case RuleVerdict.BREACH:
            return Advance(RulingState.WORK)
        case RuleVerdict.APPROVED:
            return Finish()
        case unreachable:
            assert_never(unreachable)


class RulingDeployment:
    """A gate that is green from the start, so the SUITE never drives the walk — the verdicts do.

    `rulings` is what the model-free stand-in for a policy judge reads: the first RULE visit finds
    a breach, the second does not, so the machine takes the back edge exactly once. It is
    handler-side state, so it survives a crash exactly as `CodingSuiteDomain.fixed` does — which is
    what makes `saw` an exactly-once proof under the sweep rather than a tautology, the role
    `domain.calls` plays for the coding machine."""

    def __init__(self, rulings: list[str] | None = None) -> None:
        self.gate_calls = 0
        self.rulings: list[str] = ["breach", "approved"] if rulings is None else list(rulings)
        self.saw: list[str] = []
        self.carried_outputs: list[str] = []

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        match op.name:
            case "gate":
                self.gate_calls += 1
                # STAMPED from the WALK, not from this counter, and the difference is a finding.
                # A constant output would let a state read any `Report` and still agree, so the
                # stamp has to say which measurement it is — but an earlier version stamped
                # `#{self.gate_calls}`, which is instance state on an object a FRESH WORKER does
                # not have. Measured cross-process (two `uv run python` processes over one
                # file-backed SQLite): a perfectly correct resume renumbered the stamps and the
                # assertion reddened with nothing wrong. `at` arrives from the workflow, so it
                # re-derives from replayed results and reads the same in any process.
                at = op.args.get("at")
                where = "commit" if at is None else f"visit {at}"
                return CommandRun(exit_code=0, failures=(), output=f"clean at {where}")
            case "ruling":
                # ASSERTED, not decorative. RULE passes its predecessor's summary through
                # `ctx.incoming.summary`; refusing an empty one is what makes a dropped
                # `summary=` in the lift redden a test instead of passing the whole suite.
                saw = op.args.get("saw")
                if not saw:
                    raise KeyError("RULE was handed no predecessor summary — `incoming` is empty")
                # The other half of the carried record, refused the same way and for a sharper
                # reason: `summary` is a `str` every `Report` has, but `measured` is the field
                # typed by the record parameter, so this is the one refusal that fires when a
                # `Ctx`'s `R` fails to carry anything.
                measured_output = op.args.get("measured_output")
                if not measured_output:
                    raise KeyError(
                        "RULE was handed no predecessor measurement — `incoming.measured` is empty"
                    )
                self.saw.append(saw)
                self.carried_outputs.append(measured_output)
                return CommandRun(exit_code=0, failures=(), output=self.rulings.pop(0))
            case unserved:
                raise KeyError(f"a ruling deployment does not serve {unserved!r}")


def note_row_id(ctx: Ctx[RulingState]) -> Key:
    """The id the FUSED judge authors, minted from its own coordinates.

    Two minters rather than one taking the tag, because `compose_key` refuses a key that BEGINS
    with an interpolation: a namespace tag is what makes a key registrable under one shape, and a
    tag arriving as a value is invisible to that registry. So the tag stays static in each, and
    what varies stays in the holes: the same layering rule as everywhere else, one grain finer."""
    # lint: terminal-hole -- `Ctx.visit` is an `int`; it needs no `Segment` wrapper
    return compose_key(t"note:{ctx.run_id},{Segment(ctx.state.value)},{ctx.visit}")


def ruling_row_id(ctx: Ctx[RulingState]) -> Key:
    """The id the judge-STATE authors. See `note_row_id` for why the tag is not a parameter.

    Exported so a driver can compose the id it asserts — which tracks a change in the judgement's
    COORDINATES and, deliberately, not a change in this function. A driver that also needs to catch
    a change to the minter itself must SPELL the bytes; the durable tape case does, and
    `test_conformance._judged_rows` says why the two halves differ."""
    # lint: terminal-hole -- `Ctx.visit` is an `int`; it needs no `Segment` wrapper
    return compose_key(t"ruling:{ctx.run_id},{Segment(ctx.state.value)},{ctx.visit}")


def _ruling_work(ctx: Ctx[RulingState]) -> Any:
    run: CommandRun = yield from call_tool(GATE_TOOL, {"at": ctx.visit}, CommandRun)
    return Evidence(summary=f"{ctx.state.value}: exit {run.exit_code}", measured=run)


def _judge_ruling_work(ctx: Ctx[RulingState], evidence: Evidence) -> Any:
    """A FUSED judge that authors a canonical row from its OWN coordinates.

    WORK is visited twice (the BREACH edge sends the walk back), so an `event_id` composed from the
    run id and a literal alone is the SAME on both visits, and the run dies on a
    `PlacedWriterCollision` before the postamble's two rows land. `note_row_id(ctx)` puts the
    visit in the id."""
    yield from append_ledger(
        LedgerRow(
            event_id=note_row_id(ctx),
            kind="noted",
            reason=evidence.summary,
        )
    )
    return WorkVerdict.DONE


def _ruling_review(ctx: Ctx[RulingState]) -> Any:
    """A worker-state whose judgement lives next door. It gathers and rules on nothing.

    It reads WORK's report, which is what makes the FUSED half of this machine witnessed. RULE
    reading REVIEW covers the SPLIT half only: with nothing downstream of a fused state reading
    what the lift produced, deleting `measured` from `Report.of` (the lift `fuse` performs) would
    leave every case green. This reads it, so that mutation kills the run.

    The tree is not decoration either. A walk that returns none leaves `carried` at `{}`, the
    postamble commits the digest of the empty tree, and the crash sweep's message *"the committed
    tree re-derived differently across the crash"* can never fire: a claim asserted over an empty
    set."""
    assert ctx.incoming is not None, "REVIEW is only ever entered from WORK"
    prior = ctx.incoming.measured
    assert prior is not None, "WORK's FUSED judge lifts its evidence, `measured` included"
    run: CommandRun = yield from call_tool(GATE_TOOL, {"at": ctx.visit}, CommandRun)
    return Report(
        GatherVerdict.GATHERED,
        # Both carries in one string: what WORK measured, and where this visit is.
        summary=f"gathered at visit {ctx.visit} after {prior.output}",
        measured=run,
        tree={RULING_NOTE: f"gathered at visit {ctx.visit}"},
    )


def _ruling_rule(ctx: Ctx[RulingState]) -> Any:
    """A judgement that is a STATE — it has an address, so it has a budget line and an edge.

    It reads its predecessor's report through `ctx.incoming`, which is the thing that did not
    exist: across a state boundary only the tree survived, and only into the postamble, so a state
    like this one had to re-run the measurement to see it.

    Both halves, deliberately. `summary` is prose every `Report` carries; `measured` is the field
    the RECORD parameter types, and it is what makes reading the predecessor's measurement an
    alternative to taking it again rather than a paraphrase of it."""
    assert ctx.incoming is not None, "RULE is only ever entered from REVIEW"
    measured = ctx.incoming.measured
    assert measured is not None, "REVIEW hands its gate result forward in `Report.measured`"
    answer: CommandRun = yield from call_tool(
        # `measured.output` is the point of the record parameter: `Report.measured` is typed
        # `R | None`, so this reaches a field `Measured` does not have and `ty` checks it here.
        "ruling",
        {"saw": ctx.incoming.summary, "measured_output": measured.output},
        CommandRun,
    )
    yield from append_ledger(
        LedgerRow(
            event_id=ruling_row_id(ctx),
            kind="ruled",
            reason=answer.output,
        )
    )
    verdict = RuleVerdict.APPROVED if answer.output == "approved" else RuleVerdict.BREACH
    return Report(verdict, summary=f"ruled {answer.output}", measured=answer)


def ruling_specs() -> dict[RulingState, StateSpec]:
    """Both forms in ONE spec map — WORK fuses, REVIEW and RULE do not.

    Hand-built rather than `build_specs`, because declining the fuse is the point: REVIEW and RULE
    each fill the one slot directly, and only WORK is a worker/judge pair."""
    return {
        RulingState.WORK: StateSpec(
            state=RulingState.WORK,
            run=fuse(_ruling_work, _judge_ruling_work),
            canonical=True,
        ),
        RulingState.REVIEW: StateSpec(state=RulingState.REVIEW, run=_ruling_review),
        RulingState.RULE: StateSpec(state=RulingState.RULE, run=_ruling_rule, canonical=True),
    }


def parking_ruling_machine_wf(run_id: str):
    """The same machine, budgeted so it EXHAUSTS: the other exit path.

    `run_machine`'s postamble has one call site with no `if` above it, so a run that exhausted is
    supposed to reach the canonical record exactly as one that finished does. Every other
    machine case on both engines walks to a `Finish`, and the only other `machine-parked` rows in
    the suite are recorder-based, so without this the claim holds over the empty set on the
    engines that matter.

    Budget 4 with a deployment that only ever breaches: WORK, REVIEW, RULE, WORK, and then visit 4
    is the budget, so `_visit` mints `Exhausted` WITHOUT consulting the state and `route_ruling`
    turns it into a `Park`. Three judgement rows rather than four, because the walk stops before
    the second RULE."""
    session = yield from run_machine(
        Run(run_id),
        RULING_GOAL,
        ruling_specs(),
        route_ruling,
        start=RulingState.WORK,
        budget=4,
        predicate=GATE_PREDICATE,
    )
    commitment = session.commitment
    assert commitment is not None, "the postamble is unconditional — that is what this case tests"
    return {
        "path": [state.value for state in session.path],
        "stopped": stop_record(session.stopped).kind,
        "artifact_id": commitment.artifact_id,
        "passed": commitment.passed,
        "files": list(commitment.files),
    }


def ruling_machine_wf(run_id: str):
    """14 ctx ops: WORK 2 (gate + note) + REVIEW 1 + RULE 2 (ruling + row), twice over the BREACH
    back edge, + 4 postamble (artifact, predicate, `ledger;machine:<run>;commit`,
    `ledger;machine:<run>`).

    Returns a PROJECTION rather than the `Session`, for the reason `coding_machine_wf` states:
    `DurableHandler._run` calls `_dump` exactly once at the task boundary, so asserting against a
    dumped `Finish | Park` would be asserting against a serializer."""
    session = yield from run_machine(
        # Promoted HERE, at the task boundary, for the reason `coding_machine_wf` gives: spawn
        # params arrive as `str`, so this is the one place a malformed run id can be refused
        # before the machine performs an op.
        Run(run_id),
        RULING_GOAL,
        ruling_specs(),
        route_ruling,
        start=RulingState.WORK,
        budget=12,
        predicate=GATE_PREDICATE,
    )
    commitment = session.commitment
    assert commitment is not None, "the postamble is unconditional; a Session always carries one"
    return {
        "path": [state.value for state in session.path],
        "verdicts": [str(turn.verdict) for turn in session.turns],
        "stopped": stop_record(session.stopped).kind,
        "artifact_id": commitment.artifact_id,
        "passed": commitment.passed,
        "files": list(commitment.files),
        # The run's own account of the two rows it appended, so a reader does not compose the
        # skeleton a second time to say what the postamble wrote.
        "commit_id": session.commit_id.stored(),
        "outcome_id": session.outcome_id.stored(),
    }


# ── crash injection (no psycopg, unlike tests/_durable.py) ─────────────────────
class FaultInjected(Exception):
    pass


class FaultPosition(StrEnum):
    """WHERE in an op the crash lands, and the two are not the same death.

    ``BEFORE_OP`` models "the process died before the tool ran": the resume finds no checkpoint,
    re-executes, and the effect happens once. A "survives a crash at every op" sweep at this
    position alone is silent about the other half.

    ``AFTER_THUNK`` models "the effect landed and the record did not": the thunk runs, the external
    effect happens, and the process dies before the checkpoint commits. On resume the store shows
    the same miss as if nothing had run, so the op executes AGAIN. That window is real on both
    engines and is occupied on every op: tens of microseconds on SQLite, hundreds on Absurd over
    localhost.

    **The two positions count different populations, and the arms own that difference.**
    ``BEFORE_OP`` trips on every ctx-op touch: steps, awaits, peeks, sleeps, reparks and settles,
    so ``k`` enumerates ctx ops. At ``AFTER_THUNK`` a ``k`` enumerates step executions alone, since
    the other surfaces run no thunk and so have no "after the effect landed" moment to count. A
    settle can still be aimed at BY NAME at either position, which is what lets a sweep crash a run
    at the checkpoint that decides its race; see ``FaultCtx.settle``. Read a ``k`` against its
    position, never across positions."""

    BEFORE_OP = "before-op"
    AFTER_THUNK = "after-thunk"


class Fault:
    """A crash, either before the k-th ctx op (``k``), before the first op whose
    name contains ``on_name``, or before the op whose name is ``named``; exactly one aim. The
    name-targeted forms are well-defined under concurrency, since only the matching branch's thread
    trips them, so they work on the sequential *and* concurrent gather paths, where the k-th
    *global* op is a race. ``named`` reaches an op whose name is a suffix of another's, as a join's
    is of its children's.

    ``position`` selects which half of the op the crash lands in; see ``FaultPosition``. It
    defaults to ``BEFORE_OP``, which is what every existing caller means.

    ``then`` is a second aim, at the same position, armed when the next attempt starts, which is
    how a run is crashed in two attempts. Armed at once, a gather sibling still running in the
    attempt the first crash ended could take it. It is aimed by name, since ``count`` runs on
    across attempts."""

    def __init__(
        self,
        k: int | None = None,
        on_name: str | None = None,
        position: FaultPosition = FaultPosition.BEFORE_OP,
        *,
        named: str | None = None,
        then: Fault | None = None,
    ) -> None:
        if sum(aim is not None for aim in (k, on_name, named)) > 1:
            raise ValueError(f"a fault takes one aim, not {k=}, {on_name=} and {named=}")
        if then is not None and then.k is not None:
            raise ValueError("a second crash is aimed by name")
        self.k = k
        self.on_name = on_name
        self.named = named
        self.then = then
        self.pending: Fault | None = None
        self.position = position
        self.count = 0
        self.fired = 0
        self.armed = k is not None or on_name is not None or named is not None

    def spend(self) -> None:
        """The crash fired: disarm, holding `then` for the next attempt."""
        self.fired += 1
        self.armed = False
        self.pending, self.then = self.then, None

    def begin_attempt(self) -> None:
        """A new attempt's ctx is built: arm the aim the last crash held back."""
        match self.pending:
            case Fault(on_name=on_name, named=named, then=then):
                self.k, self.on_name, self.named, self.then = None, on_name, named, then
                self.pending, self.armed = None, True


def at_every_op(unarmed: Fault) -> Iterator[tuple[int, Fault]]:
    """Each op an unarmed run reached, as a fault armed to crash there at the same position.

    After the caller's run with the fault, it must have fired: a fault that never fires is a run
    that did not crash, which would pass every assertion a crash test makes."""
    for k in range(1, unarmed.count + 1):
        fault = Fault(k, position=unarmed.position)
        yield k, fault
        assert fault.armed is False, f"k={k}: the fault never fired"


class FaultCtx:
    """Proxies a durable ctx, crashing once at the configured step/await/sleep — in the
    half of the op named by ``fault.position``."""

    def __init__(self, ctx: Any, fault: Fault) -> None:
        self._ctx = ctx
        self._f = fault
        fault.begin_attempt()

    def _trip(self, name: Key | str) -> None:
        """`Key | str` because this instrument probes EVERY ctx-op touch, and the ctx surface
        carries both types. `step` takes a `Key` (the checkpoint identity), while `await_event`
        takes a `str` because `AwaitEvent.name` is one. The synthesized probes (`peek:`, `sleep`,
        `repark:`) are instrument labels, not identities, and are strs by nature. Typing
        `AwaitEvent.name` as a `Key` narrows this union."""
        f = self._f
        f.count += 1
        if not f.armed:
            return
        # A SUBSTRING probe is over the durable spelling — `in` on an opaque `Key` is not a
        # string test, so the identity is unwrapped once, here, rather than at four call sites.
        text = name if isinstance(name, str) else name.stored()
        match f.on_name, f.named:
            case str(part), _:
                fire = part in text
            case None, str(whole):
                fire = whole == text
            case _:
                fire = f.count == f.k
        if fire:
            f.spend()
            raise FaultInjected(f"crash at {text!r} (count={f.count})")

    def _trip_before(self, name: Key | str) -> None:
        """An await, a peek, a sleep and a repark honor ``BEFORE_OP`` only. None of them runs a
        thunk, so there is no moment at which "the effect landed and the record did not" is even
        expressible for them, and a fault armed ``AFTER_THUNK`` must pass straight through rather
        than fire here, or ``k`` would silently enumerate a population that mixes the two.

        A settle runs no thunk either and takes a narrower rule of its own, in ``settle``: it keeps
        this population intact while staying reachable by name."""
        if self._f.position is FaultPosition.BEFORE_OP:
            self._trip(name)

    def _after_thunk(self, name: Key | str, thunk):
        """Run the thunk — the external effect LANDS HERE — then die before the commit.

        Wrapping the THUNK rather than the ctx method is what makes this need no engine
        cooperation: the raise escapes before ``ctx.step`` can write the checkpoint, on either
        engine, with no knowledge of how either one commits."""

        def wrapped():
            result = thunk()
            self._trip(name)
            return result

        return wrapped

    def step(self, name, thunk):
        match self._f.position:
            case FaultPosition.BEFORE_OP:
                self._trip(name)
                return self._ctx.step(name, thunk)
            case FaultPosition.AFTER_THUNK:
                return self._ctx.step(name, self._after_thunk(name, thunk))
            case unreachable:
                assert_never(unreachable)

    def step_resolved(self, name, thunk):
        """`step`'s fault, for a step whose thunk is handed the name the engine resolved."""
        match self._f.position:
            case FaultPosition.BEFORE_OP:
                self._trip(name)
                return self._ctx.step_resolved(name, thunk)
            case FaultPosition.AFTER_THUNK:

                def landed(resolved):
                    result = thunk(resolved)
                    self._trip(name)
                    return result

                return self._ctx.step_resolved(name, landed)
            case unreachable:
                assert_never(unreachable)

    def await_event(self, name):
        self._trip_before(name)
        return self._ctx.await_event(name)

    def await_until(self, name, deadline, decided):
        """A wait the clock can end is a ctx-op touch like any other.

        Without this arm it reaches the engine through `__getattr__` and trips nothing, so the
        crash-at-every-op sweep would report a clean run over a surface it never entered — the
        blind spot `peek_event` and `repark` were each added here to close.

        `decided` is the slot the wait's outcome is recorded in, handed down by the walk."""
        self._trip_before(name)
        inner: Any = self._ctx
        return inner.await_until(name, deadline, decided)

    def peek_event(self, name):
        # A gather branch's await surfaces as a non-suspending peek (V1); a crash
        # "before the branch's await" is a real death mode, so peeks trip too.
        self._trip_before(f"peek:{name}")
        return self._ctx.peek_event(name)

    def sleep_until(self, when, *, name: Key | None = None):
        self._trip_before("sleep")
        return self._ctx.sleep_until(when, name=name)

    def peek_step(self, name):
        self._trip_before(f"peek-step:{name.stored()}")
        return self._ctx.peek_step(name)

    def settle(self, name, value):
        """A race's choice and its endings, under their OWN names, which is what lets a sweep aim
        at them.

        A settled checkpoint runs no thunk, so the two positions name one moment for it: the value
        is decided before ``settle`` is called and the record is the whole effect, so a crash here
        is "the race decided and the store does not hold it" whichever position armed it. A
        synthesized label instead would put the choice out of reach of a sweep, which aims by the
        checkpoint's own name.

        At ``AFTER_THUNK`` only a fault aimed BY NAME touches this surface, so what that position
        counts is step executions and nothing else, which is what ``FaultPosition`` promises. That
        holds for an unarmed fault too, and it has to: a sweep measures how many ordinal faults to
        arm from an unarmed run's count, and a settle counted there is a ``k`` that never fires.
        A settle the store already holds writes nothing, so ``AFTER_THUNK`` passes it through, as
        a replayed step runs no thunk to crash after. At ``BEFORE_OP`` a settle is counted as
        every other ctx-op touch is.

        Neither position reaches the moment AFTER the store holds the choice and before the race
        publishes it to its branches. That window is real, and a death there is a different
        worker test's subject rather than this instrument's."""
        match self._f.position:
            case FaultPosition.BEFORE_OP:
                self._trip(name)
            case FaultPosition.AFTER_THUNK:
                aimed_by_name = self._f.named is not None or self._f.on_name is not None
                if aimed_by_name and not self._ctx.peek_step(name)[0]:
                    self._trip(name)
            case unreachable:
                assert_never(unreachable)
        return self._ctx.settle(name, value)

    def repark(self, name):
        # The no-burn wake-race reschedule is a ctx-op touch too — a crash landing
        # exactly there is a real death mode (the task must converge via the
        # engine's ordinary retry, whose replay resolves every branch at its peek).
        self._trip_before(f"repark:{name}")
        return self._ctx.repark(name)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class PeekRace:
    """One-shot shared state for ``RaceOnFirstPeek`` (the ``Fault`` pattern: state
    outlives the per-claim ctx). ``emit`` receives the fully qualified event name
    the branch peeked for; ``fired`` ensures the wake replay's peek DELEGATES and
    resolves from the durable record instead of racing again."""

    def __init__(self, emit) -> None:
        self.emit = emit
        self.fired = False


class RaceOnFirstPeek:
    """Proxies a durable ctx to make the gather wake race deterministic on a real
    engine: the FIRST branch peek emits the awaited event and still reports it
    unseen — the mid-round emission landing between a branch's peek and the
    barrier. The branch parks as a value, the post-barrier re-arm finds every
    wake condition already satisfied, and ``_join`` hits the race. Every later
    peek delegates, so the wake replay resolves the branch from the record."""

    def __init__(self, ctx: Any, race: PeekRace) -> None:
        self._ctx = ctx
        self._race = race

    def peek_event(self, name):
        if not self._race.fired:
            self._race.fired = True
            self._race.emit(name)
            return False, None
        return self._ctx.peek_event(name)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class PerEventPeekRace:
    """The per-EVENT variant of ``PeekRace``: the one-shot fires once per event
    NAME, so a branch with sequential awaits races the same gather repeatedly:
    the double-race probe."""

    def __init__(self, emit) -> None:
        self.emit = emit
        self.raced: set[str] = set()


class RaceOnEveryFirstPeek:
    """``RaceOnFirstPeek``'s per-event sibling: the first peek of EACH event
    name emits it and reports it unseen; later peeks of that name delegate."""

    def __init__(self, ctx: Any, race: PerEventPeekRace) -> None:
        self._ctx = ctx
        self._race = race

    def peek_event(self, name):
        if name not in self._race.raced:
            self._race.raced.add(name)
            self._race.emit(name)
            return False, None
        return self._ctx.peek_event(name)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


# ── backend adapters ─────────────────────────────────────────────────────────
class SqliteBackend:
    """The embedded SQLite engine (0↔1). Always available, infra-free."""

    name = "sqlite"
    mutation_error: type[Exception] = sqlite3.DatabaseError

    def __init__(self) -> None:
        self.app = SqliteApp(":memory:")

    def register(
        self,
        name: str,
        factory,
        domain,
        fault: Fault,
        layers,
        wrap=None,
        *,
        budget_limit: float | None = None,
        on_exhaust: OnExhaust = "park",
        fresh: Callable[[], Any] | None = None,
    ) -> None:
        app = self.app

        @app.register_task(name)
        def task(params, ctx):
            rid = params["run_id"]
            # The accrual contract is read from the immutable spawn params (the real
            # migration seam) and never hardcoded, so a param-less (in-flight) task stays
            # v0 under this v1-capable worker.
            contract = Contract.from_params(params)
            # Share the app's write-lock so a concurrent gather branch's ledger append
            # serializes with the checkpoint writes on the one connection. `wrap`
            # (e.g. RaceOnFirstPeek) sits INSIDE FaultCtx so faults still see every op.
            ledger = SqliteLedger(app.conn, rid, app.write_lock)
            inner = wrap(ctx) if wrap is not None else ctx
            budget = (
                MeasuredBudget(overall=budget_limit, run_id=rid, on_exhaust=on_exhaust)
                if budget_limit is not None
                else None
            )
            return DurableHandler(
                FaultCtx(inner, fault),
                # `fresh` builds the domain HERE, inside the attempt, so a deployment that keeps
                # state between calls cannot carry it across a crash — the shape the sweep is
                # otherwise blind to, since one domain object lives across attempts.
                domain if fresh is None else fresh(),
                ledger=ledger,
                op_layers=layers,
                contract=contract,
                budget=budget,
            ).run(lambda: factory(rid))

    def spawn(
        self,
        name: str,
        run_id: str,
        max_attempts: int | None = None,
        contract: Contract = Contract.V0,
    ) -> UUID:
        params = _spawn_params(run_id, contract)
        if max_attempts is not None:
            return self.app.spawn(name, params, max_attempts=max_attempts)
        return self.app.spawn(name, params)

    def task_attempts(self, task_id: UUID) -> int:
        row = self.app.conn.execute(
            "SELECT attempt FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return row[0]

    def run_until_result(self, task_id: UUID) -> Any:
        return self.app.run_until_result(task_id)

    def failure_kind(self, snapshot: Any) -> str:
        """The type name of the error a failed task was reported as: its `repr` leads the row."""
        return str(snapshot.failure).split("(", 1)[0]

    def emit_event(self, task_id: UUID, event: str, payload: Any) -> None:
        self.app.emit_event(event, payload)

    def checkpoint_keys(self, task_id: UUID) -> list[str]:
        from effective.checkpoints import keys, read_sqlite_conn

        # `read_sqlite_conn`, not `read_sqlite_task`: an `:memory:` store has no path to reopen.
        return list(keys(read_sqlite_conn(self.app.conn, task_id)))

    def checkpoint_states(self, task_id: UUID) -> dict[Key, Any]:
        from effective.checkpoints import read_sqlite_conn

        return {c.key: c.state for c in read_sqlite_conn(self.app.conn, task_id, exclude=())}

    def parked(self, task_id: UUID) -> list[Any]:
        from effective.parked import read_sqlite_parked_conn

        return [p for p in read_sqlite_parked_conn(self.app.conn) if p.task_id == task_id]

    def ledger_kinds(self, run_id: str) -> list[str]:
        rows = self.app.conn.execute(
            "SELECT kind FROM ledger WHERE workflow_run_id=? ORDER BY seq", (run_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def ledger_ids(self, run_id: str) -> list[str]:
        """The AUTHORED ids, which `ledger_kinds` cannot see.

        Two rows a state authors at different visits carry the same `kind` and differ only here,
        so the collision `Ctx`-bearing judges dissolved is invisible to the kinds. And this is the
        OTHER bookkeeper: a `ledger;` op in `checkpoint_keys` proves the op was recorded, not that
        the row reached the append-only record."""
        rows = self.app.conn.execute(
            "SELECT event_id FROM ledger WHERE workflow_run_id=? ORDER BY seq", (run_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def ledger_payloads(self, run_id: str) -> list[dict[str, Any]]:
        """The canonical rows, which is what a reader such as `runview` is handed."""
        return list(sqlite_payloads(self.app.conn, run_id))

    def ledger_append(
        self, run_id: str, event_id: str, kind: str, *, writer: Writer | None = None
    ) -> None:
        SqliteLedger(self.app.conn, run_id, self.app.write_lock).append(
            LedgerRow(event_id=Key.parse(event_id), kind=kind), writer=writer
        )

    def raw_update_kind(self, event_id: str) -> None:
        self.app.conn.execute("UPDATE ledger SET kind='y' WHERE event_id=?", (event_id,))

    def unclaimed_ctx(self, task_id: UUID) -> Any:
        """A WRITABLE ctx bound to a task this process never claimed — the exposed surface.

        Reproducing the hazard is the point: `SqliteTaskContext` is public and takes a connection,
        a task id and a lock, so anything that can open the store can build one. The viewing tests
        wrap this and assert the wrapper refuses what the bare ctx would have written."""
        return SqliteTaskContext(self.app.conn, task_id, self.app.write_lock)

    def register_body(self, name: str, body, *, deployed: bool = False) -> None:
        """A task whose body is `(params, ctx)`, over this engine's own ctx, which is also the
        ctx the engine ships, so `deployed` changes nothing here."""
        self.app.register_task(name)(body)

    def register_child(
        self, name: str, run_id: str, factory, domain, fault: Fault, fault_for=None
    ) -> None:
        """A spawned task answering its parent through `run_child`: `factory(params)` runs under a
        handler holding the task's own params, writing the ledger of `run_id`. `fault_for(params,
        ctx)`, when given, chooses each task's fault in place of `fault`."""

        @self.app.register_task(name)
        def task(params, ctx):
            chosen = fault if fault_for is None else fault_for(params, ctx)
            faulted = FaultCtx(ctx, chosen)
            ledger = SqliteLedger(self.app.conn, run_id, self.app.write_lock)
            handler = DurableHandler(faulted, domain, ledger=ledger, params=params)
            return run_child(faulted, params, handler, lambda: factory(params))

    def spawner(
        self,
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        """The `effective.interpreters.tools.Spawner` a spawn tool enqueues through on this
        engine."""
        return str(
            self.app.spawn(
                task_name, params, idempotency_key=idempotency_key, max_attempts=max_attempts
            )
        )

    def enqueued(self, name: str) -> list[tuple[UUID, dict[str, Any]]]:
        """The id and params of every task enqueued under `name`, in enqueue order."""
        rows = self.app.conn.execute(
            "SELECT task_id, params FROM tasks WHERE name = ? ORDER BY rowid", (name,)
        )
        return [(task_id, json.loads(params)) for task_id, params in rows]

    def close(self) -> None:
        self.app.close()


class AbsurdBackend:
    """Absurd/Postgres (0↔N). Imports psycopg/absurd_sdk lazily (PG-gated)."""

    name = "postgres"

    def __init__(self) -> None:
        from effective.absurd_worker import absurd_worker

        # Any, mirroring tests/_durable.py: the SDK types retry_strategy/spawn strictly,
        # but the dict-shaped IMMEDIATE_RETRY is the established test convention.
        self.app: Any = absurd_worker(PG_DSN)
        self._spawner_app: Any = None

    @property
    def mutation_error(self) -> type[Exception]:
        import psycopg

        return psycopg.errors.RaiseException

    def register(
        self,
        name: str,
        factory,
        domain,
        fault: Fault,
        layers,
        wrap=None,
        *,
        budget_limit: float | None = None,
        on_exhaust: OnExhaust = "park",
        fresh: Callable[[], Any] | None = None,
    ) -> None:
        from effective.engines.absurd import ConcurrentAbsurdCtx
        from effective.ledger import PostgresLedger

        @self.app.register_task(name, default_max_attempts=3)
        def task(params, ctx):
            rid = params["run_id"]
            contract = Contract.from_params(params)  # the real migration seam
            # Opt into concurrent gather: parallel tools, serialized writes via the SDK
            # begin/complete split. PostgresLedger pools, so its appends are already
            # concurrent; only the single SDK checkpoint connection needs the lock. `wrap`
            # (e.g. RaceOnFirstPeek) sits INSIDE FaultCtx so faults still see every op.
            ledger = PostgresLedger(PG_DSN, workflow_run_id=rid)
            inner = ConcurrentAbsurdCtx(ctx)
            if wrap is not None:
                inner = wrap(inner)
            budget = (
                MeasuredBudget(overall=budget_limit, run_id=rid, on_exhaust=on_exhaust)
                if budget_limit is not None
                else None
            )
            try:
                return DurableHandler(
                    FaultCtx(inner, fault),
                    domain if fresh is None else fresh(),
                    ledger=ledger,
                    op_layers=layers,
                    contract=contract,
                    budget=budget,
                ).run(lambda: factory(rid))
            finally:
                ledger.close()

    def spawn(
        self,
        name: str,
        run_id: str,
        max_attempts: int | None = None,
        contract: Contract = Contract.V0,
    ) -> UUID:
        return self.app.spawn(
            name,
            _spawn_params(run_id, contract),
            retry_strategy=IMMEDIATE_RETRY,
            max_attempts=max_attempts,
        )["task_id"]

    def task_attempts(self, task_id: UUID) -> int:
        import psycopg

        with psycopg.connect(PG_DSN) as conn:
            row = conn.execute(
                'SELECT attempts FROM absurd."t_default" WHERE task_id = %s::uuid', (task_id,)
            ).fetchone()
        assert row is not None
        return row[0]

    def failure_kind(self, snapshot: Any) -> str:
        """The type name of the error a failed task was reported as."""
        return snapshot.failure["name"]

    def run_until_result(self, task_id: UUID, max_batches: int = 24) -> Any:
        for _ in range(max_batches):
            snap = self.app.fetch_task_result(task_id)
            if snap is not None and snap.state in ("completed", "failed", "cancelled"):
                return snap
            self.app.work_batch()
        return self.app.fetch_task_result(task_id)

    def emit_event(self, task_id: UUID, event: str, payload: Any) -> None:
        self.app.emit_event(event, payload)  # Absurd events are by name

    def checkpoint_keys(self, task_id: UUID) -> list[str]:
        import psycopg

        from effective.bridge_absurd import read_absurd_task
        from effective.checkpoints import keys

        with psycopg.connect(PG_DSN) as conn:
            return list(keys(read_absurd_task(conn, task_id)))

    def checkpoint_states(self, task_id: UUID) -> dict[Key, Any]:
        import psycopg

        from effective.bridge_absurd import read_absurd_task

        with psycopg.connect(PG_DSN) as conn:
            return {c.key: c.state for c in read_absurd_task(conn, task_id, exclude=())}

    def parked(self, task_id: UUID) -> list[Any]:
        import psycopg

        from effective.parked import read_absurd_parked

        # Filtered to THIS task on purpose: the queue is shared, so the unfiltered list is the
        # right contract for the reader and the wrong assertion for a test.
        #
        # A bare `==` works on BOTH engines: both spawns hand back a `UUID` object and
        # `ParkedTask.task_id` is one, so there is nothing to convert on either side.
        #
        # The SDK's `SpawnResult` is a `TypedDict` annotating `task_id: str`, and a `TypedDict`
        # validates NOTHING at runtime: the SDK's `spawn` returns `row["task_id"]` from
        # `SELECT task_id … FROM absurd.spawn_task()`, a `uuid` column psycopg hydrates as a
        # `UUID` object. `type()` on the real return value (`_conformance.AbsurdBackend.spawn`,
        # run against the pg-test container) says `UUID`.
        with psycopg.connect(PG_DSN) as conn:
            return [p for p in read_absurd_parked(conn) if p.task_id == task_id]

    def unclaimed_ctx(self, task_id: UUID) -> Any:
        """A WRITABLE ctx bound to a task this process never claimed — the exposed surface.

        **Absurd refuses this in Python and not in SQL.** `TaskContext.__init__` raises
        `TypeError`, so only a claim yields one, but that is a convention `object.__new__` walks
        straight past,
        and the guard underneath is `absurd.set_task_checkpoint_state`, which raises on a run id
        absent from `r_{queue}`, a cancelled task and a run already failed, and silently skips a
        stale attempt. **Claim ownership is among none of them.** A viewer supplying the REAL
        `owner_run_id`, one `SELECT` away (below), passes all four and its writes land.

        A fabricated `uuid4()` against an already-crashed run hits the run-not-found arm instead,
        which makes the binding look unreachable on this engine; this method builds the reachable
        case.

        The connection is left open deliberately: the ctx reads and writes through it for as long
        as the test holds it, so `with psycopg.connect(...)` would close it out from under the
        caller. The test closes it.
        """
        import psycopg
        from absurd_sdk import TaskContext

        # `autocommit=True`, matching what the SDK does for its own connection
        # (`Connection.connect(conn_or_url, autocommit=True)`). Without it the checkpoint write
        # sits in an open transaction and is discarded when the connection closes — which made an
        # earlier version of this fixture report that Absurd had refused the write, when what had
        # actually happened is that the fixture threw it away. A viewing fixture that cannot
        # commit understates the hazard it exists to demonstrate.
        conn = psycopg.connect(PG_DSN, autocommit=True)
        row = conn.execute(
            "SELECT last_attempt_run, task_name, attempts, params, headers, retry_strategy, "
            'max_attempts FROM absurd."t_default" WHERE task_id = %s::uuid',
            (task_id,),
        ).fetchone()
        assert row is not None, f"no task row for {task_id}"
        # Every field read from the REAL row rather than fabricated. Only `task_id` and `run_id`
        # are consulted by the checkpoint path, so stub values would pass — and would make this
        # fixture a weaker claim than the hazard it stands for, which is a viewer holding
        # everything a claimant holds and still not being one.
        claimed: Any = {
            "task_id": str(task_id),
            "run_id": str(row[0]),
            "task_name": row[1],
            "attempt": row[2],
            "params": row[3],
            "headers": row[4],
            "retry_strategy": row[5],
            "max_attempts": row[6],
            "wake_event": None,
            "event_payload": None,
        }
        ctx = object.__new__(TaskContext)
        ctx._conn = conn
        ctx._queue_name = "default"
        ctx._task = claimed
        ctx._checkpoint_cache = {}
        ctx._step_name_counter = {}
        ctx._claim_timeout = 60
        return ctx

    def ledger_kinds(self, run_id: str) -> list[str]:
        import psycopg

        with psycopg.connect(PG_DSN) as conn:
            rows = conn.execute(
                "SELECT kind FROM ledger WHERE workflow_run_id=%s ORDER BY seq", (run_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def ledger_ids(self, run_id: str) -> list[str]:
        """The AUTHORED ids, which `ledger_kinds` cannot see.

        Two rows a state authors at different visits carry the same `kind` and differ only here,
        so the collision `Ctx`-bearing judges dissolved is invisible to the kinds. And this is the
        OTHER bookkeeper: a `ledger;` op in `checkpoint_keys` proves the op was recorded, not that
        the row reached the append-only record."""
        import psycopg

        with psycopg.connect(PG_DSN) as conn:
            rows = conn.execute(
                "SELECT event_id FROM ledger WHERE workflow_run_id=%s ORDER BY seq", (run_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def ledger_payloads(self, run_id: str) -> list[dict[str, Any]]:
        """The canonical rows, which is what a reader such as `runview` is handed."""
        import psycopg

        with psycopg.connect(PG_DSN) as conn:
            return list(pg_payloads(conn, run_id))

    def ledger_append(
        self, run_id: str, event_id: str, kind: str, *, writer: Writer | None = None
    ) -> None:
        from effective.ledger import PostgresLedger

        ledger = PostgresLedger(PG_DSN, workflow_run_id=run_id)
        try:
            ledger.append(LedgerRow(event_id=Key.parse(event_id), kind=kind), writer=writer)
        finally:
            ledger.close()

    def raw_update_kind(self, event_id: str) -> None:
        import psycopg

        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute(t"UPDATE ledger SET kind='y' WHERE event_id={event_id}")

    def register_body(self, name: str, body, *, deployed: bool = False) -> None:
        """A task whose body is `(params, ctx)`, over the concurrent-gather ctx `register` uses,
        or with `deployed` over the adapted SDK ctx a production worker hands its handler."""
        from effective.engines.absurd import ConcurrentAbsurdCtx, _adapt_ctx

        wrap = _adapt_ctx if deployed else ConcurrentAbsurdCtx
        self.app.register_task(name, default_max_attempts=3)(
            lambda params, ctx: body(params, wrap(ctx))
        )

    def register_child(
        self, name: str, run_id: str, factory, domain, fault: Fault, fault_for=None
    ) -> None:
        """A spawned task answering its parent through `run_child`: `factory(params)` runs under a
        handler holding the task's own params, writing the ledger of `run_id`. `fault_for(params,
        ctx)`, when given, chooses each task's fault in place of `fault`."""
        from effective.engines.absurd import ConcurrentAbsurdCtx
        from effective.ledger import PostgresLedger

        @self.app.register_task(name, default_max_attempts=3)
        def task(params, ctx):
            inner = ConcurrentAbsurdCtx(ctx)
            chosen = fault if fault_for is None else fault_for(params, inner)
            faulted = FaultCtx(inner, chosen)
            ledger = PostgresLedger(PG_DSN, workflow_run_id=run_id)
            try:
                handler = DurableHandler(faulted, domain, ledger=ledger, params=params)
                return run_child(faulted, params, handler, lambda: factory(params))
            finally:
                ledger.close()

    def spawner(
        self,
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        """The `effective.interpreters.tools.Spawner` a spawn tool enqueues through on this engine,
        on its own app so an enqueue from inside a running task does not share the worker's
        connection."""
        if self._spawner_app is None:
            from absurd_sdk import Absurd

            self._spawner_app = Absurd(PG_DSN, queue_name="default")
        # Any, as `self.app` is: the dict-shaped IMMEDIATE_RETRY is the test convention.
        app: Any = self._spawner_app
        spawned = app.spawn(
            task_name,
            params,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            queue=queue,
            retry_strategy=IMMEDIATE_RETRY,
        )
        return str(spawned["task_id"])

    def enqueued(self, name: str) -> list[tuple[UUID, dict[str, Any]]]:
        """The id and params of every task enqueued under `name`, in enqueue order."""
        import psycopg

        with psycopg.connect(PG_DSN) as conn:
            return conn.execute(
                t"SELECT task_id, params FROM absurd.t_default WHERE task_name = {name} "
                t"ORDER BY task_id"
            ).fetchall()

    def close(self) -> None:
        if self._spawner_app is not None:
            self._spawner_app.close()
        self.app.close()


def pg_ready() -> bool:
    """True iff a Postgres with the Absurd schema + ledger is reachable."""
    try:
        import psycopg

        with psycopg.connect(PG_DSN, connect_timeout=2) as conn:
            conn.execute("SELECT 1 FROM ledger LIMIT 0")
            conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='absurd'")
        return True
    except Exception:
        return False
