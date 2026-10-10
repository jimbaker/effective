"""The transition table holds every ctx read and clock read the durable handler makes.

| check   | its cases come from                                                      |
|---------|--------------------------------------------------------------------------|
| static  | the handler modules' source: each ctx member read and each clock read,   |
|         | by the function that reads it                                            |
| dynamic | the engines' answers: each call a run makes, by site and outcome, on     |
|         | SQLite and on Absurd                                                     |
"""

import ast
import functools
import inspect
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from test_futile_retries import Flaky, ReparkReturns, Tools, ok_branch

from effective.api import await_event, await_until, call_tool, gather, race, sleep_until
from effective.engines.absurd import AbsurdEngine, ConcurrentAbsurdCtx, sdk_signals
from effective.govern import Refused, all_refusals
from effective.handlers import base, durable
from effective.handlers.base import EngineSignal
from effective.handlers.durable import DurableHandler
from effective.handlers.transitions import FIXED, TRANSITIONS, Outcome, World
from effective.layers import op_layer
from effective.ops import Addressing

PROBES = frozenset(
    {"_supports_peek", "_supports_await_until", "_race_capable", "race_clock", "race_time"}
)
"""Functions that ask whether a capability exists, or that ARE the clock."""

WRAPPERS = frozenset(cls.__name__ for cls in durable._WRAPPERS)
CTX_PARAMS = frozenset({"ctx", "at"})
CTX_FACTORIES = frozenset({"_race_capable", "_PrefixedCtx"})


def _is_ctx(node: ast.expr, aliases: set[str]) -> bool:
    match node:
        case ast.Name(id=name):
            return name in aliases
        case ast.Attribute(attr="ctx" | "_root_ctx"):
            return True
        case ast.Call(func=ast.Name(id=factory)):
            return factory in CTX_FACTORIES
        case ast.IfExp(body=chosen, orelse=other):
            return _is_ctx(chosen, aliases) or _is_ctx(other, aliases)
        case ast.NamedExpr(value=value):
            return _is_ctx(value, aliases)
        case _:
            return False


def _reads(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The ctx members `function` reads, through any name it binds to a ctx."""
    aliases = set(CTX_PARAMS & {a.arg for a in function.args.args})
    for node in ast.walk(function):
        match node:
            case (
                ast.Assign(targets=[ast.Name(id=name)], value=value)
                | ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value)
                | ast.NamedExpr(target=ast.Name(id=name), value=value)
            ) if _is_ctx(value, aliases):
                aliases.add(name)
    read = set()
    for node in ast.walk(function):
        match node:
            case ast.Attribute(value=value, attr=member) if _is_ctx(value, aliases):
                read.add(member)
            case ast.Call(
                func=ast.Name(id="getattr"), args=[value, ast.Constant(value=str() as member), *_]
            ) if _is_ctx(value, aliases):
                read.add(member)
    return read


def _clocks(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    clocks = set()
    for node in ast.walk(function):
        match node:
            case ast.Call(func=ast.Name(id="race_clock")):
                clocks.add("race_clock")
            case ast.Call(func=ast.Attribute(value=ast.Name(id="time"), attr="time")):
                clocks.add("clock")
    return clocks


def _functions(module: Any) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Each module-level function and method of `module`, by qualified name; a nested function
    belongs to the one that defines it."""
    tree = ast.parse(inspect.getsource(module))
    found = {}
    for node in tree.body:
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                found[name] = node
            case ast.ClassDef(name=cls, body=body):
                for member in body:
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                        found[f"{cls}.{member.name}"] = member
    return found


def source_reads() -> set[tuple[str, str]]:
    """Each (site, source) the handler modules' source reads, wrappers and probes aside."""
    seen = set()
    for module in (durable, base):
        for site, function in _functions(module).items():
            if site.split(".")[0] in WRAPPERS | PROBES:
                continue
            seen |= {(site, clock) for clock in _clocks(function)}
            if module is durable:
                seen |= {(site, member) for member in _reads(function)}
    return seen


def tabled() -> set[tuple[str, str]]:
    return {(site, source) for site, source, _ in TRANSITIONS}


def test_every_read_in_the_source_is_tabled():
    untabled = {read for read in source_reads() if read[1] not in FIXED} - tabled()
    assert untabled == set()


def test_every_tabled_read_is_in_the_source():
    assert tabled() - source_reads() == set()


def test_every_counted_transition_is_counted_where_it_says():
    functions = _functions(durable)

    def counts(site: str) -> bool:
        return any(
            isinstance(node, ast.Attribute) and node.attr == "unrecorded"
            for node in ast.walk(functions[site])
        )

    missing = {
        label.counted_at
        for label in TRANSITIONS.values()
        if isinstance(label, World) and label.kept_by in ("counted", "settled")
        if label.counted_at is None or not counts(label.counted_at)
    }
    assert missing == set()


UNRECORDED: dict[str, Any] = {
    "race_clock": lambda site: (site, "race_clock", Outcome.ANSWERED),
    "_attempt_of": lambda site: ("_attempt_of", "attempt", Outcome.ANSWERED),
}
"""Each source that answers an `Unrecorded`, and the row an exit from it at `site` is held to."""

KEPT_BY = {"settled": "settled", "elided": "elides"}
"""Each exit, and the reason a row gives for keeping the read it unwraps."""


def _unrecorded_reads() -> Iterator[tuple[str, str, ast.Call | None]]:
    """Each call to an unrecorded source in the handler modules: its site, the source, and the
    exit whose receiver holds it, if any."""
    for module in (durable, base):
        for site, function in _functions(module).items():
            if site in UNRECORDED:
                continue
            exits = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in KEPT_BY
            ]
            for node in ast.walk(function):
                match node:
                    case ast.Call(func=ast.Name(id=source)) if source in UNRECORDED:
                        held = [e for e in exits if node in set(ast.walk(e.func))]
                        yield site, source, held[0] if held else None


def test_every_unrecorded_read_leaves_through_its_rows_exit():
    """An exit names why the read stays sound, so it must be the reason the read's row gives."""
    wrong = {}
    for site, source, exit_ in _unrecorded_reads():
        label = TRANSITIONS.get(UNRECORDED[source](site))
        match exit_:
            case ast.Call(func=ast.Attribute(attr=taken)):
                pass
            case _:
                taken = None
        if taken is None or not isinstance(label, World) or label.kept_by != KEPT_BY[taken]:
            wrong[(site, source)] = (taken, label)
    assert wrong == {}


UNRECORDED_SITES = Counter(
    {
        ("Racing.expired", "race_clock"): 1,
        ("Racing.stamp", "race_clock"): 1,
        ("Racing.concurrently", "race_clock"): 1,
        ("DurableHandler._run_race", "_attempt_of"): 1,
    }
)
"""Each read of an unrecorded value, by where it decides. A row keyed by the source's reader
rather than by where it decides (`_attempt_of`'s) holds a new read to nothing, so each is
declared here."""


def test_each_unrecorded_value_decides_only_where_it_is_declared():
    assert Counter((site, source) for site, source, _ in _unrecorded_reads()) == UNRECORDED_SITES


def test_an_unrecorded_source_is_only_ever_called():
    """A source handed on uncalled, to a partial or a table, would reach a decision the exit
    grader never sees."""
    handed = set()
    for module in (durable, base):
        for site, function in _functions(module).items():
            parents = {
                child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)
            }
            for node in ast.walk(function):
                match node, parents.get(node):
                    case ast.Name(id=source), ast.Call(func=func) if func is node:
                        pass
                    case ast.Name(id=source), _ if source in UNRECORDED:
                        handed.add((site, source))
    assert handed == set()


@pytest.mark.parametrize(
    ("decide", "refusal"),
    [
        (lambda clock: bool(clock), "cannot be interpreted as a boolean"),
        (lambda clock: bool(clock >= 0.0), "cannot be interpreted as a boolean"),
        (lambda clock: clock == 1.0, "decides nothing until an exit names why"),
        (lambda clock: {clock}, "unhashable"),
    ],
    ids=["a branch", "a branch on a comparison", "an equality", "a hash"],
)
def test_an_unrecorded_value_decides_nothing_until_it_exits(decide, refusal):
    with pytest.raises(TypeError, match=refusal):
        decide(base.Unrecorded(1.0))


def test_an_unrecorded_value_answers_through_its_exits():
    clock = base.Unrecorded(1.0)
    assert (clock >= 0.0).settled() is True
    assert (3.0 - clock).settled() == 2.0


@pytest.mark.parametrize(
    "call",
    [
        "self.ctx.untabled_read()",
        "store = self._root_ctx if slot else self.ctx\n        store.untabled_read()",
        "(store := self.ctx).untabled_read()",
    ],
    ids=["direct", "a chosen ctx", "a walrus"],
)
def test_a_read_the_table_lacks_fails_the_static_check(monkeypatch, call):
    """The static check sees a ctx call it was not told about, however the ctx is named."""
    source = inspect.getsource(durable).replace(
        "        name = self._step_name(op, slot)\n",
        f"        name = self._step_name(op, slot)\n        {call}\n",
        1,
    )
    assert "untabled_read" in source
    read = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda m: source if m is durable else read(m))
    assert ("DurableHandler._checkpoint", "untabled_read") in source_reads() - tabled()


SIGNALS = (EngineSignal, *sdk_signals())


def _site() -> str:
    """The handler function nearest the call, past the ctx wrappers and a race's clock read."""
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename in (durable.__file__, base.__file__):
            site = frame.f_code.co_qualname.split(".<locals>")[0]
            if site.split(".")[0] not in WRAPPERS and site != "race_clock":
                return site
        frame = frame.f_back
    return "?"


class Logged:
    """A ctx that logs each call and attribute read it answers, by site and outcome, and fails the
    first read of each (site, member) in `fail` as a busy store does."""

    def __init__(
        self, ctx: Any, seen: set[tuple[str, str, Outcome]], fail: set[tuple[str, str]]
    ) -> None:
        self._ctx = ctx
        self._seen = seen
        self._fail = fail
        self._failed: list[BaseException] = []

    def _log(self, member: str, outcome: Outcome) -> None:
        self._seen.add((_site(), member, outcome))

    def __getattr__(self, member: str) -> Any:
        try:
            attr = getattr(self._ctx, member)
            if not callable(attr):
                self._busy(member)
        except AttributeError:
            raise
        except Exception as raised:
            self._store_raised(member, raised)
            raise
        if not callable(attr):
            if member not in FIXED:
                self._log(member, Outcome.ANSWERED)
            return attr
        return functools.partial(self._call, member, attr)

    def _busy(self, member: str) -> None:
        if (read := (_site(), member)) in self._fail:
            self._fail.discard(read)
            raise sqlite3.OperationalError("database is locked")

    def _store_raised(self, member: str, raised: BaseException) -> None:
        self._failed.append(raised)
        self._log(member, Outcome.STORE_RAISED)

    def _call(self, member: str, attr: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            self._busy(member)
        except sqlite3.OperationalError as raised:
            self._store_raised(member, raised)
            raise
        ran: list[Any] = []
        if member == "step":
            args = (args[0], self._watched(args[1], ran), *args[2:])
        held = self._held(member, args)
        try:
            answer = attr(*args, **kwargs)
        except Exception as raised:
            self._ended(member, raised, ran)
            raise
        self._log(member, self._outcome(member, answer, ran, held))
        return answer

    @staticmethod
    def _watched(thunk: Any, ran: list[Any]) -> Any:
        """`thunk`, noting in `ran` that it ran and what it raised."""

        def watched() -> Any:
            ran.append(None)
            try:
                return thunk()
            except BaseException as raised:
                ran.append(raised)
                raise

        return watched

    def _held(self, member: str, args: tuple[Any, ...]) -> bool:
        """Whether the store already holds what the call settles."""
        match member:
            case "settle":
                return self._ctx.peek_step(args[0])[0]
            case "await_until":
                return self._ctx.peek_step(args[2])[0]
            case _:
                return False

    def _ended(self, member: str, raised: BaseException, ran: list[Any]) -> None:
        """Log how a call ended that raised `raised`, its thunk having run as `ran` records."""
        match raised:
            case _ if any(failed is raised for failed in self._failed):
                self._log(member, Outcome.STORE_RAISED)  # a read inside the thunk failed
            case _ if any(r is raised for r in ran):
                refused = all_refusals(raised)
                self._log(member, Outcome.REFUSED if refused else Outcome.THUNK_RAISED)
            case _ if isinstance(raised, SIGNALS):
                self._log(member, Outcome.PARKED)
            case _:
                if ran:
                    self._log(member, Outcome.MISS)  # the thunk ran, and its write failed
                self._store_raised(member, raised)

    @staticmethod
    def _outcome(member: str, answer: Any, ran: list[Any], held: bool) -> Outcome:
        match member:
            case "step":
                return Outcome.MISS if ran else Outcome.HIT
            case "peek_step" | "peek_event":
                return Outcome.HIT if answer[0] else Outcome.MISS
            case "settle":
                return Outcome.HIT if held else Outcome.MISS
            case "await_event":
                return Outcome.HIT
            case "await_until":
                expired = type(answer).__name__ == "Expired"
                return Outcome.DUE if expired and not held else Outcome.HIT
            case "sleep_until":
                return Outcome.DUE
            case _:
                return Outcome.ANSWERED


PAST = datetime(2020, 1, 1, tzinfo=UTC)
FUTURE = datetime(2050, 1, 1, tzinfo=UTC)


def _refuse(op: Any) -> Any:
    raise Refused(op, "not this one")


def _fail(op: Any) -> Any:
    raise TimeoutError(op.name)


def steps():
    yield from call_tool("a", {}, dict)
    return (yield from await_event("go", dict))


def a_failing_tool():
    return (yield from call_tool("fail", {}, dict))


def a_gather_of_awaits():
    return (yield from gather([lambda: await_event("ev", dict)]))


def a_gather_of_sleeps():
    return (yield from gather([lambda: sleep_until(FUTURE)]))


def a_race():
    answer = yield from race([ok_branch, lambda: call_tool("refused", {}, dict)])
    return type(answer).__name__


def a_refused_race():
    answer = yield from race([lambda: call_tool("refused", {}, dict), ok_branch])
    return type(answer).__name__


def a_past_sleep():
    yield from sleep_until(PAST)
    return (yield from await_until("late", dict, deadline=PAST))


def a_replayed_race():
    answer = yield from race([lambda: call_tool("refused", {}, dict), ok_branch])
    yield from call_tool("flaky", {}, dict)
    return type(answer).__name__


@op_layer
def gate(op):
    """Refuses the `gated` tool, as a governor does, and forwards every other op."""
    if getattr(getattr(op, "op", None), "name", None) == "gated":
        raise Refused(op, "gated")
    return (yield op)


def a_gated_race():
    answer = yield from race([lambda: call_tool("gated", {}, dict), ok_branch])
    yield from call_tool("flaky", {}, dict)
    return type(answer).__name__


def a_future_sleep():
    yield from sleep_until(FUTURE)


def a_race_with_a_deadline():
    answer = yield from race([ok_branch], deadline=PAST)
    return type(answer).__name__


def a_failing_race_with_a_deadline():
    answer = yield from race([lambda: call_tool("fail", {}, dict)], deadline=FUTURE)
    return type(answer).__name__


def an_absolute_await():
    return (yield from await_event("absolute", dict, Addressing.ABSOLUTE))


def bounded():
    return (yield from await_until("bounded", dict, deadline=FUTURE))


WORKLOADS = {
    "steps": (steps, [("go", {"ok": True})]),
    "a failing tool": (a_failing_tool, []),
    "a gather of awaits": (a_gather_of_awaits, [("gather:0,0;ev", {"ok": True})]),
    "a gather of sleeps": (a_gather_of_sleeps, []),
    "a race": (a_race, []),
    "a refused race": (a_refused_race, []),
    "a past sleep": (a_past_sleep, []),
    "a bounded await": (bounded, [("bounded", {"ok": True})]),
    "a replayed race": (a_replayed_race, []),
    "a gated race": (a_gated_race, []),
    "a future sleep": (a_future_sleep, []),
    "a race with a deadline": (a_race_with_a_deadline, []),
    "a failing race with a deadline": (a_failing_race_with_a_deadline, []),
    "an absolute await": (an_absolute_await, [("absolute", {"ok": True})]),
}


@dataclass
class Attempted:
    """One attempt: the transitions it took, the handler's counts at its end, the sites that
    counted a read, and whether it raised."""

    seen: set[tuple[str, str, Outcome]] = field(default_factory=set)
    ran: int = 0
    read: int = 0
    counted_at: list[str] = field(default_factory=list)
    raised: bool = False


def observe(
    engine,
    workflow,
    fail: set[tuple[str, str]],
    emits,
    *,
    missing: type[ReparkReturns] | None = None,
    name: str = "case",
) -> list[Attempted]:
    """Each attempt of one run of `workflow`, emitting `emits` in turn after it parks."""
    attempts: list[Attempted] = []
    missed: set[str] = set()
    flaky = Flaky()
    read_clock = base.race_time
    unrecorded = durable._Unreplayed.unrecorded

    def clock() -> float:
        attempts[-1].seen.add((_site(), "race_clock", Outcome.ANSWERED))
        return read_clock()

    def counting(self: Any, counted: Any = None) -> None:
        attempts[-1].counted_at.append(_site())
        unrecorded(self, counted)

    def body(params, ctx):
        from _sweep import InOrder

        inner: Any = InOrder(ConcurrentAbsurdCtx(ctx) if isinstance(engine, AbsurdEngine) else ctx)
        if missing is not None:
            inner = missing(inner, missed)
        attempt = Attempted()
        attempts.append(attempt)
        tools = Tools({"fail": _fail, "refused": _refuse, "flaky": flaky})
        logged: Any = Logged(inner, attempt.seen, fail)
        handler = DurableHandler(logged, tools, op_layers=(gate,))
        try:
            return handler.run(workflow)
        except SIGNALS:
            raise
        except Exception:
            attempt.raised = True
            raise
        finally:
            attempt.ran, attempt.read = handler._unreplayed.ran, handler._unreplayed.read

    engine.register_task(name)(body)
    task_id = engine.spawn(name, {}, max_attempts=3)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "race_time", clock)
        patch.setattr(durable._Unreplayed, "unrecorded", counting)
        engine.run_until_result(task_id)
        for event, payload in emits:
            engine.emit_event(event, payload)
            engine.run_until_result(task_id)
    return attempts


def seen_in(attempts: list[Attempted]) -> set[tuple[str, str, Outcome]]:
    return set().union(*(attempt.seen for attempt in attempts))


def mislabeled(attempts: list[Attempted]) -> list[tuple[str, Attempted]]:
    """Each attempt whose counts disagree with the labels of the transitions it took.

    | the attempt     | holds when                                                          |
    |-----------------|---------------------------------------------------------------------|
    | any             | it ran something fresh exactly when it took a `fresh` transition    |
    | any             | each site that counted a read is named by a `counted` or `settled`  |
    |                 | transition it took                                                  |
    | one that raised | each `counted` transition it took was counted                       |
    """
    wrong = []
    for attempt in attempts:
        labels = [TRANSITIONS[key] for key in attempt.seen if key in TRANSITIONS]
        worlds = [label for label in labels if isinstance(label, World)]
        counted = {label.counted_at for label in worlds if label.kept_by == "counted"}
        settled = {label.counted_at for label in worlds if label.kept_by == "settled"}
        if (attempt.ran > 0) != ("fresh" in labels):
            wrong.append(("ran without a fresh transition, or the reverse", attempt))
        if not set(attempt.counted_at) <= counted | settled:
            wrong.append(("counted where no transition says", attempt))
        if attempt.raised and not counted <= set(attempt.counted_at):
            wrong.append(("raised with a counted transition uncounted", attempt))
    return wrong


@pytest.mark.parametrize("workload", WORKLOADS, ids=WORKLOADS)
def test_every_transition_an_engine_answers_is_tabled(engine, workload):
    """The workload runs as it is, then once per read it made with that read failing first."""
    workflow, emits = WORKLOADS[workload]
    attempts = observe(engine, workflow, set(), emits)
    assert {key for key in seen_in(attempts) if key[2] is Outcome.STORE_RAISED} == set()
    for site, member, _ in sorted(seen_in(attempts)):
        attempts += observe(engine, workflow, {(site, member)}, emits, name=f"{site}:{member}")
    assert seen_in(attempts) - TRANSITIONS.keys() == set()
    assert mislabeled(attempts) == []


@pytest.mark.parametrize("workload", ["a race", "a refused race", "a replayed race"])
def test_a_race_with_no_deadline_reads_no_clock(engine, workload):
    """No branch can end too late for a race with no deadline, so it never reads the clock."""
    workflow, emits = WORKLOADS[workload]
    clocks = {
        key for key in seen_in(observe(engine, workflow, set(), emits)) if key[1] == "race_clock"
    }
    assert clocks == set()


class MissesPeek(ReparkReturns):
    """`ReparkReturns`, except that its repark parks as the engine's does."""

    def repark(self, name: Any, /) -> None:
        return self._ctx.repark(name)


def test_a_store_error_in_a_branch_is_counted_once(engine):
    """The branch's handler counts it, and the count climbs to the task's handler once."""
    counts: list[tuple[int, int]] = []
    fail = {("DurableHandler._checkpointed", "step")}

    def gathered():
        return (yield from gather([lambda: call_tool("a", {}, dict)]))

    def body(params, ctx):
        from _sweep import InOrder

        inner: Any = InOrder(ConcurrentAbsurdCtx(ctx) if isinstance(engine, AbsurdEngine) else ctx)
        logged: Any = Logged(inner, set(), fail)
        handler = DurableHandler(logged, Tools())
        try:
            return handler.run(gathered)
        finally:
            counts.append((handler._unreplayed.ran, handler._unreplayed.read))

    engine.register_task("case")(body)
    engine.run_until_result(engine.spawn("case", {}, max_attempts=2))
    assert counts[0] == (0, 1)


@pytest.mark.parametrize(
    ("missing", "repark"),
    [(ReparkReturns, Outcome.ANSWERED), (MissesPeek, Outcome.PARKED)],
    ids=["repark returns", "repark parks"],
)
def test_a_wake_race_an_engine_answers_is_tabled(engine, missing, repark):
    """Each branch's event arrives between its peek and the re-arm."""

    def workflow():
        return (yield from gather([lambda: await_event("ev", dict)]))

    engine.emit_event("gather:0,0;ev", {"ok": True})
    attempts = observe(engine, workflow, set(), [], missing=missing)
    assert ("DurableHandler._join", "repark", repark) in seen_in(attempts)
    assert seen_in(attempts) - TRANSITIONS.keys() == set()
    assert mislabeled(attempts) == []


def test_the_table_names_its_sites_by_their_qualified_names():
    names = set(_functions(durable)) | set(_functions(base))
    assert {site for site, _, _ in TRANSITIONS} - names == set()
