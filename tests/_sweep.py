"""A generated sweep over how combinators compose with failures, under two retry oracles.

Branches run in index order. A tree is JSON: `["leaf", kind]`, `["seq", a, b]`,
`["gather", a, b]`, `["race", a, b]` or `["scoped", a]`. Each runs as one task of `ATTEMPTS`
attempts, and its outcome is named by a class: its end state, the kind of its failure, and whether
it ran once, every attempt, or between.

| leaf             | its tool                                                          |
|------------------|-------------------------------------------------------------------|
| `ok`             | returns a value                                                   |
| `fresh_fail`     | raises `TimeoutError` on every call                               |
| `flaky`          | raises `RuntimeError` on its first call, then returns             |
| `refuse`         | raises `Refused`                                                  |
| `code_error`     | returns, and the workflow raises `KeyError` on it                 |
| `reject`         | returns a value its schema rejects                                |
| `fallback`       | raises `TimeoutError` on its first call, which a layer answers    |
|                  | with a value it never records, and the workflow raises on it      |
| `default`        | returns an empty record whose schema default is 0 on its first    |
|                  | load and 1 after, and the workflow raises on a 0                  |
| `layer_raises`   | returns, and a layer raises `KeyError` on the recorded result     |
| `bad_name`       | none: the step's name is no atom                                  |
| `factory`        | none: a scope's body factory raises                               |
| `unserializable` | returns, and the workflow returns a value with no JSON form       |

A task may spend one attempt to learn that its failure repeats, so each oracle allows one retry
after the attempt it judges:

| oracle    | flags a retry that follows                                       | as        |
|-----------|------------------------------------------------------------------|-----------|
| `replays` | an attempt that ran no fresh effect and raised what the attempt  | a finding |
|           | before it raised                                                 |           |
| `origin`  | two attempts in a row whose failures each hold a leaf no effect  | a spend   |
|           | raised fresh on that attempt, refusals aside                     |           |

A third oracle, `race`, flags a race that fails while one of its branches completes when run alone:
a sibling that can win is never lost to a branch's error.

`replays` is the guarantee the edge makes. `origin` covers it and also sees a futile leaf that
shares its attempt with a fresh effect, as in a gather holding a fresh failure beside a code error,
which the edge keeps retrying because the fresh effect might have changed what the code read. A run
`replays` flags and `origin` does not is a finding about the oracles.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, Field

from effective.api import Effect, call_tool, gather, race, scoped, step
from effective.combinators import Settled, fixpoint
from effective.domain import CallTool
from effective.engines import open
from effective.govern import REFUSALS, Refused
from effective.handlers.durable import DurableHandler
from effective.keys import Segment, compose_key
from effective.layers import op_layer
from effective.ops import Step, leaves

ATTEMPTS = 5
LEAVES = (
    "ok",
    "fresh_fail",
    "flaky",
    "refuse",
    "code_error",
    "reject",
    "fallback",
    "default",
    "layer_raises",
    "bad_name",
    "factory",
    "unserializable",
)
BINARY = ("seq", "gather", "race")


class Shape(BaseModel):
    """The schema a `reject` leaf's recorded result fails."""

    approve: bool


class InOrder:
    """A ctx that declines concurrency, so a gather or a race runs its branches in index order and
    a tree's outcome does not depend on which thread finishes first. Interleavings are an axis of
    their own."""

    concurrent_safe = False

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


@dataclass
class Attempted:
    """One attempt: the effects it ran fresh, and the error it raised, if any."""

    fresh: int = 0
    raised: tuple[str, ...] = ()
    from_effects: bool = True


@dataclass
class Probe:
    """The domain a tree runs against, counting per attempt what ran fresh and what it raised."""

    attempts: list[Attempted] = field(default_factory=list)
    flaked: set[str] = field(default_factory=set)
    raised_fresh: list[BaseException] = field(default_factory=list)
    loaded: set[str] = field(default_factory=set)

    def run(self, op: Any) -> Any:
        self.attempts[-1].fresh += 1
        kind = op.name.split(".", 1)[0]
        match kind:
            case "fresh_fail":
                self._raise(TimeoutError(op.name))
            case "flaky" if op.name not in self.flaked:
                self.flaked.add(op.name)
                self._raise(RuntimeError(op.name))
            case "fallback" if op.name not in self.flaked:
                self.flaked.add(op.name)
                self._raise(TimeoutError(op.name))
            case "refuse":
                raise Refused(op, "refused")
            case "reject":
                return {"approve": "maybe"}
            case "default":
                return {}
            case _:
                return {"n": 1}

    def default(self, path: str) -> int:
        """The default a `default` leaf's schema fills: 0 on its first load, then 1."""
        first = path not in self.loaded
        self.loaded.add(path)
        return 0 if first else 1

    def _raise(self, error: BaseException) -> None:
        self.raised_fresh.append(error)
        raise error


@op_layer
def layered(op: Any) -> Any:
    """The layer the `fallback` and `layer_raises` leaves run under; other steps pass through."""
    kind = getattr(op.op, "name", "").split(".", 1)[0] if isinstance(op, Step) else ""
    match kind:
        case "fallback":
            try:
                return (yield op)
            except TimeoutError:
                return {}
        case "layer_raises":
            got = yield op
            return got["missing"]
        case _:
            return (yield op)


def reads(kind: str, path: str, field: str) -> Callable[[], Effect[Any]]:
    """A leaf whose workflow reads `field` of its tool's result."""

    def read() -> Effect[Any]:
        got = yield from call_tool(f"{kind}.{path}", {}, dict)
        return got[field]

    return read


def defaulted(path: str, probe: Probe) -> Callable[[], Effect[Any]]:
    class Defaulted(BaseModel):
        n: int = Field(default_factory=lambda: probe.default(path))

    def workflow() -> Effect[Any]:
        got = yield from call_tool(f"default.{path}", {}, Defaulted)
        if got.n == 0:
            raise ValueError("the default was 0")
        return got.n

    return workflow


def failed_factory(path: str) -> Callable[[], Effect[Any]]:
    def factory() -> Effect[Any]:
        raise ValueError("the scope's body failed to build")

    return lambda: scoped(compose_key(t"factory:{Segment(path)}"), factory)


def unserializable(path: str) -> Callable[[], Effect[Any]]:
    def workflow() -> Effect[Any]:
        yield from call_tool(f"unserializable.{path}", {}, dict)
        return object()

    return workflow


def leaf(kind: str, path: str, probe: Probe) -> Callable[[], Effect[Any]]:
    """The workflow of one leaf, its tool named by the leaf's place in the tree."""
    match kind:
        case "code_error":
            return reads(kind, path, "missing")
        case "fallback":
            return reads(kind, path, "n")
        case "reject":
            return lambda: call_tool(f"reject.{path}", {}, Shape)
        case "default":
            return defaulted(path, probe)
        case "bad_name":
            return lambda: step(f"bad/{path}", CallTool(name=f"bad.{path}", result_schema=dict))
        case "factory":
            return failed_factory(path)
        case "unserializable":
            return unserializable(path)
        case _:
            return lambda: call_tool(f"{kind}.{path}", {}, dict)


def program(tree: list[Any], probe: Probe, path: str = "t") -> Callable[[], Effect[Any]]:
    """The workflow a tree spells, its tools named by their place in it."""
    match tree:
        case ["leaf", kind]:
            return leaf(kind, path, probe)
        case ["seq", a, b]:
            first, then = program(a, probe, path + "0"), program(b, probe, path + "1")

            def seq() -> Effect[Any]:
                return [(yield from first()), (yield from then())]

            return seq
        case ["gather", a, b]:
            branches = [program(a, probe, path + "0"), program(b, probe, path + "1")]
            return lambda: gather(branches)
        case ["race", a, b]:
            branches = [program(a, probe, path + "0"), program(b, probe, path + "1")]

            def raced() -> Effect[Any]:
                answer = yield from race(branches)
                return type(answer).__name__

            return raced
        case ["scoped", a]:
            body = program(a, probe, path + "0")
            return lambda: scoped(compose_key(t"scope:{Segment(path)}"), body)
        case _:
            raise ValueError(f"not a tree: {tree!r}")


@dataclass(frozen=True)
class Outcome:
    """A tree's run: its class, each oracle's verdict, and what it cost."""

    tree: list[Any]
    cls: str
    replays: bool
    origin: bool
    seconds: float

    @property
    def disagrees(self) -> bool:
        """`replays` flagged a retry that `origin`, which covers it, did not."""
        return self.replays and not self.origin


def replayed(attempts: list[Attempted]) -> bool:
    """Whether a retry followed an attempt that ran no fresh effect and raised what the attempt
    before it raised."""
    return any(
        later.fresh == 0 and later.raised and later.raised == earlier.raised
        for earlier, later in pairwise(attempts[:-1])
    )


def spent(attempts: list[Attempted]) -> bool:
    """Whether a retry followed two attempts in a row whose failures each held a leaf no effect
    raised fresh."""
    return any(
        not earlier.from_effects and not later.from_effects
        for earlier, later in pairwise(attempts[:-1])
    )


def run(engine: Any, tree: list[Any], name: str) -> Outcome:
    """Run `tree` as one task named `name` on `engine`, and classify it."""
    probe = Probe()
    workflow = program(tree, probe)

    def body(params: Any, ctx: Any) -> Any:
        probe.attempts.append(Attempted())
        before = len(probe.raised_fresh)
        try:
            in_order: Any = InOrder(ctx)
            return DurableHandler(in_order, probe, op_layers=[layered]).run(workflow)
        except Exception as raised:
            fresh = probe.raised_fresh[before:]
            probe.attempts[-1].raised = tuple(type(leaf).__name__ for leaf in leaves(raised))
            probe.attempts[-1].from_effects = all(
                any(leaf is error for error in fresh) or isinstance(leaf, REFUSALS)
                for leaf in leaves(raised)
            )
            raise

    started = time.perf_counter()
    engine.register_task(name)(body)
    snapshot = engine.run_until_result(engine.spawn(name, {}, max_attempts=ATTEMPTS))
    seconds = time.perf_counter() - started
    assert snapshot is not None
    ran = len(probe.attempts)
    count = "1" if ran == 1 else "all" if ran == ATTEMPTS else "some"
    kind = "-" if snapshot.failure is None else snapshot.failure.kind
    return Outcome(
        tree=tree,
        cls=f"{snapshot.state}:{kind}:{count}",
        replays=replayed(probe.attempts),
        origin=spent(probe.attempts),
        seconds=seconds,
    )


def composed(representatives: list[list[Any]]) -> list[list[Any]]:
    """One round: every combinator over every pair of representatives, and every scope."""
    binary = [[op, a, b] for op in BINARY for a in representatives for b in representatives]
    return binary + [["scoped", a] for a in representatives]


def initial() -> list[list[Any]]:
    return [["leaf", kind] for kind in LEAVES]


def run_alone(tree: list[Any], name: str) -> Outcome:
    """Run `tree` on a SQLite engine of its own."""
    engine = open("sqlite://")
    try:
        return run(engine, tree, name)
    finally:
        engine.close()


def seeded() -> dict[str, Any]:
    """The first value: each leaf run once, by the class it lands in."""
    classes: dict[str, Any] = {}
    for i, tree in enumerate(initial()):
        classes.setdefault(run_alone(tree, f"leaf{i}").cls, tree)
    return {"classes": classes, "findings": [], "spent": [], "rounds": []}


def lost_a_winner(tree: list[Any], cls: str, alone: dict[str, str]) -> bool:
    """Whether `tree` is a race that failed while one of its branches completes when run alone,
    each branch's class read from `alone`."""
    match tree:
        case ["race", *branches] if cls.startswith("failed"):
            return any(alone[json.dumps(branch)].startswith("completed") for branch in branches)
        case _:
            return False


class Rounds:
    """The domain that runs one round of the sweep, every tree of it on a SQLite engine of its
    own, keeping the smallest tree that showed each class."""

    def run(self, op: Any) -> Any:
        value = op.args["value"]
        started = time.perf_counter()
        classes, findings, spends = (
            dict(value["classes"]),
            list(value["findings"]),
            list(value["spent"]),
        )
        alone = {json.dumps(tree): cls for cls, tree in classes.items()}
        batch = composed(list(classes.values()))
        for i, tree in enumerate(batch):
            outcome = run_alone(tree, f"t{i}")
            known = classes.get(outcome.cls)
            if known is None or len(json.dumps(tree)) < len(json.dumps(known)):
                classes[outcome.cls] = tree
            if outcome.replays:
                findings.append(
                    {
                        "oracle": "replays",
                        "cls": outcome.cls,
                        "tree": tree,
                        "disagrees": outcome.disagrees,
                    }
                )
            if lost_a_winner(tree, outcome.cls, alone):
                findings.append({"oracle": "race", "cls": outcome.cls, "tree": tree})
            if outcome.origin:
                spends.append(tree)
        seconds = round(time.perf_counter() - started, 3)
        rounds = [*value["rounds"], {"trees": len(batch), "seconds": seconds}]
        return {"classes": classes, "findings": findings, "spent": spends, "rounds": rounds}


def same_classes(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return set(before["classes"]) == set(after["classes"])


def closure(first: dict[str, Any], budget: int) -> Effect[Settled[dict[str, Any]]]:
    """The classes closed under composition: rounds until one shows no class the last did not."""

    def step(value: dict[str, Any]) -> Effect[dict[str, Any]]:
        return (yield from call_tool("round", {"value": value}, dict))

    return (
        yield from fixpoint(first, step, budget=budget, converged=same_classes, run_id="sweep")
    )


def differ_on_absurd(classes: dict[str, Any], dsn: str) -> list[tuple[str, str, list[Any]]]:
    """Each class whose tree lands in another class on Absurd, each run on a queue of its own."""
    from uuid import uuid4

    from effective.engines.absurd import AbsurdEngine

    differ = []
    for cls, tree in classes.items():
        engine = open(dsn, queue="sweep_" + uuid4().hex[:10])
        assert isinstance(engine, AbsurdEngine)
        engine.app.create_queue()
        try:
            outcome = run(engine, tree, "confirm")
        finally:
            engine.app.drop_queue()
            engine.close()
        if outcome.cls != cls:
            differ.append((cls, outcome.cls, tree))
    return differ
