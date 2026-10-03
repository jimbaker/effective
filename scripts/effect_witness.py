"""What caused each durable write: the op on the stream, its guards, and the line that yielded it.

A pytest plugin in the shape of `vacuity_probe.py`, loaded only by this script. Three sensors write
one row per durable write attempt, tagged with the running test:

- **authored site**: `sys.monitoring` `PY_YIELD` sees every frame of a `yield from` chain, so the
  line an author wrote is read off the chain for the op the handler received, with no frame walk;
- **op stream**: `drive_through`, as each handler module binds it, marks the op, its op-layers, the
  handler, and the op the base is executing, which differs from the driven op when a layer
  injects or rewrites one;
- **write**: the SQLite checkpoint write and ledger append, `ForkLedger`, the durable handler's
  ledger write, and the recording handler's ledger and artifact arms. A row records whether the
  store wrote: a checkpoint hit and a folded re-append are attempts.

A write sits on the stream when the op the base is executing can make it (a ledger row under
`AppendLedgerRow`) and the store belongs to the driving handler, found through the attribute
names the handlers keep it under. Otherwise it is `outside`, and `beneath` names the drive it
happened inside.

The domain is RecordingHandler and the SQLite engine: a row from any other store carries its engine
and is counted, never joined. The plugin runs in one process, so the command refuses xdist.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import threading
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, assert_never, get_args, get_origin

from effective.keys.frame import split_frames
from effective.keys.grammar import parse
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Gather,
    Race,
    Respawn,
    Scoped,
    SleepUntil,
    Step,
    StoreArtifact,
    WorkflowOp,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "effect-witness.jsonl"
_ROOT_PREFIX = f"{ROOT}/"
_VENV_PREFIX = f"{ROOT}/.venv/"
_THIS = str(Path(__file__).resolve())

#: Frames an author did not write: the convenience wrappers every workflow yields through.
CONVENIENCE = frozenset({"src/effective/api.py", "src/effective/coroutine_api.py"})

#: The op-layers that decide whether an op may proceed, by qualname. Anything else in a stack is
#: reported by name and counted by no one.
GUARDS = frozenset(
    {
        "effective.permission.cascade.<locals>.gate",
        "effective.govern.govern.<locals>.gate_layer",
    }
)

#: Method names that delegate a write rather than author it; a writer site skips past them.
DELEGATES = frozenset({"step", "append", "_record_ledger"})

type Effect = Literal["checkpoint", "ledger", "artifact"]
EFFECTS: tuple[Effect, ...] = ("checkpoint", "ledger", "artifact")
IN_DOMAIN = frozenset({"sqlite", "recording"})
POSITIONS = ("outside", "stream, 0 guards", "stream, 1 guard", "stream, 2 or more")

_WRITER_DEPTH = 10


def _in_repo(filename: str) -> bool:
    return (
        filename.startswith(_ROOT_PREFIX)
        and not filename.startswith(_VENV_PREFIX)
        and filename != _THIS
    )


@dataclass(frozen=True, slots=True)
class Frame:
    """A source location: the unit every site, writer and chain in a row is made of."""

    path: str
    line: int
    qualname: str

    @classmethod
    def of(cls, code: Any, line: int) -> Frame:
        return cls(code.co_filename.removeprefix(_ROOT_PREFIX), line, code.co_qualname)

    def render(self) -> str:
        return f"{self.path}:{self.line} {self.qualname}"


def _frame(fields: dict | None) -> Frame | None:
    return None if fields is None else Frame(**fields)


def _line_at(code: Any, offset: int) -> int:
    for start, end, line in code.co_lines():
        if start <= offset < end and line is not None:
            return line
    return code.co_firstlineno


def layer_name(layer: object) -> str:
    module = getattr(layer, "__module__", None)
    qualname = getattr(layer, "__qualname__", None)
    return f"{module}.{qualname}" if module and qualname else type(layer).__qualname__


# ---------------------------------------------------------------- the authored-site sensor


class YieldSites:
    """The yield-from chain of each `WorkflowOp`, innermost frame first, per thread.

    A chain is matched to its op by identity, so an op yielded and never driven (an unlayered
    op, a replay) cannot lend its chain to the next one. Code outside the repo is disabled at its
    first yield."""

    def __init__(self, op_types: tuple[type, ...]) -> None:
        self.op_types = op_types
        self._in_repo: dict[Any, bool] = {}
        self._where: dict[tuple[Any, int], Frame] = {}
        self._local = threading.local()
        self.tool: int | None = None

    def _buffer(self) -> deque:
        buf = getattr(self._local, "buf", None)
        if buf is None:
            buf = self._local.buf = deque(maxlen=256)
        return buf

    def on_yield(self, code: Any, offset: int, value: object) -> object:
        in_repo = self._in_repo.get(code)
        if in_repo is None:
            in_repo = self._in_repo[code] = _in_repo(code.co_filename)
        if not in_repo:
            return sys.monitoring.DISABLE
        if isinstance(value, self.op_types):
            self._buffer().append((value, code, offset))
        return None

    def take(self, op: object) -> tuple[Frame, ...]:
        """The chain that yielded `op`; the buffer is emptied."""
        buf = self._buffer()
        chain = tuple(self._describe(code, offset) for value, code, offset in buf if value is op)
        buf.clear()
        return chain

    def clear(self) -> None:
        self._buffer().clear()

    def _describe(self, code: Any, offset: int) -> Frame:
        if (where := self._where.get((code, offset))) is None:
            where = self._where[(code, offset)] = Frame.of(code, _line_at(code, offset))
        return where

    def start(self) -> None:
        monitoring = sys.monitoring
        free = (t for t in (4, 3, 2, 5, 0) if monitoring.get_tool(t) is None)
        if (tool := next(free, None)) is None:
            raise RuntimeError("every sys.monitoring tool slot but coverage's is taken")
        self.tool = tool
        monitoring.use_tool_id(tool, "effect-witness")
        monitoring.register_callback(tool, monitoring.events.PY_YIELD, self.on_yield)
        monitoring.set_events(tool, monitoring.events.PY_YIELD)

    def stop(self) -> None:
        if self.tool is not None:
            sys.monitoring.set_events(self.tool, 0)
            sys.monitoring.register_callback(self.tool, sys.monitoring.events.PY_YIELD, None)
            sys.monitoring.free_tool_id(self.tool)
            self.tool = None


def authored(chain: Iterable[Frame]) -> Frame | None:
    """The innermost frame outside the convenience wrappers: the line an author wrote."""
    return next((frame for frame in chain if frame.path not in CONVENIENCE), None)


# ---------------------------------------------------------------- the op-stream sensor


@dataclass(frozen=True, slots=True, eq=False)
class Drive:
    op: WorkflowOp
    layers: tuple[str, ...]
    guards: int
    site: Frame | None
    handler: object | None


@dataclass(frozen=True, slots=True, eq=False)
class Based:
    """The op a drive's base is executing, and the chain that yielded it when a layer did."""

    drive: Drive
    op: WorkflowOp
    chain: tuple[Frame, ...]


_DRIVE: ContextVar[tuple[Drive, ...]] = ContextVar("effect_witness_drive", default=())
_BASED: ContextVar[Based | None] = ContextVar("effect_witness_based", default=None)


@dataclass(frozen=True, slots=True, eq=False)
class Forwarding:
    """The base a drive hands its layer stack: it marks the op the base is executing.

    A witness installed over this one receives a `Forwarding` as its base, and reads `handler`
    from it to learn whose stores the drive owns."""

    drive: Drive
    base: Callable[[Any], Any]
    sites: YieldSites

    @property
    def handler(self) -> object | None:
        return self.drive.handler

    def __call__(self, inner: WorkflowOp) -> Any:
        chain = () if inner is self.drive.op else self.sites.take(inner)
        token = _BASED.set(Based(self.drive, inner, chain))
        try:
            return self.base(inner)
        finally:
            _BASED.reset(token)


def position(guards: int | None) -> str:
    match guards:
        case None:
            return "outside"
        case 0:
            return "stream, 0 guards"
        case 1:
            return "stream, 1 guard"
        case _:
            return "stream, 2 or more"


def _stores(owner: object | None, *names: str) -> list[object]:
    """`owner` and everything it wraps through `names`, outermost first."""
    chain: list[object] = []
    while owner is not None and owner not in chain:
        chain.append(owner)
        owner = next((vars(owner)[n] for n in names if n in getattr(owner, "__dict__", {})), None)
    return chain


def can_write(effect: Effect, op: WorkflowOp) -> bool:
    """Whether executing `op` makes a write of this kind: total over `WorkflowOp`."""
    match op:
        case AppendLedgerRow():
            return effect in ("ledger", "checkpoint")
        case StoreArtifact():
            return effect in ("artifact", "checkpoint")
        case Step() | Race():
            return effect == "checkpoint"  # a race checkpoints its choice and its endings
        case SleepUntil() | AwaitEvent() | Gather() | Scoped() | Respawn():
            return False
        case unreachable:
            assert_never(unreachable)


def owns(handler: object | None, effect: Effect, store: object) -> bool:
    """Whether `store` is the one `handler` writes this kind of effect to."""
    match effect:
        case "checkpoint":
            return store in _stores(getattr(handler, "ctx", None), "_ctx", "ctx")
        case "ledger":
            return store is handler or store in _stores(getattr(handler, "ledger", None), "_base")
        case "artifact":
            return store is handler
        case unreachable:
            assert_never(unreachable)


def attributed(stack: tuple[Drive, ...], based: Based | None) -> Drive | None:
    """The drive whose base is executing, when every drive pushed after it drives the same op.

    A later drive of the same op is this drive seen by a second witness installed over this one;
    a later drive of another op is a nested handler, and its layers are what is running."""
    if based is None:
        return None
    index = next((i for i, drive in enumerate(stack) if drive is based.drive), None)
    if index is None:
        return None
    later = stack[index + 1 :]
    return based.drive if all(drive.op is based.drive.op for drive in later) else None


# ---------------------------------------------------------------- the recorder


def writer_frames(skip: int) -> list[Frame]:
    """The repo frames above a sensor, nearest first."""
    frames: list[Frame] = []
    frame = sys._getframe(skip)
    while frame is not None and len(frames) < _WRITER_DEPTH:
        if _in_repo(frame.f_code.co_filename):
            frames.append(Frame.of(frame.f_code, frame.f_lineno))
        frame = frame.f_back
    return frames


def writer_site(frames: list[Frame]) -> Frame | None:
    """The first frame that authored the write rather than delegating it."""
    return next((f for f in frames if f.qualname.rsplit(".", 1)[-1] not in DELEGATES), None)


def identity_class(identity: str) -> str:
    """A durable name's head tag, with its frames dropped."""
    return parse(split_frames(identity)[1]).terms[0].tag


@dataclass(slots=True)
class Pending:
    """One write in progress, which the innermost store call beneath it completes."""

    effect: Effect
    store: object
    frames: list[Frame]
    engine: str = "unscanned"
    identity: str | None = None
    written: bool = True
    nested_frames: list[Frame] = field(default_factory=list)


class Witness:
    """The plugin. `rows` is the run's evidence; `install` and `uninstall` bracket it."""

    def __init__(self, out: Path | None = OUT) -> None:
        self.out = out
        self.current: str | None = "<outside pytest>" if out is None else None
        self.rows: list[dict] = []
        self._pending = threading.local()
        self._restore: list[tuple[type | ModuleType, str, object]] = []
        self.sites: YieldSites | None = None

    # -- pytest hooks

    def pytest_configure(self, config) -> None:
        self.install()

    def pytest_unconfigure(self, config) -> None:
        self.uninstall()

    def pytest_runtest_logstart(self, nodeid: str, location) -> None:
        self.current = nodeid

    def pytest_runtest_logfinish(self, nodeid: str, location) -> None:
        self.current = None

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if self.out is None:
            return
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("w") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")

    # -- the sensors

    def _open(self, key: str) -> Pending | None:
        return getattr(self._pending, "open", {}).get(key)

    def _write(
        self,
        key: str | None,
        effect: Effect,
        store: object,
        call: Callable[[], Any],
        finish: Callable[[Pending, Any], None],
    ) -> Any:
        """Run `call` as one write. A `key` names a delegation chain: a call under an open write
        of the same key is that write, one store further down."""
        opened: dict[str, Pending] = self._pending.__dict__.setdefault("open", {})
        if key is not None and (outer := opened.get(key)) is not None:
            outer.nested_frames.extend(writer_frames(3))
            return call()
        pending = Pending(effect, store, writer_frames(3))
        if key is not None:
            opened[key] = pending
        try:
            result = call()
        finally:
            if key is not None:
                del opened[key]
        finish(pending, result)
        self.record(pending)
        return result

    def record(self, pending: Pending) -> None:
        if self.current is None:
            return
        if pending.identity is None:
            raise ValueError(f"a {pending.effect} write reached the recorder with no identity")
        stack = _DRIVE.get()
        based = _BASED.get()
        drive = attributed(stack, based)
        caused = (
            drive is not None
            and based is not None
            and can_write(pending.effect, based.op)
            and owns(drive.handler, pending.effect, pending.store)
        )
        enclosing = stack[-1] if stack else None
        injected = caused and based is not None and drive is not None and based.op is not drive.op
        site = None
        if caused and based is not None and drive is not None:
            site = authored(based.chain) if injected else drive.site
        writer = writer_site(pending.frames)
        task = getattr(pending.store, "task_id", None)
        self.rows.append(
            {
                "test": self.current,
                "effect": pending.effect,
                "engine": pending.engine,
                "written": pending.written,
                "class": identity_class(pending.identity)
                if pending.effect == "checkpoint"
                else None,
                "identity": pending.identity,
                "scope": None if task is None else str(task),
                "position": position(drive.guards if caused and drive else None),
                "beneath": None if caused or enclosing is None else type(enclosing.op).__name__,
                "injected": injected,
                "op": type(based.op).__name__ if caused and based else None,
                "layers": list(drive.layers) if caused and drive else [],
                "site": None if site is None else asdict(site),
                "writer": None if writer is None else asdict(writer),
                "frames": [asdict(f) for f in pending.frames + pending.nested_frames],
            }
        )

    def _patch(self, owner: type | ModuleType, name: str, replacement: object) -> None:
        self._restore.append((owner, name, vars(owner)[name]))
        setattr(owner, name, replacement)

    def install(self) -> None:
        arms = tuple(get_origin(arm) or arm for arm in get_args(WorkflowOp.__value__))
        self.sites = YieldSites(arms)
        self.sites.start()
        self._install_op_stream(self.sites)
        self._install_checkpoint()
        self._install_ledger()
        self._install_recording()

    def _install_op_stream(self, sites: YieldSites) -> None:
        from effective.handlers import absurd, recording

        def driving(original: Callable[..., Any]) -> Callable[..., Any]:
            def drive_through(op_layers, op, base):
                names = tuple(layer_name(layer) for layer in op_layers)
                drive = Drive(
                    op=op,
                    layers=names,
                    guards=sum(name in GUARDS for name in names),
                    site=authored(sites.take(op)),
                    handler=getattr(base, "__self__", None) or getattr(base, "handler", None),
                )
                token = _DRIVE.set((*_DRIVE.get(), drive))
                try:
                    return original(op_layers, op, Forwarding(drive, base, sites))
                finally:
                    _DRIVE.reset(token)
                    sites.clear()

            return drive_through

        for module in (recording, absurd):
            self._patch(module, "drive_through", driving(vars(module)["drive_through"]))

    def _install_checkpoint(self) -> None:
        from effective.sqlite import SqliteTaskContext

        witness = self
        original_step = SqliteTaskContext.step

        def step(ctx, name, thunk, /):
            ran: list[bool] = []

            def watched():
                ran.append(True)
                return thunk()

            def call():
                return original_step(ctx, name, watched)

            def finish(pending: Pending, _result: object) -> None:
                pending.engine = "sqlite"
                pending.identity = name.stored()
                pending.written = bool(ran)

            return witness._write(None, "checkpoint", ctx, call, finish)

        self._patch(SqliteTaskContext, "step", step)

    def _install_ledger(self) -> None:
        from effective.counterfactual import ForkLedger
        from effective.handlers import absurd
        from effective.sqlite import SqliteLedger

        witness = self
        original_sqlite = SqliteLedger.append
        original_fork = ForkLedger.append
        original_record = absurd.DurableHandler._record_ledger

        def delegating(original: Callable[..., Any]) -> Callable[..., Any]:
            """A writer above the store: the store beneath it, if SQLite, fills in the rest."""

            def append(owner, row, *args, **kwargs):
                def call():
                    return original(owner, row, *args, **kwargs)

                def finish(pending: Pending, _result: object) -> None:
                    pending.identity = pending.identity or row.event_id.stored()

                return witness._write("ledger", "ledger", owner, call, finish)

            return append

        record_ledger = delegating(original_record)

        def durable_record_ledger(handler, row):
            if handler.ledger is None:  # the no-commit mode: a checkpoint, no append
                return original_record(handler, row)
            return record_ledger(handler, row)

        self._patch(SqliteLedger, "append", self._store_sensor(original_sqlite))
        self._patch(ForkLedger, "append", delegating(original_fork))
        self._patch(absurd.DurableHandler, "_record_ledger", durable_record_ledger)

    def _store_sensor(self, original_sqlite: Callable[..., Any]) -> Callable[..., Any]:
        """The SQLite ledger: it tells the write in progress what was stored, and whether."""
        witness = self

        def sqlite_append(ledger, row, *args, **kwargs):
            before = ledger.conn.total_changes

            def call():
                return original_sqlite(ledger, row, *args, **kwargs)

            def finish(pending: Pending, _result: object) -> None:
                pending.engine = "sqlite"
                pending.identity = row.event_id.stored()
                pending.written = ledger.conn.total_changes > before

            if (outer := witness._open("ledger")) is None:
                return witness._write("ledger", "ledger", ledger, call, finish)
            outer.nested_frames.extend(writer_frames(2))
            result = call()
            finish(outer, result)
            return result

        return sqlite_append

    def _install_recording(self) -> None:
        from effective.handlers import recording

        witness = self
        original_interpret = recording.RecordingHandler._interpret
        arm = Frame.of(original_interpret.__code__, original_interpret.__code__.co_firstlineno)

        def interpret(handler, op):
            result = original_interpret(handler, op)
            match op:
                case AppendLedgerRow(row=row):
                    effect: Effect = "ledger"
                    identity = row.event_id.stored()
                case StoreArtifact():
                    effect = "artifact"
                    identity = str(result)
                case _:
                    return result
            frames = [arm, *writer_frames(2)]
            witness.record(Pending(effect, handler, frames, "recording", identity))
            return result

        self._patch(recording.RecordingHandler, "_interpret", interpret)

    def uninstall(self) -> None:
        while self._restore:
            owner, name, original = self._restore.pop()
            setattr(owner, name, original)
        if self.sites is not None:
            self.sites.stop()
            self.sites = None


# ---------------------------------------------------------------- the joins


def _substrate(where: dict | None) -> bool:
    return where is not None and where["path"].startswith("src/")


def _render(where: dict | None) -> str:
    return "(none)" if (frame := _frame(where)) is None else frame.render()


def in_domain(rows: list[dict]) -> list[dict]:
    """The writes the joins read: a store in the declared domain, and a store that wrote."""
    return [row for row in rows if row["engine"] in IN_DOMAIN and row["written"]]


def join_a(rows: list[dict]) -> str:
    """Join A: every write's position relative to the op stream, split by who wrote it."""
    unscanned = Counter(row["engine"] for row in rows if row["engine"] not in IN_DOMAIN)
    attempts = sum(1 for row in rows if row["engine"] in IN_DOMAIN and not row["written"])
    rows = in_domain(rows)
    table: Counter = Counter()
    outside: Counter = Counter()
    for row in rows:
        author = "substrate" if _substrate(row["writer"]) else "test"
        table[(row["effect"], row["position"], author)] += 1
        if row["position"] == "outside" and author == "substrate":
            writer = _render(row["writer"])
            outside[(row["effect"], row["class"] or "", writer, row["beneath"])] += 1
    lines = ["JOIN A: position relative to the op stream", ""]
    lines.append(
        "out of domain, counted and not joined: "
        + (", ".join(f"{n} {engine}" for engine, n in unscanned.items()) or "none")
    )
    lines.append(f"attempts that did not write (a hit, a folded append): {attempts}")
    lines.append("")
    lines.append(f"{'effect':<11} {'position':<18} {'substrate':>10} {'test':>8}")
    for effect in EFFECTS:
        for pos in POSITIONS:
            sub, test = table[(effect, pos, "substrate")], table[(effect, pos, "test")]
            lines.append(f"{effect:<11} {pos:<18} {sub:>10} {test:>8}")
    lines += ["", "substrate writes outside the op stream, by writer site:"]
    for (effect, klass, writer, beneath), n in sorted(outside.items(), key=lambda kv: -kv[1]):
        under = f"  (beneath {beneath})" if beneath else ""
        lines.append(f"  {n:>6}  {effect:<10} {klass:<14} {writer}{under}")
    if not outside:
        lines.append("  (none)")
    injected = Counter((r["effect"], _render(r["site"])) for r in rows if r["injected"])
    lines += ["", "writes a layer injected into the stream, by the layer's line:"]
    lines += [f"  {n:>6}  {effect:<10} {site}" for (effect, site), n in injected.items()]
    if not injected:
        lines.append("  (none)")
    stacks = sorted(
        {(tuple(r["layers"]), r["position"]) for r in rows if r["position"] != "outside"}
    )
    lines += ["", "layer stacks on the op stream:"]
    lines += [f"  [{pos}] {', '.join(stack) or '(none)'}" for stack, pos in stacks]
    return "\n".join(lines)


def join_b(rows: list[dict]) -> str:
    """Join B: one durable identity in one test, reached from two or more places.

    A place is the authored site on the stream and the writer site off it, so a side door and
    its guarded twin pair. Groups whose members sit at different positions come first."""
    groups: dict[tuple, set[tuple[Frame, str]]] = defaultdict(set)
    for row in in_domain(rows):
        place = _frame(row["site"] or row["writer"])
        if row["effect"] == "artifact" or place is None:
            continue
        identity = row["identity"].split("#", 1)[0]
        groups[(row["test"], row["effect"], row["scope"], identity)].add((place, row["position"]))
    multi = {g: found for g, found in groups.items() if len({p for p, _ in found}) > 1}

    def mixed(found: set[tuple[Frame, str]]) -> bool:
        return len({pos for _, pos in found}) > 1

    kinds: Counter = Counter()
    body: list[str] = []
    for (test, effect, _scope, identity), found in sorted(
        multi.items(), key=lambda kv: (not mixed(kv[1]), kv[0])
    ):
        paths = {"src" if p.path.startswith("src/") else "test" for p, _ in found}
        label = "+".join(sorted(paths)) + (", mixed" if mixed(found) else "")
        kinds[label] += 1
        body.append(f"  [{label}] {effect} {identity}  in {test}")
        body.extend(
            f"      [{pos}] {place.render()}"
            for place, pos in sorted(found, key=lambda pair: (pair[0].render(), pair[1]))
        )
    summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "none"
    return "\n".join(
        ["JOIN B: one identity, many places", f"{len(multi)} groups: {summary}", "", *body]
    )


# -- Join C: the static half

EFFECT_METHODS = frozenset({"step", "append", "_record_ledger"})


def effect_definitions(files: list[Path]) -> set[tuple[str, int]]:
    """`(path, line)` of every definition a durable write can resolve to.

    A `step` on a class whose name ends in `Ctx` or `Context`, an `append` on one ending in
    `Ledger` or `LedgerWriter`, and `_record_ledger` anywhere."""
    found: set[tuple[str, int]] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if (
                    (item.name == "step" and node.name.endswith(("Ctx", "Context")))
                    or (item.name == "append" and node.name.endswith(("Ledger", "LedgerWriter")))
                    or item.name == "_record_ledger"
                ):
                    found.add((str(path), item.lineno))
    return found


@dataclass(frozen=True, slots=True)
class StaticSite:
    path: str
    line: int
    method: str


@dataclass(frozen=True, slots=True)
class StaticSites:
    candidates: int
    resolved: list[StaticSite]
    unresolved: list[StaticSite]
    """Candidates `ty` had no definition for, where the name is specific enough to report."""


def static_sites(roots: list[str]) -> StaticSites:
    """ast-grep's candidates, split by whether `ty` resolves them to an effect definition.

    An `append` that resolves nowhere is dropped, since the name is every list's. A `step` or
    `_record_ledger` that resolves nowhere is kept: an untyped receiver is how a durable write
    escapes the resolver, and the runtime can still dispose of it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import whouses

    rule = {
        "kind": "identifier",
        "regex": f"^({'|'.join(sorted(EFFECT_METHODS))})$",
        "inside": {
            "kind": "attribute",
            "field": "attribute",
            "inside": {"kind": "call", "field": "function"},
        },
    }
    candidates = whouses._scan(rule, roots)
    files = sorted({Path(ROOT / root) for root in roots})
    targets = effect_definitions(
        [p for root in files for p in (root.rglob("*.py") if root.is_dir() else [root])]
    )
    server = whouses.Server(ROOT)
    resolved: list[StaticSite] = []
    unresolved: list[StaticSite] = []
    try:
        for match in candidates:
            path = ROOT / match["file"]
            start = match["range"]["start"]
            site = StaticSite(str(match["file"]), start["line"] + 1, match["text"])
            match server.definition(path, start["line"], start["column"]):
                case None if match["text"] != "append":
                    unresolved.append(site)
                case found if found in targets:
                    resolved.append(site)
    finally:
        server.proc.terminate()
    return StaticSites(len(candidates), resolved, unresolved)


def disposition(reached: set[str]) -> str:
    """A static site's verdict from how the run reached it: `stream`, `outside`, `attempt`."""
    wrote = reached - {"attempt"}
    match sorted(wrote), "attempt" in reached:
        case [], False:
            return "unwitnessed"
        case [], True:
            return "reached, never wrote"
        case ["stream"], _:
            return "witnessed in stream"
        case ["outside"], _:
            return "witnessed outside"
        case _:
            return "witnessed both"


def join_c(rows: list[dict], sites: StaticSites) -> str:
    """Join C: each statically found write site, disposed by in-domain rows of its own kind.

    A site reached only by attempts (a checkpoint hit, a folded append) is reported as such: it
    ran, and it never wrote."""
    seen: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row["engine"] not in IN_DOMAIN:
            continue
        where = (
            "attempt"
            if not row["written"]
            else "outside"
            if row["position"] == "outside"
            else "stream"
        )
        for frame in row["frames"]:
            seen[(row["effect"], frame["path"], frame["line"])].add(where)

    def disposed(found: list[StaticSite]) -> tuple[Counter, list[str]]:
        verdicts: Counter = Counter()
        lines: list[str] = []
        for site in sorted(found, key=lambda s: (s.path, s.line, s.method)):
            effect = "checkpoint" if site.method == "step" else "ledger"
            said = disposition(seen.get((effect, site.path, site.line), set()))
            verdicts[said] += 1
            lines.append(f"  [{said}] {site.path}:{site.line} .{site.method}")
        return verdicts, lines

    def summary(verdicts: Counter) -> str:
        return ", ".join(f"{n} {v}" for v, n in sorted(verdicts.items())) or "none"

    resolved, resolved_lines = disposed(sites.resolved)
    unresolved, unresolved_lines = disposed(sites.unresolved)
    return "\n".join(
        [
            "JOIN C: static write sites against the witness",
            f"{sites.candidates} ast-grep candidates",
            "",
            f"resolved by ty to an effect definition: {summary(resolved)}",
            *resolved_lines,
            "",
            f"no definition from ty (step, _record_ledger): {summary(unresolved)}",
            *unresolved_lines,
        ]
    )


# ---------------------------------------------------------------- the command


def pytest_args(paths: list[str]) -> list[str]:
    """The arguments the plugin's run passes. `-n 0` comes after the caller's arguments, so the
    tests run in this process, where the sensors are."""
    return [
        *paths,
        "-q",
        "--no-header",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "-n",
        "0",
    ]


def _run(paths: list[str], out: Path) -> int:
    import pytest

    return int(pytest.main(pytest_args(paths), plugins=[Witness(out)]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pytest_args", nargs="*", help="what to run (default tests/)")
    parser.add_argument("--report", action="store_true", help="report from a previous run")
    parser.add_argument("--out", type=Path, default=OUT, help="the rows file")
    parser.add_argument(
        "--static",
        nargs="*",
        metavar="ROOT",
        help="add Join C over these roots (default src/effective src/agent)",
    )
    args = parser.parse_args(argv)
    if args.report and args.pytest_args:
        parser.error("--report reads a previous run; it takes no pytest arguments")
    if not args.report and (code := _run(args.pytest_args or ["tests/"], args.out)) not in (0, 1):
        return code
    rows = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
    if not rows:
        print(f"no writes recorded in {args.out}: the sensors saw nothing", file=sys.stderr)
        return 2
    tests = len({row["test"] for row in rows})
    print(f"{len(rows)} write attempts from {tests} tests  ({args.out})\n")
    print(join_a(rows), join_b(rows), sep="\n\n")
    if args.static is not None:
        print("\n" + join_c(rows, static_sites(args.static or ["src/effective", "src/agent"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
