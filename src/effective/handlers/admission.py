"""What a race loser may still do, and how every thread learns it.

A loser checks twice. At each walk op, before the layers, its cursor asks whether an earlier
attempt admitted this op before its choice was saved, which bounds what a stopped loser replays.
Where an op reaches the engine, a guard asks whether the engine can serve it, which stops any new
effect, a layer's injected ops included.

| state                 | lock                        | read by                                   |
|-----------------------|-----------------------------|-------------------------------------------|
| a cursor's count, its | the cursor's own            | its branch, per op; the publisher, once   |
| bounds, `sealed`      |                             | per decision                              |
| a race's members, cut | the race tree's lock `T`    | registration, retirement, publication     |
| a race's choice       | its `ChoiceBox`'s own       | the barrier; the recorder's stop tests    |
| a handler's `_free`,  | none: confined to the       | that thread; its parent only after the    |
| `_halted`, `_gated`,  | handler's thread            | branch's future has joined                |
| `_inputs`             |                             |                                           |

Locks are taken in the order `T`, then a cursor's, then the engine's, and a cursor's lock is never
held while taking another. No flag two threads share is read without a lock, so nothing here
rests on the atomicity of a builtin.
"""

import hashlib
import json
import threading
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Any, assert_never

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python

from effective.choice import Choice

STEP_BUDGET = 0
"""Steps a stopped loser may take past its horizon: structures entered, or ops a layer yields
after a resume. Zero, so a stopped loser starts nothing new and finishes what it had admitted,
its layers' code after the engine answers included. `Cursor.tick`'s `leaf`, `Bound.exact` and
`Cursor.late` decide only above zero; they are kept so the budget can be raised by this
constant alone."""


class Halt(Exception):
    """A cursor's tick past its horizon, which its handler turns into a stop."""


@dataclass(frozen=True)
class Bound:
    """One enclosing race's limit on a cursor: `limit` walk ops, exact when the choice's horizon
    named this cursor and a cap folded from a finished parent otherwise."""

    race: RaceState
    index: int
    choice: Choice | None = None
    limit: int = 0
    exact: bool = True

    @property
    def loser(self) -> bool:
        return self.choice is not None and self.choice.loses(self.index)


class Cursor:
    """The walk position of one thread of control inside a race tree: a race or gather branch's
    handler. A scoped body continues its handler's cursor."""

    def __init__(self, frame: str) -> None:
        self.frame = frame
        self.lock = threading.Lock()
        self.ready = threading.Condition(self.lock)
        self.sealed = False
        self.count = 0
        self.bounds: tuple[Bound, ...] = ()
        self.late = 0

    def stopped(self) -> bool:
        """Whether a stored choice names this thread of control a loser."""
        with self.lock:
            return any(bound.loser for bound in self.bounds)

    def tick(self, *, leaf: bool) -> bool:
        """Count one walk op, or one op a layer yielded after a resume, and whether this thread is
        still running free. Waits while a publisher has it sealed; raises `Halt` past a horizon."""
        with self.lock:
            while self.sealed:
                self.ready.wait()
            n = self.count + 1
            late = [bound for bound in self.bounds if bound.loser and n > bound.limit]
            if late:
                if leaf or not all(bound.exact for bound in late) or self.late >= STEP_BUDGET:
                    raise Halt
                self.late += 1
            self.count = n
            return not any(bound.loser for bound in self.bounds)

    def inherit(self, frame: str) -> Cursor:
        """A child thread's cursor, bounded as this one is. The caller holds `T`.

        A child the saved cut names gets its exact limit. One it omits was finished before the cut
        or started after it, and gets what remains of this cursor's limit, which bounds a child
        whose replay loops even though it finished the first time."""
        child = Cursor(frame)
        with self.lock:
            child.bounds = tuple(
                Bound(bound.race, bound.index, bound.choice, saved, True)
                if (saved := bound.race.cut.get(frame)) is not None
                else Bound(
                    bound.race,
                    bound.index,
                    bound.choice,
                    max(0, bound.limit - self.count) if bound.loser else 0,
                    False,
                )
                for bound in self.bounds
            )
        return child

    def fold(self, finished: int) -> None:
        """Count a structure whose children have all ended, and their work, as this thread's own,
        so a replay's cap covers them."""
        with self.lock:
            self.count += 1 + finished


class ChoiceBox:
    """A race's choice once it is stored, behind its own lock."""

    def __init__(self, choice: Choice | None = None) -> None:
        self._lock = threading.Lock()
        self._choice = choice

    def current(self) -> Choice | None:
        with self._lock:
            return self._choice

    def put(self, choice: Choice) -> None:
        with self._lock:
            self._choice = choice

    def loses(self, index: int) -> bool:
        choice = self.current()
        return choice is not None and choice.loses(index)


class RaceState:
    """One race's live cursors and the cut its choice saved. `T` guards both."""

    def __init__(self, choice: Choice | None, cut: dict[str, int]) -> None:
        self.box = ChoiceBox(choice)
        self.cut = cut
        self.members: dict[str, Cursor] = {}

    def register(self, cursor: Cursor, index: int | None = None) -> None:
        """Enroll `cursor`: a branch, when `index` is given, and its descendants otherwise. The
        caller holds `T`."""
        if index is not None:
            choice = self.box.current()
            with cursor.lock:
                cursor.bounds += (Bound(self, index, choice, self.cut.get(cursor.frame, 0)),)
        self.members[cursor.frame] = cursor

    def retire(self, cursor: Cursor) -> None:
        """Remove a cursor whose thread has ended. The caller holds `T`."""
        self.members.pop(cursor.frame, None)

    def publish(self, proposal: Choice, settle: Callable[[dict[str, Any]], Any]) -> Choice:
        """Seal every live cursor, save the choice with the cut, install what the store returned,
        and unseal. The caller holds `T`.

        The cut holds a count for each live cursor the proposal names a loser, so a retry stops
        that thread where this attempt had got to. Sealing first means no admission slips between
        the count read and the save; the unseal runs even when the save fails, which reopens
        admission with no loser told to stop."""
        live = sorted(self.members.values(), key=lambda cursor: cursor.frame)
        try:
            cut = {}
            for cursor in live:
                with cursor.lock:
                    cursor.sealed = True
                    (entry,) = (bound for bound in cursor.bounds if bound.race is self)
                    if proposal.loses(entry.index):
                        cut[cursor.frame] = cursor.count
            stored = settle(proposal.stored() | {"horizon": cut})
            choice = Choice.from_stored(stored)
            self.cut = dict(stored.get("horizon", {}))
            for cursor in live:
                with cursor.lock:
                    cursor.bounds = tuple(
                        Bound(self, bound.index, choice, self.cut.get(cursor.frame, 0))
                        if bound.race is self
                        else bound
                        for bound in cursor.bounds
                    )
            self.box.put(choice)
            return choice
        finally:
            for cursor in live:
                with cursor.lock:
                    cursor.sealed = False
                    cursor.ready.notify_all()


def drive_racing(
    layers: Sequence[Callable[[Any], Generator[Any, Any, Any]]],
    op: Any,
    base: Callable[[Any], Any],
    *,
    new_work: Callable[[], None],
    stopped: tuple[type[BaseException], ...],
) -> Any:
    """`layers.drive_through` for a race branch.

    | differs in                              | so that                                     |
    |-----------------------------------------|---------------------------------------------|
    | `stopped` escapes every layer           | a layer that catches broadly cannot turn a  |
    |                                         | stop into a value                           |
    | `new_work()` runs when a resume yields  | a stopped loser starts no new op a layer    |
    | an op                                   | yields, while the op it admitted finishes   |
    | every layer generator is closed on exit | a stopped layer's `finally` runs            |
    """
    if not layers:
        return base(op)
    head, *rest = layers
    gen = head(op)
    try:
        try:
            inner = gen.send(None)
        except StopIteration as done:
            return done.value
        while True:
            try:
                value = drive_racing(rest, inner, base, new_work=new_work, stopped=stopped)
            except stopped:
                raise
            except Exception as raised:
                try:
                    inner = gen.throw(raised)
                except StopIteration as done:
                    return done.value
            else:
                try:
                    inner = gen.send(value)
                except StopIteration as done:
                    return done.value
            new_work()
    finally:
        gen.close()


def observed(value: Any) -> Any | None:
    """`value`'s witness, taken when the branch is handed it so a later mutation cannot change
    it, or `None` when it has none: bytes no text covers, and a value holding itself, raise where
    they are walked or encoded rather than answering."""
    try:
        return witness(value)
    except TypeError, ValueError, RecursionError:
        return None


def digest(witnesses: Sequence[Any | None]) -> str | None:
    """A digest of what a branch was handed, in order, from the witnesses `observed` took; `None`
    when any of them is `None`, so a retry fails rather than trusting a partial digest."""
    if any(taken is None for taken in witnesses):
        return None
    encoded = json.dumps(list(witnesses), separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def witness(value: Any) -> Any:
    """`value` as the store keeps it: its checkpoint encoding, canonical for the quotient both
    engines share, so two values durability cannot tell apart witness alike.

    Absurd holds a checkpoint as `jsonb`, which sorts an object's keys, drops the sign of a zero
    and reads `1e+16` back as an integer. A branch is handed the live value on the attempt that
    ran the op and the store's value on the next, so a witness finer than the store fails a retry
    that was handed the same input.

    | in the encoding | witnessed as |
    |---|---|
    | an object | its members, sorted by key |
    | an array | its items in order, a tuple and a list alike |
    | a number | its own digits, so `-0.0` is `0.0` and `1e+16` is the integer |
    | a string, a boolean, null | itself, under its tag |

    Every set is ordered first, wherever it sits, since the encoding lists a set's items in the
    order the process's hash seed gives them and the attempt that retries is another process. A
    value no encoding takes has never been stored, so it is witnessed by its structure, where a
    finer witness is safe; one with neither raises `TypeError`, which `observed` turns into no
    witness.
    """
    try:
        return _canonical(_encoded(_ordered_sets(value)))
    except TypeError:
        return _structural(value)


def _ordered_sets(value: Any) -> Any:
    """`value` with every set inside it ordered by its items' witnesses. A model and a dataclass
    are opened up first, since a set in one of their fields encodes in iteration order too."""
    match value:
        case set() | frozenset():
            return sorted((_ordered_sets(item) for item in value), key=_by_witness)
        case list() | tuple():
            return [_ordered_sets(item) for item in value]
        case dict():
            return {key: _ordered_sets(item) for key, item in value.items()}
        case BaseModel():
            return _ordered_sets(value.model_dump(mode="python"))
        case _ if is_dataclass(value) and not isinstance(value, type):
            return _ordered_sets(asdict(value))
        case _:
            return value


def _by_witness(item: Any) -> str:
    return json.dumps(witness(item), separators=(",", ":"))


def _structural(value: Any) -> Any:
    """A value no checkpoint encoding takes, by the parts a workflow reads: a type, a model's
    fields, a dataclass's fields, a mapping's members. Nothing here is ever stored, so this is
    finer than the store and cannot fail a retry that was handed the same value."""
    match value:
        case type():
            return ["type", _named(value)]
        case BaseModel():
            return [_named(type(value)), witness(dict(value))]
        case list() | tuple():
            return ["array", [witness(item) for item in value]]
        case dict():
            members = sorted((str(key), witness(item)) for key, item in value.items())
            return [_named(type(value)), [[key, item] for key, item in members]]
        case _ if is_dataclass(value) and not isinstance(value, type):
            named = [[held.name, witness(getattr(value, held.name))] for held in fields(value)]
            return [_named(type(value)), named]
        case _:
            raise TypeError(f"no stable witness for a {type(value).__qualname__}")


def observed_error(raised: BaseException) -> Any | None:
    """A witness of an error a branch was handed, or `None` when a part of it has none. An error
    is no stored value, so it is its class, its arguments and its attributes, each witnessed as
    one."""
    try:
        attributes = sorted((name, witness(value)) for name, value in vars(raised).items())
        return [
            "error",
            _named(type(raised)),
            [witness(argument) for argument in raised.args],
            [[name, value] for name, value in attributes],
        ]
    except TypeError, ValueError, RecursionError:
        return None


def _canonical(encoded: Any) -> Any:
    """One node of a checkpoint encoding, tagged so no two kinds collide."""
    match encoded:
        case None:
            return ["null"]
        case bool():
            return ["bool", encoded]
        case str():
            return ["str", encoded]
        case int() | float() | Decimal():
            return ["number", _number(encoded)]
        case list() | tuple():
            return ["array", [_canonical(item) for item in encoded]]
        case dict():
            members = sorted((str(key), _canonical(item)) for key, item in encoded.items())
            return ["object", [[key, item] for key, item in members]]
        case _:  # pragma: no cover - `to_jsonable_python` returns no other kind
            raise TypeError(f"no stable witness for a {type(encoded).__qualname__}")


def _number(value: int | float | Decimal) -> str:
    """A number by its value: what the store reads back, whatever form it was written in."""
    match value:
        case int():  # exact at any width, where a decimal's context would round it
            return str(value)
        case float():
            quantity = Decimal(repr(value))
        case Decimal():
            quantity = value
        case unreachable:
            assert_never(unreachable)
    if not quantity.is_finite():
        return "nan" if quantity.is_nan() else format(quantity)
    if quantity == 0:
        return "0"
    # `format` and the arithmetic around it, never `normalize`, whose rounding a workflow's own
    # decimal context decides: two numbers the store keeps apart would witness alike under it.
    written = format(quantity, "f")
    return written.rstrip("0").rstrip(".") if "." in written else written


def _encoded(value: Any) -> Any:
    """`value` as a checkpoint encodes it."""
    try:
        return to_jsonable_python(value)
    except PydanticSerializationError as unencodable:
        raise TypeError(f"no stable witness for a {type(value).__qualname__}") from unencodable


def _named(kind: type) -> list[str]:
    """A class, by where it is defined."""
    return [kind.__module__, kind.__qualname__]
