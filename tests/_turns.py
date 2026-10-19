"""Every schedule of a few attempts that share a transport, one call at a time.

Each attempt runs in its own thread and holds the turn from one transport call to the next, so
the code between two calls runs inside the turn and a schedule is the order of the calls. The
attempts start one at a time in name order, each running to its first call, and a start is no
choice: it decides nothing the order of the calls can see. `schedules` explores them depth first:
a run follows a prefix of choices and then lets the attempt that last ran go on, and records each
point where more than one was waiting, which names the runs still to make. A switch is a turn
taken by another attempt than the one that ran last while it waited; bounding the switches, as
CHESS bounds preemptions, keeps the count to hundreds, and two switches already place one whole
attempt inside any window of the other.

The attempts can be a gather's branches, their calls its workflow ops: `gathered` names each branch
by its frame and makes its entry its start, and `layer` takes a turn before each op a named branch
runs, so the commit falls inside the turn.

| rule                                           | what it prevents                       |
|------------------------------------------------|----------------------------------------|
| a turn passes once every attempt waits or ends | two calls racing inside one schedule   |
| a prefix chooses an attempt that waits         | a schedule nobody can supply hanging   |
| a wait is bounded                              | an attempt that never calls hanging    |
| a claim restarts every branch, choices go on   | a park or a crash hiding later choices |
| a crash ends its branch's turn at once         | a sibling waiting on a dead branch     |
"""

import threading
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from _conformance import FaultInjected

from effective.keys import Index, compose_key
from effective.keys.frame import branch_frames
from effective.layers import current_placement, op_layer
from effective.ops import WorkflowOp

TIMEOUT = 10.0


class Abandoned(Exception):
    """The run stopped before this attempt's turn came."""


@dataclass
class Turns:
    """One run: the attempts take turns in the order `prefix` chooses, then the first waiting."""

    names: Sequence[str]
    prefix: Sequence[str] = ()
    choices: list[tuple[tuple[str, ...], str, str | None]] = field(default_factory=list)
    """Each point where more than one attempt waited, which one went, and which ran last."""
    _turn: threading.Condition = field(default_factory=threading.Condition)
    _waiting: set[str] = field(default_factory=set)
    _ended: set[str] = field(default_factory=set)
    _started: set[str] = field(default_factory=set)
    _abandoned: bool = False
    _running: str | None = None
    _last: str | None = None
    _refused: ValueError | None = None
    _committed: set[str] = field(default_factory=set)
    _claims: int = 0
    _stalled: str | None = None

    def wait(self, name: str) -> None:
        """Give up the turn, if `name` holds it, and wait for `name`'s next one."""
        with self._turn:
            assert name in self.names, f"{name!r} is no attempt of {self.names}"
            assert name not in self._waiting, f"{name!r} waits twice: one thread per name"
            if self._running == name:
                self._running = None
            self._waiting.add(name)
            self._advance()
            if not self._turn.wait_for(lambda: self._running == name or self._abandoned, TIMEOUT):
                self._stalled = f"{name!r} waited out its turn: {self._waiting=}, {self._ended=}"
                self._abandoned = True
                self._turn.notify_all()
            if self._abandoned:
                raise Abandoned(name)

    def end(self, name: str, claim: int | None = None) -> None:
        """`name` ends, in `claim` if given: an end a finished claim's branch reaches late, as a
        generator finalized after the next claim began, changes nothing."""
        with self._turn:
            if claim is not None and claim != self._claims:
                return
            self._running = None if self._running == name else self._running
            self._waiting.discard(name)
            self._ended.add(name)
            self._advance()

    def _advance(self) -> None:
        """Pass the turn, once every attempt waits or has ended; the caller holds `_turn`."""
        settled = self._waiting | self._ended == set(self.names)
        if self._running is not None or self._abandoned or not self._waiting or not settled:
            return
        if starting := sorted(self._waiting - self._started):
            chosen = starting[0]
            self._started.add(chosen)
        else:
            try:
                chosen = self._chosen(tuple(sorted(self._waiting)))
            except ValueError as refused:
                self._refused, self._abandoned = refused, True
                self._turn.notify_all()
                return
        self._waiting.discard(chosen)
        self._running = chosen
        self._turn.notify_all()

    def claim(self) -> None:
        """A claim of the run begins: every branch starts again, and the choices go on, so a
        schedule spans a park or a crash. Call it where the attempt builds its domain."""
        with self._turn:
            assert self._running is None, f"{self._running!r} holds the turn of the last claim"
            assert not self._waiting, f"{self._waiting} wait in the last claim"
            self._ended, self._started, self._last = set(), set(), None
            self._claims += 1

    def claiming(self, domain: Any) -> Callable[[], Any]:
        """A backend's `fresh`: begin a claim, and answer with `domain`."""

        def fresh() -> Any:
            self.claim()
            return domain

        return fresh

    def settle(self) -> None:
        """Raise what stopped the run early: a prefix nobody could take, or a turn never passed."""
        if self._refused is not None:
            raise self._refused
        assert self._stalled is None, self._stalled

    def run(self, attempts: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
        """Each attempt's answer, or what it raised."""
        answers: dict[str, Any] = {}

        def attempt(name: str) -> None:
            try:
                self.wait(name)
                answers[name] = attempts[name]()
            except Exception as raised:
                answers[name] = raised
            finally:
                self.end(name)

        threads = [threading.Thread(target=attempt, args=(name,)) for name in self.names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(TIMEOUT)
        self.settle()
        return answers

    def _chosen(self, ready: tuple[str, ...]) -> str:
        """The attempt the prefix, or else the attempt that last ran, takes the next call for."""
        taken, chosen = len(self.choices), self._last if self._last in ready else ready[0]
        if len(ready) > 1:
            chosen = self.prefix[taken] if taken < len(self.prefix) else chosen
            if chosen not in ready:
                raise ValueError(f"the prefix chooses {chosen!r} where {ready} wait")
            self.choices.append((ready, chosen, self._last))
        self._last = chosen
        return chosen

    def gathered(
        self, bodies: Sequence[Callable[[], Generator[WorkflowOp, Any, Any]]]
    ) -> list[Callable[[], Generator[WorkflowOp, Any, Any]]]:
        """Each body as a gather branch under its name: its entry waits for the branch's start, so
        every branch waits before the first choice, and its exit ends the branch."""

        def turned(name: str, body: Callable[[], Generator[WorkflowOp, Any, Any]]) -> Any:
            def branch() -> Generator[WorkflowOp, Any, Any]:
                self.wait(name)
                with self._turn:
                    claim = self._claims
                try:
                    return (yield from body())
                finally:
                    self.end(name, claim)

            return branch

        return [turned(name, body) for name, body in zip(self.names, bodies, strict=True)]

    def layer(self) -> Any:
        """An op layer under which each op of a named branch takes a turn before it runs, so the
        turn spans the op and the branch's code up to its next op. Every other op passes, and so
        does an op a resumed claim replays: its placement committed in an earlier claim. A crash in
        the op ends the branch, so its sibling goes on in the claim the crash ends. A park never
        reaches the layer: the branch ends when its abandoned generator is finalized, as a
        turnstile's turn does."""

        @op_layer
        def take_turn(op: WorkflowOp) -> Generator[WorkflowOp, Any, Any]:
            placement = current_placement()
            frames = () if placement is None else branch_frames(placement.stored())
            name = next((f for f in frames if f in self.names), None)
            with self._turn:
                replayed = placement is not None and placement.stored() in self._committed
            if name is None or placement is None or replayed:
                return (yield op)
            self.wait(name)
            try:
                result = yield op
            except FaultInjected:
                self.end(name)
                raise
            with self._turn:
                self._committed.add(placement.stored())
            return result

        return take_turn

    def transport(self, name: str, inner: Any) -> Any:
        """`inner`, with `name` taking a turn before each of its calls."""
        turns = self

        class Turned:
            def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
                turns.wait(name)
                return inner.call(model, method, *args, **kwargs)

        return Turned()


def gather_branches(n: int) -> list[str]:
    """The frames of the first gather's `n` branches, which name them as attempts."""
    return [compose_key(t"gather:{Index(0)},{Index(i)}").stored() for i in range(n)]


def _switch(ready: tuple[str, ...], chosen: str, last: str | None) -> bool:
    return last in ready and chosen != last


def schedules(names: Sequence[str], switches: int) -> Iterator[Turns]:
    """A fresh `Turns` for every schedule with at most `switches` switches, each to be `run`
    before the next is drawn."""
    pending: list[tuple[str, ...]] = [()]
    while pending:
        prefix = pending.pop()
        turns = Turns(names, prefix)
        yield turns
        for at, (ready, chosen, last) in enumerate(turns.choices):
            if at < len(prefix):
                continue
            taken = turns.choices[:at]
            made = sum(_switch(*choice) for choice in taken)
            for other in ready:
                if other != chosen and made + _switch(ready, other, last) <= switches:
                    pending.append((*(choice for _, choice, _ in taken), other))
