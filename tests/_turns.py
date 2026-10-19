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

| rule                                           | what it prevents                       |
|------------------------------------------------|----------------------------------------|
| a turn passes once every attempt waits or ends | two calls racing inside one schedule   |
| a prefix chooses an attempt that waits         | a schedule nobody can supply hanging   |
| a wait is bounded                              | an attempt that never calls hanging    |
"""

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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

    def wait(self, name: str) -> None:
        """Give up the turn, if `name` holds it, and wait for `name`'s next one."""
        with self._turn:
            if self._running == name:
                self._running = None
            self._waiting.add(name)
            self._turn.notify_all()
            assert self._turn.wait_for(lambda: self._running == name or self._abandoned, TIMEOUT)
            if self._abandoned:
                raise Abandoned(name)

    def end(self, name: str) -> None:
        with self._turn:
            self._running = None if self._running == name else self._running
            self._waiting.discard(name)
            self._ended.add(name)
            self._turn.notify_all()

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
        with self._turn:
            while len(self._ended) < len(self.names):
                settled = self._turn.wait_for(
                    lambda: (
                        self._running is None and self._waiting | self._ended == set(self.names)
                    ),
                    TIMEOUT,
                )
                assert settled, (self._waiting, self._ended, self._running)
                if not self._waiting:
                    break
                if starting := sorted(self._waiting - self._started):
                    chosen = starting[0]
                    self._started.add(chosen)
                else:
                    chosen = self._chosen(tuple(sorted(self._waiting)))
                self._waiting.discard(chosen)
                self._running = chosen
                self._turn.notify_all()
        for thread in threads:
            thread.join(TIMEOUT)
        return answers

    def _chosen(self, ready: tuple[str, ...]) -> str:
        """The attempt the prefix, or else the attempt that last ran, takes the next call for."""
        taken, chosen = len(self.choices), self._last if self._last in ready else ready[0]
        if len(ready) > 1:
            chosen = self.prefix[taken] if taken < len(self.prefix) else chosen
            if chosen not in ready:
                self._abandoned = True
                self._turn.notify_all()
                raise ValueError(f"the prefix chooses {chosen!r} where {ready} wait")
            self.choices.append((ready, chosen, self._last))
        self._last = chosen
        return chosen

    def transport(self, name: str, inner: Any) -> Any:
        """`inner`, with `name` taking a turn before each of its calls."""
        turns = self

        class Turned:
            def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
                turns.wait(name)
                return inner.call(model, method, *args, **kwargs)

        return Turned()


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
