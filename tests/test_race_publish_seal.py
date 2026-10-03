"""`RaceState.publish`'s seal, with a loser's admission released by the store write itself.

The choice is stored before the cursors' bounds carry it. A loser that ticks in that window, with
no seal, sees no choice, runs free and counts an op past the cut the store just saved. Sealed, its
tick waits until the bounds carry the choice and then stops.

| the order `publish` keeps              | pinned by                               |
|----------------------------------------|-----------------------------------------|
| seal, then read the count into the cut | the count is read under the seal        |
| write the bounds, then unseal          | the unseal follows the bounds           |
| a tick in the store write waits it out | a loser admitted during the store write |

The first two watch the cursor's own state, so they hold on any scheduler; the third drives a
real tick through the window.
"""

import threading
from typing import Any

from test_race_shapes import HOLD, until

from effective.choice import Choice
from effective.handlers.admission import Cursor, Halt, RaceState


class _Watched(threading.Condition):
    """A cursor's condition that says when a tick starts waiting on the seal."""

    def __init__(self, lock: threading.Lock, waiting: threading.Event) -> None:
        super().__init__(lock)
        self._waiting = waiting

    def wait(self, timeout: float | None = None) -> bool:
        self._waiting.set()
        return super().wait(timeout)


def test_a_loser_admitted_during_the_store_write_stops_at_the_cut():
    race = RaceState(None, {})
    loser = Cursor("branch:1")
    race.register(loser, index=1)
    loser.count = 2
    parked_or_done, go = threading.Event(), threading.Event()
    loser.ready = _Watched(loser.lock, parked_or_done)
    ticked: list[Any] = []

    def admit() -> None:
        until(go)
        try:
            ticked.append(loser.tick(leaf=True))
        except Halt as halted:
            ticked.append(halted)
        parked_or_done.set()

    def settle(value: dict[str, Any]) -> dict[str, Any]:
        go.set()
        until(parked_or_done)
        return value

    thread = threading.Thread(target=admit)
    thread.start()
    race.publish(Choice("winners", (0,), (0,)), settle)
    thread.join(HOLD)
    assert not thread.is_alive(), "the loser never woke: an unseal that did not notify"

    assert [type(outcome) for outcome in ticked] == [Halt]
    assert (loser.count, race.cut) == (2, {"branch:1": 2})


class _Observed(Cursor):
    """A cursor that records, at each read of its count, whether it was sealed, and at each
    unseal, whether its bounds already named it a loser."""

    def __init__(self, frame: str) -> None:
        self.reads_sealed: list[bool] = []
        self.unsealed_as_loser: list[bool] = []
        self._sealed, self._count = False, 0
        super().__init__(frame)

    @property
    def sealed(self) -> bool:
        return self._sealed

    @sealed.setter
    def sealed(self, value: bool) -> None:
        if self._sealed and not value:
            self.unsealed_as_loser.append(any(bound.loser for bound in self.bounds))
        self._sealed = value

    @property
    def count(self) -> int:
        self.reads_sealed.append(self._sealed)
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        self._count = value


def _published() -> _Observed:
    race = RaceState(None, {})
    loser = _Observed("branch:1")
    race.register(loser, index=1)
    loser.count = 2
    loser.reads_sealed.clear()
    race.publish(Choice("winners", (0,), (0,)), lambda value: value)
    return loser


def test_the_count_is_read_under_the_seal():
    assert _published().reads_sealed == [True]


def test_the_unseal_follows_the_bounds():
    assert _published().unsealed_as_loser == [True]
