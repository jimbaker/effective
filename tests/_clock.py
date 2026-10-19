"""One clock a test holds and moves forward: a race's deadline reads it, and so does the engine a
run is on, so a row places each read by moving the clock rather than by waiting for a real one.

| reads it                | held by                                                       |
|-------------------------|---------------------------------------------------------------|
| a race's deadline       | `on_races`, installing the clock as `base.race_time`          |
| the Absurd SDK          | `on_absurd`, installing it as `_get_current_time`             |
| `absurd.current_time()` | `absurd.fake_now`, set by `on_absurd` and `hold` on a session |

A held clock only moves forward. A row that needs an earlier instant starts there, and a row whose
clocks disagree, as two workers' do, holds one clock each.
"""

import contextlib
import datetime as dt
import threading
from collections.abc import Generator
from typing import Any

import pytest

from effective.engines.absurd import AbsurdEngine
from effective.handlers import base


class HeldClock:
    def __init__(self, at: dt.datetime) -> None:
        self._at = at
        self._sessions: list[Any] = []
        self._lock = threading.Lock()

    @property
    def at(self) -> dt.datetime:
        with self._lock:
            return self._at

    def __call__(self) -> float:
        return self.at.timestamp()

    def advance(self, by: dt.timedelta) -> None:
        with self._lock:
            self._move(self._at + by)

    def advance_to(self, at: dt.datetime) -> None:
        with self._lock:
            self._move(at)

    def on_races(self, monkeypatch: pytest.MonkeyPatch) -> HeldClock:
        monkeypatch.setattr(base, "race_time", self)
        return self

    def hold(self, session: Any) -> None:
        """Hold `absurd.fake_now` on a session the test opened, so what it writes is stamped
        by this clock. The session autocommits: a rollback would undo the setting."""
        assert session.autocommit, "a held session autocommits, or a rollback drops its clock"
        with self._lock:
            self._sessions.append(session)
            self._set(session)

    @contextlib.contextmanager
    def on_absurd(self, engine: AbsurdEngine) -> Generator[HeldClock]:
        """Hold both clocks a durable wait consults: the server's, on the engine's session, and
        the SDK's, which re-derives a sleep's remainder on resume.

        The two move together. A server clock moved alone lets a run be claimed, re-park, and
        reschedule to the frozen clock plus its remainder, past any later advance to the same
        instant. An instant before the real clock is refused for the same reason: a session
        this clock does not hold stamps real time, and the held clock would have to pass it to
        reach what that session wrote.
        """
        import absurd_sdk

        now = dt.datetime.now(dt.UTC)
        assert self._at >= now, f"a held Absurd clock starts at or after real time: {self._at}"
        real = absurd_sdk._get_current_time
        # The SDK names this function as its test seam. `setattr` because the checker models a
        # plain assignment as rebinding that exact function object.
        setattr(absurd_sdk, "_get_current_time", lambda: self.at)  # noqa: B010
        with self._lock:
            outer = len(self._sessions)
        try:
            self.hold(engine.app._conn)
            yield self
        finally:
            setattr(absurd_sdk, "_get_current_time", real)  # noqa: B010
            with self._lock:
                released, self._sessions = self._sessions[outer:], self._sessions[:outer]
            for session in released:
                if not session.closed:
                    session.execute("SELECT set_config('absurd.fake_now', '', false)")

    def _move(self, at: dt.datetime) -> None:
        assert at >= self._at, f"a held clock moves forward: {at} is before {self._at}"
        self._at = at
        for session in self._sessions:
            self._set(session)

    def _set(self, session: Any) -> None:
        session.execute("SELECT set_config('absurd.fake_now', %s, false)", (self._at.isoformat(),))
