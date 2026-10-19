"""`race` and `quorum` on the recording and replay core.

Each row here has one answer whatever the schedule, so the threads the recorder runs its branches
in cannot move it. A loser that steps `LONG` times is how a stop is observed: it ends `Stopped`
only if it stopped at an admission once the choice was saved.
"""

import threading
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any

import pytest
from _steps import called_step

from effective.api import await_event, call_tool, gather, quorum, race, scoped
from effective.choice import Chosen, Impossible, Raised, Refusal, Stopped, Won
from effective.domain import CallTool
from effective.govern import Refused
from effective.handlers import recording
from effective.handlers.base import BranchRaised, Racing, TraceEntry, op_key
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Key, compose_key
from effective.ops import CompositionRefused, RaceUndecided, Step, leaves


class _AnyTool(Mapping[str, object]):
    """Answers every tool call with its own name, so a branch can call as many as it likes."""

    def __getitem__(self, key: str) -> object:
        if key.startswith("tool:"):
            return key.removeprefix("tool:")
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _returns(name: str):
    def branch():
        return (yield from call_tool(name, {}, str))

    return branch


def _refuses(reason: str):
    def branch():
        yield from call_tool("before", {}, str)
        raise Refused(
            Step(name="gate", op=CallTool(name="gate", args={}, result_schema=str)), reason
        )

    return branch


LONG = 20_000
"""Steps a looping loser takes before it returns on its own. A loser stops long before this; a
handler that fails to stop it sees it return, and the race's endings then say so."""


def _forever(name: str):
    """A branch that ends on its own only after `LONG` steps; a loser stops at an admission."""

    def branch():
        for n in range(LONG):
            yield from scoped(compose_key(t"loop:{Index(n)}"), partial(call_tool, name, {}, str))
        return "ran out"

    return branch


class _Held(_AnyTool):
    """Answers `key` only once `until` is set, and sets each event in `marks` as its key is
    asked."""

    def __init__(
        self,
        key: str,
        until: threading.Event,
        marks: Mapping[str, threading.Event] | None = None,
    ) -> None:
        self._key, self._until, self._marks = key, until, marks or {}

    def __getitem__(self, key: str) -> object:
        if (asked := self._marks.get(key)) is not None:
            asked.set()
        if key == self._key:
            assert self._until.wait(10), f"{key} was held and never released"
        return super().__getitem__(key)


def _recorded(program, responses: Mapping[str, object] | None = None) -> tuple[Any, Any]:
    handler = RecordingHandler(responses=_AnyTool() if responses is None else responses)
    return handler.run(program), handler


def test_a_refusal_loses_and_a_success_wins():
    def program():
        return (yield from race([_refuses("no"), _returns("yes")]))

    answer, handler = _recorded(program)
    assert answer == Chosen((Won(1, "yes"),), (Refusal(0, "no"), Won(1, "yes")))
    assert ReplayHandler(handler.trace).run(program) == answer


def test_every_branch_refusing_makes_the_race_impossible():
    def program():
        return (yield from race([_refuses("a"), _refuses("b")]))

    answer, handler = _recorded(program)
    assert answer == Impossible((Refusal(0, "a"), Refusal(1, "b")))
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_full_quorum_returns_every_winner_in_branch_order():
    def program():
        return (yield from quorum(3, [_returns("a"), _returns("b"), _returns("c")]))

    answer, _ = _recorded(program)
    assert answer == Chosen(
        (Won(0, "a"), Won(1, "b"), Won(2, "c")), (Won(0, "a"), Won(1, "b"), Won(2, "c"))
    )


def test_a_quorum_of_none_starts_no_branch():
    started: list[int] = []

    def branch(i: int):
        def body():
            started.append(i)
            return (yield from call_tool("x", {}, str))

        return body

    def program():
        return (yield from quorum(0, [branch(0), branch(1)]))

    answer, handler = _recorded(program)
    assert answer == Chosen((), (Stopped(0), Stopped(1)))
    assert started == []
    assert handler.trace == []


@pytest.mark.parametrize("want", [-1, 3])
def test_a_quorum_out_of_range_is_refused_when_built(want):
    def program():
        return (yield from quorum(want, [_returns("a"), _returns("b")]))

    with pytest.raises(CompositionRefused, match="cannot want"):
        _recorded(program)


def test_a_loser_that_never_ends_stops_at_an_admission():
    def program():
        return (yield from race([_returns("won"), _forever("loop")]))

    answer, handler = _recorded(program)
    assert answer == Chosen((Won(0, "won"),), (Won(0, "won"), Stopped(1)))
    keys = [entry.key.stored() for entry in handler.trace]
    assert keys[0] == "race:0;choice"
    assert keys[-1] == "race:0;endings"
    assert all(key.startswith(("race:0,0;", "race:0,1;")) for key in keys[1:-1])
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_race_inside_a_loser_never_chooses():
    """The flag passes down, and an inner race with no saved choice saves none. The
    outer winner's answer waits until the inner race has started, so the flag finds it running."""

    def inner():
        return (yield from race([_forever("p"), _forever("q")]))

    def program():
        return (yield from race([_returns("won"), inner]))

    inner_started = threading.Event()
    held = _Held("tool:won", until=inner_started, marks={"tool:p": inner_started})
    answer, handler = _recorded(program, held)
    assert answer == Chosen((Won(0, "won"),), (Won(0, "won"), Stopped(1)))
    keys = [entry.key.stored() for entry in handler.trace]
    assert not [key for key in keys if key.startswith("race:0,1;race:0;")]
    for inner_branch in ("race:0,1;race:0,0;", "race:0,1;race:0,1;"):
        assert sum(key.startswith(inner_branch) for key in keys) < LONG, "never told to stop"
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_race_branch_that_awaits_is_refused():
    def waits():
        return (yield from await_event(Key.parse("ev:never"), str))

    def program():
        return (yield from race([waits]))

    with pytest.raises(ExceptionGroup) as raised:
        _recorded(program)
    assert [type(leaf) for leaf in leaves(raised.value)] == [CompositionRefused]


def _pure():
    """A body that returns without yielding an op, so it runs to its end wherever it is entered."""
    return 7
    yield  # a generator, which never reaches its first yield


def _after_a_read(
    monkeypatch: pytest.MonkeyPatch, fires: Callable[[Racing, list[int], list[Any]], bool]
) -> threading.Event:
    """An event set once a race's read of a batch leaves `fires` true."""
    read, fired = Racing.read, threading.Event()

    def reading(self: Racing, batch: list[int], slots: list[Any], *rest: Any) -> bool:
        failing = read(self, batch, slots, *rest)
        if fires(self, batch, slots):
            fired.set()
        return failing

    monkeypatch.setattr(Racing, "read", reading)
    return fired


@pytest.mark.parametrize(
    "structure",
    [
        "scoped",
        "gather",
        "race",
    ],
)
def test_a_flagged_loser_stops_at_a_structure(monkeypatch, structure):
    """A loser flagged before it reaches a `scoped`, `gather` or `race` stops there, on every
    interpreter: a structure is new work past its horizon, and a stopped loser starts none."""
    asked = threading.Event()
    decided = _after_a_read(monkeypatch, lambda race, batch, slots: race.decided())

    def holds(op: Any) -> Any:
        if isinstance(op, Step) and op_key(op).stored() == called_step("a"):
            assert asked.wait(10)
        elif isinstance(op, Step):
            asked.set()
            assert decided.wait(10)
        return (yield op)

    def loser():
        yield from call_tool("b", {}, str)
        match structure:
            case "scoped":
                return (yield from scoped(Key.parse("s"), _pure))
            case "gather":
                return (yield from gather([_pure]))[0]
            case _:
                return (yield from race([_pure]))  # an inner race in a loser saves no choice

    def program():
        return (yield from race([_returns("a"), loser]))

    handler = RecordingHandler(responses=_AnyTool(), op_layers=[holds])
    answer = handler.run(program)
    assert answer == Chosen((Won(0, "a"),), (Won(0, "a"), Stopped(1)))
    assert ReplayHandler(handler.trace).run(program) == answer


def _raises_after(asked: str):
    """A branch whose own code raises once its first call returns."""

    def branch():
        yield from call_tool(asked, {}, str)
        raise KeyError("the branch's own code")

    return branch


def _read_before_the_choice(
    monkeypatch: pytest.MonkeyPatch, raising: type[BaseException]
) -> threading.Event:
    """An event set once a race reads a branch's `raising` error before it chooses. The type
    names which race: an inner race reads its own branches' errors first."""

    def read_one(race: Racing, batch: list[int], slots: list[Any]) -> bool:
        return any(
            isinstance(slots[i], BranchRaised)
            and slots[i].before_choice
            and isinstance(slots[i].error, raising)
            for i in batch
        )

    return _after_a_read(monkeypatch, read_one)


def test_replay_answers_a_race_whose_branch_raised_before_the_choice_as_the_recorder_did(
    monkeypatch,
):
    """The sibling wins past the error, and replay serves the endings the recorder settled.

    branch 0               the race                   branch 1
    calls `first`                                     calls `ok`, held
    raises ────────────►  reads the raise
                          before any choice ───────►  `ok` answered
                          chooses branch 1  ◄───────  returns
    """

    def program():
        return (yield from race([_raises_after("first"), _returns("ok")]))

    held = _Held("tool:ok", until=_read_before_the_choice(monkeypatch, KeyError))
    answer, handler = _recorded(program, held)
    ending = Raised(0, repr(KeyError("the branch's own code")))
    assert answer == Chosen((Won(1, "ok"),), (ending, Won(1, "ok")))
    assert ReplayHandler(handler.trace).run(program) == answer


def _raises_at_once():
    def branch():
        raise ValueError("a programming error in the branch")
        yield  # a generator, which never reaches its first yield

    return branch


def test_replay_serves_a_raise_its_record_settled_under_a_choice_of_no_winners():
    """A record can hold an impossible answer with a `Raised` ending, as a recorder answering
    impossible before every branch ended wrote one. Replay is a reading of that record."""

    def recorded():
        return (yield from quorum(3, [_refuses("no"), _returns("b"), _returns("c")]))

    _, handler = _recorded(recorded)
    raised = {"ending": "raised", "error": "ValueError('a programming error in the branch')"}
    trace = [
        TraceEntry(e.key, e.op, [e.result[0], raised, *e.result[2:]], e.error)
        if "endings" in e.key.stored()
        else e
        for e in handler.trace
        if not e.key.stored().startswith("race:0,1;")
    ]

    def replayed():
        return (yield from quorum(3, [_refuses("no"), _raises_at_once(), _returns("c")]))

    answer = ReplayHandler(trace).run(replayed)
    assert isinstance(answer, Impossible)
    assert [type(ending) for ending in answer.endings][:2] == [Refusal, Raised]


class _HeldUntilChosen(_AnyTool):
    """Holds `tool:slow` until `chosen` is set, and `tool:w` until `tool:slow` has been asked."""

    def __init__(self, chosen: threading.Event) -> None:
        self._chosen, self._inside = chosen, threading.Event()

    def __getitem__(self, key: str) -> object:
        if key == "tool:slow":
            self._inside.set()
            assert self._chosen.wait(10), "the outer race never chose"
        if key == "tool:w":
            assert self._inside.wait(10), "the inner race never reached its first call"
        return super().__getitem__(key)


def test_an_inner_race_its_outer_choice_ends_stops_and_replays_stopped(monkeypatch):
    """The recorder's inner race is held open past the outer choice, which names it a loser, so it
    stops; replay of the trace stops it too."""
    chosen = threading.Event()
    put = recording.ChoiceBox.put

    def putting(self: Any, choice: Any) -> None:
        put(self, choice)
        chosen.set()

    monkeypatch.setattr(recording.ChoiceBox, "put", putting)

    def program():
        inner = lambda: quorum(3, [_refuses("no"), _raises_at_once(), _returns("slow")])  # noqa: E731
        return (yield from race([inner, _returns("w")]))

    answer, handler = _recorded(program, _HeldUntilChosen(chosen))
    assert answer == Chosen((Won(1, "w"),), (Stopped(0), Won(1, "w")))
    assert any("tool:slow" in e.key.stored() for e in handler.trace), "slow's call was not kept"
    assert ReplayHandler(handler.trace).run(program) == answer


def test_an_inner_race_that_failed_before_the_outer_choice_replays_raised(monkeypatch):
    """The inner race fails, the outer race reads that as a loss ending `Raised`, and the inner
    branches' calls stay on the trace for replay to walk.

    branch 0, an inner quorum      the outer race              branch 1
                                                              calls `w`, held
    fails: `RaceUndecided` ────►  reads the raise
                                  before any choice ───────►  `w` answered
                                  chooses branch 1  ◄───────  returns
    """

    def program():
        inner = lambda: quorum(3, [_refuses("no"), _raises_at_once(), _returns("a")])  # noqa: E731
        return (yield from race([inner, _returns("w")]))

    held = _Held("tool:w", until=_read_before_the_choice(monkeypatch, RaceUndecided))
    answer, handler = _recorded(program, held)
    failure = RaceUndecided(
        "race branches raised", [ValueError("a programming error in the branch")]
    )
    assert answer == Chosen((Won(1, "w"),), (Raised(0, repr(failure)), Won(1, "w")))
    assert ReplayHandler(handler.trace).run(program) == answer
