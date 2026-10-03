"""The scripted path through `_coding` — a state machine with real backedges, end to end.

Role: **journey**. Its subject is the one the other fixtures do not have: control flow
that goes ROUND. The cart's fan-out and the search's loop both go forward; a failing test here
sends the machine back to `code` and a refactor sends it back to `test`, so the tape is a cycle
and a projection of it is a state diagram.

**Nothing here spells a key.** Every expectation is computed from `_coding.TASK` and the
transition table, so a longer task changes no assertion.
"""

import _coding as coding
import pytest

from effective.api import Effect, ask_llm
from effective.combinators import route
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler

pytestmark = pytest.mark.journey


def expected_path(task: coding.Task) -> tuple[str, ...]:
    """The states the machine must visit — walked from `advance` alone, with the world's verdict
    supplied by the case rather than by a run.

    Derived rather than recorded, so this is a claim ABOUT the machine. Calling `work` to find out
    what `work` does would be a mirror."""
    state, left, turn, path = "plan", task.refactors, 0, []
    while state != coding.DONE:
        path.append(state)
        outcome = "pass" if turn >= task.passes_at else "fail"
        state, left = coding.advance(state, outcome, left)
        turn += 1
    return tuple(path)


def test_the_machine_walks_the_path_the_transition_table_implies():
    """The whole run, against the table — plan through finalize, with the red-to-green cycle and
    the refactor round where the case puts them."""
    session, _, _ = coding.scripted_run()
    assert session.path == expected_path(coding.TASK)
    assert session.path[0] == "plan", "nothing may run before the intent is stated"
    assert session.path[-1] == "finalize"


def test_both_backedges_are_taken():
    """Anti-vacuity, and this fixture's reason to exist. A machine that never went round would be
    a straight-line workflow with extra vocabulary, and every projection question it is meant to
    answer would be the cart's question again.

    Named rather than counted, so a change says which edge lapsed."""
    session, _, _ = coding.scripted_run()
    pairs = list(zip(session.path, session.path[1:], strict=False))
    assert ("test", "code") in pairs, "a red suite must send the machine back to code"
    assert ("refactor", "test") in pairs, "a tidy must be re-tested"


def test_only_the_commitment_points_reach_the_canonical_record():
    """The two-bookkeepers rule, as this workload states it: `explore`, `repl` and `test` are
    sandbox work and leave checkpoints only; `code` and `finalize` leave rows.

    Reddens when: a phase starts appending — the drift that turns a ledger into a log."""
    session, ledger, _ = coding.scripted_run()
    assert [row.kind for row in ledger] == ["committed"] * len(session.commits) + ["finalized"]
    assert len(session.commits) == session.path.count("code")


def test_the_plan_is_approved_before_anything_touches_the_world():
    """The gate's whole point: the park sits in `plan`, so a run that has not been approved has
    appended nothing. Asserted at the PARK rather than after it, because "approved first" is a
    claim about the prefix."""
    handler = RecordingHandler(responses=coding.Answers(coding.TASK))
    parked = handler.run(lambda: coding.work(coding.RUN_ID, coding.TASK))
    assert isinstance(parked, Suspended)
    assert handler.ledger == []
    assert not any("ledger" in entry.key.stored() for entry in handler.trace)


def test_every_op_names_exactly_one_phase():
    """Structural isolation, the machine's version: an op carries the scope of the state that ran
    it, so a projection can tell a `run_tests` in `test` from one anywhere else.

    The count says how much the loop looked at — every op in the run sits under a phase."""
    _, _, tape = coding.scripted_run()
    examined = 0
    for key in tape:
        under = coding.phases_in(key)
        assert len(under) == 1, f"{key} sits under {len(under)} phases"
        assert under[0] in coding.PHASES, key
        examined += 1
    assert examined == len(tape)


def test_replay_re_derives_every_transition():
    """The determinism claim: `advance` is pure over recorded outcomes, so a replay that re-binds
    those outcomes must walk the identical path — and any drift shows up as a key that cannot
    bind rather than as a different answer."""
    handler = RecordingHandler(responses=coding.Answers(coding.TASK))
    outcome = handler.run(lambda: coding.work(coding.RUN_ID, coding.TASK))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(coding.APPROVED)
    assert ReplayHandler(handler.trace).run(lambda: coding.work(coding.RUN_ID, coding.TASK)) == (
        outcome
    )


@pytest.mark.parametrize(
    ("state", "outcome", "left", "expected"),
    [
        ("plan", "ok", 1, ("explore", 1)),
        ("explore", "read", 1, ("repl", 1)),
        ("repl", "tried", 1, ("code", 1)),
        ("code", "sha", 1, ("test", 1)),
        ("test", "fail", 1, ("code", 1)),  # the backedge
        ("test", "pass", 1, ("refactor", 1)),
        ("test", "pass", 0, ("finalize", 0)),  # no rounds left, so straight out
        ("refactor", "tidied", 1, ("test", 0)),  # the other backedge, and it spends a round
        ("finalize", coding.DONE, 0, (coding.DONE, 0)),
    ],
)
def test_the_transition_table_is_total_over_its_states(state, outcome, left, expected):
    """`advance` as a decision table, off a run: each row pins a successor and the rounds left,
    both backedges included. Which states exist is
    `test_advance_covers_every_phase_the_machine_can_be_in`'s question."""
    assert coding.advance(state, outcome, left) == expected


@pytest.mark.parametrize("state", list(coding.PHASES))
def test_advance_covers_every_phase_the_machine_can_be_in(state):
    """Each phase in `PHASES` has a transition out on a passing outcome, into a phase or `DONE`."""
    successor, _ = coding.advance(state, "pass", 1)
    assert successor in coding.PHASES.keys() | {coding.DONE}


def test_route_would_cost_one_recorded_op_per_turn():
    """The design question this fixture was built to settle, measured rather than argued.

    `combinators.route` dispatches on a recorded CLASSIFICATION — its `classifier` is a sealed op
    that must be yielded, because in route's world the choice comes from OUTSIDE the recorded
    prefix (a model picks the arm) and recording it is what makes replay re-dispatch identically.
    A state machine's next state is a PURE function of the outcome already recorded, so yielding
    a classifier to re-derive it adds an op and no information.

    **The general rule underneath: record a choice exactly when it is not derivable from what is
    already recorded.** Route's choice is not; a transition's is.

    Measured on the same machine, driven both ways."""
    _, _, tape = coding.scripted_run()
    routed_tape = _drive_with_route()
    turns = len(expected_path(coding.TASK))
    assert len(routed_tape) == len(tape) + turns
    assert sum("classify" in key for key in routed_tape) == turns


def _drive_with_route() -> list[str]:
    """The same machine expressed with `route`, purely to price it. Not the fixture's shape."""

    def machine() -> Effect[coding.Session]:
        state, left, turns = "plan", coding.TASK.refactors, []
        commits: list[str] = []
        while state != coding.DONE:
            turn = len(turns)

            def classifier(_chunk: str, s=state) -> Effect[str]:
                # the redundant op: it re-records a state the loop already holds
                yield from ask_llm("classify", f"which phase? ({s})", str)
                return s

            arms = {
                name: (lambda _chunk, f=fn: f(coding.TASK, coding.RUN_ID))
                for name, fn in coding.PHASES.items()
            }
            outcome = yield from _in_turn(
                turn, state, lambda a=arms, c=classifier: route("", c, a)
            )
            if state == "code":
                commits.append(outcome)
            turns.append(coding.Turn(state, outcome))
            state, left = coding.advance(state, outcome, left)
        return coding.Session(coding.RUN_ID, tuple(turns), tuple(commits))

    handler = RecordingHandler(responses=_RoutedAnswers(coding.TASK))
    outcome = handler.run(machine)
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(coding.APPROVED)
    return [entry.key.stored() for entry in handler.trace]


def _in_turn(turn: int, state: str, body):
    from effective.api import scoped
    from effective.keys import Segment, compose_key

    return scoped(
        # lint: terminal-hole — `turn` is the loop counter, an `int`, already an atom
        compose_key(t"turn:{turn}"),
        lambda: scoped(compose_key(t"phase:{Segment(state)}"), body),
    )


class _RoutedAnswers(coding.Answers):
    """`coding.Answers` plus the classifier's own answer — the op that exists only to be paid."""

    def __getitem__(self, key: str) -> object:
        _, name = coding._placed(str(key))
        if name == "classify":
            return "unused — route reads the return value, not this"
        return super().__getitem__(key)
