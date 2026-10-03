"""A coding agent as a STATE MACHINE — the fixture whose projection is a state diagram.

A shared fixture like `_funnel.py`, `_cart.py` and `_mcts.py`. What it has that none of them does
is **genuine backedges**: a failing test sends the machine back to `code`, and a refactor sends it
back to `test`. The cart's loop and the search's loop both go forward; this one goes round, and a
projection that recovers it recovers the *program*, not a summary of it.

**The states.** `plan` first, because a coding agent that edits before it has said what it intends
is the one nobody wants::

    plan -> explore -> repl -> code -> test -> refactor -> finalize
                                 ^       |        |
                                 +-------+--------+

**Where the transition lives is the design question this fixture answers.** `combinators.route`
dispatches on a recorded CLASSIFICATION: a classifier op runs, and its result picks the arm. A
state machine does not work that way — the arm to run is already known (it is the state), and what
is decided by a recorded value is the arm to run *next*. Expressing this with `route` would spend
one recorded op per turn re-deriving a state the workflow already holds, so the machine here is a
plain loop over a dict of phase generators with a PURE transition over the recorded outcome.
`test_coding_machine_example.py` measures what `route` would have cost.

**Two bookkeepers, and the fixture is careful about which.** `explore`, `repl` and `test` are
sandbox work — disposable, checkpointed, and nothing more. Only `code` (a commit) and `finalize`
reach the canonical record. That is the sandbox-disposable/ledger-canonical rule, and it is what
makes selection worth running here: a hundred-turn session restricted to the ledger is the handful
of commitments a reviewer actually reads.

**The plan is gated.** A human approves it before any world change, which is the one park.

**Nothing here spells a key.** `Answers` reads the PLACEMENT off the key it is asked for.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from effective.api import (
    Effect,
    append_ledger,
    ask_llm,
    await_event,
    call_tool,
    scoped,
)
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Segment, compose_key
from effective.keys.grammar import KeySyntaxError, ParsedKey, parse
from effective.ops import LedgerRow

RUN_ID = "wk-2026"

DONE = "done"
"""The terminal state. Not a phase — nothing runs in it, which is what makes the loop's exit a
property of the transition table rather than a `break` somewhere in a body."""


@dataclass(frozen=True)
class Task:
    """What the agent was asked to do, and how the world will answer — the case, written first.

    `passes_at` is the turn from which the suite goes green. Keying the world's answer on the TURN
    rather than on the agent's own state is deliberate: it keeps `Answers` a function of the
    placement alone, so it says the same thing on replay, and it keeps the fixture from deciding
    its own test results."""

    goal: str
    passes_at: int
    refactors: int


TASK = Task(goal="fix the off-by-one in the pager", passes_at=6, refactors=1)
"""One red-to-green cycle and one refactor round, so both backedges are taken exactly once and a
reader can count the turns by hand."""


@dataclass(frozen=True)
class Turn:
    """One transition, recorded: which state ran and what it returned."""

    state: str
    outcome: str


@dataclass(frozen=True)
class Session:
    """The finished run."""

    run_id: str
    turns: tuple[Turn, ...]
    commits: tuple[str, ...]

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(turn.state for turn in self.turns)


# --- the phases: one generator each, all naming their ops plainly -----------------------------


def _plan(task: Task, run_id: str) -> Effect[str]:
    """Say what will be done, and get a human's word before anything touches the world."""
    yield from ask_llm("draft", f"Plan the work: {task.goal}", str)
    # `review:`, not `approve:` — the latter is a substrate authority namespace and is refused for
    # an authored await. The name is scoped on the RUN, which is this question's subject.
    return (yield from await_event(f"review:{run_id}", str))


def _explore(task: Task, run_id: str) -> Effect[str]:
    return (yield from call_tool("read_files", {"goal": task.goal}, str))


def _repl(task: Task, run_id: str) -> Effect[str]:
    return (yield from call_tool("evaluate", {"goal": task.goal}, str))


def _code(task: Task, run_id: str) -> Effect[str]:
    """Edit and commit. The one phase that reaches the canonical record on every visit — so a
    machine that loops through here twice leaves two commits, which is the point."""
    edit = yield from ask_llm("edit", f"Write the change: {task.goal}", str)
    sha = yield from call_tool("commit", {"edit": edit}, str)
    # THE WORLD CHANGES HERE. Inside the turn's scope, so the axis is visible to a projection —
    # the placement rule `_mcts` measured.
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"committed:{Segment(sha)}"), kind="committed")
    )
    return sha


def _test(task: Task, run_id: str) -> Effect[str]:
    return (yield from call_tool("run_tests", {"goal": task.goal}, str))


def _refactor(task: Task, run_id: str) -> Effect[str]:
    return (yield from ask_llm("tidy", f"Tidy the change: {task.goal}", str))


def _finalize(task: Task, run_id: str) -> Effect[str]:
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"finalized:{Segment(run_id)}"), kind="finalized")
    )
    return DONE


PHASES = {
    "plan": _plan,
    "explore": _explore,
    "repl": _repl,
    "code": _code,
    "test": _test,
    "refactor": _refactor,
    "finalize": _finalize,
}


def advance(state: str, outcome: str, refactors_left: int) -> tuple[str, int]:
    """The transition — PURE, over the recorded outcome and a count the workflow holds.

    Written as a total function rather than as `if`s inside the phases, which is what lets the
    machine's shape be read (and tested) without running it. The two branching states are the
    two backedges, and they are the whole reason this fixture exists."""
    match state:
        case "plan":
            return "explore", refactors_left
        case "explore":
            return "repl", refactors_left
        case "repl":
            return "code", refactors_left
        case "code":
            return "test", refactors_left
        case "test":
            if outcome != "pass":
                return "code", refactors_left  # BACKEDGE: red tests send it back
            if refactors_left > 0:
                return "refactor", refactors_left
            return "finalize", refactors_left
        case "refactor":
            return "test", refactors_left - 1  # BACKEDGE: a tidy is re-tested
        case "finalize":
            return DONE, refactors_left
    raise ValueError(f"no transition out of {state!r}")


def work(run_id: str, task: Task) -> Effect[Session]:
    state, refactors_left, turns = "plan", task.refactors, []
    commits: list[str] = []
    while state != DONE:
        turn = len(turns)
        outcome = yield from scoped(
            # lint: terminal-hole — `turn` is `len(turns)`, an `int`, so it is already an atom
            compose_key(t"turn:{turn}"),
            lambda s=state: scoped(
                compose_key(t"phase:{Segment(s)}"), lambda: PHASES[s](task, run_id)
            ),
        )
        if state == "code":
            commits.append(outcome)
        turns.append(Turn(state, outcome))
        state, refactors_left = advance(state, outcome, refactors_left)
    return Session(run_id, tuple(turns), tuple(commits))


# --- the scripted run: the answers, and the spine that drives them -------------

_FRAME_TAGS = ("turn", "phase")
"""The frames this run mints. `turn:{n}` counts executions of one position, so it declares
`Index`; `phase:{name}` says WHICH state ran, which a projection must keep or the state machine
collapses to a single box, so it declares `Name`. `--coordinate-roles` reads
`lint.KEY_REGISTRY_SRCS` and this file is not in it, so nothing gates the reading here; it is
written down so a reader does not have to infer it."""


def _placed(key: str) -> tuple[dict[str, str], str]:
    """Split a placed key into `{frame tag: its first coordinate}` and the author's own op name.

    `turn:3;phase:code;step:edit` -> `({"turn": "3", "phase": "code"}, "edit")`. Total: text the
    parser refuses comes back as its own name, which the caller then fails to find rather than
    mis-answering."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return {}, key
    frames = {
        term.tag: term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag in _FRAME_TAGS and term.coordinates
    }
    body = next((i for i, term in enumerate(terms) if term.tag not in _FRAME_TAGS), len(terms))
    return frames, ParsedKey(terms[body:]).render() if body < len(terms) else key


def phases_in(key: str) -> tuple[str, ...]:
    """Every `phase:` coordinate a placed key carries, read through the production parser.

    The sibling of `_funnel.lanes_in`, `_cart.items_in` and `_mcts.nodes_in`."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return ()
    return tuple(
        term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag == "phase" and term.coordinates
    )


class Answers(Mapping[str, object]):
    """The canned side of the run, answered by PLACEMENT rather than by a spelled key.

    The only interesting answer is `run_tests`, which reads the TURN it was asked at — so the
    world's verdict is a function of the placement and nothing else, and says the same thing on
    replay. Note what is NOT here: `review:{run_id}`, the plan's approval."""

    def __init__(self, task: Task) -> None:
        self._task = task

    def __iter__(self):
        """The names answerable WITHOUT a frame — none. Every op here needs a `turn:` frame,
        because the suite's verdict depends on when it was asked, so enumerating any name would
        hand a caller a `KeyError`."""
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: str) -> object:
        frames, name = _placed(str(key))
        turn = int(frames.get("turn", -1))
        match name:
            case "draft":
                return f"plan: {self._task.goal}"
            case "tool:read_files":
                return "pager.py, test_pager.py"
            case "tool:evaluate":
                return "reproduced at n=1"
            case "edit":
                return "range(n) -> range(n + 1)"
            case "tool:commit":
                return f"sha{turn}"
            case "tool:run_tests":
                return "pass" if turn >= self._task.passes_at else "fail"
            case "tidy":
                return "extracted the bound into a constant"
        raise KeyError(key)


APPROVED = "looks right — go"


def scripted_run() -> tuple[Session, list[LedgerRow], list[str]]:
    """Drive the scripted task: plan, get approval, red-to-green, refactor, finalize.

    Returns the session, the canonical rows, and the tape — so a caller asserts over all three
    bookkeepers without re-driving."""
    handler = RecordingHandler(responses=Answers(TASK))
    outcome = handler.run(lambda: work(RUN_ID, TASK))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(APPROVED)
    assert isinstance(outcome, Session)
    return outcome, handler.ledger, [entry.key.stored() for entry in handler.trace]
