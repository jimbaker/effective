"""Two claims pinned by showing them: the postamble's SHAPE, and the incumbent's defect.

**Role: adversarial.** Both tests exist to fail if a plausible edit lands, and neither
is reachable by the journey suite:

- `test_coding_machine.py` proves the postamble runs on the two exit paths its fixtures take. It
  cannot see a guard above the call site that those fixtures happen to satisfy, and "a fixture
  never takes that branch" is precisely how an unconditional tail stops being unconditional. Only
  reading the source's shape catches that, so the first test here parses it.
- The delta over the incumbent is the reason this package exists, and the journey suite shows
  only half of it (the new machine terminates). The second test drives the INCUMBENT on the same
  class of input and watches it fail, so the comparison is shown rather than claimed.

*Probing a pure function proves things about the pure function.* `advance` in isolation looks
fail-safe (an unrecognised outcome "re-tries"), and that reading is wrong, because the retry
target is the one phase that appends to the canonical record on every visit. So the second test
drives the composition and counts ledger rows, not the function.
"""

import ast
from collections.abc import Mapping
from pathlib import Path

import _coding
import pytest

from effective.api import Effect
from effective.coding.states import PlanVerdict, State, Verdict
from effective.coding.transition import Park, transition
from effective.coding.verdicts import StructuralJudgementRequired
from effective.handlers.recording import RecordingHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence, StateSpec
from effective.machine.specs import fuse
from effective.machine.trampoline import Session, run_machine

MACHINE = (
    Path(__file__).resolve().parent.parent / "src" / "effective" / "machine" / "trampoline.py"
)
"""The trampoline's own source. These pins parse a FILE, so pointing them at a re-export shim
would leave them asserting about nothing, which is how a structural pin goes vacuous in a move."""
POSTAMBLE = "commitment_postamble"

# --- the postamble's shape, pinned structurally ------------------------------------------------

_BRANCHING = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.ExceptHandler)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from {MACHINE.name} — this pin names a real function")


def _callee_name(func: ast.expr) -> str:
    """The name a call invokes, through an attribute access as well as a bare name.

    **This closes a wrong DIAGNOSTIC, not an escape.** Under an `isinstance(call.func, ast.Name)`
    test, both `_m.commitment_postamble(...)` and an aliased `cp(...)` are seen as ZERO call
    sites, so `test_the_postamble_has_one_call_site_per_ending` fails either way. It fails saying
    *the postamble is gone*, which is the hazard: the obvious response to that message is to
    loosen the pin, and the pin would then be loose for real. Recognising the callee makes a
    legal refactor stay green and an actual removal say what it is."""
    match func:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=attr):
            return attr
        case _:
            return ""


def _postamble_names(tree: ast.Module) -> set[str]:
    """Every local name that reaches the postamble — the bare one plus any import alias."""
    names = {POSTAMBLE}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            names |= {a.asname for a in node.names if a.name == POSTAMBLE and a.asname}
    return names


def _walk_local(node: ast.AST):
    """`ast.walk`, stopping at every nested `def`/`lambda` — a SCOPE-local walk.

    The distinction is the difference between "this code runs when `run_machine` runs" and "this
    code appears somewhere inside `run_machine`'s source", and every pin below wants the first.
    A plain walk conflates them in both directions: a call moved into a nested helper still looks
    present, and a `return` inside a nested helper looks like an early exit of the outer function.
    (Measured: the nested-def escape passed the positional pin until this walk replaced
    `ast.walk`.)
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        yield child
        yield from _walk_local(child)


def _postamble_calls(node: ast.AST, names: set[str], *, local: bool = False) -> list[ast.Call]:
    walk = _walk_local(node) if local else ast.walk(node)
    return [
        call for call in walk if isinstance(call, ast.Call) and _callee_name(call.func) in names
    ]


def _calls_postamble(node: ast.AST, names: set[str], *, local: bool = False) -> bool:
    return bool(_postamble_calls(node, names, local=local))


COMMITTERS = {"run_machine", "committing"}
"""The functions that may run the postamble, one call each: `run_machine` for a run that ended,
and `committing` for a run that was refused, which `run_machine` raises before reaching its own."""


def test_the_postamble_has_one_call_site_per_ending():
    """Two call sites for one run is two records for it, and the ledger is append-only, so the
    second is permanent. A refused run skips `run_machine`'s call by raising above it, and a caller
    commits it through `committing`, so no run reaches both."""
    tree = ast.parse(MACHINE.read_text(), filename=str(MACHINE))
    names = _postamble_names(tree)
    callers = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for _call in _postamble_calls(node, names, local=True)
    ]
    assert "run_machine" in callers, "the pin found no call in run_machine"
    assert len(callers) == len(set(callers)), f"a committer calls {POSTAMBLE} twice: {callers}"
    assert set(callers) <= COMMITTERS, f"{POSTAMBLE} is called from {set(callers) - COMMITTERS}"


def test_no_branch_stands_above_the_postamble():
    """**The pin no behavioural test can replace.** A guard here would be invisible to every
    fixture that satisfies it, and the failure mode is silent: a run that exhausted or was
    rejected simply never reaches the canonical record, and nothing reddens.

    So: inside `run_machine`, no `if`/`for`/`while`/`try`/`with`/`match` may CONTAIN the call. The
    loop is above it in the source, which is fine — the call must not be inside it."""
    tree = ast.parse(MACHINE.read_text(), filename=str(MACHINE))
    names = _postamble_names(tree)
    run = _function(tree, "run_machine")
    guarded = [
        type(node).__name__
        for node in ast.walk(run)
        if isinstance(node, _BRANCHING) and _calls_postamble(node, names)
    ]
    assert not guarded, (
        f"{POSTAMBLE} sits inside {guarded} — the tail is conditional again. A run that "
        "exhausted or was rejected would leave no canonical record, and no fixture would notice."
    )
    assert _calls_postamble(run, names), (
        "the pin found no call at all — it is watching the wrong function"
    )


def test_the_postamble_is_a_direct_element_of_run_machines_body():
    """Not inside a branch is weaker than it reads, and the gap is a NESTED DEF.

    `no_branch_stands_above` walks `run_machine` and asks whether any `if`/`for`/… contains the
    call. Move the call into a helper `def` inside `run_machine` and no branching node contains
    it, so the pin passes — while the run writes zero ledger rows unless something calls the
    helper. The stronger statement is positional: the call is a statement of `run_machine`'s own
    body, at the top level of that body, and that is what "unconditional tail" actually means."""
    tree = ast.parse(MACHINE.read_text(), filename=str(MACHINE))
    names = _postamble_names(tree)
    run = _function(tree, "run_machine")
    direct = [
        stmt
        for stmt in run.body
        if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef)
        and _calls_postamble(stmt, names, local=True)
    ]
    assert len(direct) == 1, (
        f"{POSTAMBLE} is not a direct statement of run_machine's body ({len(direct)} found). "
        "A call nested in a helper def satisfies every branch check and still runs never."
    )


ADDRESS = "commit_id"
"""The assignment that decides the record's ADDRESS. It is the floor for the exit rule below — see
that test for why the rule is relative to it rather than to the top of the function."""


def _raises_the_refused_run(node: ast.AST) -> bool:
    """The one exit a run may take after its address: a refusal, raised as the run it stopped,
    for the caller to commit or not."""
    match node:
        case ast.Raise(exc=ast.Call(func=func)):
            return _callee_name(func) == "RunRefused"
        case _:
            return False


def test_nothing_returns_or_raises_between_the_ADDRESS_and_the_postamble():
    """The third escape, and the one no branch check can see: an early `return` ABOVE the call.

    A guard is not the only way to make a tail conditional: `if exhausted: return session` needs
    no branch *around* the postamble to skip it. So the pin also reads source order. The `break`
    in the visit loop is deliberately not in that set; it leaves the loop, not the function.

    **One exit is the design: a refused run raises `RunRefused`** and stays uncommitted until a
    caller commits it. Any other exit is the defect.

    **The floor is the ADDRESS, not the top of the function.** The refusal of a `run_id` that
    cannot form an atom also exits `run_machine` above the postamble, and this pin does not see
    it, because it is raised inside `compose_key` rather than written as an `ast.Raise` here. So
    the rule is "no exit once a run EXISTS", and a run exists once it has an address to be
    recorded at. Above that line there is nothing to commit and nothing has been done; below it,
    an exit is the defect this pin guards
    (`test_a_partial_spec_map_is_REFUSED_before_the_first_op` is the other side of the same rule).
    """
    tree = ast.parse(MACHINE.read_text(), filename=str(MACHINE))
    names = _postamble_names(tree)
    run = _function(tree, "run_machine")
    calls = _postamble_calls(run, names, local=True)
    assert calls, (
        f"no {POSTAMBLE} call runs when run_machine runs — it has moved into a nested scope, "
        "which the positional pin above is the one that should be read first"
    )
    call = calls[0]

    address = [
        node.lineno
        for node in _walk_local(run)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == ADDRESS for t in node.targets)
    ]
    assert address, (
        f"no `{ADDRESS}` assignment in run_machine — the exit rule below is relative to it, so "
        "without it this test grades nothing. Renaming the address means renaming ADDRESS here."
    )
    floor = min(address)

    below = [
        node
        for node in _walk_local(run)
        if isinstance(node, ast.stmt) and floor < node.lineno < call.lineno
    ]
    refusals = [node for node in below if _raises_the_refused_run(node)]
    guards = [
        node
        for node in below
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "walked.refusal is not None"
        and node.body == refusals
        and not node.orelse
    ]
    assert (len(refusals), len(guards)) == (1, 1), (
        f"the refused run's exit is {refusals} under {guards}: it is one `raise RunRefused`, "
        "alone under `if walked.refusal is not None`"
    )
    early = [
        (type(node).__name__, node.lineno)
        for node in below
        if isinstance(node, ast.Return | ast.Raise) and node not in refusals
    ]
    assert not early, (
        f"{early} sits between the address (line {floor}) and the {POSTAMBLE} call — the tail is "
        "conditional again, by exit rather than by guard, and no fixture that never takes the "
        "branch would notice."
    )


# --- the incumbent's defect, reproduced --------------------------------------------------------


class Unrecognised(Mapping[str, object]):
    """`tests/_coding.Answers`, with the suite answering a CASE VARIANT of the expected token.

    Not a contrived value: `"PASS"` is what a real provider returns often enough that the
    incumbent's own note called it out. The point is that nothing in the incumbent's types, tests
    or `ty` distinguishes it from `"pass"`."""

    def __init__(self, cap: int) -> None:
        self.cap, self.asked = cap, 0

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: str) -> object:
        self.asked += 1
        if self.asked > self.cap:
            raise RuntimeError("did-not-terminate")
        name = str(key)
        if "run_tests" in name:
            return "PASS"  # <-- the whole defect, one keystroke wide
        if "commit" in name:
            return f"sha{self.asked}"
        return "x"


def test_the_incumbent_livelocks_and_appends_forever_on_a_case_variant():
    """The defect this package exists to remove, DEMONSTRATED on the incumbent.

    `advance("test", "PASS", 1) -> "code"`, and `code` is the one phase that reaches the canonical
    record on every visit, so the typo does not re-try; it commits, forever. Probing `advance`
    alone would call this fail-safe; driving the composition shows what it costs.

    **A THRESHOLD, not an equality.** This instrument measures **65 rows against 401 ANSWERED
    ops**; a cap on *recorded ops* gives a different count over the same walk (99 rows in 400),
    because the two caps are different denominators. What is reproducible, and what the defect
    is, is the shape: unbounded growth of the canonical record with no exit. So the assertion is
    a floor that any honest cap clears."""
    answers = Unrecognised(cap=400)
    handler = RecordingHandler(responses=answers)
    with pytest.raises(RuntimeError, match="did-not-terminate"):
        handler.run(lambda: _coding.work(_coding.RUN_ID, _coding.TASK))
    committed = [row for row in handler.ledger if row.kind == "committed"]
    assert len(committed) > 50, (
        "the incumbent should have appended a canonical row per loop; if this is small the "
        "fixture changed and the comparison below is no longer against the real defect"
    )
    assert len({row.event_id for row in committed}) == len(committed), (
        "each append is a DISTINCT permanent row: the defect is unbounded growth of the "
        "canonical record, not one row rewritten"
    )


def test_the_machine_terminates_and_appends_once_on_the_same_shape_of_input():
    """The other half of the comparison, on the same shape: a verdict that always loops.

    The machine cannot livelock, and the reason is structural rather than lucky — the interpreter
    counts the budget and mints `Exhausted` without asking, so a judge that never says stop is
    bounded anyway. One record, not ninety-nine."""

    def always_loop(_ctx: Ctx, _evidence: Evidence) -> Effect[Verdict]:
        return PlanVerdict.REVISE
        yield  # pragma: no cover  -- generator, as `Judge` requires

    def worker(ctx: Ctx) -> Effect[Evidence]:
        return Evidence(summary=ctx.state.value)
        yield  # pragma: no cover

    specs = {s: StateSpec(state=s, run=fuse(worker, always_loop)) for s in State}
    handler = RecordingHandler(responses={"tool:run_suite": CommandRun(exit_code=0)})
    out = handler.run(
        lambda: run_machine(
            Run("lv-2026"), "loop forever", specs, transition, start=State.PLAN, budget=12
        )
    )
    assert isinstance(out, Session)
    assert isinstance(out.stopped, Park)
    assert len(out.turns) == 13
    assert len(handler.ledger) == 2


def test_a_RAISING_judge_escapes_the_postamble_and_that_is_UNRULED():
    """The boundary of the unconditional-tail property, pinned so it is a fact rather than a
    docstring claim.

    The postamble runs on every *outcome* — approved, exhausted, rejected. It does NOT run when a
    judge or worker RAISES, because an exception propagates past the call site rather than
    reaching it. That path is reachable in shipped code: `verdicts.verdict_for_finalize` raises
    `StructuralJudgementRequired` by design when the suite is green and nobody supplied the
    structural answer.

    **The AST pin above cannot see this**, and that is the point of writing it down here. It checks
    for an `if` CONTAINING the call; an escape BEFORE the call is a different shape, and no other
    fixture exercises it.

    This test asserts the CURRENT behavior, not a desired one. Whether an exceptional run should
    commit what it had is an open design question: the incumbent's property is that honest
    failures land on the ledger, and a raised judge is a failure that lands nowhere. When it is
    decided, this test changes with it."""

    def raising_judge(_ctx: Ctx, _evidence: Evidence) -> Effect[Verdict]:
        raise StructuralJudgementRequired("the suite is green and nobody answered")
        yield  # pragma: no cover

    def worker(ctx: Ctx) -> Effect[Evidence]:
        return Evidence(summary=ctx.state.value)
        yield  # pragma: no cover

    specs = {s: StateSpec(state=s, run=fuse(worker, raising_judge)) for s in State}
    handler = RecordingHandler(responses={"tool:run_suite": CommandRun(exit_code=0)})
    with pytest.raises(StructuralJudgementRequired):
        handler.run(
            lambda: run_machine(
                Run("raise-2026"), "goal", specs, transition, start=State.FINALIZE, budget=3
            )
        )
    assert handler.ledger == [], "the postamble ran on an exception — the ruling changed"
    assert handler.trace == [], "ops reached the tape before the raise"
